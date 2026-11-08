/* hymt2 内部头文件。
 * 本项目是 sdcb/HyMT2Sharp (Apache-2.0) 中 CPU 路径的 C 语言移植，
 * 只覆盖运行 Tencent Hy-MT2-1.8B-1.25Bit (STQ1_0) GGUF 所需的部分。 */
#ifndef HYMT2_INTERNAL_H
#define HYMT2_INTERNAL_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define QK_K 256
#define MAX_BATCH 16 /* 一次前向最多处理的 token 数 (prefill 分块) */

#pragma pack(push, 1)
typedef struct { uint8_t qs[32]; uint8_t sign[8]; uint16_t d; } block_stq1_0; /* 42 B / 256 权重 */
typedef struct { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; uint16_t d; } block_q6_k; /* 210 B */
typedef struct { uint16_t d; int8_t qs[32]; } block_q8_0; /* 34 B */
typedef struct { uint16_t d, dmin; uint8_t scales[12]; uint8_t qs[128]; } block_q4_k; /* 144 B */
typedef struct { uint16_t d; uint8_t qs[128]; } block_q2_0c; /* 130 B / 512 权重, 每字节 4 个连续 2-bit 码, 值 = (2c-3)*d */
#pragma pack(pop)

/* 激活量化块 (自定义布局, 64 字节对齐):
 *   qp  : STQ1_0 内核用的"重排"副本。对每个 lane l (0..3), 64 字节依次存放 chunk 0,2,1,3 的 16 字节,
 *         这样 SSE (16B)、AVX2 (32B)、AVX-512 (64B) 都能用连续对齐的整块加载。
 *   qs  : 自然顺序, Q6_K 与标量路径用。 */
#define Q8K_PIECE_ORDER {0, 2, 1, 3}
typedef struct __attribute__((aligned(64))) {
    int8_t qp[QK_K];
    float d;
    int32_t sum;
    int16_t bsums[QK_K / 16];
    uint8_t _pad[64 - 8 - 32];
    int8_t qs[QK_K];
} q8k_block; /* 576 B */

void *hy_aligned_alloc(size_t size);
void hy_aligned_free(void *p);

enum { GT_F32 = 0, GT_F16 = 1, GT_Q8_0 = 8, GT_Q4_K = 12, GT_Q6_K = 14, GT_Q2_0C = 40, GT_STQ1_0 = 43, GT_UNSUPPORTED = -1 };

typedef struct {
    const uint8_t *data;
    int type;
    int k;          /* 输入维 (ne[0]) */
    int n;          /* 输出行数 (ne[1]) */
    size_t row_bytes;
    /* 面板重排 (STQ1_0 / Q2_0C): 加载时把权重预解码成 2-bit 码, 运行时内层循环不再做码本查表/符号选择。
     * pk: n * (k/256) 个 64 字节块 (64 字节对齐); ps: 每块一个 float 缩放。
     * 结果 = Σ_b ps[b] * y.d * ((enc << pshift) - poff * y.sum)  (STQ: pshift=0,poff=1; Q2_0C: pshift=1,poff=3) */
    const uint8_t *pk;
    const float *ps;
    int pshift, poff;
    void *gpu;      /* 设备权重句柄 (hy_dweight*), NULL = 仅 CPU */
} weight_t;

/* ---- kernels.c ---- */
void kernels_init(void);
const char *kernels_cpu_features(void); /* 例如 "ssse3 sse4.2 avx avx2 f16c fma avx512f avx512bw avx512vnni" */
bool kernels_set_isa(const char *name); /* NULL = 自动; 测试用: 强制 scalar/ssse3/avx2/neon */
const char *kernels_isa(void);
float f16_to_f32(uint16_t h);
float bf16_to_f32(uint16_t h);
uint16_t f32_to_bf16(float f);
size_t row_size(int type, int k);
void quantize_q8k(const float *x, int k, q8k_block *y);
void dequant_row(const weight_t *w, int row, float *dst);
/* 把 [r0,r1) 行重排成面板 (dst_pk/dst_ps 已按整张权重分配好) */
void panel_repack_rows(const weight_t *src, uint8_t *dst_pk, float *dst_ps, int r0, int r1);
bool panel_supported(int type);
/* out[t * w->n + r] = dot(row r, act t),  r in [r0, r1),  t in [0, nt) */
void matmul_rows(const weight_t *w, int r0, int r1, const float *xf, const q8k_block *xq, int nt, float *out);
float dot_bf16(const float *q, const uint16_t *k, int n);
void axpy_bf16(float *o, const uint16_t *v, float p, int n);
float dot_bf16_ref(const float *q, const uint16_t *k, int n);
/* 仅测试用: 标量参考实现 (STQ1_0 与 Q8_K 点积) */
float stq1_0_dot_scalar(const block_stq1_0 *x, const q8k_block *y, int nb);
float stq1_0_dot_dispatch(const block_stq1_0 *x, const q8k_block *y, int nb);
float q6_k_dot_scalar(const block_q6_k *x, const q8k_block *y, int nb);
float q4_k_dot_scalar(const block_q4_k *x, const q8k_block *y, int nb);
int type_from_ggml(unsigned raw); /* ggml 类型号 -> GT_*, 不支持返回 GT_UNSUPPORTED */
void stq1_0_dequant_row(const block_stq1_0 *x, float *dst, int k);
void q6_k_dequant_row(const block_q6_k *x, float *y, int k);

/* ---- threadpool.c ---- */
typedef struct threadpool threadpool;
threadpool *tp_create(int n_threads);
void tp_destroy(threadpool *tp);
int tp_threads(const threadpool *tp);
/* 物理核数 (不含超线程逻辑核)。探测失败时回退 1。 */
int hy_physical_cpu_count(void);
typedef void (*tp_fn)(void *ctx, int begin, int end);
/* 把 [0, n) 按 grain 切块动态分给各线程 (含调用线程) 并行执行 fn */
void tp_for(threadpool *tp, int n, int grain, tp_fn fn, void *ctx);

/* ---- gguf.c ---- */
typedef struct { const char *p; uint64_t len; } gstr;
typedef struct {
    gstr name;
    uint32_t n_dims;
    uint64_t dims[4];
    uint32_t type;      /* 原始 ggml 类型 */
    uint64_t offset;
} gguf_tensor;
typedef struct {
    gstr key;
    uint32_t type;      /* gguf 值类型 */
    uint32_t arr_type;
    uint64_t arr_len;
    const uint8_t *arr_data;   /* 数组: 指向元素起点 (字符串数组需自行遍历) */
    union { uint64_t u; int64_t i; double f; gstr s; } v;
} gguf_kv;
typedef struct gguf gguf;
gguf *gguf_open(const char *path, char *err, size_t errcap);
void gguf_close(gguf *g);
const gguf_kv *gguf_find(const gguf *g, const char *key);
int64_t gguf_get_int(const gguf *g, const char *key, int64_t def);
double gguf_get_float(const gguf *g, const char *key, double def);
bool gguf_get_str(const gguf *g, const char *key, gstr *out);
const gguf_tensor *gguf_find_tensor(const gguf *g, const char *name);
const uint8_t *gguf_tensor_data(const gguf *g, const gguf_tensor *t);
void gguf_prefault(const gguf *g);
uint64_t gguf_tensor_bytes_avail(const gguf *g, const gguf_tensor *t);
/* 把 kv 数组里第 i 个字符串取出 (顺序遍历, 构建时一次性展开) */
bool gguf_array_strings(const gguf_kv *kv, gstr *out, size_t n);

/* ---- vulkan.c (可选, HYMT2_NO_VULKAN 关闭; 运行时动态加载 Vulkan 加载器, 没有也不影响) ---- */
typedef struct vkctx vkctx;
typedef struct vkweight vkweight;
/* pref: NULL/"auto" 选最好的非软件设备; "force" 允许软件设备 (lavapipe, 测试用); "<n>" 指定第 n 个设备; "list" 只打印设备列表 */
vkctx *vk_create(const char *pref, char *info, size_t cap);
bool vk_is_software(const vkctx *c);
void vk_destroy(vkctx *c);
/* 上传一张权重 (STQ1_0/Q2_0C 需要已有面板; Q4_K; Q6_K)。不支持或失败返回 NULL。 */
vkweight *vk_upload(vkctx *c, const weight_t *w);
void vk_release(vkweight *w);
float *vk_weight_out(vkweight *w); /* host-visible 输出; 布局 out[t*n+r] */
/* 一组共享同一输入的权重一次提交。输出布局与 CPU 一致: out[t*n + r]。失败返回 false (调用方回退 CPU)。
 * outs[i]==vk_weight_out(ws[i]) 时跳过二次 memcpy。 */
bool vk_matmul(vkctx *c, vkweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt);

/* ---- tokenizer.c ---- */
typedef struct tokenizer tokenizer;
tokenizer *tok_create(const gguf *g, char *err, size_t errcap);
void tok_destroy(tokenizer *t);
/* 返回 token 个数; 调用方 free(*ids) */
int tok_encode(const tokenizer *t, const char *text, size_t len, int **ids);
int tok_vocab_size(const tokenizer *t);
bool tok_is_stop(const tokenizer *t, int id);
/* 把 token 序列还原成 UTF-8 (跳过 ｜hy_ 特殊 token); 返回字节数, 写入 out (不含 NUL) */
size_t tok_decode_visible(const tokenizer *t, const int *ids, int n, char *out, size_t cap);

#endif
