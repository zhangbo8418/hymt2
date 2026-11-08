/* 统一加速设备抽象: CPU 外可选 Vulkan / CUDA / Metal / NPU。
 * 算子与运行时思路参考 llama.cpp ggml-cuda / ggml-metal / ggml-vulkan，不链接 llama。 */
#ifndef HYMT2_DEVICE_H
#define HYMT2_DEVICE_H

#include "internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hy_device hy_device;
typedef struct hy_dweight hy_dweight;

typedef enum {
    HY_DEV_NONE = 0,
    HY_DEV_VULKAN = 1,
    HY_DEV_CUDA = 2,
    HY_DEV_METAL = 3,
    HY_DEV_NPU = 4
} hy_dev_kind;

/* pref: NULL/"auto" 按 cuda→metal→vulkan(非软件) 探测;
 * "cpu"/"0"/"off" = 不建设备;
 * "vulkan"|"cuda"|"metal"|"npu" = 强制;
 * "force" = Vulkan 允许软件设备 (兼容 HYMT2_VK);
 * "<n>" = Vulkan 第 n 个设备。也读环境变量 HYMT2_DEVICE, 未设则回退 HYMT2_VK。 */
hy_device *hy_device_create(const char *pref, char *info, size_t cap);
hy_dev_kind hy_device_kind(const hy_device *d);
const char *hy_device_name(const hy_device *d);
bool hy_device_is_software(const hy_device *d);
void hy_device_destroy(hy_device *d);

hy_dweight *hy_device_upload(hy_device *d, const weight_t *w);
void hy_device_release_on(hy_device *d, hy_dweight *w);
/* 一组共享同一激活的权重: encode 全部 GEMM → 同步一次 → 结果对 CPU 可见。
 * 若 outs[i] 等于 hy_device_weight_out(ws[i]) (共享/映射输出), 后端跳过二次 memcpy。 */
bool hy_device_matmul(hy_device *d, hy_dweight *const *ws, float *const *outs, int cnt,
                     const q8k_block *xq, int nt);
/* 权重的 host 可见输出缓冲 (Metal/Vulkan shared, CUDA pinned); 不可用返回 NULL。布局 out[t*n+r]。 */
float *hy_device_weight_out(hy_device *d, hy_dweight *w);

/* ---- 整层/整网常驻前向 (Metal / CUDA; 一次 commit/stream 内跑完, ≤1 次 GPU 同步) ---- */
typedef struct hy_fused_layer {
    hy_dweight *wq, *wk, *wv, *wo, *wg, *wu, *wd;
    const float *attn_norm, *ffn_norm, *q_norm, *k_norm;
} hy_fused_layer;

typedef struct hy_fused_args {
    int n_layer, hidden, ffn, heads, kv_heads, hdim, qdim, kvdim, ctx, rope_half, nt, pos0;
    float eps;
    int want_logits;
    hy_fused_layer *layers;
    hy_dweight *outw;
    const float *out_norm;
    const float *rope_cos, *rope_sin; /* [ctx * rope_half] */
    float *h;      /* in/out [nt * hidden], host 可写 (embed 后) */
    float *logits; /* want_logits 时写入 [outw->n] */
} hy_fused_args;

bool hy_device_has_fused(const hy_device *d);
/* 按模型形状分配 KV/工作缓冲; 可重复调用 (同形状 noop) */
bool hy_device_fused_prepare(hy_device *d, const hy_fused_args *shape);
bool hy_device_fused_forward(hy_device *d, hy_fused_args *a);

/* 探测本机构建/运行时是否具备各后端。例: "isa=sse4.1 cuda=0 metal=0 vulkan=1 npu=nnapi|0" */
int hy_device_capabilities(char *out, int cap);

/* 后端实现用: 把私有 ctx 包成 hy_device (vtable)。destroy_impl 只释放 impl, wrap 层 free hy_device。 */
typedef struct hy_device_ops {
    void (*destroy_impl)(void *impl);
    hy_dweight *(*upload)(void *impl, const weight_t *w);
    void (*release)(hy_dweight *w);
    bool (*matmul)(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt);
    bool (*is_software)(const void *impl);
    float *(*weight_out)(hy_dweight *w); /* 可选; NULL = 无零拷贝输出 */
    bool (*fused_prepare)(void *impl, const hy_fused_args *shape);
    bool (*fused_forward)(void *impl, hy_fused_args *a);
} hy_device_ops;
hy_device *hy_device_wrap(hy_dev_kind kind, const char *name, void *impl, const hy_device_ops *ops, bool software);

/* auto 校准用: 按名字尝试创建 (cuda/metal/vulkan/nnapi/qnn/rkllm/hiai)。失败返回 NULL。 */
hy_device *hy_device_try(const char *name, char *info, size_t cap);

#ifdef __cplusplus
}
#endif
#endif
