/* Hunyuan-dense 前向 + 贪心生成。
 * 计算顺序移植自 HyMT2Sharp (HunyuanDenseModel*.cs / Ops.cs):
 *   RMSNorm → QKV → NeoX RoPE → 逐头 q/k RMSNorm (在 RoPE 之后!) → KV(bf16) → GQA 注意力 → 输出投影 → 残差
 *   RMSNorm → gate/up → SiLU(gate)*up → down → 残差;  最后 output_norm → output.weight (缺省复用 token_embd)。 */
#include "../include/hymt2.h"
#include "device.h"
#include "internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif
#if defined(__aarch64__)
#include <arm_neon.h>
#define HY_MODEL_NEON 1
#endif

typedef struct {
    float *attn_norm, *ffn_norm, *q_norm, *k_norm;
    weight_t wq, wk, wv, wo, wg, wu, wd;
} layer_t;

struct hymt2_model {
    gguf *g;
    tokenizer *tok;
    threadpool *tp;
    int n_layer, hidden, ffn, heads, kv_heads, hdim, qdim, kvdim, vocab, vocab_out, ctx, rope_half, batch;
    float eps;
    weight_t embd, outw;
    float *out_norm;
    layer_t *L;
    float *rope_cos, *rope_sin;
    uint16_t **kc, **vc;
    int *cache_tok;
    int n_cache;
    /* scratch (最多 MAX_BATCH 个 token) */
    float *h, *xn, *q, *k, *v, *ao, *gate, *up, *tmp, *scores, *logits;
    q8k_block *xq_h, *xq_q, *xq_f;
    /* 面板重排缓冲 (加载时生成, 释放时统一 free) */
    uint8_t **panel_pk;
    float **panel_ps;
    int n_panel, cap_panel;
    int repacked; /* 0 = 未重排, 1 = 已重排 */
    /* 加速设备 (Vulkan / CUDA / Metal / NPU, 经 device.h) */
    hy_device *dev;
    bool use_gpu;
    bool use_fused; /* Metal/CUDA: 仅整网常驻前向; 不用「逐 matmul+CPU 注意力」假加速 */
    hy_fused_layer *fused_layers; /* prepare 后常驻, forward 不再每步 calloc */
    hy_dweight **gpu_ws;
    int n_gpu, cap_gpu;
    /* 最近一次 generate 的统计 */
    int st_prefill_tok, st_cached_tok, st_decode_tok;
    double st_prefill_ms, st_decode_ms;
    char err[512];
};

#ifdef HYMT2_PROFILE
static double prof_ms[8]; static const char *prof_name[8] = {"embed","qkv","rope/kv","attn","wo","ffn","head","other"};
static double prof_now(void);
#define PROF_BEGIN() double _pt = prof_now()
#define PROF_END(i) do { double _n = prof_now(); prof_ms[i] += _n - _pt; _pt = _n; } while (0)
#else
#define PROF_BEGIN() do {} while (0)
#define PROF_END(i) do {} while (0)
#endif

static void set_err(hymt2_model *m, const char *fmt, const char *a) { snprintf(m->err, sizeof(m->err), fmt, a); }

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1000.0 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
#endif
}

static bool build_panel(hymt2_model *m, weight_t *w);
static void matmul(hymt2_model *m, const weight_t *w, const float *xf, const q8k_block *xq, int nt, float *out);
static void matmul_multi(hymt2_model *m, const weight_t **ws, float **outs, int cnt, const float *xf, const q8k_block *xq, int nt);
static void quantize_rows(const float *x, int nt, int k, q8k_block *y);

/* ------------------------------------------------------------------ 加载 */

static int map_type(const gguf *g, uint32_t raw) {
    switch (raw) {
    case 0: return GT_F32;
    case 1: return GT_F16;
    case 8: return GT_Q8_0;
    case 12: return GT_Q4_K;
    case 14: return GT_Q6_K;
    case 40: return GT_Q2_0C;
    case 43: return GT_STQ1_0;
    case 42: { /* 早期 HyMT2 Stride16 文件把 STQ1_0 放在 42, 与标准 Q2_0 撞号: 用元数据区分 */
        gstr name = { "", 0 };
        gguf_get_str(g, "general.name", &name);
        bool stride16 = false;
        for (uint64_t i = 0; i + 8 <= name.len; i++) {
            char tmp[9];
            memcpy(tmp, name.p + i, 8); tmp[8] = 0;
            for (int j = 0; j < 8; j++) if (tmp[j] >= 'A' && tmp[j] <= 'Z') tmp[j] = (char)(tmp[j] + 32);
            if (!strcmp(tmp, "stride16")) { stride16 = true; break; }
        }
        if (gguf_get_int(g, "general.file_type", -1) == 41 || stride16) return GT_STQ1_0;
        return GT_UNSUPPORTED;
    }
    }
    return GT_UNSUPPORTED;
}

static bool load_weight(hymt2_model *m, const char *name, weight_t *w, int k_expect, int n_expect, bool optional) {
    const gguf_tensor *t = gguf_find_tensor(m->g, name);
    if (!t) {
        if (!optional) set_err(m, "missing tensor %s", name);
        return false;
    }
    int type = map_type(m->g, t->type);
    if (type == GT_UNSUPPORTED) {
        snprintf(m->err, sizeof(m->err), "tensor %s: unsupported ggml type %u (本库支持 F32/F16/Q8_0/Q4_K/Q6_K/Q2_0C/STQ1_0)", name, t->type);
        return false;
    }
    if (t->n_dims < 2) { set_err(m, "tensor %s is not a matrix", name); return false; }
    w->type = type;
    w->k = (int)t->dims[0];
    w->n = (int)t->dims[1];
    if ((k_expect && w->k != k_expect) || (n_expect && w->n != n_expect)) {
        snprintf(m->err, sizeof(m->err), "tensor %s has shape [%d,%d], expected [%d,%d]", name, w->k, w->n, k_expect, n_expect);
        return false;
    }
    int blk = (type == GT_Q8_0) ? 32 : (type == GT_F32 || type == GT_F16) ? 1 : (type == GT_Q2_0C) ? 512 : QK_K;
    if (w->k % blk) { set_err(m, "tensor %s: inner dim not a multiple of block size", name); return false; }
    w->row_bytes = row_size(type, w->k);
    w->data = gguf_tensor_data(m->g, t);
    if ((uint64_t)w->n * w->row_bytes > gguf_tensor_bytes_avail(m->g, t)) { set_err(m, "tensor %s extends past end of file", name); return false; }
    return true;
}

static float *load_vec(hymt2_model *m, const char *name, int n) {
    const gguf_tensor *t = gguf_find_tensor(m->g, name);
    if (!t) { set_err(m, "missing tensor %s", name); return NULL; }
    if (t->type != 0 && t->type != 1) { set_err(m, "norm tensor %s must be F32/F16", name); return NULL; }
    if ((int)t->dims[0] != n) { set_err(m, "norm tensor %s has wrong length", name); return NULL; }
    float *v = (float *)malloc(sizeof(float) * (size_t)n);
    const uint8_t *p = gguf_tensor_data(m->g, t);
    for (int i = 0; i < n; i++) {
        if (t->type == 0) memcpy(&v[i], p + (size_t)i * 4, 4);
        else { uint16_t h; memcpy(&h, p + (size_t)i * 2, 2); v[i] = f16_to_f32(h); }
    }
    return v;
}

static int default_threads(void) {
    const char *e = getenv("HYMT2_THREADS");
    if (e && atoi(e) > 0) return atoi(e);
    /* 默认绑定物理核 (不含超线程逻辑核); 争用同一执行单元时逻辑核并行常更慢。
     * 可用 HYMT2_THREADS 覆盖。Android 额外封顶 4。 */
    int n = hy_physical_cpu_count();
    if (n < 1) n = 1;
#if defined(__ANDROID__)
    if (n > 4) n = 4;
#endif
    return n;
}

void hymt2_model_free(hymt2_model *m) {
    if (!m) return;
    if (m->L) {
        for (int i = 0; i < m->n_layer; i++) {
            free(m->L[i].attn_norm); free(m->L[i].ffn_norm); free(m->L[i].q_norm); free(m->L[i].k_norm);
        }
        free(m->L);
    }
    if (m->kc) for (int i = 0; i < m->n_layer; i++) free(m->kc[i]);
    if (m->vc) for (int i = 0; i < m->n_layer; i++) free(m->vc[i]);
    free(m->kc); free(m->vc); free(m->out_norm); free(m->rope_cos); free(m->rope_sin); free(m->cache_tok);
    free(m->h); free(m->xn); free(m->q); free(m->k); free(m->v); free(m->ao); free(m->gate); free(m->up); free(m->tmp);
    free(m->scores); free(m->logits); hy_aligned_free(m->xq_h); hy_aligned_free(m->xq_q); hy_aligned_free(m->xq_f);
    for (int i = 0; i < m->n_gpu; i++) hy_device_release_on(m->dev, m->gpu_ws[i]);
    free(m->gpu_ws);
    free(m->fused_layers);
    hy_device_destroy(m->dev);
    for (int i = 0; i < m->n_panel; i++) { hy_aligned_free(m->panel_pk[i]); free(m->panel_ps[i]); }
    free(m->panel_pk); free(m->panel_ps);
    tp_destroy(m->tp);
    tok_destroy(m->tok);
    gguf_close(m->g);
    free(m);
}

/* 用真实的线程池在第 0 层 FFN 上实测: 面板 (更多访存, 更少指令) 与原生 (更少访存, 更多解码指令) 谁更快。
 * 内存带宽很低的机器上面板可能更慢, 所以不凭猜测, 现场测。 */
static bool calibrate_panel_wins(hymt2_model *m) {
    /* 跨最多 8 层的 FFN (原生约 100MB / 面板约 160MB), 总量超过常见的 L2/L3, 反映"权重要从内存流式读取"的真实情况 */
    const int ncal = m->n_layer < 8 ? m->n_layer : 8;
    for (int l = 0; l < ncal; l++)
        if (m->L[l].wg.type != GT_STQ1_0 || m->L[l].wu.type != GT_STQ1_0 || m->L[l].wd.type != GT_STQ1_0) return true;
    const int saved = m->n_panel;
    for (int l = 0; l < ncal; l++) {
        if (!build_panel(m, &m->L[l].wg) || !build_panel(m, &m->L[l].wu) || !build_panel(m, &m->L[l].wd)) return false;
    }
    /* 固定的伪随机激活 */
    uint32_t s = 12345;
    for (int i = 0; i < m->hidden; i++) { s = s * 1664525u + 1013904223u; m->xn[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    for (int i = 0; i < m->ffn; i++) { s = s * 1664525u + 1013904223u; m->gate[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    quantize_rows(m->xn, 1, m->hidden, m->xq_h);
    quantize_rows(m->gate, 1, m->ffn, m->xq_f);
    double best[2] = { 1e30, 1e30 };
    for (int rep = 0; rep < 4; rep++) { /* 两种实现交替测, 减少背景负载漂移的影响; 第 0 轮兼作预热 (缺页) */
        for (int variant = 0; variant < 2; variant++) { /* 0 = 原生, 1 = 面板 */
            double t0 = now_ms();
            for (int l = 0; l < ncal; l++) {
                weight_t v[3] = { m->L[l].wg, m->L[l].wu, m->L[l].wd };
                if (variant == 0) for (int i = 0; i < 3; i++) v[i].pk = NULL;
                const weight_t *gu[2] = { &v[0], &v[1] };
                float *os[2] = { m->gate, m->up };
                matmul_multi(m, gu, os, 2, m->xn, m->xq_h, 1);
                matmul(m, &v[2], m->xn, m->xq_f, 1, m->tmp);
            }
            double dt = (now_ms() - t0) / ncal;
            if (rep > 0 && dt < best[variant]) best[variant] = dt;
        }
    }
    bool panel_wins = best[1] < best[0] * 0.97;
    if (getenv("HYMT2_VERBOSE"))
        fprintf(stderr, "[hymt2] repack 校准 (每层 FFN, %d 层轮流): 原生 %.2f ms vs 面板 %.2f ms -> %s\n", ncal, best[0], best[1], panel_wins ? "面板" : "原生");
    if (!panel_wins) { /* 丢弃试用面板 */
        for (int i = saved; i < m->n_panel; i++) { hy_aligned_free(m->panel_pk[i]); free(m->panel_ps[i]); }
        m->n_panel = saved;
        for (int l = 0; l < ncal; l++) {
            weight_t *ws[3] = { &m->L[l].wg, &m->L[l].wu, &m->L[l].wd };
            for (int i = 0; i < 3; i++) { ws[i]->pk = NULL; ws[i]->ps = NULL; }
        }
    }
    return true;
}

static bool repack_weights(hymt2_model *m) {
    const char *mode = getenv("HYMT2_REPACK");
    const bool force_on = mode && !strcmp(mode, "1"), force_off = mode && !strcmp(mode, "0");
    /* Q2_0C 的原生路径是标量参考实现, 必须重排; STQ1_0 按模式/实测决定 */
    bool want_stq = !force_off;
    if (want_stq && !force_on && m->n_layer > 0 && m->L[0].wg.type == GT_STQ1_0 && !calibrate_panel_wins(m)) return false;
    else if (want_stq && !force_on && m->n_layer > 0 && m->L[0].wg.type == GT_STQ1_0 && m->L[0].wg.pk == NULL) want_stq = false; /* 校准结果: 原生更快 */
    for (int i = 0; i < m->n_layer; i++) {
        weight_t *ws[7] = { &m->L[i].wq, &m->L[i].wk, &m->L[i].wv, &m->L[i].wo, &m->L[i].wg, &m->L[i].wu, &m->L[i].wd };
        for (int j = 0; j < 7; j++) {
            if (ws[j]->type == GT_Q2_0C || (ws[j]->type == GT_STQ1_0 && want_stq)) {
                if (!build_panel(m, ws[j])) return false;
                m->repacked = 1;
            }
        }
    }
    return true;
}


/* ------------------------------------------------------------------ 加速设备 (HYMT2_DEVICE, 兼容 HYMT2_VK) */

static float *gpu_out(hymt2_model *m, const weight_t *w, float *host); /* 见 matmul 段 */
static void forward_batch(hymt2_model *m, const int *toks, int nt, int pos0, bool want_logits);

static void gpu_register(hymt2_model *m, weight_t *w, hy_dweight *g) {
    if (m->n_gpu == m->cap_gpu) {
        m->cap_gpu = m->cap_gpu ? m->cap_gpu * 2 : 256;
        m->gpu_ws = (hy_dweight **)realloc(m->gpu_ws, sizeof(hy_dweight *) * (size_t)m->cap_gpu);
    }
    m->gpu_ws[m->n_gpu++] = g;
    w->gpu = g;
}

static void gpu_drop(hymt2_model *m) {
    m->use_gpu = false;
    m->use_fused = false;
    free(m->fused_layers);
    m->fused_layers = NULL;
    for (int i = 0; i < m->n_gpu; i++) hy_device_release_on(m->dev, m->gpu_ws[i]);
    m->n_gpu = 0;
    for (int l = 0; l < m->n_layer; l++) {
        weight_t *ws[7] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv, &m->L[l].wo, &m->L[l].wg, &m->L[l].wu, &m->L[l].wd };
        for (int j = 0; j < 7; j++) ws[j]->gpu = NULL;
    }
    m->outw.gpu = NULL;
    hy_device_destroy(m->dev);
    m->dev = NULL;
}

static hy_fused_layer *build_fused_layers(hymt2_model *m) {
    hy_fused_layer *fl = (hy_fused_layer *)calloc((size_t)m->n_layer, sizeof(hy_fused_layer));
    if (!fl) return NULL;
    for (int i = 0; i < m->n_layer; i++) {
        fl[i].wq = (hy_dweight *)m->L[i].wq.gpu; fl[i].wk = (hy_dweight *)m->L[i].wk.gpu;
        fl[i].wv = (hy_dweight *)m->L[i].wv.gpu; fl[i].wo = (hy_dweight *)m->L[i].wo.gpu;
        fl[i].wg = (hy_dweight *)m->L[i].wg.gpu; fl[i].wu = (hy_dweight *)m->L[i].wu.gpu;
        fl[i].wd = (hy_dweight *)m->L[i].wd.gpu;
        fl[i].attn_norm = m->L[i].attn_norm; fl[i].ffn_norm = m->L[i].ffn_norm;
        fl[i].q_norm = m->L[i].q_norm; fl[i].k_norm = m->L[i].k_norm;
        if (!fl[i].wq || !fl[i].wk || !fl[i].wv || !fl[i].wo || !fl[i].wg || !fl[i].wu || !fl[i].wd) {
            free(fl); return NULL;
        }
    }
    return fl;
}

static void fill_fused_shape(hymt2_model *m, hy_fused_args *sh, hy_fused_layer *fl) {
    memset(sh, 0, sizeof(*sh));
    sh->n_layer = m->n_layer; sh->hidden = m->hidden; sh->ffn = m->ffn;
    sh->heads = m->heads; sh->kv_heads = m->kv_heads; sh->hdim = m->hdim;
    sh->qdim = m->qdim; sh->kvdim = m->kvdim; sh->ctx = m->ctx; sh->rope_half = m->rope_half;
    sh->eps = m->eps; sh->layers = fl; sh->outw = (hy_dweight *)m->outw.gpu;
    sh->out_norm = m->out_norm; sh->rope_cos = m->rope_cos; sh->rope_sin = m->rope_sin;
}

/* CUDA/Metal: 只认整网 fused（权重/激活/KV 常驻, ≤1 sync）。
 * 旧改法「逐 matmul GPU + CPU 注意力」来回拷, 常慢于物理核 CPU, 且 fused 失败后仍留下 use_gpu=true。 */
static bool enable_fused(hymt2_model *m, bool verbose) {
    if (!m->dev || !hy_device_has_fused(m->dev) || !m->outw.gpu) return false;
    const char *fe = getenv("HYMT2_FUSED");
    int force = fe && (!strcmp(fe, "1") || !strcmp(fe, "force") || !strcmp(fe, "yes") || !strcmp(fe, "on"));
    int forbid = fe && (!strcmp(fe, "0") || !strcmp(fe, "off") || !strcmp(fe, "no"));
    if (forbid) return false;

    hy_fused_layer *fl = build_fused_layers(m);
    if (!fl) return false;
    hy_fused_args sh;
    fill_fused_shape(m, &sh, fl);
    if (!hy_device_fused_prepare(m->dev, &sh)) { free(fl); return false; }

    if (!force) {
        for (int i = 0; i < m->hidden; i++) m->h[i] = 0.01f * (float)((i * 17) % 100);
        hy_fused_args a = sh;
        a.nt = 1; a.pos0 = 0; a.want_logits = 1; a.h = m->h; a.logits = m->logits;
        double t0 = now_ms();
        for (int r = 0; r < 2; r++) {
            if (!hy_device_fused_forward(m->dev, &a)) { free(fl); return false; }
        }
        double fused_ms = (now_ms() - t0) / 2.0;
        m->use_gpu = false;
        m->use_fused = false;
        double t1 = now_ms();
        int _tok1 = 1; forward_batch(m, &_tok1, 1, 0, true);
        double cpu_ms = now_ms() - t1;
        if (fused_ms > cpu_ms * 0.9) {
            fprintf(stderr, "[hymt2] fused 校准 %.1f ms/步 vs CPU %.1f ms — 不用 %s (HYMT2_FUSED=1 可强制)\n",
                    fused_ms, cpu_ms, hy_device_name(m->dev));
            free(fl);
            return false;
        }
        fprintf(stderr, "[hymt2] fused 校准 %.1f ms/步 vs CPU %.1f ms — 启用 %s\n",
                fused_ms, cpu_ms, hy_device_name(m->dev));
    }

    free(m->fused_layers);
    m->fused_layers = fl;
    m->use_gpu = true;
    m->use_fused = true;
    if (verbose) fprintf(stderr, "[hymt2] %s 整网常驻前向已启用 (≤1 sync/步)\n", hy_device_name(m->dev));
    return true;
}

/* 上传权重; 不擅自 use_gpu=true (由 fused / matmul 校准决定)。 */
static bool device_upload_all(hymt2_model *m, bool verbose, bool forced) {
    for (int l = 0; l < m->n_layer; l++) {
        weight_t *ws[7] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv, &m->L[l].wo, &m->L[l].wg, &m->L[l].wu, &m->L[l].wd };
        for (int j = 0; j < 7; j++) if (panel_supported(ws[j]->type) && !build_panel(m, ws[j])) { gpu_drop(m); return false; }
    }
    double t0 = now_ms();
    for (int l = 0; l < m->n_layer; l++) {
        weight_t *ws[7] = { &m->L[l].wq, &m->L[l].wk, &m->L[l].wv, &m->L[l].wo, &m->L[l].wg, &m->L[l].wu, &m->L[l].wd };
        for (int j = 0; j < 7; j++) {
            hy_dweight *g = hy_device_upload(m->dev, ws[j]);
            if (!g) {
                if (verbose || forced) fprintf(stderr, "[hymt2] %s 上传失败 (层 %d 张量 %d)\n", hy_device_name(m->dev), l, j);
                gpu_drop(m);
                return false;
            }
            gpu_register(m, ws[j], g);
        }
    }
    hy_dweight *gh = hy_device_upload(m->dev, &m->outw);
    if (gh) gpu_register(m, &m->outw, gh);
    if (verbose) fprintf(stderr, "[hymt2] %s 权重上传完成 (%d 张, %.1f s)\n", hy_device_name(m->dev), m->n_gpu, (now_ms() - t0) / 1000.0);
    return true;
}

/* Vulkan/NPU 等无 fused: 只校 FFN matmul; 必须明显快于 CPU 才启用逐算子路径。 */
static double device_matmul_calibrate_ms(hymt2_model *m) {
    const int ncal = m->n_layer < 4 ? m->n_layer : 4;
    uint32_t s = 777;
    for (int i = 0; i < m->hidden; i++) { s = s * 1664525u + 1013904223u; m->xn[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    for (int i = 0; i < m->ffn; i++) { s = s * 1664525u + 1013904223u; m->gate[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    quantize_rows(m->xn, 1, m->hidden, m->xq_h);
    quantize_rows(m->gate, 1, m->ffn, m->xq_f);
    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        m->use_gpu = true;
        m->use_fused = false;
        double t1 = now_ms();
        for (int l = 0; l < ncal; l++) {
            const weight_t *gu[2] = { &m->L[l].wg, &m->L[l].wu };
            float *os[2] = { gpu_out(m, &m->L[l].wg, m->gate), gpu_out(m, &m->L[l].wu, m->up) };
            matmul_multi(m, gu, os, 2, m->xn, m->xq_h, 1);
            matmul(m, &m->L[l].wd, m->xn, m->xq_f, 1, gpu_out(m, &m->L[l].wd, m->tmp));
        }
        double dt = (now_ms() - t1) / ncal;
        if (!m->use_gpu) return 1e30;
        if (rep > 0 && dt < best) best = dt;
    }
    return best;
}

static double cpu_calibrate_ms(hymt2_model *m) {
    const int ncal = m->n_layer < 4 ? m->n_layer : 4;
    uint32_t s = 777;
    for (int i = 0; i < m->hidden; i++) { s = s * 1664525u + 1013904223u; m->xn[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    for (int i = 0; i < m->ffn; i++) { s = s * 1664525u + 1013904223u; m->gate[i] = ((float)(s >> 9) / 4194304.f) - 1.f; }
    quantize_rows(m->xn, 1, m->hidden, m->xq_h);
    quantize_rows(m->gate, 1, m->ffn, m->xq_f);
    double best = 1e30;
    for (int rep = 0; rep < 4; rep++) {
        m->use_gpu = false;
        m->use_fused = false;
        double t1 = now_ms();
        for (int l = 0; l < ncal; l++) {
            const weight_t *gu[2] = { &m->L[l].wg, &m->L[l].wu };
            float *os[2] = { m->gate, m->up };
            matmul_multi(m, gu, os, 2, m->xn, m->xq_h, 1);
            matmul(m, &m->L[l].wd, m->xn, m->xq_f, 1, m->tmp);
        }
        double dt = (now_ms() - t1) / ncal;
        if (rep > 0 && dt < best) best = dt;
    }
    return best;
}

/* HYMT2_DEVICE: auto|cpu|cuda|metal|vulkan|nnapi|qnn|rkllm|hiai|npu|force|<n>
 * auto: cuda/metal 必须整网 fused 赢 CPU ≥10%; vulkan/npu 用 matmul 校准。输了就彻底放下, 不留假加速。 */
static void setup_device(hymt2_model *m) {
    const char *pref = getenv("HYMT2_DEVICE");
    if (!pref || !*pref) pref = getenv("HYMT2_VK");
    if (pref && (!strcmp(pref, "0") || !strcmp(pref, "off") || !strcmp(pref, "cpu"))) return;
    const bool verbose = getenv("HYMT2_VERBOSE") != NULL;
    const bool forced = pref && strcmp(pref, "auto") != 0;
    char caps[512] = "";
    hy_device_capabilities(caps, sizeof caps);
    fprintf(stderr, "[hymt2] capabilities: %s\n", caps);

    if (forced) {
        char info[256] = "";
        m->dev = hy_device_create(pref, info, sizeof info);
        if (!m->dev) { fprintf(stderr, "[hymt2] 加速设备不可用: %s\n", info); return; }
        fprintf(stderr, "[hymt2] 设备(强制): %s (%s)\n", hy_device_name(m->dev), info);
        if (!device_upload_all(m, verbose, true)) return;
        if (hy_device_has_fused(m->dev)) {
            if (!enable_fused(m, verbose)) {
                /* 强制设备但 fused 不够快: 仍开逐算子, 便于调试; 日志已提示 */
                m->use_gpu = true;
                m->use_fused = false;
                fprintf(stderr, "[hymt2] 强制 %s: fused 未胜出, 回退逐算子 (可能慢于 CPU)\n", hy_device_name(m->dev));
            }
        } else {
            m->use_gpu = true;
            m->use_fused = false;
        }
        return;
    }

    static const char *cands[] = { "cuda", "metal", "vulkan", "qnn", "nnapi", "rkllm", "hiai", NULL };
    double cpu_ms = cpu_calibrate_ms(m);
    for (int i = 0; cands[i]; i++) {
        char info[256] = "";
        hy_device *d = hy_device_try(cands[i], info, sizeof info);
        if (!d) {
            if (verbose) fprintf(stderr, "[hymt2] auto skip %s: %s\n", cands[i], info);
            continue;
        }
        if (m->dev) gpu_drop(m);
        m->dev = d;
        if (!device_upload_all(m, verbose, false)) continue;

        if (hy_device_has_fused(m->dev)) {
            if (enable_fused(m, verbose)) {
                fprintf(stderr, "[hymt2] auto -> %s+fused (%s)\n", hy_device_name(d), info);
                return;
            }
            fprintf(stderr, "[hymt2] %s fused 未优于 CPU, 继续\n", hy_device_name(d));
            gpu_drop(m);
            continue;
        }

        double ms = device_matmul_calibrate_ms(m);
        fprintf(stderr, "[hymt2] 校准 %s matmul: %.2f ms/层 (CPU %.2f)\n", hy_device_name(d), ms, cpu_ms);
        if (ms < cpu_ms * 0.9) {
            m->use_gpu = true;
            m->use_fused = false;
            fprintf(stderr, "[hymt2] auto -> %s (matmul %.2f < CPU %.2f×0.9; %s)\n",
                    hy_device_name(d), ms, cpu_ms, info);
            return;
        }
        fprintf(stderr, "[hymt2] %s 未优于 CPU, 继续\n", hy_device_name(d));
        gpu_drop(m);
    }
    fprintf(stderr, "[hymt2] auto -> cpu (无设备快过物理核 CPU 10%%; CPU %.2f ms/层)\n", cpu_ms);
}

hymt2_model *hymt2_model_load(const char *path, int n_threads, int ctx_len, char *err, int err_cap) {
    kernels_init();
    hymt2_model *m = (hymt2_model *)calloc(1, sizeof(*m));
    if (!m) return NULL;
#define FAIL() do { if (err && err_cap > 0) snprintf(err, (size_t)err_cap, "%s", m->err); hymt2_model_free(m); return NULL; } while (0)
    m->g = gguf_open(path, m->err, sizeof(m->err));
    if (!m->g) FAIL();
    { const char *pf = getenv("HYMT2_PREFAULT"); if (!pf || strcmp(pf, "0") != 0) gguf_prefault(m->g); }

    gstr archs = { "hunyuan-dense", 13 };
    gguf_get_str(m->g, "general.architecture", &archs);
    char arch[64];
    snprintf(arch, sizeof(arch), "%.*s", (int)(archs.len < 63 ? archs.len : 63), archs.p);
    char key[160];
#define KI(suffix, def) (snprintf(key, sizeof(key), "%s." suffix, arch), gguf_get_int(m->g, key, (def)))
#define KF(suffix, def) (snprintf(key, sizeof(key), "%s." suffix, arch), gguf_get_float(m->g, key, (def)))
    m->hidden = (int)KI("embedding_length", 0);
    m->heads = (int)KI("attention.head_count", 0);
    m->kv_heads = (int)KI("attention.head_count_kv", m->heads);
    m->n_layer = (int)KI("block_count", 0);
    m->ffn = (int)KI("feed_forward_length", 0);
    m->hdim = (int)KI("attention.key_length", 0);
    if (m->hdim == 0 && m->heads) m->hdim = m->hidden / m->heads;
    int rope_dim = (int)KI("rope.dimension_count", m->hdim);
    float base = (float)KF("rope.freq_base", 10000.0);
    float alpha = (float)KF("rope.scaling.alpha", 0.0);
    m->eps = (float)KF("attention.layer_norm_rms_epsilon", 1e-5);
    if (alpha > 0 && m->hdim > 2) base *= powf(alpha, (float)m->hdim / (float)(m->hdim - 2));
    if (!m->hidden || !m->heads || !m->kv_heads || !m->n_layer || !m->ffn || !m->hdim || m->heads % m->kv_heads) {
        set_err(m, "bad model hyper-parameters (arch=%s)", arch);
        FAIL();
    }
    if (rope_dim <= 0 || rope_dim > m->hdim || (rope_dim & 1)) { set_err(m, "bad rope dimension (arch=%s)", arch); FAIL(); }
    m->qdim = m->heads * m->hdim;
    m->kvdim = m->kv_heads * m->hdim;
    m->ctx = ctx_len > 0 ? ctx_len : 2048;
    m->rope_half = rope_dim / 2;

    m->tok = tok_create(m->g, m->err, sizeof(m->err));
    if (!m->tok) FAIL();

    if (!load_weight(m, "token_embd.weight", &m->embd, m->hidden, 0, false)) FAIL();
    m->vocab = m->embd.n;
    if (!load_weight(m, "output.weight", &m->outw, m->hidden, 0, true)) {
        if (m->err[0]) FAIL();
        m->outw = m->embd; /* 输出头与词嵌入共用 */
    }
    m->vocab_out = m->outw.n < tok_vocab_size(m->tok) ? m->outw.n : tok_vocab_size(m->tok);
    m->out_norm = load_vec(m, "output_norm.weight", m->hidden);
    if (!m->out_norm) FAIL();

    m->L = (layer_t *)calloc((size_t)m->n_layer, sizeof(layer_t));
    char nm[96];
    for (int i = 0; i < m->n_layer; i++) {
        layer_t *l = &m->L[i];
#define W(field, suffix, k, n) (snprintf(nm, sizeof(nm), "blk.%d." suffix, i), load_weight(m, nm, &l->field, (k), (n), false))
#define V(field, suffix, n) (snprintf(nm, sizeof(nm), "blk.%d." suffix, i), (l->field = load_vec(m, nm, (n))) != NULL)
        if (!V(attn_norm, "attn_norm.weight", m->hidden) || !V(ffn_norm, "ffn_norm.weight", m->hidden) ||
            !V(q_norm, "attn_q_norm.weight", m->hdim) || !V(k_norm, "attn_k_norm.weight", m->hdim) ||
            !W(wq, "attn_q.weight", m->hidden, m->qdim) || !W(wk, "attn_k.weight", m->hidden, m->kvdim) ||
            !W(wv, "attn_v.weight", m->hidden, m->kvdim) || !W(wo, "attn_output.weight", m->qdim, m->hidden) ||
            !W(wg, "ffn_gate.weight", m->hidden, m->ffn) || !W(wu, "ffn_up.weight", m->hidden, m->ffn) ||
            !W(wd, "ffn_down.weight", m->ffn, m->hidden))
            FAIL();
    }

    /* RoPE 表 */
    m->rope_cos = (float *)malloc(sizeof(float) * (size_t)m->ctx * (size_t)m->rope_half);
    m->rope_sin = (float *)malloc(sizeof(float) * (size_t)m->ctx * (size_t)m->rope_half);
    for (int i = 0; i < m->rope_half; i++) {
        float freq = 1.0f / powf(base, (float)i / (float)m->rope_half);
        for (int p = 0; p < m->ctx; p++) {
            float a = (float)p * freq;
            m->rope_cos[(size_t)p * m->rope_half + i] = cosf(a);
            m->rope_sin[(size_t)p * m->rope_half + i] = sinf(a);
        }
    }

    m->kc = (uint16_t **)calloc((size_t)m->n_layer, sizeof(uint16_t *));
    m->vc = (uint16_t **)calloc((size_t)m->n_layer, sizeof(uint16_t *));
    for (int i = 0; i < m->n_layer; i++) {
        m->kc[i] = (uint16_t *)calloc((size_t)m->ctx * (size_t)m->kvdim, sizeof(uint16_t));
        m->vc[i] = (uint16_t *)calloc((size_t)m->ctx * (size_t)m->kvdim, sizeof(uint16_t));
    }
    m->cache_tok = (int *)malloc(sizeof(int) * (size_t)m->ctx);

    const size_t B = MAX_BATCH;
    int maxk = m->hidden > m->ffn ? m->hidden : m->ffn;
    if (m->qdim > maxk) maxk = m->qdim;
    m->h = (float *)malloc(sizeof(float) * B * (size_t)m->hidden);
    m->xn = (float *)malloc(sizeof(float) * B * (size_t)maxk);
    m->q = (float *)malloc(sizeof(float) * B * (size_t)m->qdim);
    m->k = (float *)malloc(sizeof(float) * B * (size_t)m->kvdim);
    m->v = (float *)malloc(sizeof(float) * B * (size_t)m->kvdim);
    m->ao = (float *)malloc(sizeof(float) * B * (size_t)m->qdim);
    m->gate = (float *)malloc(sizeof(float) * B * (size_t)m->ffn);
    m->up = (float *)malloc(sizeof(float) * B * (size_t)m->ffn);
    m->tmp = (float *)malloc(sizeof(float) * B * (size_t)m->hidden);
    m->scores = (float *)malloc(sizeof(float) * B * (size_t)m->heads * (size_t)m->ctx);
    m->logits = (float *)malloc(sizeof(float) * (size_t)m->outw.n);
    m->xq_h = (q8k_block *)hy_aligned_alloc(sizeof(q8k_block) * B * (size_t)(m->hidden / QK_K + 1));
    m->xq_q = (q8k_block *)hy_aligned_alloc(sizeof(q8k_block) * B * (size_t)(m->qdim / QK_K + 1));
    m->xq_f = (q8k_block *)hy_aligned_alloc(sizeof(q8k_block) * B * (size_t)(m->ffn / QK_K + 1));
    if (!m->h || !m->xn || !m->q || !m->k || !m->v || !m->ao || !m->gate || !m->up || !m->tmp || !m->scores || !m->logits ||
        !m->xq_h || !m->xq_q || !m->xq_f || !m->cache_tok) {
        set_err(m, "out of memory (%s)", "scratch");
        FAIL();
    }
    m->tp = tp_create(n_threads > 0 ? n_threads : default_threads());
    /* prefill 的 token 分块大小: 没有 AVX 的小核心 (L1 只有 24KB) 用 8, 其余 16; HYMT2_BATCH 可覆盖 */
    m->batch = (!strcmp(kernels_isa(), "ssse3") || !strcmp(kernels_isa(), "sse4.1") || !strcmp(kernels_isa(), "scalar")) ? 8 : MAX_BATCH;
    if (getenv("HYMT2_BATCH") && atoi(getenv("HYMT2_BATCH")) >= 1) m->batch = atoi(getenv("HYMT2_BATCH"));
    if (m->batch > MAX_BATCH) m->batch = MAX_BATCH;
    m->err[0] = 0;
    if (!repack_weights(m)) { snprintf(m->err, sizeof m->err, "%s", "out of memory while repacking weights"); FAIL(); }
    setup_device(m);
    if (getenv("HYMT2_VERBOSE"))
        fprintf(stderr, "[hymt2] kernel=%s threads=%d batch=%d repack=%s gpu=%s\n", kernels_isa(), tp_threads(m->tp), m->batch, m->repacked ? "panel" : "native",
                m->use_gpu ? hy_device_name(m->dev) : "off");
    return m;
#undef FAIL
}

/* ------------------------------------------------------------------ 数学 */

typedef struct { const weight_t *w; uint8_t *pk; float *ps; } repack_ctx;
static void repack_task(void *c, int b, int e) {
    repack_ctx *x = (repack_ctx *)c;
    panel_repack_rows(x->w, x->pk, x->ps, b, e);
}

/* 为一张权重生成面板 (STQ1_0 / Q2_0C), 缓冲登记到模型里统一释放 */
static bool build_panel(hymt2_model *m, weight_t *w) {
    if (w->pk || !panel_supported(w->type)) return true;
    const size_t nbl = (size_t)w->n * (size_t)(w->k / QK_K);
    uint8_t *pk = (uint8_t *)hy_aligned_alloc(nbl * 64);
    float *ps = (float *)malloc(sizeof(float) * nbl);
    if (!pk || !ps) { hy_aligned_free(pk); free(ps); return false; }
    repack_ctx c = { w, pk, ps };
    tp_for(m->tp, w->n, 16, repack_task, &c);
    if (m->n_panel == m->cap_panel) {
        m->cap_panel = m->cap_panel ? m->cap_panel * 2 : 256;
        m->panel_pk = (uint8_t **)realloc(m->panel_pk, sizeof(uint8_t *) * (size_t)m->cap_panel);
        m->panel_ps = (float **)realloc(m->panel_ps, sizeof(float *) * (size_t)m->cap_panel);
    }
    m->panel_pk[m->n_panel] = pk; m->panel_ps[m->n_panel] = ps; m->n_panel++;
    w->pk = pk; w->ps = ps;
    w->pshift = w->type == GT_Q2_0C ? 1 : 0;
    w->poff = w->type == GT_Q2_0C ? 3 : 1;
    return true;
}

/* GPU 共享/映射输出缓冲; 与 hy_device_weight_out 别名时可跳过 matmul 后 memcpy */
static float *gpu_out(hymt2_model *m, const weight_t *w, float *host) {
    if (!m->use_gpu || !m->dev || !w || !w->gpu) return host;
    float *op = hy_device_weight_out(m->dev, (hy_dweight *)w->gpu);
    return op ? op : host;
}

typedef struct { const weight_t *w; const float *xf; const q8k_block *xq; int nt; float *out; } mm_ctx;
static void mm_task(void *c, int b, int e) {
    mm_ctx *x = (mm_ctx *)c;
    matmul_rows(x->w, b, e, x->xf, x->xq, x->nt, x->out);
}
static void matmul(hymt2_model *m, const weight_t *w, const float *xf, const q8k_block *xq, int nt, float *out) {
    if (m->use_gpu && w->gpu && m->dev) {
        hy_dweight *g[1] = { (hy_dweight *)w->gpu };
        float *o[1] = { out };
        if (hy_device_matmul(m->dev, g, o, 1, xq, nt)) return;
        m->use_gpu = false; /* GPU 出错: 之后全部回退 CPU (CPU 上的权重一直保留) */
    }
    mm_ctx c = { w, xf, xq, nt, out };
    int th = tp_threads(m->tp);
    int grain = w->n / (th * 8);
    if (grain < 4) grain = 4;
    tp_for(m->tp, w->n, grain, mm_task, &c);
}

/* 多个共享同一输入的权重 (q/k/v 或 gate/up) 合并成一次并行区, 减少线程同步次数并改善负载均衡 */
typedef struct { const weight_t *w[4]; float *out[4]; int cnt, start[5], nt; const float *xf; const q8k_block *xq; } mmm_ctx;
static void mmm_task(void *c, int b, int e) {
    mmm_ctx *x = (mmm_ctx *)c;
    for (int i = 0; i < x->cnt; i++) {
        int lo = b > x->start[i] ? b : x->start[i], hi = e < x->start[i + 1] ? e : x->start[i + 1];
        if (lo < hi) matmul_rows(x->w[i], lo - x->start[i], hi - x->start[i], x->xf, x->xq, x->nt, x->out[i]);
    }
}
static void matmul_multi(hymt2_model *m, const weight_t **ws, float **outs, int cnt, const float *xf, const q8k_block *xq, int nt) {
    if (m->use_gpu && m->dev) {
        hy_dweight *g[4];
        bool all = true;
        for (int i = 0; i < cnt; i++) { g[i] = (hy_dweight *)ws[i]->gpu; if (!g[i]) all = false; }
        if (all) {
            if (hy_device_matmul(m->dev, g, outs, cnt, xq, nt)) return;
            m->use_gpu = false;
        }
    }
    mmm_ctx c;
    c.cnt = cnt; c.nt = nt; c.xf = xf; c.xq = xq; c.start[0] = 0;
    for (int i = 0; i < cnt; i++) { c.w[i] = ws[i]; c.out[i] = outs[i]; c.start[i + 1] = c.start[i] + ws[i]->n; }
    int grain = c.start[cnt] / (tp_threads(m->tp) * 8);
    if (grain < 4) grain = 4;
    tp_for(m->tp, c.start[cnt], grain, mmm_task, &c);
}

static void quantize_rows(const float *x, int nt, int k, q8k_block *y) {
    for (int t = 0; t < nt; t++) quantize_q8k(x + (size_t)t * k, k, y + (size_t)t * (k / QK_K));
}

static void rmsnorm(const float *x, const float *w, float *y, int dim, float eps) { /* y 可与 x 相同 */
    float sum = 0.f;
    int i = 0;
#if defined(HY_MODEL_NEON)
    float32x4_t vsum = vdupq_n_f32(0.f);
    for (; i + 4 <= dim; i += 4) {
        float32x4_t v = vld1q_f32(x + i);
        vsum = vfmaq_f32(vsum, v, v);
    }
    sum = vaddvq_f32(vsum);
#endif
    for (; i < dim; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / (float)dim + eps);
    i = 0;
#if defined(HY_MODEL_NEON)
    float32x4_t vs = vdupq_n_f32(scale);
    for (; i + 4 <= dim; i += 4)
        vst1q_f32(y + i, vmulq_f32(vmulq_f32(vld1q_f32(x + i), vs), vld1q_f32(w + i)));
#endif
    for (; i < dim; i++) y[i] = x[i] * scale * w[i];
}

static void residual_add(float *h, const float *tmp, int n) {
    int i = 0;
#if defined(HY_MODEL_NEON)
    for (; i + 4 <= n; i += 4)
        vst1q_f32(h + i, vaddq_f32(vld1q_f32(h + i), vld1q_f32(tmp + i)));
#endif
    for (; i < n; i++) h[i] += tmp[i];
}

static void store_bf16_pair(const float *k, const float *v, uint16_t *kd, uint16_t *vd, int n) {
    for (int d = 0; d < n; d++) { kd[d] = f32_to_bf16(k[d]); vd[d] = f32_to_bf16(v[d]); }
}

static void rope_neox(const hymt2_model *m, float *x, int heads, int pos) {
    int half = m->rope_half;
    const float *c = m->rope_cos + (size_t)pos * half, *s = m->rope_sin + (size_t)pos * half;
    for (int h = 0; h < heads; h++) {
        float *row = x + (size_t)h * m->hdim;
        for (int i = 0; i < half; i++) {
            float x0 = row[i], x1 = row[i + half];
            row[i] = x0 * c[i] - x1 * s[i];
            row[i + half] = x0 * s[i] + x1 * c[i];
        }
    }
}

typedef struct { hymt2_model *m; int layer, nt, pos0; const float *q; } attn_ctx;
static void attn_task(void *cx, int b, int e) {
    attn_ctx *a = (attn_ctx *)cx;
    hymt2_model *m = a->m;
    const float *qbase = a->q ? a->q : m->q;
    const int group = m->heads / m->kv_heads;
    const float scale = 1.0f / sqrtf((float)m->hdim);
    for (int item = b; item < e; item++) {
        int t = item / m->heads, h = item % m->heads;
        int pos = a->pos0 + t, n = pos + 1, kvh = h / group;
        const float *qh = qbase + (size_t)t * m->qdim + (size_t)h * m->hdim;
        float *sc = m->scores + (size_t)item * m->ctx;
        const uint16_t *K = m->kc[a->layer] + (size_t)kvh * m->hdim, *V = m->vc[a->layer] + (size_t)kvh * m->hdim;
        float mx = -INFINITY;
        for (int j = 0; j < n; j++) {
            float s = dot_bf16(qh, K + (size_t)j * m->kvdim, m->hdim) * scale;
            sc[j] = s;
            if (s > mx) mx = s;
        }
        float sum = 0.f;
        for (int j = 0; j < n; j++) { sc[j] = expf(sc[j] - mx); sum += sc[j]; }
        float inv = 1.0f / sum;
        float *o = m->ao + (size_t)t * m->qdim + (size_t)h * m->hdim;
        for (int d = 0; d < m->hdim; d++) o[d] = 0.f;
        for (int j = 0; j < n; j++) {
            axpy_bf16(o, V + (size_t)j * m->kvdim, sc[j] * inv, m->hdim);
        }
    }
}

/* 处理 nt (<=MAX_BATCH) 个连续位置的 token; want_logits 时把最后一个 token 的 logits 写入 m->logits */
static void forward_batch(hymt2_model *m, const int *toks, int nt, int pos0, bool want_logits) {
    PROF_BEGIN();
    const int H = m->hidden, nbh = H / QK_K, nbq = m->qdim / QK_K, nbf = m->ffn / QK_K;
    (void)nbh; (void)nbq; (void)nbf;
    for (int t = 0; t < nt; t++) dequant_row(&m->embd, toks[t], m->h + (size_t)t * H);
    PROF_END(0);

    if (m->use_fused && m->use_gpu && m->dev && m->fused_layers) {
        hy_fused_args a;
        fill_fused_shape(m, &a, m->fused_layers);
        a.nt = nt; a.pos0 = pos0; a.want_logits = want_logits ? 1 : 0;
        a.h = m->h; a.logits = m->logits;
        if (hy_device_fused_forward(m->dev, &a)) { PROF_END(6); return; }
        /* fused 运行失败: 彻底回退纯 CPU, 不留逐算子 GPU 假加速 */
        fprintf(stderr, "[hymt2] fused forward 失败, 回退 CPU\n");
        m->use_fused = false;
        m->use_gpu = false;
        m->n_cache = 0;
    }

    /* 纯 CPU: 走原始 zip 同款直写缓冲路径 (无 gpu_out 间接), 避免 23→16 tok/s 回归。
     * GPU/逐算子设备: 仍用 gpu_out 映射输出缓冲。 */
    const bool cpu_fast = !(m->use_gpu && m->dev);

    for (int li = 0; li < m->n_layer; li++) {
        const layer_t *L = &m->L[li];
        /* ---- 注意力 ---- */
        for (int t = 0; t < nt; t++) rmsnorm(m->h + (size_t)t * H, L->attn_norm, m->xn + (size_t)t * H, H, m->eps);
        quantize_rows(m->xn, nt, H, m->xq_h);
        float *qbuf = cpu_fast ? m->q : gpu_out(m, &L->wq, m->q);
        float *kbuf = cpu_fast ? m->k : gpu_out(m, &L->wk, m->k);
        float *vbuf = cpu_fast ? m->v : gpu_out(m, &L->wv, m->v);
        {
            const weight_t *ws[3] = { &L->wq, &L->wk, &L->wv };
            float *os[3] = { qbuf, kbuf, vbuf };
            matmul_multi(m, ws, os, 3, m->xn, m->xq_h, nt);
        }
        PROF_END(1);
        for (int t = 0; t < nt; t++) {
            float *q = qbuf + (size_t)t * m->qdim, *k = kbuf + (size_t)t * m->kvdim, *v = vbuf + (size_t)t * m->kvdim;
            rope_neox(m, q, m->heads, pos0 + t);
            rope_neox(m, k, m->kv_heads, pos0 + t);
            for (int h = 0; h < m->heads; h++) rmsnorm(q + (size_t)h * m->hdim, L->q_norm, q + (size_t)h * m->hdim, m->hdim, m->eps);
            for (int h = 0; h < m->kv_heads; h++) rmsnorm(k + (size_t)h * m->hdim, L->k_norm, k + (size_t)h * m->hdim, m->hdim, m->eps);
            uint16_t *kd = m->kc[li] + (size_t)(pos0 + t) * m->kvdim, *vd = m->vc[li] + (size_t)(pos0 + t) * m->kvdim;
            if (cpu_fast) {
                for (int d = 0; d < m->kvdim; d++) { kd[d] = f32_to_bf16(k[d]); vd[d] = f32_to_bf16(v[d]); }
            } else {
                store_bf16_pair(k, v, kd, vd, m->kvdim);
            }
        }
        PROF_END(2);
        attn_ctx ac = { m, li, nt, pos0, cpu_fast ? NULL : qbuf };
        tp_for(m->tp, nt * m->heads, 1, attn_task, &ac);
        PROF_END(3);
        quantize_rows(m->ao, nt, m->qdim, m->xq_q);
        float *wo_out = cpu_fast ? m->tmp : gpu_out(m, &L->wo, m->tmp);
        matmul(m, &L->wo, m->ao, m->xq_q, nt, wo_out);
        if (cpu_fast) {
            for (size_t i = 0; i < (size_t)nt * H; i++) m->h[i] += wo_out[i];
        } else {
            residual_add(m->h, wo_out, nt * H);
        }
        PROF_END(4);

        /* ---- 前馈 ---- */
        for (int t = 0; t < nt; t++) rmsnorm(m->h + (size_t)t * H, L->ffn_norm, m->xn + (size_t)t * H, H, m->eps);
        quantize_rows(m->xn, nt, H, m->xq_h);
        float *gbuf = cpu_fast ? m->gate : gpu_out(m, &L->wg, m->gate);
        float *ubuf = cpu_fast ? m->up : gpu_out(m, &L->wu, m->up);
        {
            const weight_t *ws[2] = { &L->wg, &L->wu };
            float *os[2] = { gbuf, ubuf };
            matmul_multi(m, ws, os, 2, m->xn, m->xq_h, nt);
        }
        for (size_t i = 0; i < (size_t)nt * m->ffn; i++) {
            float g = gbuf[i];
            gbuf[i] = g / (1.0f + expf(-g)) * ubuf[i];
        }
        quantize_rows(gbuf, nt, m->ffn, m->xq_f);
        float *wd_out = cpu_fast ? m->tmp : gpu_out(m, &L->wd, m->tmp);
        matmul(m, &L->wd, gbuf, m->xq_f, nt, wd_out);
        if (cpu_fast) {
            for (size_t i = 0; i < (size_t)nt * H; i++) m->h[i] += wd_out[i];
        } else {
            residual_add(m->h, wd_out, nt * H);
        }
        PROF_END(5);
    }

    if (want_logits) {
        float *last = m->h + (size_t)(nt - 1) * H;
        rmsnorm(last, m->out_norm, m->xn, H, m->eps);
        quantize_rows(m->xn, 1, H, m->xq_h);
        if (cpu_fast) {
            matmul(m, &m->outw, m->xn, m->xq_h, 1, m->logits);
        } else {
            float *lout = gpu_out(m, &m->outw, m->logits);
            matmul(m, &m->outw, m->xn, m->xq_h, 1, lout);
            if (lout != m->logits) memcpy(m->logits, lout, sizeof(float) * (size_t)m->outw.n);
        }
        PROF_END(6);
    }
}

/* 处理任意长度的 token 序列 (分块), 最后一块返回 logits */
static void forward_tokens(hymt2_model *m, const int *toks, int n, int pos0, bool want_logits) {
    for (int i = 0; i < n; i += m->batch) {
        int nt = n - i < m->batch ? n - i : m->batch;
        forward_batch(m, toks + i, nt, pos0 + i, want_logits && i + nt == n);
    }
}

/* ------------------------------------------------------------------ 生成 */

/* 贪心生成 token id。返回生成个数 (>=0), 或 -2 出错 / -3 超时; *out 由调用方 free。 */
static int gen_ids(hymt2_model *m, const char *user, int max_tokens, int timeout_ms, int **out) {
    static const char *PRE = "<｜hy_begin▁of▁sentence｜><｜hy_User｜>";
    static const char *POST = "<｜hy_Assistant｜>";
    *out = NULL;
    size_t ul = strlen(user), pl = strlen(PRE), ql = strlen(POST);
    char *prompt = (char *)malloc(pl + ul + ql + 1);
    memcpy(prompt, PRE, pl); memcpy(prompt + pl, user, ul); memcpy(prompt + pl + ul, POST, ql + 1);
    int *ids = NULL;
    int n = tok_encode(m->tok, prompt, pl + ul + ql, &ids);
    free(prompt);
    if (n <= 0) { free(ids); set_err(m, "empty prompt (%s)", "tokenizer returned nothing"); return -2; }
    if (n + max_tokens > m->ctx) max_tokens = m->ctx - n;
    if (max_tokens <= 0) { free(ids); set_err(m, "prompt too long for context (%s)", "increase ctx_len"); return -2; }

    double t0 = now_ms();
    m->st_decode_tok = 0; m->st_decode_ms = 0;
    /* 复用与上一次相同的前缀 KV; 至少重算最后一个 token 以得到 logits */
    int keep = 0;
    while (keep < m->n_cache && keep < n - 1 && m->cache_tok[keep] == ids[keep]) keep++;
    m->n_cache = keep;
    forward_tokens(m, ids + keep, n - keep, keep, true);
    memcpy(m->cache_tok + keep, ids + keep, sizeof(int) * (size_t)(n - keep));
    m->n_cache = n;
    m->st_prefill_tok = n - keep; m->st_cached_tok = keep; m->st_prefill_ms = now_ms() - t0;
    free(ids);
    double t_dec = now_ms();

    int *gen = (int *)malloc(sizeof(int) * (size_t)(max_tokens + 1));
    int ng = 0;
    for (int i = 0; i < max_tokens; i++) {
        if (timeout_ms > 0 && now_ms() - t0 > timeout_ms) { set_err(m, "timeout (%s)", "generation"); free(gen); return -3; }
        int best = 0;
        float bv = m->logits[0];
        for (int j = 1; j < m->vocab_out; j++)
            if (m->logits[j] > bv) { bv = m->logits[j]; best = j; }
        if (tok_is_stop(m->tok, best)) break;
        gen[ng++] = best;
        if (i + 1 == max_tokens) break;
        forward_tokens(m, &best, 1, m->n_cache, true);
        m->cache_tok[m->n_cache++] = best;
    }
    m->st_decode_tok = ng; m->st_decode_ms = now_ms() - t_dec;
    *out = gen;
    return ng;
}

int hymt2_model_info(const hymt2_model *m, char *out, int cap) {
    int n = snprintf(out, (size_t)cap, "kernel=%s threads=%d batch=%d repack=%s gpu=%s%s", kernels_isa(), tp_threads(m->tp), m->batch,
                     m->repacked ? "panel" : "native", m->use_gpu && m->dev ? hy_device_name(m->dev) : "off",
                     m->use_fused ? "+fused" : "");
    return n < 0 ? 0 : (n < cap ? n : cap - 1);
}

int hymt2_capabilities(char *out, int cap) {
    kernels_init();
    return hy_device_capabilities(out, cap);
}

void hymt2_model_stats(const hymt2_model *m, int *prefill_tokens, int *cached_tokens, double *prefill_ms, int *decode_tokens, double *decode_ms) {
    if (prefill_tokens) *prefill_tokens = m->st_prefill_tok;
    if (cached_tokens) *cached_tokens = m->st_cached_tok;
    if (prefill_ms) *prefill_ms = m->st_prefill_ms;
    if (decode_tokens) *decode_tokens = m->st_decode_tok;
    if (decode_ms) *decode_ms = m->st_decode_ms;
}

int hymt2_model_generate(hymt2_model *m, const char *user, char *out, int out_cap, int max_tokens, int timeout_ms) {
    if (!m) return -1;
    m->err[0] = 0;
    int *gen = NULL;
    int ng = gen_ids(m, user, max_tokens, timeout_ms, &gen);
    if (ng < 0) return ng;

    size_t cap = out_cap > 0 ? (size_t)out_cap : 0;
    char *tmp = (char *)malloc(cap + 8);
    size_t w = tok_decode_visible(m->tok, gen, ng, tmp, cap);
    free(gen);
    /* 去掉末尾被截断的 UTF-8 序列 */
    size_t e = w;
    while (e > 0 && ((uint8_t)tmp[e - 1] & 0xC0) == 0x80) e--;
    if (e > 0 && (uint8_t)tmp[e - 1] >= 0xC0) {
        uint8_t lead = (uint8_t)tmp[e - 1];
        size_t need = lead >= 0xF0 ? 4 : lead >= 0xE0 ? 3 : 2;
        if (w - (e - 1) < need) w = e - 1;
    }
    /* 去掉首尾空白 */
    size_t s = 0;
    while (s < w && (tmp[s] == ' ' || tmp[s] == '\n' || tmp[s] == '\t' || tmp[s] == '\r')) s++;
    while (w > s && (tmp[w - 1] == ' ' || tmp[w - 1] == '\n' || tmp[w - 1] == '\t' || tmp[w - 1] == '\r')) w--;
    memcpy(out, tmp + s, w - s);
    free(tmp);
    return (int)(w - s);
}

const char *hymt2_model_last_error(const hymt2_model *m) { return m ? m->err : ""; }
const char *hymt2_isa(void) { kernels_init(); return kernels_isa(); }
const char *hymt2_cpu_features(void) { kernels_init(); return kernels_cpu_features(); }

/* ---- 测试用: 暴露一次前向的 logits (见 tests/test_model.c) ---- */
HYMT2_API const float *hymt2_debug_logits(hymt2_model *m, const int *ids, int n, int *vocab_out) {
    m->n_cache = 0;
    forward_tokens(m, ids, n, 0, true);
    m->n_cache = 0;
    *vocab_out = m->vocab_out;
    return m->logits;
}
HYMT2_API int hymt2_debug_encode(hymt2_model *m, const char *text, int **ids) { return tok_encode(m->tok, text, strlen(text), ids); }

/* 测试用: 逐 token (nt=1) 前向, 用来和批量 prefill 的结果对拍 */
HYMT2_API const float *hymt2_debug_logits_stepwise(hymt2_model *m, const int *ids, int n, int *vocab_out) {
    m->n_cache = 0;
    for (int i = 0; i < n; i++) forward_batch(m, ids + i, 1, i, i == n - 1);
    m->n_cache = 0;
    *vocab_out = m->vocab_out;
    return m->logits;
}

/* 测试用: 返回贪心生成的 token id (写入 ids_out, 至多 max 个) */
HYMT2_API int hymt2_debug_generate_ids(hymt2_model *m, const char *user, int *ids_out, int max) {
    int *gen = NULL;
    int ng = gen_ids(m, user, max, 0, &gen);
    if (ng > 0) memcpy(ids_out, gen, sizeof(int) * (size_t)ng);
    free(gen);
    return ng;
}

#ifdef HYMT2_PROFILE
static double prof_now(void) { return now_ms(); }
HYMT2_API void hymt2_debug_profile_dump(void) {
    for (int i = 0; i < 7; i++) fprintf(stderr, "  %-8s %9.1f ms\n", prof_name[i], prof_ms[i]);
}
HYMT2_API void hymt2_debug_profile_reset(void) { for (int i = 0; i < 8; i++) prof_ms[i] = 0; }
#endif

/* ---- 测试用: 直接调用量化内核 (tests/test_quants.py 用 gguf-py 的独立实现对拍) ---- */
HYMT2_API int hymt2_debug_dequant_row(unsigned ggml_type, const void *data, int k, int row, float *dst) {
    int t = type_from_ggml(ggml_type);
    if (t == GT_UNSUPPORTED) return -1;
    weight_t w = { (const uint8_t *)data, t, k, row + 1, row_size(t, k), NULL, NULL, 0, 0, NULL };
    dequant_row(&w, row, dst);
    return 0;
}
/* out[t*rows + r] = dot(row r, quantize_q8k(x[t])) ; panel != 0 时先重排 (仅 STQ1_0 / Q2_0C) */
HYMT2_API int hymt2_debug_matmul(unsigned ggml_type, const void *data, int rows, int k, const float *x, int nt, float *out, int panel) {
    kernels_init();
    int t = type_from_ggml(ggml_type);
    if (t == GT_UNSUPPORTED || nt > MAX_BATCH) return -1;
    weight_t w = { (const uint8_t *)data, t, k, rows, row_size(t, k), NULL, NULL, 0, 0, NULL };
    uint8_t *pk = NULL; float *ps = NULL;
    if (panel && panel_supported(t)) {
        pk = (uint8_t *)hy_aligned_alloc((size_t)rows * (size_t)(k / QK_K) * 64);
        ps = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)(k / QK_K));
        panel_repack_rows(&w, pk, ps, 0, rows);
        w.pk = pk; w.ps = ps; w.pshift = t == GT_Q2_0C ? 1 : 0; w.poff = t == GT_Q2_0C ? 3 : 1;
    }
    q8k_block *xq = (q8k_block *)hy_aligned_alloc(sizeof(q8k_block) * (size_t)nt * (size_t)(k / QK_K));
    for (int i = 0; i < nt; i++) quantize_q8k(x + (size_t)i * k, k, xq + (size_t)i * (k / QK_K));
    matmul_rows(&w, 0, rows, x, xq, nt, out);
    hy_aligned_free(xq); hy_aligned_free(pk); free(ps);
    return 0;
}

/* backend: "metal"|"cuda"|"vulkan"|"force"(软件 Vulkan) */
HYMT2_API int hymt2_debug_device_matmul(const char *backend, unsigned ggml_type, const void *data, int rows, int k, const float *x, int nt, float *out) {
    kernels_init();
    int t = type_from_ggml(ggml_type);
    if (t == GT_UNSUPPORTED || nt > MAX_BATCH) return -1;
    char info[256] = "";
    const char *name = backend && *backend ? backend : "metal";
    hy_device *d = hy_device_try(!strcmp(name, "force") ? "force" : name, info, sizeof info);
    if (!d && !strcmp(name, "vulkan")) d = hy_device_try("force", info, sizeof info);
    if (!d) { fprintf(stderr, "[hymt2] debug device %s: %s\n", name, info); return -2; }
    weight_t w = { (const uint8_t *)data, t, k, rows, row_size(t, k), NULL, NULL, 0, 0, NULL };
    uint8_t *pk = NULL; float *ps = NULL;
    if (panel_supported(t)) {
        pk = (uint8_t *)hy_aligned_alloc((size_t)rows * (size_t)(k / QK_K) * 64);
        ps = (float *)malloc(sizeof(float) * (size_t)rows * (size_t)(k / QK_K));
        panel_repack_rows(&w, pk, ps, 0, rows);
        w.pk = pk; w.ps = ps; w.pshift = t == GT_Q2_0C ? 1 : 0; w.poff = t == GT_Q2_0C ? 3 : 1;
    }
    int rc = -3;
    hy_dweight *g = hy_device_upload(d, &w);
    if (g) {
        q8k_block *xq = (q8k_block *)hy_aligned_alloc(sizeof(q8k_block) * (size_t)nt * (size_t)(k / QK_K));
        for (int i = 0; i < nt; i++) quantize_q8k(x + (size_t)i * k, k, xq + (size_t)i * (k / QK_K));
        hy_dweight *gs[1] = { g };
        float *os[1] = { out };
        rc = hy_device_matmul(d, gs, os, 1, xq, nt) ? 0 : -4;
        hy_aligned_free(xq);
        hy_device_release_on(d, g);
    }
    hy_aligned_free(pk); free(ps);
    hy_device_destroy(d);
    return rc;
}

HYMT2_API int hymt2_debug_vk_matmul(unsigned ggml_type, const void *data, int rows, int k, const float *x, int nt, float *out) {
    return hymt2_debug_device_matmul("force", ggml_type, data, rows, k, x, nt, out);
}
