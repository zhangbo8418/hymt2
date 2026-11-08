/* CUDA 设备层: probe 始终尝试; create 仅在 HYMT2_CUDA=1 且链入 cuda_kernels.o、有 GPU 时成功。
 * fused_prepare/forward: 激活/KV/中间量常驻显存，整网一次 D2H（≤1 次同步）。 */
#include "device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <windows.h>
static void *hy_dlopen(const char *n) { return (void *)LoadLibraryA(n); }
static void hy_dlclose(void *h) { if (h) FreeLibrary((HMODULE)h); }
#else
#include <dlfcn.h>
static void *hy_dlopen(const char *n) { return dlopen(n, RTLD_LAZY); }
static void hy_dlclose(void *h) { if (h) dlclose(h); }
#endif

int hy_cuda_probe(char *out, int cap) {
    static const char *names[] = {
#if defined(_WIN32)
        "cudart64_12.dll", "cudart64_110.dll", "cudart64_102.dll", "cudart64_10.dll", "nvcuda.dll",
#else
        "libcudart.so.12", "libcudart.so.11.0", "libcudart.so", "libcuda.so.1", "libcuda.so",
#endif
        NULL
    };
    int ok = 0;
    for (int i = 0; names[i]; i++) {
        void *h = hy_dlopen(names[i]);
        if (h) { ok = 1; hy_dlclose(h); break; }
    }
    if (out && cap) snprintf(out, (size_t)cap, "%d", ok);
    return ok;
}

#ifndef HYMT2_CUDA

hy_device *hy_cuda_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    char b[8] = "0";
    hy_cuda_probe(b, sizeof b);
    if (info && cap)
        snprintf(info, cap, "本构建未链 CUDA 内核 (make CUDA=1 + nvcc); 探测 cuda=%s", b);
    return NULL;
}

#else /* HYMT2_CUDA: 需要 cuda_kernels.o + cuda_fused.o + libcudart */

#include <cuda_runtime.h>

enum { KIND_PANEL = 0, KIND_Q4K = 1, KIND_Q6K = 2 };
#define Q6K_PADDED_BYTES 212

extern int hy_cuda_launch_panel(const void *pk, const void *ps, const void *act, void *out,
                                uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total,
                                uint32_t row0, uint32_t shift, uint32_t off);
extern int hy_cuda_launch_q4k(const void *wq, const void *act, void *out,
                              uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total, uint32_t row0);
extern int hy_cuda_launch_q6k(const void *wq, const void *act, void *out,
                              uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total, uint32_t row0);
extern int hy_cuda_launch_rmsnorm(const float *x, const float *w, float *y,
                                  uint32_t n, uint32_t nt, uint32_t stride, float eps);
extern int hy_cuda_launch_rmsnorm_rows(float *x, const float *w,
                                       uint32_t n, uint32_t nrows, uint32_t stride, float eps);
extern int hy_cuda_launch_add(float *h, const float *x, uint32_t n);
extern int hy_cuda_launch_silu(float *g, const float *u, uint32_t n);
extern int hy_cuda_launch_rope(float *x, const float *cosv, const float *sinv,
                               uint32_t heads, uint32_t hdim, uint32_t nhalf,
                               uint32_t pos0, uint32_t nt);
extern int hy_cuda_launch_quantize(const float *x, void *y, uint32_t k, uint32_t nt, uint32_t nb);
extern int hy_cuda_launch_store_bf16(const float *k, const float *v, uint16_t *kc, uint16_t *vc,
                                     uint32_t kvdim, uint32_t pos0, uint32_t nt);
extern int hy_cuda_launch_attn(const float *q, const uint16_t *kc, const uint16_t *vc, float *ao,
                               uint32_t heads, uint32_t kv_heads, uint32_t hdim, uint32_t qdim,
                               uint32_t kvdim, uint32_t nt, uint32_t pos0, uint32_t ctx);

typedef struct {
    void *d_w, *d_s, *d_out;
    float *op;
    size_t w_bytes, s_bytes, out_bytes;
    int kind, n, k, nb;
    uint32_t shift, off;
    struct cuctx *c;
} cuweight;

typedef struct { void *kc, *vc; } cu_kv;

typedef struct cuctx {
    void *d_act;
    size_t act_cap;
    bool broken;
    int device;
    /* fused 工作区 */
    int f_n_layer, f_hidden, f_ffn, f_heads, f_kv_heads, f_hdim, f_qdim, f_kvdim, f_ctx, f_rope_half;
    void *d_h, *d_xn, *d_q, *d_k, *d_v, *d_ao, *d_gate, *d_up, *d_tmp;
    void *d_act_h, *d_act_q, *d_act_f;
    void *d_rope_c, *d_rope_s, *d_out_norm;
    cu_kv *kv;
    void **norm_attn, **norm_ffn, **norm_q, **norm_k;
} cuctx;

static void cu_destroy_impl(void *impl);
static hy_dweight *cu_upload(void *impl, const weight_t *w);
static void cu_release(hy_dweight *w);
static bool cu_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt);
static bool cu_is_sw(const void *impl) { (void)impl; return false; }
static float *cu_weight_out(hy_dweight *w) { return w ? ((cuweight *)w)->op : NULL; }
static bool cu_fused_prepare(void *impl, const hy_fused_args *shape);
static bool cu_fused_forward(void *impl, hy_fused_args *a);

static const hy_device_ops cu_ops = {
    cu_destroy_impl, cu_upload, cu_release, cu_matmul, cu_is_sw, cu_weight_out, cu_fused_prepare, cu_fused_forward
};

static void cu_free_fused(cuctx *c) {
    if (!c) return;
    cudaFree(c->d_h); cudaFree(c->d_xn); cudaFree(c->d_q); cudaFree(c->d_k); cudaFree(c->d_v);
    cudaFree(c->d_ao); cudaFree(c->d_gate); cudaFree(c->d_up); cudaFree(c->d_tmp);
    cudaFree(c->d_act_h); cudaFree(c->d_act_q); cudaFree(c->d_act_f);
    cudaFree(c->d_rope_c); cudaFree(c->d_rope_s); cudaFree(c->d_out_norm);
    c->d_h = c->d_xn = c->d_q = c->d_k = c->d_v = NULL;
    c->d_ao = c->d_gate = c->d_up = c->d_tmp = NULL;
    c->d_act_h = c->d_act_q = c->d_act_f = NULL;
    c->d_rope_c = c->d_rope_s = c->d_out_norm = NULL;
    if (c->kv) {
        for (int i = 0; i < c->f_n_layer; i++) { cudaFree(c->kv[i].kc); cudaFree(c->kv[i].vc); }
        free(c->kv); c->kv = NULL;
    }
    if (c->norm_attn) {
        for (int i = 0; i < c->f_n_layer; i++) {
            cudaFree(c->norm_attn[i]); cudaFree(c->norm_ffn[i]);
            cudaFree(c->norm_q[i]); cudaFree(c->norm_k[i]);
        }
        free(c->norm_attn); free(c->norm_ffn); free(c->norm_q); free(c->norm_k);
        c->norm_attn = c->norm_ffn = c->norm_q = c->norm_k = NULL;
    }
    c->f_n_layer = 0;
}

static int cu_mm(cuweight *w, void *act, void *out, int nt) {
    if (w->kind == KIND_PANEL)
        return hy_cuda_launch_panel(w->d_w, w->d_s, act, out, (uint32_t)w->n, (uint32_t)w->nb,
                                    (uint32_t)nt, (uint32_t)w->n, 0, w->shift, w->off);
    if (w->kind == KIND_Q4K)
        return hy_cuda_launch_q4k(w->d_w, act, out, (uint32_t)w->n, (uint32_t)w->nb,
                                  (uint32_t)nt, (uint32_t)w->n, 0);
    return hy_cuda_launch_q6k(w->d_w, act, out, (uint32_t)w->n, (uint32_t)w->nb,
                              (uint32_t)nt, (uint32_t)w->n, 0);
}

hy_device *hy_cuda_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
        if (info && cap) snprintf(info, cap, "无 CUDA 设备");
        return NULL;
    }
    if (cudaSetDevice(0) != cudaSuccess) {
        if (info && cap) snprintf(info, cap, "cudaSetDevice 失败");
        return NULL;
    }
    cuctx *c = (cuctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->device = 0;
    c->act_cap = (size_t)MAX_BATCH * 4096 * sizeof(q8k_block);
    if (cudaMalloc(&c->d_act, c->act_cap) != cudaSuccess) {
        if (info && cap) snprintf(info, cap, "cudaMalloc act 失败");
        free(c);
        return NULL;
    }
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (info && cap) snprintf(info, cap, "%s", prop.name);
    hy_device *d = hy_device_wrap(HY_DEV_CUDA, "cuda", c, &cu_ops, false);
    if (!d) cu_destroy_impl(c);
    return d;
}

static void cu_destroy_impl(void *impl) {
    cuctx *c = (cuctx *)impl;
    if (!c) return;
    cu_free_fused(c);
    if (c->d_act) cudaFree(c->d_act);
    free(c);
}

static void cu_release(hy_dweight *w) {
    cuweight *cw = (cuweight *)w;
    if (!cw) return;
    if (cw->d_w) cudaFree(cw->d_w);
    if (cw->d_s) cudaFree(cw->d_s);
    if (cw->d_out) cudaFree(cw->d_out);
    if (cw->op) cudaFreeHost(cw->op);
    free(cw);
}

static hy_dweight *cu_upload(void *impl, const weight_t *w) {
    cuctx *c = (cuctx *)impl;
    if (!c || c->broken) return NULL;
    int kind;
    size_t wrow;
    const int nb = w->k / QK_K;
    if ((w->type == GT_STQ1_0 || w->type == GT_Q2_0C) && w->pk) { kind = KIND_PANEL; wrow = (size_t)nb * 64; }
    else if (w->type == GT_Q4_K) { kind = KIND_Q4K; wrow = (size_t)nb * 144; }
    else if (w->type == GT_Q6_K) { kind = KIND_Q6K; wrow = (size_t)nb * Q6K_PADDED_BYTES; }
    else return NULL;

    size_t need = (size_t)MAX_BATCH * (size_t)nb * sizeof(q8k_block);
    if (need > c->act_cap) {
        cudaFree(c->d_act);
        c->act_cap = need;
        if (cudaMalloc(&c->d_act, c->act_cap) != cudaSuccess) { c->broken = true; return NULL; }
    }

    cuweight *cw = (cuweight *)calloc(1, sizeof(*cw));
    cw->c = c; cw->kind = kind; cw->n = w->n; cw->k = w->k; cw->nb = nb;
    cw->shift = (uint32_t)w->pshift; cw->off = (uint32_t)w->poff;
    cw->w_bytes = (size_t)w->n * wrow;
    cw->out_bytes = (size_t)w->n * MAX_BATCH * sizeof(float);

    uint8_t *host = (uint8_t *)malloc(cw->w_bytes);
    if (!host) { free(cw); return NULL; }
    if (kind == KIND_PANEL) memcpy(host, w->pk, cw->w_bytes);
    else if (kind == KIND_Q4K) memcpy(host, w->data, cw->w_bytes);
    else {
        for (size_t i = 0; i < (size_t)w->n * (size_t)nb; i++) {
            memcpy(host + i * Q6K_PADDED_BYTES, w->data + i * sizeof(block_q6_k), sizeof(block_q6_k));
            host[i * Q6K_PADDED_BYTES + 210] = host[i * Q6K_PADDED_BYTES + 211] = 0;
        }
    }
    if (cudaMalloc(&cw->d_w, cw->w_bytes) != cudaSuccess ||
        cudaMemcpy(cw->d_w, host, cw->w_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        free(host); cu_release((hy_dweight *)cw); return NULL;
    }
    free(host);
    if (kind == KIND_PANEL) {
        cw->s_bytes = (size_t)w->n * (size_t)nb * sizeof(float);
        if (cudaMalloc(&cw->d_s, cw->s_bytes) != cudaSuccess ||
            cudaMemcpy(cw->d_s, w->ps, cw->s_bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cu_release((hy_dweight *)cw); return NULL;
        }
    }
    if (cudaMalloc(&cw->d_out, cw->out_bytes) != cudaSuccess) {
        cu_release((hy_dweight *)cw); return NULL;
    }
    if (cudaHostAlloc((void **)&cw->op, cw->out_bytes, cudaHostAllocDefault) != cudaSuccess) {
        cu_release((hy_dweight *)cw); return NULL;
    }
    return (hy_dweight *)cw;
}

static bool cu_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    cuctx *c = (cuctx *)impl;
    if (!c || c->broken || nt < 1 || nt > MAX_BATCH) return false;
    int nb = ((cuweight *)ws[0])->nb;
    for (int i = 1; i < cnt; i++) if (((cuweight *)ws[i])->nb != nb) return false;
    size_t abytes = (size_t)nt * (size_t)nb * sizeof(q8k_block);
    if (abytes > c->act_cap) return false;
    if (cudaMemcpy(c->d_act, xq, abytes, cudaMemcpyHostToDevice) != cudaSuccess) { c->broken = true; return false; }
    for (int i = 0; i < cnt; i++) {
        cuweight *w = (cuweight *)ws[i];
        if (cu_mm(w, c->d_act, w->d_out, nt) != 0) { c->broken = true; return false; }
    }
    /* 一组 GEMM 一次同步 + D2H */
    if (cudaDeviceSynchronize() != cudaSuccess) { c->broken = true; return false; }
    for (int i = 0; i < cnt; i++) {
        cuweight *w = (cuweight *)ws[i];
        float *dst = (outs[i] && outs[i] != w->op) ? outs[i] : w->op;
        if (cudaMemcpy(dst, w->d_out, (size_t)w->n * (size_t)nt * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            c->broken = true; return false;
        }
    }
    return true;
}

static bool cu_fused_prepare(void *impl, const hy_fused_args *sh) {
    cuctx *c = (cuctx *)impl;
    if (!c || !sh) return false;
    if (c->kv && c->f_n_layer == sh->n_layer && c->f_hidden == sh->hidden && c->f_ffn == sh->ffn &&
        c->f_ctx == sh->ctx && c->f_qdim == sh->qdim && c->f_kvdim == sh->kvdim)
        return true;
    cu_free_fused(c);
    c->f_n_layer = sh->n_layer; c->f_hidden = sh->hidden; c->f_ffn = sh->ffn;
    c->f_heads = sh->heads; c->f_kv_heads = sh->kv_heads; c->f_hdim = sh->hdim;
    c->f_qdim = sh->qdim; c->f_kvdim = sh->kvdim; c->f_ctx = sh->ctx; c->f_rope_half = sh->rope_half;
    size_t H = (size_t)sh->hidden, F = (size_t)sh->ffn, B = MAX_BATCH;
    size_t Q = (size_t)sh->qdim, KV = (size_t)sh->kvdim;
#define CU_ALLOC(p, n) do { if (cudaMalloc(&(p), (n)) != cudaSuccess) { cu_free_fused(c); return false; } } while (0)
    CU_ALLOC(c->d_h, B * H * 4); CU_ALLOC(c->d_xn, B * H * 4);
    CU_ALLOC(c->d_q, B * Q * 4); CU_ALLOC(c->d_k, B * KV * 4); CU_ALLOC(c->d_v, B * KV * 4);
    CU_ALLOC(c->d_ao, B * Q * 4);
    CU_ALLOC(c->d_gate, B * F * 4); CU_ALLOC(c->d_up, B * F * 4); CU_ALLOC(c->d_tmp, B * H * 4);
    CU_ALLOC(c->d_act_h, B * (H / 256) * sizeof(q8k_block));
    CU_ALLOC(c->d_act_q, B * (Q / 256) * sizeof(q8k_block));
    CU_ALLOC(c->d_act_f, B * (F / 256) * sizeof(q8k_block));
    CU_ALLOC(c->d_rope_c, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
    CU_ALLOC(c->d_rope_s, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
    CU_ALLOC(c->d_out_norm, H * 4);
#undef CU_ALLOC
    if (cudaMemcpy(c->d_rope_c, sh->rope_cos, (size_t)sh->ctx * (size_t)sh->rope_half * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(c->d_rope_s, sh->rope_sin, (size_t)sh->ctx * (size_t)sh->rope_half * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(c->d_out_norm, sh->out_norm, H * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        cu_free_fused(c); return false;
    }
    c->kv = (cu_kv *)calloc((size_t)sh->n_layer, sizeof(cu_kv));
    c->norm_attn = (void **)calloc((size_t)sh->n_layer, sizeof(void *));
    c->norm_ffn = (void **)calloc((size_t)sh->n_layer, sizeof(void *));
    c->norm_q = (void **)calloc((size_t)sh->n_layer, sizeof(void *));
    c->norm_k = (void **)calloc((size_t)sh->n_layer, sizeof(void *));
    if (!c->kv || !c->norm_attn) { cu_free_fused(c); return false; }
    size_t kvbytes = (size_t)sh->ctx * (size_t)sh->kvdim * sizeof(uint16_t);
    for (int i = 0; i < sh->n_layer; i++) {
        if (cudaMalloc(&c->kv[i].kc, kvbytes) != cudaSuccess ||
            cudaMalloc(&c->kv[i].vc, kvbytes) != cudaSuccess ||
            cudaMalloc(&c->norm_attn[i], H * 4) != cudaSuccess ||
            cudaMalloc(&c->norm_ffn[i], H * 4) != cudaSuccess ||
            cudaMalloc(&c->norm_q[i], (size_t)sh->hdim * 4) != cudaSuccess ||
            cudaMalloc(&c->norm_k[i], (size_t)sh->hdim * 4) != cudaSuccess) {
            cu_free_fused(c); return false;
        }
        cudaMemset(c->kv[i].kc, 0, kvbytes);
        cudaMemset(c->kv[i].vc, 0, kvbytes);
        if (cudaMemcpy(c->norm_attn[i], sh->layers[i].attn_norm, H * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(c->norm_ffn[i], sh->layers[i].ffn_norm, H * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(c->norm_q[i], sh->layers[i].q_norm, (size_t)sh->hdim * 4, cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(c->norm_k[i], sh->layers[i].k_norm, (size_t)sh->hdim * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            cu_free_fused(c); return false;
        }
    }
    return true;
}

static bool cu_fused_forward(void *impl, hy_fused_args *a) {
    cuctx *c = (cuctx *)impl;
    if (!c || !a || !c->kv || c->broken || a->nt < 1 || a->nt > MAX_BATCH) return false;
    size_t hbytes = (size_t)a->nt * (size_t)a->hidden * sizeof(float);
    if (cudaMemcpy(c->d_h, a->h, hbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        c->broken = true; return false;
    }

#define FAIL() do { c->broken = true; return false; } while (0)
#define CHK(x) do { if ((x) != 0) FAIL(); } while (0)

    for (int li = 0; li < a->n_layer; li++) {
        hy_fused_layer *L = &a->layers[li];
        cuweight *wq = (cuweight *)L->wq, *wk = (cuweight *)L->wk, *wv = (cuweight *)L->wv;
        cuweight *wo = (cuweight *)L->wo, *wg = (cuweight *)L->wg, *wu = (cuweight *)L->wu, *wd = (cuweight *)L->wd;
        if (!wq || !wk || !wv || !wo || !wg || !wu || !wd) FAIL();

        CHK(hy_cuda_launch_rmsnorm((const float *)c->d_h, (const float *)c->norm_attn[li], (float *)c->d_xn,
                                   (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)a->hidden, a->eps));
        CHK(hy_cuda_launch_quantize((const float *)c->d_xn, c->d_act_h,
                                    (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)(a->hidden / 256)));
        CHK(cu_mm(wq, c->d_act_h, c->d_q, a->nt));
        CHK(cu_mm(wk, c->d_act_h, c->d_k, a->nt));
        CHK(cu_mm(wv, c->d_act_h, c->d_v, a->nt));
        CHK(hy_cuda_launch_rope((float *)c->d_q, (const float *)c->d_rope_c, (const float *)c->d_rope_s,
                                (uint32_t)a->heads, (uint32_t)a->hdim, (uint32_t)a->rope_half,
                                (uint32_t)a->pos0, (uint32_t)a->nt));
        CHK(hy_cuda_launch_rope((float *)c->d_k, (const float *)c->d_rope_c, (const float *)c->d_rope_s,
                                (uint32_t)a->kv_heads, (uint32_t)a->hdim, (uint32_t)a->rope_half,
                                (uint32_t)a->pos0, (uint32_t)a->nt));
        CHK(hy_cuda_launch_rmsnorm_rows((float *)c->d_q, (const float *)c->norm_q[li],
                                        (uint32_t)a->hdim, (uint32_t)(a->nt * a->heads), (uint32_t)a->hdim, a->eps));
        CHK(hy_cuda_launch_rmsnorm_rows((float *)c->d_k, (const float *)c->norm_k[li],
                                        (uint32_t)a->hdim, (uint32_t)(a->nt * a->kv_heads), (uint32_t)a->hdim, a->eps));
        CHK(hy_cuda_launch_store_bf16((const float *)c->d_k, (const float *)c->d_v,
                                      (uint16_t *)c->kv[li].kc, (uint16_t *)c->kv[li].vc,
                                      (uint32_t)a->kvdim, (uint32_t)a->pos0, (uint32_t)a->nt));
        CHK(hy_cuda_launch_attn((const float *)c->d_q, (const uint16_t *)c->kv[li].kc, (const uint16_t *)c->kv[li].vc,
                                (float *)c->d_ao, (uint32_t)a->heads, (uint32_t)a->kv_heads, (uint32_t)a->hdim,
                                (uint32_t)a->qdim, (uint32_t)a->kvdim, (uint32_t)a->nt, (uint32_t)a->pos0, (uint32_t)a->ctx));
        CHK(hy_cuda_launch_quantize((const float *)c->d_ao, c->d_act_q,
                                    (uint32_t)a->qdim, (uint32_t)a->nt, (uint32_t)(a->qdim / 256)));
        CHK(cu_mm(wo, c->d_act_q, c->d_tmp, a->nt));
        CHK(hy_cuda_launch_add((float *)c->d_h, (const float *)c->d_tmp, (uint32_t)(a->hidden * a->nt)));

        CHK(hy_cuda_launch_rmsnorm((const float *)c->d_h, (const float *)c->norm_ffn[li], (float *)c->d_xn,
                                   (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)a->hidden, a->eps));
        CHK(hy_cuda_launch_quantize((const float *)c->d_xn, c->d_act_h,
                                    (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)(a->hidden / 256)));
        CHK(cu_mm(wg, c->d_act_h, c->d_gate, a->nt));
        CHK(cu_mm(wu, c->d_act_h, c->d_up, a->nt));
        CHK(hy_cuda_launch_silu((float *)c->d_gate, (const float *)c->d_up, (uint32_t)(a->ffn * a->nt)));
        CHK(hy_cuda_launch_quantize((const float *)c->d_gate, c->d_act_f,
                                    (uint32_t)a->ffn, (uint32_t)a->nt, (uint32_t)(a->ffn / 256)));
        CHK(cu_mm(wd, c->d_act_f, c->d_tmp, a->nt));
        CHK(hy_cuda_launch_add((float *)c->d_h, (const float *)c->d_tmp, (uint32_t)(a->hidden * a->nt)));
    }

    if (a->want_logits && a->outw && a->logits) {
        float *last = (float *)c->d_h + (size_t)(a->nt - 1) * (size_t)a->hidden;
        CHK(hy_cuda_launch_rmsnorm(last, (const float *)c->d_out_norm, (float *)c->d_xn,
                                   (uint32_t)a->hidden, 1, (uint32_t)a->hidden, a->eps));
        CHK(hy_cuda_launch_quantize((const float *)c->d_xn, c->d_act_h,
                                    (uint32_t)a->hidden, 1, (uint32_t)(a->hidden / 256)));
        cuweight *ow = (cuweight *)a->outw;
        CHK(cu_mm(ow, c->d_act_h, ow->d_out, 1));
    }

#undef CHK
#undef FAIL

    /* 整网一次同步 + 回读 */
    if (cudaMemcpy(a->h, c->d_h, hbytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        c->broken = true; return false;
    }
    if (a->want_logits && a->outw && a->logits) {
        cuweight *ow = (cuweight *)a->outw;
        if (cudaMemcpy(a->logits, ow->d_out, sizeof(float) * (size_t)ow->n, cudaMemcpyDeviceToHost) != cudaSuccess) {
            c->broken = true; return false;
        }
    }
    return true;
}

#endif /* HYMT2_CUDA */
