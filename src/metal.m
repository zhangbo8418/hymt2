/* Metal 后端: 真实 create/upload/matmul (panel / Q4_K / Q6_K), 与 vulkan.c 同布局。 */
#if defined(__APPLE__) && defined(HYMT2_METAL)

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "device.h"
#include "metal_shaders.h"
#include "metal_fused_shaders.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { KIND_PANEL = 0, KIND_Q4K = 1, KIND_Q6K = 2 };
enum {
    P_PANEL = 0, P_Q4K, P_Q6K, P_RMS, P_ADD, P_SILU, P_ROPE, P_QUANT, P_STORE, P_ATTN, P_RMS_ROWS, P_COUNT
};
#define Q6K_PADDED_BYTES 212
#define MAX_BATCH_M 16

typedef struct {
    uint32_t rows, nb, nt, n_total, row0, shift, off, pad0;
} pushc;

typedef struct {
    id<MTLBuffer> wb, sb;
    uint32_t rows, row0;
} mtpart;

typedef struct {
    struct mtctx *c;
    int kind, n, k, nb;
    uint32_t shift, off;
    mtpart *parts;
    int np;
    id<MTLBuffer> ob;
    float *op; /* ob.contents */
} mtweight;

typedef struct { id<MTLBuffer> kc, vc; } mt_kv;

typedef struct mtctx {
    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    id<MTLComputePipelineState> pipe[P_COUNT];
    id<MTLBuffer> act;
    id<MTLBuffer> dummy;
    size_t act_cap;
    bool broken;
    bool fused_ok;
    int f_n_layer, f_hidden, f_ffn, f_heads, f_kv_heads, f_hdim, f_qdim, f_kvdim, f_ctx, f_rope_half;
    id<MTLBuffer> buf_h, buf_xn, buf_q, buf_k, buf_v, buf_ao, buf_gate, buf_up, buf_tmp;
    id<MTLBuffer> buf_act_h, buf_act_q, buf_act_f, buf_rope_c, buf_rope_s, buf_out_norm;
    __strong id<MTLBuffer> *norm_attn, *norm_ffn, *norm_q, *norm_k;
    mt_kv *kv;
} mtctx;

static void mt_destroy_impl(void *impl);
static hy_dweight *mt_upload(void *impl, const weight_t *w);
static void mt_release(hy_dweight *w);
static bool mt_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt);
static bool mt_is_sw(const void *impl) { (void)impl; return false; }
static float *mt_weight_out(hy_dweight *w) { return w ? ((mtweight *)w)->op : NULL; }
static bool mt_fused_prepare(void *impl, const hy_fused_args *shape);
static bool mt_fused_forward(void *impl, hy_fused_args *a);

static const hy_device_ops mt_ops = {
    mt_destroy_impl, mt_upload, mt_release, mt_matmul, mt_is_sw, mt_weight_out, mt_fused_prepare, mt_fused_forward
};

int hy_metal_probe(char *out, int cap) {
    @autoreleasepool {
        id<MTLDevice> d = MTLCreateSystemDefaultDevice();
        int ok = d != nil ? 1 : 0;
        if (out && cap) snprintf(out, (size_t)cap, "%d", ok);
        return ok;
    }
}

static mtctx *mt_create_ctx(char *info, size_t cap) {
    @autoreleasepool {
        mtctx *c = (mtctx *)calloc(1, sizeof(*c));
        if (!c) return NULL;
        c->dev = MTLCreateSystemDefaultDevice();
        if (!c->dev) {
            if (info) snprintf(info, cap, "无 Metal 设备");
            free(c);
            return NULL;
        }
        c->queue = [c->dev newCommandQueue];
        NSError *err = nil;
        NSMutableString *src = [NSMutableString stringWithUTF8String:HY_MSL_SRC];
        [src appendString:[NSString stringWithUTF8String:HY_MSL_FUSED]];
        c->lib = [c->dev newLibraryWithSource:src options:nil error:&err];
        if (!c->lib) {
            if (info) snprintf(info, cap, "MSL 编译失败 (src=%luB): %s", (unsigned long)src.length,
                               err ? [[err localizedDescription] UTF8String] : "?");
            free(c);
            return NULL;
        }
        const char *names[P_COUNT] = {
            "hy_panel", "hy_q4k", "hy_q6k", "hy_rmsnorm", "hy_add", "hy_silu_mul",
            "hy_rope_neox", "hy_quantize_q8k", "hy_store_bf16", "hy_attn", "hy_rmsnorm_rows"
        };
        for (int i = 0; i < P_COUNT; i++) {
            id<MTLFunction> fn = [c->lib newFunctionWithName:[NSString stringWithUTF8String:names[i]]];
            if (!fn) {
                if (info) snprintf(info, cap, "缺少 kernel %s (src=%luB err=%s)", names[i], (unsigned long)src.length,
                                   err ? [[err localizedDescription] UTF8String] : "nil");
                free(c);
                return NULL;
            }
            c->pipe[i] = [c->dev newComputePipelineStateWithFunction:fn error:&err];
            if (!c->pipe[i]) {
                if (info) snprintf(info, cap, "pipeline %s: %s", names[i], err ? [[err localizedDescription] UTF8String] : "?");
                free(c);
                return NULL;
            }
        }
        c->fused_ok = true;
        c->act_cap = (size_t)MAX_BATCH_M * 8192 * sizeof(q8k_block); /* 上限: 大 hidden 时再扩 */
        if (c->act_cap < (size_t)MAX_BATCH_M * 64 * sizeof(q8k_block))
            c->act_cap = (size_t)MAX_BATCH_M * 256 * sizeof(q8k_block);
        c->act_cap = (size_t)MAX_BATCH_M * 4096 * sizeof(q8k_block);
        c->act = [c->dev newBufferWithLength:c->act_cap options:MTLResourceStorageModeShared];
        c->dummy = [c->dev newBufferWithLength:16 options:MTLResourceStorageModeShared];
        if (info) snprintf(info, cap, "%s", [[c->dev name] UTF8String]);
        /* retain for C lifetime */
        c->dev = c->dev;
        return c;
    }
}

hy_device *hy_metal_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    char buf[256] = "";
    mtctx *c = mt_create_ctx(buf, sizeof buf);
    if (info && cap) snprintf(info, cap, "%s", buf);
    if (!c) return NULL;
    hy_device *d = hy_device_wrap(HY_DEV_METAL, "metal", c, &mt_ops, false);
    if (!d) mt_destroy_impl(c);
    return d;
}

static void mt_destroy_impl(void *impl) {
    mtctx *c = (mtctx *)impl;
    if (!c) return;
    @autoreleasepool {
        c->act = nil; c->dummy = nil;
        c->buf_h = nil; c->buf_xn = nil; c->buf_q = nil; c->buf_k = nil; c->buf_v = nil;
        c->buf_ao = nil; c->buf_gate = nil; c->buf_up = nil; c->buf_tmp = nil;
        c->buf_act_h = nil; c->buf_act_q = nil; c->buf_act_f = nil;
        c->buf_rope_c = nil; c->buf_rope_s = nil; c->buf_out_norm = nil;
        if (c->kv) {
            for (int i = 0; i < c->f_n_layer; i++) { c->kv[i].kc = nil; c->kv[i].vc = nil; }
            free(c->kv);
        }
        if (c->norm_attn) {
            for (int i = 0; i < c->f_n_layer; i++) {
                c->norm_attn[i] = nil; c->norm_ffn[i] = nil; c->norm_q[i] = nil; c->norm_k[i] = nil;
            }
            free(c->norm_attn); free(c->norm_ffn); free(c->norm_q); free(c->norm_k);
        }
        for (int i = 0; i < P_COUNT; i++) c->pipe[i] = nil;
        c->lib = nil; c->queue = nil; c->dev = nil;
    }
    free(c);
}

static void mt_release(hy_dweight *w) {
    mtweight *mw = (mtweight *)w;
    if (!mw) return;
    @autoreleasepool {
        for (int i = 0; i < mw->np; i++) { mw->parts[i].wb = nil; mw->parts[i].sb = nil; }
        mw->ob = nil;
    }
    free(mw->parts);
    free(mw);
}

static hy_dweight *mt_upload(void *impl, const weight_t *w) {
    mtctx *c = (mtctx *)impl;
    if (!c || c->broken) return NULL;
    int kind;
    size_t wrow;
    const int nb = w->k / QK_K;
    if ((w->type == GT_STQ1_0 || w->type == GT_Q2_0C) && w->pk) { kind = KIND_PANEL; wrow = (size_t)nb * 64; }
    else if (w->type == GT_Q4_K) { kind = KIND_Q4K; wrow = (size_t)nb * 144; }
    else if (w->type == GT_Q6_K) { kind = KIND_Q6K; wrow = (size_t)nb * Q6K_PADDED_BYTES; }
    else return NULL;

    @autoreleasepool {
        /* 需要更大 act 时扩容 */
        size_t need_act = (size_t)MAX_BATCH_M * (size_t)nb * sizeof(q8k_block);
        if (need_act > c->act_cap) {
            c->act_cap = need_act;
            c->act = [c->dev newBufferWithLength:c->act_cap options:MTLResourceStorageModeShared];
        }

        const size_t rows_per = (size_t)w->n; /* Metal 共享缓冲, 一次一块 */
        mtweight *mw = (mtweight *)calloc(1, sizeof(*mw));
        mw->c = c; mw->kind = kind; mw->n = w->n; mw->k = w->k; mw->nb = nb;
        mw->shift = (uint32_t)w->pshift; mw->off = (uint32_t)w->poff;
        mw->np = 1;
        mw->parts = (mtpart *)calloc(1, sizeof(mtpart));
        mw->parts[0].row0 = 0;
        mw->parts[0].rows = (uint32_t)w->n;

        size_t out_bytes = (size_t)w->n * MAX_BATCH_M * sizeof(float);
        mw->ob = [c->dev newBufferWithLength:out_bytes options:MTLResourceStorageModeShared];
        mw->op = (float *)mw->ob.contents;

        size_t wbytes = (size_t)w->n * wrow;
        NSMutableData *wdata = [NSMutableData dataWithLength:wbytes];
        uint8_t *dst = (uint8_t *)wdata.mutableBytes;
        if (kind == KIND_PANEL) memcpy(dst, w->pk, wbytes);
        else if (kind == KIND_Q4K) memcpy(dst, w->data, wbytes);
        else {
            for (size_t i = 0; i < (size_t)w->n * (size_t)nb; i++) {
                const uint8_t *src = w->data + i * sizeof(block_q6_k);
                uint8_t *d = dst + i * Q6K_PADDED_BYTES;
                memcpy(d, src, sizeof(block_q6_k)); d[210] = d[211] = 0;
            }
        }
        mw->parts[0].wb = [c->dev newBufferWithBytes:wdata.bytes length:wbytes options:MTLResourceStorageModeShared];
        if (kind == KIND_PANEL) {
            size_t sbytes = (size_t)w->n * (size_t)nb * sizeof(float);
            mw->parts[0].sb = [c->dev newBufferWithBytes:w->ps length:sbytes options:MTLResourceStorageModeShared];
        } else {
            mw->parts[0].sb = c->dummy;
        }
        (void)rows_per;
        return (hy_dweight *)mw;
    }
}

static bool mt_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    mtctx *c = (mtctx *)impl;
    if (!c || c->broken || nt < 1 || nt > MAX_BATCH_M) return false;
    const int nb = ((mtweight *)ws[0])->nb;
    for (int i = 1; i < cnt; i++) if (((mtweight *)ws[i])->nb != nb) return false;
    size_t abytes = (size_t)nt * (size_t)nb * sizeof(q8k_block);
    if (abytes > c->act_cap) return false;

    @autoreleasepool {
        memcpy(c->act.contents, xq, abytes);
        [c->act didModifyRange:NSMakeRange(0, abytes)];
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        int last = -1;
        for (int i = 0; i < cnt; i++) {
            mtweight *w = (mtweight *)ws[i];
            if (w->kind != last) {
                [enc setComputePipelineState:c->pipe[w->kind]];
                last = w->kind;
            }
            for (int p = 0; p < w->np; p++) {
                mtpart *pt = &w->parts[p];
                pushc pc = { pt->rows, (uint32_t)nb, (uint32_t)nt, (uint32_t)w->n, pt->row0, w->shift, w->off, 0 };
                [enc setBuffer:pt->wb offset:0 atIndex:0];
                [enc setBuffer:pt->sb offset:0 atIndex:1];
                [enc setBuffer:c->act offset:0 atIndex:2];
                [enc setBuffer:w->ob offset:0 atIndex:3];
                [enc setBytes:&pc length:sizeof(pc) atIndex:4];
                MTLSize tg = MTLSizeMake(64, 1, 1);
                MTLSize grid = MTLSizeMake((pt->rows + 7u) / 8u, (NSUInteger)nt, 1);
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            }
        }
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted]; /* 整组 GEMM 只同步一次 */
        if (cb.error) { c->broken = true; return false; }
        for (int i = 0; i < cnt; i++) {
            mtweight *w = (mtweight *)ws[i];
            if (!outs[i] || outs[i] == w->op) continue; /* 零拷贝: 调用方直接读 shared out */
            memcpy(outs[i], w->op, (size_t)w->n * (size_t)nt * sizeof(float));
        }
        return true;
    }
}

static id<MTLBuffer> mt_newbuf(mtctx *c, size_t n) {
    return [c->dev newBufferWithLength:n options:MTLResourceStorageModeShared];
}

static void mt_enc_mm(mtctx *c, id<MTLComputeCommandEncoder> enc, mtweight *w, id<MTLBuffer> act, id<MTLBuffer> outb, int nt) {
    [enc setComputePipelineState:c->pipe[w->kind]];
    for (int p = 0; p < w->np; p++) {
        mtpart *pt = &w->parts[p];
        pushc pc = { pt->rows, (uint32_t)w->nb, (uint32_t)nt, (uint32_t)w->n, pt->row0, w->shift, w->off, 0 };
        [enc setBuffer:pt->wb offset:0 atIndex:0];
        [enc setBuffer:pt->sb offset:0 atIndex:1];
        [enc setBuffer:act offset:0 atIndex:2];
        [enc setBuffer:outb offset:0 atIndex:3];
        [enc setBytes:&pc length:sizeof(pc) atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake((pt->rows + 7u) / 8u, (NSUInteger)nt, 1)
            threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    }
}

static void mt_enc_rms_batch(mtctx *c, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> w,
                             id<MTLBuffer> y, uint32_t n, uint32_t nt, uint32_t stride, float eps) {
    struct { uint32_t n, nt, stride, pad; float eps; } pc = { n, nt, stride, 0, eps };
    [enc setComputePipelineState:c->pipe[P_RMS]];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBytes:&pc length:sizeof(pc) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, nt, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static void mt_enc_rms_rows(mtctx *c, id<MTLComputeCommandEncoder> enc, id<MTLBuffer> x, id<MTLBuffer> w,
                            uint32_t n, uint32_t nrows, uint32_t stride, float eps) {
    struct { uint32_t n, nrows, stride, pad; float eps; } pc = { n, nrows, stride, 0, eps };
    [enc setComputePipelineState:c->pipe[P_RMS_ROWS]];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBytes:&pc length:sizeof(pc) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(nrows, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

static bool mt_fused_prepare(void *impl, const hy_fused_args *sh) {
    mtctx *c = (mtctx *)impl;
    if (!c || !sh || !c->fused_ok) return false;
    @autoreleasepool {
        if (c->kv && c->f_n_layer == sh->n_layer && c->f_hidden == sh->hidden && c->f_ffn == sh->ffn &&
            c->f_ctx == sh->ctx && c->f_qdim == sh->qdim && c->f_kvdim == sh->kvdim)
            return true;
        c->buf_h = nil; c->buf_xn = nil; c->buf_q = nil; c->buf_k = nil; c->buf_v = nil;
        c->buf_ao = nil; c->buf_gate = nil; c->buf_up = nil; c->buf_tmp = nil;
        c->buf_act_h = nil; c->buf_act_q = nil; c->buf_act_f = nil;
        c->buf_rope_c = nil; c->buf_rope_s = nil; c->buf_out_norm = nil;
        if (c->kv) {
            for (int i = 0; i < c->f_n_layer; i++) { c->kv[i].kc = nil; c->kv[i].vc = nil; }
            free(c->kv); c->kv = NULL;
        }
        if (c->norm_attn) {
            for (int i = 0; i < c->f_n_layer; i++) {
                c->norm_attn[i] = nil; c->norm_ffn[i] = nil; c->norm_q[i] = nil; c->norm_k[i] = nil;
            }
            free(c->norm_attn); free(c->norm_ffn); free(c->norm_q); free(c->norm_k);
            c->norm_attn = c->norm_ffn = c->norm_q = c->norm_k = NULL;
        }
        c->f_n_layer = sh->n_layer; c->f_hidden = sh->hidden; c->f_ffn = sh->ffn;
        c->f_heads = sh->heads; c->f_kv_heads = sh->kv_heads; c->f_hdim = sh->hdim;
        c->f_qdim = sh->qdim; c->f_kvdim = sh->kvdim; c->f_ctx = sh->ctx; c->f_rope_half = sh->rope_half;
        size_t H = (size_t)sh->hidden, F = (size_t)sh->ffn, B = MAX_BATCH_M;
        c->buf_h = mt_newbuf(c, B * H * 4); c->buf_xn = mt_newbuf(c, B * H * 4);
        c->buf_q = mt_newbuf(c, B * (size_t)sh->qdim * 4);
        c->buf_k = mt_newbuf(c, B * (size_t)sh->kvdim * 4);
        c->buf_v = mt_newbuf(c, B * (size_t)sh->kvdim * 4);
        c->buf_ao = mt_newbuf(c, B * (size_t)sh->qdim * 4);
        c->buf_gate = mt_newbuf(c, B * F * 4); c->buf_up = mt_newbuf(c, B * F * 4);
        c->buf_tmp = mt_newbuf(c, B * H * 4);
        c->buf_act_h = mt_newbuf(c, B * (H / 256) * sizeof(q8k_block));
        c->buf_act_q = mt_newbuf(c, B * ((size_t)sh->qdim / 256) * sizeof(q8k_block));
        c->buf_act_f = mt_newbuf(c, B * (F / 256) * sizeof(q8k_block));
        c->buf_rope_c = mt_newbuf(c, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
        c->buf_rope_s = mt_newbuf(c, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
        memcpy(c->buf_rope_c.contents, sh->rope_cos, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
        memcpy(c->buf_rope_s.contents, sh->rope_sin, (size_t)sh->ctx * (size_t)sh->rope_half * 4);
        c->buf_out_norm = mt_newbuf(c, H * 4);
        memcpy(c->buf_out_norm.contents, sh->out_norm, H * 4);
        c->kv = (mt_kv *)calloc((size_t)sh->n_layer, sizeof(mt_kv));
        c->norm_attn = (__strong id<MTLBuffer> *)calloc((size_t)sh->n_layer, sizeof(id<MTLBuffer>));
        c->norm_ffn = (__strong id<MTLBuffer> *)calloc((size_t)sh->n_layer, sizeof(id<MTLBuffer>));
        c->norm_q = (__strong id<MTLBuffer> *)calloc((size_t)sh->n_layer, sizeof(id<MTLBuffer>));
        c->norm_k = (__strong id<MTLBuffer> *)calloc((size_t)sh->n_layer, sizeof(id<MTLBuffer>));
        if (!c->kv || !c->norm_attn) return false;
        size_t kvbytes = (size_t)sh->ctx * (size_t)sh->kvdim * 2;
        for (int i = 0; i < sh->n_layer; i++) {
            c->kv[i].kc = mt_newbuf(c, kvbytes); c->kv[i].vc = mt_newbuf(c, kvbytes);
            c->norm_attn[i] = mt_newbuf(c, H * 4); c->norm_ffn[i] = mt_newbuf(c, H * 4);
            c->norm_q[i] = mt_newbuf(c, (size_t)sh->hdim * 4); c->norm_k[i] = mt_newbuf(c, (size_t)sh->hdim * 4);
            memcpy(c->norm_attn[i].contents, sh->layers[i].attn_norm, H * 4);
            memcpy(c->norm_ffn[i].contents, sh->layers[i].ffn_norm, H * 4);
            memcpy(c->norm_q[i].contents, sh->layers[i].q_norm, (size_t)sh->hdim * 4);
            memcpy(c->norm_k[i].contents, sh->layers[i].k_norm, (size_t)sh->hdim * 4);
        }
        return true;
    }
}

static bool mt_fused_forward(void *impl, hy_fused_args *a) {
    mtctx *c = (mtctx *)impl;
    if (!c || !a || !c->kv || c->broken || a->nt < 1 || a->nt > MAX_BATCH_M) return false;
    @autoreleasepool {
        size_t hbytes = (size_t)a->nt * (size_t)a->hidden * sizeof(float);
        memcpy(c->buf_h.contents, a->h, hbytes);
        [c->buf_h didModifyRange:NSMakeRange(0, hbytes)];
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (int li = 0; li < a->n_layer; li++) {
            hy_fused_layer *L = &a->layers[li];
            mtweight *wq = (mtweight *)L->wq, *wk = (mtweight *)L->wk, *wv = (mtweight *)L->wv;
            mtweight *wo = (mtweight *)L->wo, *wg = (mtweight *)L->wg, *wu = (mtweight *)L->wu, *wd = (mtweight *)L->wd;
            if (!wq || !wk || !wv || !wo || !wg || !wu || !wd) { [enc endEncoding]; return false; }

            mt_enc_rms_batch(c, enc, c->buf_h, c->norm_attn[li], c->buf_xn,
                             (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)a->hidden, a->eps);
            {
                struct { uint32_t k, nt, nb, pad; } qpc =
                    { (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)(a->hidden / 256), 0 };
                [enc setComputePipelineState:c->pipe[P_QUANT]];
                [enc setBuffer:c->buf_xn offset:0 atIndex:0];
                [enc setBuffer:c->buf_act_h offset:0 atIndex:1];
                [enc setBytes:&qpc length:sizeof(qpc) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(qpc.nb, (NSUInteger)a->nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            mt_enc_mm(c, enc, wq, c->buf_act_h, c->buf_q, a->nt);
            mt_enc_mm(c, enc, wk, c->buf_act_h, c->buf_k, a->nt);
            mt_enc_mm(c, enc, wv, c->buf_act_h, c->buf_v, a->nt);
            {
                struct { uint32_t heads, hdim, nhalf, pos0, nt, pad0; } rpc =
                    { (uint32_t)a->heads, (uint32_t)a->hdim, (uint32_t)a->rope_half, (uint32_t)a->pos0, (uint32_t)a->nt, 0 };
                /* 旧: nhalf×heads×nt 个 TG、每 TG 1 线程 → Iris 上调度开销爆炸 */
                NSUInteger nh = (NSUInteger)a->rope_half;
                NSUInteger tgx = nh < 64 ? nh : 64;
                [enc setComputePipelineState:c->pipe[P_ROPE]];
                [enc setBuffer:c->buf_q offset:0 atIndex:0];
                [enc setBuffer:c->buf_rope_c offset:0 atIndex:1];
                [enc setBuffer:c->buf_rope_s offset:0 atIndex:2];
                [enc setBytes:&rpc length:sizeof(rpc) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((nh + tgx - 1) / tgx, (NSUInteger)a->heads, (NSUInteger)a->nt)
                    threadsPerThreadgroup:MTLSizeMake(tgx, 1, 1)];
                rpc.heads = (uint32_t)a->kv_heads;
                [enc setBuffer:c->buf_k offset:0 atIndex:0];
                [enc setBytes:&rpc length:sizeof(rpc) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake((nh + tgx - 1) / tgx, (NSUInteger)a->kv_heads, (NSUInteger)a->nt)
                    threadsPerThreadgroup:MTLSizeMake(tgx, 1, 1)];
            }
            /* q: nt*heads 行, stride=hdim; k: nt*kv_heads 行 */
            mt_enc_rms_rows(c, enc, c->buf_q, c->norm_q[li], (uint32_t)a->hdim,
                            (uint32_t)(a->nt * a->heads), (uint32_t)a->hdim, a->eps);
            mt_enc_rms_rows(c, enc, c->buf_k, c->norm_k[li], (uint32_t)a->hdim,
                            (uint32_t)(a->nt * a->kv_heads), (uint32_t)a->hdim, a->eps);
            {
                struct { uint32_t kvdim, pos0, nt, pad; } spc =
                    { (uint32_t)a->kvdim, (uint32_t)a->pos0, (uint32_t)a->nt, 0 };
                [enc setComputePipelineState:c->pipe[P_STORE]];
                [enc setBuffer:c->buf_k offset:0 atIndex:0];
                [enc setBuffer:c->buf_v offset:0 atIndex:1];
                [enc setBuffer:c->kv[li].kc offset:0 atIndex:2];
                [enc setBuffer:c->kv[li].vc offset:0 atIndex:3];
                [enc setBytes:&spc length:sizeof(spc) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake(((NSUInteger)a->kvdim + 63) / 64, (NSUInteger)a->nt, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            {
                struct { uint32_t heads, kv_heads, hdim, qdim, kvdim, nt, pos0, ctx; } apc =
                    { (uint32_t)a->heads, (uint32_t)a->kv_heads, (uint32_t)a->hdim, (uint32_t)a->qdim,
                      (uint32_t)a->kvdim, (uint32_t)a->nt, (uint32_t)a->pos0, (uint32_t)a->ctx };
                [enc setComputePipelineState:c->pipe[P_ATTN]];
                [enc setBuffer:c->buf_q offset:0 atIndex:0];
                [enc setBuffer:c->kv[li].kc offset:0 atIndex:1];
                [enc setBuffer:c->kv[li].vc offset:0 atIndex:2];
                [enc setBuffer:c->buf_ao offset:0 atIndex:3];
                [enc setBytes:&apc length:sizeof(apc) atIndex:4];
                [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)a->heads, (NSUInteger)a->nt, 1)
                    threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            {
                struct { uint32_t k, nt, nb, pad; } qpc =
                    { (uint32_t)a->qdim, (uint32_t)a->nt, (uint32_t)(a->qdim / 256), 0 };
                [enc setComputePipelineState:c->pipe[P_QUANT]];
                [enc setBuffer:c->buf_ao offset:0 atIndex:0];
                [enc setBuffer:c->buf_act_q offset:0 atIndex:1];
                [enc setBytes:&qpc length:sizeof(qpc) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(qpc.nb, (NSUInteger)a->nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            mt_enc_mm(c, enc, wo, c->buf_act_q, c->buf_tmp, a->nt);
            {
                struct { uint32_t n, nt, stride, pad; float eps; } pc = { (uint32_t)(a->hidden * a->nt), 1, 0, 0, 0 };
                [enc setComputePipelineState:c->pipe[P_ADD]];
                [enc setBuffer:c->buf_h offset:0 atIndex:0];
                [enc setBuffer:c->buf_tmp offset:0 atIndex:1];
                [enc setBytes:&pc length:sizeof(pc) atIndex:2];
                NSUInteger n = (NSUInteger)pc.n;
                [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            mt_enc_rms_batch(c, enc, c->buf_h, c->norm_ffn[li], c->buf_xn,
                             (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)a->hidden, a->eps);
            {
                struct { uint32_t k, nt, nb, pad; } qpc =
                    { (uint32_t)a->hidden, (uint32_t)a->nt, (uint32_t)(a->hidden / 256), 0 };
                [enc setComputePipelineState:c->pipe[P_QUANT]];
                [enc setBuffer:c->buf_xn offset:0 atIndex:0];
                [enc setBuffer:c->buf_act_h offset:0 atIndex:1];
                [enc setBytes:&qpc length:sizeof(qpc) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(qpc.nb, (NSUInteger)a->nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            mt_enc_mm(c, enc, wg, c->buf_act_h, c->buf_gate, a->nt);
            mt_enc_mm(c, enc, wu, c->buf_act_h, c->buf_up, a->nt);
            {
                struct { uint32_t n, nt, stride, pad; float eps; } pc = { (uint32_t)(a->ffn * a->nt), 1, 0, 0, 0 };
                [enc setComputePipelineState:c->pipe[P_SILU]];
                [enc setBuffer:c->buf_gate offset:0 atIndex:0];
                [enc setBuffer:c->buf_up offset:0 atIndex:1];
                [enc setBytes:&pc length:sizeof(pc) atIndex:2];
                NSUInteger n = (NSUInteger)pc.n;
                [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            {
                struct { uint32_t k, nt, nb, pad; } qpc =
                    { (uint32_t)a->ffn, (uint32_t)a->nt, (uint32_t)(a->ffn / 256), 0 };
                [enc setComputePipelineState:c->pipe[P_QUANT]];
                [enc setBuffer:c->buf_gate offset:0 atIndex:0];
                [enc setBuffer:c->buf_act_f offset:0 atIndex:1];
                [enc setBytes:&qpc length:sizeof(qpc) atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(qpc.nb, (NSUInteger)a->nt, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            }
            mt_enc_mm(c, enc, wd, c->buf_act_f, c->buf_tmp, a->nt);
            {
                struct { uint32_t n, nt, stride, pad; float eps; } pc = { (uint32_t)(a->hidden * a->nt), 1, 0, 0, 0 };
                [enc setComputePipelineState:c->pipe[P_ADD]];
                [enc setBuffer:c->buf_h offset:0 atIndex:0];
                [enc setBuffer:c->buf_tmp offset:0 atIndex:1];
                [enc setBytes:&pc length:sizeof(pc) atIndex:2];
                NSUInteger n = (NSUInteger)pc.n;
                [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
        }

        if (a->want_logits && a->outw && a->logits) {
            NSUInteger off = (NSUInteger)(a->nt - 1) * (NSUInteger)a->hidden * 4;
            /* 单 token：用 buffer offset 指向最后一行 */
            {
                struct { uint32_t n, nt, stride, pad; float eps; } pc =
                    { (uint32_t)a->hidden, 1, (uint32_t)a->hidden, 0, a->eps };
                [enc setComputePipelineState:c->pipe[P_RMS]];
                [enc setBuffer:c->buf_h offset:off atIndex:0];
                [enc setBuffer:c->buf_out_norm offset:0 atIndex:1];
                [enc setBuffer:c->buf_xn offset:0 atIndex:2];
                [enc setBytes:&pc length:sizeof(pc) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }
            struct { uint32_t k, nt, nb, pad; } qpc = { (uint32_t)a->hidden, 1, (uint32_t)(a->hidden / 256), 0 };
            [enc setComputePipelineState:c->pipe[P_QUANT]];
            [enc setBuffer:c->buf_xn offset:0 atIndex:0];
            [enc setBuffer:c->buf_act_h offset:0 atIndex:1];
            [enc setBytes:&qpc length:sizeof(qpc) atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(qpc.nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
            mt_enc_mm(c, enc, (mtweight *)a->outw, c->buf_act_h, ((mtweight *)a->outw)->ob, 1);
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted]; /* 整网一次 GPU 同步 */
        if (cb.error) { c->broken = true; return false; }
        /* 下一步会重新 embed 覆盖 h, 无需整缓冲 D2H; 只要 logits */
        if (a->want_logits && a->outw && a->logits)
            memcpy(a->logits, ((mtweight *)a->outw)->op, sizeof(float) * (size_t)((mtweight *)a->outw)->n);
        (void)hbytes;
        return true;
    }
}

#else /* 非 Apple 或未开 HYMT2_METAL */

#include "device.h"
#include <stdio.h>
#include <string.h>
#if defined(__APPLE__)
#include <dlfcn.h>
int hy_metal_probe(char *out, int cap) {
    void *h = dlopen("/System/Library/Frameworks/Metal.framework/Metal", RTLD_LAZY);
    int ok = h != NULL;
    if (h) dlclose(h);
    if (out && cap) snprintf(out, (size_t)cap, "%d", ok);
    return ok;
}
hy_device *hy_metal_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    if (info && cap) snprintf(info, cap, "请用 make METAL=1 (Darwin) 启用 Metal 内核");
    return NULL;
}
#else
int hy_metal_probe(char *out, int cap) {
    if (out && cap) snprintf(out, (size_t)cap, "0");
    return 0;
}
hy_device *hy_metal_create(const char *pref, char *info, size_t cap) {
    (void)pref;
    if (info && cap) snprintf(info, cap, "Metal 仅支持 macOS/iOS");
    return NULL;
}
#endif
#endif
