/* Android NNAPI float FC: 与 tools/nnapi_device_probe 一致。
 * 关键发现 (MTK/rodin): Compilation_finish 之后若立刻 Model_free, 随后 Execution
 * 会在 libneuralnetworks MetaModel 里 bad_variant_access abort。
 * 因此每个 tile 必须把 Model 与 Compilation 一起活到 Compilation_free。 */
#if defined(__ANDROID__)
#include "nnapi_gemm.h"
#include "device.h"
#include "internal.h"
#include <android/NeuralNetworks.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { NNAPI_ROW_TILE = 1024 };

typedef struct {
    uint32_t n_accel;
    ANeuralNetworksDevice *accels[8];
    char info[192];
    float *xbuf, *ybuf, *wbuf, *bias;
    size_t xcap, ycap, wcap, bcap;
} nnapi_ctx;

typedef struct {
    ANeuralNetworksModel *model; /* 必须与 comp 同寿命 */
    ANeuralNetworksCompilation *comp;
    int rows;
} nnapi_tile;

typedef struct {
    nnapi_ctx *c;
    weight_t src;
    nnapi_tile *tiles;
    int n_tiles;
    int ready;
} nnapi_weight;

static int ensure_f32(float **p, size_t *cap, size_t need) {
    if (need <= *cap) return 1;
    float *nbuf = (float *)realloc(*p, need * sizeof(float));
    if (!nbuf) return 0;
    *p = nbuf;
    *cap = need;
    return 1;
}

static void dequant_q8k(const q8k_block *xq, int k, int nt, float *dst) {
    const int nb = k / QK_K;
    for (int t = 0; t < nt; t++) {
        for (int b = 0; b < nb; b++) {
            const q8k_block *blk = &xq[(size_t)t * (size_t)nb + (size_t)b];
            float *o = dst + (size_t)t * (size_t)k + (size_t)b * QK_K;
            const float d = blk->d;
            for (int j = 0; j < QK_K; j++) o[j] = (float)blk->qs[j] * d;
        }
    }
}

/* 成功时 *out_model / *out_comp 均非空; 失败两者都为 NULL */
static int compile_fc_const(nnapi_ctx *c, const float *W, const float *bias, int n, int k,
                            ANeuralNetworksModel **out_model, ANeuralNetworksCompilation **out_comp) {
    *out_model = NULL;
    *out_comp = NULL;
    ANeuralNetworksModel *model = NULL;
    if (ANeuralNetworksModel_create(&model) != ANEURALNETWORKS_NO_ERROR || !model) return 0;

    uint32_t dx[] = {1u, (uint32_t)k};
    uint32_t dw[] = {(uint32_t)n, (uint32_t)k};
    uint32_t db[] = {(uint32_t)n};
    uint32_t dy[] = {1u, (uint32_t)n};
    ANeuralNetworksOperandType tx = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dx, .scale = 0.f, .zeroPoint = 0};
    ANeuralNetworksOperandType tw = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dw, .scale = 0.f, .zeroPoint = 0};
    ANeuralNetworksOperandType tb = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 1, .dimensions = db, .scale = 0.f, .zeroPoint = 0};
    ANeuralNetworksOperandType ty = {.type = ANEURALNETWORKS_TENSOR_FLOAT32, .dimensionCount = 2, .dimensions = dy, .scale = 0.f, .zeroPoint = 0};
    ANeuralNetworksOperandType tact = {.type = ANEURALNETWORKS_INT32, .dimensionCount = 0, .dimensions = NULL, .scale = 0.f, .zeroPoint = 0};

    if (ANeuralNetworksModel_addOperand(model, &tx) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &tw) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &tb) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &tact) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_addOperand(model, &ty) != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksModel_free(model);
        return 0;
    }
    if (ANeuralNetworksModel_setOperandValue(model, 1, W, (size_t)n * (size_t)k * sizeof(float)) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_setOperandValue(model, 2, bias, (size_t)n * sizeof(float)) != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksModel_free(model);
        return 0;
    }
    int32_t act = ANEURALNETWORKS_FUSED_NONE;
    ANeuralNetworksModel_setOperandValue(model, 3, &act, sizeof act);

    uint32_t op_ins[] = {0, 1, 2, 3}, op_outs[] = {4};
    if (ANeuralNetworksModel_addOperation(model, ANEURALNETWORKS_FULLY_CONNECTED, 4, op_ins, 1, op_outs) != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksModel_free(model);
        return 0;
    }
    uint32_t mins[] = {0}, mouts[] = {4};
    if (ANeuralNetworksModel_identifyInputsAndOutputs(model, 1, mins, 1, mouts) != ANEURALNETWORKS_NO_ERROR ||
        ANeuralNetworksModel_finish(model) != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksModel_free(model);
        return 0;
    }

    ANeuralNetworksCompilation *comp = NULL;
    int rc = ANeuralNetworksCompilation_createForDevices(model, (const ANeuralNetworksDevice *const *)c->accels, c->n_accel, &comp);
    if (rc != ANEURALNETWORKS_NO_ERROR || !comp) {
        ANeuralNetworksModel_free(model);
        return 0;
    }
    ANeuralNetworksCompilation_setPreference(comp, ANEURALNETWORKS_PREFER_FAST_SINGLE_ANSWER);
    rc = ANeuralNetworksCompilation_finish(comp);
    if (rc != ANEURALNETWORKS_NO_ERROR) {
        ANeuralNetworksCompilation_free(comp);
        ANeuralNetworksModel_free(model);
        return 0;
    }
    /* 故意不 free model */
    *out_model = model;
    *out_comp = comp;
    return 1;
}

static void free_tile(nnapi_tile *t) {
    if (!t) return;
    if (t->comp) ANeuralNetworksCompilation_free(t->comp);
    if (t->model) ANeuralNetworksModel_free(t->model);
    t->comp = NULL;
    t->model = NULL;
}

static void free_tiles(nnapi_weight *nw) {
    if (!nw || !nw->tiles) return;
    for (int i = 0; i < nw->n_tiles; i++) free_tile(&nw->tiles[i]);
    free(nw->tiles);
    nw->tiles = NULL;
    nw->n_tiles = 0;
    nw->ready = 0;
}

static int ensure_compiled(nnapi_weight *nw) {
    if (nw->ready) return 1;
    nnapi_ctx *c = nw->c;
    const weight_t *w = &nw->src;
    int n_tiles = (w->n + NNAPI_ROW_TILE - 1) / NNAPI_ROW_TILE;
    nw->tiles = (nnapi_tile *)calloc((size_t)n_tiles, sizeof(nnapi_tile));
    if (!nw->tiles) return 0;
    if (!ensure_f32(&c->wbuf, &c->wcap, (size_t)NNAPI_ROW_TILE * (size_t)w->k)) { free_tiles(nw); return 0; }
    if (!ensure_f32(&c->bias, &c->bcap, (size_t)NNAPI_ROW_TILE)) { free_tiles(nw); return 0; }
    memset(c->bias, 0, (size_t)NNAPI_ROW_TILE * sizeof(float));

    if (getenv("HYMT2_VERBOSE"))
        fprintf(stderr, "[nnapi] compile n=%d k=%d tiles=%d\n", w->n, w->k, n_tiles);

    int ti = 0;
    for (int r0 = 0; r0 < w->n; r0 += NNAPI_ROW_TILE, ti++) {
        int rows = w->n - r0;
        if (rows > NNAPI_ROW_TILE) rows = NNAPI_ROW_TILE;
        for (int r = 0; r < rows; r++)
            dequant_row(w, r0 + r, c->wbuf + (size_t)r * (size_t)w->k);
        ANeuralNetworksModel *model = NULL;
        ANeuralNetworksCompilation *comp = NULL;
        if (!compile_fc_const(c, c->wbuf, c->bias, rows, w->k, &model, &comp)) {
            if (getenv("HYMT2_VERBOSE"))
                fprintf(stderr, "[nnapi] compile fail rows=%d k=%d r0=%d\n", rows, w->k, r0);
            free_tiles(nw);
            return 0;
        }
        nw->tiles[ti].model = model;
        nw->tiles[ti].comp = comp;
        nw->tiles[ti].rows = rows;
    }
    nw->n_tiles = n_tiles;
    nw->ready = 1;
    return 1;
}

static void nnapi_destroy(void *impl) {
    nnapi_ctx *c = (nnapi_ctx *)impl;
    if (!c) return;
    free(c->xbuf);
    free(c->ybuf);
    free(c->wbuf);
    free(c->bias);
    free(c);
}

static hy_dweight *nnapi_upload(void *impl, const weight_t *w) {
    nnapi_ctx *c = (nnapi_ctx *)impl;
    if (!c || !w || w->n < 1 || w->k < 1 || (w->k % QK_K) != 0) return NULL;
    nnapi_weight *nw = (nnapi_weight *)calloc(1, sizeof(*nw));
    if (!nw) return NULL;
    nw->c = c;
    nw->src = *w;
    nw->src.gpu = NULL;
    return (hy_dweight *)nw;
}

static void nnapi_release(hy_dweight *w) {
    nnapi_weight *nw = (nnapi_weight *)w;
    if (!nw) return;
    free_tiles(nw);
    free(nw);
}

static bool nnapi_run_tile(ANeuralNetworksCompilation *comp, const float *x, float *y, int k, int n) {
    ANeuralNetworksExecution *exec = NULL;
    if (ANeuralNetworksExecution_create(comp, &exec) != ANEURALNETWORKS_NO_ERROR || !exec) return false;
    int ok = 1;
    if (ANeuralNetworksExecution_setInput(exec, 0, NULL, x, (size_t)k * sizeof(float)) != ANEURALNETWORKS_NO_ERROR) ok = 0;
    if (ok && ANeuralNetworksExecution_setOutput(exec, 0, NULL, y, (size_t)n * sizeof(float)) != ANEURALNETWORKS_NO_ERROR) ok = 0;
    if (ok && ANeuralNetworksExecution_compute(exec) != ANEURALNETWORKS_NO_ERROR) ok = 0;
    ANeuralNetworksExecution_free(exec);
    return ok != 0;
}

static bool nnapi_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    nnapi_ctx *c = (nnapi_ctx *)impl;
    if (!c || !ws || !outs || !xq || cnt < 1 || nt < 1) return false;

    const int k0 = ((nnapi_weight *)ws[0])->src.k;
    for (int i = 1; i < cnt; i++)
        if (((nnapi_weight *)ws[i])->src.k != k0) return false;

    if (!ensure_f32(&c->xbuf, &c->xcap, (size_t)nt * (size_t)k0)) return false;
    dequant_q8k(xq, k0, nt, c->xbuf);
    if (!ensure_f32(&c->ybuf, &c->ycap, (size_t)NNAPI_ROW_TILE)) return false;

    for (int i = 0; i < cnt; i++) {
        nnapi_weight *nw = (nnapi_weight *)ws[i];
        if (!ensure_compiled(nw)) return false;
        for (int t = 0; t < nt; t++) {
            const float *xt = c->xbuf + (size_t)t * (size_t)k0;
            float *yt = outs[i] + (size_t)t * (size_t)nw->src.n;
            int r0 = 0;
            for (int ti = 0; ti < nw->n_tiles; ti++) {
                int rows = nw->tiles[ti].rows;
                if (!nnapi_run_tile(nw->tiles[ti].comp, xt, c->ybuf, k0, rows)) return false;
                memcpy(yt + r0, c->ybuf, (size_t)rows * sizeof(float));
                r0 += rows;
            }
        }
    }
    return true;
}

static bool nnapi_is_sw(const void *impl) {
    (void)impl;
    return false;
}

static const hy_device_ops nnapi_ops = { nnapi_destroy, nnapi_upload, nnapi_release, nnapi_matmul, nnapi_is_sw, NULL, NULL, NULL };

int hy_nnapi_probe(void) {
    uint32_t n = 0;
    return ANeuralNetworks_getDeviceCount(&n) == ANEURALNETWORKS_NO_ERROR && n > 0;
}

hy_device *hy_nnapi_create(char *info, size_t cap) {
    nnapi_ctx *c = (nnapi_ctx *)calloc(1, sizeof(*c));
    if (!c) return NULL;

    uint32_t ndev = 0;
    if (ANeuralNetworks_getDeviceCount(&ndev) != ANEURALNETWORKS_NO_ERROR) {
        if (info && cap) snprintf(info, cap, "nnapi: getDeviceCount 失败");
        free(c);
        return NULL;
    }

    char names[160] = "";
    size_t nlen = 0;
    for (uint32_t i = 0; i < ndev; i++) {
        ANeuralNetworksDevice *dev = NULL;
        if (ANeuralNetworks_getDevice(i, &dev) != ANEURALNETWORKS_NO_ERROR || !dev) continue;
        int32_t dtype = -1;
        ANeuralNetworksDevice_getType(dev, &dtype);
        if (dtype != ANEURALNETWORKS_DEVICE_ACCELERATOR) continue;
        const char *nm = NULL;
        ANeuralNetworksDevice_getName(dev, &nm);
        if (c->n_accel < 8) {
            c->accels[c->n_accel++] = dev;
            if (nm && nlen + strlen(nm) + 2 < sizeof names)
                nlen += (size_t)snprintf(names + nlen, sizeof names - nlen, "%s%s", nlen ? "," : "", nm);
        }
    }
    if (c->n_accel == 0) {
        if (info && cap) snprintf(info, cap, "nnapi: 无 ACCELERATOR");
        free(c);
        return NULL;
    }

    {
        float W[6] = {1, 0, 0, 1, 0.5f, 0.5f}, b[3] = {0}, x[2] = {1, 2}, y[3] = {0};
        ANeuralNetworksModel *model = NULL;
        ANeuralNetworksCompilation *comp = NULL;
        if (!compile_fc_const(c, W, b, 3, 2, &model, &comp)) {
            if (info && cap) snprintf(info, cap, "nnapi: 冒烟编译失败 [%s]", names);
            free(c);
            return NULL;
        }
        int ok = nnapi_run_tile(comp, x, y, 2, 3);
        ANeuralNetworksCompilation_free(comp);
        ANeuralNetworksModel_free(model);
        if (!ok) {
            if (info && cap) snprintf(info, cap, "nnapi: 冒烟 compute 失败 out=[%.3f %.3f %.3f]", y[0], y[1], y[2]);
            free(c);
            return NULL;
        }
        if (getenv("HYMT2_VERBOSE"))
            fprintf(stderr, "[nnapi] smoke ok out=[%.3f %.3f %.3f]\n", y[0], y[1], y[2]);
    }

    snprintf(c->info, sizeof c->info, "nnapi accel=%u [%s] const-FC tile=%d", c->n_accel, names, NNAPI_ROW_TILE);
    if (info && cap) snprintf(info, cap, "%s", c->info);
    hy_device *d = hy_device_wrap(HY_DEV_NPU, "nnapi", c, &nnapi_ops, false);
    if (!d) nnapi_destroy(c);
    return d;
}

#endif /* __ANDROID__ */
