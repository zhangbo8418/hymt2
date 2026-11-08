/* 统一设备层: HYMT2_DEVICE (兼容 HYMT2_VK); auto 时由 model.c 多候选校准选优。 */
#include "device.h"
#include "internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

hy_device *hy_cuda_create(const char *pref, char *info, size_t cap);
int hy_cuda_probe(char *out, int cap);
hy_device *hy_metal_create(const char *pref, char *info, size_t cap);
int hy_metal_probe(char *out, int cap);
int hy_npu_probe(char *out, int cap);
hy_device *hy_npu_create(const char *pref, char *info, size_t cap);
/* 细分 NPU 强制名 */
hy_device *hy_npu_create_named(const char *name, char *info, size_t cap);

struct hy_device {
    hy_dev_kind kind;
    void *impl;
    bool software;
    char name[64];
    hy_device_ops ops;
};

hy_device *hy_device_wrap(hy_dev_kind kind, const char *name, void *impl, const hy_device_ops *ops, bool software) {
    if (!impl || !ops) return NULL;
    hy_device *d = (hy_device *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->kind = kind;
    d->impl = impl;
    d->software = software;
    d->ops = *ops;
    snprintf(d->name, sizeof d->name, "%s", name ? name : "gpu");
    return d;
}

/* ---- Vulkan 适配 ---- */
static void vk_adapt_destroy_impl(void *impl) { vk_destroy((vkctx *)impl); }
static hy_dweight *vk_adapt_upload(void *impl, const weight_t *w) {
    return (hy_dweight *)vk_upload((vkctx *)impl, w);
}
static void vk_adapt_release(hy_dweight *w) { vk_release((vkweight *)w); }
static bool vk_adapt_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    return vk_matmul((vkctx *)impl, (vkweight *const *)ws, outs, cnt, xq, nt);
}
static bool vk_adapt_is_sw(const void *impl) { return vk_is_software((const vkctx *)impl); }
static float *vk_adapt_weight_out(hy_dweight *w) { return vk_weight_out((vkweight *)w); }

static const hy_device_ops vk_ops = {
    vk_adapt_destroy_impl, vk_adapt_upload, vk_adapt_release, vk_adapt_matmul, vk_adapt_is_sw, vk_adapt_weight_out, NULL, NULL
};

static hy_device *wrap_vulkan(const char *pref, char *info, size_t cap) {
    char buf[256] = "";
    vkctx *c = vk_create(pref, buf, sizeof buf);
    if (info && cap) snprintf(info, cap, "%s", buf);
    if (!c) return NULL;
    hy_device *d = hy_device_wrap(HY_DEV_VULKAN, "vulkan", c, &vk_ops, vk_is_software(c));
    if (!d) vk_destroy(c);
    return d;
}

static bool is_off(const char *p) {
    return p && (!strcmp(p, "0") || !strcmp(p, "off") || !strcmp(p, "cpu") || !strcmp(p, "none"));
}

static const char *resolve_pref(void) {
    const char *d = getenv("HYMT2_DEVICE");
    if (d && *d) return d;
    const char *v = getenv("HYMT2_VK");
    if (v && *v) return v;
    return "auto";
}

hy_device *hy_device_try(const char *name, char *info, size_t cap) {
    if (!name || !*name) return NULL;
    char buf[256] = "";
    hy_device *d = NULL;
    if (!strcmp(name, "cuda")) d = hy_cuda_create("auto", buf, sizeof buf);
    else if (!strcmp(name, "metal")) d = hy_metal_create("auto", buf, sizeof buf);
    else if (!strcmp(name, "vulkan")) {
        d = wrap_vulkan("auto", buf, sizeof buf);
        if (d && d->software) { hy_device_destroy(d); d = NULL; snprintf(buf, sizeof buf, "software vulkan skipped in auto"); }
    }
    else if (!strcmp(name, "force") || !strcmp(name, "vulkan-force"))
        d = wrap_vulkan("force", buf, sizeof buf);
    else if (!strcmp(name, "nnapi") || !strcmp(name, "qnn") || !strcmp(name, "rkllm") || !strcmp(name, "hiai") || !strcmp(name, "npu"))
        d = hy_npu_create_named(name, buf, sizeof buf);
    else if (name[0] >= '0' && name[0] <= '9')
        d = wrap_vulkan(name, buf, sizeof buf);
    if (info && cap) snprintf(info, cap, "%s", buf[0] ? buf : (d ? "ok" : "unavailable"));
    return d;
}

hy_device *hy_device_create(const char *pref, char *info, size_t cap) {
    if (!pref || !*pref) pref = resolve_pref();
    if (is_off(pref)) {
        if (info && cap) snprintf(info, cap, "device=cpu (显式关闭加速器)");
        return NULL;
    }
    if (!strcmp(pref, "list")) {
        char caps[512];
        hy_device_capabilities(caps, sizeof caps);
        if (info && cap) snprintf(info, cap, "%s", caps);
        fprintf(stderr, "[hymt2] devices: %s\n", caps);
        wrap_vulkan("list", info, cap);
        return NULL;
    }

    /* auto: 只返回「第一个能 create 的」; 多候选校准在 model.c setup_device */
    if (!strcmp(pref, "auto")) {
        static const char *order[] = { "cuda", "metal", "vulkan", "qnn", "nnapi", "rkllm", "hiai", NULL };
        char buf[256] = "";
        for (int i = 0; order[i]; i++) {
            hy_device *d = hy_device_try(order[i], buf, sizeof buf);
            if (d) {
                if (info && cap) snprintf(info, cap, "%s (%s)", hy_device_name(d), buf);
                return d;
            }
        }
        if (info && cap) snprintf(info, cap, "无可用加速器, 使用 CPU");
        return NULL;
    }

    return hy_device_try(pref, info, cap);
}

hy_dev_kind hy_device_kind(const hy_device *d) { return d ? d->kind : HY_DEV_NONE; }
const char *hy_device_name(const hy_device *d) {
    if (!d) return "cpu";
    return d->name[0] ? d->name : "gpu";
}
bool hy_device_is_software(const hy_device *d) {
    if (!d) return false;
    if (d->ops.is_software) return d->ops.is_software(d->impl);
    return d->software;
}
void hy_device_destroy(hy_device *d) {
    if (!d) return;
    if (d->ops.destroy_impl) d->ops.destroy_impl(d->impl);
    free(d);
}
hy_dweight *hy_device_upload(hy_device *d, const weight_t *w) {
    return d && d->ops.upload ? d->ops.upload(d->impl, w) : NULL;
}
void hy_device_release_on(hy_device *d, hy_dweight *w) {
    if (d && d->ops.release) d->ops.release(w);
}
bool hy_device_matmul(hy_device *d, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    return d && d->ops.matmul ? d->ops.matmul(d->impl, ws, outs, cnt, xq, nt) : false;
}
float *hy_device_weight_out(hy_device *d, hy_dweight *w) {
    if (!d || !w || !d->ops.weight_out) return NULL;
    return d->ops.weight_out(w);
}
bool hy_device_has_fused(const hy_device *d) {
    return d && d->ops.fused_prepare && d->ops.fused_forward;
}
bool hy_device_fused_prepare(hy_device *d, const hy_fused_args *shape) {
    return d && d->ops.fused_prepare ? d->ops.fused_prepare(d->impl, shape) : false;
}
bool hy_device_fused_forward(hy_device *d, hy_fused_args *a) {
    return d && d->ops.fused_forward ? d->ops.fused_forward(d->impl, a) : false;
}

int hy_device_capabilities(char *out, int cap) {
    char cuda[32] = "0", metal[32] = "0", npu[96] = "0";
    hy_cuda_probe(cuda, sizeof cuda);
    hy_metal_probe(metal, sizeof metal);
    hy_npu_probe(npu, sizeof npu);
#ifndef HYMT2_NO_VULKAN
    const char *vk = "1";
#else
    const char *vk = "0";
#endif
    int n = snprintf(out, (size_t)cap, "isa=%s cuda=%s metal=%s vulkan=%s npu=%s",
                     kernels_isa(), cuda, metal, vk, npu);
    return n < 0 ? 0 : (n < cap ? n : cap - 1);
}
