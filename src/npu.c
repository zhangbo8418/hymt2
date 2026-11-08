/* NPU 插件: nnapi / qnn / rkllm / hiai。
 * - probe: 真库/真 SDK 路径；主机交叉编可探测 aarch64 包是否就位（sdk=）。
 * - create: Android 上 NNAPI 走真实 ANeuralNetworks*；无运行时绝不假加速。
 * - QNN/RKLLM/HiAI: 库就位时可 create 运行时句柄；matmul 需目标 SoC + 已转换模型，
 *   未接线前 upload/matmul 失败，auto 不会靠假路径加速。 */
#include "device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if defined(_WIN32)
#include <windows.h>
#ifndef S_ISREG
#define S_ISREG(m) (((m) & _S_IFMT) == _S_IFREG)
#endif
#ifndef S_ISDIR
#define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#endif
static void *hy_dlopen(const char *n) { return (void *)LoadLibraryA(n); }
static void *hy_dlsym(void *h, const char *n) { return (void *)GetProcAddress((HMODULE)h, n); }
static void hy_dlclose(void *h) { if (h) FreeLibrary((HMODULE)h); }
#else
#include <dlfcn.h>
#include <unistd.h>
static void *hy_dlopen(const char *n) { return dlopen(n, RTLD_LAZY); }
static void *hy_dlsym(void *h, const char *n) { return dlsym(h, n); }
static void hy_dlclose(void *h) { if (h) dlclose(h); }
#endif

static int file_ok(const char *p) {
    if (!p || !*p) return 0;
#if defined(_WIN32)
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p, &st) == 0 && S_ISREG(st.st_mode);
#endif
}
static int dir_ok(const char *p) {
    if (!p || !*p) return 0;
#if defined(_WIN32)
    DWORD a = GetFileAttributesA(p);
    return a != INVALID_FILE_ATTRIBUTES && (a & FILE_ATTRIBUTE_DIRECTORY);
#else
    struct stat st;
    return stat(p, &st) == 0 && S_ISDIR(st.st_mode);
#endif
}

/* 相对本仓库 third_party/sdks：可执行文件旁、或 HYMT2_SDK_ROOT / 源码布局。 */
static void sdk_root(char *out, size_t cap) {
    const char *e = getenv("HYMT2_SDK_ROOT");
    if (e && *e) { snprintf(out, cap, "%s", e); return; }
    e = getenv("HYMT2_ROOT");
    if (e && *e) { snprintf(out, cap, "%s/third_party/sdks", e); return; }
    /* 常见开发布局: .../hymt2/libhymt2.dylib → .../hymt2/third_party/sdks */
    snprintf(out, cap, "third_party/sdks");
}

static int find_under(char *out, size_t cap, const char *root, const char *rel) {
    if (!root || !*root || !rel) return 0;
    snprintf(out, cap, "%s/%s", root, rel);
    return file_ok(out) || dir_ok(out);
}

/* ---- NNAPI ---- */
#if defined(__ANDROID__)
#include "nnapi_gemm.h"
#else
static int probe_nnapi_sdk(void) {
    char root[512], path[768];
    const char *ndk = getenv("ANDROID_NDK_HOME");
    static const char *tails[] = {
        "toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include/android/NeuralNetworks.h",
        "toolchains/llvm/prebuilt/darwin-arm64/sysroot/usr/include/android/NeuralNetworks.h",
        "toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/android/NeuralNetworks.h",
        NULL
    };
    if (ndk && *ndk) {
        for (int i = 0; tails[i]; i++) {
            snprintf(path, sizeof path, "%s/%s", ndk, tails[i]);
            if (file_ok(path)) return 1;
        }
    }
    sdk_root(root, sizeof root);
    static const char *local[] = {
        "ndk/android-ndk-r27c/toolchains/llvm/prebuilt/darwin-x86_64/sysroot/usr/include/android/NeuralNetworks.h",
        "ndk/android-ndk-r27c/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/android/NeuralNetworks.h",
        NULL
    };
    for (int i = 0; local[i]; i++) {
        snprintf(path, sizeof path, "%s/%s", root, local[i]);
        if (file_ok(path)) return 1;
    }
    return 0;
}
static hy_device *create_nnapi(char *info, size_t cap) {
    if (probe_nnapi_sdk()) {
        if (info && cap)
            snprintf(info, cap, "nnapi: NDK/头文件已就绪; 需在 Android 进程加载 libneuralnetworks.so");
    } else if (info && cap) {
        snprintf(info, cap, "nnapi: 无 NDK 头文件且非 Android 运行时");
    }
    return NULL;
}
#endif

/* ---- QNN ---- */
static int read_env_file_key(const char *path, const char *key, char *out, size_t cap) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    size_t klen = strlen(key);
    char line[768];
    int ok = 0;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, key, klen) != 0 || line[klen] != '=') continue;
        char *v = line + klen + 1;
        size_t n = strlen(v);
        while (n && (v[n - 1] == '\n' || v[n - 1] == '\r')) v[--n] = 0;
        snprintf(out, cap, "%s", v);
        ok = 1;
        break;
    }
    fclose(f);
    return ok;
}

static int locate_qnn(char *libpath, size_t cap) {
    char held[512], sdk[512], try[768];
    const char *root = getenv("QNN_SDK_ROOT");
    if (!root || !*root) root = getenv("QAIRT_SDK_ROOT");
    held[0] = 0;
    if (!root || !*root) {
        sdk_root(sdk, sizeof sdk);
        snprintf(try, sizeof try, "%s/qnn/current", sdk);
        if (dir_ok(try)) {
            snprintf(held, sizeof held, "%s", try);
            root = held;
        } else {
            snprintf(try, sizeof try, "%s/qnn/env.sh", sdk);
            if (read_env_file_key(try, "QNN_SDK_ROOT", held, sizeof held)) root = held;
        }
    }
    if (!root || !*root) return 0;
    static const char *rels[] = {
        "lib/aarch64-android/libQnnHtp.so",
        "lib/x86_64-linux-clang/libQnnHtp.so",
        "lib/aarch64-android/libQnnCpu.so",
        "lib/x86_64-linux-clang/libQnnCpu.so",
        NULL
    };
    for (int i = 0; rels[i]; i++) {
        snprintf(try, sizeof try, "%s/%s", root, rels[i]);
        if (file_ok(try)) {
            if (libpath && cap) snprintf(libpath, cap, "%s", try);
            return 1;
        }
    }
    return 0;
}

static int probe_qnn(void) {
    char p[768];
    if (!locate_qnn(p, sizeof p)) return 0;
    void *h = hy_dlopen(p);
    if (h) { hy_dlclose(h); return 2; } /* 本机可加载 */
    return 1; /* SDK 文件在，架构不匹配则仍计为 sdk 就位 */
}

/* ---- RKLLM ---- */
static int locate_rkllm(char *libpath, size_t cap) {
    const char *r = getenv("RKLLM_LIB");
    if (r && file_ok(r)) { if (libpath && cap) snprintf(libpath, cap, "%s", r); return 1; }
    char sdk[512], try[768];
    sdk_root(sdk, sizeof sdk);
    static const char *cands[] = {
        "rknn/rkllm-runtime/Linux/librkllm_api/aarch64/librkllmrt.so",
        "rknn/rkllm-runtime/Android/librkllm_api/arm64-v8a/librkllmrt.so",
        "rknn/librkllmrt.so",
        NULL
    };
    for (int i = 0; cands[i]; i++) {
        if (find_under(try, sizeof try, sdk, cands[i]) && file_ok(try)) {
            if (libpath && cap) snprintf(libpath, cap, "%s", try);
            return 1;
        }
    }
    void *h = hy_dlopen("librkllmrt.so");
    if (h) {
        hy_dlclose(h);
        if (libpath && cap) snprintf(libpath, cap, "librkllmrt.so");
        return 1;
    }
    return 0;
}

static int probe_rkllm(void) {
    char p[768];
    if (!locate_rkllm(p, sizeof p)) return 0;
    void *h = hy_dlopen(p);
    if (h) {
        void *sym = hy_dlsym(h, "rkllm_init");
        hy_dlclose(h);
        return sym ? 2 : 1;
    }
    return 1;
}

/* ---- HiAI / MindSpore Lite NPU ---- */
static int locate_hiai(char *libpath, size_t cap) {
    const char *d = getenv("HWHIAI_DDK");
    char sdk[512], try[768];
    if (d && *d) {
        snprintf(try, sizeof try, "%s/lib64/libhiai.so", d);
        if (file_ok(try)) { if (libpath && cap) snprintf(libpath, cap, "%s", try); return 1; }
        snprintf(try, sizeof try, "%s/lib/libhiai.so", d);
        if (file_ok(try)) { if (libpath && cap) snprintf(libpath, cap, "%s", try); return 1; }
    }
    char held_ms[512];
    held_ms[0] = 0;
    const char *ms = getenv("MINDSPORE_LITE_HOME");
    sdk_root(sdk, sizeof sdk);
    if (!ms || !*ms) {
        snprintf(try, sizeof try, "%s/hiai/mindspore-lite-2.12.0-android-aarch64", sdk);
        if (dir_ok(try)) {
            snprintf(held_ms, sizeof held_ms, "%s", try);
            ms = held_ms;
        } else {
            snprintf(try, sizeof try, "%s/hiai/env.sh", sdk);
            if (read_env_file_key(try, "MINDSPORE_LITE_HOME", held_ms, sizeof held_ms)) ms = held_ms;
        }
    }
    if (ms && *ms) {
        snprintf(try, sizeof try, "%s/runtime/third_party/hiai_ddk/lib/libhiai.so", ms);
        if (file_ok(try)) { if (libpath && cap) snprintf(libpath, cap, "%s", try); return 1; }
        snprintf(try, sizeof try, "%s/runtime/lib/libmindspore-lite.so", ms);
        if (file_ok(try)) { if (libpath && cap) snprintf(libpath, cap, "%s", try); return 1; }
    }
    void *h = hy_dlopen("libhiai.so");
    if (h) { hy_dlclose(h); if (libpath && cap) snprintf(libpath, cap, "libhiai.so"); return 1; }
    return 0;
}

static int probe_hiai(void) {
    char p[768];
    if (!locate_hiai(p, sizeof p)) return 0;
    void *h = hy_dlopen(p);
    if (h) { hy_dlclose(h); return 2; }
    return 1;
}

int hy_npu_probe(char *out, int cap) {
#if defined(__ANDROID__)
    int nn = hy_nnapi_probe() ? 2 : 0;
#else
    int nn = probe_nnapi_sdk() ? 1 : 0;
#endif
    int qn = probe_qnn(), rk = probe_rkllm(), hi = probe_hiai();
    char buf[128];
    buf[0] = 0;
    char *p = buf;
    int left = (int)sizeof(buf);
#define ADD(cond, tag_rt, tag_sdk) do { \
    if (cond) { \
        const char *t = (cond) >= 2 ? tag_rt : tag_sdk; \
        int n = snprintf(p, (size_t)left, "%s%s", p == buf ? "" : "|", t); \
        if (n > 0 && n < left) { p += n; left -= n; } \
    } \
} while (0)
    ADD(nn, "nnapi", "nnapi-sdk");
    ADD(qn, "qnn", "qnn-sdk");
    ADD(rk, "rkllm", "rkllm-sdk");
    ADD(hi, "hiai", "hiai-sdk");
#undef ADD
    if (p == buf) snprintf(buf, sizeof buf, "0");
    if (out && cap) snprintf(out, (size_t)cap, "%s", buf);
    return nn || qn || rk || hi;
}

typedef struct {
    char name[32];
    void *lib;
    char path[768];
} npu_rt;

static void npu_rt_destroy(void *impl) {
    npu_rt *s = (npu_rt *)impl;
    if (!s) return;
    if (s->lib) hy_dlclose(s->lib);
    free(s);
}
static hy_dweight *npu_rt_upload(void *impl, const weight_t *w) {
    (void)impl; (void)w;
    return NULL;
}
static void npu_rt_release(hy_dweight *w) { (void)w; }
static bool npu_rt_matmul(void *impl, hy_dweight *const *ws, float *const *outs, int cnt, const q8k_block *xq, int nt) {
    (void)impl; (void)ws; (void)outs; (void)cnt; (void)xq; (void)nt;
    return false;
}
static bool npu_rt_is_sw(const void *impl) { (void)impl; return false; }
static const hy_device_ops npu_rt_ops = { npu_rt_destroy, npu_rt_upload, npu_rt_release, npu_rt_matmul, npu_rt_is_sw, NULL, NULL, NULL };

/* 仅当本机能 dlopen 厂商库时 create；aarch64 .so 在 Mac/x86 上只 probe=sdk，不 create。 */
static hy_device *make_vendor(const char *name, int (*locate)(char *, size_t), const char *need_sym, char *info, size_t cap) {
    char path[768];
    if (!locate(path, sizeof path)) {
        if (info && cap) snprintf(info, cap, "%s 运行时/SDK 不可用", name);
        return NULL;
    }
    void *h = hy_dlopen(path);
    if (!h) {
        if (info && cap)
            snprintf(info, cap, "%s SDK 已就位 (%s) 但本机无法 dlopen (架构/依赖); 请在目标 SoC 上跑", name, path);
        return NULL;
    }
    if (need_sym && *need_sym && !hy_dlsym(h, need_sym)) {
        if (info && cap) snprintf(info, cap, "%s: 缺符号 %s", name, need_sym);
        hy_dlclose(h);
        return NULL;
    }
    npu_rt *s = (npu_rt *)calloc(1, sizeof(*s));
    if (!s) { hy_dlclose(h); return NULL; }
    snprintf(s->name, sizeof s->name, "%s", name);
    snprintf(s->path, sizeof s->path, "%s", path);
    s->lib = h;
    if (info && cap)
        snprintf(info, cap, "%s 库已加载 (%s); quant/GEMM 委派需转换模型后接线", name, path);
    hy_device *d = hy_device_wrap(HY_DEV_NPU, name, s, &npu_rt_ops, false);
    if (!d) npu_rt_destroy(s);
    return d;
}

hy_device *hy_npu_create_named(const char *name, char *info, size_t cap) {
    if (!name) return NULL;
    if (!strcmp(name, "npu")) {
        static const char *ord[] = { "qnn", "nnapi", "rkllm", "hiai", NULL };
        for (int i = 0; ord[i]; i++) {
            hy_device *d = hy_npu_create_named(ord[i], info, cap);
            if (d) return d;
        }
        if (info && cap) snprintf(info, cap, "无可用 NPU 插件");
        return NULL;
    }
#if defined(__ANDROID__)
    if (!strcmp(name, "nnapi")) return hy_nnapi_create(info, cap);
#else
    if (!strcmp(name, "nnapi")) return create_nnapi(info, cap);
#endif
    if (!strcmp(name, "qnn")) return make_vendor("qnn", locate_qnn, NULL, info, cap);
    if (!strcmp(name, "rkllm")) return make_vendor("rkllm", locate_rkllm, "rkllm_init", info, cap);
    if (!strcmp(name, "hiai")) return make_vendor("hiai", locate_hiai, NULL, info, cap);
    if (info && cap) snprintf(info, cap, "未知 NPU 名 %s", name);
    return NULL;
}

hy_device *hy_npu_create(const char *pref, char *info, size_t cap) {
    return hy_npu_create_named(pref && *pref ? pref : "npu", info, cap);
}
