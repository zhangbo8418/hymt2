/* 与旧 NativeAOT 版同名的四个全局导出函数 (单例模型)。 */
#include "../include/hymt2.h"
#include "device.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static hymt2_model *g_model;
static char g_err[512];
static atomic_flag g_lock = ATOMIC_FLAG_INIT;
static void lock(void) { while (atomic_flag_test_and_set(&g_lock)) {} }
static void unlock(void) { atomic_flag_clear(&g_lock); }

int hymt2_init_ex(const char *model_path, int n_threads, int ctx_len) {
    lock();
    if (g_model) { hymt2_model_free(g_model); g_model = NULL; }
    g_err[0] = 0;
    g_model = hymt2_model_load(model_path, n_threads, ctx_len, g_err, sizeof(g_err));
    int rc = g_model ? 0 : -2;
    unlock();
    return rc;
}

int hymt2_init(const char *model_path, const char *backend) {
    (void)backend;
    return hymt2_init_ex(model_path, 0, 0);
}

int hymt2_generate(const char *user_prompt, char *out, int out_cap, int max_tokens, int timeout_ms) {
    lock();
    int rc;
    if (!g_model) { snprintf(g_err, sizeof(g_err), "not initialized"); rc = -1; }
    else {
        rc = hymt2_model_generate(g_model, user_prompt, out, out_cap, max_tokens, timeout_ms);
        if (rc < 0) snprintf(g_err, sizeof(g_err), "%s", hymt2_model_last_error(g_model));
    }
    unlock();
    return rc;
}

int hymt2_last_error(char *out, int out_cap) {
    size_t n = strlen(g_err);
    if ((int)n > out_cap) n = (size_t)out_cap;
    memcpy(out, g_err, n);
    return (int)n;
}

void hymt2_free(void) {
    lock();
    if (g_model) { hymt2_model_free(g_model); g_model = NULL; }
    unlock();
}

/* 把"检测到的 CPU 特性 / 选中的内核 / 设备能力"写进 out, 返回字节数 (给 Go 侧打日志) */
int hymt2_info(char *out, int out_cap) {
    char caps[256];
    hy_device_capabilities(caps, sizeof caps);
    char tmp[768];
    int n = snprintf(tmp, sizeof(tmp), "kernel=%s cpu=[%s] %s", hymt2_isa(), hymt2_cpu_features(), caps);
    if (n < 0) n = 0;
    if (n > out_cap) n = out_cap;
    memcpy(out, tmp, (size_t)n);
    return n;
}

