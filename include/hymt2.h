/* hymt2: Tencent Hy-MT2 (hunyuan-dense) 的纯 C CPU 推理库, 面向 1.25-bit STQ1_0 GGUF。
 * 移植自 sdcb/HyMT2Sharp (Apache-2.0)。线程模型: 一个模型句柄不是线程安全的, 调用方需串行化。 */
#ifndef HYMT2_H
#define HYMT2_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define HYMT2_API __declspec(dllexport)
#else
#define HYMT2_API __attribute__((visibility("default")))
#endif

typedef struct hymt2_model hymt2_model;

/* 返回码约定 (generate): >=0 写入的字节数; -1 未初始化; -2 出错 (见 last_error); -3 超时 */

/* n_threads <= 0 : 自动 (环境变量 HYMT2_THREADS 可覆盖); ctx_len <= 0 : 2048 */
HYMT2_API hymt2_model *hymt2_model_load(const char *gguf_path, int n_threads, int ctx_len, char *err, int err_cap);
HYMT2_API void hymt2_model_free(hymt2_model *m);
/* 贪心解码。user_prompt 是完整的用户提示词 (本函数负责套聊天模板)。输出 UTF-8, 不带 NUL。 */
HYMT2_API int hymt2_model_generate(hymt2_model *m, const char *user_prompt, char *out, int out_cap, int max_tokens, int timeout_ms);
HYMT2_API const char *hymt2_model_last_error(const hymt2_model *m);
/* 当前模型实际使用的配置: "kernel=sse4.1 threads=4 batch=8 repack=panel gpu=off" */
HYMT2_API int hymt2_model_info(const hymt2_model *m, char *out, int cap);
/* 最近一次 generate 的耗时分解: prefill (新算的 token 数 / 命中前缀缓存的 token 数 / 毫秒) 与 decode (token 数 / 毫秒) */
HYMT2_API void hymt2_model_stats(const hymt2_model *m, int *prefill_tokens, int *cached_tokens, double *prefill_ms, int *decode_tokens, double *decode_ms);
HYMT2_API const char *hymt2_isa(void); /* 实际使用的内核: "avx512-vnni" / "avx512" / "avx2" / "ssse3" / "neon" / "neon+dotprod" / "scalar"; 环境变量 HYMT2_ISA 可强制降级 */
HYMT2_API const char *hymt2_cpu_features(void); /* 检测到的 CPU 特性, 供日志使用 */
/* 本机加速能力: "isa=… cuda=0|1 metal=0|1 vulkan=0|1 npu=0|1"。设备选择见 HYMT2_DEVICE。 */
HYMT2_API int hymt2_capabilities(char *out, int cap);

/* ---- 全局单例接口 (给 Go / purego 用): init_ex / generate / last_error / free ---- */
HYMT2_API int hymt2_init_ex(const char *model_path, int n_threads /* <=0 自动 */, int ctx_len /* <=0 为 2048 */);
HYMT2_API int hymt2_init(const char *model_path, const char *backend /* 忽略 */);
HYMT2_API int hymt2_info(char *out, int out_cap); /* "kernel=sse4.1 cpu=[ssse3 sse4.1 sse4.2]" */
HYMT2_API int hymt2_generate(const char *user_prompt, char *out, int out_cap, int max_tokens, int timeout_ms);
HYMT2_API int hymt2_last_error(char *out, int out_cap);
HYMT2_API void hymt2_free(void);

#ifdef __cplusplus
}
#endif
#endif
