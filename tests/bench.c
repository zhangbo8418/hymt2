/* 速度测试: 随机权重的"真实尺寸"模型 (见 tools/make_bench_gguf.c)。输出无意义, 只看耗时。 */
#include "../include/hymt2.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
const float *hymt2_debug_logits(hymt2_model *m, const int *ids, int n, int *vocab_out);
const float *hymt2_debug_logits_stepwise(hymt2_model *m, const int *ids, int n, int *vocab_out);
void hymt2_debug_profile_dump(void); void hymt2_debug_profile_reset(void);
int hymt2_debug_generate_ids(hymt2_model *m, const char *user, int *ids_out, int max);
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
int main(int argc, char **argv) {
    char err[512];
    int th = argc > 2 ? atoi(argv[2]) : 1;
    double t0 = ms();
    hymt2_model *m = hymt2_model_load(argv[1], th, 512, err, sizeof(err));
    if (!m) { fprintf(stderr, "load: %s\n", err); return 1; }
    printf("isa=%s threads=%d load=%.0f ms\n", hymt2_isa(), th, ms() - t0);
    int ids[64], v;
    for (int i = 0; i < 64; i++) ids[i] = 10 + i * 7;
    hymt2_debug_logits(m, ids, 8, &v);                       /* 预热 (缺页) */
    t0 = ms(); hymt2_debug_logits(m, ids, 48, &v); double pf = ms() - t0;
    t0 = ms(); hymt2_debug_logits_stepwise(m, ids, 48, &v); double st = ms() - t0;
    printf("prefill 48 tokens (batched)   : %7.1f ms  -> %6.1f tok/s\n", pf, 48 * 1000.0 / pf);
    printf("48 tokens one-by-one          : %7.1f ms  -> %6.1f tok/s (含 1 次输出头)\n", st, 48 * 1000.0 / st);
    hymt2_debug_profile_reset();
    int out[40];
    t0 = ms(); int n = hymt2_debug_generate_ids(m, "t5 t6 t7 t8 t9 t10 t11 t12", out, 32); double g = ms() - t0;
    printf("generate: %d tokens in %.1f ms  -> %.1f ms/token (含输出头)\n", n, g, g / (n > 0 ? n : 1));
    hymt2_debug_profile_dump();
    hymt2_model_free(m);
    return 0;
}
