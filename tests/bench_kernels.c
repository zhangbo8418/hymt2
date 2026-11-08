/* 单个内核的吞吐测试: Q6_K 输出头 (12 万行 × 2048) 与 STQ1_0 (6144 × 2048), 每种 ISA 各测一遍 */
#include "../src/internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
static double ms(void) { LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c); return (double)c.QuadPart * 1000.0 / (double)f.QuadPart; }
#else
static double ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec / 1e6; }
#endif
int main(void) {
    kernels_init();
    const int K = 2048, nb = K / QK_K;
    size_t rows6 = 120818, rowsS = 6144;
    block_q6_k *q6 = malloc(rows6 * nb * sizeof(block_q6_k));
    block_stq1_0 *st = malloc(rowsS * nb * sizeof(block_stq1_0));
    srand(1);
    for (size_t i = 0; i < rows6 * nb; i++) { for (int j = 0; j < 128; j++) q6[i].ql[j] = rand(); for (int j = 0; j < 64; j++) q6[i].qh[j] = rand();
        for (int j = 0; j < 16; j++) q6[i].scales[j] = rand() % 41 - 20; q6[i].d = 0x1400; }
    for (size_t i = 0; i < rowsS * nb; i++) { for (int j = 0; j < 32; j++) st[i].qs[j] = rand(); for (int j = 0; j < 8; j++) st[i].sign[j] = rand(); st[i].d = 0x2800; }
    float *x = malloc(sizeof(float) * K * MAX_BATCH);
    for (int i = 0; i < K * MAX_BATCH; i++) x[i] = (float)rand() / RAND_MAX - 0.5f;
    q8k_block *xq = hy_aligned_alloc(sizeof(q8k_block) * nb * MAX_BATCH);
    for (int t = 0; t < MAX_BATCH; t++) quantize_q8k(x + t * K, K, xq + t * nb);
    float *out = malloc(sizeof(float) * rows6 * MAX_BATCH);
    weight_t w6 = { (const uint8_t *)q6, GT_Q6_K, K, (int)rows6, nb * sizeof(block_q6_k) };
    weight_t ws = { (const uint8_t *)st, GT_STQ1_0, K, (int)rowsS, nb * sizeof(block_stq1_0) };
    /* 面板版 (权重已预解码) */
    uint8_t *pk = hy_aligned_alloc((size_t)rowsS * nb * 64); float *ps = malloc(sizeof(float) * rowsS * nb);
    weight_t wpn = ws; panel_repack_rows(&ws, pk, ps, 0, (int)rowsS); wpn.pk = pk; wpn.ps = ps; wpn.pshift = 0; wpn.poff = 1;
    const char *isas[] = { "scalar", "ssse3", "sse4.1", "avx2", "avx512-bw", "avx512-vnni", "neon" };
    for (int i = 0; i < 7; i++) {
        if (!kernels_set_isa(isas[i])) continue;
        matmul_rows(&w6, 0, 2000, x, xq, 1, out);
        double t0 = ms(); matmul_rows(&w6, 0, (int)rows6, x, xq, 1, out); double d6 = ms() - t0;
        t0 = ms(); matmul_rows(&ws, 0, (int)rowsS, x, xq, 1, out); double ds1 = ms() - t0;
        t0 = ms(); matmul_rows(&ws, 0, (int)rowsS, x, xq, 16, out); double ds16 = ms() - t0;
        printf("%-7s Q6_K head 120818x2048 : %7.1f ms  (%.2f GMAC/s)\n", kernels_isa(), d6, (double)rows6 * K / d6 / 1e6);
        printf("        STQ1_0 6144x2048 nt=1 : %7.2f ms  (%.2f GMAC/s)\n", ds1, (double)rowsS * K / ds1 / 1e6);
        matmul_rows(&wpn, 0, 500, x, xq, 1, out);
        t0 = ms(); matmul_rows(&wpn, 0, (int)rowsS, x, xq, 1, out); double dp1 = ms() - t0;
        t0 = ms(); matmul_rows(&wpn, 0, (int)rowsS, x, xq, 16, out); double dp16 = ms() - t0;
        printf("        panel  6144x2048 nt=1 : %7.2f ms  (%.2f GMAC/s)   nt=16: %7.2f ms (%.2f GMAC/s)\n", dp1, (double)rowsS * K / dp1 / 1e6, dp16, 16.0 * rowsS * K / dp16 / 1e6);
        printf("        STQ1_0 6144x2048 nt=16: %7.2f ms  (%.2f GMAC/s)\n", ds16, 16.0 * rowsS * K / ds16 / 1e6);
    }
    return 0;
}
