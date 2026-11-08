/* 内核自测: SIMD 路径必须与标量参考、以及"先反量化再做浮点点积"一致。 */
#include "../src/internal.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t rng = 12345;
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static float frand(void) { return ((float)(rnd() & 0xFFFFFF) / (float)0x800000) - 1.0f; }

static uint16_t f32_to_f16(float f) { /* 仅用于造测试数据: 正常范围内的简易转换 */
    union { float f; uint32_t u; } v = { f };
    uint32_t s = (v.u >> 16) & 0x8000, e = ((v.u >> 23) & 0xFF) - 112, m = (v.u >> 13) & 0x3FF;
    return (uint16_t)(s | (e << 10) | m);
}

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { printf("FAIL: " __VA_ARGS__); printf("\n"); fails++; } } while (0)

static int run(void) {
    printf("--- isa = %s\n", kernels_isa());
    const int nb = 6, K = nb * QK_K, ROWS = 7, NT = 5;

    /* ---- STQ1_0 ---- */
    block_stq1_0 *w = malloc(sizeof(block_stq1_0) * nb * ROWS);
    for (int i = 0; i < nb * ROWS; i++) {
        for (int j = 0; j < 32; j++) w[i].qs[j] = (uint8_t)rnd();
        for (int j = 0; j < 8; j++) w[i].sign[j] = (uint8_t)rnd();
        w[i].d = f32_to_f16(0.01f + 0.05f * (float)(rnd() % 100) / 100.f);
    }
    float *xf = malloc(sizeof(float) * K * NT);
    for (int i = 0; i < K * NT; i++) xf[i] = frand() * (i % 37 == 0 ? 8.f : 1.f);
    q8k_block *xq = hy_aligned_alloc(sizeof(q8k_block) * nb * NT);
    for (int t = 0; t < NT; t++) quantize_q8k(xf + t * K, K, xq + t * nb);

    /* 量化往返误差 */
    float maxerr = 0;
    for (int t = 0; t < NT; t++)
        for (int b = 0; b < nb; b++)
            for (int j = 0; j < QK_K; j++) {
                float deq = xq[t * nb + b].qs[j] * xq[t * nb + b].d;
                float e = fabsf(deq - xf[t * K + b * QK_K + j]) / (fabsf(xq[t * nb + b].d) + 1e-12f);
                if (e > maxerr) maxerr = e;
            }
    CHECK(maxerr <= 0.5001f, "q8k round-trip err %g (>0.5 step)", maxerr);

    weight_t wt = { (const uint8_t *)w, GT_STQ1_0, K, ROWS, (size_t)nb * sizeof(block_stq1_0) };
    float *out = malloc(sizeof(float) * ROWS * NT);
    matmul_rows(&wt, 0, ROWS, xf, xq, NT, out);
    double worst = 0;
    for (int r = 0; r < ROWS; r++) {
        float *dw = malloc(sizeof(float) * K);
        stq1_0_dequant_row(w + r * nb, dw, K);
        for (int t = 0; t < NT; t++) {
            double ref = 0, mag = 0; /* 用反量化权重 × 反量化激活, 做双精度; mag = Σ|项| (误差基准, 抗相消) */
            for (int b = 0; b < nb; b++)
                for (int j = 0; j < QK_K; j++) {
                    double term = (double)dw[b * QK_K + j] * ((double)xq[t * nb + b].qs[j] * (double)xq[t * nb + b].d);
                    ref += term; mag += fabs(term);
                }
            float sc = stq1_0_dot_scalar(w + r * nb, xq + t * nb, nb);
            float di = stq1_0_dot_dispatch(w + r * nb, xq + t * nb, nb);
            float bt = out[t * ROWS + r];
            double den = mag + 1e-6;
            double e1 = fabs(sc - ref) / den, e2 = fabs(di - ref) / den, e3 = fabs(bt - ref) / den;
            if (e1 > worst) worst = e1; if (e2 > worst) worst = e2; if (e3 > worst) worst = e3;
            CHECK(e1 < 1e-5 && e2 < 1e-5 && e3 < 1e-5, "stq row %d tok %d ref=%g scalar=%g simd=%g batch=%g", r, t, ref, sc, di, bt);
        }
        free(dw);
    }
    printf("STQ1_0   worst err / Σ|term| = %.3g\n", worst);

    /* 每个 4 组必须"恰好一个 0, 三个等幅": 检查码本语义 */
    {
        float dw[QK_K];
        int bad = 0;
        for (int i = 0; i < nb * ROWS; i++) {
            stq1_0_dequant_row(&w[i], dw, QK_K);
            for (int g = 0; g < 64; g++) {
                int chunk = g >> 4, gloc = g & 15, zeros = 0;
                for (int lane = 0; lane < 4; lane++)
                    if (dw[chunk * 64 + gloc + lane * 16] == 0.f) zeros++;
                if (zeros != 1) bad++;
            }
        }
        CHECK(bad == 0, "codebook: %d groups without exactly one zero", bad);
        printf("codebook: every 4-group has exactly one zero: %s\n", bad ? "NO" : "yes");
    }

    /* ---- Q6_K ---- */
    block_q6_k *q6 = malloc(sizeof(block_q6_k) * nb * ROWS);
    for (int i = 0; i < nb * ROWS; i++) {
        for (int j = 0; j < 128; j++) q6[i].ql[j] = (uint8_t)rnd();
        for (int j = 0; j < 64; j++) q6[i].qh[j] = (uint8_t)rnd();
        for (int j = 0; j < 16; j++) q6[i].scales[j] = (int8_t)((int)(rnd() % 61) - 30);
        q6[i].d = f32_to_f16(0.002f + 0.004f * (float)(rnd() % 100) / 100.f);
    }
    weight_t w6 = { (const uint8_t *)q6, GT_Q6_K, K, ROWS, (size_t)nb * sizeof(block_q6_k) };
    matmul_rows(&w6, 0, ROWS, xf, xq, NT, out);
    worst = 0;
    for (int r = 0; r < ROWS; r++) {
        float dw[K];
        q6_k_dequant_row(q6 + r * nb, dw, K);
        for (int t = 0; t < NT; t++) {
            double ref = 0, mag = 0;
            for (int b = 0; b < nb; b++)
                for (int j = 0; j < QK_K; j++) {
                    double term = (double)dw[b * QK_K + j] * ((double)xq[t * nb + b].qs[j] * (double)xq[t * nb + b].d);
                    ref += term; mag += fabs(term);
                }
            double e = fabs(out[t * ROWS + r] - ref) / (mag + 1e-6);
            if (e > worst) worst = e;
            CHECK(e < 1e-5, "q6k row %d tok %d ref=%g got=%g", r, t, ref, out[t * ROWS + r]);
        }
    }
    printf("Q6_K     worst err / Σ|term| = %.3g\n", worst);

    /* ---- Q4_K: SIMD 与标量一致 ---- */
    {
        block_q4_k *q4 = malloc(sizeof(block_q4_k) * nb * ROWS);
        for (int i = 0; i < nb * ROWS; i++) {
            q4[i].d = f32_to_f16(0.01f + 0.02f * (float)(rnd() % 100) / 100.f);
            q4[i].dmin = f32_to_f16(0.001f + 0.002f * (float)(rnd() % 100) / 100.f);
            for (int j = 0; j < 12; j++) q4[i].scales[j] = (uint8_t)rnd();
            for (int j = 0; j < 128; j++) q4[i].qs[j] = (uint8_t)rnd();
        }
        weight_t w4 = { (const uint8_t *)q4, GT_Q4_K, K, ROWS, (size_t)nb * sizeof(block_q4_k) };
        matmul_rows(&w4, 0, ROWS, xf, xq, NT, out);
        worst = 0;
        for (int r = 0; r < ROWS; r++)
            for (int t = 0; t < NT; t++) {
                float sc = q4_k_dot_scalar(q4 + r * nb, xq + t * nb, nb);
                float got = out[t * ROWS + r];
                double den = fabs(sc) + 1.0;
                double e = fabs(got - sc) / den;
                if (e > worst) worst = e;
                CHECK(e < 1e-5, "q4k row %d tok %d scalar=%g got=%g", r, t, sc, got);
            }
        printf("Q4_K     worst err vs scalar = %.3g\n", worst);
        free(q4);
    }

    /* ---- Q8_0: SIMD 与标量一致 ---- */
    {
        const int nb8 = K / 32;
        block_q8_0 *q8 = malloc(sizeof(block_q8_0) * nb8 * ROWS);
        for (int i = 0; i < nb8 * ROWS; i++) {
            q8[i].d = f32_to_f16(0.01f + 0.02f * (float)(rnd() % 100) / 100.f);
            for (int j = 0; j < 32; j++) q8[i].qs[j] = (int8_t)((int)(rnd() % 255) - 128);
        }
        weight_t w8 = { (const uint8_t *)q8, GT_Q8_0, K, ROWS, (size_t)nb8 * sizeof(block_q8_0) };
        matmul_rows(&w8, 0, ROWS, xf, xq, NT, out);
        worst = 0;
        for (int r = 0; r < ROWS; r++)
            for (int t = 0; t < NT; t++) {
                float sc = 0.f;
                const block_q8_0 *row = q8 + r * nb8;
                const float *yy = xf + t * K;
                for (int b = 0; b < nb8; b++) {
                    float s = 0.f;
                    for (int j = 0; j < 32; j++) s += (float)row[b].qs[j] * yy[b * 32 + j];
                    sc += f16_to_f32(row[b].d) * s;
                }
                float got = out[t * ROWS + r];
                double den = fabs(sc) + 1.0;
                double e = fabs(got - sc) / den;
                if (e > worst) worst = e;
                CHECK(e < 1e-5, "q8_0 row %d tok %d scalar=%g got=%g", r, t, sc, got);
            }
        printf("Q8_0     worst err vs scalar = %.3g\n", worst);
        free(q8);
    }

    /* ---- 面板重排: STQ1_0 和 Q2_0C, 与"反量化 × 反量化激活"的双精度参考比较 ---- */
    for (int kind = 0; kind < 2; kind++) {
        const int type = kind == 0 ? GT_STQ1_0 : GT_Q2_0C;
        size_t rb = row_size(type, K);
        uint8_t *raw = malloc(rb * ROWS);
        for (size_t i = 0; i < rb * ROWS; i++) raw[i] = (uint8_t)rnd();
        for (int r = 0; r < ROWS; r++) { /* 把 f16 缩放字段改成有限的小数 */
            if (type == GT_STQ1_0) for (int b = 0; b < nb; b++) ((block_stq1_0 *)(raw + r * rb))[b].d = f32_to_f16(0.01f + 0.03f * (float)(rnd() % 100) / 100.f);
            else for (int b = 0; b < K / 512; b++) ((block_q2_0c *)(raw + r * rb))[b].d = f32_to_f16(0.01f + 0.03f * (float)(rnd() % 100) / 100.f);
        }
        weight_t wp = { raw, type, K, ROWS, rb, NULL, NULL, kind == 0 ? 0 : 1, kind == 0 ? 1 : 3, NULL };
        uint8_t *pk = hy_aligned_alloc((size_t)ROWS * nb * 64);
        float *ps = malloc(sizeof(float) * ROWS * nb);
        panel_repack_rows(&wp, pk, ps, 0, ROWS);
        weight_t wpanel = wp; wpanel.pk = pk; wpanel.ps = ps;
        float *o2 = malloc(sizeof(float) * ROWS * NT);
        matmul_rows(&wpanel, 0, ROWS, xf, xq, NT, o2);
        double worstp = 0;
        for (int r = 0; r < ROWS; r++) {
            float *dw = malloc(sizeof(float) * K);
            dequant_row(&wp, r, dw);
            for (int t = 0; t < NT; t++) {
                double ref = 0, mag = 0;
                for (int b = 0; b < nb; b++)
                    for (int j = 0; j < QK_K; j++) {
                        double term = (double)dw[b * QK_K + j] * ((double)xq[t * nb + b].qs[j] * (double)xq[t * nb + b].d);
                        ref += term; mag += fabs(term);
                    }
                double e = fabs(o2[t * ROWS + r] - ref) / (mag + 1e-6);
                if (e > worstp) worstp = e;
                CHECK(e < 1e-5, "panel %s row %d tok %d ref=%g got=%g", kind == 0 ? "stq" : "q2_0c", r, t, ref, o2[t * ROWS + r]);
            }
            free(dw);
        }
        printf("panel %-7s worst err / Σ|term| = %.3g\n", kind == 0 ? "STQ1_0" : "Q2_0C", worstp);
        /* nt=1 路径 (解码) 也要覆盖: 单 token 单独算一遍 */
        float o1[ROWS];
        matmul_rows(&wpanel, 0, ROWS, xf, xq, 1, o1);
        for (int r = 0; r < ROWS; r++) CHECK(fabsf(o1[r] - o2[r]) <= 1e-4f * (fabsf(o2[r]) + 1.f), "panel nt=1 vs nt=%d row %d: %g vs %g", NT, r, o1[r], o2[r]);
        /* 无面板的原生路径 (STQ: 各 ISA 内核; Q2_0C: 标量) 应与面板结果一致 */
        matmul_rows(&wp, 0, ROWS, xf, xq, NT, out);
        for (int r = 0; r < ROWS; r++) for (int t = 0; t < NT; t++)
            CHECK(fabsf(out[t * ROWS + r] - o2[t * ROWS + r]) <= 1e-4f * (fabsf(o2[t * ROWS + r]) + 1.f), "native vs panel row %d tok %d: %g vs %g", r, t, out[t * ROWS + r], o2[t * ROWS + r]);
        free(raw); hy_aligned_free(pk); free(ps); free(o2);
    }

    /* ---- 注意力用的 bf16 点积 / axpy ---- */
    {
        enum { N = 128 };
        float q[N], o1[N], o2[N];
        uint16_t kv[N];
        double worst2 = 0;
        for (int trial = 0; trial < 50; trial++) {
            for (int i = 0; i < N; i++) { q[i] = frand() * 3; kv[i] = f32_to_bf16(frand() * 2); o1[i] = o2[i] = frand(); }
            float p = frand();
            float got = dot_bf16(q, kv, N), want = dot_bf16_ref(q, kv, N);
            double mag = 0;
            for (int i = 0; i < N; i++) mag += fabs((double)q[i] * bf16_to_f32(kv[i]));
            double e = fabs(got - want) / (mag + 1e-9);
            if (e > worst2) worst2 = e;
            CHECK(e < 1e-6, "dot_bf16 got=%g want=%g", got, want);
            axpy_bf16(o1, kv, p, N);
            for (int i = 0; i < N; i++) o2[i] += p * bf16_to_f32(kv[i]);
            for (int i = 0; i < N; i++) CHECK(fabsf(o1[i] - o2[i]) < 1e-5f, "axpy_bf16 [%d] %g vs %g", i, o1[i], o2[i]);
        }
        printf("bf16 dot/axpy worst err / Σ|term| = %.3g\n", worst2);
    }
    return fails;
}

int main(void) {
    kernels_init();
    const char *isas[] = { "scalar", "ssse3", "sse4.1", "avx2", "avx512-bw", "avx512-vnni", "neon" };
    int total = 0, ran = 0;
    for (size_t i = 0; i < sizeof(isas) / sizeof(isas[0]); i++) {
        if (!kernels_set_isa(isas[i])) { printf("--- isa %s: not available, skipped\n", isas[i]); continue; }
        total += run();
        ran++;
    }
    printf(total ? "\n%d FAILURES\n" : "\nALL KERNEL TESTS PASSED (%d ISA paths)\n", total ? total : ran);
    return total ? 1 : 0;
}
