/* 量化内核: STQ1_0 / Q6_K / Q8_0 与 Q8_K 激活量化。
 * STQ1_0 的编码规则、码本和 SIMD 取数方式移植自 HyMT2Sharp 的 Kernels/STQ1_0.cs (Apache-2.0)。 */
#include "internal.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
#define HY_X86 1
#include <immintrin.h>
#endif
#if defined(__aarch64__)
#define HY_NEON 1
#include <arm_neon.h>
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
/* glibc/bionic: ASIMDDP=1<<20, I8MM 在 HWCAP2。运行时探测, 避免整库 -march=dotprod 在老核上 SIGILL。 */
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#endif
#if defined(__clang__) || defined(__GNUC__)
#define HY_ARM_DOTPROD_TARGET __attribute__((target("+dotprod")))
#else
#define HY_ARM_DOTPROD_TARGET
#endif
#endif

#include <stdlib.h>
#ifdef _WIN32
#include <malloc.h>
void *hy_aligned_alloc(size_t size) { return _aligned_malloc(size, 64); }
void hy_aligned_free(void *p) { _aligned_free(p); }
#else
void *hy_aligned_alloc(size_t size) {
    void *p = NULL;
    if (posix_memalign(&p, 64, size ? size : 64) != 0) return NULL;
    return p;
}
void hy_aligned_free(void *p) { free(p); }
#endif

/* ------------------------------------------------------------------ 基础转换 */

float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else { /* 次正规数 */
            int e = -1;
            do { e++; man <<= 1; } while ((man & 0x400u) == 0);
            man &= 0x3FFu;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);
    } else {
        bits = sign | ((exp + 112u) << 23) | (man << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

uint16_t f32_to_bf16(float f) { /* 与 C# ConvertToBf16 一致: 就近舍入 */
    uint32_t bits;
    memcpy(&bits, &f, 4);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

size_t row_size(int type, int k) {
    switch (type) {
    case GT_F32: return (size_t)k * 4;
    case GT_F16: return (size_t)k * 2;
    case GT_Q8_0: return (size_t)(k / 32) * sizeof(block_q8_0);
    case GT_Q4_K: return (size_t)(k / QK_K) * sizeof(block_q4_k);
    case GT_Q6_K: return (size_t)(k / QK_K) * sizeof(block_q6_k);
    case GT_STQ1_0: return (size_t)(k / QK_K) * sizeof(block_stq1_0);
    case GT_Q2_0C: return (size_t)(k / 512) * sizeof(block_q2_0c);
    }
    return 0;
}

/* ------------------------------------------------------------------ Q8_K 激活量化 */

void quantize_q8k(const float *x, int k, q8k_block *y) {
    int nb = k / QK_K;
    static const int cord[4] = Q8K_PIECE_ORDER;
    for (int i = 0; i < nb; i++) {
        const float *src = x + (size_t)i * QK_K;
        float amax = 0.f, max = 0.f;
#if defined(HY_NEON)
        {
            float32x4_t vmax = vdupq_n_f32(0.f), vamax = vdupq_n_f32(0.f);
            for (int j = 0; j < QK_K; j += 4) {
                float32x4_t v = vld1q_f32(src + j);
                float32x4_t ax = vabsq_f32(v);
                uint32x4_t m = vcgtq_f32(ax, vamax);
                vamax = vbslq_f32(m, ax, vamax);
                vmax = vbslq_f32(m, v, vmax);
            }
            float ax4[4], vx4[4];
            vst1q_f32(ax4, vamax);
            vst1q_f32(vx4, vmax);
            for (int j = 0; j < 4; j++)
                if (ax4[j] > amax) { amax = ax4[j]; max = vx4[j]; }
        }
#else
        for (int j = 0; j < QK_K; j++) { /* 取"第一个绝对值最大"的元素 (带符号) */
            float ax = fabsf(src[j]);
            if (ax > amax) { amax = ax; max = src[j]; }
        }
#endif
        q8k_block *b = &y[i];
        if (amax == 0.f) {
            memset(b, 0, sizeof(*b));
            continue;
        }
        memset(b->_pad, 0, sizeof(b->_pad));
        float iscale = -127.f / max;
        b->d = 1.f / iscale;
        int total = 0;
#if defined(HY_NEON)
        {
            const float32x4_t scale = vdupq_n_f32(iscale);
            const float32x4_t lo = vdupq_n_f32(-127.f), hi = vdupq_n_f32(127.f);
            for (int g = 0; g < QK_K / 16; g++) {
                int32x4_t s4 = vdupq_n_s32(0);
                for (int j = 0; j < 16; j += 4) {
                    float32x4_t v = vminq_f32(hi, vmaxq_f32(lo, vmulq_f32(scale, vld1q_f32(src + g * 16 + j))));
                    int32x4_t qi = vcvtnq_s32_f32(v); /* 就近偶数, 对齐 lrintf */
                    int32_t tmp[4];
                    vst1q_s32(tmp, qi);
                    b->qs[g * 16 + j + 0] = (int8_t)tmp[0];
                    b->qs[g * 16 + j + 1] = (int8_t)tmp[1];
                    b->qs[g * 16 + j + 2] = (int8_t)tmp[2];
                    b->qs[g * 16 + j + 3] = (int8_t)tmp[3];
                    s4 = vaddq_s32(s4, qi);
                }
                int s = (int)vaddvq_s32(s4);
                b->bsums[g] = (int16_t)s;
                total += s;
            }
        }
#else
        for (int g = 0; g < QK_K / 16; g++) {
            int s = 0;
            for (int j = 0; j < 16; j++) {
                int v = (int)lrintf(iscale * src[g * 16 + j]); /* 就近偶数舍入 */
                if (v > 127) v = 127;
                if (v < -127) v = -127;
                b->qs[g * 16 + j] = (int8_t)v;
                s += v;
            }
            b->bsums[g] = (int16_t)s;
            total += s;
        }
#endif
        b->sum = total;
        /* 重排副本 qp[l][j] = qs 的 chunk CORD[j] 的 lane l (见 q8k_block 注释) */
        for (int l = 0; l < 4; l++)
            for (int j = 0; j < 4; j++)
                memcpy(b->qp + l * 64 + j * 16, b->qs + cord[j] * 64 + l * 16, 16);
    }
}

#ifdef HY_X86
__attribute__((target("f16c")))
static inline float f16_hw(uint16_t h) { return _cvtsh_ss(h); }
#endif
#ifdef HY_NEON
static inline float f16_hw(uint16_t h) { return vgetq_lane_f32(vcvt_f32_f16(vcreate_f16((uint64_t)h)), 0); }
#endif

/* ------------------------------------------------------------------ STQ1_0 */

/* 每个 4 权重组的 2-bit 三值模式: 0=-1, 1=0, 2=+1。下标 = (sign<<4)|slot */
static const uint8_t STQ_CODEBOOK[32] = {
    0xA9, 0x89, 0x29, 0x09, 0xA6, 0x86, 0x26, 0x06, 0x9A, 0x92, 0x1A, 0x12, 0x6A, 0x62, 0x4A, 0x42,
    0x01, 0x21, 0x81, 0xA1, 0x04, 0x24, 0x84, 0xA4, 0x10, 0x18, 0x90, 0x98, 0x40, 0x48, 0x60, 0x68,
};
static uint64_t SIGN_LUT[256]; /* 每个字节 8 位 -> 8 个 0x00/0xFF 选择字节 */

static inline uint8_t stq_qpack(const block_stq1_0 *x, int g) {
    int slot = (x->qs[g >> 1] >> ((g & 1) * 4)) & 0x0F;
    int sign = (x->sign[g >> 3] >> (g & 7)) & 1;
    return STQ_CODEBOOK[(sign << 4) | slot];
}

void stq1_0_dequant_row(const block_stq1_0 *x, float *dst, int k) {
    for (int b = 0; b < k / QK_K; b++) {
        float d = f16_to_f32(x[b].d);
        for (int g = 0; g < QK_K / 4; g++) {
            uint8_t qp = stq_qpack(&x[b], g);
            int chunk = g >> 4, gloc = g & 15;
            for (int lane = 0; lane < 4; lane++)
                dst[b * QK_K + chunk * 64 + gloc + lane * 16] = (float)(((qp >> (lane * 2)) & 3) - 1) * d;
        }
    }
}

float stq1_0_dot_scalar(const block_stq1_0 *x, const q8k_block *y, int nb) {
    float sum = 0.f;
    for (int b = 0; b < nb; b++) {
        int si = 0;
        for (int g = 0; g < QK_K / 4; g++) {
            uint8_t qp = stq_qpack(&x[b], g);
            int chunk = g >> 4, gloc = g & 15;
            for (int lane = 0; lane < 4; lane++)
                si += (((qp >> (lane * 2)) & 3) - 1) * y[b].qs[chunk * 64 + gloc + lane * 16];
        }
        sum += f16_to_f32(x[b].d) * y[b].d * (float)si;
    }
    return sum;
}

typedef void (*stq_rows_fn)(const weight_t *, int, int, const q8k_block *, int, float *);

static void stq_rows_scalar(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    int nb = w->k / QK_K;
    for (int r = r0; r < r1; r++) {
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);
        for (int t = 0; t < nt; t++)
            out[(size_t)t * w->n + r] = stq1_0_dot_scalar(row, xq + (size_t)t * nb, nb);
    }
}

#ifdef HY_X86
/* 解出一个 64 权重 chunk (16 组) 的 4 个 lane 平面, 每个字节是 0/1/2 (= 权重+1) */
__attribute__((target("ssse3")))
static inline void stq_chunk_codes_ssse3(const uint8_t *qs, const uint8_t *sg, __m128i out[4], __m128i tlo, __m128i thi) {
    const __m128i m0f = _mm_set1_epi8(0x0F), m3 = _mm_set1_epi8(3);
    __m128i qb = _mm_loadl_epi64((const __m128i *)qs);
    __m128i lo = _mm_and_si128(qb, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), m0f);
    __m128i slots = _mm_unpacklo_epi8(lo, hi); /* [lo0,hi0,lo1,hi1,...] = 组 0..15 的 slot */
    __m128i sm = _mm_set_epi64x((long long)SIGN_LUT[sg[1]], (long long)SIGN_LUT[sg[0]]);
    __m128i q0 = _mm_shuffle_epi8(tlo, slots);
    __m128i q1 = _mm_shuffle_epi8(thi, slots);
    __m128i qp = _mm_or_si128(_mm_andnot_si128(sm, q0), _mm_and_si128(sm, q1));
    out[0] = _mm_and_si128(qp, m3);
    out[1] = _mm_and_si128(_mm_srli_epi16(qp, 2), m3);
    out[2] = _mm_and_si128(_mm_srli_epi16(qp, 4), m3);
    out[3] = _mm_and_si128(_mm_srli_epi16(qp, 6), m3);
}

__attribute__((target("ssse3")))
static void stq_rows_ssse3(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    /* 面向没有 AVX 的 CPU (如 J4125 / Goldmont Plus): 尽量少的指令。
     *  - maddubs 的 int16 结果在一个权重块内直接累加 (每通道最多 16×508 = 8128, 不会溢出), 只在块末做一次 madd
     *  - 不再每块做 hadd: 浮点累加留在向量里, 整行结束才做一次水平求和 */
    static const int cord[4] = Q8K_PIECE_ORDER;
    const int nb = w->k / QK_K;
    const __m128i tlo = _mm_loadu_si128((const __m128i *)STQ_CODEBOOK);
    const __m128i thi = _mm_loadu_si128((const __m128i *)(STQ_CODEBOOK + 16));
    const __m128i ones = _mm_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);
        __m128 accf[MAX_BATCH];
        for (int t = 0; t < nt; t++) accf[t] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m128i codes[4][4];
            for (int c = 0; c < 4; c++)
                stq_chunk_codes_ssse3(row[b].qs + c * 8, row[b].sign + c * 2, codes[c], tlo, thi);
            const float wd = f16_to_f32(row[b].d);
            for (int t = 0; t < nt; t++) {
                const q8k_block *y = xq + (size_t)t * nb + b;
                __m128i s16 = _mm_setzero_si128();
                for (int l = 0; l < 4; l++)
                    for (int c = 0; c < 4; c++) {
                        __m128i a = _mm_load_si128((const __m128i *)(y->qp + l * 64 + cord[c] * 16));
                        s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(codes[c][l], a));
                    }
                __m128i s32 = _mm_madd_epi16(s16, ones);
                s32 = _mm_sub_epi32(s32, _mm_cvtsi32_si128(y->sum)); /* 修正量只扣在 0 号通道, 四个通道之和正好是 enc - sum */
                accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_cvtepi32_ps(s32), _mm_set1_ps(wd * y->d)));
            }
        }
        for (int t = 0; t < nt; t++) {
            __m128 v = _mm_add_ps(accf[t], _mm_movehl_ps(accf[t], accf[t]));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            out[(size_t)t * w->n + r] = _mm_cvtss_f32(v);
        }
    }
}

__attribute__((target("sse4.1")))
static inline void stq_chunk_codes_sse41(const uint8_t *qs, const uint8_t *sg, __m128i out[4], __m128i tlo, __m128i thi) {
    const __m128i m0f = _mm_set1_epi8(0x0F), m3 = _mm_set1_epi8(3);
    __m128i qb = _mm_loadl_epi64((const __m128i *)qs);
    __m128i lo = _mm_and_si128(qb, m0f);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(qb, 4), m0f);
    __m128i slots = _mm_unpacklo_epi8(lo, hi);
    __m128i sm = _mm_set_epi64x((long long)SIGN_LUT[sg[1]], (long long)SIGN_LUT[sg[0]]);
    __m128i qp = _mm_blendv_epi8(_mm_shuffle_epi8(tlo, slots), _mm_shuffle_epi8(thi, slots), sm);
    out[0] = _mm_and_si128(qp, m3);
    out[1] = _mm_and_si128(_mm_srli_epi16(qp, 2), m3);
    out[2] = _mm_and_si128(_mm_srli_epi16(qp, 4), m3);
    out[3] = _mm_and_si128(_mm_srli_epi16(qp, 6), m3);
}

__attribute__((target("sse4.1")))
static void stq_rows_sse41(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    /* SSE4.1 版 (J4125 / Goldmont Plus 走这条): 同 ssse3 版, 另用 pblendvb 把符号选择从 3 条指令降到 1 条。
     *  - maddubs 的 int16 结果在一个权重块内直接累加 (每通道最多 16×508 = 8128, 不会溢出), 只在块末做一次 madd
     *  - 不再每块做 hadd: 浮点累加留在向量里, 整行结束才做一次水平求和 */
    static const int cord[4] = Q8K_PIECE_ORDER;
    const int nb = w->k / QK_K;
    const __m128i tlo = _mm_loadu_si128((const __m128i *)STQ_CODEBOOK);
    const __m128i thi = _mm_loadu_si128((const __m128i *)(STQ_CODEBOOK + 16));
    const __m128i ones = _mm_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);
        __m128 accf[MAX_BATCH];
        for (int t = 0; t < nt; t++) accf[t] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m128i codes[4][4];
            for (int c = 0; c < 4; c++)
                stq_chunk_codes_sse41(row[b].qs + c * 8, row[b].sign + c * 2, codes[c], tlo, thi);
            const float wd = f16_to_f32(row[b].d);
            for (int t = 0; t < nt; t++) {
                const q8k_block *y = xq + (size_t)t * nb + b;
                __m128i s16 = _mm_setzero_si128();
                for (int l = 0; l < 4; l++)
                    for (int c = 0; c < 4; c++) {
                        __m128i a = _mm_load_si128((const __m128i *)(y->qp + l * 64 + cord[c] * 16));
                        s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(codes[c][l], a));
                    }
                __m128i s32 = _mm_madd_epi16(s16, ones);
                s32 = _mm_sub_epi32(s32, _mm_cvtsi32_si128(y->sum)); /* 修正量只扣在 0 号通道, 四个通道之和正好是 enc - sum */
                accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_cvtepi32_ps(s32), _mm_set1_ps(wd * y->d)));
            }
        }
        for (int t = 0; t < nt; t++) {
            __m128 v = _mm_add_ps(accf[t], _mm_movehl_ps(accf[t], accf[t]));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            out[(size_t)t * w->n + r] = _mm_cvtss_f32(v);
        }
    }
}
#endif /* HY_X86 */

#ifdef HY_NEON
static inline void stq_chunk_codes_neon(const uint8_t *qs, const uint8_t *sg, uint8x16_t out[4], uint8x16_t tlo, uint8x16_t thi) {
    const uint8x16_t m0f = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3);
    uint8x16_t qb = vcombine_u8(vld1_u8(qs), vdup_n_u8(0));
    uint8x16_t lo = vandq_u8(qb, m0f);
    uint8x16_t hi = vshrq_n_u8(qb, 4);
    uint8x16_t slots = vzip1q_u8(lo, hi);
    uint8x16_t sm = vreinterpretq_u8_u64(vcombine_u64(vcreate_u64(SIGN_LUT[sg[0]]), vcreate_u64(SIGN_LUT[sg[1]])));
    uint8x16_t qp = vbslq_u8(sm, vqtbl1q_u8(thi, slots), vqtbl1q_u8(tlo, slots));
    out[0] = vandq_u8(qp, m3);
    out[1] = vandq_u8(vshrq_n_u8(qp, 2), m3);
    out[2] = vandq_u8(vshrq_n_u8(qp, 4), m3);
    out[3] = vshrq_n_u8(qp, 6);
}

/* 无 DOTPROD 的通用 NEON 点积 (老 aarch64 / 未探测到 asimddp)。 */
static inline int32x4_t neon_dot_mul(int32x4_t acc, int8x16_t c, int8x16_t a) {
    acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(c), vget_low_s8(a)));
    return vpadalq_s16(acc, vmull_high_s8(c, a));
}

#define STQ_ROWS_NEON_BODY(DOT)                                                                                        \
    static const int cord[4] = Q8K_PIECE_ORDER;                                                                         \
    const int nb = w->k / QK_K;                                                                                         \
    const uint8x16_t tlo = vld1q_u8(STQ_CODEBOOK), thi = vld1q_u8(STQ_CODEBOOK + 16);                                   \
    for (int r = r0; r < r1; r++) {                                                                                     \
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);                           \
        float acc[MAX_BATCH];                                                                                          \
        for (int t = 0; t < nt; t++) acc[t] = 0.f;                                                                      \
        for (int b = 0; b < nb; b++) {                                                                                  \
            uint8x16_t codes[4][4];                                                                                    \
            for (int c = 0; c < 4; c++)                                                                                 \
                stq_chunk_codes_neon(row[b].qs + c * 8, row[b].sign + c * 2, codes[c], tlo, thi);                       \
            const float wd = f16_hw(row[b].d);                                                                          \
            for (int t = 0; t < nt; t++) {                                                                              \
                const q8k_block *y = xq + (size_t)t * nb + b;                                                           \
                int32x4_t s = vdupq_n_s32(0);                                                                           \
                for (int c = 0; c < 4; c++)                                                                             \
                    for (int l = 0; l < 4; l++)                                                                         \
                        s = (DOT)(s, vreinterpretq_s8_u8(codes[c][l]), vld1q_s8(y->qp + l * 64 + cord[c] * 16));      \
                acc[t] += wd * y->d * (float)(vaddvq_s32(s) - y->sum);                                                  \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < nt; t++) out[(size_t)t * w->n + r] = acc[t];                                                 \
    }

static void stq_rows_neon(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    STQ_ROWS_NEON_BODY(neon_dot_mul)
}

HY_ARM_DOTPROD_TARGET
static void stq_rows_neon_dotprod(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    STQ_ROWS_NEON_BODY(vdotq_s32)
}
#undef STQ_ROWS_NEON_BODY
#endif /* HY_NEON */

#ifdef HY_X86
/* 从 chunk c 的两个符号字节展开出 16 字节的 0x00/0xFF 掩码 */
__attribute__((target("avx2,f16c")))
static inline __m128i stq_sign_mask128(const uint8_t *sg, int c) {
    return _mm_set_epi64x((long long)SIGN_LUT[sg[2 * c + 1]], (long long)SIGN_LUT[sg[2 * c]]);
}

/* 解出一个权重块: A=[chunk0|chunk2], B=[chunk1|chunk3] 两个 256-bit 的 qpack 字节 */
__attribute__((target("avx2,f16c")))
static inline void stq_block_qpack_avx2(const block_stq1_0 *x, __m256i *outA, __m256i *outB, __m256i tlo, __m256i thi) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    __m256i qb = _mm256_loadu_si256((const __m256i *)x->qs);
    __m256i lo = _mm256_and_si256(qb, m0f);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m0f);
    __m256i sA = _mm256_unpacklo_epi8(lo, hi), sB = _mm256_unpackhi_epi8(lo, hi);
    __m256i mA = _mm256_set_m128i(stq_sign_mask128(x->sign, 2), stq_sign_mask128(x->sign, 0));
    __m256i mB = _mm256_set_m128i(stq_sign_mask128(x->sign, 3), stq_sign_mask128(x->sign, 1));
    __m256i a0 = _mm256_shuffle_epi8(tlo, sA), a1 = _mm256_shuffle_epi8(thi, sA);
    __m256i b0 = _mm256_shuffle_epi8(tlo, sB), b1 = _mm256_shuffle_epi8(thi, sB);
    *outA = _mm256_or_si256(_mm256_andnot_si256(mA, a0), _mm256_and_si256(mA, a1));
    *outB = _mm256_or_si256(_mm256_andnot_si256(mB, b0), _mm256_and_si256(mB, b1));
}

__attribute__((target("avx2,f16c")))
static void stq_rows_avx2(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    const int nb = w->k / QK_K;
    const __m256i tlo = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)STQ_CODEBOOK));
    const __m256i thi = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)(STQ_CODEBOOK + 16)));
    const __m256i ones = _mm256_set1_epi16(1), m3 = _mm256_set1_epi8(3);
    for (int r = r0; r < r1; r++) {
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);
        __m256 accf[MAX_BATCH];
        for (int t = 0; t < nt; t++) accf[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            __m256i qA, qB;
            stq_block_qpack_avx2(&row[b], &qA, &qB, tlo, thi);
            __m256i pA[4], pB[4];
            pA[0] = _mm256_and_si256(qA, m3); pB[0] = _mm256_and_si256(qB, m3);
            pA[1] = _mm256_and_si256(_mm256_srli_epi16(qA, 2), m3); pB[1] = _mm256_and_si256(_mm256_srli_epi16(qB, 2), m3);
            pA[2] = _mm256_and_si256(_mm256_srli_epi16(qA, 4), m3); pB[2] = _mm256_and_si256(_mm256_srli_epi16(qB, 4), m3);
            pA[3] = _mm256_and_si256(_mm256_srli_epi16(qA, 6), m3); pB[3] = _mm256_and_si256(_mm256_srli_epi16(qB, 6), m3);
            const float wd = f16_hw(row[b].d);
            for (int t = 0; t < nt; t++) {
                const q8k_block *y = xq + (size_t)t * nb + b;
                __m256i s16 = _mm256_setzero_si256();
                for (int l = 0; l < 4; l++) {
                    s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pA[l], _mm256_load_si256((const __m256i *)(y->qp + l * 64))));
                    s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pB[l], _mm256_load_si256((const __m256i *)(y->qp + l * 64 + 32))));
                }
                __m256i s32 = _mm256_madd_epi16(s16, ones);
                s32 = _mm256_sub_epi32(s32, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, y->sum));
                accf[t] = _mm256_add_ps(accf[t], _mm256_mul_ps(_mm256_cvtepi32_ps(s32), _mm256_set1_ps(wd * y->d)));
            }
        }
        for (int t = 0; t < nt; t++) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(accf[t]), _mm256_extractf128_ps(accf[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            out[(size_t)t * w->n + r] = _mm_cvtss_f32(v);
        }
    }
}

/* ---- AVX-512: 一个 zmm 装下全部 4 个 chunk; 符号处理 = 一次 pshufb + 掩码减法 (hi 码本 == 0xAA - lo 码本, 启动时自检) ---- */
#define STQ_AVX512_BODY(DOT_BLOCK)                                                                                     \
    const int nb = w->k / QK_K;                                                                                         \
    const __m512i tlo = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)STQ_CODEBOOK));                        \
    const __m512i c_aa = _mm512_set1_epi8((char)0xAA), m3 = _mm512_set1_epi8(3);                                       \
    const __m256i m0f = _mm256_set1_epi8(0x0F);                                                                         \
    const __m512i ones = _mm512_set1_epi16(1);                                                                          \
    (void)ones;                                                                                                         \
    for (int r = r0; r < r1; r++) {                                                                                     \
        const block_stq1_0 *row = (const block_stq1_0 *)(w->data + (size_t)r * w->row_bytes);                          \
        __m512 accf[MAX_BATCH];                                                                                         \
        for (int t = 0; t < nt; t++) accf[t] = _mm512_setzero_ps();                                                     \
        for (int b = 0; b < nb; b++) {                                                                                  \
            __m256i qb = _mm256_loadu_si256((const __m256i *)row[b].qs);                                                \
            __m256i lo = _mm256_and_si256(qb, m0f);                                                                     \
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(qb, 4), m0f);                                               \
            __m512i slots = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_unpacklo_epi8(lo, hi)),                   \
                                               _mm256_unpackhi_epi8(lo, hi), 1); /* [c0|c2|c1|c3] */                   \
            uint64_t sw; memcpy(&sw, row[b].sign, 8);                                                                    \
            __mmask64 km = (__mmask64)((sw & 0xFFFFull) | (((sw >> 32) & 0xFFFFull) << 16) |                           \
                                       (((sw >> 16) & 0xFFFFull) << 32) | ((sw >> 48) << 48));                         \
            __m512i q0 = _mm512_shuffle_epi8(tlo, slots);                                                               \
            __m512i qpk = _mm512_mask_sub_epi8(q0, km, c_aa, q0);                                                       \
            __m512i p0 = _mm512_and_si512(qpk, m3), p1 = _mm512_and_si512(_mm512_srli_epi16(qpk, 2), m3);               \
            __m512i p2 = _mm512_and_si512(_mm512_srli_epi16(qpk, 4), m3), p3 = _mm512_and_si512(_mm512_srli_epi16(qpk, 6), m3); \
            const float wd = f16_hw(row[b].d);                                                                          \
            for (int t = 0; t < nt; t++) {                                                                              \
                const q8k_block *y = xq + (size_t)t * nb + b;                                                           \
                __m512i a0 = _mm512_load_si512((const void *)(y->qp)), a1 = _mm512_load_si512((const void *)(y->qp + 64)); \
                __m512i a2 = _mm512_load_si512((const void *)(y->qp + 128)), a3 = _mm512_load_si512((const void *)(y->qp + 192)); \
                __m512i s32;                                                                                            \
                DOT_BLOCK                                                                                               \
                s32 = _mm512_mask_sub_epi32(s32, 1, s32, _mm512_set1_epi32(y->sum));                                    \
                accf[t] = _mm512_add_ps(accf[t], _mm512_mul_ps(_mm512_cvtepi32_ps(s32), _mm512_set1_ps(wd * y->d)));    \
            }                                                                                                           \
        }                                                                                                               \
        for (int t = 0; t < nt; t++) out[(size_t)t * w->n + r] = _mm512_reduce_add_ps(accf[t]);                         \
    }

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2,f16c")))
static void stq_rows_avx512(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    STQ_AVX512_BODY(
        __m512i s16 = _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(p0, a0), _mm512_maddubs_epi16(p1, a1)),
                                       _mm512_add_epi16(_mm512_maddubs_epi16(p2, a2), _mm512_maddubs_epi16(p3, a3)));
        s32 = _mm512_madd_epi16(s16, ones);
    )
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2,f16c")))
static void stq_rows_avx512_vnni(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    STQ_AVX512_BODY(
        s32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), p0, a0);
        s32 = _mm512_dpbusd_epi32(s32, p1, a1);
        s32 = _mm512_dpbusd_epi32(s32, p2, a2);
        s32 = _mm512_dpbusd_epi32(s32, p3, a3);
    )
}
#endif /* HY_X86 (avx2 stq) */

static stq_rows_fn g_stq_rows = stq_rows_scalar;
static const char *g_isa = "scalar";

float stq1_0_dot_dispatch(const block_stq1_0 *x, const q8k_block *y, int nb) {
    weight_t w = { (const uint8_t *)x, GT_STQ1_0, nb * QK_K, 1, (size_t)nb * sizeof(block_stq1_0), NULL, NULL, 0, 0, NULL };
    float o = 0.f;
    g_stq_rows(&w, 0, 1, y, 1, &o);
    return o;
}

#ifndef HY_PF_BYTES
#define HY_PF_BYTES 4096 /* 面板内核向前预取的字节数 (行长 512B 时预取 8 行, 1.5KB 时预取 2 行; 不要超过小 L1 的一半) */
#endif
/* ------------------------------------------------------------------ 面板重排 (STQ1_0 / Q2_0C) */
/* 面板块 (64 字节) 的字节顺序 = [chunk0 | chunk2 | chunk1 | chunk3] 的 16 字节段, 与 q8k_block.qp 的段顺序一致;
 * 每个字节的 4 个 2-bit 码依次是 lane 0..3 (权重下标 c*64 + gloc + 16*lane)。 */

bool panel_supported(int type) { return type == GT_STQ1_0 || type == GT_Q2_0C; }

void panel_repack_rows(const weight_t *src, uint8_t *dst_pk, float *dst_ps, int r0, int r1) {
    static const int cord[4] = Q8K_PIECE_ORDER;
    const int nb = src->k / QK_K;
    for (int r = r0; r < r1; r++) {
        const uint8_t *row = src->data + (size_t)r * src->row_bytes;
        for (int b = 0; b < nb; b++) {
            uint8_t *pk = dst_pk + ((size_t)r * nb + b) * 64;
            if (src->type == GT_STQ1_0) {
                const block_stq1_0 *x = (const block_stq1_0 *)row + b;
                dst_ps[(size_t)r * nb + b] = f16_to_f32(x->d);
                for (int g = 0; g < 64; g++) pk[cord[g >> 4] * 16 + (g & 15)] = stq_qpack(x, g);
            } else { /* Q2_0C: 512 权重一个块, 两个 256 半块共用缩放 */
                const block_q2_0c *x = (const block_q2_0c *)row + (b >> 1);
                const int half = b & 1;
                dst_ps[(size_t)r * nb + b] = f16_to_f32(x->d);
                for (int g = 0; g < 64; g++) {
                    int c = g >> 4, gloc = g & 15, qp = 0;
                    for (int l = 0; l < 4; l++) {
                        int widx = half * 256 + c * 64 + gloc + 16 * l;
                        qp |= ((x->qs[widx >> 2] >> ((widx & 3) * 2)) & 3) << (2 * l);
                    }
                    pk[cord[c] * 16 + (g & 15)] = (uint8_t)qp;
                }
            }
        }
    }
}

typedef void (*panel_rows_fn)(const weight_t *, int, int, const q8k_block *, int, float *);

static void panel_rows_scalar(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    const int nb = w->k / QK_K;
    for (int r = r0; r < r1; r++) {
        for (int t = 0; t < nt; t++) {
            float acc = 0.f;
            for (int b = 0; b < nb; b++) {
                const uint8_t *pk = w->pk + ((size_t)r * nb + b) * 64;
                const q8k_block *y = xq + (size_t)t * nb + b;
                int enc = 0;
                for (int l = 0; l < 4; l++)
                    for (int i = 0; i < 64; i++) enc += ((pk[i] >> (2 * l)) & 3) * y->qp[l * 64 + i];
                acc += w->ps[(size_t)r * nb + b] * y->d * (float)((enc << w->pshift) - w->poff * y->sum);
            }
            out[(size_t)t * w->n + r] = acc;
        }
    }
}

#ifdef HY_X86
/* SSE (ssse3 起, SSE4.1/4.2 机器同样用它): 内层循环只剩 and / shift / maddubs / add */
__attribute__((target("ssse3")))
static void panel_rows_ssse3(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    const int nb = w->k / QK_K;
    const int pfr = (int)(HY_PF_BYTES / ((size_t)nb * 64)) < 1 ? 1 : (int)(HY_PF_BYTES / ((size_t)nb * 64));
    const __m128i m3 = _mm_set1_epi8(3), ones = _mm_set1_epi16(1), vshift = _mm_cvtsi32_si128(w->pshift);
    const int off = w->poff;
    for (int r = r0; r < r1; r++) {
        const uint8_t *pk = w->pk + (size_t)r * nb * 64;
        const float *ps = w->ps + (size_t)r * nb;
        if (r + pfr < r1) { /* 预取前方若干行的权重 (低带宽 CPU 上有助于隐藏访存延迟) */
            const char *nx = (const char *)(pk + (size_t)nb * 64 * pfr);
            for (int b = 0; b < nb; b++) _mm_prefetch(nx + b * 64, _MM_HINT_T0);
        }
        __m128 accf[MAX_BATCH];
        for (int t = 0; t < nt; t++) accf[t] = _mm_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const float sc = ps[b];
            if (nt == 1) {
                const q8k_block *y = xq + b;
                __m128i s16 = _mm_setzero_si128();
                for (int j = 0; j < 4; j++) {
                    __m128i v = _mm_load_si128((const __m128i *)(pk + b * 64 + j * 16));
                    s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(v, m3), _mm_load_si128((const __m128i *)(y->qp + j * 16))));
                    s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(v, 2), m3), _mm_load_si128((const __m128i *)(y->qp + 64 + j * 16))));
                    s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(v, 4), m3), _mm_load_si128((const __m128i *)(y->qp + 128 + j * 16))));
                    s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(v, 6), m3), _mm_load_si128((const __m128i *)(y->qp + 192 + j * 16))));
                }
                __m128i s32 = _mm_sll_epi32(_mm_madd_epi16(s16, ones), vshift);
                s32 = _mm_sub_epi32(s32, _mm_cvtsi32_si128(off * y->sum));
                accf[0] = _mm_add_ps(accf[0], _mm_mul_ps(_mm_cvtepi32_ps(s32), _mm_set1_ps(sc * y->d)));
            } else {
                __m128i p[4][4]; /* [j][l] */
                for (int j = 0; j < 4; j++) {
                    __m128i v = _mm_load_si128((const __m128i *)(pk + b * 64 + j * 16));
                    p[j][0] = _mm_and_si128(v, m3);
                    p[j][1] = _mm_and_si128(_mm_srli_epi16(v, 2), m3);
                    p[j][2] = _mm_and_si128(_mm_srli_epi16(v, 4), m3);
                    p[j][3] = _mm_and_si128(_mm_srli_epi16(v, 6), m3);
                }
                for (int t = 0; t < nt; t++) {
                    const q8k_block *y = xq + (size_t)t * nb + b;
                    __m128i s16 = _mm_setzero_si128();
                    for (int l = 0; l < 4; l++)
                        for (int j = 0; j < 4; j++)
                            s16 = _mm_add_epi16(s16, _mm_maddubs_epi16(p[j][l], _mm_load_si128((const __m128i *)(y->qp + l * 64 + j * 16))));
                    __m128i s32 = _mm_sll_epi32(_mm_madd_epi16(s16, ones), vshift);
                    s32 = _mm_sub_epi32(s32, _mm_cvtsi32_si128(off * y->sum));
                    accf[t] = _mm_add_ps(accf[t], _mm_mul_ps(_mm_cvtepi32_ps(s32), _mm_set1_ps(sc * y->d)));
                }
            }
        }
        for (int t = 0; t < nt; t++) {
            __m128 v = _mm_add_ps(accf[t], _mm_movehl_ps(accf[t], accf[t]));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            out[(size_t)t * w->n + r] = _mm_cvtss_f32(v);
        }
    }
}

__attribute__((target("avx2")))
static void panel_rows_avx2(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    const int nb = w->k / QK_K;
    const int pfr = (int)(HY_PF_BYTES / ((size_t)nb * 64)) < 1 ? 1 : (int)(HY_PF_BYTES / ((size_t)nb * 64));
    const __m256i m3 = _mm256_set1_epi8(3), ones = _mm256_set1_epi16(1);
    const __m128i vshift = _mm_cvtsi32_si128(w->pshift);
    const int off = w->poff;
    for (int r = r0; r < r1; r++) {
        const uint8_t *pk = w->pk + (size_t)r * nb * 64;
        const float *ps = w->ps + (size_t)r * nb;
        if (r + pfr < r1) {
            const char *nx = (const char *)(pk + (size_t)nb * 64 * pfr);
            for (int b = 0; b < nb; b++) _mm_prefetch(nx + b * 64, _MM_HINT_T0);
        }
        __m256 accf[MAX_BATCH];
        for (int t = 0; t < nt; t++) accf[t] = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const float sc = ps[b];
            __m256i vA = _mm256_load_si256((const __m256i *)(pk + b * 64)), vB = _mm256_load_si256((const __m256i *)(pk + b * 64 + 32));
            __m256i pA0 = _mm256_and_si256(vA, m3), pB0 = _mm256_and_si256(vB, m3);
            __m256i pA1 = _mm256_and_si256(_mm256_srli_epi16(vA, 2), m3), pB1 = _mm256_and_si256(_mm256_srli_epi16(vB, 2), m3);
            __m256i pA2 = _mm256_and_si256(_mm256_srli_epi16(vA, 4), m3), pB2 = _mm256_and_si256(_mm256_srli_epi16(vB, 4), m3);
            __m256i pA3 = _mm256_and_si256(_mm256_srli_epi16(vA, 6), m3), pB3 = _mm256_and_si256(_mm256_srli_epi16(vB, 6), m3);
            for (int t = 0; t < nt; t++) {
                const q8k_block *y = xq + (size_t)t * nb + b;
                const __m256i *q = (const __m256i *)y->qp;
                __m256i s16 = _mm256_maddubs_epi16(pA0, _mm256_load_si256(q));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pB0, _mm256_load_si256(q + 1)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pA1, _mm256_load_si256(q + 2)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pB1, _mm256_load_si256(q + 3)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pA2, _mm256_load_si256(q + 4)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pB2, _mm256_load_si256(q + 5)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pA3, _mm256_load_si256(q + 6)));
                s16 = _mm256_add_epi16(s16, _mm256_maddubs_epi16(pB3, _mm256_load_si256(q + 7)));
                __m256i s32 = _mm256_sll_epi32(_mm256_madd_epi16(s16, ones), vshift);
                s32 = _mm256_sub_epi32(s32, _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, off * y->sum));
                accf[t] = _mm256_add_ps(accf[t], _mm256_mul_ps(_mm256_cvtepi32_ps(s32), _mm256_set1_ps(sc * y->d)));
            }
        }
        for (int t = 0; t < nt; t++) {
            __m128 v = _mm_add_ps(_mm256_castps256_ps128(accf[t]), _mm256_extractf128_ps(accf[t], 1));
            v = _mm_add_ps(v, _mm_movehl_ps(v, v));
            v = _mm_add_ss(v, _mm_shuffle_ps(v, v, 1));
            out[(size_t)t * w->n + r] = _mm_cvtss_f32(v);
        }
    }
}

#define PANEL_AVX512_BODY(DOT_BLOCK)                                                                                  \
    const int nb = w->k / QK_K;                                                                                         \
    const int pfr = (int)(HY_PF_BYTES / ((size_t)nb * 64)) < 1 ? 1 : (int)(HY_PF_BYTES / ((size_t)nb * 64));          \
    const __m512i m3 = _mm512_set1_epi8(3);                                                                             \
    const __m512i ones = _mm512_set1_epi16(1);                                                                          \
    const __m128i vshift = _mm_cvtsi32_si128(w->pshift);                                                                \
    const int off = w->poff;                                                                                            \
    (void)ones;                                                                                                         \
    for (int r = r0; r < r1; r++) {                                                                                     \
        const uint8_t *pk = w->pk + (size_t)r * nb * 64;                                                               \
        const float *ps = w->ps + (size_t)r * nb;                                                                       \
        if (r + pfr < r1) {                                                                                             \
            const char *nx = (const char *)(pk + (size_t)nb * 64 * pfr);                                                \
            for (int b = 0; b < nb; b++) _mm_prefetch(nx + b * 64, _MM_HINT_T0);                                        \
        }                                                                                                               \
        __m512 accf[MAX_BATCH];                                                                                         \
        for (int t = 0; t < nt; t++) accf[t] = _mm512_setzero_ps();                                                     \
        for (int b = 0; b < nb; b++) {                                                                                  \
            const float sc = ps[b];                                                                                     \
            __m512i v = _mm512_load_si512((const void *)(pk + b * 64));                                                 \
            __m512i p0 = _mm512_and_si512(v, m3), p1 = _mm512_and_si512(_mm512_srli_epi16(v, 2), m3);                   \
            __m512i p2 = _mm512_and_si512(_mm512_srli_epi16(v, 4), m3), p3 = _mm512_and_si512(_mm512_srli_epi16(v, 6), m3); \
            for (int t = 0; t < nt; t++) {                                                                              \
                const q8k_block *y = xq + (size_t)t * nb + b;                                                           \
                __m512i a0 = _mm512_load_si512((const void *)(y->qp)), a1 = _mm512_load_si512((const void *)(y->qp + 64)); \
                __m512i a2 = _mm512_load_si512((const void *)(y->qp + 128)), a3 = _mm512_load_si512((const void *)(y->qp + 192)); \
                __m512i s32;                                                                                            \
                DOT_BLOCK                                                                                               \
                s32 = _mm512_sll_epi32(s32, vshift);                                                                    \
                s32 = _mm512_mask_sub_epi32(s32, 1, s32, _mm512_set1_epi32(off * y->sum));                              \
                accf[t] = _mm512_add_ps(accf[t], _mm512_mul_ps(_mm512_cvtepi32_ps(s32), _mm512_set1_ps(sc * y->d)));    \
            }                                                                                                           \
        }                                                                                                               \
        for (int t = 0; t < nt; t++) out[(size_t)t * w->n + r] = _mm512_reduce_add_ps(accf[t]);                         \
    }

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx2")))
static void panel_rows_avx512(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    PANEL_AVX512_BODY(
        __m512i s16 = _mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(p0, a0), _mm512_maddubs_epi16(p1, a1)),
                                       _mm512_add_epi16(_mm512_maddubs_epi16(p2, a2), _mm512_maddubs_epi16(p3, a3)));
        s32 = _mm512_madd_epi16(s16, ones);
    )
}

__attribute__((target("avx512f,avx512bw,avx512vl,avx512dq,avx512vnni,avx2")))
static void panel_rows_avx512_vnni(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    PANEL_AVX512_BODY(
        s32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), p0, a0);
        s32 = _mm512_dpbusd_epi32(s32, p1, a1);
        s32 = _mm512_dpbusd_epi32(s32, p2, a2);
        s32 = _mm512_dpbusd_epi32(s32, p3, a3);
    )
}
#endif /* HY_X86 (panel) */

#ifdef HY_NEON
/* 行外提 + 预取下一组行: 权重顺序读 (带宽主导); nt==1 不解包临时表; 多 token 时权重只解一次。
 * 注意: 块外提多行会跨 nb*64 大跨度读权重, 在手机上实测更慢, 不要改回去。 */
#define PANEL_ROWS_NEON_BODY(DOT)                                                                                      \
    const int nb = w->k / QK_K;                                                                                         \
    const int pfr = (int)(HY_PF_BYTES / ((size_t)nb * 64));                                                             \
    const int pref = pfr < 1 ? 1 : pfr;                                                                                 \
    const uint8x16_t m3 = vdupq_n_u8(3);                                                                                \
    const int sh = w->pshift, off = w->poff;                                                                            \
    for (int r = r0; r < r1; r++) {                                                                                     \
        const uint8_t *pk = w->pk + (size_t)r * (size_t)nb * 64;                                                        \
        const float *ps = w->ps + (size_t)r * (size_t)nb;                                                                \
        if (r + pref < r1) {                                                                                           \
            const uint8_t *nx = pk + (size_t)nb * 64 * (size_t)pref;                                                    \
            for (int b = 0; b < nb; b += 4) {                                                                           \
                __builtin_prefetch(nx + (size_t)b * 64, 0, 3);                                                         \
                if (b + 2 < nb) __builtin_prefetch(nx + (size_t)(b + 2) * 64, 0, 3);                                   \
            }                                                                                                          \
        }                                                                                                              \
        float acc[MAX_BATCH];                                                                                          \
        for (int t = 0; t < nt; t++) acc[t] = 0.f;                                                                      \
        if (nt == 1) {                                                                                                 \
            const q8k_block *y0 = xq;                                                                                   \
            int b = 0;                                                                                                 \
            for (; b + 1 < nb; b += 2) {                                                                                \
                int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);                                                     \
                const int8_t *qp0 = y0[b].qp, *qp1 = y0[b + 1].qp;                                                      \
                for (int j = 0; j < 4; j++) {                                                                           \
                    uint8x16_t v0 = vld1q_u8(pk + b * 64 + j * 16);                                                     \
                    uint8x16_t v1 = vld1q_u8(pk + (b + 1) * 64 + j * 16);                                               \
                    s0 = (DOT)(s0, vreinterpretq_s8_u8(vandq_u8(v0, m3)), vld1q_s8(qp0 + j * 16));                      \
                    s1 = (DOT)(s1, vreinterpretq_s8_u8(vandq_u8(v1, m3)), vld1q_s8(qp1 + j * 16));                      \
                    s0 = (DOT)(s0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v0, 2), m3)), vld1q_s8(qp0 + 64 + j * 16)); \
                    s1 = (DOT)(s1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v1, 2), m3)), vld1q_s8(qp1 + 64 + j * 16)); \
                    s0 = (DOT)(s0, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v0, 4), m3)), vld1q_s8(qp0 + 128 + j * 16));\
                    s1 = (DOT)(s1, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v1, 4), m3)), vld1q_s8(qp1 + 128 + j * 16));\
                    s0 = (DOT)(s0, vreinterpretq_s8_u8(vshrq_n_u8(v0, 6)), vld1q_s8(qp0 + 192 + j * 16));               \
                    s1 = (DOT)(s1, vreinterpretq_s8_u8(vshrq_n_u8(v1, 6)), vld1q_s8(qp1 + 192 + j * 16));               \
                }                                                                                                      \
                acc[0] += ps[b] * y0[b].d * (float)((vaddvq_s32(s0) << sh) - off * y0[b].sum);                          \
                acc[0] += ps[b + 1] * y0[b + 1].d * (float)((vaddvq_s32(s1) << sh) - off * y0[b + 1].sum);             \
            }                                                                                                          \
            for (; b < nb; b++) {                                                                                      \
                const int8_t *qp = y0[b].qp;                                                                            \
                int32x4_t s = vdupq_n_s32(0);                                                                           \
                for (int j = 0; j < 4; j++) {                                                                           \
                    uint8x16_t v = vld1q_u8(pk + b * 64 + j * 16);                                                      \
                    s = (DOT)(s, vreinterpretq_s8_u8(vandq_u8(v, m3)), vld1q_s8(qp + j * 16));                          \
                    s = (DOT)(s, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3)), vld1q_s8(qp + 64 + j * 16));     \
                    s = (DOT)(s, vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3)), vld1q_s8(qp + 128 + j * 16));    \
                    s = (DOT)(s, vreinterpretq_s8_u8(vshrq_n_u8(v, 6)), vld1q_s8(qp + 192 + j * 16));                  \
                }                                                                                                      \
                acc[0] += ps[b] * y0[b].d * (float)((vaddvq_s32(s) << sh) - off * y0[b].sum);                           \
            }                                                                                                          \
        } else {                                                                                                       \
            for (int b = 0; b < nb; b++) {                                                                              \
                int8x16_t p[4][4];                                                                                     \
                for (int j = 0; j < 4; j++) {                                                                           \
                    uint8x16_t v = vld1q_u8(pk + b * 64 + j * 16);                                                      \
                    p[j][0] = vreinterpretq_s8_u8(vandq_u8(v, m3));                                                    \
                    p[j][1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 2), m3));                                     \
                    p[j][2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(v, 4), m3));                                     \
                    p[j][3] = vreinterpretq_s8_u8(vshrq_n_u8(v, 6));                                                   \
                }                                                                                                      \
                const float sc = ps[b];                                                                                \
                int t = 0;                                                                                             \
                for (; t + 1 < nt; t += 2) {                                                                            \
                    const q8k_block *y0 = xq + (size_t)t * (size_t)nb + (size_t)b;                                      \
                    const q8k_block *y1 = y0 + nb;                                                                      \
                    int32x4_t s0 = vdupq_n_s32(0), s1 = vdupq_n_s32(0);                                                 \
                    for (int l = 0; l < 4; l++)                                                                         \
                        for (int j = 0; j < 4; j++) {                                                                   \
                            int8x16_t pv = p[j][l];                                                                     \
                            s0 = (DOT)(s0, pv, vld1q_s8(y0->qp + l * 64 + j * 16));                                    \
                            s1 = (DOT)(s1, pv, vld1q_s8(y1->qp + l * 64 + j * 16));                                    \
                        }                                                                                              \
                    acc[t] += sc * y0->d * (float)((vaddvq_s32(s0) << sh) - off * y0->sum);                             \
                    acc[t + 1] += sc * y1->d * (float)((vaddvq_s32(s1) << sh) - off * y1->sum);                         \
                }                                                                                                      \
                for (; t < nt; t++) {                                                                                  \
                    const q8k_block *y = xq + (size_t)t * (size_t)nb + (size_t)b;                                       \
                    int32x4_t s = vdupq_n_s32(0);                                                                       \
                    for (int l = 0; l < 4; l++)                                                                         \
                        for (int j = 0; j < 4; j++)                                                                     \
                            s = (DOT)(s, p[j][l], vld1q_s8(y->qp + l * 64 + j * 16));                                  \
                    acc[t] += sc * y->d * (float)((vaddvq_s32(s) << sh) - off * y->sum);                                \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        for (int t = 0; t < nt; t++) out[(size_t)t * (size_t)w->n + (size_t)r] = acc[t];                                 \
    }

static void panel_rows_neon(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    PANEL_ROWS_NEON_BODY(neon_dot_mul)
}

HY_ARM_DOTPROD_TARGET
static void panel_rows_neon_dotprod(const weight_t *w, int r0, int r1, const q8k_block *xq, int nt, float *out) {
    PANEL_ROWS_NEON_BODY(vdotq_s32)
}
#undef PANEL_ROWS_NEON_BODY
#endif /* HY_NEON (panel) */

static panel_rows_fn g_panel_rows = panel_rows_scalar;

/* ------------------------------------------------------------------ Q6_K (标准 ggml 布局) */

void q6_k_dequant_row(const block_q6_k *x, float *y, int k) {
    for (int i = 0; i < k / QK_K; i++) {
        float d = f16_to_f32(x[i].d);
        const uint8_t *ql = x[i].ql, *qh = x[i].qh;
        const int8_t *sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int q1 = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                y[l] = d * sc[is + 0] * (float)(q1 - 32);
                y[l + 32] = d * sc[is + 2] * (float)(q2 - 32);
                y[l + 64] = d * sc[is + 4] * (float)(q3 - 32);
                y[l + 96] = d * sc[is + 6] * (float)(q4 - 32);
            }
            y += 128; ql += 64; qh += 32; sc += 8;
        }
    }
}

float q6_k_dot_scalar(const block_q6_k *x, const q8k_block *y, int nb) {
    float sumf = 0.f;
    int8_t aux[QK_K];
    for (int i = 0; i < nb; i++) {
        const uint8_t *q4 = x[i].ql, *qh = x[i].qh;
        int8_t *a = aux;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; l++) {
                a[l] = (int8_t)(((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                a[l + 32] = (int8_t)(((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                a[l + 64] = (int8_t)(((q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                a[l + 96] = (int8_t)(((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
            }
            a += 128; q4 += 64; qh += 32;
        }
        int acc = 0;
        const int8_t *q8 = y[i].qs;
        a = aux;
        for (int j = 0; j < QK_K / 16; j++) {
            int s = x[i].scales[j], t = 0;
            for (int l = 0; l < 16; l++) t += q8[l] * a[l];
            acc += s * t;
            q8 += 16; a += 16;
        }
        sumf += f16_to_f32(x[i].d) * y[i].d * (float)acc;
    }
    return sumf;
}

/* ------------------------------------------------------------------ Q4_K (标准 ggml 布局) */

static inline void k4_scale_min(const uint8_t *q, int j, uint8_t *d, uint8_t *m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)); *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4)); }
}

static void q4_k_dequant_row(const block_q4_k *x, float *y, int k) {
    for (int i = 0; i < k / QK_K; i++) {
        const float d = f16_to_f32(x[i].d), mn = f16_to_f32(x[i].dmin);
        const uint8_t *q = x[i].qs;
        for (int j = 0, is = 0; j < QK_K; j += 64, is += 2, q += 32) {
            uint8_t sc, m;
            k4_scale_min(x[i].scales, is, &sc, &m);
            float d1 = d * sc, m1 = mn * m;
            k4_scale_min(x[i].scales, is + 1, &sc, &m);
            float d2 = d * sc, m2 = mn * m;
            for (int l = 0; l < 32; l++) y[l] = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; l++) y[32 + l] = d2 * (q[l] >> 4) - m2;
            y += 64;
        }
    }
}

float q4_k_dot_scalar(const block_q4_k *x, const q8k_block *y, int nb) {
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) k4_scale_min(x[i].scales, j, &sc[j], &mn[j]);
        int sumi = 0, summ = 0;
        for (int g = 0; g < 4; g++) {
            const uint8_t *q = x[i].qs + 32 * g;
            int s0 = 0, s1 = 0;
            for (int l = 0; l < 32; l++) { s0 += (q[l] & 0xF) * y[i].qs[64 * g + l]; s1 += (q[l] >> 4) * y[i].qs[64 * g + 32 + l]; }
            sumi += sc[2 * g] * s0 + sc[2 * g + 1] * s1;
        }
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_to_f32(x[i].d) * (float)sumi - f16_to_f32(x[i].dmin) * (float)summ);
    }
    return sumf;
}

typedef float (*kq_dot_fn4)(const block_q4_k *, const q8k_block *, int);
static kq_dot_fn4 g_q4k_dot = q4_k_dot_scalar;

#ifdef HY_X86
__attribute__((target("ssse3")))
static inline int hsum_epi32_ssse3(__m128i v) {
    v = _mm_hadd_epi32(v, v);
    v = _mm_hadd_epi32(v, v);
    return _mm_cvtsi128_si32(v);
}

/* SSSE3: 每个 32 元素子块 = 两个 16 字节向量的 maddubs, 之后乘子块缩放 */
__attribute__((target("ssse3")))
static float q4_k_dot_ssse3(const block_q4_k *x, const q8k_block *y, int nb) {
    const __m128i m0f = _mm_set1_epi8(0x0F);
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) k4_scale_min(x[i].scales, j, &sc[j], &mn[j]);
        __m128i sumi = _mm_setzero_si128();
        for (int g = 0; g < 4; g++) {
            __m128i q0 = _mm_loadu_si128((const __m128i *)(x[i].qs + 32 * g)), q1 = _mm_loadu_si128((const __m128i *)(x[i].qs + 32 * g + 16));
            const int8_t *a = y[i].qs + 64 * g;
            __m128i lo = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(q0, m0f), _mm_loadu_si128((const __m128i *)a)),
                                       _mm_maddubs_epi16(_mm_and_si128(q1, m0f), _mm_loadu_si128((const __m128i *)(a + 16))));
            __m128i hi = _mm_add_epi16(_mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(q0, 4), m0f), _mm_loadu_si128((const __m128i *)(a + 32))),
                                       _mm_maddubs_epi16(_mm_and_si128(_mm_srli_epi16(q1, 4), m0f), _mm_loadu_si128((const __m128i *)(a + 48))));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(lo, _mm_set1_epi16(sc[2 * g])));
            sumi = _mm_add_epi32(sumi, _mm_madd_epi16(hi, _mm_set1_epi16(sc[2 * g + 1])));
        }
        int summ = 0;
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_to_f32(x[i].d) * (float)hsum_epi32_ssse3(sumi) - f16_to_f32(x[i].dmin) * (float)summ);
    }
    return sumf;
}

__attribute__((target("avx2,f16c")))
static inline int hsum_epi32_avx2(__m256i v) {
    __m128i r = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    r = _mm_hadd_epi32(r, r);
    r = _mm_hadd_epi32(r, r);
    return _mm_cvtsi128_si32(r);
}

__attribute__((target("avx2,f16c")))
static float q4_k_dot_avx2(const block_q4_k *x, const q8k_block *y, int nb) {
    const __m256i m0f = _mm256_set1_epi8(0x0F);
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) k4_scale_min(x[i].scales, j, &sc[j], &mn[j]);
        __m256i sumi = _mm256_setzero_si256();
        for (int g = 0; g < 4; g++) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(x[i].qs + 32 * g));
            __m256i lo = _mm256_and_si256(q, m0f), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m0f);
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 64 * g)), a1 = _mm256_loadu_si256((const __m256i *)(y[i].qs + 64 * g + 32));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(lo, a0), _mm256_set1_epi16(sc[2 * g])));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(hi, a1), _mm256_set1_epi16(sc[2 * g + 1])));
        }
        int summ = 0;
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_hw(x[i].d) * (float)hsum_epi32_avx2(sumi) - f16_hw(x[i].dmin) * (float)summ);
    }
    return sumf;
}

#endif /* HY_X86 (q4k) */

#ifdef HY_NEON
static float q4_k_dot_neon(const block_q4_k *x, const q8k_block *y, int nb) {
    const uint8x16_t m0f = vdupq_n_u8(0x0F);
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        uint8_t sc[8], mn[8];
        for (int j = 0; j < 8; j++) k4_scale_min(x[i].scales, j, &sc[j], &mn[j]);
        int32x4_t sumi = vdupq_n_s32(0);
        for (int g = 0; g < 4; g++) {
            const uint8_t *q = x[i].qs + 32 * g;
            const int8_t *a = y[i].qs + 64 * g;
            uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
            uint8x16_t lo0 = vandq_u8(q0, m0f), lo1 = vandq_u8(q1, m0f);
            uint8x16_t hi0 = vandq_u8(vshrq_n_u8(q0, 4), m0f), hi1 = vandq_u8(vshrq_n_u8(q1, 4), m0f);
            int8x16_t a0 = vld1q_s8(a), a1 = vld1q_s8(a + 16), a2 = vld1q_s8(a + 32), a3 = vld1q_s8(a + 48);
#ifdef __ARM_FEATURE_DOTPROD
            int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(lo0), a0);
            s0 = vdotq_s32(s0, vreinterpretq_s8_u8(lo1), a1);
            int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(hi0), a2);
            s1 = vdotq_s32(s1, vreinterpretq_s8_u8(hi1), a3);
#else
            int32x4_t s0 = vpaddlq_s16(vmull_s8(vget_low_s8(vreinterpretq_s8_u8(lo0)), vget_low_s8(a0)));
            s0 = vpadalq_s16(s0, vmull_high_s8(vreinterpretq_s8_u8(lo0), a0));
            s0 = vpadalq_s16(s0, vmull_s8(vget_low_s8(vreinterpretq_s8_u8(lo1)), vget_low_s8(a1)));
            s0 = vpadalq_s16(s0, vmull_high_s8(vreinterpretq_s8_u8(lo1), a1));
            int32x4_t s1 = vpaddlq_s16(vmull_s8(vget_low_s8(vreinterpretq_s8_u8(hi0)), vget_low_s8(a2)));
            s1 = vpadalq_s16(s1, vmull_high_s8(vreinterpretq_s8_u8(hi0), a2));
            s1 = vpadalq_s16(s1, vmull_s8(vget_low_s8(vreinterpretq_s8_u8(hi1)), vget_low_s8(a3)));
            s1 = vpadalq_s16(s1, vmull_high_s8(vreinterpretq_s8_u8(hi1), a3));
#endif
            sumi = vmlaq_n_s32(sumi, s0, sc[2 * g]);
            sumi = vmlaq_n_s32(sumi, s1, sc[2 * g + 1]);
        }
        int summ = 0;
        for (int j = 0; j < 8; j++) summ += mn[j] * (y[i].bsums[2 * j] + y[i].bsums[2 * j + 1]);
        sumf += y[i].d * (f16_to_f32(x[i].d) * (float)vaddvq_s32(sumi) - f16_to_f32(x[i].dmin) * (float)summ);
    }
    return sumf;
}
#endif /* HY_NEON (q4k) */

typedef float (*q6_dot_fn)(const block_q6_k *, const q8k_block *, int);
static q6_dot_fn g_q6_dot = q6_k_dot_scalar;

#ifdef HY_X86
/* 思路同 llama.cpp: 用无符号 6-bit 值做 maddubs, 再减去 32 * Σ(scale_j * bsum_j) */
__attribute__((target("ssse3")))
static float q6_k_dot_ssse3(const block_q6_k *x, const q8k_block *y, int nb) {
    const __m128i m0f = _mm_set1_epi8(0x0F), m30 = _mm_set1_epi8(0x30), zero = _mm_setzero_si128();
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        __m128i sumi = zero;
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = x[i].ql + 64 * n, *qh = x[i].qh + 32 * n;
            const int8_t *q8 = y[i].qs + 128 * n;
            for (int h = 0; h < 2; h++) {
                __m128i qlA = _mm_loadu_si128((const __m128i *)(ql + 16 * h));
                __m128i qlB = _mm_loadu_si128((const __m128i *)(ql + 32 + 16 * h));
                __m128i qhv = _mm_loadu_si128((const __m128i *)(qh + 16 * h));
                __m128i v[4];
                v[0] = _mm_or_si128(_mm_and_si128(qlA, m0f), _mm_and_si128(_mm_slli_epi16(qhv, 4), m30));
                v[1] = _mm_or_si128(_mm_and_si128(qlB, m0f), _mm_and_si128(_mm_slli_epi16(qhv, 2), m30));
                v[2] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlA, 4), m0f), _mm_and_si128(qhv, m30));
                v[3] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlB, 4), m0f), _mm_and_si128(_mm_srli_epi16(qhv, 2), m30));
                for (int m = 0; m < 4; m++) {
                    __m128i a = _mm_loadu_si128((const __m128i *)(q8 + 32 * m + 16 * h));
                    int j = 8 * n + 2 * m + h;
                    sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_maddubs_epi16(v[m], a), _mm_set1_epi16(x[i].scales[j])));
                }
            }
        }
        int corr = 0;
        for (int j = 0; j < 16; j++) corr += x[i].scales[j] * y[i].bsums[j];
        sumi = _mm_hadd_epi32(sumi, sumi);
        sumi = _mm_hadd_epi32(sumi, sumi);
        sumf += f16_to_f32(x[i].d) * y[i].d * (float)(_mm_cvtsi128_si32(sumi) - 32 * corr);
    }
    return sumf;
}

__attribute__((target("sse4.1")))
static float q6_k_dot_sse41(const block_q6_k *x, const q8k_block *y, int nb) {
    const __m128i m0f = _mm_set1_epi8(0x0F), m30 = _mm_set1_epi8(0x30), zero = _mm_setzero_si128();
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        __m128i sumi = zero;
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = x[i].ql + 64 * n, *qh = x[i].qh + 32 * n;
            const int8_t *q8 = y[i].qs + 128 * n;
            for (int h = 0; h < 2; h++) {
                __m128i qlA = _mm_loadu_si128((const __m128i *)(ql + 16 * h));
                __m128i qlB = _mm_loadu_si128((const __m128i *)(ql + 32 + 16 * h));
                __m128i qhv = _mm_loadu_si128((const __m128i *)(qh + 16 * h));
                __m128i v[4];
                v[0] = _mm_or_si128(_mm_and_si128(qlA, m0f), _mm_and_si128(_mm_slli_epi16(qhv, 4), m30));
                v[1] = _mm_or_si128(_mm_and_si128(qlB, m0f), _mm_and_si128(_mm_slli_epi16(qhv, 2), m30));
                v[2] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlA, 4), m0f), _mm_and_si128(qhv, m30));
                v[3] = _mm_or_si128(_mm_and_si128(_mm_srli_epi16(qlB, 4), m0f), _mm_and_si128(_mm_srli_epi16(qhv, 2), m30));
                for (int m = 0; m < 4; m++) {
                    __m128i a = _mm_loadu_si128((const __m128i *)(q8 + 32 * m + 16 * h));
                    int j = 8 * n + 2 * m + h;
                    sumi = _mm_add_epi32(sumi, _mm_madd_epi16(_mm_maddubs_epi16(v[m], a), _mm_set1_epi16(x[i].scales[j])));
                }
            }
        }
        /* pmovsxbw 把 16 个 int8 scale 扩成 int16, 与 bsums 做 madd: 修正项全部向量化 */
        __m128i scA = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        __m128i scB = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i *)(x[i].scales + 8)));
        __m128i corr = _mm_add_epi32(_mm_madd_epi16(scA, _mm_loadu_si128((const __m128i *)y[i].bsums)),
                                     _mm_madd_epi16(scB, _mm_loadu_si128((const __m128i *)(y[i].bsums + 8))));
        sumi = _mm_sub_epi32(sumi, _mm_slli_epi32(corr, 5));
        sumi = _mm_hadd_epi32(sumi, sumi);
        sumi = _mm_hadd_epi32(sumi, sumi);
        sumf += f16_to_f32(x[i].d) * y[i].d * (float)_mm_cvtsi128_si32(sumi);
    }
    return sumf;
}

__attribute__((target("avx2,f16c")))
static float q6_k_dot_avx2(const block_q6_k *x, const q8k_block *y, int nb) {
    const __m256i m0f = _mm256_set1_epi8(0x0F), m30 = _mm256_set1_epi8(0x30);
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; i++) {
        __m256i sumi = _mm256_setzero_si256();
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = x[i].ql + 64 * n, *qh = x[i].qh + 32 * n;
            const int8_t *q8 = y[i].qs + 128 * n;
            __m256i ql0 = _mm256_loadu_si256((const __m256i *)ql), ql1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i qhv = _mm256_loadu_si256((const __m256i *)qh);
            __m256i v[4];
            v[0] = _mm256_or_si256(_mm256_and_si256(ql0, m0f), _mm256_and_si256(_mm256_slli_epi16(qhv, 4), m30));
            v[1] = _mm256_or_si256(_mm256_and_si256(ql1, m0f), _mm256_and_si256(_mm256_slli_epi16(qhv, 2), m30));
            v[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql0, 4), m0f), _mm256_and_si256(qhv, m30));
            v[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(ql1, 4), m0f), _mm256_and_si256(_mm256_srli_epi16(qhv, 2), m30));
            for (int m = 0; m < 4; m++) {
                __m256i a = _mm256_loadu_si256((const __m256i *)(q8 + 32 * m));
                int j = 8 * n + 2 * m;
                __m256i sc = _mm256_inserti128_si256(_mm256_castsi128_si256(_mm_set1_epi16(x[i].scales[j])), _mm_set1_epi16(x[i].scales[j + 1]), 1);
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(_mm256_maddubs_epi16(v[m], a), sc));
            }
        }
        __m256i sc16 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)x[i].scales));
        __m256i bs = _mm256_loadu_si256((const __m256i *)y[i].bsums);
        __m256i corr = _mm256_slli_epi32(_mm256_madd_epi16(sc16, bs), 5);
        __m256 d = _mm256_set1_ps(f16_hw(x[i].d) * y[i].d);
        accf = _mm256_add_ps(accf, _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_sub_epi32(sumi, corr)), d));
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(accf), _mm256_extractf128_ps(accf, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return _mm_cvtss_f32(r);
}
#endif /* HY_X86 (q6) */

#ifdef HY_NEON
static float q6_k_dot_neon(const block_q6_k *x, const q8k_block *y, int nb) {
    const uint8x16_t m0f = vdupq_n_u8(0x0F), m3 = vdupq_n_u8(3), c32 = vdupq_n_u8(32);
    float sumf = 0.f;
    for (int i = 0; i < nb; i++) {
        int32x4_t sumi = vdupq_n_s32(0);
        for (int n = 0; n < 2; n++) {
            const uint8_t *ql = x[i].ql + 64 * n, *qh = x[i].qh + 32 * n;
            const int8_t *q8 = y[i].qs + 128 * n;
            for (int h = 0; h < 2; h++) {
                uint8x16_t qlA = vld1q_u8(ql + 16 * h), qlB = vld1q_u8(ql + 32 + 16 * h), qhv = vld1q_u8(qh + 16 * h);
                uint8x16_t v[4];
                v[0] = vorrq_u8(vandq_u8(qlA, m0f), vshlq_n_u8(vandq_u8(qhv, m3), 4));
                v[1] = vorrq_u8(vandq_u8(qlB, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 2), m3), 4));
                v[2] = vorrq_u8(vshrq_n_u8(qlA, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(qhv, 4), m3), 4));
                v[3] = vorrq_u8(vshrq_n_u8(qlB, 4), vshlq_n_u8(vshrq_n_u8(qhv, 6), 4));
                for (int m = 0; m < 4; m++) {
                    int8x16_t q = vreinterpretq_s8_u8(vsubq_u8(v[m], c32)); /* 6-bit 值 - 32, 范围 [-32, 31] */
                    int8x16_t a = vld1q_s8(q8 + 32 * m + 16 * h);
                    int j = 8 * n + 2 * m + h;
#ifdef __ARM_FEATURE_DOTPROD
                    int32x4_t t = vdotq_s32(vdupq_n_s32(0), q, a);
#else
                    int32x4_t t = vpaddlq_s16(vmull_s8(vget_low_s8(q), vget_low_s8(a)));
                    t = vpadalq_s16(t, vmull_high_s8(q, a));
#endif
                    sumi = vmlaq_n_s32(sumi, t, x[i].scales[j]);
                }
            }
        }
        sumf += f16_hw(x[i].d) * y[i].d * (float)vaddvq_s32(sumi);
    }
    return sumf;
}
#endif /* HY_NEON (q6) */

/* ------------------------------------------------------------------ Q8_0 / F16 / F32 (浮点激活路径) */

static float dot_q8_0_f32(const block_q8_0 *x, const float *y, int k) {
    float sum = 0.f;
    for (int b = 0; b < k / 32; b++) {
        float s = 0.f;
        for (int j = 0; j < 32; j++) s += (float)x[b].qs[j] * y[b * 32 + j];
        sum += f16_to_f32(x[b].d) * s;
    }
    return sum;
}

typedef float (*q8_dot_fn)(const block_q8_0 *, const float *, int);
static q8_dot_fn g_q8_dot = dot_q8_0_f32;

#ifdef HY_X86
/* SSE4.1: pmovsxbd 扩展 int8→int32 再转 float 点积 */
__attribute__((target("sse4.1")))
static float dot_q8_0_sse41(const block_q8_0 *x, const float *y, int k) {
    float sum = 0.f;
    for (int b = 0; b < k / 32; b++) {
        const float *yy = y + b * 32;
        __m128 acc = _mm_setzero_ps();
        for (int j = 0; j < 32; j += 4) {
            __m128i q = _mm_cvtepi8_epi32(_mm_cvtsi32_si128(*(const int *)(x[b].qs + j)));
            acc = _mm_add_ps(acc, _mm_mul_ps(_mm_cvtepi32_ps(q), _mm_loadu_ps(yy + j)));
        }
        float t[4];
        _mm_storeu_ps(t, acc);
        sum += f16_to_f32(x[b].d) * ((t[0] + t[1]) + (t[2] + t[3]));
    }
    return sum;
}

__attribute__((target("avx2,f16c")))
static float dot_q8_0_avx2(const block_q8_0 *x, const float *y, int k) {
    float sum = 0.f;
    for (int b = 0; b < k / 32; b++) {
        const float *yy = y + b * 32;
        __m256 acc = _mm256_setzero_ps();
        for (int j = 0; j < 32; j += 8) {
            __m256i q = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(x[b].qs + j)));
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_cvtepi32_ps(q), _mm256_loadu_ps(yy + j)));
        }
        __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
        r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
        sum += f16_hw(x[b].d) * _mm_cvtss_f32(r);
    }
    return sum;
}
#endif

#ifdef HY_NEON
static float dot_q8_0_neon(const block_q8_0 *x, const float *y, int k) {
    float sum = 0.f;
    for (int b = 0; b < k / 32; b++) {
        const float *yy = y + b * 32;
        float32x4_t acc = vdupq_n_f32(0);
        for (int j = 0; j < 32; j += 16) {
            int8x16_t q = vld1q_s8(x[b].qs + j);
            int16x8_t q0 = vmovl_s8(vget_low_s8(q)), q1 = vmovl_high_s8(q);
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q0))), vld1q_f32(yy + j));
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_high_s16(q0)), vld1q_f32(yy + j + 4));
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(q1))), vld1q_f32(yy + j + 8));
            acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_high_s16(q1)), vld1q_f32(yy + j + 12));
        }
        sum += f16_to_f32(x[b].d) * vaddvq_f32(acc);
    }
    return sum;
}
#endif

/* ------------------------------------------------------------------ 注意力: bf16 KV 的点积 / axpy */

static float dot_bf16_scalar(const float *q, const uint16_t *k, int n) {
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    int d = 0;
    for (; d + 4 <= n; d += 4) {
        a0 += q[d] * bf16_to_f32(k[d]); a1 += q[d + 1] * bf16_to_f32(k[d + 1]);
        a2 += q[d + 2] * bf16_to_f32(k[d + 2]); a3 += q[d + 3] * bf16_to_f32(k[d + 3]);
    }
    float s = (a0 + a1) + (a2 + a3);
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
}
static void axpy_bf16_scalar(float *o, const uint16_t *v, float p, int n) {
    for (int d = 0; d < n; d++) o[d] += p * bf16_to_f32(v[d]);
}

#ifdef HY_X86
__attribute__((target("avx2")))
static float dot_bf16_avx2(const float *q, const uint16_t *k, int n) {
    __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
    int d = 0;
    for (; d + 16 <= n; d += 16) {
        __m256 k0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(k + d))), 16));
        __m256 k1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(k + d + 8))), 16));
        a0 = _mm256_add_ps(a0, _mm256_mul_ps(_mm256_loadu_ps(q + d), k0));
        a1 = _mm256_add_ps(a1, _mm256_mul_ps(_mm256_loadu_ps(q + d + 8), k1));
    }
    a0 = _mm256_add_ps(a0, a1);
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(a0), _mm256_extractf128_ps(a0, 1));
    r = _mm_hadd_ps(r, r); r = _mm_hadd_ps(r, r);
    float s = _mm_cvtss_f32(r);
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
}
__attribute__((target("avx2")))
static void axpy_bf16_avx2(float *o, const uint16_t *v, float p, int n) {
    __m256 pv = _mm256_set1_ps(p);
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        __m256 x = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(v + d))), 16));
        _mm256_storeu_ps(o + d, _mm256_add_ps(_mm256_loadu_ps(o + d), _mm256_mul_ps(pv, x)));
    }
    for (; d < n; d++) o[d] += p * bf16_to_f32(v[d]);
}
/* SSE2 版本 (x86-64 基线) */
static float dot_bf16_sse2(const float *q, const uint16_t *k, int n) {
    __m128 a0 = _mm_setzero_ps(), a1 = _mm_setzero_ps();
    const __m128i z = _mm_setzero_si128();
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(k + d));
        __m128 k0 = _mm_castsi128_ps(_mm_unpacklo_epi16(z, raw)); /* bf16 = 高 16 位 */
        __m128 k1 = _mm_castsi128_ps(_mm_unpackhi_epi16(z, raw));
        a0 = _mm_add_ps(a0, _mm_mul_ps(_mm_loadu_ps(q + d), k0));
        a1 = _mm_add_ps(a1, _mm_mul_ps(_mm_loadu_ps(q + d + 4), k1));
    }
    a0 = _mm_add_ps(a0, a1);
    float t[4];
    _mm_storeu_ps(t, a0);
    float s = (t[0] + t[1]) + (t[2] + t[3]);
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
}
static void axpy_bf16_sse2(float *o, const uint16_t *v, float p, int n) {
    const __m128 pv = _mm_set1_ps(p);
    const __m128i z = _mm_setzero_si128();
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        __m128i raw = _mm_loadu_si128((const __m128i *)(v + d));
        __m128 x0 = _mm_castsi128_ps(_mm_unpacklo_epi16(z, raw)), x1 = _mm_castsi128_ps(_mm_unpackhi_epi16(z, raw));
        _mm_storeu_ps(o + d, _mm_add_ps(_mm_loadu_ps(o + d), _mm_mul_ps(pv, x0)));
        _mm_storeu_ps(o + d + 4, _mm_add_ps(_mm_loadu_ps(o + d + 4), _mm_mul_ps(pv, x1)));
    }
    for (; d < n; d++) o[d] += p * bf16_to_f32(v[d]);
}
#endif
#ifdef HY_NEON
static float dot_bf16_neon(const float *q, const uint16_t *k, int n) {
    float32x4_t a0 = vdupq_n_f32(0), a1 = vdupq_n_f32(0);
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        uint16x8_t raw = vld1q_u16(k + d);
        float32x4_t k0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
        float32x4_t k1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw), 16));
        a0 = vmlaq_f32(a0, vld1q_f32(q + d), k0);
        a1 = vmlaq_f32(a1, vld1q_f32(q + d + 4), k1);
    }
    float s = vaddvq_f32(vaddq_f32(a0, a1));
    for (; d < n; d++) s += q[d] * bf16_to_f32(k[d]);
    return s;
}
static void axpy_bf16_neon(float *o, const uint16_t *v, float p, int n) {
    int d = 0;
    for (; d + 8 <= n; d += 8) {
        uint16x8_t raw = vld1q_u16(v + d);
        float32x4_t x0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(raw), 16));
        float32x4_t x1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(raw), 16));
        vst1q_f32(o + d, vmlaq_n_f32(vld1q_f32(o + d), x0, p));
        vst1q_f32(o + d + 4, vmlaq_n_f32(vld1q_f32(o + d + 4), x1, p));
    }
    for (; d < n; d++) o[d] += p * bf16_to_f32(v[d]);
}
#endif

typedef float (*dot_bf16_fn)(const float *, const uint16_t *, int);
typedef void (*axpy_bf16_fn)(float *, const uint16_t *, float, int);
static dot_bf16_fn g_dot_bf16 = dot_bf16_scalar;
static axpy_bf16_fn g_axpy_bf16 = axpy_bf16_scalar;
float dot_bf16(const float *q, const uint16_t *k, int n) { return g_dot_bf16(q, k, n); }
void axpy_bf16(float *o, const uint16_t *v, float p, int n) { g_axpy_bf16(o, v, p, n); }
float dot_bf16_ref(const float *q, const uint16_t *k, int n) { return dot_bf16_scalar(q, k, n); }

static float q2_0c_dot_scalar(const block_q2_0c *x, const q8k_block *y, int nb512) {
    float sum = 0.f;
    for (int b = 0; b < nb512; b++) {
        float d = f16_to_f32(x[b].d), acc = 0.f;
        for (int h = 0; h < 2; h++) {
            int s = 0;
            for (int j = 0; j < 256; j++) {
                int widx = h * 256 + j;
                s += (2 * ((x[b].qs[widx >> 2] >> ((widx & 3) * 2)) & 3) - 3) * y[2 * b + h].qs[j];
            }
            acc += (float)s * y[2 * b + h].d;
        }
        sum += d * acc;
    }
    return sum;
}

/* ------------------------------------------------------------------ 统一入口 */

void matmul_rows(const weight_t *w, int r0, int r1, const float *xf, const q8k_block *xq, int nt, float *out) {
    if (w->pk) { g_panel_rows(w, r0, r1, xq, nt, out); return; }
    switch (w->type) {
    case GT_Q2_0C: {
        int nb = w->k / 512;
        for (int r = r0; r < r1; r++) {
            const block_q2_0c *row = (const block_q2_0c *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++) out[(size_t)t * w->n + r] = q2_0c_dot_scalar(row, xq + (size_t)t * nb * 2, nb);
        }
        return;
    }
    case GT_STQ1_0:
        g_stq_rows(w, r0, r1, xq, nt, out);
        return;
    case GT_Q4_K: {
        int nb = w->k / QK_K;
        for (int r = r0; r < r1; r++) {
            const block_q4_k *row = (const block_q4_k *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++) out[(size_t)t * w->n + r] = g_q4k_dot(row, xq + (size_t)t * nb, nb);
        }
        return;
    }
    case GT_Q6_K: {
        int nb = w->k / QK_K;
        for (int r = r0; r < r1; r++) {
            const block_q6_k *row = (const block_q6_k *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++)
                out[(size_t)t * w->n + r] = g_q6_dot(row, xq + (size_t)t * nb, nb);
        }
        return;
    }
    case GT_Q8_0:
        for (int r = r0; r < r1; r++) {
            const block_q8_0 *row = (const block_q8_0 *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++)
                out[(size_t)t * w->n + r] = g_q8_dot(row, xf + (size_t)t * w->k, w->k);
        }
        return;
    case GT_F16:
        for (int r = r0; r < r1; r++) {
            const uint16_t *row = (const uint16_t *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++) {
                float s = 0.f;
                const float *x = xf + (size_t)t * w->k;
                for (int j = 0; j < w->k; j++) s += f16_to_f32(row[j]) * x[j];
                out[(size_t)t * w->n + r] = s;
            }
        }
        return;
    case GT_F32:
        for (int r = r0; r < r1; r++) {
            const float *row = (const float *)(w->data + (size_t)r * w->row_bytes);
            for (int t = 0; t < nt; t++) {
                float s = 0.f;
                const float *x = xf + (size_t)t * w->k;
                for (int j = 0; j < w->k; j++) s += row[j] * x[j];
                out[(size_t)t * w->n + r] = s;
            }
        }
        return;
    }
}

void dequant_row(const weight_t *w, int row, float *dst) {
    const uint8_t *p = w->data + (size_t)row * w->row_bytes;
    switch (w->type) {
    case GT_F32: memcpy(dst, p, (size_t)w->k * 4); break;
    case GT_F16:
        for (int j = 0; j < w->k; j++) dst[j] = f16_to_f32(((const uint16_t *)p)[j]);
        break;
    case GT_Q8_0:
        for (int b = 0; b < w->k / 32; b++) {
            const block_q8_0 *x = (const block_q8_0 *)p + b;
            float d = f16_to_f32(x->d);
            for (int j = 0; j < 32; j++) dst[b * 32 + j] = d * (float)x->qs[j];
        }
        break;
    case GT_Q4_K: q4_k_dequant_row((const block_q4_k *)p, dst, w->k); break;
    case GT_Q6_K: q6_k_dequant_row((const block_q6_k *)p, dst, w->k); break;
    case GT_STQ1_0: stq1_0_dequant_row((const block_stq1_0 *)p, dst, w->k); break;
    case GT_Q2_0C:
        for (int b = 0; b < w->k / 512; b++) {
            const block_q2_0c *x = (const block_q2_0c *)p + b;
            float d = f16_to_f32(x->d);
            for (int j = 0; j < 512; j++) dst[b * 512 + j] = (float)(2 * ((x->qs[j >> 2] >> ((j & 3) * 2)) & 3) - 3) * d;
        }
        break;
    }
}

static void sign_lut_init(void) {
    for (int v = 0; v < 256; v++) {
        uint64_t bits = 0;
        for (int i = 0; i < 8; i++)
            if ((v >> i) & 1) bits |= 0xFFull << (i * 8);
        SIGN_LUT[v] = bits;
    }
}

/* ISA 等级: x86: 0 scalar < 1 ssse3 < 2 sse4.1 (J4125) < 3 avx2 < 4 avx512
 *          arm: 0 scalar < 1 neon < 2 neon+dotprod */
static int g_level = 0;
static bool g_cb_ok = true;

#ifdef HY_NEON
static int hy_arm_has_dotprod(void) {
#if defined(__APPLE__)
    return 1; /* Apple Silicon 均有 DOTPROD */
#elif defined(__linux__) || defined(__ANDROID__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}
#endif

static int isa_level_by_name(const char *name) {
    if (!strcmp(name, "scalar")) return 0;
    if (!strcmp(name, "ssse3")) return 1;
    if (!strcmp(name, "sse4.1") || !strcmp(name, "sse4.2") || !strcmp(name, "sse4")) return 2;
    if (!strcmp(name, "avx2")) return 3;
    if (!strncmp(name, "avx512", 6)) return 4; /* "avx512" / "avx512-bw" (强制不用 VNNI) / "avx512-vnni" */
#ifdef HY_NEON
    if (!strcmp(name, "neon+dotprod") || !strcmp(name, "dotprod")) return 2;
    if (!strcmp(name, "neon")) return 1;
#endif
    return -1;
}

/* 选择指令集实现: name == NULL 或 "auto": 自动选当前 CPU 能用的最优实现; 否则取 "不超过指定等级的最优实现"
 * (scalar / ssse3 / avx2 / avx512 / neon)。返回 true 表示恰好得到了所指定的等级。 */
bool kernels_set_isa(const char *name) {
    sign_lut_init();
    g_stq_rows = stq_rows_scalar; g_q6_dot = q6_k_dot_scalar; g_isa = "scalar"; g_level = 0;
    g_dot_bf16 = dot_bf16_scalar; g_axpy_bf16 = axpy_bf16_scalar; g_panel_rows = panel_rows_scalar;
    g_q4k_dot = q4_k_dot_scalar; g_q8_dot = dot_q8_0_f32;
    const bool want_auto = (name == NULL || !strcmp(name, "auto"));
    int max_level = 99;
    if (!want_auto) {
        max_level = isa_level_by_name(name);
        if (max_level < 0) return false;
    }
#ifdef HY_X86
    __builtin_cpu_init();
    if (max_level >= 1 && __builtin_cpu_supports("ssse3")) {
        g_stq_rows = stq_rows_ssse3; g_q6_dot = q6_k_dot_ssse3; g_isa = "ssse3"; g_level = 1;
        g_dot_bf16 = dot_bf16_sse2; g_axpy_bf16 = axpy_bf16_sse2; g_panel_rows = panel_rows_ssse3;
        g_q4k_dot = q4_k_dot_ssse3;
    }
    if (max_level >= 2 && __builtin_cpu_supports("sse4.1")) {
        g_stq_rows = stq_rows_sse41; g_q6_dot = q6_k_dot_sse41; g_isa = "sse4.1"; g_level = 2;
        g_dot_bf16 = dot_bf16_sse2; g_axpy_bf16 = axpy_bf16_sse2;
        g_q8_dot = dot_q8_0_sse41;
    }
    if (max_level >= 3 && __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c")) {
        g_stq_rows = stq_rows_avx2; g_q6_dot = q6_k_dot_avx2; g_isa = "avx2"; g_level = 3;
        g_dot_bf16 = dot_bf16_avx2; g_axpy_bf16 = axpy_bf16_avx2; g_panel_rows = panel_rows_avx2;
        g_q4k_dot = q4_k_dot_avx2; g_q8_dot = dot_q8_0_avx2;
    }
    /* AVX-512: 显式指定 (HYMT2_ISA=avx512) 时只要 CPU 支持就用; auto 时仅在有 VNNI 的较新核心上启用
     * (没有 VNNI 的 Skylake-X 类核心跑 512 位指令会降频, 收益不稳, 需要的话手动指定) */
    if (max_level >= 4 && g_cb_ok && __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") && __builtin_cpu_supports("f16c")) {
        bool vnni = __builtin_cpu_supports("avx512vnni") && !(name && !strcmp(name, "avx512-bw"));
        if (!want_auto || vnni) {
            g_stq_rows = vnni ? stq_rows_avx512_vnni : stq_rows_avx512;
            g_panel_rows = vnni ? panel_rows_avx512_vnni : panel_rows_avx512;
            g_isa = vnni ? "avx512-vnni" : "avx512";
            g_level = 4;
        }
    }
#endif
#ifdef HY_NEON
    if (max_level >= 1) {
        g_stq_rows = stq_rows_neon; g_q6_dot = q6_k_dot_neon;
        g_dot_bf16 = dot_bf16_neon; g_axpy_bf16 = axpy_bf16_neon; g_panel_rows = panel_rows_neon;
        g_q4k_dot = q4_k_dot_neon; g_q8_dot = dot_q8_0_neon;
        g_isa = "neon";
        g_level = 1;
    }
    /* 与官方 llama.cpp STQ 一致: 有 ASIMDDP 时用 vdot (面板/原生 STQ)。不碰 x86/J4125 路径。 */
    if (max_level >= 2 && hy_arm_has_dotprod()) {
        g_stq_rows = stq_rows_neon_dotprod;
        g_panel_rows = panel_rows_neon_dotprod;
        g_isa = "neon+dotprod";
        g_level = 2;
    }
#endif
    if (want_auto) return true;
    return g_level == max_level;
}

void kernels_init(void) {
    static bool done = false;
    if (done) return;
    done = true;
    for (int i = 0; i < 16; i++)
        if (STQ_CODEBOOK[16 + i] != (uint8_t)(0xAA - STQ_CODEBOOK[i])) g_cb_ok = false;
    kernels_set_isa(NULL);
    const char *e = getenv("HYMT2_ISA");
    if (e && *e && strcmp(e, "auto") != 0) {
        if (!kernels_set_isa(e))
            fprintf(stderr, "[hymt2] HYMT2_ISA=%s 在此 CPU/编译配置下不可用, 改用 %s\n", e, g_isa);
    }
}

const char *kernels_isa(void) { return g_isa; }

const char *kernels_cpu_features(void) {
    static char buf[256];
    buf[0] = 0;
#ifdef HY_X86
    __builtin_cpu_init();
#define FEAT(n) do { if (__builtin_cpu_supports(n)) { if (buf[0]) strcat(buf, " "); strcat(buf, n); } } while (0)
    FEAT("ssse3"); FEAT("sse4.1"); FEAT("sse4.2"); FEAT("avx"); FEAT("avx2"); FEAT("f16c"); FEAT("fma");
    FEAT("avx512f"); FEAT("avx512bw"); FEAT("avx512vl"); FEAT("avx512dq"); FEAT("avx512vnni");
#undef FEAT
#elif defined(HY_NEON)
    strcpy(buf, "neon");
    if (hy_arm_has_dotprod()) strcat(buf, " dotprod");
#endif
    return buf[0] ? buf : "(unknown)";
}

int type_from_ggml(unsigned raw) {
    switch (raw) {
    case 0: return GT_F32;
    case 1: return GT_F16;
    case 8: return GT_Q8_0;
    case 12: return GT_Q4_K;
    case 14: return GT_Q6_K;
    case 40: return GT_Q2_0C;
    case 43: return GT_STQ1_0;
    }
    return GT_UNSUPPORTED;
}
