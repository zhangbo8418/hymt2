/* CUDA quant matmul kernels — 与 vulkan shaders 同布局。由 nvcc 编译 (make CUDA=1)。 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

struct PC { uint32_t rows, nb, nt, n_total, row0, shift, off, pad0; };

__device__ inline uint32_t hy_wbyte(const uint32_t *wq, uint32_t base, uint32_t i) {
    return (wq[base + (i >> 2)] >> (8u * (i & 3u))) & 255u;
}
__device__ inline int hy_abyte(const uint32_t *act, uint32_t base, uint32_t i) {
    int v = (int)act[base + (i >> 2)];
    uint32_t s = 8u * (i & 3u);
    return (v << (24 - s)) >> 24;
}

extern "C" __global__ void hy_panel(const uint32_t *pk, const float *ps, const uint32_t *act, float *outv, PC pc) {
    __shared__ float red[64];
    uint32_t tid = threadIdx.x;
    uint32_t lr = tid >> 3, lane = tid & 7u;
    uint32_t r = blockIdx.x * 8u + lr;
    uint32_t t = blockIdx.y;
    float acc = 0.f;
    if (r < pc.rows) {
        for (uint32_t b = lane; b < pc.nb; b += 8u) {
            uint32_t wbase = (r * pc.nb + b) * 16u;
            uint32_t abase = (t * pc.nb + b) * 144u;
            int enc = 0;
            for (uint32_t j = 0; j < 4; j++) {
                for (uint32_t wi = 0; wi < 4; wi++) {
                    uint32_t wv = pk[wbase + j * 4u + wi];
                    for (uint32_t l = 0; l < 4; l++) {
                        int aw = (int)act[abase + l * 16u + j * 4u + wi];
                        uint32_t wl = wv >> (2u * l);
                        enc += (int)(wl & 3u) * ((aw << 24) >> 24);
                        enc += (int)((wl >> 8) & 3u) * ((aw << 16) >> 24);
                        enc += (int)((wl >> 16) & 3u) * ((aw << 8) >> 24);
                        enc += (int)((wl >> 24) & 3u) * (aw >> 24);
                    }
                }
            }
            float d = __int_as_float((int)act[abase + 64u]);
            int sm = (int)act[abase + 65u];
            acc += ps[r * pc.nb + b] * d * (float)((enc << (int)pc.shift) - (int)pc.off * sm);
        }
    }
    red[tid] = acc;
    __syncthreads();
    if (lane == 0 && r < pc.rows) {
        float s = 0.f;
        for (int i = 0; i < 8; i++) s += red[lr * 8 + i];
        outv[t * pc.n_total + pc.row0 + r] = s;
    }
}

extern "C" __global__ void hy_q4k(const uint32_t *wq, const float * /*unused*/, const uint32_t *act, float *outv, PC pc) {
    __shared__ float red[64];
    uint32_t tid = threadIdx.x;
    uint32_t lr = tid >> 3, lane = tid & 7u;
    uint32_t r = blockIdx.x * 8u + lr;
    uint32_t t = blockIdx.y;
    float acc = 0.f;
    if (r < pc.rows) {
        for (uint32_t b = lane; b < pc.nb; b += 8u) {
            uint32_t wb = (r * pc.nb + b) * 36u;
            uint32_t ab = (t * pc.nb + b) * 144u;
            uint32_t dpack = wq[wb];
            float d0 = __half2float(((const __half *)&dpack)[0]);
            float d1 = __half2float(((const __half *)&dpack)[1]);
            uint32_t sc[8], mn[8];
            for (uint32_t j = 0; j < 8; j++) {
                if (j < 4) { sc[j] = hy_wbyte(wq, wb + 1, j) & 63u; mn[j] = hy_wbyte(wq, wb + 1, j + 4) & 63u; }
                else {
                    sc[j] = (hy_wbyte(wq, wb + 1, j + 4) & 15u) | ((hy_wbyte(wq, wb + 1, j - 4) >> 6) << 4);
                    mn[j] = (hy_wbyte(wq, wb + 1, j + 4) >> 4) | ((hy_wbyte(wq, wb + 1, j) >> 6) << 4);
                }
            }
            int sumi = 0;
            for (uint32_t g = 0; g < 4; g++) {
                int s0 = 0, s1 = 0;
                for (uint32_t l = 0; l < 32; l++) {
                    uint32_t q = hy_wbyte(wq, wb + 4, 32 * g + l);
                    s0 += (int)(q & 15u) * hy_abyte(act, ab + 80, 64 * g + l);
                    s1 += (int)(q >> 4) * hy_abyte(act, ab + 80, 64 * g + 32 + l);
                }
                sumi += (int)sc[2 * g] * s0 + (int)sc[2 * g + 1] * s1;
            }
            int summ = 0;
            for (uint32_t j = 0; j < 8; j++) {
                uint32_t w = act[ab + 66 + j];
                int16_t lo = (int16_t)(w & 0xffffu);
                int16_t hi = (int16_t)((w >> 16) & 0xffffu);
                summ += (int)mn[j] * ((int)lo + (int)hi);
            }
            float yd = __int_as_float((int)act[ab + 64]);
            acc += yd * (d0 * (float)sumi - d1 * (float)summ);
        }
    }
    red[tid] = acc;
    __syncthreads();
    if (lane == 0 && r < pc.rows) {
        float s = 0.f;
        for (int i = 0; i < 8; i++) s += red[lr * 8 + i];
        outv[t * pc.n_total + pc.row0 + r] = s;
    }
}

extern "C" __global__ void hy_q6k(const uint32_t *wq, const float * /*unused*/, const uint32_t *act, float *outv, PC pc) {
    __shared__ float red[64];
    uint32_t tid = threadIdx.x;
    uint32_t lr = tid >> 3, lane = tid & 7u;
    uint32_t r = blockIdx.x * 8u + lr;
    uint32_t t = blockIdx.y;
    float acc = 0.f;
    if (r < pc.rows) {
        for (uint32_t b = lane; b < pc.nb; b += 8u) {
            uint32_t wb = (r * pc.nb + b) * 53u;
            uint32_t ab = (t * pc.nb + b) * 144u;
            int isum = 0;
            for (uint32_t n = 0; n < 2; n++) {
                for (uint32_t l = 0; l < 32; l++) {
                    uint32_t qlo = hy_wbyte(wq, wb, 64 * n + l), qhi = hy_wbyte(wq, wb, 64 * n + l + 32);
                    uint32_t qh = hy_wbyte(wq, wb + 32, 32 * n + l);
                    int q1 = (int)((qlo & 15u) | (((qh >> 0) & 3u) << 4)) - 32;
                    int q2 = (int)((qhi & 15u) | (((qh >> 2) & 3u) << 4)) - 32;
                    int q3 = (int)((qlo >> 4) | (((qh >> 4) & 3u) << 4)) - 32;
                    int q4 = (int)((qhi >> 4) | (((qh >> 6) & 3u) << 4)) - 32;
                    uint32_t e = 128 * n + l;
                    isum += (int8_t)hy_wbyte(wq, wb + 48, e / 16) * q1 * hy_abyte(act, ab + 80, e)
                          + (int8_t)hy_wbyte(wq, wb + 48, (e + 32) / 16) * q2 * hy_abyte(act, ab + 80, e + 32)
                          + (int8_t)hy_wbyte(wq, wb + 48, (e + 64) / 16) * q3 * hy_abyte(act, ab + 80, e + 64)
                          + (int8_t)hy_wbyte(wq, wb + 48, (e + 96) / 16) * q4 * hy_abyte(act, ab + 80, e + 96);
                }
            }
            float d = __half2float(((const __half *)&wq[wb + 52])[0]);
            float yd = __int_as_float((int)act[ab + 64]);
            acc += d * yd * (float)isum;
        }
    }
    red[tid] = acc;
    __syncthreads();
    if (lane == 0 && r < pc.rows) {
        float s = 0.f;
        for (int i = 0; i < 8; i++) s += red[lr * 8 + i];
        outv[t * pc.n_total + pc.row0 + r] = s;
    }
}

/* Host launchers (linked when CUDA=1) */
extern "C" int hy_cuda_launch_panel(const void *pk, const void *ps, const void *act, void *out,
                                    uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total,
                                    uint32_t row0, uint32_t shift, uint32_t off) {
    PC pc = { rows, nb, nt, n_total, row0, shift, off, 0 };
    dim3 block(64), grid((rows + 7) / 8, nt);
    hy_panel<<<grid, block>>>((const uint32_t *)pk, (const float *)ps, (const uint32_t *)act, (float *)out, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1; /* 同步由 cu_matmul 合并一次 */
}

extern "C" int hy_cuda_launch_q4k(const void *wq, const void *act, void *out,
                                  uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total, uint32_t row0) {
    PC pc = { rows, nb, nt, n_total, row0, 0, 0, 0 };
    dim3 block(64), grid((rows + 7) / 8, nt);
    hy_q4k<<<grid, block>>>((const uint32_t *)wq, NULL, (const uint32_t *)act, (float *)out, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_q6k(const void *wq, const void *act, void *out,
                                  uint32_t rows, uint32_t nb, uint32_t nt, uint32_t n_total, uint32_t row0) {
    PC pc = { rows, nb, nt, n_total, row0, 0, 0, 0 };
    dim3 block(64), grid((rows + 7) / 8, nt);
    hy_q6k<<<grid, block>>>((const uint32_t *)wq, NULL, (const uint32_t *)act, (float *)out, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
