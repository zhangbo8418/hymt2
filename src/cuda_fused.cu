/* CUDA 整网常驻算子: rmsnorm / add / silu / rope / quantize / attn / store。
 * 由 fused_forward 串在同一 stream，整网末尾才同步。需 CUDA Toolkit ≥10.2。 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

struct UniPC { uint32_t n, nt, stride, pad; float eps; };
struct RopePC { uint32_t heads, hdim, nhalf, pos0, nt, pad0; };
struct Q8PC { uint32_t k, nt, nb, pad; };
struct AttnPC { uint32_t heads, kv_heads, hdim, qdim, kvdim, nt, pos0, ctx; };
struct StorePC { uint32_t kvdim, pos0, nt, pad; };
struct RmsRowsPC { uint32_t n, nrows, stride, pad; float eps; };

__device__ inline float bf16_f(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    return __int_as_float((int)u);
}
__device__ inline uint16_t f_bf16(float x) {
    uint32_t bits = (uint32_t)__float_as_int(x);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

extern "C" __global__ void hy_cu_rmsnorm(const float *x, const float *w, float *y, UniPC pc) {
    uint32_t t = blockIdx.y;
    if (t >= pc.nt) return;
    uint32_t tid = threadIdx.x;
    const float *xr = x + t * pc.stride;
    float *yr = y + t * pc.stride;
    __shared__ float red[256];
    float loc = 0.f;
    for (uint32_t i = tid; i < pc.n; i += 256u) loc += xr[i] * xr[i];
    red[tid] = loc;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / (float)pc.n + pc.eps);
    for (uint32_t i = tid; i < pc.n; i += 256u) yr[i] = xr[i] * scale * w[i];
}

extern "C" __global__ void hy_cu_rmsnorm_rows(float *x, const float *w, RmsRowsPC pc) {
    uint32_t rid = blockIdx.x;
    if (rid >= pc.nrows) return;
    uint32_t tid = threadIdx.x;
    float *xr = x + rid * pc.stride;
    __shared__ float red[256];
    float loc = 0.f;
    for (uint32_t i = tid; i < pc.n; i += 256u) loc += xr[i] * xr[i];
    red[tid] = loc;
    __syncthreads();
    for (uint32_t s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    float scale = rsqrtf(red[0] / (float)pc.n + pc.eps);
    for (uint32_t i = tid; i < pc.n; i += 256u) xr[i] = xr[i] * scale * w[i];
}

extern "C" __global__ void hy_cu_add(float *h, const float *x, UniPC pc) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < pc.n) h[idx] += x[idx];
}

extern "C" __global__ void hy_cu_silu_mul(float *g, const float *u, UniPC pc) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pc.n) return;
    float v = g[idx];
    g[idx] = (v / (1.f + expf(-v))) * u[idx];
}

extern "C" __global__ void hy_cu_rope_neox(float *x, const float *cosv, const float *sinv, RopePC pc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t h = blockIdx.y;
    uint32_t t = blockIdx.z;
    if (t >= pc.nt || h >= pc.heads || i >= pc.nhalf) return;
    float *row = x + t * (pc.heads * pc.hdim) + h * pc.hdim;
    uint32_t pos = pc.pos0 + t;
    float c = cosv[pos * pc.nhalf + i], s = sinv[pos * pc.nhalf + i];
    float x0 = row[i], x1 = row[i + pc.nhalf];
    row[i] = x0 * c - x1 * s;
    row[i + pc.nhalf] = x0 * s + x1 * c;
}

extern "C" __global__ void hy_cu_quantize_q8k(const float *x, uint8_t *y, Q8PC pc) {
    uint32_t t = blockIdx.y, b = blockIdx.x;
    if (t >= pc.nt || b >= pc.nb) return;
    uint32_t tid = threadIdx.x;
    const float *src = x + t * pc.k + b * 256u;
    uint8_t *blk = y + (size_t)(t * pc.nb + b) * 576u;
    __shared__ float red_amax[64];
    __shared__ float red_max[64];
    float amax = 0.f, mx = 0.f;
    for (uint32_t j = tid; j < 256u; j += 64u) {
        float v = src[j], ax = fabsf(v);
        if (ax > amax) { amax = ax; mx = v; }
    }
    red_amax[tid] = amax; red_max[tid] = mx;
    __syncthreads();
    for (uint32_t s = 32u; s > 0u; s >>= 1u) {
        if (tid < s && red_amax[tid + s] > red_amax[tid]) {
            red_amax[tid] = red_amax[tid + s]; red_max[tid] = red_max[tid + s];
        }
        __syncthreads();
    }
    amax = red_amax[0]; mx = red_max[0];
    if (tid == 0u) {
        for (uint32_t j = 0u; j < 576u; j++) blk[j] = 0;
    }
    __syncthreads();
    if (amax == 0.f) return;
    float iscale = -127.f / mx;
    if (tid == 0u) ((float *)(blk + 256))[0] = 1.f / iscale;
    __syncthreads();
    char *qs = (char *)(blk + 320);
    for (uint32_t j = tid; j < 256u; j += 64u) {
        int v = (int)rintf(iscale * src[j]);
        if (v > 127) v = 127; if (v < -127) v = -127;
        qs[j] = (char)v;
    }
    __syncthreads();
    if (tid == 0u) {
        int total = 0;
        for (uint32_t g = 0u; g < 16u; g++) {
            int s = 0;
            for (uint32_t j = 0u; j < 16u; j++) s += (int)qs[g * 16u + j];
            ((short *)(blk + 264))[g] = (short)s;
            total += s;
        }
        ((int *)(blk + 260))[0] = total;
        char *qp = (char *)blk;
        uint32_t cordj[4] = {0u, 2u, 1u, 3u};
        for (uint32_t l = 0u; l < 4u; l++)
            for (uint32_t j = 0u; j < 4u; j++)
                for (uint32_t i = 0u; i < 16u; i++)
                    qp[l * 64u + j * 16u + i] = qs[cordj[j] * 64u + l * 16u + i];
    }
}

extern "C" __global__ void hy_cu_store_bf16(const float *k, const float *v, uint16_t *kc, uint16_t *vc, StorePC pc) {
    uint32_t d = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t t = blockIdx.y;
    if (t >= pc.nt || d >= pc.kvdim) return;
    uint32_t dst = (pc.pos0 + t) * pc.kvdim + d;
    kc[dst] = f_bf16(k[t * pc.kvdim + d]);
    vc[dst] = f_bf16(v[t * pc.kvdim + d]);
}

extern "C" __global__ void hy_cu_attn(const float *q, const uint16_t *kc, const uint16_t *vc, float *ao, AttnPC pc) {
    uint32_t h = blockIdx.x, t = blockIdx.y;
    if (h >= pc.heads || t >= pc.nt) return;
    uint32_t lid = threadIdx.x;
    uint32_t group = pc.heads / (pc.kv_heads ? pc.kv_heads : 1u);
    uint32_t kvh = h / group;
    uint32_t n = pc.pos0 + t + 1u;
    float scale = rsqrtf((float)pc.hdim);
    const float *qh = q + t * pc.qdim + h * pc.hdim;
    __shared__ float q_sh[256];
    __shared__ float red[64];
    __shared__ float scores[2048];
    for (uint32_t d = lid; d < pc.hdim; d += 64u) q_sh[d] = qh[d];
    __syncthreads();
    uint32_t ncap = n < 2048u ? n : 2048u;
    for (uint32_t j = lid; j < ncap; j += 64u) {
        const uint16_t *Krow = kc + j * pc.kvdim + kvh * pc.hdim;
        float s = 0.f;
        for (uint32_t d = 0u; d < pc.hdim; d++) s += q_sh[d] * bf16_f(Krow[d]);
        scores[j] = s * scale;
    }
    __syncthreads();
    float mx = -1e30f;
    for (uint32_t j = lid; j < ncap; j += 64u) if (scores[j] > mx) mx = scores[j];
    red[lid] = mx;
    __syncthreads();
    for (uint32_t s = 32u; s > 0u; s >>= 1u) {
        if (lid < s) red[lid] = fmaxf(red[lid], red[lid + s]);
        __syncthreads();
    }
    mx = red[0];
    float sum = 0.f;
    for (uint32_t j = lid; j < ncap; j += 64u) sum += expf(scores[j] - mx);
    red[lid] = sum;
    __syncthreads();
    for (uint32_t s = 32u; s > 0u; s >>= 1u) {
        if (lid < s) red[lid] += red[lid + s];
        __syncthreads();
    }
    float inv = 1.f / red[0];
    float acc0 = 0.f, acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    uint32_t d0 = lid, d1 = lid + 64u, d2 = lid + 128u, d3 = lid + 192u;
    bool h1 = d1 < pc.hdim, h2 = d2 < pc.hdim, h3 = d3 < pc.hdim;
    for (uint32_t j = 0u; j < ncap; j++) {
        float w = expf(scores[j] - mx) * inv;
        const uint16_t *Vrow = vc + j * pc.kvdim + kvh * pc.hdim;
        acc0 += bf16_f(Vrow[d0]) * w;
        if (h1) acc1 += bf16_f(Vrow[d1]) * w;
        if (h2) acc2 += bf16_f(Vrow[d2]) * w;
        if (h3) acc3 += bf16_f(Vrow[d3]) * w;
    }
    float *o = ao + t * pc.qdim + h * pc.hdim;
    o[d0] = acc0;
    if (h1) o[d1] = acc1;
    if (h2) o[d2] = acc2;
    if (h3) o[d3] = acc3;
}

/* ---- host launchers (不 sync) ---- */
extern "C" int hy_cuda_launch_rmsnorm(const float *x, const float *w, float *y,
                                      uint32_t n, uint32_t nt, uint32_t stride, float eps) {
    UniPC pc = { n, nt, stride, 0, eps };
    dim3 grid(1, nt), block(256);
    hy_cu_rmsnorm<<<grid, block>>>(x, w, y, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_rmsnorm_rows(float *x, const float *w,
                                           uint32_t n, uint32_t nrows, uint32_t stride, float eps) {
    RmsRowsPC pc = { n, nrows, stride, 0, eps };
    hy_cu_rmsnorm_rows<<<nrows, 256>>>(x, w, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_add(float *h, const float *x, uint32_t n) {
    UniPC pc = { n, 1, 0, 0, 0.f };
    uint32_t blocks = (n + 255u) / 256u;
    hy_cu_add<<<blocks, 256>>>(h, x, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_silu(float *g, const float *u, uint32_t n) {
    UniPC pc = { n, 1, 0, 0, 0.f };
    uint32_t blocks = (n + 255u) / 256u;
    hy_cu_silu_mul<<<blocks, 256>>>(g, u, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_rope(float *x, const float *cosv, const float *sinv,
                                   uint32_t heads, uint32_t hdim, uint32_t nhalf,
                                   uint32_t pos0, uint32_t nt) {
    RopePC pc = { heads, hdim, nhalf, pos0, nt, 0 };
    dim3 grid(nhalf, heads, nt), block(1);
    hy_cu_rope_neox<<<grid, block>>>(x, cosv, sinv, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_quantize(const float *x, void *y, uint32_t k, uint32_t nt, uint32_t nb) {
    Q8PC pc = { k, nt, nb, 0 };
    dim3 grid(nb, nt), block(64);
    hy_cu_quantize_q8k<<<grid, block>>>(x, (uint8_t *)y, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_store_bf16(const float *k, const float *v, uint16_t *kc, uint16_t *vc,
                                         uint32_t kvdim, uint32_t pos0, uint32_t nt) {
    StorePC pc = { kvdim, pos0, nt, 0 };
    dim3 grid((kvdim + 63u) / 64u, nt), block(64);
    hy_cu_store_bf16<<<grid, block>>>(k, v, kc, vc, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int hy_cuda_launch_attn(const float *q, const uint16_t *kc, const uint16_t *vc, float *ao,
                                   uint32_t heads, uint32_t kv_heads, uint32_t hdim, uint32_t qdim,
                                   uint32_t kvdim, uint32_t nt, uint32_t pos0, uint32_t ctx) {
    AttnPC pc = { heads, kv_heads, hdim, qdim, kvdim, nt, pos0, ctx };
    dim3 grid(heads, nt), block(64);
    hy_cu_attn<<<grid, block>>>(q, kc, vc, ao, pc);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
