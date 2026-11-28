#include "mmvq-f32.cuh"

#include <cstdlib>
#include <cstring>

// the threads of a block split one row like MMVQ: thread t takes 8 of the 32 quants (part t % 4) of the blocks
// t / 4, t / 4 + nthreads / 4, ...; at most MMVQ_F32_MAX_ITERS such steps per thread
#define MMVQ_F32_NWARPS    4
#define MMVQ_F32_MAX_ITERS 4
#define MMVQ_F32_MAX_NCOLS (MMVQ_F32_MAX_ITERS*MMVQ_F32_NWARPS*WARP_SIZE/4*QK8_0)

// int8 -> float without I2F: byte k of w ^ 0x80808080 below 0x4B000000 is the float 2^23 + 128 + q, exact for |q| <= 128
static __device__ __forceinline__ float mmvq_f32_i8(const uint32_t w, const int k) {
    return __uint_as_float(__byte_perm(w ^ 0x80808080u, 0x4B000000u, 0x7440 + k)) - 8388736.0f;
}

// prefetch: src0 is a weight (no earlier kernel writes it), so its loads are issued before the PDL wait on src1
template <int rows_per_block, bool prefetch>
__launch_bounds__(MMVQ_F32_NWARPS*WARP_SIZE, 1)
static __global__ void mul_mat_vec_q8_0_f32(
        const char * vx, const float * y, float * dst, const int nblocks, const int64_t stride_row, const int nrows) {
    constexpr int nthreads = MMVQ_F32_NWARPS*WARP_SIZE;
    const int tid  = threadIdx.x;
    const int part = tid % 4;
    const int kb0  = tid / 4;
    const int row0 = blockIdx.x*rows_per_block;

    uint2 q[rows_per_block][MMVQ_F32_MAX_ITERS];
    float d[rows_per_block][MMVQ_F32_MAX_ITERS];

#define MMVQ_F32_LOAD() \
    _Pragma("unroll") \
    for (int r = 0; r < rows_per_block; ++r) { \
        const block_q8_0 * x = (const block_q8_0 *) (vx + (int64_t) min(row0 + r, nrows - 1)*stride_row); \
        _Pragma("unroll") \
        for (int it = 0; it < MMVQ_F32_MAX_ITERS; ++it) { \
            const int kb = kb0 + it*(nthreads/4); \
            if (kb < nblocks) { \
                const uint16_t * q16 = (const uint16_t *) (x[kb].qs + 8*part); \
                q[r][it].x = q16[0] | ((uint32_t) q16[1] << 16); \
                q[r][it].y = q16[2] | ((uint32_t) q16[3] << 16); \
                d[r][it]   = __half2float(x[kb].d); \
            } \
        } \
    }

    if constexpr (prefetch) {
        MMVQ_F32_LOAD();
    }
    ggml_cuda_pdl_sync();
    if constexpr (!prefetch) {
        MMVQ_F32_LOAD();
    }
#undef MMVQ_F32_LOAD

    float sum[rows_per_block] = {0.0f};
#pragma unroll
    for (int it = 0; it < MMVQ_F32_MAX_ITERS; ++it) {
        const int kb = kb0 + it*(nthreads/4);
        if (kb < nblocks) {
            const float4 * y4 = (const float4 *) (y + kb*QK8_0 + 8*part);
            const float4 a = y4[0];
            const float4 b = y4[1];
#pragma unroll
            for (int r = 0; r < rows_per_block; ++r) {
                float p = mmvq_f32_i8(q[r][it].x, 0)*a.x;
                p = fmaf(mmvq_f32_i8(q[r][it].x, 1), a.y, p);
                p = fmaf(mmvq_f32_i8(q[r][it].x, 2), a.z, p);
                p = fmaf(mmvq_f32_i8(q[r][it].x, 3), a.w, p);
                p = fmaf(mmvq_f32_i8(q[r][it].y, 0), b.x, p);
                p = fmaf(mmvq_f32_i8(q[r][it].y, 1), b.y, p);
                p = fmaf(mmvq_f32_i8(q[r][it].y, 2), b.z, p);
                p = fmaf(mmvq_f32_i8(q[r][it].y, 3), b.w, p);
                sum[r] = fmaf(d[r][it], p, sum[r]);
            }
        }
    }
    ggml_cuda_pdl_lc();

    __shared__ float partial[rows_per_block][MMVQ_F32_NWARPS];
    const int warp = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
        sum[r] = warp_reduce_sum(sum[r]);
        if (lane == 0) {
            partial[r][warp] = sum[r];
        }
    }
    __syncthreads();
    if (tid < rows_per_block && row0 + tid < nrows) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < MMVQ_F32_NWARPS; ++w) {
            v += partial[tid][w];
        }
        dst[row0 + tid] = v;
    }
}

bool ggml_cuda_should_use_mmvq_f32(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc) {
    static const bool disabled = [] {
        const char * v = getenv("GGML_CUDA_DISABLE_MMVQ_F32_ACT");
        return v != nullptr && strcmp(v, "0") != 0;
    }();
    if (disabled || !GGML_CUDA_CC_IS_NVIDIA(cc) || cc < 1000) {
        return false;
    }
    return src0->type == GGML_TYPE_Q8_0 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 && src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src0->ne[0] % QK8_0 == 0 && src0->ne[0] <= MMVQ_F32_MAX_NCOLS && src0->ne[0] == src1->ne[0] &&
        src0->nb[0] == ggml_type_size(GGML_TYPE_Q8_0) && src0->nb[1] % 2 == 0 && (uintptr_t) src0->data % 2 == 0 &&
        ggml_is_contiguous(src1) && (uintptr_t) src1->data % 16 == 0 && ggml_is_contiguous(dst);
}

void ggml_cuda_mul_mat_vec_q_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int nrows   = src0->ne[1];
    const int nblocks = src0->ne[0] / QK8_0;
    const int nsm     = ggml_cuda_info().devices[ctx.device].nsm;
    const bool prefetch = src0->buffer && ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;

    const char  * x = (const char *) src0->data;
    const float * y = (const float *) src1->data;
    float       * d = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

#define MMVQ_F32_LAUNCH(RPB) do { \
        const ggml_cuda_kernel_launch_params lp(dim3((nrows + RPB - 1)/RPB, 1, 1), dim3(MMVQ_F32_NWARPS*WARP_SIZE, 1, 1), 0, stream); \
        if (prefetch) { ggml_cuda_kernel_launch(mul_mat_vec_q8_0_f32<RPB, true>,  lp, x, y, d, nblocks, (int64_t) src0->nb[1], nrows); } \
        else          { ggml_cuda_kernel_launch(mul_mat_vec_q8_0_f32<RPB, false>, lp, x, y, d, nblocks, (int64_t) src0->nb[1], nrows); } \
    } while (0)
    // two rows per block share the activation loads once that still leaves several blocks per SM
    if (nrows >= 2*4*nsm) {
        MMVQ_F32_LAUNCH(2);
    } else {
        MMVQ_F32_LAUNCH(1);
    }
#undef MMVQ_F32_LAUNCH
}
