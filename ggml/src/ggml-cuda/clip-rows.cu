#include "clip-rows.cuh"

// one block per row; the sum of squares is accumulated exactly like reduce_row_f32 (reduce_rows.cuh) so that the
// result is bit-identical to SQR + SUM_ROWS for the same block size
template <bool has_w>
static __global__ void clip_rows_f32(const float * __restrict__ x, const float * __restrict__ w, const float * __restrict__ limit,
        float * __restrict__ dst, const int ncols, const int w_rows, const float limit_value, const float bias,
        const float lo, const float hi) {
    const int64_t row = blockIdx.x;
    const int     col = threadIdx.x;

    const float * xr = x + row*ncols;
    const float * wr = has_w ? w + int64_t(row % w_rows)*ncols : nullptr;

    auto value = [&](int i) {
        return has_w ? __fmul_rn(xr[i], wr[i]) : xr[i];
    };

    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    ggml_cuda_pdl_sync();
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float c = value(i);
                temp[j] = __fmul_rn(c, c);
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] = __fadd_rn(sum_temp[j], temp[j]);
        }
    }
    float sum = 0.0f;
    for (int j = 0; j < num_unroll; ++j) {
        sum = __fadd_rn(sum, sum_temp[j]);
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);
    ggml_cuda_pdl_lc();

    // SQRT, SCALE (scale 1: x*1 + bias rounds once either way), DIV, CLAMP as in their own kernels
    const float norm   = sqrtf(sum) + bias;
    const float factor = fminf(fmaxf((limit ? limit[row] : limit_value) / norm, lo), hi);

    for (int i = col; i < ncols; i += blockDim.x) {
        dst[row*ncols + i] = __fmul_rn(value(i), factor);
    }
}

void ggml_cuda_op_clip_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        const ggml_tensor * limit, float limit_value, float bias, float lo, float hi, ggml_tensor * dst) {
    const int     ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda for this shape
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int block_size = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());

    const float * w_d     = w ? (const float *) w->data : nullptr;
    const float * limit_d = limit ? (const float *) limit->data : nullptr;
    const int     w_rows  = w ? (int) ggml_nrows(w) : 1;

    if (w) {
        ggml_cuda_kernel_launch(clip_rows_f32<true>, launch_params, (const float *) x->data, w_d, limit_d, (float *) dst->data,
            ncols, w_rows, limit_value, bias, lo, hi);
    } else {
        ggml_cuda_kernel_launch(clip_rows_f32<false>, launch_params, (const float *) x->data, w_d, limit_d, (float *) dst->data,
            ncols, w_rows, limit_value, bias, lo, hi);
    }
}

// one block per row; the same loop, reduction and rounding as l2_norm_f32 (norm.cu) and scale_f32 (scale.cu) for this
// block size, so the result matches L2_NORM then SCALE run as separate ops
template <int block_size, bool has_w>
static __global__ void l2_clip_rows_f32(const float * __restrict__ x, const float * __restrict__ w,
        float * __restrict__ dst, const int ncols, const int w_rows, const float eps, const float scale, const float bias) {
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;

    const float * xr = x + row*ncols;
    const float * wr = has_w ? w + int64_t(row % w_rows)*ncols : nullptr;

    auto value = [&](int i) {
        return has_w ? __fmul_rn(xr[i], wr[i]) : xr[i];
    };

    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();
    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += block_size) {
        const float xi = value(col);
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, block_size>(tmp, s_sum);
    ggml_cuda_pdl_lc();

    const float rs = rsqrtf(fmaxf(tmp, eps * eps));
    for (int col = tid; col < ncols; col += block_size) {
        const float y = rs * value(col);
        dst[row*ncols + col] = scale * y + bias;
    }
}

void ggml_cuda_op_l2_clip_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        float eps, float scale, float bias, ggml_tensor * dst) {
    const int     ncols  = x->ne[0];
    const int64_t nrows  = ggml_nrows(x);
    const float * w_d    = w ? (const float *) w->data : nullptr;
    const int     w_rows = w ? (int) ggml_nrows(w) : 1;
    const float * x_d    = (const float *) x->data;
    float       * dst_d  = (float *) dst->data;

    // the block size of l2_norm_f32_cuda for this row length
    if (ncols < 1024) {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(WARP_SIZE, 1, 1), 0, ctx.stream());
        if (w) {
            ggml_cuda_kernel_launch(l2_clip_rows_f32<WARP_SIZE, true>,  lp, x_d, w_d, dst_d, ncols, w_rows, eps, scale, bias);
        } else {
            ggml_cuda_kernel_launch(l2_clip_rows_f32<WARP_SIZE, false>, lp, x_d, w_d, dst_d, ncols, w_rows, eps, scale, bias);
        }
    } else {
        const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(1024, 1, 1), 0, ctx.stream());
        if (w) {
            ggml_cuda_kernel_launch(l2_clip_rows_f32<1024, true>,  lp, x_d, w_d, dst_d, ncols, w_rows, eps, scale, bias);
        } else {
            ggml_cuda_kernel_launch(l2_clip_rows_f32<1024, false>, lp, x_d, w_d, dst_d, ncols, w_rows, eps, scale, bias);
        }
    }
}
