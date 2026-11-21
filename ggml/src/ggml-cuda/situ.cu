#include "situ.cuh"

static __global__ void situ_f32(const float * __restrict__ g, const float * __restrict__ u, float * __restrict__ dst,
        const int64_t k, const int64_t n, const int64_t s, const float sa0, const float ba0, const float sa1, const float ba1,
        const float sb0, const float bb0, const float sb1, const float bb1) {
    ggml_cuda_pdl_lc();
    const int64_t i = int64_t(blockDim.x)*blockIdx.x + threadIdx.x;
    if (i >= k) {
        return;
    }
    const int64_t j = (i / n)*s + (i % n);

    ggml_cuda_pdl_sync();
    const float x = g[j];
    const float y = u[j];

    // SCALE is an FMA (scale*x + bias), TANH tanhf, SIGMOID 1/(1 + exp(-x)), as in their own kernels
    const float a = __fmaf_rn(sa1, tanhf(__fmaf_rn(sa0, x, ba0)), ba1);
    const float b = __fmaf_rn(sb1, tanhf(__fmaf_rn(sb0, y, bb0)), bb1);
    const float sig = 1.0f / (1.0f + expf(-x));
    dst[i] = __fmul_rn(__fmul_rn(sig, a), b);
}

void ggml_cuda_op_situ_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
        const float * scales, ggml_tensor * dst) {
    const int64_t n = g->ne[0];
    const int64_t k = ggml_nelements(dst);
    const int64_t s = g->nb[1] / sizeof(float);

    const int block_size = 256;
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(
        dim3((k + block_size - 1)/block_size, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(situ_f32, launch_params, (const float *) g->data, (const float *) u->data, (float *) dst->data,
        k, n, s, scales[0], scales[1], scales[2], scales[3], scales[4], scales[5], scales[6], scales[7]);
}
