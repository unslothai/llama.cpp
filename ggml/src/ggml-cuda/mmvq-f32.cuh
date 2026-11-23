#pragma once

#include "common.cuh"

// Batch-1 MUL_MAT of Q8_0 weights with F32 activations: the weights are dequantized in registers and dotted with the
// F32 src1 directly, so src1 is not quantized to q8_1 first. This changes rounding against MMVQ (whose src1 is
// quantized to q8_1, i.e. 8 bits per activation): the products are formed from the exact activations instead.
// Taken only for one src1 column, contiguous 2D operands and NVIDIA cc >= 10.0.
// Kill switch: GGML_CUDA_DISABLE_MMVQ_F32_ACT.
bool ggml_cuda_should_use_mmvq_f32(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, int cc);

void ggml_cuda_mul_mat_vec_q_f32(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);
