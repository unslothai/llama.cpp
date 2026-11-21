#pragma once

#include "common.cuh"

// Fused SiTU gate, the chain over the two halves g, u of a gate_up product (views into the same tensor)
//   a = sa1*tanh(sa0*g + ba0) + ba1, b = sb1*tanh(sb0*u + bb0) + bb1     (SCALE, TANH, SCALE)
//   y = (sigmoid(g)*a)*b                                                  (SIGMOID, MUL, MUL)
// in one kernel with the rounding of the separate ops. g and u: rows of n elements, row stride s (elements).
// scales: {sa0, ba0, sa1, ba1, sb0, bb0, sb1, bb1}
void ggml_cuda_op_situ_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * g, const ggml_tensor * u,
        const float * scales, ggml_tensor * dst);
