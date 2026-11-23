#pragma once

#include "common.cuh"

// Fused depth mix (attention over earlier block outputs), the chain after normed = rms_norm(x) [ncols, 1, T]:
//   states_t = concat(prev_t, reshape(normed), 0) (CONCAT, [n - 1, ncols, T] + [1, ncols, T])
//   states   = concat(prev, normed, 1)            (CONCAT, [ncols, n - 1, T] + [ncols, 1, T])
//   scores   = soft_max(states^T q * scale)       ([REPEAT,] MUL_MAT, SOFT_MAX_EXT without mask and sinks)
//   out      = states_t^T scores                  (MUL_MAT, optional; mul_mat(states_t, scores) [ncols, nq, T] or mul_mat(scores, states_t) [nq, ncols, T])
// in one kernel, with the same rounding as the separate ops: both products follow mul_mat_vec_f for the block size it picks, the softmax follows soft_max_f32.
// With the scores as src0 of the last MUL_MAT the separate op sums in another order: that form matches it to rounding, not bit for bit.
// Only taken where the separate ops would run exactly those kernels (MMVF for both products, one warp per softmax row).
// Intermediates with uses outside the chain are written as well.
// On NVIDIA cc >= 10.0 the kernel also takes the norms around the chain, with the rounding of the separate kernels (the vector path of rms_norm_f32 with one float4 per thread, ncols/4 <= 1024).
// Before the chain: the RMS_NORM of x, or of x = a + b after an ADD [and RESHAPE] (the sum is written); after it: [RESHAPE,] RMS_NORM, MUL by a weight row of a one-query mix output.
// GGML_CUDA_DISABLE_DEPTH_MIX_NORM_FUSION=1 keeps those on their own kernels.
// Returns the number of nodes after node i (the first CONCAT, or the ADD / RMS_NORM before it) taken by the fused kernel, 0 if the nodes at i do not form one.
// memory_ok(cgraph, i, count, outs, n_outs): the outputs of nodes [i, i + count) overlap no input of the chain
typedef bool (*depth_mix_memory_check_t)(const ggml_cgraph * cgraph, int node_idx, int node_count, const int * out_nodes, int out_count);
int ggml_cuda_try_fuse_depth_mix(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i, depth_mix_memory_check_t memory_ok);

// GGML_CUDA_DEPTH_MIX_SHADOW=1 only: compare and free the shadow outputs still pending at the end of a graph
void ggml_cuda_depth_mix_shadow_flush(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph);
