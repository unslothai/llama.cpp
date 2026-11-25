#pragma once

#include "common.cuh"

// Fused depth mix (attention over earlier block outputs), the chain after normed = rms_norm(x) [ncols, 1, T] (the RMS_NORM stays its own op):
//   states_t = concat(prev_t, reshape(normed), 0) (CONCAT, [n - 1, ncols, T] + [1, ncols, T])
//   states   = concat(prev, normed, 1)            (CONCAT, [ncols, n - 1, T] + [ncols, 1, T])
//   scores   = soft_max(states^T q * scale)       ([REPEAT,] MUL_MAT, SOFT_MAX_EXT without mask and sinks)
//   out      = states_t^T scores                  (MUL_MAT, optional; mul_mat(states_t, scores) [ncols, nq, T] or mul_mat(scores, states_t) [nq, ncols, T])
// in one kernel, with the same rounding as the separate ops: both products follow mul_mat_vec_f for the block size it picks, the softmax follows soft_max_f32.
// Only taken where the separate ops would run exactly those kernels (MMVF for both products, one warp per softmax row).
// Intermediates with uses outside the chain are written as well.
// Returns the number of nodes after node i (the first CONCAT) taken by the fused kernel, 0 if the nodes at i do not form one.
// memory_ok(cgraph, i, count, outs, n_outs): the outputs of nodes [i, i + count) overlap no input of the chain
typedef bool (*depth_mix_memory_check_t)(const ggml_cgraph * cgraph, int node_idx, int node_count, const int * out_nodes, int out_count);
int ggml_cuda_try_fuse_depth_mix(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i, depth_mix_memory_check_t memory_ok);
