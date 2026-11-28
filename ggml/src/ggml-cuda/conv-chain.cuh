#pragma once

#include "common.cuh"

// Fused causal short convolution with a residual, the chain
//   x   = concat(past, transpose(cur), 0)                         (CONCAT; past: the last n_conv - 1 inputs per sequence)
//   the tail of x stored into the recurrent state                 (CPY of the last n_conv - 1 columns, or SET_ROWS of
//                                                                  one window per rollback slot after ARANGE, SCALE, CAST)
//   out = ssm_conv(x, w)                                          (SSM_CONV)
//   out = out + window(x, lag) * tap(lag), lag = 0 .. n_conv - 1  (optional: CONT of transpose(x), then MUL, ADD per lag)
//   y   = scale*(cur + out) + bias                                (ADD, SCALE)
// in one kernel per sequence and channel, with the same rounding as the separate kernels.
// Returns the number of nodes after node i taken by the fused chain, 0 if the nodes at i do not form one.
int ggml_cuda_try_fuse_conv_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i);
