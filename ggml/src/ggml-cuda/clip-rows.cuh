#pragma once

#include "common.cuh"

// Fused row L2 clip, the chain
//   c = x (* w)                                  (MUL, optional, w broadcast over dims 2 and 3)
//   n = sqrt(sum_rows(c*c)) + bias               (SQR, SUM_ROWS, SQRT, SCALE with scale 1)
//   f = clamp(limit/n, lo, hi)                   (DIV by n, CLAMP; limit is a tensor or a constant)
//   y = c*f                                      (MUL)
// in one kernel, with the same rounding as the separate ops: the row sum follows the reduction of
// sum_rows_f32_cuda for the same row count, the products are not contracted.
// x, w, limit and dst are contiguous F32; x is [ncols, nrows], w is [ncols, w_rows] (row r reads row r % w_rows),
// limit (if not null) holds one value per row.
void ggml_cuda_op_clip_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        const ggml_tensor * limit, float limit_value, float bias, float lo, float hi, ggml_tensor * dst);

// The same clip written as an L2 norm, the chain
//   c = x (* w)                                  (MUL, optional, w as above)
//   y = scale*(c/max(|c|, eps)) + bias           (L2_NORM with eps, SCALE)
// in one kernel, with the reduction and rounding of l2_norm_f32_cuda and scale_f32 for the same row length.
// For a clip to threshold t the graph uses eps = scale = t, bias 0: y = t*c/max(|c|, t).
void ggml_cuda_op_l2_clip_rows_fused(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * w,
        float eps, float scale, float bias, ggml_tensor * dst);
