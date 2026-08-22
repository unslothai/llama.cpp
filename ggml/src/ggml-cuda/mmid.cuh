#pragma once

#include "common.cuh"

// Whether ggml_cuda_launch_mm_ids_helper can handle this shape on the device at all. The two
// implementations behind it have different limits: the single-warp-per-expert kernel needs one
// shared memory word per token, the counting sort needs one per expert. Callers that have a host
// fallback should ask here rather than reimplementing either bound.
bool ggml_cuda_mm_ids_helper_supported(int n_experts, int n_tokens);

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// Inverts the ids_dst map produced by ggml_cuda_launch_mm_ids_helper, i.e. turns the
// compact row -> token slot mapping into the token slot -> compact row mapping.
// Equivalent to calling the helper a second time with write_inverse == true, but much cheaper.
void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, int n, cudaStream_t stream);
