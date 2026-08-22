#pragma once

void ggml_cuda_launch_mm_ids_helper(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// Inverts the ids_dst map produced by ggml_cuda_launch_mm_ids_helper, i.e. turns the
// compact row -> token slot mapping into the token slot -> compact row mapping.
// Equivalent to calling the helper a second time with write_inverse == true, but much cheaper.
void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, int n, cudaStream_t stream);
