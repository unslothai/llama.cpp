#pragma once

#include "common.cuh"

// whether the cuBLAS fallback of mul_mat_id should sort the ids on the device (complete = true) instead of on the host
bool ggml_cuda_mm_ids_device_sort_supported(int n_experts, int n_tokens, int n_expert_used);

// complete = true: always the counting sort, the only one with a complete permutation for repeated ids
void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream,
        bool complete = false);

// inverts ids_dst of ggml_cuda_launch_mm_ids_helper (compact row -> token slot) into token slot -> compact row
void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, int n, cudaStream_t stream);
