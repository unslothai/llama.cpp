#pragma once

#include "common.cuh"

struct ggml_cuda_mm_id_tile_spec {
    int32_t expert;
    int32_t col_low;
    int32_t col_high;

    int32_t padding;
};
static_assert(sizeof(ggml_cuda_mm_id_tile_spec) == 16, "bad ggml_cuda_mm_id_tile_spec size");

// whether the cuBLAS fallback of mul_mat_id should sort the ids on the device instead of on the host
bool ggml_cuda_mm_ids_device_sort_supported(int n_experts, int n_tokens, int n_expert_used);

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream);

// cuBLAS fallback: the counting sort, whatever the shape; the bounds of expert i are expert_bounds[i:i+1]
void ggml_cuda_launch_mm_ids_sort(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        int n_experts, int n_tokens, int n_expert_used, int nchannels_y, int si1, int sis1, bool write_inverse, cudaStream_t stream);

// inverts ids_dst of the sort (compact row -> token slot) into token slot -> compact row
void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, int n, cudaStream_t stream);
