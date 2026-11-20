#include "common.cuh"
#include "mmid.cuh"

// To reduce shared memory use, store "it" and "iex_used" with 22/10 bits each.
struct mm_ids_helper_store {
    uint32_t data;

    __device__ mm_ids_helper_store(const uint32_t it, const uint32_t iex_used) {
        data = (it & 0x003FFFFF) | (iex_used << 22);
    }

    __device__ uint32_t it() const {
        return data & 0x003FFFFF;
    }

    __device__ uint32_t iex_used() const {
        return data >> 22;
    }
};
static_assert(sizeof(mm_ids_helper_store) == 4, "unexpected size for mm_ids_helper_store");

// the generic path passes 0, which needs no padding since it never groups lanes by token
template <int n> struct mm_ids_pow2 { static constexpr int value = 2*mm_ids_pow2<(n + 1)/2>::value; };
template <>      struct mm_ids_pow2<1> { static constexpr int value = 1; };
template <>      struct mm_ids_pow2<0> { static constexpr int value = 1; };
// HIP has no __syncwarp; the kernels that use this run one warp per block, so __syncthreads is equivalent
static __device__ __forceinline__ void mm_ids_warp_sync() {
#ifdef GGML_USE_HIP
    __syncthreads();
#else
    __syncwarp();
#endif // GGML_USE_HIP
}

template <bool increment_nex_prev, bool write_to_store>
static __device__ __forceinline__ void check_experts(
        mm_ids_helper_store * store, int & nex_prev, int & it_compact,
        const int expert, const int it, const int iex, const int expert_found) {
    if (increment_nex_prev) {
        nex_prev += expert_found < expert;
    }

    if (write_to_store) {
        const ggml_cuda_thread_mask_t iex_used = ggml_cuda_ballot(expert_found == expert);
        const ggml_cuda_thread_mask_t mask = (ggml_cuda_thread_mask_t(1) << threadIdx.x) - 1;
        const int it_compact_offset = ggml_cuda_popc(iex_used & mask);

        if (expert_found == expert) {
            store[it_compact + it_compact_offset] = mm_ids_helper_store(it, iex);
        }

        it_compact += ggml_cuda_popc(iex_used);
    }
}

static __device__ __forceinline__ void dump_store(
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int n_expert_used, const int nchannels_y, const int sis1, const bool write_inverse,
        const mm_ids_helper_store * __restrict__ store, const int nex_prev, int & n_dumped, const int it_compact) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + n_dumped + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + n_dumped + itc;
        } else {
            ids_src1[nex_prev + n_dumped + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }
    n_dumped += it_compact;
}

constexpr int n_unroll = 4; // Manually unroll this many loop iterations for better performance.

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The first (n_tokens*n_expert_used + J - 1) / tile_width elements of tile_spec contain the specication for
//   matrix multiplication tiles at token positions which are exactly divided by tile_width.
// Additionally the following n_expert elements of tile_spec may contain additional tiles that fall between these seams.
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const uint3 tile_width,
        const int si1, const int sis1, const bool write_inverse, const int ne_store) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;

    const int expert = blockIdx.x; // The expert this CUDA block handles.

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int n_dumped   = 0; // Number of data values that have been dumped to output buffers.
    int it_compact = 0; // Running index for the compact slice of this expert.

    // Step 1:   Iterate over n_tokens, increment nex_prev to determine the output indices, fill up store.
    // Step 1.5: If the store does not have enough space to hold all values, iterate over the rest of the tokens just to increment nex_prev.
    // Step 2:   Dump the stored values using nex_prev to get the correct output indices.
    // Step 3:   If we were not able to process all values, resume at the index up to which we were able to dump the output and re-do.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        int it = 0;
        for (; it < n_tokens && it_compact + max(warp_size, n_expert_used) <= ne_store; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }
        int it_dumped = it;
        for (; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = false;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }

        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        ggml_cuda_syncwarp();
        dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        it_compact = 0;

        for (int it = it_dumped; it < n_tokens; ++it) {
            for (int iex0 = 0; iex0 < n_expert_used; iex0 += warp_size) {
                const int iex = iex0 + threadIdx.x;
                const int expert_found = iex < n_expert_used ? ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = false;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
                if (it_compact + max(warp_size, n_expert_used) >= ne_store) {
                    ggml_cuda_syncwarp();
                    dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
                    it_compact = 0;
                }
            }
        }
        if (it_compact > 0) {
            ggml_cuda_syncwarp();
            dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.

        int it0 = 0;
        for (; it0 < n_tokens && it_compact + n_unroll*warp_size <= ne_store; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }
        int it0_dumped = it0;
        for (; it0 < n_tokens; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = true;
                constexpr bool write_to_store     = false;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
            }
        }

        nex_prev = warp_reduce_sum<warp_size>(nex_prev);
        ggml_cuda_syncwarp();
        dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        it_compact = 0;

        for (it0 = it0_dumped; it0 < n_tokens; it0 += n_unroll*warp_size/neu_padded) {
#pragma unroll
            for (int it1 = 0; it1 < n_unroll*warp_size/neu_padded; it1 += warp_size/neu_padded) {
                const int it = it0 + it1 + threadIdx.x / neu_padded;
                const int expert_found = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                    ids[it*si1 + iex] : INT_MAX;

                constexpr bool increment_nex_prev = false;
                constexpr bool write_to_store     = true;
                check_experts<increment_nex_prev, write_to_store>
                    (store, nex_prev, it_compact, expert, it, iex, expert_found);
                if (it_compact + warp_size >= ne_store) {
                    ggml_cuda_syncwarp();
                    dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
                    it_compact = 0;
                }
            }
        }
        if (it_compact > 0) {
            ggml_cuda_syncwarp();
            dump_store(ids_src1, ids_dst, n_expert_used, nchannels_y, sis1, write_inverse, store, nex_prev, n_dumped, it_compact);
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), n_dumped);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + n_dumped);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_tokens*n_expert_used + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const uint3 tile_width_fd = init_fastdiv_values(tile_width);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t ne_shared_pad = n_expert_used_template == 0 ? std::max(n_expert_used_var, warp_size) : n_unroll*warp_size;
    const size_t nbytes_store_core = std::min(smpbo - ne_shared_pad * sizeof(mm_ids_helper_store), n_tokens * sizeof(mm_ids_helper_store));
    const size_t nbytes_shared = nbytes_store_core + ne_shared_pad * sizeof(mm_ids_helper_store);
    const size_t ne_store = nbytes_shared / sizeof(mm_ids_helper_store);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, tile_spec, n_tokens, n_expert_used_var, nchannels_y, tile_width_fd, si1, sis1, write_inverse, ne_store);
}

// Stable counting sort: the same mapping as mm_ids_helper, with work proportional to the number of assignments instead of n_experts*n_tokens.

#define MM_IDS_SCAN_BLOCK 256

// per-block histogram over experts, block w owns tokens [w*tpb, min((w+1)*tpb, n_tokens))
template <int n_expert_used_template>
static __global__ void mm_ids_hist(
        const int32_t * __restrict__ ids, int32_t * __restrict__ hist,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int si1,
        const int tpb, const int W) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;

    const int w   = blockIdx.x;
    const int it0 = w*tpb;
    const int it1 = min(it0 + tpb, n_tokens);

    extern __shared__ char data_mm_ids_helper[];
    int32_t * counts = (int32_t *) data_mm_ids_helper;

    for (int e = threadIdx.x; e < n_experts; e += warp_size) {
        counts[e] = 0;
    }
    mm_ids_warp_sync();

    const int npairs = (it1 - it0)*n_expert_used;
    for (int p = threadIdx.x; p < npairs; p += warp_size) {
        const int itl = p / n_expert_used;
        const int iex = p - itl*n_expert_used;
        const int expert_used = ids[(size_t)(it0 + itl)*si1 + iex];
        // skip out of range ids: expert_bounds[n_experts] then comes out short, which mm_ids_expert_prefix checks
        if (expert_used >= 0 && expert_used < n_experts) {
            atomicAdd(&counts[expert_used], 1);
        }
    }
    mm_ids_warp_sync();

    for (int e = threadIdx.x; e < n_experts; e += warp_size) {
        hist[(size_t) e*W + w] = counts[e];
    }
}

// Exclusive scan of each histogram row in place, plus the row total. One block per expert.
static __global__ void mm_ids_scan_rows(
        int32_t * __restrict__ hist, int32_t * __restrict__ row_total, const int W) {
    int32_t * row = hist + (size_t) blockIdx.x*W;

    __shared__ int32_t sdata[MM_IDS_SCAN_BLOCK];
    __shared__ int32_t carry;

    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (int base = 0; base < W; base += MM_IDS_SCAN_BLOCK) {
        const int     i = base + threadIdx.x;
        const int32_t v = i < W ? row[i] : 0;

        sdata[threadIdx.x] = v;
        __syncthreads();

        for (int offset = 1; offset < MM_IDS_SCAN_BLOCK; offset <<= 1) {
            const int32_t add = (int) threadIdx.x >= offset ? sdata[threadIdx.x - offset] : 0;
            __syncthreads();
            sdata[threadIdx.x] += add;
            __syncthreads();
        }

        const int32_t inclusive  = sdata[threadIdx.x];
        const int32_t base_carry = carry;
        __syncthreads();

        if (i < W) {
            row[i] = base_carry + inclusive - v;
        }
        if (threadIdx.x == MM_IDS_SCAN_BLOCK - 1) {
            carry = base_carry + inclusive;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        row_total[blockIdx.x] = carry;
    }
}

// Exclusive scan of the per-expert totals into expert_bounds. Single block.
static __global__ void mm_ids_expert_prefix(
        const int32_t * __restrict__ row_total, int32_t * __restrict__ expert_bounds, const int n_experts,
        const int64_t n_assignments) {
    __shared__ int32_t sdata[MM_IDS_SCAN_BLOCK];
    __shared__ int32_t carry;

    if (threadIdx.x == 0) {
        carry = 0;
    }
    __syncthreads();

    for (int base = 0; base < n_experts; base += MM_IDS_SCAN_BLOCK) {
        const int     i = base + threadIdx.x;
        const int32_t v = i < n_experts ? row_total[i] : 0;

        sdata[threadIdx.x] = v;
        __syncthreads();

        for (int offset = 1; offset < MM_IDS_SCAN_BLOCK; offset <<= 1) {
            const int32_t add = (int) threadIdx.x >= offset ? sdata[threadIdx.x - offset] : 0;
            __syncthreads();
            sdata[threadIdx.x] += add;
            __syncthreads();
        }

        const int32_t inclusive  = sdata[threadIdx.x];
        const int32_t base_carry = carry;
        __syncthreads();

        if (i < n_experts) {
            expert_bounds[i] = base_carry + inclusive - v;
        }
        if (threadIdx.x == MM_IDS_SCAN_BLOCK - 1) {
            carry = base_carry + inclusive;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // the histogram skips out of range ids, and every consumer of the sort needs a complete permutation:
        // stop here, as the host checks of the other paths do, instead of handing on unwritten entries
        if (int64_t(carry) != n_assignments) {
            printf("mm_ids: expert id out of range\n");
            __trap();
        }
        expert_bounds[n_experts] = carry;
    }
}

// Stable scatter: block w writes its first assignment to expert e at expert_bounds[e] + hist[e][w].
template <int n_expert_used_template>
static __global__ void mm_ids_scatter(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ hist,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y,
        const int si1, const int sis1, const bool write_inverse, const int tpb, const int W) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;

    const int lane = threadIdx.x;
    const int w    = blockIdx.x;
    const int it0  = w*tpb;
    const int it1  = min(it0 + tpb, n_tokens);

    extern __shared__ char data_mm_ids_helper[];
    int32_t * cursor = (int32_t *) data_mm_ids_helper;

    for (int e = lane; e < n_experts; e += warp_size) {
        cursor[e] = expert_bounds[e] + hist[(size_t) e*W + w];
    }
    mm_ids_warp_sync();

    const int npairs = (it1 - it0)*n_expert_used;
    for (int p0 = 0; p0 < npairs; p0 += warp_size) {
        const int p = p0 + lane;

        int expert_used = -1;
        int it  = 0;
        int iex = 0;
        if (p < npairs) {
            const int itl = p / n_expert_used;
            iex = p - itl*n_expert_used;
            it  = it0 + itl;
            const int e = ids[(size_t) it*si1 + iex];
            expert_used = e >= 0 && e < n_experts ? e : -1;
        }

        // rank by lane among lanes with the same expert: keeps the token-major, slot-minor order of mm_ids_helper
        int rank = 0;
        int cnt  = 0;
#pragma unroll
        for (int j = 0; j < warp_size; ++j) {
            const int ej = ggml_cuda_shfl(expert_used, j, warp_size);
            if (ej == expert_used) {
                cnt++;
                rank += j < lane;
            }
        }

        int pos = 0;
        if (expert_used >= 0) {
            pos = cursor[expert_used] + rank;
        }
        mm_ids_warp_sync(); // every lane has read its cursor before the leader advances it
        if (expert_used >= 0 && rank == 0) {
            cursor[expert_used] += cnt;
        }
        mm_ids_warp_sync();

        if (expert_used >= 0) {
            ids_dst[pos] = it*n_expert_used + iex;
            // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
            if (write_inverse) {
                ids_src1[it*n_expert_used + iex] = pos;
            } else {
                ids_src1[pos] = it*sis1 + iex % nchannels_y;
            }
        }
    }
}

template <int n_expert_used_template>
static void launch_mm_ids_counting_sort(
        ggml_cuda_pool & pool,
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const int id        = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const int nsm       = ggml_cuda_info().devices[id].nsm;

    // one warp per block over a contiguous token range; cap the block count to keep the histogram small
    int tpb = 16;
    int W   = (n_tokens + tpb - 1) / tpb;
    const int max_W = 8*nsm;
    while (W > max_W) {
        tpb *= 2;
        W    = (n_tokens + tpb - 1) / tpb;
    }

    ggml_cuda_pool_alloc<int32_t> hist(pool, (size_t) n_experts*W);
    ggml_cuda_pool_alloc<int32_t> row_total(pool, n_experts);

    const size_t nbytes_shared = (size_t) n_experts*sizeof(int32_t);

    mm_ids_hist<n_expert_used_template><<<W, warp_size, nbytes_shared, stream>>>
        (ids, hist.ptr, n_experts, n_tokens, n_expert_used_var, si1, tpb, W);
    CUDA_CHECK(cudaGetLastError());

    mm_ids_scan_rows<<<n_experts, MM_IDS_SCAN_BLOCK, 0, stream>>>(hist.ptr, row_total.ptr, W);
    CUDA_CHECK(cudaGetLastError());

    mm_ids_expert_prefix<<<1, MM_IDS_SCAN_BLOCK, 0, stream>>>(row_total.ptr, expert_bounds, n_experts, int64_t(n_tokens)*n_expert_used_var);
    CUDA_CHECK(cudaGetLastError());

    mm_ids_scatter<n_expert_used_template><<<W, warp_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, hist.ptr, n_experts, n_tokens, n_expert_used_var,
         nchannels_y, si1, sis1, write_inverse, tpb, W);
    CUDA_CHECK(cudaGetLastError());
}

static bool mm_ids_counting_sort_possible(const int n_experts) {
    const int id = ggml_cuda_get_device();
    return (size_t) n_experts*sizeof(int32_t) <= ggml_cuda_info().devices[id].smpb;
}

static bool mm_ids_legacy_possible(const int n_tokens, const int n_expert_used) {
    const int id = ggml_cuda_get_device();
    return (size_t) n_tokens*sizeof(mm_ids_helper_store) <= ggml_cuda_info().devices[id].smpbo &&
        n_tokens < (1 << 22) && n_expert_used < (1 << 10);
}

// GGML_CUDA_MM_IDS_SORT=0/1 forces the single-warp kernel/the counting sort (cuBLAS fallback: the host/device sort), for testing
static int mm_ids_sort_override() {
    static const int value = []() {
        const char * env = getenv("GGML_CUDA_MM_IDS_SORT");
        return env ? atoi(env) : -1;
    }();
    return value;
}

// MMQ/MMF: one warp per expert scans all ids in the single-warp kernel, the counting sort costs 4 launches instead of 1.
// Both scale with n_expert_used the same way and the experts run in parallel, so the crossover is in n_tokens:
// measured at 256-512 tokens for 64-512 experts and 2-8 used on T4, L4 and RTX PRO 6000
static bool mm_ids_counting_sort_faster(const int n_tokens) {
    return n_tokens >= 512;
}

// cuBLAS fallback: the device counting sort against the host sort, which loops over all n_tokens*n_expert_used ids once per expert;
// the device sort was faster in every measured cell from n_experts*n_tokens*n_expert_used >= 16384 on, and up to 3% slower below
static bool mm_ids_device_sort_faster(const int n_experts, const int n_tokens, const int n_expert_used) {
    return (int64_t) n_experts*n_tokens*n_expert_used >= 16384;
}

static bool mm_ids_use_counting_sort(const int n_experts, const int n_tokens, const int n_expert_used) {
    // an empty batch would launch an empty grid, the single-warp kernel writes the zero bounds instead
    if (n_tokens == 0 || !mm_ids_counting_sort_possible(n_experts)) {
        return false;
    }
    if (!mm_ids_legacy_possible(n_tokens, n_expert_used)) {
        return true; // the single-warp kernel cannot run at all
    }
    const int forced = mm_ids_sort_override();
    if (forced >= 0) {
        return forced != 0;
    }
    return mm_ids_counting_sort_faster(n_tokens);
}

static __global__ void mm_ids_invert(
        const int32_t * __restrict__ ids_dst, int32_t * __restrict__ ids_inverse, const int n) {
    const int ic = blockIdx.x*blockDim.x + threadIdx.x;

    if (ic >= n) {
        return;
    }

    // out of range ids leave unwritten rows in ids_dst: skip them, no OOB write
    const int32_t id = ids_dst[ic];
    if (id < 0 || id >= n) {
        return;
    }

    ids_inverse[id] = ic;
}

void ggml_cuda_launch_mm_ids_invert(
        const int32_t * ids_dst, int32_t * ids_inverse, const int n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }

    const int block_size = 256;
    const int num_blocks = (n + block_size - 1) / block_size;
    mm_ids_invert<<<num_blocks, block_size, 0, stream>>>(ids_dst, ids_inverse, n);
}

// the device sort of the cuBLAS fallback always uses the counting sort (complete = true), it is only picked where that is faster
bool ggml_cuda_mm_ids_device_sort_supported(const int n_experts, const int n_tokens, const int n_expert_used) {
    if (n_tokens == 0 || !mm_ids_counting_sort_possible(n_experts)) {
        return false;
    }
    const int forced = mm_ids_sort_override();
    if (forced >= 0) {
        return forced != 0;
    }
    return mm_ids_device_sort_faster(n_experts, n_tokens, n_expert_used);
}

// tile_spec from the expert bounds of the counting sort: the same tiles as the tail of mm_ids_helper. One thread per expert.
static __global__ void mm_ids_tile_spec(
        const int32_t * __restrict__ expert_bounds, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_slots, const uint3 tile_width) {
    const int expert = blockIdx.x*blockDim.x + threadIdx.x;

    if (expert >= n_experts) {
        return;
    }

    const int nex_prev = expert_bounds[expert];
    const int n_match  = expert_bounds[expert + 1] - nex_prev;

    int col_low  = nex_prev;
    int col_high = nex_prev + min(int(tile_width.z), n_match);

    // The while loop covers all tiles at the seams containing the column with % tile_width == 0.
    while (col_high > fastdiv(col_low + tile_width.z - 1, tile_width) * tile_width.z) {
        tile_spec[fastdiv(col_low + tile_width.z - 1, tile_width)]
            = {expert, col_low, col_high, /*padding =*/ 0};

        col_low  = col_high;
        col_high = min(col_high + int(tile_width.z), nex_prev + n_match);
    }

    // Finally, at most one extra tile_spec between the seams per expert.
    tile_spec[fastdiv(n_slots + tile_width.z - 1, tile_width) + expert]
        = {expert, col_low, col_high, /*padding =*/ 0};
}

#define MM_IDS_SWITCH(stmt)                     \
    switch (n_expert_used) {                    \
        case  2: { constexpr int neu =  2; stmt; } break; \
        case  4: { constexpr int neu =  4; stmt; } break; \
        case  6: { constexpr int neu =  6; stmt; } break; \
        case  8: { constexpr int neu =  8; stmt; } break; \
        case 10: { constexpr int neu = 10; stmt; } break; \
        case 16: { constexpr int neu = 16; stmt; } break; \
        case 32: { constexpr int neu = 32; stmt; } break; \
        default: { constexpr int neu =  0; stmt; } break; \
    }

void ggml_cuda_launch_mm_ids_sort(
        ggml_cuda_pool & pool,
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, int32_t * expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens > 0 && mm_ids_counting_sort_possible(n_experts));

    MM_IDS_SWITCH(launch_mm_ids_counting_sort<neu>(pool, ids, ids_src1, ids_dst, expert_bounds,
        n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream));
}

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * __restrict__ ids,
        int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, ggml_cuda_mm_id_tile_spec * __restrict__ tile_spec,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int tile_width,
        const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    if (mm_ids_use_counting_sort(n_experts, n_tokens, n_expert_used)) {
        ggml_cuda_pool_alloc<int32_t> expert_bounds(pool, n_experts + 1);

        ggml_cuda_launch_mm_ids_sort(pool, ids, ids_src1, ids_dst, expert_bounds.ptr,
            n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);

        const int block_size = 256;
        mm_ids_tile_spec<<<(n_experts + block_size - 1)/block_size, block_size, 0, stream>>>
            (expert_bounds.ptr, tile_spec, n_experts, n_tokens*n_expert_used, init_fastdiv_values(tile_width));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    MM_IDS_SWITCH(launch_mm_ids_helper<neu>(ids, ids_src1, ids_dst, tile_spec,
        n_experts, n_tokens, n_expert_used, nchannels_y, tile_width, si1, sis1, write_inverse, stream));
}

#undef MM_IDS_SWITCH
