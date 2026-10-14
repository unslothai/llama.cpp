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

// Helper function for mul_mat_id, converts ids to a more convenient format.
// ids_src1 describes how to permute the flattened column indices of src1 in order to get a compact src1 tensor sorted by expert.
// ids_dst describes the same mapping but for the dst tensor.
// The upper and lower bounds for the ith expert in the compact src1 tensor are stored in expert_bounds[i:i+1].
template <int n_expert_used_template>
__launch_bounds__(ggml_cuda_get_physical_warp_size(), 1)
static __global__ void mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    const int n_expert_used = n_expert_used_template == 0 ? n_expert_used_var : n_expert_used_template;
    const int expert = blockIdx.x;

    extern __shared__ char data_mm_ids_helper[];
    mm_ids_helper_store * store = (mm_ids_helper_store *) data_mm_ids_helper;

    int nex_prev   = 0; // Number of columns for experts with a lower index.
    int it_compact = 0; // Running index for the compact slice of this expert.

    if constexpr (n_expert_used_template == 0) {
        // Generic implementation:
        for (int it = 0; it < n_tokens; ++it) {
            int iex_used = -1; // The index at which the expert is used, if any.
            for (int iex = threadIdx.x; iex < n_expert_used; iex += warp_size) {
                const int expert_used = ids[it*si1 + iex];
                nex_prev += expert_used < expert;
                if (expert_used == expert) {
                    iex_used = iex;
                }
            }

            if (iex_used != -1) {
                store[it_compact] = mm_ids_helper_store(it, iex_used);
            }

            if (warp_reduce_any<warp_size>(iex_used != -1)) {
                it_compact++;
            }
        }
    } else {
        // Implementation optimized for specific numbers of experts used:
        // a warp holds a whole number of token slots, so the slot count is padded to a power of 2
        constexpr int neu_padded = mm_ids_pow2<n_expert_used_template>::value;
        static_assert(neu_padded <= warp_size && warp_size % neu_padded == 0, "bad n_expert_used");
        for (int it0 = 0; it0 < n_tokens; it0 += warp_size/neu_padded) {
            const int it = it0 + threadIdx.x / neu_padded;

            const int iex = threadIdx.x % neu_padded; // The index at which the expert is used, if any.
            const int expert_used = (neu_padded == n_expert_used || iex < n_expert_used) && it < n_tokens ?
                ids[it*si1 + iex] : INT_MAX;
            const int iex_used = expert_used == expert ? iex : -1;
            nex_prev += expert_used < expert;

            // Whether the threads at this token position have used the expert:
            const int it_compact_add_self = warp_reduce_any<neu_padded>(iex_used != -1);

            // Do a scan over threads at lower token positions in warp to get the correct index for writing data:
            int it_compact_add_lower = 0;
#pragma unroll
            for (int offset = neu_padded; offset < warp_size; offset += neu_padded) {
                const int tmp = __shfl_up_sync(0xFFFFFFFF, it_compact_add_self, offset, warp_size);
                if (threadIdx.x >= static_cast<unsigned int>(offset)) {
                    it_compact_add_lower += tmp;
                }
            }

            if (iex_used != -1) {
                store[it_compact + it_compact_add_lower] = mm_ids_helper_store(it, iex_used);
            }

            // The thread with the highest index in the warp always has the sum over the whole warp, use it to increment all threads:
            it_compact += __shfl_sync(0xFFFFFFFF, it_compact_add_lower + it_compact_add_self, warp_size - 1, warp_size);
        }
    }
    nex_prev = warp_reduce_sum<warp_size>(nex_prev);
    ggml_cuda_syncwarp();

    for (int itc = threadIdx.x; itc < it_compact; itc += warp_size) {
        const mm_ids_helper_store store_it = store[itc];
        const int it       = store_it.it();
        const int iex_used = store_it.iex_used();
        ids_dst[nex_prev + itc] = it*n_expert_used + iex_used;
        // ids_src1 holds the forward map, or the inverse map (token slot -> compact row) for quant dedup
        if (write_inverse) {
            ids_src1[it*n_expert_used + iex_used] = nex_prev + itc;
        } else {
            ids_src1[nex_prev + itc] = it*sis1 + iex_used % nchannels_y;
        }
    }

    if (threadIdx.x != 0) {
        return;
    }

    expert_bounds[expert] = nex_prev;

    if (expert < static_cast<int>(gridDim.x) - 1) {
        return;
    }

    expert_bounds[gridDim.x] = nex_prev + it_compact;
}

template <int n_expert_used_template>
static void launch_mm_ids_helper(
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used_var, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    GGML_ASSERT(n_tokens          < (1 << 22) && "too few bits in mm_ids_helper_store");
    GGML_ASSERT(n_expert_used_var < (1 << 10) && "too few bits in mm_ids_helper_store");

    const int id = ggml_cuda_get_device();
    const int warp_size = ggml_cuda_info().devices[id].warp_size;
    const size_t smpbo = ggml_cuda_info().devices[id].smpbo;
    CUDA_SET_SHARED_MEMORY_LIMIT(mm_ids_helper<n_expert_used_template>, smpbo);

    const dim3 num_blocks(n_experts, 1, 1);
    const dim3 block_size(warp_size, 1, 1);
    const size_t nbytes_shared = n_tokens*sizeof(mm_ids_helper_store);
    GGML_ASSERT(nbytes_shared <= smpbo);
    mm_ids_helper<n_expert_used_template><<<num_blocks, block_size, nbytes_shared, stream>>>
        (ids, ids_src1, ids_dst, expert_bounds, n_tokens, n_expert_used_var, nchannels_y, si1, sis1, write_inverse);
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
        // skip out of range ids: expert_bounds[n_experts] then comes out short, which the caller checks
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
        const int32_t * __restrict__ row_total, int32_t * __restrict__ expert_bounds, const int n_experts) {
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
            const int ej = __shfl_sync(0xFFFFFFFF, expert_used, j, warp_size);
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

    mm_ids_expert_prefix<<<1, MM_IDS_SCAN_BLOCK, 0, stream>>>(row_total.ptr, expert_bounds, n_experts);
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

// GGML_CUDA_MM_IDS_SORT=0/1 forces the single-warp kernel/the counting sort, for testing
static int mm_ids_sort_override() {
    static const int value = []() {
        const char * env = getenv("GGML_CUDA_MM_IDS_SORT");
        return env ? atoi(env) : -1;
    }();
    return value;
}

// the counting sort needs 4 launches instead of 1, so use it only for large shapes or when the single-warp kernel cannot run
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
    return (int64_t) n_experts*n_tokens >= (1 << 16);
}

static __global__ void mm_ids_invert(
        const int32_t * __restrict__ ids_dst, int32_t * __restrict__ ids_inverse, const int n) {
    const int ic = blockIdx.x*blockDim.x + threadIdx.x;

    if (ic >= n) {
        return;
    }

    // repeated ids leave holes with garbage in ids_dst on the single-warp path: skip them, no OOB write
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

// only the counting sort gives a complete permutation for repeated ids, the single-warp kernel leaves holes,
// so the host sort keeps rejecting those for the shapes the counting sort is not used for
bool ggml_cuda_mm_ids_helper_supported(const int n_experts, const int n_tokens, const int n_expert_used) {
    return mm_ids_use_counting_sort(n_experts, n_tokens, n_expert_used);
}

#define MM_IDS_DISPATCH(neu)                                                                                                    \
    do {                                                                                                                        \
        if (counting_sort) {                                                                                                    \
            launch_mm_ids_counting_sort<neu>(pool, ids, ids_src1, ids_dst, expert_bounds,                                       \
                n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);                             \
        } else {                                                                                                                \
            launch_mm_ids_helper<neu>(ids, ids_src1, ids_dst, expert_bounds,                                                    \
                n_experts, n_tokens, n_expert_used, nchannels_y, si1, sis1, write_inverse, stream);                             \
        }                                                                                                                       \
    } while (0)

void ggml_cuda_launch_mm_ids_helper(
        ggml_cuda_pool & pool,
        const int32_t * __restrict__ ids, int32_t * __restrict__ ids_src1, int32_t * __restrict__ ids_dst, int32_t * __restrict__ expert_bounds,
        const int n_experts, const int n_tokens, const int n_expert_used, const int nchannels_y, const int si1, const int sis1, const bool write_inverse, cudaStream_t stream) {
    const bool counting_sort = mm_ids_use_counting_sort(n_experts, n_tokens, n_expert_used);

    switch (n_expert_used) {
        case  2: MM_IDS_DISPATCH( 2); break;
        case  4: MM_IDS_DISPATCH( 4); break;
        case  6: MM_IDS_DISPATCH( 6); break;
        case  8: MM_IDS_DISPATCH( 8); break;
        case 10: MM_IDS_DISPATCH(10); break;
        case 16: MM_IDS_DISPATCH(16); break;
        case 32: MM_IDS_DISPATCH(32); break;
        default: MM_IDS_DISPATCH( 0); break;
    }
}

#undef MM_IDS_DISPATCH
