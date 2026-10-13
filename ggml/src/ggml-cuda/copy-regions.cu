// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "copy-regions.cuh"

// device-to-device copies of many regions in one launch, run on the SMs so they do not slow down uploads on the copy engines

#define COPY_REGIONS_MAX 128 // regions per launch, the kernel parameters stay below 4 KB

struct copy_regions_args {
    char       * dst  [COPY_REGIONS_MAX];
    const char * src  [COPY_REGIONS_MAX];
    int64_t      ne   [COPY_REGIONS_MAX];     // elements of the region
    int32_t      first[COPY_REGIONS_MAX + 1]; // index of the first chunk of the region, then the number of chunks
};

template <typename T>
static __global__ void copy_regions_kernel(const copy_regions_args args, const int n, const int64_t chunk) {
    const int n_items = args.first[n];
    for (int item = blockIdx.x; item < n_items; item += gridDim.x) {
        // region of the chunk: the last one whose first chunk is <= item
        int lo = 0;
        int hi = n - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1)/2;
            if (args.first[mid] <= item) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        const int64_t beg = (int64_t) (item - args.first[lo])*chunk;
        const int64_t end = min(beg + chunk, args.ne[lo]);

        const T * __restrict__ src = (const T *) args.src[lo];
        T       * __restrict__ dst = (T *) args.dst[lo];

        constexpr int unroll = 4;
        int64_t i = beg + threadIdx.x;
        for (; i + (unroll - 1)*(int64_t) blockDim.x < end; i += unroll*(int64_t) blockDim.x) {
            T v[unroll];
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                v[u] = src[i + u*blockDim.x];
            }
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                dst[i + u*blockDim.x] = v[u];
            }
        }
        for (; i < end; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

template <typename T>
static void copy_regions_launch(copy_regions_args & args, const int n, const size_t * sizes, const int max_ctas, cudaStream_t stream) {
    const int64_t chunk = (64*1024)/sizeof(T);
    int64_t n_items = 0;
    for (int i = 0; i < n; ++i) {
        args.ne[i]    = (int64_t) (sizes[i]/sizeof(T));
        args.first[i] = (int32_t) n_items;
        n_items += (args.ne[i] + chunk - 1)/chunk;
    }
    GGML_ASSERT(n_items <= INT32_MAX);
    args.first[n] = (int32_t) n_items;
    if (n_items == 0) {
        return;
    }
    const int n_ctas = (int) std::min<int64_t>(n_items, max_ctas);
    copy_regions_kernel<T><<<n_ctas, 256, 0, stream>>>(args, n, chunk);
}

void ggml_cuda_copy_regions_d2d(cudaStream_t stream, const int nsm, void * const * dst, const void * const * src, const size_t * sizes, const int n) {
    const int max_ctas = std::max(8, nsm);

    for (int i0 = 0; i0 < n; i0 += COPY_REGIONS_MAX) {
        const int nc = std::min(n - i0, COPY_REGIONS_MAX);

        copy_regions_args args;
        uintptr_t align = 0;
        for (int i = 0; i < nc; ++i) {
            args.dst[i] = (char *) dst[i0 + i];
            args.src[i] = (const char *) src[i0 + i];
            align |= (uintptr_t) args.dst[i] | (uintptr_t) args.src[i] | (uintptr_t) sizes[i0 + i];
        }

        if (align % 16 == 0) {
            copy_regions_launch<int4>(args, nc, sizes + i0, max_ctas, stream);
        } else if (align % 8 == 0) {
            copy_regions_launch<int2>(args, nc, sizes + i0, max_ctas, stream);
        } else if (align % 4 == 0) {
            copy_regions_launch<int>(args, nc, sizes + i0, max_ctas, stream);
        } else {
            copy_regions_launch<char>(args, nc, sizes + i0, max_ctas, stream);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
