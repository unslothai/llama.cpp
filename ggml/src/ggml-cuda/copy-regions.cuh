#pragma once

#include "common.cuh"

// copies n regions dst[i] <- src[i] of sizes[i] bytes in stream with a kernel (both pointers device memory of the stream's
// device), one launch per 128 regions, using at most one CTA per SM
void ggml_cuda_copy_regions_d2d(cudaStream_t stream, int nsm, void * const * dst, const void * const * src, const size_t * sizes, int n);
