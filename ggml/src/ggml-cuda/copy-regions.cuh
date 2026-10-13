// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#pragma once

#include "common.cuh"

// copies n regions dst[i] <- src[i] of sizes[i] bytes (device memory of the stream's device) with a kernel, at most one CTA per SM
void ggml_cuda_copy_regions_d2d(cudaStream_t stream, int nsm, void * const * dst, const void * const * src, const size_t * sizes, int n);
