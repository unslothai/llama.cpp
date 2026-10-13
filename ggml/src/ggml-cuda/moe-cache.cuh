// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_moe_cache_lookup_supported(const ggml_tensor * op);
