#pragma once

#include "common.cuh"

// fuse a relative-position bias gathered into a dense F16 mask into the flash attention that reads it:
//   [SUB,] [CONT,] GET_ROWS, CPY (F32 -> F16), ADD (F16 mask), FLASH_ATTN_EXT, with only views in between, no concurrent streams;
// other nodes in between that do not read the chain (e.g. the FILL of the sinks) are run first with compute;
// returns the number of nodes after node i taken by the fused kernel, 0 if the nodes at i do not form one
int ggml_cuda_try_fuse_rel_bias_fa(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*compute)(ggml_backend_cuda_context & ctx, ggml_tensor * dst));
