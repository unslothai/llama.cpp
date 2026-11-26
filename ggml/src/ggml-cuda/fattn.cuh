#include "common.cuh"

// whether ggml_cuda_flash_attn_ext runs dst with the vector kernel compiled for its K/V types
bool ggml_cuda_flash_attn_ext_uses_vec(int device, const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
