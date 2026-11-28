#include "common.cuh"

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_supported(int device, const ggml_tensor * dst);

bool ggml_cuda_flash_attn_ext_banded_use_mma(int device, const ggml_tensor * dst);

inline bool ggml_cuda_banded_mma_disabled() {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_DISABLE_BANDED_MMA");
        return env != nullptr && atoi(env) != 0;
    }();
    return disabled;
}

size_t ggml_cuda_flash_attn_ext_get_alloc_size(int device, const ggml_tensor * dst);
