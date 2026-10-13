// [TAG_EXACT_CONCURRENCY] which buffer types the mode accepts a weight in, as used by the context weight check and the lora refusal.
// A host buffer is refused: its ops move to the GPU only once the batch is wide enough, so results depend on the step width.

#include "ggml-backend.h"

#include "../src/llama-impl.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

int main() {
    ggml_backend_load_all();

    // nothing placed anywhere is nothing to trust
    assert(!llama_exact_buft_invariant(nullptr));

    // the plain CPU buffer, and the pinned host buffer a GPU backend offers, are both host memory
    assert(!llama_exact_buft_invariant(ggml_backend_cpu_buffer_type()));

    bool checked_gpu = false;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

        const bool invariant = reg && llama_exact_backend_name(ggml_backend_reg_name(reg));

        // a device's own buffer follows its backend, and its host buffer never does
        assert(llama_exact_buft_invariant(ggml_backend_dev_buffer_type(dev)) == invariant);

        if (auto * host = ggml_backend_dev_host_buffer_type(dev)) {
            assert(!llama_exact_buft_invariant(host));
        }

        checked_gpu = checked_gpu || invariant;
    }

    // the registry names the mode trusts, whatever this build has
    assert(llama_exact_backend_name("CUDA"));
    assert(llama_exact_backend_name("ROCm"));
    assert(llama_exact_backend_name("MUSA"));
    assert(!llama_exact_backend_name("CPU"));
    assert(!llama_exact_backend_name("BLAS"));
    assert(!llama_exact_backend_name(nullptr));

    printf("%s: all tests passed%s\n", __func__,
            checked_gpu ? "" : " (no batch-invariant device here, the positive case was not exercised)");

    return 0;
}
