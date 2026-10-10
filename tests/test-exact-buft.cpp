// [TAG_EXACT_CONCURRENCY] which buffer types the mode accepts a weight in. A host buffer is not one
// of them: the scheduler runs an operation on the backend holding its weight, and moves a host
// weight's operation to the GPU only once the batch is wide enough, so its result would depend on
// how many sequences share the step. This is the predicate behind both the context's weight check
// and the refusal of a lora that would inherit such a buffer.

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
