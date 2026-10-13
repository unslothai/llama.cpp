// [TAG_EXACT_CONCURRENCY] which buffer types the mode accepts a weight in.
// A host buffer is not accepted: the scheduler offloads its ops only for wide batches, so results depend on the step width.

#include "ggml-backend.h"

#include "../src/llama-impl.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

int main() {
    ggml_backend_load_all();

    assert(!llama_exact_buft_invariant(nullptr));

    assert(!llama_exact_buft_invariant(ggml_backend_cpu_buffer_type()));

    bool checked_gpu = false;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

        const bool invariant = reg && llama_exact_backend_name(ggml_backend_reg_name(reg));

        assert(llama_exact_buft_invariant(ggml_backend_dev_buffer_type(dev)) == invariant);

        if (auto * host = ggml_backend_dev_host_buffer_type(dev)) {
            assert(!llama_exact_buft_invariant(host));
        }

        checked_gpu = checked_gpu || invariant;
    }

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
