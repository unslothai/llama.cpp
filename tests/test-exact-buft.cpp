// [TAG_EXACT_CONCURRENCY] A host buffer is refused: its ops move to the GPU only once the batch is wide enough, so results depend on the step width.

#include "ggml-backend.h"
#include "../ggml/src/ggml-backend-impl.h"

#include "../src/llama-impl.h"

#include <cstdio>
#include <cstring>

#undef NDEBUG
#include <cassert>

int main() {
    ggml_backend_load_all();

    assert(!llama_exact_buft_invariant(nullptr));

    assert(!llama_exact_buft_invariant(ggml_backend_cpu_buffer_type()));

    bool checked_gpu = false;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);

        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU && ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_IGPU) {
            continue;
        }

        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);

        const bool invariant = llama_exact_backend_reg(reg);

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

    // a CUDA-named registry from before these kernels carries the name but not the extension: it is not exact
    static void * the_extension = (void *) &the_extension;
    struct fake {
        static const char * name(ggml_backend_reg_t) { return "CUDA"; }
        static size_t       count(ggml_backend_reg_t) { return 0; }
        static ggml_backend_dev_t dev(ggml_backend_reg_t, size_t) { return nullptr; }
        static void * none(ggml_backend_reg_t, const char *) { return nullptr; }
        static void * with(ggml_backend_reg_t, const char * n) {
            return strcmp(n, "ggml_backend_cuda_set_exact_decode_width") == 0 ? the_extension : nullptr;
        }
    };
    ggml_backend_reg old_cuda = { GGML_BACKEND_API_VERSION, { fake::name, fake::count, fake::dev, fake::none }, nullptr };
    ggml_backend_reg new_cuda = { GGML_BACKEND_API_VERSION, { fake::name, fake::count, fake::dev, fake::with }, nullptr };
    assert(!llama_exact_backend_reg(&old_cuda));
    assert(llama_exact_backend_reg(&new_cuda));
    assert(!llama_exact_backend_reg(nullptr));

    printf("%s: all tests passed%s\n", __func__,
            checked_gpu ? "" : " (no batch-invariant device here, the positive case was not exercised)");

    return 0;
}
