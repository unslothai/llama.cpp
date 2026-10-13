// [TAG_EXACT_CONCURRENCY] The context refuses a non-layer weight (output_norm) kept on host by an override, and accepts the same model fully offloaded.
// Needs a batch-invariant GPU backend; without one the test reports what it skipped and passes.

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Windows has no setenv
static void set_env_default(const char * name, const char * value) {
    if (getenv(name)) {
        return;
    }
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

static bool context_ok(const char * path, const llama_model_tensor_buft_override * overrides) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers          = 999;
    mparams.tensor_buft_overrides = overrides;

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) {
        fprintf(stderr, "failed to load %s\n", path);
        exit(1);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = 2048;
    cparams.n_seq_max       = 2;
    cparams.kv_unified      = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cparams);
    const bool ok = ctx != nullptr;

    llama_free(ctx);
    llama_model_free(model);

    return ok;
}

int main(int argc, char ** argv) {
    // read before the model is loaded: latched on first use
    set_env_default("LLAMA_EXACT_CONCURRENCY", "1");

    const char * path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[i + 1];
        }
    }

    if (path == nullptr) {
        fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]);
        return 1;
    }

    ggml_backend_load_all();
    llama_backend_init();

    if (!context_ok(path, nullptr)) {
        printf("%s : skipped, this build and model cannot run exact concurrency\n", __func__);
        llama_backend_free();
        return 0;
    }

    const llama_model_tensor_buft_override host_norm[] = {
        { "^output_norm\\.weight$", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };

    if (context_ok(path, host_norm)) {
        fprintf(stderr, "%s : FAIL, output_norm.weight on the host passed the exact concurrency weight check\n", __func__);
        llama_backend_free();
        return 1;
    }

    printf("%s : all tests passed\n", __func__);
    llama_backend_free();
    return 0;
}
