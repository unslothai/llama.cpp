// [TAG_EXACT_CONCURRENCY] llama_set_exact_decode_tokens() checks the explicit column bound before any context exists: a figure past
// it is refused and leaves the stored one, which is never lowered, so nothing later is refused because of it.
// With -m MODEL it checks instead that the figure cannot outgrow a live exact context's batch geometry, which that context's
// constructor enforces (n_batch >= n_ubatch + n_seq_max * figure): a vocab-only load, as the geometry needs no weights.

#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

static int check(const char * what, bool ok) {
    printf("%-72s %s\n", what, ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static llama_context * exact_context(llama_model * model, uint32_t n_batch, uint32_t n_ubatch, uint32_t n_seq_max) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_batch    = n_batch;
    cparams.n_ubatch   = n_ubatch;
    cparams.n_seq_max  = n_seq_max;
    cparams.kv_unified = true;
    return llama_init_from_model(model, cparams);
}

static int live_contexts(const char * path) {
    // a fresh process: the figure starts at 1 and only grows
    set_env("LLAMA_EXACT_CONCURRENCY", "1");

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;

    llama_model * model = llama_model_load_from_file(path, mparams);
    if (model == nullptr) {
        printf("could not load %s\n", path);
        return 1;
    }

    int bad = 0;

    // valid at a figure of 1: 128 + 2 * 1 <= 132
    llama_context * tight = exact_context(model, 132, 128, 2);
    bad += check("a 132/128/2 context at a figure of 1: created", tight != nullptr && llama_exact_decode_tokens() == 1);
    bad += check("8 beside it (needs a batch of 144): refused, 1 kept", !llama_set_exact_decode_tokens(8) && llama_exact_decode_tokens() == 1);
    bad += check("2 beside it (needs exactly 132): accepted", llama_set_exact_decode_tokens(2) && llama_exact_decode_tokens() == 2);
    bad += check("3 beside it: refused, 2 kept", !llama_set_exact_decode_tokens(3) && llama_exact_decode_tokens() == 2);

    llama_context * single = exact_context(model, 132, 128, 1);
    bad += check("a one-sequence context: created", single != nullptr);

    llama_free(tight);
    bad += check("8 once the tight context is freed, a one-sequence one alive: accepted", llama_set_exact_decode_tokens(8) && llama_exact_decode_tokens() == 8);
    llama_free(single);

    llama_context * again = exact_context(model, 132, 128, 2);
    bad += check("an identical 132/128/2 context at a figure of 8: refused", again == nullptr);
    llama_free(again);

    llama_context * wide = exact_context(model, 144, 128, 2);
    bad += check("a 144/128/2 context at a figure of 8: created", wide != nullptr);
    bad += check("9 beside it: refused, 8 kept", !llama_set_exact_decode_tokens(9) && llama_exact_decode_tokens() == 8);
    llama_free(wide);

    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", bad == 0 ? "ok" : "FAIL");
    return bad == 0 ? 0 : 1;
}

int main(int argc, char ** argv) {
    if (argc == 3 && strcmp(argv[1], "-m") == 0) {
        return live_contexts(argv[2]);
    }

#ifdef _WIN32
    _putenv_s("GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "8");
#else
    setenv("GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "8", 1);
#endif

    int bad = 0;

    const uint32_t tokens0 = llama_exact_decode_tokens();
    const uint32_t width0  = llama_exact_decode_width();

    bad += check("9 tokens per sequence, bound 8, no context yet: refused", !llama_set_exact_decode_tokens(9));
    bad += check("  the stored figure is unchanged", llama_exact_decode_tokens() == tokens0);
    bad += check("  the reported width is unchanged", llama_exact_decode_width() == width0);
    bad += check("8 tokens per sequence: accepted", llama_set_exact_decode_tokens(8));
    bad += check("  stored as 8, width 8", llama_exact_decode_tokens() == 8 && llama_exact_decode_width() == 8);
    bad += check("9 after 8: refused, 8 kept", !llama_set_exact_decode_tokens(9) && llama_exact_decode_tokens() == 8);
    bad += check("1 after 8: accepted, never lowered", llama_set_exact_decode_tokens(1) && llama_exact_decode_tokens() == 8);
    bad += check("0 is 1: accepted, 8 kept", llama_set_exact_decode_tokens(0) && llama_exact_decode_tokens() == 8);
    bad += check("a width of 9 past the bound: refused", !llama_set_exact_decode_width(9) && llama_exact_decode_width() == 8);
    bad += check("a width of 8: accepted", llama_set_exact_decode_width(8));

    printf("%s\n", bad == 0 ? "ok" : "FAIL");
    return bad == 0 ? 0 : 1;
}
