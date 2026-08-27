// whole-context state save/restore while more than one sequence is live.
//
// a non-unified kv cache keeps one stream per sequence and the whole-context restore walks the
// streams in turn, so a restore step that resets the cache as a whole undoes every stream that
// came before it and only the last one survives the round trip. the same shape of mistake is
// easy to reintroduce, so the check here is on the content of each sequence rather than on the
// call merely returning the right byte count.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

static constexpr int    N_SEQ    = 4;
static constexpr int    N_PROMPT = 12;
static constexpr size_t N_CMP    = 64; // logits compared per sequence

static int g_failures = 0;

static void check(bool ok, const std::string & what) {
    if (!ok) {
        fprintf(stderr, "  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// distinct, deterministic token stream per sequence
static llama_token tok_of(int seq, int pos, int n_vocab) {
    return (llama_token) ((13*(unsigned) pos + 101*(unsigned) seq + 3) % (unsigned) n_vocab);
}

// decode seq's prompt, then one extra token, and return the logits of that extra token
static bool decode_seq(llama_context * ctx, int seq, int n_vocab, int pos0, int n_tokens, std::vector<float> * logits_out) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);

    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tok_of(seq, pos0 + i, n_vocab), pos0 + i, { seq }, i + 1 == n_tokens);
    }

    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);

    if (!ok) {
        return false;
    }

    if (logits_out) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        if (logits == nullptr) {
            return false;
        }
        logits_out->assign(logits, logits + std::min((size_t) n_vocab, N_CMP));
    }

    return true;
}

static bool logits_agree(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return false;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i]) || std::isnan(b[i])) {
            return false;
        }
        // same model, same backend and the same batch shape either side of the restore, so the
        // two are expected to agree exactly; the tolerance only guards against a backend that
        // reorders reductions between calls
        if (std::fabs(a[i] - b[i]) > 1e-3f*(1.0f + std::fabs(a[i]))) {
            return false;
        }
    }

    return true;
}

static void run_case(llama_model * model, const common_params & params, bool kv_unified) {
    const char * label = kv_unified ? "kv-unified" : "kv-per-seq";
    fprintf(stderr, "\n== %s ==\n", label);

    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = N_SEQ;
    cparams.n_ctx      = 256*N_SEQ;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 256;
    cparams.kv_unified = kv_unified;

    llama_context * ctx_a = llama_init_from_model(model, cparams);
    if (ctx_a == nullptr) {
        fprintf(stderr, "  FAIL: %s: could not create the source context\n", label);
        ++g_failures;
        return;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // fill every sequence with its own tokens
    for (int s = 0; s < N_SEQ; ++s) {
        if (!decode_seq(ctx_a, s, n_vocab, 0, N_PROMPT, nullptr)) {
            fprintf(stderr, "  FAIL: %s: prompt decode failed on seq %d\n", label, s);
            ++g_failures;
            llama_free(ctx_a);
            return;
        }
    }

    // save the whole context, with all N_SEQ sequences in it
    std::vector<uint8_t> state(llama_state_get_size(ctx_a));
    const size_t n_written = llama_state_get_data(ctx_a, state.data(), state.size());
    check(n_written > 0 && n_written <= state.size(), std::string(label) + ": llama_state_get_data wrote nothing");
    state.resize(n_written);

    // printed so that two builds can be compared byte for byte from the outside
    {
        uint64_t h = 1469598103934665603ull;
        for (uint8_t b : state) {
            h = (h ^ b)*1099511628211ull;
        }
        fprintf(stderr, "  state blob: %zu bytes, fnv1a %016llx\n", state.size(), (unsigned long long) h);
    }

    // reference continuation, taken after the save so it reflects the saved state
    std::vector<std::vector<float>> ref(N_SEQ);
    for (int s = 0; s < N_SEQ; ++s) {
        if (!decode_seq(ctx_a, s, n_vocab, N_PROMPT, 1, &ref[s])) {
            fprintf(stderr, "  FAIL: %s: reference decode failed on seq %d\n", label, s);
            ++g_failures;
            llama_free(ctx_a);
            return;
        }
    }

    llama_free(ctx_a);

    // restore into a context that has never seen any of it
    llama_context * ctx_b = llama_init_from_model(model, cparams);
    if (ctx_b == nullptr) {
        fprintf(stderr, "  FAIL: %s: could not create the destination context\n", label);
        ++g_failures;
        return;
    }

    const size_t n_read = llama_state_set_data(ctx_b, state.data(), state.size());
    check(n_read == state.size(), std::string(label) + ": llama_state_set_data consumed the wrong size");

    llama_memory_t mem = llama_get_memory(ctx_b);

    for (int s = 0; s < N_SEQ; ++s) {
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, s);
        check(pos_max == N_PROMPT - 1,
                std::string(label) + ": seq " + std::to_string(s) + " has pos_max " + std::to_string(pos_max) +
                ", expected " + std::to_string(N_PROMPT - 1));
    }

    for (int s = 0; s < N_SEQ; ++s) {
        std::vector<float> got;
        if (!decode_seq(ctx_b, s, n_vocab, N_PROMPT, 1, &got)) {
            fprintf(stderr, "  FAIL: %s: continuation decode failed on seq %d\n", label, s);
            ++g_failures;
            continue;
        }

        check(logits_agree(ref[s], got),
                std::string(label) + ": seq " + std::to_string(s) + " does not continue as it did before the save");
    }

    llama_free(ctx_b);
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx     = 256*N_SEQ;
    params.n_parallel = N_SEQ;
    params.n_predict = 0;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = common_model_params_to_llama(params);

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    run_case(model, params, true);
    run_case(model, params, false);

    llama_model_free(model);
    llama_backend_free();

    if (g_failures > 0) {
        fprintf(stderr, "\n%d check(s) failed\n", g_failures);
        return 1;
    }

    fprintf(stderr, "\nall checks passed\n");

    return 0;
}
