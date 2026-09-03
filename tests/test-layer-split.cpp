// Numerical-equivalence test for layer-split pipeline parallelism.
//
// Runs one prompt three ways and compares the resulting logits:
//   (1) the whole model in a single context                          -> reference
//   (2) stage A = layers [0, k), emitting the raw residual stream
//       stage B = layers [k, n_layer), ingesting it via llama_batch.embd
//
// Stage A must produce the residual entering layer k BEFORE output_norm; stage B must
// consume it as its layer-k input. If the split is correct the two logit vectors are
// identical up to floating-point reassociation, and the argmax token must match exactly.
//
// usage: test-layer-split <model.gguf> [split_layer] [prompt]

#include "llama.h"
#include "llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void set_stage(int beg, int end) {
    char b[32], e[32];
    snprintf(b, sizeof(b), "%d", beg);
    snprintf(e, sizeof(e), "%d", end);
    setenv("LLAMA_PP_IL_BEG", b, 1);
    setenv("LLAMA_PP_IL_END", e, 1);
}

// Build a batch that feeds hidden states through llama_batch.embd.
//
// Subtlety that costs a day if missed: for an M-RoPE model (llama_hparams::n_pos_per_embd() == 4,
// i.e. any model with rope.dimension_sections, which includes Qwen3.5), llama_batch_allocr reads
// positions as FOUR sections of n_tokens when the batch carries embeddings rather than tokens:
//
//   src_off = batch.token ? 0 : j*batch.n_tokens;          // src/llama-batch.cpp:785
//   udata->pos[j*n_tokens + i] = batch.pos[src_off + idxs[i]];
//
// The token path broadcasts one position across all four sections for you; the embd path does not,
// because there it assumes image embeddings that carry genuine 3D positions. Supplying only
// n_tokens positions therefore reads past the end of the array for j = 1,2,3, producing garbage
// RoPE positions -- nondeterministic logits that look exactly like a broken layer split.
static llama_batch make_embd_batch(int n_tok, int n_embd, int n_pos_per_embd, const float * hidden,
                                   bool all_logits) {
    llama_batch b = llama_batch_init(n_tok * n_pos_per_embd, n_embd, 1);
    memcpy(b.embd, hidden, (size_t) n_tok * n_embd * sizeof(float));
    for (int j = 0; j < n_pos_per_embd; ++j) {
        for (int i = 0; i < n_tok; ++i) {
            b.pos[j * n_tok + i] = i;   // text semantics: same position in every RoPE section
        }
    }
    for (int i = 0; i < n_tok; ++i) {
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0;
        b.logits[i] = all_logits ? 1 : (i == n_tok - 1);
    }
    b.n_tokens = n_tok;
    return b;
}

static void clear_stage() {
    unsetenv("LLAMA_PP_IL_BEG");
    unsetenv("LLAMA_PP_IL_END");
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [split_layer] [prompt]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    const int  split_layer_arg   = argc > 2 ? atoi(argv[2]) : -1;
    const std::string prompt     = argc > 3 ? argv[3] : "The capital of France is";

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 999;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab  = llama_model_get_vocab(model);
    const int n_vocab  = llama_vocab_n_tokens(vocab);
    const int n_embd   = llama_model_n_embd(model);
    const int n_layer  = llama_model_n_layer(model);
    const int split    = split_layer_arg > 0 ? split_layer_arg : n_layer / 2;

    const enum llama_rope_type rt = llama_model_rope_type(model);
    const int n_pos_per_embd = (rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    printf("model: n_layer=%d n_embd=%d n_vocab=%d, split at layer %d, n_pos_per_embd=%d\n",
           n_layer, n_embd, n_vocab, split, n_pos_per_embd);

    // tokenize
    std::vector<llama_token> tokens(prompt.size() + 8);
    int n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                               tokens.data(), (int) tokens.size(), true, false);
    if (n_tok < 0) { tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(),
                               tokens.data(), (int) tokens.size(), true, false); }
    tokens.resize(n_tok);
    printf("prompt: \"%s\" -> %d tokens\n", prompt.c_str(), n_tok);

    const int n_ctx_use = 512;

    // ---------------- reference: whole model ----------------
    std::vector<float> ref_logits(n_vocab);
    {
        clear_stage();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "ref ctx failed\n"); return 1; }

        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tok - 1);
        }
        batch.n_tokens = n_tok;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "ref decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(ctx, n_tok - 1);
        memcpy(ref_logits.data(), lg, n_vocab * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }

    auto full_layer_inp = [&](int lid, std::vector<float> & out) {
        clear_stage();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_set_embeddings_layer_inp(ctx, (uint32_t) lid, true);
        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tok - 1);
        }
        batch.n_tokens = n_tok;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "full_layer_inp decode failed\n"); exit(1); }
        out.assign((size_t) n_tok * n_embd, 0.0f);
        memcpy(out.data(), llama_get_embeddings_layer_inp(ctx, (uint32_t) lid), out.size() * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    };

    auto cmp = [&](const char * what, const std::vector<float> & a, const std::vector<float> & b) {
        double m = 0.0, sq = 0.0;
        for (size_t i = 0; i < a.size(); ++i) { double d = (double) a[i] - b[i]; m = std::max(m, std::fabs(d)); sq += d*d; }
        printf("  %-52s max=%.6g rmse=%.6g\n", what, m, std::sqrt(sq / a.size()));
        return m;
    };

    // ---------------- independent reference for the hidden states ----------------
    // The fork already exposes the residual stream entering any layer (used by common/speculative.cpp
    // for EAGLE3 drafting). Extracting t_layer_inp[split] from the FULL model gives a ground truth
    // for what stage A must emit, which bisects a mismatch into "stage A wrong" vs "stage B wrong".
    std::vector<float> hidden_ref((size_t) n_tok * n_embd);
    {
        clear_stage();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "hidden-ref ctx failed\n"); return 1; }
        llama_set_embeddings_layer_inp(ctx, (uint32_t) split, true);

        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == n_tok - 1);
        }
        batch.n_tokens = n_tok;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "hidden-ref decode failed\n"); return 1; }
        const float * h = llama_get_embeddings_layer_inp(ctx, (uint32_t) split);
        if (!h) { fprintf(stderr, "hidden-ref: no layer input\n"); return 1; }
        memcpy(hidden_ref.data(), h, hidden_ref.size() * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }

    // ---------------- stage A: layers [0, split), emit hidden states ----------------
    std::vector<float> hidden((size_t) n_tok * n_embd);
    {
        set_stage(0, split);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        cp.embeddings   = true;                      // required: emit t_embd
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;   // per-token, not pooled
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "stage A ctx failed\n"); return 1; }

        llama_batch batch = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; ++i) {
            batch.token[i] = tokens[i]; batch.pos[i] = i;
            batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;                     // all tokens' states are needed downstream
        }
        batch.n_tokens = n_tok;
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "stage A decode failed\n"); return 1; }

        const float * e = llama_get_embeddings(ctx);
        if (!e) { fprintf(stderr, "stage A produced no embeddings\n"); return 1; }
        memcpy(hidden.data(), e, hidden.size() * sizeof(float));

        double s = 0.0; for (float v : hidden) s += (double) v * v;
        printf("stage A: emitted %d x %d hidden states, rms=%.6f\n",
               n_tok, n_embd, std::sqrt(s / hidden.size()));

        double hmax = 0.0, hsq = 0.0;
        for (size_t i = 0; i < hidden.size(); ++i) {
            const double d = (double) hidden[i] - hidden_ref[i];
            hmax = std::max(hmax, std::fabs(d));
            hsq += d * d;
        }
        printf("stage A vs t_layer_inp[%d] of the full model: max|dh|=%.6g rmse=%.6g\n",
               split, hmax, std::sqrt(hsq / hidden.size()));
        llama_batch_free(batch);
        llama_free(ctx);
    }

    // ---------------- stage B: layers [split, n_layer), ingest hidden states ----------------
    std::vector<float> split_logits(n_vocab);
    {
        set_stage(split, n_layer);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "stage B ctx failed\n"); return 1; }

        llama_batch batch = make_embd_batch(n_tok, n_embd, n_pos_per_embd, hidden.data(), false);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "stage B decode failed\n"); return 1; }
        const float * lg = llama_get_logits_ith(ctx, n_tok - 1);
        memcpy(split_logits.data(), lg, n_vocab * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }

    // ---------------- stage B, run a second time in a fresh context ----------------
    // Same weights, same input hidden states. Any difference between the two is nondeterminism
    // inside stage B itself rather than a systematic error in the split.
    std::vector<float> split_logits2(n_vocab);
    {
        set_stage(split, n_layer);
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch batch = make_embd_batch(n_tok, n_embd, n_pos_per_embd, hidden.data(), false);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "stage B2 decode failed\n"); return 1; }
        memcpy(split_logits2.data(), llama_get_logits_ith(ctx, n_tok - 1), n_vocab * sizeof(float));
        llama_batch_free(batch);
        llama_free(ctx);
    }
    {
        double m = 0.0;
        for (int i = 0; i < n_vocab; ++i) m = std::max(m, std::fabs((double) split_logits[i] - split_logits2[i]));
        printf("stage B run-to-run max|dlogit| = %.6g  %s\n", m,
               m == 0.0 ? "(deterministic)" : "(NONDETERMINISTIC - stage B reads uninitialised state)");
    }

    // ---------------- control: FULL layer range, fed through batch.embd ----------------
    // Extract the model's own layer-0 input (the token embeddings) and feed them straight back in
    // as embeddings with il_beg = 0, il_end = n_layer. This isolates the two suspects: if this
    // matches the reference, llama_batch.embd ingestion is sound and any error is caused by
    // slicing the layer range; if it does not, the embd input path itself is at fault.
    {
        std::vector<float> embd0;
        full_layer_inp(0, embd0);

        clear_stage();
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
        llama_context * ctx = llama_init_from_model(model, cp);
        llama_batch batch = make_embd_batch(n_tok, n_embd, n_pos_per_embd, embd0.data(), false);
        if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "embd-full decode failed\n"); return 1; }
        std::vector<float> lg(n_vocab);
        memcpy(lg.data(), llama_get_logits_ith(ctx, n_tok - 1), n_vocab * sizeof(float));
        cmp("CONTROL full range via batch.embd vs reference", lg, ref_logits);
        llama_batch_free(batch);
        llama_free(ctx);
    }

    // ---------------- per-layer bisect through stage B ----------------
    // Feed the full model's true residual into a one-layer stage [L, L+1) and compare the residual
    // it emits against the full model's residual entering L+1. This localises corruption to a
    // single layer instead of blaming the whole second half.
    if (getenv("BISECT")) {
        printf("\nper-layer bisect (inject true residual at L, compare output at L+1):\n");
        for (int L = split; L < n_layer - 1; ++L) {
            std::vector<float> in_L, out_ref;
            full_layer_inp(L,     in_L);
            full_layer_inp(L + 1, out_ref);

            set_stage(L, L + 1);
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx = n_ctx_use; cp.n_batch = n_ctx_use; cp.n_ubatch = n_ctx_use;
            cp.embeddings = true; cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
            llama_context * ctx = llama_init_from_model(model, cp);
            llama_batch batch = make_embd_batch(n_tok, n_embd, n_pos_per_embd, in_L.data(), true);
            if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "bisect decode failed at L=%d\n", L); return 1; }
            std::vector<float> got((size_t) n_tok * n_embd);
            memcpy(got.data(), llama_get_embeddings(ctx), got.size() * sizeof(float));
            char lbl[64]; snprintf(lbl, sizeof(lbl), "layer %d (%s)", L, "single-layer stage");
            cmp(lbl, got, out_ref);
            llama_batch_free(batch);
            llama_free(ctx);
        }
        printf("\n");
    }

    // ---------------- compare ----------------
    int   ref_arg = 0, spl_arg = 0;
    double max_abs = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        if (ref_logits[i]   > ref_logits[ref_arg]) ref_arg = i;
        if (split_logits[i] > split_logits[spl_arg]) spl_arg = i;
        const double d = (double) ref_logits[i] - split_logits[i];
        max_abs = std::max(max_abs, std::fabs(d));
        sum_sq += d * d;
    }
    const double rmse = std::sqrt(sum_sq / n_vocab);

    char buf_r[128] = {0}, buf_s[128] = {0};
    llama_token_to_piece(vocab, ref_arg, buf_r, sizeof(buf_r), 0, true);
    llama_token_to_piece(vocab, spl_arg, buf_s, sizeof(buf_s), 0, true);

    printf("\nreference argmax : %6d \"%s\"  logit=%.5f\n", ref_arg, buf_r, ref_logits[ref_arg]);
    printf("split     argmax : %6d \"%s\"  logit=%.5f\n", spl_arg, buf_s, split_logits[spl_arg]);
    printf("max |dlogit| = %.6f   rmse = %.6f\n", max_abs, rmse);

    const bool ok = (ref_arg == spl_arg) && (max_abs < 0.05);
    printf("\n%s\n", ok ? "PASS: layer split is numerically equivalent"
                        : "FAIL: layer split diverges from the reference");

    llama_model_free(model);
    llama_backend_free();
    return ok ? 0 : 1;
}
