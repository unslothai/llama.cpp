#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"
#include "llama-cpp.h"
#include "common.h"
#include "sampling.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

struct test_args {
    std::string model;
    std::string test;
    std::string device = "auto";
};

struct test_params {
    llama_model_ptr model;
};

static llama_model_ptr load_model(const test_args & args) {
    auto mparams = llama_model_default_params();

    ggml_backend_dev_t devs[2] = { nullptr, nullptr };

    if (args.device != "auto") {
        if (args.device == "gpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);

            if (devs[0] == nullptr) {
                fprintf(stderr, "Error: GPU requested but not available\n");
                return nullptr;
            }

            mparams.n_gpu_layers = 999;
        } else if (args.device == "cpu") {
            devs[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);

            mparams.n_gpu_layers = 0;
        } else {
            fprintf(stderr, "Error: invalid device '%s'\n", args.device.c_str());
            return nullptr;
        }

        mparams.devices = devs;

        fprintf(stderr, "Using device: %s\n", ggml_backend_dev_name(devs[0]));
    }

    llama_model_ptr res;

    res.reset(llama_model_load_from_file(args.model.c_str(), mparams));

    if (!res) {
        fprintf(stderr, "Warning: failed to load model '%s', skipping test\n", args.model.c_str());
        return nullptr;
    }

    return res;
}

struct test_context {
    llama_context_ptr ctx;

    int n_vocab = 0;

    const llama_vocab * vocab = nullptr;

    std::unordered_map<llama_seq_id, int32_t> seq_positions;
    std::unordered_map<llama_seq_id, int32_t> last_batch_info;

    test_context(
            const test_params & params,
            std::vector<llama_sampler_seq_config> & configs,
            int32_t n_seq_max = -1,
            uint32_t n_outputs_max = 0,
            uint32_t n_ubatch = 0,
            uint32_t n_outputs_max_per_seq = 1) {
        auto * model = params.model.get();

        GGML_ASSERT(model);
        GGML_ASSERT(!ctx);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;
        cparams.n_batch = 512;
        if (n_ubatch > 0) {
            cparams.n_ubatch = n_ubatch;
        }
        cparams.n_outputs_max = n_outputs_max;
        cparams.n_outputs_max_per_seq = n_outputs_max_per_seq;
        cparams.samplers = configs.data();
        cparams.n_samplers = configs.size();
        cparams.kv_unified = true;

        // If n_seq_max is not specified, calculate it from configs
        if (n_seq_max < 0) {
            int32_t max_seq_id = 0;
            for (const auto & config : configs) {
                max_seq_id = std::max(config.seq_id, max_seq_id);
            }
            cparams.n_seq_max = max_seq_id + 1;
        } else {
            cparams.n_seq_max = n_seq_max;
        }

        ctx.reset(llama_init_from_model(model, cparams));
        if (!ctx) {
            throw std::runtime_error("failed to create context");
        }

        vocab = llama_model_get_vocab(model);
        n_vocab = llama_vocab_n_tokens(vocab);
    }

    bool decode(const std::map<llama_seq_id, std::string> & prompts) {
        GGML_ASSERT(ctx);

        last_batch_info.clear();
        common_batch batch(ctx.get());

        for (const auto & [seq_id, prompt] : prompts) {
            std::vector<llama_token> tokens;
            tokens.push_back(llama_vocab_bos(vocab));

            std::vector<llama_token> prompt_tokens(32);
            int n_tokens = llama_tokenize(vocab, prompt.c_str(), prompt.length(),
                                           prompt_tokens.data(), prompt_tokens.size(),
                                           false, false);
            if (n_tokens < 0) {
                fprintf(stderr, "Warning: tokenization failed for seq_id %d\n", seq_id);
                return false;
            }

            for (int i = 0; i < n_tokens; i++) {
                tokens.push_back(prompt_tokens[i]);
            }

            if (seq_positions.find(seq_id) == seq_positions.end()) {
                seq_positions[seq_id] = 0;
            }

            int32_t start_pos = seq_positions[seq_id];
            for (size_t i = 0; i < tokens.size(); i++) {
                batch.add(tokens[i], start_pos + i, seq_id, i == tokens.size() - 1);
            }

            seq_positions[seq_id] = start_pos + tokens.size();
        }


        printf("Batch contents:\n");
        printf("n_tokens: %d\n", batch.size());
        for (int i = 0; i < batch.size(); i++) {
            const auto & t = batch.tokens[i];
            printf("token[%d]: tok=%-5d, pos=%d, seq_id=%d, logits=%d\n", i, t.id, t.pos[0], t.seq_id, t.output);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed\n");
            return false;
        }

        update_batch_info(batch);
        return true;
    }

    int32_t idx_for_seq(llama_seq_id seq_id) {
        auto it = last_batch_info.find(seq_id);
        if (it == last_batch_info.end()) {
            fprintf(stderr, "Error: no batch index found for seq_id %d\n", seq_id);
            return -1;
        }
        return it->second;
    }

    // build mapping from seq id to batch token idx
    void update_batch_info(const common_batch & batch) {
        last_batch_info.clear();
        for (int i = 0; i < batch.size(); i++) {
            if (batch.tokens[i].output) {
                llama_seq_id cur_seq = batch.tokens[i].seq_id;
                last_batch_info[cur_seq] = i;
            }
        }
    }

    bool decode_token(llama_token token, llama_seq_id seq_id = 0) {
        GGML_ASSERT(ctx);

        common_batch batch(ctx.get());
        int32_t pos = seq_positions[seq_id];
        batch.add(token, pos, seq_id, true);

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for token %d in seq %d\n", token, seq_id);
            return false;
        }

        update_batch_info(batch);

        seq_positions[seq_id]++;

        return true;
    }

    bool decode_tokens(const std::map<llama_seq_id, llama_token> & seq_tokens) {
        GGML_ASSERT(ctx);

        common_batch batch(ctx.get());

        for (const auto & [seq_id, token] : seq_tokens) {
            int32_t pos = seq_positions[seq_id];
            batch.add(token, pos, seq_id, true);
        }

        if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "Warning: llama_decode failed for batch tokens\n");
            return false;
        }

        for (const auto & [seq_id, _] : seq_tokens) {
            seq_positions[seq_id]++;
        }

        update_batch_info(batch);

        return true;
    }

    std::string token_to_piece(llama_token token, bool special) const {
        std::string piece;
        piece.resize(piece.capacity());  // using string internal cache, 15 bytes + '\n'
        const int n_chars = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
        if (n_chars < 0) {
            piece.resize(-n_chars);
            int check = llama_token_to_piece(vocab, token, &piece[0], piece.size(), 0, special);
            GGML_ASSERT(check == -n_chars);
        } else {
            piece.resize(n_chars);
        }

        return piece;
    }
};

struct test_single_output_backend_sampler {
    bool backend_initialized = false;
    uint32_t backend_outputs_max_per_seq = 0;
    int backend_apply_count = 0;
    int apply_count = 0;
};

static const char * test_single_output_backend_sampler_name(const llama_sampler * /*smpl*/) {
    return "single-output-backend";
}

static void test_single_output_backend_sampler_apply(
        llama_sampler * smpl, llama_token_data_array * /*cur_p*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->apply_count++;
}

static void test_single_output_backend_sampler_free(llama_sampler * smpl) {
    delete (test_single_output_backend_sampler *) smpl->ctx;
}

static bool test_single_output_backend_sampler_backend_init(
        llama_sampler * smpl, ggml_backend_buffer_type_t /*buft*/, uint32_t n_outputs_max_per_seq) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_outputs_max_per_seq = n_outputs_max_per_seq;
    if (n_outputs_max_per_seq > 1) {
        return false;
    }
    ctx->backend_initialized = true;
    return true;
}

static void test_single_output_backend_sampler_backend_apply(
        llama_sampler * smpl, ggml_context * /*ctx*/, ggml_cgraph * /*gf*/, llama_sampler_data * /*data*/) {
    auto * ctx = (test_single_output_backend_sampler *) smpl->ctx;
    ctx->backend_apply_count++;
}

static llama_sampler_i test_single_output_backend_sampler_i = {
    /* .name              = */ test_single_output_backend_sampler_name,
    /* .accept            = */ nullptr,
    /* .apply             = */ test_single_output_backend_sampler_apply,
    /* .reset             = */ nullptr,
    /* .clone             = */ nullptr,
    /* .free              = */ test_single_output_backend_sampler_free,
    /* .backend_init      = */ test_single_output_backend_sampler_backend_init,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ test_single_output_backend_sampler_backend_apply,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static llama_sampler * test_single_output_backend_sampler_init(
        test_single_output_backend_sampler ** sampler_ctx) {
    auto * ctx = new test_single_output_backend_sampler;
    *sampler_ctx = ctx;
    return llama_sampler_init(&test_single_output_backend_sampler_i, ctx);
}

static void test_greedy(const test_params & params) {
    const int seq_id = 0;

    struct llama_sampler_chain_params backend_sampler_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_sampler_params));

    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_greedy());
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("greedy sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), loop_idx);
        printf("Generation step %d: token id:%d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }
}


static void test_greedy_filtered_common(const test_params & params) {
    auto check = [&](common_params_sampling sp, bool greedy) {
        common_sampler_ptr sampler(common_sampler_init(params.model.get(), sp));
        auto * chain = common_sampler_get(sampler.get());
        auto * last = llama_sampler_chain_get(chain, llama_sampler_chain_n(chain) - 1);
        GGML_ASSERT(strcmp(llama_sampler_name(last), greedy ? "greedy" : "dist") == 0);

        llama_token_data tokens[] = {{0, -2.0f, 0.0f}, {1, -1.0f, 0.0f}, {2, 0.0f, 0.0f}, {3, 1.0f, 0.0f}};
        llama_token_data_array candidates = {tokens, 4, -1, false};
        llama_sampler_apply(chain, &candidates);
        GGML_ASSERT(candidates.selected >= 0);
        if (greedy || sp.n_probs > 0) {
            GGML_ASSERT(candidates.data[candidates.selected].id == 3);
        }
        if (sp.n_probs > 0) {
            GGML_ASSERT(candidates.data[candidates.selected].p == 1.0f);
        }
        if (sp.dynatemp_range > 0.0f) {
            int positive = 0;
            for (size_t i = 0; i < candidates.size; ++i) {
                positive += candidates.data[i].p > 0.0f;
            }
            GGML_ASSERT(positive > 1);
        }
    };

    common_params_sampling sp;
    sp.temp = 0.0f;
    sp.samplers = {COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TEMPERATURE};
    for (bool backend : {false, true}) {
        sp.backend_sampling = backend;
        check(sp, true);
        auto probabilities = sp;
        probabilities.n_probs = 4;
        check(probabilities, false);
        auto dynamic = sp;
        dynamic.dynatemp_range = 1.0f;
        check(dynamic, false);
    }
    auto grammar = sp;
    grammar.grammar = {COMMON_GRAMMAR_TYPE_USER, "root ::= [a-z]+"};
    check(grammar, true);
    auto budget = sp;
    budget.reasoning_budget_start = {1};
    budget.reasoning_budget_end = {{2}};
    budget.reasoning_budget_tokens = 1;
    check(budget, true);
    sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K};
    check(sp, false);
    sp.samplers.clear();
    check(sp, false);

    sp.temp = 0.8f;
    sp.samplers = {COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K};
    for (bool backend : {false, true}) {
        sp.backend_sampling = backend;
        sp.top_k = 1;
        check(sp, true);
        auto probabilities = sp;
        probabilities.n_probs = 4;
        check(probabilities, false);
        sp.top_k = 8;
        check(sp, false);
    }
}

static void test_greedy_filtered(const test_params & params) {
    for (int k : {1, 8}) {
        llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        std::vector<llama_sampler_seq_config> configs = {{0, chain.get()}};
        test_context test_ctx(params, configs);
        GGML_ASSERT(test_ctx.decode({{0, "Write a Python function"}}));
        for (int step = 0; step < 4; ++step) {
            const int idx = test_ctx.idx_for_seq(0);
            const auto * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), idx);
            const auto * ids = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), idx);
            GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), idx) == (uint32_t) k);
            GGML_ASSERT(llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), idx) == (uint32_t) k);
            const auto expected = ids[std::max_element(logits, logits + k) - logits];
            GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), idx) == expected);
            GGML_ASSERT(test_ctx.decode_token(expected));
        }
    }
}

static void test_top_k(const test_params & params) {
    const int seq_id = 0;
    const int32_t k = 8;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_k(k));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_logits; ++i) {
        printf("top_k logit[%zu] = %.6f\n", i, logits[i]);
    }

    llama_token * candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), batch_idx);
    for (size_t i = 0; i < n_candidates; ++i) {
        printf("top_k candidate[%zu] = %d : %s\n", i, candidates[i],
               test_ctx.token_to_piece(candidates[i], false).c_str());
    }

    // Sample using CPU sampler for verification that it is possible to do hybrid
    // sampling, first top_k on the backend and then dist on the CPU.
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    GGML_ASSERT(chain->iface->backend_apply != nullptr);

    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));
    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend top-k hybrid sampling test PASSED\n");
}

static void test_temp(const test_params & params) {
    {
        const float temp_0 = 0.8f;
        struct llama_sampler_chain_params backend_chain_params_0 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_0(llama_sampler_chain_init(backend_chain_params_0));
        llama_sampler_chain_add(backend_sampler_chain_0.get(), llama_sampler_init_temp(temp_0));

        const float temp_1 = 0.1f;
        struct llama_sampler_chain_params backend_chain_params_1 = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain_1(llama_sampler_chain_init(backend_chain_params_1));
        llama_sampler_chain_add(backend_sampler_chain_1.get(), llama_sampler_init_temp(temp_1));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { 0, backend_sampler_chain_0.get() },
            { 1, backend_sampler_chain_1.get() }
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{0, "Some where over the"}, {1, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(0);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);

            // Sample from sequence 0 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 0 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }


        // Verify sequence 1
        {
            int32_t batch_idx = test_ctx.idx_for_seq(1);

            // Sample from sequence 1 using CPU sampler
            struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
            llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
            llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

            llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("Sequence 1 sampled token id:%d, string: '%s'\n", token, token_str.c_str());
            GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temperature values.
    auto test_argmax_temp = [&](float temp) {
        printf("\nTesting temperature = %.1f\n", temp);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp(temp));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == 1);
    };

    test_argmax_temp(0.0f);
    test_argmax_temp(-1.0f);

    printf("backend temp sampling test PASSED\n");
}

static void test_temp_ext(const test_params & params) {
    {
        int seq_id = 0;
        const float temp = 0.8f;
        const float delta = 0.5f;
        const float exponent = 1.5f;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once upon a"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        // Verify sequence 0
        {
            int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
            int n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
            GGML_ASSERT(n_logits == test_ctx.n_vocab);
        }
    }

    // lambda for testing non-positive temp/delta/exponent values.
    auto test_argmax_temp = [&](float temp, float delta, float exponent) {
        printf("\nTesting temperature = %.1f, delta = %1.f, exponent = %1.f\n", temp, delta, exponent);

        int seq_id = 0;
        struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
        llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_temp_ext(temp, delta, exponent));

        std::vector<llama_sampler_seq_config> backend_sampler_configs = {
            { seq_id, backend_sampler_chain.get() },
        };

        test_context test_ctx(params, backend_sampler_configs);

        if (!test_ctx.decode({{seq_id, "Once"}})) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

        uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

        // a positive delta is dynamic temperature on the CPU, also for temp <= 0 (llama_sampler_temp_ext_apply)
        if (temp <= 0.0f && delta <= 0.0f) {
            GGML_ASSERT(n_logits == 1);
        } else {
            GGML_ASSERT(n_logits == (uint32_t) test_ctx.n_vocab);
        }
    };

    test_argmax_temp(0.0f,  0.0f, 1.0f); // Greedy (temp=0)
    test_argmax_temp(-1.0f, 0.0f, 2.0f); // Greedy (temp<0)
    test_argmax_temp(0.0f,  0.3f, 1.0f); // Dynamic temperature in [0, 0.3]
    test_argmax_temp(0.8f,  0.0f, 2.0f); // Temperature scaling

    printf("backend temp_ext sampling test PASSED\n");
}

static void test_min_p(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.1;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_min_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
            //printf("min_p logit[%zu] = %.6f\n", i, logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("min-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("min-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        if (!test_ctx.decode_token(token, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("min-p sampling test PASSED\n");
}

static void test_top_p(const test_params & params) {
    const int seq_id = 0;
    const float p = 0.9;
    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_top_p(p, 0));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        return;
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
    uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);

    // Print the logits that are above the min-p threshold
    std::vector<float> filtered_logits;
    for (size_t i = 0; i < n_logits; ++i) {
        if (logits[i] > -1e9f) {
            filtered_logits.push_back(logits[i]);
        }
    }
    GGML_ASSERT(filtered_logits.size() < (size_t) test_ctx.n_vocab);
    GGML_ASSERT(filtered_logits.size() > 0);

    // Sample using CPU sampler for verification to inspect they are reasonable
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));

    llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    const std::string token_str = test_ctx.token_to_piece(token, false);
    printf("top-p cpu sampled token id:%d, string: '%s'\n", token, token_str.c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    // Decode and sample 10 more tokens
    for (int i = 0; i < 10; i++) {
        int32_t loop_idx = test_ctx.idx_for_seq(seq_id);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), loop_idx);
        printf("top-p gen step %d: token id :%5.d, string: %s\n", i, token, test_ctx.token_to_piece(token, false).c_str());
        test_ctx.decode_token(token, 0);
    }

    printf("top-p sampling test PASSED\n");
}

static void test_multi_sequence(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_greedy());

    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_greedy());

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1
    {
        int32_t batch_idx= test_ctx.idx_for_seq(1);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Generate tokens for each sequence
    printf("\nMulti-sequence generation:\n");
    for (int step = 0; step < 4; step++) {
        std::map<llama_seq_id, llama_token> tokens;

        for (llama_seq_id seq_id : {0, 1}) {
            int32_t idx = test_ctx.idx_for_seq(seq_id);
            llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), idx);
            const std::string token_str = test_ctx.token_to_piece(token, false);
            printf("  Seq %d, step %d: token id=%d, string='%s'\n", seq_id, step, token, token_str.c_str());
            tokens[seq_id] = token;
        }

        // Decode all tokens in a single batch
        if (!test_ctx.decode_tokens(tokens)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    printf("backend multi-sequence sampling test PASSED\n");
}

static void test_dist(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);
    llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);

    token = llama_get_sampled_token_ith(test_ctx.ctx.get(), -1);
    printf("dist sampled id:%d, string:'%s'\n", token, test_ctx.token_to_piece(token, false).c_str());
    GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);

    printf("backend dist sampling test PASSED\n");
}

static void test_dist_and_cpu(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Some"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    llama_token cpu_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
    printf("dist & cpu sampled id:%d, string:'%s'\n", cpu_token, test_ctx.token_to_piece(cpu_token, false).c_str());
    GGML_ASSERT(backend_token == cpu_token);

    printf("backend dist & cpu sampling test PASSED\n");
}

static void test_logit_bias(const test_params & params) {
    const auto * model = params.model.get();
    const auto * vocab = llama_model_get_vocab(model);

    const int seq_id = 0;

    std::vector<llama_logit_bias> logit_bias;

    // Get the token for the piece "World".
    const std::string piece = "World";
    std::vector<llama_token> tokens(16);
    llama_tokenize(vocab, piece.c_str(), piece.size(), tokens.data(), tokens.size(), false, false);

    llama_token bias_token = tokens[0];
    // TODO: biasing too much here makes the Vulkan sampling fail - should be investigated further
    //       https://github.com/ggml-org/llama.cpp/actions/runs/20894267644/job/60030252675?pr=18753#step:3:23350
    //logit_bias.push_back({ bias_token, +100.0f });
    logit_bias.push_back({ bias_token, +10.0f });

    printf("biasing token piece '%s' -> token id %d\n", piece.c_str(), bias_token);

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_logit_bias(
                llama_vocab_n_tokens(vocab),
                logit_bias.size(),
                logit_bias.data()));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { seq_id, backend_sampler_chain.get() },
    };

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    printf("sampled token = %d, expected = %d\n", backend_token, bias_token);
    GGML_ASSERT(backend_token == bias_token);

    printf("backend logit bias sampling test PASSED\n");
}

static void accept_prompt(llama_sampler * smpl, const llama_vocab * vocab, const std::string & prompt) {
    const llama_token bos = llama_vocab_bos(vocab);
    if (bos != LLAMA_TOKEN_NULL) {
        llama_sampler_accept(smpl, bos);
    }

    std::vector<llama_token> tokens(64);
    int32_t n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
            tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(),
                tokens.data(), (int32_t) tokens.size(), false, false);
    }

    for (int32_t i = 0; i < n_tokens; ++i) {
        llama_sampler_accept(smpl, tokens[i]);
    }
}

static std::vector<float> decode_raw_logits(const test_params & params, const std::string & prompt) {
    const int seq_id = 0;
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(params.model.get()));
    std::vector<llama_sampler_seq_config> empty_configs;
    test_context ctx(params, empty_configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));

    float * logits = llama_get_logits_ith(ctx.ctx.get(), ctx.idx_for_seq(seq_id));
    GGML_ASSERT(logits != nullptr);
    return std::vector<float>(logits, logits + n_vocab);
}

static std::vector<llama_token_data> apply_cpu_sampler(
        const std::vector<float> & raw_logits,
        llama_sampler * sampler) {
    std::vector<llama_token_data> data;
    data.reserve(raw_logits.size());
    for (llama_token token = 0; token < (llama_token) raw_logits.size(); ++token) {
        data.push_back({ token, raw_logits[token], 0.0f });
    }

    llama_token_data_array cur_p = { data.data(), data.size(), -1, false };
    llama_sampler_apply(sampler, &cur_p);
    data.resize(cur_p.size);
    return data;
}

using sampler_setup_fn = std::function<void(llama_sampler *)>;
using sampler_init_fn = std::function<llama_sampler *()>;

enum class penalties_position {
    before_filter,
    after_filter,
};

static void add_filter_and_penalties(
        llama_sampler * chain,
        const sampler_init_fn & init_filter,
        int32_t n_vocab,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        penalties_position position) {
    const auto add_penalties = [&]() {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };

    if (position == penalties_position::before_filter) {
        add_penalties();
        llama_sampler_chain_add(chain, init_filter());
    } else {
        llama_sampler_chain_add(chain, init_filter());
        add_penalties();
    }
}

static llama_sampler_ptr make_sampler_chain(
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    add_samplers(chain.get());
    accept_history(chain.get());
    return chain;
}

struct backend_sampler_output {
    std::vector<float> logits;
    std::vector<llama_token> candidates;
};

static backend_sampler_output run_backend_sampler(
        const test_params & params,
        const std::string & prompt,
        llama_sampler * sampler) {
    const int seq_id = 0;
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, sampler }};
    test_context ctx(params, configs);

    GGML_ASSERT(ctx.decode({{ seq_id, prompt }}));
    llama_synchronize(ctx.ctx.get());

    const int32_t idx = ctx.idx_for_seq(seq_id);
    const uint32_t n_logits = llama_get_sampled_logits_count_ith(ctx.ctx.get(), idx);
    const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(ctx.ctx.get(), idx);
    float * logits = llama_get_sampled_logits_ith(ctx.ctx.get(), idx);
    llama_token * candidates = llama_get_sampled_candidates_ith(ctx.ctx.get(), idx);
    GGML_ASSERT(logits != nullptr);

    backend_sampler_output result;
    result.logits.assign(logits, logits + n_logits);
    result.candidates.resize(n_logits);

    if (n_candidates == 0) {
        for (uint32_t i = 0; i < n_logits; ++i) {
            result.candidates[i] = (llama_token) i;
        }
    } else {
        GGML_ASSERT(candidates != nullptr);
        GGML_ASSERT(n_candidates == n_logits);
        std::memcpy(result.candidates.data(), candidates, n_candidates * sizeof(llama_token));
    }

    return result;
}

struct sampler_comparison_output {
    std::vector<llama_token_data> expected;
    backend_sampler_output actual;
};

static sampler_comparison_output run_sampler_comparison(
        const test_params & params,
        const std::string & prompt,
        const std::vector<float> & raw_logits,
        const sampler_setup_fn & add_samplers,
        const sampler_setup_fn & accept_history) {
    llama_sampler_ptr cpu_chain = make_sampler_chain(add_samplers, accept_history);
    llama_sampler_ptr backend_chain = make_sampler_chain(add_samplers, accept_history);
    return {
        apply_cpu_sampler(raw_logits, cpu_chain.get()),
        run_backend_sampler(params, prompt, backend_chain.get()),
    };
}

static std::unordered_map<llama_token, float> map_logits(const std::vector<llama_token_data> & data) {
    std::unordered_map<llama_token, float> result;
    result.reserve(data.size());
    for (const auto & item : data) {
        result[item.id] = item.logit;
    }
    return result;
}

struct sampler_comparison_stats {
    int n_mismatch = 0;
    int n_masked = 0;
    float max_diff = 0.0f;
};

static sampler_comparison_stats compare_sampler_outputs(
        const char * name,
        const std::unordered_map<llama_token, float> & expected,
        const backend_sampler_output & actual,
        bool allow_extra_candidates = false) {
    GGML_ASSERT(actual.logits.size() == actual.candidates.size());

    sampler_comparison_stats result;
    std::unordered_set<llama_token> seen;
    seen.reserve(actual.candidates.size());

    for (size_t i = 0; i < actual.logits.size(); ++i) {
        const llama_token token = actual.candidates[i];
        const float logit = actual.logits[i];
        if (!seen.insert(token).second || std::isnan(logit)) {
            if (result.n_mismatch < 5) {
                printf("%s token %d has invalid backend output\n", name, token);
            }
            ++result.n_mismatch;
            continue;
        }

        const auto it = expected.find(token);
        if (it == expected.end()) {
            if (std::isinf(logit) && logit < 0.0f) {
                ++result.n_masked;
            } else if (!allow_extra_candidates) {
                if (result.n_mismatch < 5) {
                    printf("%s token %d was not masked\n", name, token);
                }
                ++result.n_mismatch;
            }
            continue;
        }

        const float diff = fabsf(it->second - logit);
        result.max_diff = std::max(result.max_diff, diff);
        if (!std::isfinite(logit) || diff > 1e-3f) {
            if (result.n_mismatch < 5) {
                printf("%s mismatch token %d: cpu=%.6f backend=%.6f diff=%.6f\n",
                        name, token, it->second, logit, diff);
            }
            ++result.n_mismatch;
        }
    }

    for (const auto & item : expected) {
        if (seen.find(item.first) == seen.end()) {
            if (result.n_mismatch < 5) {
                printf("%s missing backend token %d\n", name, item.first);
            }
            ++result.n_mismatch;
        }
    }

    printf("%s logits: max_diff=%.6f n_masked=%d n_mismatch=%d\n",
            name, result.max_diff, result.n_masked, result.n_mismatch);
    return result;
}

static float find_backend_logit(const backend_sampler_output & output, llama_token token) {
    for (size_t i = 0; i < output.candidates.size(); ++i) {
        if (output.candidates[i] == token) {
            return output.logits[i];
        }
    }
    GGML_ABORT("backend token not found");
}

static sampler_comparison_output run_penalties_comparison(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const auto add_samplers = [&](llama_sampler * chain) {
        llama_sampler_chain_add(chain, llama_sampler_init_penalties(
                    llama_vocab_n_tokens(vocab), penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
    };
    const auto accept_history = [&](llama_sampler * chain) {
        accept_prompt(chain, vocab, prompt);
        if (extra_accept) {
            extra_accept(chain);
        }
    };

    return run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);
}

static void compare_penalties_logits(
        const test_params & params,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        const std::function<void(llama_sampler *)> & extra_accept = {}) {
    const sampler_comparison_output output = run_penalties_comparison(
            params, penalty_last_n, penalty_repeat, penalty_freq, penalty_present, prompt, extra_accept);

    GGML_ASSERT(output.expected.size() == output.actual.logits.size());

    const sampler_comparison_stats stats = compare_sampler_outputs(
            "penalties", map_logits(output.expected), output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void check_penalty_parameter_values(const test_params & params) {
    struct penalty_test_case {
        const char * name;
        float repeat;
        float frequency;
        float presence;
    };

    const penalty_test_case cases[] = {
        { "frequency -1",   1.0f, -1.0f,     0.0f },
        { "frequency 0",    1.0f,  0.0f,     0.0f },
        { "frequency 1",    1.0f,  1.0f,     0.0f },
        { "presence -1",    1.0f,  0.0f,    -1.0f },
        { "presence 0",     1.0f,  0.0f,     0.0f },
        { "presence 1",     1.0f,  0.0f,     1.0f },
        { "repeat 1",       1.0f,  0.0f,     0.0f },
    };

    int n_failed = 0;
    for (const auto & test : cases) {
        const sampler_comparison_output output = run_penalties_comparison(
                params, 64, test.repeat, test.frequency, test.presence, "Hello Hello world");
        GGML_ASSERT(output.expected.size() == output.actual.logits.size());
        const sampler_comparison_stats stats = compare_sampler_outputs(
                test.name, map_logits(output.expected), output.actual);
        n_failed += stats.n_mismatch != 0;
    }

    GGML_ASSERT(n_failed == 0);
}

static void compare_top_k_penalties_logits(
        const test_params & params,
        int32_t k,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();

    GGML_ASSERT(n_vocab > k);

    const sampler_init_fn init_top_k = [k]() {
        return llama_sampler_init_top_k(k);
    };
    llama_sampler_ptr top_k(init_top_k());
    const std::vector<llama_token_data> top_k_data = apply_cpu_sampler(raw_logits, top_k.get());
    GGML_ASSERT(top_k_data.size() == (size_t) k);
    const llama_token retained_history_token = top_k_data[0].id;

    llama_token excluded_history_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        const auto it = std::find_if(top_k_data.begin(), top_k_data.end(), [token](const llama_token_data & data) {
            return data.id == token;
        });
        if (it == top_k_data.end()) {
            excluded_history_token = token;
            break;
        }
    }
    GGML_ASSERT(excluded_history_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_top_k, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };

    auto accept_history = [&](llama_sampler * smpl) {
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, excluded_history_token);
        llama_sampler_accept(smpl, retained_history_token);
        llama_sampler_accept(smpl, retained_history_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.expected.size() == (size_t) k);
    GGML_ASSERT(output.actual.logits.size() == (size_t) k);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    if (position == penalties_position::after_filter) {
        GGML_ASSERT(expected_logits.find(retained_history_token) != expected_logits.end());
        GGML_ASSERT(fabsf(expected_logits.at(retained_history_token) - raw_logits[retained_history_token]) > 1e-6f);
        GGML_ASSERT(expected_logits.find(excluded_history_token) == expected_logits.end());
        GGML_ASSERT(std::find(output.actual.candidates.begin(), output.actual.candidates.end(),
                    excluded_history_token) == output.actual.candidates.end());
    } else {
        const std::unordered_map<llama_token, float> unpenalized_logits = map_logits(top_k_data);
        bool changed = false;
        for (const auto & item : expected_logits) {
            const auto it = unpenalized_logits.find(item.first);
            if (it == unpenalized_logits.end() || fabsf(it->second - item.second) > 1e-6f) {
                changed = true;
                break;
            }
        }
        GGML_ASSERT(changed);
    }

    const char * name = position == penalties_position::before_filter
        ? "penalties top-k"
        : "top-k penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name, expected_logits, output.actual);
    GGML_ASSERT(stats.n_masked == 0);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void compare_masking_penalties_logits(
        const test_params & params,
        const char * filter_name,
        const sampler_init_fn & init_filter,
        int32_t penalty_last_n,
        float penalty_repeat,
        float penalty_freq,
        float penalty_present,
        const std::string & prompt,
        penalties_position position,
        bool allow_extra_candidates,
        bool add_history = true) {
    const auto * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<float> raw_logits = decode_raw_logits(params, prompt);
    const int n_vocab = (int) raw_logits.size();
    llama_sampler_ptr filter(init_filter());
    const std::vector<llama_token_data> filtered_data = apply_cpu_sampler(raw_logits, filter.get());
    GGML_ASSERT(!filtered_data.empty());
    GGML_ASSERT(filtered_data.size() < (size_t) n_vocab);

    const llama_token penalized_token = filtered_data[0].id;
    std::unordered_set<llama_token> retained_tokens;
    retained_tokens.reserve(filtered_data.size());
    for (const auto & data : filtered_data) {
        retained_tokens.insert(data.id);
    }

    llama_token masked_token = LLAMA_TOKEN_NULL;
    for (llama_token token = 0; token < n_vocab; ++token) {
        if (retained_tokens.find(token) == retained_tokens.end()) {
            masked_token = token;
            break;
        }
    }
    GGML_ASSERT(masked_token != LLAMA_TOKEN_NULL);

    const auto add_samplers = [&](llama_sampler * chain) {
        add_filter_and_penalties(chain, init_filter, n_vocab,
                penalty_last_n, penalty_repeat, penalty_freq, penalty_present, position);
    };
    auto accept_history = [&](llama_sampler * smpl) {
        if (!add_history) {
            return;
        }
        accept_prompt(smpl, vocab, prompt);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, penalized_token);
        llama_sampler_accept(smpl, masked_token);
        llama_sampler_accept(smpl, masked_token);
    };

    const sampler_comparison_output output = run_sampler_comparison(
            params, prompt, raw_logits, add_samplers, accept_history);

    GGML_ASSERT(output.actual.logits.size() == (size_t) n_vocab);

    const std::unordered_map<llama_token, float> expected_logits = map_logits(output.expected);

    GGML_ASSERT(expected_logits.find(masked_token) == expected_logits.end());
    if (add_history) {
        if (position == penalties_position::after_filter) {
            GGML_ASSERT(expected_logits.find(penalized_token) != expected_logits.end());
            GGML_ASSERT(fabsf(expected_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        } else {
            llama_sampler_ptr penalties(llama_sampler_init_penalties(
                        n_vocab, penalty_last_n, penalty_repeat, penalty_freq, penalty_present));
            accept_history(penalties.get());
            const std::unordered_map<llama_token, float> penalized_logits =
                map_logits(apply_cpu_sampler(raw_logits, penalties.get()));
            GGML_ASSERT(fabsf(penalized_logits.at(penalized_token) - raw_logits[penalized_token]) > 1e-6f);
        }
    }

    const std::string name = position == penalties_position::before_filter
        ? "penalties " + std::string(filter_name)
        : std::string(filter_name) + " penalties";
    const sampler_comparison_stats stats = compare_sampler_outputs(
            name.c_str(), expected_logits, output.actual, allow_extra_candidates);
    const float masked_logit = find_backend_logit(output.actual, masked_token);
    GGML_ASSERT(stats.n_masked > 0);
    GGML_ASSERT(std::isinf(masked_logit) && masked_logit < 0.0f);
    GGML_ASSERT(stats.n_mismatch == 0);
}

static void test_penalties(const test_params & params) {
    printf("Testing backend penalties (repeat + freq + presence)\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello Hello world");

    printf("Testing backend penalties with penalty_last_n > 64\n");
    const auto * vocab = llama_model_get_vocab(params.model.get());
    std::vector<llama_token> tokens(8);
    int32_t n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    if (n_tok < 0) {
        tokens.resize(-n_tok);
        n_tok = llama_tokenize(vocab, "a", 1, tokens.data(), (int32_t) tokens.size(), false, false);
    }
    GGML_ASSERT(n_tok > 0);
    const llama_token tok = tokens[0];

    compare_penalties_logits(params, 80, 1.15f, 0.1f, 0.05f, "a", [tok](llama_sampler * smpl) {
        // accept_prompt already accepted BOS + one 'a'; fill the ring to n=80
        for (int i = 0; i < 78; ++i) {
            llama_sampler_accept(smpl, tok);
        }
    });

    printf("Testing backend penalties without filler entries\n");
    compare_penalties_logits(params, 64, 1.1f, 0.5f, 0.25f, "Hello", [](llama_sampler * smpl) {
        for (llama_token token = 0; token < 64; ++token) {
            llama_sampler_accept(smpl, token);
        }
    });

    printf("Testing backend top-k followed by penalties\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::after_filter);

    printf("Testing backend penalties followed by top-k\n");
    compare_top_k_penalties_logits(params, 8, 64, 1.1f, 0.5f, 0.25f, "Hello",
            penalties_position::before_filter);

    printf("Testing backend top-p followed by penalties\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend top-p followed by penalties with a large history window\n");
    compare_masking_penalties_logits(params, "top-p large-window", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 4096, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalties followed by top-p\n");
    compare_masking_penalties_logits(params, "top-p", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, true);

    printf("Testing backend min-p followed by penalties\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, false);

    printf("Testing backend penalties followed by min-p\n");
    compare_masking_penalties_logits(params, "min-p", []() {
        return llama_sampler_init_min_p(0.1f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::before_filter, false);

    printf("Testing backend top-p followed by penalties with empty history\n");
    compare_masking_penalties_logits(params, "top-p empty", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.5f, 0.25f, "Hello", penalties_position::after_filter, true, false);

    printf("Testing backend top-p followed by individual penalties\n");
    compare_masking_penalties_logits(params, "top-p repeat", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.1f, 0.0f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p frequency", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.5f, 0.0f, "Hello", penalties_position::after_filter, true);
    compare_masking_penalties_logits(params, "top-p presence", []() {
        return llama_sampler_init_top_p(0.9f, 0);
    }, 64, 1.0f, 0.0f, 0.25f, "Hello", penalties_position::after_filter, true);

    printf("Testing backend penalty parameter values\n");
    check_penalty_parameter_values(params);

    printf("backend penalties sampling test PASSED\n");
}

// This test verifies that it is possible to have two different backend samplers,
// one that uses the backend dist sampler, and another that uses CPU dist sampler.
static void test_mixed(const test_params & params) {
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    int k = 40;
    struct llama_sampler_chain_params chain_params_1 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_1(llama_sampler_chain_init(chain_params_1));
    llama_sampler_chain_add(sampler_chain_1.get(), llama_sampler_init_top_k(k));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
        { 1, sampler_chain_1.get() }
    };

    test_context test_ctx(params, backend_sampler_configs);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"},
        {1, "Some"}
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 that used the dist backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
        //GGML_ASSERT(llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx) == nullptr);
        //GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx) == 0);
    }

    // Verify sequence 1 that used the top-k backend sampler.
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);
        float * logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(logits != nullptr);
        size_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(n_logits == (size_t) k);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx) == LLAMA_TOKEN_NULL);
    }

    printf("backend mixed sampling test PASSED\n");
}

static void test_set_sampler(const test_params & params) {
    const int seq_id = 0;
    const int32_t seed = 88;

    struct llama_sampler_chain_params backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr backend_sampler_chain(llama_sampler_chain_init(backend_chain_params));
    llama_sampler_chain_add(backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> backend_sampler_configs = {{ seq_id, backend_sampler_chain.get() }};

    test_context test_ctx(params, backend_sampler_configs);

    if (!test_ctx.decode({{seq_id, "Hello"}})) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    int32_t batch_idx = test_ctx.idx_for_seq(seq_id);

    // Sample using backend sampler configured above
    llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
    const std::string backend_token_str = test_ctx.token_to_piece(backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", backend_token, backend_token_str.c_str());

    // Now clear the backend sampler for this sequence.
    llama_set_sampler(test_ctx.ctx.get(), seq_id, nullptr);
    printf("Cleared backend sampler for seq_id %d\n", seq_id);

    // Sample using CPU sampler
    struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(18));

    std::map<llama_seq_id, llama_token> tokens = { { seq_id, backend_token}, };
    if (!test_ctx.decode_tokens(tokens)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Should not have any sampled token or probs after clearing the backend sampler.
    const int32_t idx = test_ctx.idx_for_seq(seq_id);
    GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), idx) == LLAMA_TOKEN_NULL);
    GGML_ASSERT(llama_get_sampled_probs_ith(test_ctx.ctx.get(), idx) == nullptr);

    // Sample the token using the CPU sampler chain.
    llama_token token2 = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), seq_id);
    const std::string token2_str = test_ctx.token_to_piece(token2, false);
    printf("CPU sampled token after clearing backend sampler: id=%d, string='%s'\n", token2, token2_str.c_str());
    std::map<llama_seq_id, llama_token> tokens2 = { { seq_id, token2}, };

    // Set a new backend sampler for the sequence.
    struct llama_sampler_chain_params new_backend_chain_params = llama_sampler_chain_default_params();
    llama_sampler_ptr new_backend_sampler_chain(llama_sampler_chain_init(new_backend_chain_params));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_top_k(20));
    llama_sampler_chain_add(new_backend_sampler_chain.get(), llama_sampler_init_dist(seed));
    llama_set_sampler(test_ctx.ctx.get(), seq_id, new_backend_sampler_chain.get());

    if (!test_ctx.decode_tokens(tokens2)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    llama_token new_backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), test_ctx.idx_for_seq(seq_id));
    const std::string new_backend_token_str = test_ctx.token_to_piece(new_backend_token, false);
    printf("dist sampled token = %d, string='%s'\n", new_backend_token, new_backend_token_str.c_str());

    printf("backend set sampler test PASSED\n");
}

static void test_cpu_mixed(const test_params & params) {
    // Sequence 0 uses backend sampling
    struct llama_sampler_chain_params chain_params_0 = llama_sampler_chain_default_params();
    llama_sampler_ptr sampler_chain_0(llama_sampler_chain_init(chain_params_0));
    llama_sampler_chain_add(sampler_chain_0.get(), llama_sampler_init_dist(88));

    std::vector<llama_sampler_seq_config> backend_sampler_configs = {
        { 0, sampler_chain_0.get() },
    };

    // We need 2 sequences: seq 0 with backend sampling, seq 1 with CPU sampling
    test_context test_ctx(params, backend_sampler_configs, 2);

    std::map<llama_seq_id, std::string> prompts = {
        {0, "Hello"}, // Will use backend sampling
        {1, "Some"}   // Will use CPU sampling
    };

    if (!test_ctx.decode(prompts)) {
        GGML_ASSERT(false && "Failed to decode token");
    }

    // Verify sequence 0 (backend sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 0 (backend) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Verify sequence 1 (CPU sampled)
    {
        int32_t batch_idx = test_ctx.idx_for_seq(1);

        llama_token backend_token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        GGML_ASSERT(backend_token == LLAMA_TOKEN_NULL);

        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("Seq 1 (CPU) sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    // Clear/remove the backend sampler, and sample again
    {
        // clear the backend sampler for seq 0 so that there are no backend
        // samplers.
        llama_set_sampler(test_ctx.ctx.get(), 0, nullptr);

        // Create a CPU sampler and verify we can sample from it.
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());

        int32_t batch_idx = test_ctx.idx_for_seq(1);
        llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), batch_idx);
        if (!test_ctx.decode_token(token, 1)) {
            GGML_ASSERT(false && "Failed to decode token");
        }
    }

    // Set a backend sampler so that we can verify that it can be reset
    {
        struct llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        llama_sampler_ptr sampler_chain(llama_sampler_chain_init(chain_params));
        llama_sampler_chain_add(sampler_chain.get(), llama_sampler_init_dist(88));

        llama_set_sampler(test_ctx.ctx.get(), 0, sampler_chain.get());

        if (!test_ctx.decode_token(3834, 0)) {
            GGML_ASSERT(false && "Failed to decode token");
        }

        int32_t batch_idx = test_ctx.idx_for_seq(0);
        llama_token token = llama_get_sampled_token_ith(test_ctx.ctx.get(), batch_idx);
        const std::string token_str = test_ctx.token_to_piece(token, false);
        printf("re-added backend sampled token id=%d, string='%s'\n", token, token_str.c_str());
        GGML_ASSERT(token >= 0 && token < test_ctx.n_vocab);
    }

    printf("backend-cpu mixed batch test PASSED\n");
}

static void test_multi_output_limit(const test_params & params) {
    const llama_seq_id seq_id = 0;

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(88));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 0, 2);

    common_batch batch(test_ctx.ctx.get());
    for (int i = 0; i < 3; ++i) {
        batch.add(llama_vocab_bos(test_ctx.vocab), i, seq_id, true);
    }

    printf(">>> test_multi_output_limit expected error start:\n");
    const int ret = llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
    GGML_ASSERT(ret != 0 && "llama_decode should reject outputs above the per-sequence limit");
    printf("<<< test_multi_output_limit expected error end.\n");

    printf("backend multi-output limit test PASSED\n");
}

static void test_multi_output_multi_sequence_dist(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t seeds[] = { 88, 1337 };
    // reduce the chance that swapped random inputs select the same token
    const float temp = 10.0f;

    llama_sampler_ptr chain_0(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_ptr chain_1(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_0.get(), llama_sampler_init_dist(seeds[0]));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_temp(temp));
    llama_sampler_chain_add(chain_1.get(), llama_sampler_init_dist(seeds[1]));
    std::vector<llama_sampler_seq_config> configs = {
        { 0, chain_0.get() },
        { 1, chain_1.get() },
    };
    test_context test_ctx(params, configs, 2, 4, 0, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 2, 4);

    const llama_token seq_tokens[2][2] = {
        { llama_vocab_bos(vocab), llama_vocab_eos(vocab) },
        { llama_vocab_eos(vocab), llama_vocab_bos(vocab) },
    };

    // a batch belongs to one context, so it is built per context
    auto make_batch = [&](llama_context * ctx) {
        common_batch batch(ctx);
        for (int pos = 0; pos < 2; ++pos) {
            batch.add(seq_tokens[0][pos], pos, 0, true);
            batch.add(seq_tokens[1][pos], pos, 1, true);
        }
        return batch;
    };

    common_batch batch = make_batch(test_ctx.ctx.get());
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    {
        common_batch batch_ref = make_batch(reference_ctx.ctx.get());
        GGML_ASSERT(llama_process(reference_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_ref.get()) == 0);
    }

    std::mt19937 reference_rngs[] = {
        std::mt19937(seeds[0]),
        std::mt19937(seeds[1]),
    };
    std::uniform_real_distribution<double> reference_dist(0.0, 1.0);

    for (int i = 0; i < batch.size(); ++i) {
        const llama_seq_id seq_id = batch.tokens[i].seq_id;
        GGML_ASSERT(seq_id == 0 || seq_id == 1);

        llama_sampler * chain = seq_id == 0 ? chain_0.get() : chain_1.get();
        const llama_token backend_token = llama_sampler_sample(chain, test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == (uint32_t) n_vocab);
        GGML_ASSERT(n_probs == (uint32_t) n_vocab);

        float prob_sum = 0.0f;
        float cumsum_before = 0.0f;
        for (llama_token token = 0; token < n_vocab; ++token) {
            const float expected_logit = reference_logits[token] / temp;
            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(expected_logit));
            GGML_ASSERT(std::fabs(sampled_logits[token] - expected_logit) <= tolerance);
            GGML_ASSERT(std::isfinite(sampled_probs[token]));
            GGML_ASSERT(sampled_probs[token] >= 0.0f);

            prob_sum += sampled_probs[token];
            if (token < backend_token) {
                cumsum_before += sampled_probs[token];
            }
        }

        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);

        const float rnd = reference_dist(reference_rngs[seq_id]);
        const float cumsum_sampled = cumsum_before + sampled_probs[backend_token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    }

    printf("backend multi-sequence multi-output dist test PASSED\n");
}

static void test_multi_output_dist_transaction(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 95;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    llama_sampler_ptr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(10.0f));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 3, 2, 3);

    auto verify_random = [&](int32_t row, float rnd, bool accept = true) {
        const llama_token token = accept ?
            llama_sampler_sample(chain.get(), test_ctx.ctx.get(), row) :
            llama_get_sampled_token_ith(test_ctx.ctx.get(), row);
        const float * probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), row);

        GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        GGML_ASSERT(probs != nullptr);

        float cumsum_before = 0.0f;
        for (llama_token i = 0; i < token; ++i) {
            cumsum_before += probs[i];
        }

        const float cumsum_sampled = cumsum_before + probs[token];
        GGML_ASSERT(rnd >= cumsum_before - 1e-4f);
        GGML_ASSERT(rnd <= cumsum_sampled + 1e-4f);
    };

    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    float randoms[3];
    for (float & rnd : randoms) {
        rnd = dist(rng);
    }

    int32_t pos = 0;
    auto decode = [&]() {
        common_batch batch(test_ctx.ctx.get());
        for (int32_t i = 0; i < 3; ++i) {
            batch.add(llama_vocab_bos(vocab), pos++, seq_id, true);
        }
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    };

    decode();
    verify_random(0, randoms[0], false);

    decode();
    verify_random(0, randoms[0]);
    verify_random(1, randoms[1]);

    decode();
    llama_sampler_ptr saved(llama_sampler_clone(chain.get()));
    verify_random(0, randoms[2]);

    llama_sampler_copy(saved.get(), chain.get());

    decode();
    verify_random(0, randoms[2]);

    printf("backend multi-output dist transaction test PASSED\n");
}

static void test_multi_output_sampling_chain(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const uint32_t seed = 88;
    const float p = 0.9f;
    const float temp = 0.8f;
    const float cdf_epsilon = 1e-4f;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const uint32_t k = std::min<uint32_t>(512, n_vocab);
    const llama_logit_bias bias = { llama_vocab_bos(vocab), -0.1f };

    auto make_filter_chain = [&]() {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_logit_bias(n_vocab, 1, &bias));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_p(p, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_min_p(0.01f, 1));
        llama_sampler_chain_add(result.get(), llama_sampler_init_temp(temp));
        return result;
    };

    llama_sampler_ptr chain = make_filter_chain();
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(seed));
    std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
    test_context test_ctx(params, configs, 1, 2, 2, 2);

    std::vector<llama_sampler_seq_config> reference_configs;
    test_context reference_ctx(params, reference_configs, 1, 2, 2);

    llama_sampler_ptr reference_bias(llama_sampler_init_logit_bias(n_vocab, 1, &bias));
    llama_sampler_ptr reference_top_k(llama_sampler_init_top_k(k));
    llama_sampler_ptr reference_top_p(llama_sampler_init_top_p(p, 1));
    llama_sampler_ptr reference_min_p(llama_sampler_init_min_p(0.01f, 1));
    llama_sampler_ptr reference_temp(llama_sampler_init_temp(temp));
    std::vector<llama_token_data> reference_data(n_vocab);

    // a batch belongs to one context, so it is built per context
    auto make_batch = [&](llama_context * ctx, int32_t pos) {
        common_batch batch(ctx);
        for (int i = 0; i < 2; ++i) {
            batch.add(llama_vocab_bos(vocab), pos + i, seq_id, true);
        }
        return batch;
    };

    common_batch batch = make_batch(test_ctx.ctx.get(), 0);
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    {
        common_batch batch_ref = make_batch(reference_ctx.ctx.get(), 0);
        GGML_ASSERT(llama_process(reference_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_ref.get()) == 0);
    }

    for (int i = 0; i < batch.size(); ++i) {
        const llama_token backend_token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
        const float * sampled_logits = llama_get_sampled_logits_ith(test_ctx.ctx.get(), i);
        const float * sampled_probs = llama_get_sampled_probs_ith(test_ctx.ctx.get(), i);
        const llama_token * sampled_candidates = llama_get_sampled_candidates_ith(test_ctx.ctx.get(), i);
        const uint32_t n_logits = llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_probs = llama_get_sampled_probs_count_ith(test_ctx.ctx.get(), i);
        const uint32_t n_candidates = llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i);
        const float * reference_logits = llama_get_logits_ith(reference_ctx.ctx.get(), i);

        GGML_ASSERT(backend_token >= 0 && backend_token < n_vocab);
        GGML_ASSERT(sampled_logits != nullptr);
        GGML_ASSERT(sampled_probs != nullptr);
        GGML_ASSERT(sampled_candidates != nullptr);
        GGML_ASSERT(reference_logits != nullptr);
        GGML_ASSERT(n_logits == k);
        GGML_ASSERT(n_probs == n_logits);
        GGML_ASSERT(n_candidates == n_logits);

        for (llama_token token = 0; token < n_vocab; ++token) {
            reference_data[token] = { token, reference_logits[token], 0.0f };
        }

        llama_token_data_array reference = {
            /* .data     = */ reference_data.data(),
            /* .size     = */ reference_data.size(),
            /* .selected = */ LLAMA_TOKEN_NULL,
            /* .sorted   = */ false,
        };

        llama_sampler_apply(reference_bias.get(), &reference);
        llama_sampler_apply(reference_top_k.get(), &reference);
        llama_sampler_apply(reference_top_p.get(), &reference);
        GGML_ASSERT(reference.size > 0);

        float cdf = 0.0f;
        for (size_t j = 0; j < reference.size; ++j) {
            cdf += reference.data[j].p;
        }
        const float cdf_before = cdf - reference.data[reference.size - 1].p;
        const float boundary_distance = std::min(std::fabs(cdf_before - p), std::fabs(cdf - p));

        llama_sampler_apply(reference_min_p.get(), &reference);
        llama_sampler_apply(reference_temp.get(), &reference);

        std::unordered_map<llama_token, float> reference_by_id;
        for (size_t j = 0; j < reference.size; ++j) {
            reference_by_id.emplace(reference.data[j].id, reference.data[j].logit);
        }
        size_t n_backend_only = 0;
        int32_t sampled_index = -1;
        float prob_sum = 0.0f;

        for (uint32_t j = 0; j < n_logits; ++j) {
            GGML_ASSERT(sampled_candidates[j] >= 0 && sampled_candidates[j] < n_vocab);
            GGML_ASSERT(std::isfinite(sampled_probs[j]));
            GGML_ASSERT(sampled_probs[j] >= 0.0f);
            prob_sum += sampled_probs[j];

            if (sampled_candidates[j] == backend_token) {
                sampled_index = j;
            }
            if (!std::isfinite(sampled_logits[j])) {
                GGML_ASSERT(std::isinf(sampled_logits[j]) && sampled_logits[j] < 0.0f);
                GGML_ASSERT(sampled_probs[j] == 0.0f);
                continue;
            }

            const auto match = reference_by_id.find(sampled_candidates[j]);
            if (match == reference_by_id.end()) {
                ++n_backend_only;
                continue;
            }

            const float tolerance = 1e-4f * std::max(1.0f, std::fabs(match->second));
            GGML_ASSERT(std::fabs(sampled_logits[j] - match->second) <= tolerance);
            reference_by_id.erase(match);
        }

        const size_t n_reference_only = reference_by_id.size();

        if (n_backend_only != 0 || n_reference_only != 0) {
            GGML_ASSERT(n_backend_only <= 1);
            GGML_ASSERT(n_reference_only <= 1);
            GGML_ASSERT(boundary_distance <= cdf_epsilon);
        }

        GGML_ASSERT(sampled_index >= 0);
        GGML_ASSERT(std::isfinite(sampled_logits[sampled_index]));
        GGML_ASSERT(sampled_probs[sampled_index] > 0.0f);
        GGML_ASSERT(std::fabs(prob_sum - 1.0f) <= 1e-3f);
    }

    batch = make_batch(test_ctx.ctx.get(), 2);
    GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

    printf("backend multi-output sampling chain test PASSED\n");
}

static void test_multi_output_cpu_suffix(const test_params & params) {
    const llama_seq_id seq_id = 0;
    const int32_t k = 8;
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());

    auto make_chain = [&](test_single_output_backend_sampler ** sampler_ctx) {
        llama_sampler_ptr result(llama_sampler_chain_init(llama_sampler_chain_default_params()));
        llama_sampler_chain_add(result.get(), llama_sampler_init_top_k(k));
        llama_sampler_chain_add(result.get(), test_single_output_backend_sampler_init(sampler_ctx));
        llama_sampler_chain_add(result.get(), llama_sampler_init_dist(88));
        return result;
    };

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 1, 0, 4);

        common_batch batch(test_ctx.ctx.get());
        batch.add(llama_vocab_bos(vocab), 0, seq_id, true);
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

        GGML_ASSERT(sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 1);
        GGML_ASSERT(sampler_ctx->backend_apply_count > 0);
        GGML_ASSERT(sampler_ctx->apply_count == 0);
        GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), 0) != LLAMA_TOKEN_NULL);
    }

    {
        test_single_output_backend_sampler * sampler_ctx = nullptr;
        llama_sampler_ptr chain = make_chain(&sampler_ctx);
        std::vector<llama_sampler_seq_config> configs = {{ seq_id, chain.get() }};
        test_context test_ctx(params, configs, 1, 2, 0, 0);

        common_batch batch(test_ctx.ctx.get());
        for (int i = 0; i < 2; ++i) {
            batch.add(llama_vocab_bos(vocab), i, seq_id, true);
        }
        GGML_ASSERT(llama_process(test_ctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);

        GGML_ASSERT(!sampler_ctx->backend_initialized);
        GGML_ASSERT(sampler_ctx->backend_outputs_max_per_seq == 2);
        GGML_ASSERT(sampler_ctx->backend_apply_count == 0);
        for (int i = 0; i < batch.size(); ++i) {
            GGML_ASSERT(llama_get_sampled_token_ith(test_ctx.ctx.get(), i) == LLAMA_TOKEN_NULL);
            GGML_ASSERT(llama_get_sampled_logits_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            GGML_ASSERT(llama_get_sampled_candidates_count_ith(test_ctx.ctx.get(), i) == (uint32_t) k);
            const llama_token token = llama_sampler_sample(chain.get(), test_ctx.ctx.get(), i);
            GGML_ASSERT(token >= 0 && token < llama_vocab_n_tokens(vocab));
        }
        GGML_ASSERT(sampler_ctx->apply_count == batch.size());
    }

    printf("backend multi-output CPU suffix test PASSED\n");
}

//
// GPU vs CPU parity of whole sampler chains (common_sampler, as llama-server builds them)
//
// The same batch is decoded by a context without samplers (CPU reference: identical logits) and by one that
// samples on the backend; every sequence gets its own seed. For every output the sampled token must be the
// same and the post-sampling candidates and probabilities must match.
//

struct parity_case {
    std::string name;
    std::function<void(common_params_sampling &)> setup;
    bool greedy = false; // no probabilities to compare
};

struct parity_stats {
    int n_draws    = 0;
    int n_equal    = 0;
    int n_set_diff = 0;
    int n_tied     = 0; // mismatches tolerated as provable near-ties of the running sum (parity_near_tie)
    float max_p_diff = 0.0f;
};

// The CPU dist sampler picks the first candidate whose running sum (double) reaches u * total, the backend one
// does the same on an fp32 cumulative sum. Its expected error over n terms is about sqrt(n) * FLT_EPSILON * total
// (rounding errors of a sum of n floats add like a random walk); the worst-case n * FLT_EPSILON * total bound is far
// too loose here (3e-2 at a 248k vocabulary) and tolerated draws of a deliberately broken sampler (dist picking
// index + 1) whose token had less mass than the bound. A different token is a near-tie only if u * total lies within
// sqrt(n) * FLT_EPSILON * total of the CPU cumulative interval
// of both tokens (so it sits on the boundary between them, with at most that much mass in between) and both
// tokens' probabilities agree within the same bound. u is replayed from the seed: one draw of
// std::uniform_real_distribution<double> on std::mt19937 per sample, as llama_sampler_dist_apply does. Any other
// mismatch, or one where the replayed u does not fall in the CPU token's interval, is not tolerated.
static bool parity_near_tie(const char * name, int seq, uint32_t row, common_sampler * cpu, common_sampler * gpu,
                            llama_token tc, llama_token tg, double u) {
    const auto * cp = common_sampler_get_candidates(cpu, false);
    const auto * gp = common_sampler_get_candidates(gpu, false);

    double total = 0.0;
    for (size_t i = 0; i < cp->size; ++i) {
        total += cp->data[i].p;
    }
    const double eps = sqrt((double) cp->size) * FLT_EPSILON * total;

    // CPU cumulative interval [lo, hi) of a token, in the order the CPU accumulated
    auto interval = [&](llama_token t, double & lo, double & hi, float & p) {
        double c = 0.0;
        for (size_t i = 0; i < cp->size; ++i) {
            if (cp->data[i].id == t) {
                lo = c; hi = c + cp->data[i].p; p = cp->data[i].p;
                return true;
            }
            c += cp->data[i].p;
        }
        return false;
    };
    auto gpu_p = [&](llama_token t, float & p) {
        for (size_t i = 0; i < gp->size; ++i) {
            if (gp->data[i].id == t) {
                p = gp->data[i].p;
                return true;
            }
        }
        return false;
    };
    auto dist_to = [](double x, double lo, double hi) { return x < lo ? lo - x : x > hi ? x - hi : 0.0; };

    double lc = 0, hc = 0, lg = 0, hg = 0;
    float  pcc = 0, pcg = 0, pgc = 0, pgg = 0;
    if (!interval(tc, lc, hc, pcc) || !interval(tg, lg, hg, pcg) || !gpu_p(tc, pgc) || !gpu_p(tg, pgg)) {
        printf("  near-tie %s seq %d row %u: token missing from a candidate list, not tolerated\n", name, seq, row);
        return false;
    }
    const double ut = u * total;
    const double dc = dist_to(ut, lc, hc);
    const double dg = dist_to(ut, lg, hg);
    const bool ok = dc <= eps && dg <= eps && fabs((double) pcc - pgc) <= eps && fabs((double) pcg - pgg) <= eps;
    printf("  near-tie %s seq %d row %u: u*total %.9f cpu tok %d [%.9f, %.9f) gpu tok %d [%.9f, %.9f) "
           "dist %.2e / %.2e p diff %.2e / %.2e eps %.3e (%zu cand) %s\n",
           name, seq, row, ut, tc, lc, hc, tg, lg, hg, dc, dg, fabs((double) pcc - pgc), fabs((double) pcg - pgg),
           eps, cp->size, ok ? "tolerated" : "NOT tolerated");
    return ok;
}

static std::vector<llama_token> tokenize_prompt(const llama_vocab * vocab, const std::string & prompt, bool add_special = true) {
    std::vector<llama_token> tokens(256);
    int32_t n = llama_tokenize(vocab, prompt.c_str(), (int32_t) prompt.size(), tokens.data(), (int32_t) tokens.size(), add_special, false);
    GGML_ASSERT(n > 0);
    tokens.resize(n);
    return tokens;
}

static common_params_sampling parity_base_params() {
    common_params_sampling sp;
    sp.top_k    = 0;
    sp.top_p    = 1.0f;
    sp.min_p    = 0.0f;
    sp.temp     = 1.0f;
    sp.min_keep = 0;
    sp.no_perf  = true;
    return sp;
}

static void compare_candidates(
        const parity_case & pc,
        common_sampler * cpu,
        common_sampler * gpu,
        parity_stats & st) {
    if (pc.greedy) {
        return;
    }
    auto * cp = common_sampler_get_candidates(cpu, false);
    auto * gp = common_sampler_get_candidates(gpu, false);

    // both lists must hold probabilities, the same check on each side (a NaN would pass p <= 0 tests)
    for (const auto * cands : { cp, gp }) {
        for (size_t i = 0; i < cands->size; ++i) {
            const float p = cands->data[i].p;
            if (!std::isfinite(p) || p < 0.0f || p > 1.0f + 1e-5f) {
                st.n_set_diff++;
                st.max_p_diff = INFINITY;
                return;
            }
        }
    }

    // the CPU array keeps what the chain kept; the backend one, once compacted, too (p > 0 for both)
    std::unordered_map<llama_token, float> cmap;
    for (size_t i = 0; i < cp->size; ++i) {
        if (cp->data[i].p > 0.0f) {
            cmap[cp->data[i].id] = cp->data[i].p;
        }
    }
    bool set_diff = false;
    for (size_t i = 0; i < gp->size; ++i) {
        if (gp->data[i].p <= 0.0f) {
            continue;
        }
        const auto it = cmap.find(gp->data[i].id);
        if (it == cmap.end()) {
            // a candidate the CPU dropped: only acceptable if its probability is negligible (rounding at a cut)
            set_diff = set_diff || gp->data[i].p > 1e-6f;
            st.max_p_diff = std::max(st.max_p_diff, gp->data[i].p);
            continue;
        }
        st.max_p_diff = std::max(st.max_p_diff, fabsf(it->second - gp->data[i].p));
    }
    // every CPU candidate must be on the backend side too, also when both lists have the same size
    {
        std::unordered_map<llama_token, float> gmap;
        for (size_t i = 0; i < gp->size; ++i) {
            if (gp->data[i].p > 0.0f) {
                gmap[gp->data[i].id] = gp->data[i].p;
            }
        }
        for (const auto & [id, p] : cmap) {
            if (gmap.find(id) == gmap.end()) {
                set_diff = set_diff || p > 1e-6f;
                st.max_p_diff = std::max(st.max_p_diff, p);
            }
        }
    }
    st.n_set_diff += set_diff ? 1 : 0;
}

// single output per sequence, n_seeds sequences per prompt
static parity_stats run_parity_single(const test_params & params, const parity_case & pc, int n_seeds) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<std::string> prompts = {
        "The capital of France is",
        "Once upon a time, there",
    };

    struct seq_info {
        llama_seq_id seq_id = 0;
        std::vector<llama_token> tokens = {};
        uint32_t seed = 0;
        common_sampler * cpu = nullptr;
        common_sampler * gpu = nullptr;
        bool has_dist = false;
        std::mt19937 rng = {}; // replays the CPU dist sampler's draws
    };
    std::vector<seq_info> seqs;
    for (size_t ip = 0; ip < prompts.size(); ++ip) {
        for (int s = 0; s < n_seeds; ++s) {
            seqs.push_back({ (llama_seq_id) seqs.size(), tokenize_prompt(vocab, prompts[ip]), (uint32_t) (1234 + 7919*s + ip) });
        }
    }

    std::vector<llama_sampler_seq_config> configs;
    for (auto & si : seqs) {
        common_params_sampling sp = parity_base_params();
        pc.setup(sp);
        sp.seed = si.seed;
        si.has_dist = !pc.greedy && sp.temp > 0.0f && sp.mirostat == 0;
        si.rng.seed(si.seed);
        sp.backend_sampling = false;
        si.cpu = common_sampler_init(params.model.get(), sp);
        sp.backend_sampling = true;
        si.gpu = common_sampler_init(params.model.get(), sp);
        // penalties need a history: the prompt
        for (llama_token t : si.tokens) {
            common_sampler_accept(si.cpu, t, false);
            common_sampler_accept(si.gpu, t, false);
        }
        configs.push_back({ si.seq_id, common_sampler_get(si.gpu) });
    }

    std::vector<llama_sampler_seq_config> no_configs;
    test_context ctx_cpu(params, no_configs, (int32_t) seqs.size());
    test_context ctx_gpu(params, configs,    (int32_t) seqs.size());

    common_batch batch_cpu(ctx_cpu.ctx.get());
    common_batch batch_gpu(ctx_gpu.ctx.get());
    std::vector<int32_t> idx;
    for (auto & si : seqs) {
        for (size_t i = 0; i < si.tokens.size(); ++i) {
            const bool out = i + 1 == si.tokens.size();
            batch_cpu.add(si.tokens[i], (llama_pos) i, si.seq_id, out);
            batch_gpu.add(si.tokens[i], (llama_pos) i, si.seq_id, out);
        }
        idx.push_back(batch_cpu.size() - 1);
    }
    GGML_ASSERT(llama_process(ctx_cpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_cpu.get()) == 0);
    GGML_ASSERT(llama_process(ctx_gpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_gpu.get()) == 0);

    parity_stats st;
    for (size_t i = 0; i < seqs.size(); ++i) {
        const llama_token tc = common_sampler_sample(seqs[i].cpu, ctx_cpu.ctx.get(), idx[i]);
        const llama_token tg = common_sampler_sample(seqs[i].gpu, ctx_gpu.ctx.get(), idx[i]);
        const double u = std::uniform_real_distribution<double>(0.0, 1.0)(seqs[i].rng);
        st.n_draws++;
        st.n_equal += tc == tg;
        if (tc != tg && st.n_draws - st.n_equal <= 3) {
            const float * lg = llama_get_logits_ith(ctx_cpu.ctx.get(), idx[i]);
            printf("  %s: seq %zu cpu %d gpu %d logits %.6f %.6f\n", pc.name.c_str(), i, tc, tg, lg[tc], lg[tg]);
        }
        if (tc != tg && seqs[i].has_dist &&
            parity_near_tie(pc.name.c_str(), (int) i, 0, seqs[i].cpu, seqs[i].gpu, tc, tg, u)) {
            st.n_tied++;
        }
        compare_candidates(pc, seqs[i].cpu, seqs[i].gpu, st);
    }

    for (auto & si : seqs) {
        common_sampler_free(si.cpu);
        common_sampler_free(si.gpu);
    }

    return st;
}

// draft verification: one sequence outputs n_draft + 1 rows; row j must be sampled with the history extended by
// the draft tokens 1..j (the server accepts them before it samples row j), as the CPU sampler does
static parity_stats run_parity_multi(const test_params & params, const parity_case & pc, int n_seeds, uint32_t n_ubatch) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const std::vector<llama_token> prompt = tokenize_prompt(vocab, "The quick brown fox jumps over the lazy dog. The quick brown fox");
    // rows: the last prompt token, then the draft tokens; draft[j] is the input of row j + 1
    const std::vector<llama_token> draft  = tokenize_prompt(vocab, " jumps over the lazy dog. The", false);
    const uint32_t n_rows = (uint32_t) draft.size() + 1;
    GGML_ASSERT(n_rows >= 3);

    struct seq_info {
        llama_seq_id seq_id = 0;
        uint32_t seed = 0;
        common_sampler * cpu = nullptr;
        common_sampler * gpu = nullptr;
        std::vector<int32_t> idx = {};
        bool has_dist = false;
        std::mt19937 rng = {}; // replays the CPU dist sampler's draws
    };
    std::vector<seq_info> seqs;
    for (int s = 0; s < n_seeds; ++s) {
        seqs.push_back({ (llama_seq_id) s, (uint32_t) (4321 + 104729*s) });
    }

    std::vector<llama_sampler_seq_config> configs;
    for (auto & si : seqs) {
        common_params_sampling sp = parity_base_params();
        pc.setup(sp);
        sp.seed = si.seed;
        si.has_dist = !pc.greedy && sp.temp > 0.0f && sp.mirostat == 0;
        si.rng.seed(si.seed);
        sp.backend_sampling = false;
        si.cpu = common_sampler_init(params.model.get(), sp);
        sp.backend_sampling = true;
        si.gpu = common_sampler_init(params.model.get(), sp);
        for (llama_token t : prompt) {
            common_sampler_accept(si.cpu, t, false);
            common_sampler_accept(si.gpu, t, false);
        }
        configs.push_back({ si.seq_id, common_sampler_get(si.gpu) });
    }

    std::vector<llama_sampler_seq_config> no_configs;
    test_context ctx_cpu(params, no_configs, (int32_t) seqs.size(), 0, n_ubatch, n_rows);
    test_context ctx_gpu(params, configs,    (int32_t) seqs.size(), 0, n_ubatch, n_rows);

    common_batch batch_cpu(ctx_cpu.ctx.get());
    common_batch batch_gpu(ctx_gpu.ctx.get());
    for (auto & si : seqs) {
        llama_pos pos = 0;
        for (size_t i = 0; i < prompt.size(); ++i, ++pos) {
            const bool out = i + 1 == prompt.size();
            batch_cpu.add(prompt[i], pos, si.seq_id, out);
            batch_gpu.add(prompt[i], pos, si.seq_id, out);
            if (out) {
                si.idx.push_back(batch_cpu.size() - 1);
            }
        }
        for (uint32_t j = 1; j < n_rows; ++j, ++pos) {
            batch_cpu.add(draft[j - 1], pos, si.seq_id, true);
            batch_gpu.add(draft[j - 1], pos, si.seq_id, true);
            si.idx.push_back(batch_cpu.size() - 1);
        }
    }
    GGML_ASSERT(llama_process(ctx_cpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_cpu.get()) == 0);
    GGML_ASSERT(llama_process(ctx_gpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_gpu.get()) == 0);

    parity_stats st;
    for (auto & si : seqs) {
        for (uint32_t j = 0; j < n_rows; ++j) {
            const llama_token tc = common_sampler_sample(si.cpu, ctx_cpu.ctx.get(), si.idx[j]);
            const llama_token tg = common_sampler_sample(si.gpu, ctx_gpu.ctx.get(), si.idx[j]);
            const double u = std::uniform_real_distribution<double>(0.0, 1.0)(si.rng);
            st.n_draws++;
            st.n_equal += tc == tg;
            if (tc != tg && st.n_draws - st.n_equal <= 3) {
                printf("  %s: seq %d row %u cpu %d gpu %d\n", pc.name.c_str(), si.seq_id, j, tc, tg);
            }
            if (tc != tg && si.has_dist &&
                parity_near_tie(pc.name.c_str(), si.seq_id, j, si.cpu, si.gpu, tc, tg, u)) {
                st.n_tied++;
            }
            {
                const int before = st.n_set_diff; const float mb = st.max_p_diff; st.max_p_diff = 0;
                compare_candidates(pc, si.cpu, si.gpu, st);
                if (getenv("PARITY_DEBUG") && (st.n_set_diff != before || st.max_p_diff > 1e-4f)) {
                    printf("  DBG %s seq %d row %u idx %d cpu_n %zu gpu_n %zu probs_count %u cand_count %u maxd %.3g tok %d/%d\n", pc.name.c_str(), si.seq_id, j, si.idx[j],
                        common_sampler_get_candidates(si.cpu, false)->size, common_sampler_get_candidates(si.gpu, false)->size,
                        llama_get_sampled_probs_count_ith(ctx_gpu.ctx.get(), si.idx[j]), llama_get_sampled_candidates_count_ith(ctx_gpu.ctx.get(), si.idx[j]), st.max_p_diff, tc, tg);
                }
                if (getenv("PARITY_DEBUG") && (st.n_set_diff != before || st.max_p_diff > 1e-4f)) {
                    auto * cp = common_sampler_get_candidates(si.cpu, true);
                    auto * gp = common_sampler_get_candidates(si.gpu, true);
                    printf("    cpu:"); for (size_t q = 0; q < std::min<size_t>(4, cp->size); ++q) printf(" (%d %.4f %.3f)", cp->data[q].id, cp->data[q].p, cp->data[q].logit); printf("\n");
                    printf("    gpu:"); for (size_t q = 0; q < std::min<size_t>(4, gp->size); ++q) printf(" (%d %.4f %.3f)", gp->data[q].id, gp->data[q].p, gp->data[q].logit); printf("\n");
                }
                st.max_p_diff = std::max(mb, st.max_p_diff);
            }
            // accept the draft token, as the server does when the sampled token matches it
            if (j + 1 < n_rows) {
                common_sampler_accept(si.cpu, draft[j], true);
                common_sampler_accept(si.gpu, draft[j], true);
            }
        }
    }

    for (auto & si : seqs) {
        common_sampler_free(si.cpu);
        common_sampler_free(si.gpu);
    }

    return st;
}

static std::vector<parity_case> parity_cases(const llama_vocab * vocab) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token paris = tokenize_prompt(vocab, " Paris", false)[0];

    using sp_t = common_params_sampling;
    std::vector<parity_case> cases = {
        // Studio default chains (studio/backend/assets/configs/inference_defaults.json)
        { "qwen3",       [](sp_t & sp) { sp.temp = 0.6f; sp.top_p = 0.95f; sp.top_k = 20; } },
        { "qwen3.5",     [](sp_t & sp) { sp.temp = 0.6f; sp.top_p = 0.95f; sp.top_k = 20; sp.penalty_present = 1.5f; } },
        { "qwen3.5-nt",  [](sp_t & sp) { sp.temp = 0.7f; sp.top_p = 0.8f;  sp.top_k = 20; sp.penalty_present = 1.5f; } },
        { "gemma-4",     [](sp_t & sp) { sp.temp = 1.0f; sp.top_p = 0.95f; sp.top_k = 64; } },
        { "gpt-oss",     [](sp_t & sp) { sp.temp = 1.0f; sp.top_p = 1.0f;  sp.min_p = 0.01f; } },
        { "llama-3",     [](sp_t & sp) { sp.temp = 1.5f; sp.top_p = 0.95f; sp.min_p = 0.1f; } },
        { "mistral",     [](sp_t & sp) { sp.temp = 0.7f; sp.top_p = 0.95f; sp.min_p = 0.01f; } },
        { "lfm2",        [](sp_t & sp) { sp.temp = 0.1f; sp.top_p = 0.1f;  sp.top_k = 50; sp.min_p = 0.15f; sp.penalty_repeat = 1.05f; } },
        { "granite",     [](sp_t & sp) { sp.temp = 0.0f; sp.min_p = 0.01f; }, true },
        // penalties of every kind, before the filters
        { "penalties",   [](sp_t & sp) { sp.temp = 0.8f; sp.top_p = 0.9f; sp.top_k = 40; sp.penalty_repeat = 1.3f; sp.penalty_freq = 0.7f; sp.penalty_present = 0.4f; } },
        // top-k past the candidates (aborted the server)
        { "top_k>n_vocab", [](sp_t & sp) { sp.top_k = 1000000; sp.top_p = 0.9f; } },
        { "top_k=140000",  [](sp_t & sp) { sp.top_k = 140000;  sp.top_p = 0.9f; } },
        { "full_nofilter_k",  [](sp_t & sp) { sp.top_k = 1000000; } },
        { "full_nofilter",    [](sp_t & sp) { sp.top_k = 0; } },
        { "temp0>top_k",   [](sp_t & sp) { sp.temp = 0.0f; sp.top_k = 40; sp.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE, COMMON_SAMPLER_TYPE_TOP_K }; sp.n_probs = 1; } },
        // top-k without top-p: dist over the top-k order (was unsorted, seed not reproducible)
        { "top_k_only",  [](sp_t & sp) { sp.top_k = 40; } },
        { "top_k=1",     [](sp_t & sp) { sp.top_k = 1; }, true },
        // min-p
        { "min_p=1",     [](sp_t & sp) { sp.min_p = 1.0f; } },
        { "min_p=0.999", [](sp_t & sp) { sp.min_p = 0.999f; } },
        { "min_p_keep3", [](sp_t & sp) { sp.min_p = 0.9f; sp.min_keep = 3; } },
        { "min_p=2",     [](sp_t & sp) { sp.min_p = 2.0f; sp.min_keep = 2; } },
        // top-p
        { "top_p_keep3", [](sp_t & sp) { sp.top_k = 3; sp.top_p = 0.1f; sp.min_keep = 3; } },
        { "top_p=0",     [](sp_t & sp) { sp.top_p = 0.0f; } },
        { "top_p_full",  [](sp_t & sp) { sp.top_p = 0.9f; } },
        // dynamic temperature
        { "dynatemp_t0", [](sp_t & sp) { sp.top_k = 8; sp.temp = 0.0f; sp.dynatemp_range = 1.0f; } },
        { "dynatemp",    [](sp_t & sp) { sp.top_p = 0.95f; sp.temp = 0.8f; sp.dynatemp_range = 0.5f; sp.dynatemp_exponent = 1.5f; } },
        { "dynatemp_e0", [](sp_t & sp) { sp.top_k = 40; sp.temp = 0.7f; sp.dynatemp_range = 0.3f; sp.dynatemp_exponent = 0.0f; } },
        // logit bias: a repeated token sums its biases
        { "logit_bias_dup", [paris](sp_t & sp) { sp.top_k = 8; sp.logit_bias = { { paris, 10.0f }, { paris, -100.0f } }; } },
        { "logit_bias",     [paris, n_vocab](sp_t & sp) { sp.top_k = 8; sp.logit_bias = { { paris, -3.0f }, { n_vocab - 1, 30.0f } }; } },
        // a CPU suffix after a backend prefix (XTC has no backend implementation): it must not see removed candidates
        { "xtc_suffix",  [](sp_t & sp) { sp.top_k = 3; sp.top_p = 0.8f; sp.xtc_probability = 1.0f; sp.xtc_threshold = 0.0f; } },
        { "typical_suffix", [](sp_t & sp) { sp.top_k = 40; sp.top_p = 0.9f; sp.typ_p = 0.5f;
                                            sp.samplers = { COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TYPICAL_P, COMMON_SAMPLER_TYPE_TEMPERATURE }; } },
    };
    return cases;
}

// A backend sampler chain on given logits, computed on one backend device, against the same chain on the CPU
// (llama_sampler_apply): cases that need exact logits (ties, a single candidate, -INF entries) a model rarely gives.
struct synth_candidates {
    std::map<llama_token, float> finite; // id -> logit of the candidates that are not -INF
    int n_dims = 1;                      // of the exported logits
};

static synth_candidates synth_run_cpu(llama_sampler * chain, const std::vector<float> & logits) {
    std::vector<llama_token_data> cur;
    for (size_t i = 0; i < logits.size(); ++i) {
        cur.push_back({ (llama_token) i, logits[i], 0.0f });
    }
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(chain, &arr);
    synth_candidates res;
    for (size_t i = 0; i < arr.size; ++i) {
        if (arr.data[i].logit != -INFINITY) {
            res.finite[arr.data[i].id] = arr.data[i].logit;
        }
    }
    return res;
}

static synth_candidates synth_run_backend(llama_sampler * chain, const std::vector<float> & logits, ggml_backend_dev_t dev) {
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    GGML_ASSERT(backend != nullptr);
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backend);
    GGML_ASSERT(chain->iface->backend_init(chain, buft, 1) && "every sampler of a synthetic case runs on the backend");

    const size_t n_graph = 4096;
    ggml_init_params ip = { ggml_tensor_overhead()*n_graph + ggml_graph_overhead_custom(n_graph, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, n_graph, false);

    ggml_tensor * inp = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) logits.size());
    ggml_set_input(inp);
    if (chain->iface->backend_reset) {
        chain->iface->backend_reset(chain);
    }
    llama_sampler_data data = { inp, nullptr, nullptr, nullptr };
    chain->iface->backend_apply(chain, ctx, gf, &data);

    synth_candidates res;
    res.n_dims = ggml_n_dims(data.logits);
    ggml_tensor * out_logits = ggml_cont(ctx, ggml_reshape_1d(ctx, data.logits, ggml_nelements(data.logits)));
    ggml_set_output(out_logits);
    ggml_build_forward_expand(gf, out_logits);
    ggml_tensor * out_ids = nullptr;
    if (data.candidates) {
        out_ids = ggml_cont(ctx, ggml_reshape_1d(ctx, data.candidates, ggml_nelements(data.candidates)));
        ggml_set_output(out_ids);
        ggml_build_forward_expand(gf, out_ids);
    }

    ggml_gallocr_t galloc = ggml_gallocr_new(buft);
    GGML_ASSERT(ggml_gallocr_alloc_graph(galloc, gf));
    ggml_backend_tensor_set(inp, logits.data(), 0, logits.size()*sizeof(float));
    if (chain->iface->backend_set_input) {
        chain->iface->backend_set_input(chain);
    }
    GGML_ASSERT(ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS);

    std::vector<float>   lg(ggml_nelements(out_logits));
    std::vector<int32_t> ids(lg.size());
    ggml_backend_tensor_get(out_logits, lg.data(), 0, lg.size()*sizeof(float));
    if (out_ids) {
        GGML_ASSERT(ggml_nelements(out_ids) == (int64_t) lg.size());
        ggml_backend_tensor_get(out_ids, ids.data(), 0, ids.size()*sizeof(int32_t));
    } else {
        for (size_t i = 0; i < ids.size(); ++i) {
            ids[i] = (int32_t) i;
        }
    }
    for (size_t i = 0; i < lg.size(); ++i) {
        if (lg[i] != -INFINITY) {
            res.finite[ids[i]] = lg[i];
        }
    }

    ggml_gallocr_free(galloc);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return res;
}

static void test_synthetic_parity(const test_params & params) {
    GGML_UNUSED(params);

    struct synth_case {
        const char * name;
        std::vector<float> logits;
        std::function<void(llama_sampler *)> build;
    };
    const float ninf = -INFINITY;
    const std::vector<synth_case> cases = {
        // min-p fallback with a tie at the cut: the CPU keeps exactly min_keep candidates
        { "min_p_keep_tie", { 2, 1, 1, 0 }, [](llama_sampler * c) { llama_sampler_chain_add(c, llama_sampler_init_min_p(1.0f, 2)); } },
        // dynamic temperature whose whole range is <= 0 is greedy on the CPU (the first maximum)
        { "dynatemp_neg_range", { 4, 4, 2, 1 }, [](llama_sampler * c) { llama_sampler_chain_add(c, llama_sampler_init_temp_ext(-1.0f, 0.5f, 1.0f)); } },
        // one candidate left: the CPU leaves its logit as it is
        { "dynatemp_one_cand", { 3, ninf, ninf, ninf }, [](llama_sampler * c) { llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.5f, 0.5f, 1.0f)); } },
        // a -INF candidate is not a candidate for the maximum entropy, on both sides
        { "dynatemp_masked", { 1, 4, 2, 3 }, [](llama_sampler * c) {
            const llama_logit_bias lb[] = { { 0, -INFINITY } };
            llama_sampler_chain_add(c, llama_sampler_init_logit_bias(4, 1, lb));
            llama_sampler_chain_add(c, llama_sampler_init_temp_ext(0.8f, 0.5f, 1.5f)); } },
        // duplicate biases after a reordering sampler: summed on both sides
        { "logit_bias_dup_reordered", { 0, -1, 1, 2 }, [](llama_sampler * c) {
            const llama_logit_bias lb[] = { { 3, 10.0f }, { 3, -100.0f } };
            llama_sampler_chain_add(c, llama_sampler_init_top_k(2));
            llama_sampler_chain_add(c, llama_sampler_init_logit_bias(4, 2, lb)); } },
        // top-k exports 1-D logits like every other backend sampler
        { "top_k_temp", { 0, 3, 1, 2, 5 }, [](llama_sampler * c) {
            llama_sampler_chain_add(c, llama_sampler_init_top_k(3));
            llama_sampler_chain_add(c, llama_sampler_init_temp(0.5f)); } },
    };

    std::vector<ggml_backend_dev_t> devs;
    if (auto * d = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU)) {
        devs.push_back(d);
    }
    devs.push_back(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));

    int n_fail = 0;
    for (auto * dev : devs) {
        for (const auto & sc : cases) {
            llama_sampler * c_cpu = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler * c_bk  = llama_sampler_chain_init(llama_sampler_chain_default_params());
            sc.build(c_cpu);
            sc.build(c_bk);
            const synth_candidates rc = synth_run_cpu(c_cpu, sc.logits);
            const synth_candidates rb = synth_run_backend(c_bk, sc.logits, dev);
            bool ok = rc.finite.size() == rb.finite.size() && rb.n_dims == 1;
            for (const auto & [id, l] : rc.finite) {
                const auto it = rb.finite.find(id);
                ok = ok && it != rb.finite.end() && fabsf(it->second - l) <= 1e-4f*std::max(1.0f, fabsf(l));
            }
            printf("synthetic %-26s %-6s cpu %zu cand, backend %zu cand (dims %d) %s\n", sc.name, ggml_backend_dev_name(dev),
                    rc.finite.size(), rb.finite.size(), rb.n_dims, ok ? "OK" : "FAIL");
            if (!ok) {
                for (const auto & [id, l] : rc.finite) { printf("    cpu %d %.6g\n", id, l); }
                for (const auto & [id, l] : rb.finite) { printf("    bk  %d %.6g\n", id, l); }
            }
            n_fail += ok ? 0 : 1;
            llama_sampler_free(c_cpu);
            llama_sampler_free(c_bk);
        }
    }
    printf("synthetic parity: %d failing cases\n", n_fail);
    GGML_ASSERT(n_fail == 0);
    printf("backend synthetic parity test PASSED\n");
}

// a backend prefix that masks every candidate of a row: the CPU samplers that follow get this row's candidates as
// they are (all -INF), the same as without a backend, never an empty, default (token 0) or previous row
static void test_all_masked_row(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<llama_logit_bias> all_inf;
    for (llama_token t = 0; t < n_vocab; ++t) {
        all_inf.push_back({ t, -INFINITY });
    }

    // common_sampler path: logit bias (backend) masks everything, XTC (CPU only) follows
    common_params_sampling sp = parity_base_params();
    sp.logit_bias = all_inf;
    sp.xtc_probability = 1.0f;
    sp.xtc_threshold   = 0.0f;
    sp.seed = 7;
    sp.backend_sampling = true;
    common_sampler * gpu = common_sampler_init(params.model.get(), sp);

    // the same chain directly through llama_sampler_sample
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_logit_bias(n_vocab, (int32_t) all_inf.size(), all_inf.data()));
    llama_sampler_chain_add(chain, llama_sampler_init_xtc(1.0f, 0.0f, 0, 7));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(7));
    llama_sampler_ptr chain_owner(chain);

    std::vector<llama_sampler_seq_config> configs = { { 0, common_sampler_get(gpu) }, { 1, chain } };
    test_context tctx(params, configs, 2);

    const std::vector<llama_token> prompt = tokenize_prompt(vocab, "The capital of France is");
    common_batch batch(tctx.ctx.get());
    for (llama_seq_id s = 0; s < 2; ++s) {
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch.add(prompt[i], (llama_pos) i, s, i + 1 == prompt.size());
        }
    }
    GGML_ASSERT(llama_process(tctx.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
    const int32_t idx0 = (int32_t) prompt.size() - 1;
    const int32_t idx1 = 2*(int32_t) prompt.size() - 1;

    (void) common_sampler_sample(gpu, tctx.ctx.get(), idx0);
    const auto * cp = common_sampler_get_candidates(gpu, false);
    std::unordered_set<llama_token> ids;
    bool all_inf_logits = true;
    for (size_t i = 0; i < cp->size; ++i) {
        ids.insert(cp->data[i].id);
        all_inf_logits = all_inf_logits && cp->data[i].logit == -INFINITY;
    }
    printf("all-masked row (common_sampler): %zu candidates, %zu distinct ids, all -INF: %d\n", cp->size, ids.size(), all_inf_logits);
    GGML_ASSERT(cp->size == (size_t) n_vocab && ids.size() == (size_t) n_vocab && all_inf_logits);

    // CPU reference of the same chain on the same (all -INF) candidates: the token the CPU samplers pick
    llama_sampler * ref = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(ref, llama_sampler_init_xtc(1.0f, 0.0f, 0, 7));
    llama_sampler_chain_add(ref, llama_sampler_init_dist(7));
    std::vector<llama_token_data> cur;
    for (llama_token t = 0; t < n_vocab; ++t) {
        cur.push_back({ t, -INFINITY, 0.0f });
    }
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(ref, &arr);
    const llama_token t_ref = arr.data[arr.selected].id;
    const llama_token t_bk  = llama_sampler_sample(chain, tctx.ctx.get(), idx1);
    printf("all-masked row (llama_sampler_sample): token %d, CPU reference %d\n", t_bk, t_ref);
    GGML_ASSERT(t_bk == t_ref);

    llama_sampler_free(ref);
    common_sampler_free(gpu);
    printf("backend all-masked row test PASSED\n");
}

static void test_cpu_parity(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int n_seeds = getenv("PARITY_SEEDS") ? atoi(getenv("PARITY_SEEDS")) : 24;

    int n_draws = 0;
    int n_fail  = 0;
    int n_tied  = 0;
    for (const auto & pc : parity_cases(vocab)) {
        const parity_stats st = run_parity_single(params, pc, n_seeds);
        const bool ok = st.n_equal + st.n_tied == st.n_draws && st.n_set_diff == 0 && st.max_p_diff < 1e-4f;
        printf("parity %-16s draws %4d equal %4d ties %d set_diff %d max_p_diff %.2e %s\n",
                pc.name.c_str(), st.n_draws, st.n_equal, st.n_tied, st.n_set_diff, st.max_p_diff, ok ? "OK" : "FAIL");
        n_draws += st.n_draws;
        n_fail  += ok ? 0 : 1;
        n_tied  += st.n_tied;
    }
    printf("parity single-output: %d draws, %d tolerated near-ties, %d failing cases\n", n_draws, n_tied, n_fail);
    GGML_ASSERT(n_fail == 0);

    printf("backend CPU parity test PASSED\n");
}

static void test_cpu_parity_multi_output(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int n_seeds = getenv("PARITY_SEEDS") ? atoi(getenv("PARITY_SEEDS")) / 2 : 12;

    int n_draws = 0;
    int n_fail  = 0;
    int n_tied  = 0;
    for (const auto & pc : parity_cases(vocab)) {
        // n_ubatch 8 splits a sequence's rows across ubatches
        for (uint32_t n_ubatch : { 512u, 8u }) {
            const parity_stats st = run_parity_multi(params, pc, n_seeds, n_ubatch);
            const bool ok = st.n_equal + st.n_tied == st.n_draws && st.n_set_diff == 0 && st.max_p_diff < 1e-4f;
            printf("parity-multi %-16s ubatch %3u draws %4d equal %4d ties %d set_diff %d max_p_diff %.2e %s\n",
                    pc.name.c_str(), n_ubatch, st.n_draws, st.n_equal, st.n_tied, st.n_set_diff, st.max_p_diff, ok ? "OK" : "FAIL");
            n_draws += st.n_draws;
            n_fail  += ok ? 0 : 1;
            n_tied  += st.n_tied;
        }
    }
    printf("parity multi-output: %d draws, %d tolerated near-ties, %d failing cases\n", n_draws, n_tied, n_fail);
    GGML_ASSERT(n_fail == 0);

    printf("backend CPU parity multi-output test PASSED\n");
}

// Draft verification rows of several sequences split across ubatches: every row's sampled probabilities must be
// the softmax of its own logits. The graph has the same shape in every ubatch and only the chains of the rows a
// ubatch samples are active; when the allocator kept a plan made for other active chains, a later tensor
// overwrote the results of an active chain before they were read back.
static void test_multi_output_ubatch_split(const test_params & params) {
    const llama_vocab * vocab = llama_model_get_vocab(params.model.get());
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    const std::vector<llama_token> prompt = tokenize_prompt(vocab, "The quick brown fox jumps over the lazy dog. The quick brown fox");
    const std::vector<llama_token> draft  = tokenize_prompt(vocab, " jumps over the lazy dog. The", false);
    const uint32_t n_rows = (uint32_t) draft.size() + 1;

    const int n_seqs = 2;
    std::vector<llama_sampler_seq_config> no_configs;

    for (uint32_t n_ubatch : { 512u, 8u }) {
        // a chain is initialised for one context only
        std::vector<llama_sampler_ptr> chains;
        std::vector<llama_sampler_seq_config> configs;
        for (int s = 0; s < n_seqs; ++s) {
            chains.emplace_back(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(chains.back().get(), llama_sampler_init_dist(1000 + s));
            configs.push_back({ s, chains.back().get() });
        }

        test_context ctx_cpu(params, no_configs, n_seqs, 0, n_ubatch, n_rows);
        test_context ctx_gpu(params, configs,    n_seqs, 0, n_ubatch, n_rows);

        common_batch batch_cpu(ctx_cpu.ctx.get());
        common_batch batch_gpu(ctx_gpu.ctx.get());
        std::vector<int32_t> idx;
        for (int s = 0; s < n_seqs; ++s) {
            llama_pos pos = 0;
            for (size_t i = 0; i < prompt.size(); ++i, ++pos) {
                const bool out = i + 1 == prompt.size();
                batch_cpu.add(prompt[i], pos, s, out);
                batch_gpu.add(prompt[i], pos, s, out);
                if (out) {
                    idx.push_back(batch_cpu.size() - 1);
                }
            }
            for (llama_token t : draft) {
                batch_cpu.add(t, pos, s, true);
                batch_gpu.add(t, pos, s, true);
                idx.push_back(batch_cpu.size() - 1);
                ++pos;
            }
        }
        GGML_ASSERT(llama_process(ctx_cpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_cpu.get()) == 0);
        GGML_ASSERT(llama_process(ctx_gpu.ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch_gpu.get()) == 0);
        llama_synchronize(ctx_gpu.ctx.get());

        int n_bad = 0;
        float max_diff = 0.0f;
        for (size_t r = 0; r < idx.size(); ++r) {
            const float * logits = llama_get_logits_ith(ctx_cpu.ctx.get(), idx[r]);
            const float * probs  = llama_get_sampled_probs_ith(ctx_gpu.ctx.get(), idx[r]);
            GGML_ASSERT(logits && probs);
            GGML_ASSERT(llama_get_sampled_probs_count_ith(ctx_gpu.ctx.get(), idx[r]) == (uint32_t) n_vocab);
            double m = -INFINITY, z = 0.0;
            for (int i = 0; i < n_vocab; ++i) { m = std::max(m, (double) logits[i]); }
            for (int i = 0; i < n_vocab; ++i) { z += exp(logits[i] - m); }
            float d = 0.0f;
            for (int i = 0; i < n_vocab; ++i) {
                d = std::max(d, fabsf((float) (exp(logits[i] - m)/z) - probs[i]));
            }
            if (!(d < 1e-4f)) {
                if (n_bad < 4) {
                    printf("  ubatch %u: row %zu (batch index %d): sampled probs differ from the softmax by %g\n", n_ubatch, r, idx[r], d);
                }
                ++n_bad;
            }
            max_diff = std::max(max_diff, std::isnan(d) ? INFINITY : d);
        }
        printf("multi-output ubatch split: n_ubatch %u, %zu rows, %d wrong, max diff %g\n", n_ubatch, idx.size(), n_bad, max_diff);
        GGML_ASSERT(n_bad == 0);
    }

    printf("backend multi-output ubatch split test PASSED\n");
}

struct backend_test_case {
    std::string name;
    void (*fn)(const test_params &);
    bool enabled_by_default;
};

// note: test names are "test_<suffix>" and match the function implementing them
static const backend_test_case BACKEND_TESTS[] = {
    // single sampler
    { "test_greedy",                 test_greedy,                 true },
    { "test_greedy_filtered",        test_greedy_filtered,        true },
    { "test_greedy_filtered_common", test_greedy_filtered_common, true },
    { "test_temp",                   test_temp,                   true },
    { "test_temp_ext",               test_temp_ext,               true },
    { "test_top_k",                  test_top_k,                  true },
    { "test_top_p",                  test_top_p,                  true },
    { "test_min_p",                  test_min_p,                  true },
    { "test_logit_bias",             test_logit_bias,             true },
    { "test_penalties",              test_penalties,              true },
    // multiple sequences
    { "test_multi_sequence",         test_multi_sequence,         true },
    { "test_dist",                   test_dist,                   true },
    { "test_dist_and_cpu",           test_dist_and_cpu,           true },
    { "test_mixed",                  test_mixed,                  true },
    { "test_cpu_mixed",              test_cpu_mixed,              true },
    { "test_set_sampler",            test_set_sampler,            true },
    // multiple outputs per sequence
    { "test_multi_output_limit",                  test_multi_output_limit,               true },
    { "test_multi_output_multi_sequence_dist",    test_multi_output_multi_sequence_dist, true },
    { "test_multi_output_dist_transaction",       test_multi_output_dist_transaction,    true },
    { "test_multi_output_sampling_chain",         test_multi_output_sampling_chain,      true },
    { "test_multi_output_cpu_suffix",             test_multi_output_cpu_suffix,          true },
    { "test_multi_output_ubatch_split",           test_multi_output_ubatch_split,        true },
    // GPU vs CPU parity of whole chains, per sampler and edge value
    { "test_synthetic_parity",                    test_synthetic_parity,                 true },
    { "test_all_masked_row",                      test_all_masked_row,                   true },
    { "test_cpu_parity",                          test_cpu_parity,                       true },
    { "test_cpu_parity_multi_output",             test_cpu_parity_multi_output,          true },
};

// TODO: add usage, examples
static test_args parse_cli(int argc, char ** argv) {
    test_args out;

    for (int i = 1; i < argc; ++i) {
        const char * arg = argv[i];

        if (std::strcmp(arg, "--test") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--test expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.test = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--test=", 7) == 0) {
            out.test = arg + 7;
            continue;
        }
        if (std::strcmp(arg, "--model") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--model expects a value\n");
                exit(EXIT_FAILURE);
            }
            out.model = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--model=", 8) == 0) {
            out.model = arg + 8;
            continue;
        }
        if (std::strcmp(arg, "--device") == 0) {
            if (i + 1 >= argc) {
                fprintf(stderr, "--device expects a value (cpu or gpu)\n");
                exit(EXIT_FAILURE);
            }
            out.device = argv[++i];
            continue;
        }
        if (std::strncmp(arg, "--device=", 9) == 0) {
            out.device = arg + 9;
            continue;
        }
        if (out.model.empty()) {
            out.model = arg;
            continue;
        }

        fprintf(stderr, "Unexpected argument: %s\n", arg);
        exit(EXIT_FAILURE);
    }

    if (out.device != "cpu" && out.device != "gpu" && out.device != "auto") {
        fprintf(stderr, "Invalid device '%s'. Must be 'cpu', 'gpu' or 'auto'\n", out.device.c_str());
        exit(EXIT_FAILURE);
    }

    return out;
}

static std::vector<const backend_test_case *> collect_tests_to_run(const std::string & requested) {
    std::vector<const backend_test_case *> selected;

    if (!requested.empty()) {
        for (const auto & test : BACKEND_TESTS) {
            if (test.name == requested) {
                selected.push_back(&test);
                break;
            }
        }
        if (selected.empty()) {
            fprintf(stderr, "Unknown test '%s'. Available tests:\n", requested.c_str());
            for (const auto & test : BACKEND_TESTS) {
                fprintf(stderr, "  %s\n", test.name.c_str());
            }
            exit(EXIT_FAILURE);
        }
    } else {
        for (const auto & test : BACKEND_TESTS) {
            if (!test.enabled_by_default) {
                continue;
            }
#ifdef GGML_USE_HIP
            // TODO: remove this when https://github.com/ggml-org/llama.cpp/pull/26592 is merged
            if (test.name == "test_penalties"                   ||
                test.name == "test_set_sampler"                 ||
                test.name == "test_mixed"                       ||
                test.name == "test_top_p"                       ||
                test.name == "test_multi_output_sampling_chain" ||
                test.name == "test_multi_output_cpu_suffix") {
                fprintf(stderr, "Skipping test '%s' on HIP backend (no backend TOP_K support)\n", test.name.c_str());
                continue;
            }
#endif // GGML_USE_HIP
            selected.push_back(&test);
        }
    }

    if (selected.empty()) {
        fprintf(stderr, "No backend sampling tests selected. Use --test=<name> to pick one.\n");
    }

    return selected;
}

static void run_tests(const std::vector<const backend_test_case *> & tests, const test_params & args) {
    for (const auto & test : tests) {
        fprintf(stderr, "\n=== %s ===\n", test->name.c_str());
        try {
            test->fn(args);
        } catch (const std::exception & e) {
            fprintf(stderr, "Error running test '%s': %s\n", test->name.c_str(), e.what());
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char ** argv) {
    test_args args = parse_cli(argc, argv);

    if (args.model.empty()) {
        args.model = common_get_model_or_exit(1, argv);
    }

    {
        std::ifstream file(args.model);
        if (!file.is_open()) {
            fprintf(stderr, "no model '%s' found\n", args.model.c_str());
            return EXIT_FAILURE;
        }
    }

    fprintf(stderr, "using '%s'\n", args.model.c_str());

    llama_backend_init();

    test_params params = {
        /*.model =*/ load_model(args),
    };

    const std::vector<const backend_test_case *> tests = collect_tests_to_run(args.test);
    if (!tests.empty()) {
        run_tests(tests, params);
    }

    return 0;
}
