#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <cstring>
#include <vector>

// max elements of the dense relative bias per attention call, and max chunks per layer
static constexpr int64_t n_kq_bias_max  = int64_t(1) << 29;
static constexpr int64_t n_kq_chunk_max = 64;
// ubatches up to this many tokens take the banded path over the padded n_kv
static constexpr int64_t n_tokens_banded_min = 32;

// the banded attention takes the distances from the positions when the cells are not the positions
// (several sequences in a unified cache, a wrapped SWA cache); LLAMA_INKLING_BANDED_POS=0 keeps those dense,
// =2 takes the positions for every banded ubatch (a check: the cells that are the positions give the same result)
// LLAMA_INKLING_BANDED_CAUSAL=0 keeps the mask input of the full-attention layers when the cache holds one sequence
static bool inkling2_banded_causal_enabled() {
    static const bool enabled = [] {
        const char * env = getenv("LLAMA_INKLING_BANDED_CAUSAL");
        return env == nullptr || atoi(env) != 0;
    }();
    return enabled;
}

// ubatches of 2 to this many tokens over at most n_kv_dense_max cells take the dense attention with the indexed
// relative bias (one FA tile kernel), not the banded op and its merge. On a B200 the dense path is faster for 2 to 8
// tokens up to 128K cells; 16 tokens lose from 64K cells. LLAMA_INKLING_DENSE_NMAX and LLAMA_INKLING_DENSE_KV_MAX
// override them (for sweeps; NMAX=1 restores the banded path for every ubatch of 2+ tokens)
static constexpr int64_t n_tokens_dense_max = 8;
static constexpr int64_t n_kv_dense_max     = INT64_MAX;

static int64_t inkling2_env_i64(const char * name, int64_t def) {
    const char * env = getenv(name);
    return env == nullptr ? def : std::max<int64_t>(0, atoll(env));
}

static int64_t inkling2_dense_n_max() {
    static const int64_t v = inkling2_env_i64("LLAMA_INKLING_DENSE_NMAX", n_tokens_dense_max);
    return v;
}

static int64_t inkling2_dense_kv_max() {
    static const int64_t v = inkling2_env_i64("LLAMA_INKLING_DENSE_KV_MAX", n_kv_dense_max);
    return v;
}

static int inkling2_banded_pos_mode() {
    static const int mode = [] {
        const char * env = getenv("LLAMA_INKLING_BANDED_POS");
        return env == nullptr ? 1 : std::clamp(atoi(env), 0, 2);
    }();
    return mode;
}

void llama_model_inkling2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_QK_CLIP,           hparams.f_qk_clip);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);
    ml.get_key(LLM_KV_SHORTCONV_L_CACHE,           hparams.n_shortconv_l_cache);

    // relative-position bias, and the log-scaled positions of the full-attention layers
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_LENGTH, hparams.n_relative);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_EXTENT, hparams.n_relative_extent);
    ml.get_key(LLM_KV_ATTENTION_RELATIVE_CLIP,   hparams.f_relative_clip);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_FLOOR, hparams.n_log_scale_floor);
    ml.get_key(LLM_KV_ATTENTION_LOG_SCALE_ALPHA, hparams.f_log_scale_alpha);

    // the embedding and output tensors are padded past the vocabulary
    ml.get_key(LLM_KV_VOCAB_SIZE,          hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);

    // MoE
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);

    ml.get_key(LLM_KV_ACTIVATION_SITU_BETA,        hparams.situ_beta);
    ml.get_key(LLM_KV_ACTIVATION_SITU_LINEAR_BETA, hparams.situ_linear_beta);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);

    hparams.f_final_logit_softcapping = 0.0f;
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    // sliding window
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,         hparams.n_swa);
    ml.get_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl);

    uint32_t n_pattern = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, n_pattern);

    if (n_pattern != hparams.n_layer() || !hparams.n_relative || !hparams.n_relative_extent || !hparams.n_swa ||
        !hparams.attn_res_block_size || !hparams.n_log_scale_floor || hparams.n_shortconv_l_cache < 2 ||
        hparams.n_shortconv_l_cache > 64 || hparams.n_relative_extent > INT32_MAX || hparams.n_swa > INT32_MAX ||
        !hparams.n_vocab_unpadded || hparams.n_vocab_unpadded > hparams.n_vocab_padded) {
        throw std::runtime_error("inkling2: invalid dimensions, vocabulary, or layer pattern");
    }

    for (float value : {hparams.f_norm_rms_eps, hparams.f_qk_clip, hparams.f_relative_clip, hparams.situ_beta, hparams.situ_linear_beta}) {
        if (!std::isfinite(value) || value <= 0.0f) {
            throw std::runtime_error("inkling2: normalization, clipping and scaling parameters must be finite and positive");
        }
    }

    if (!std::isfinite(hparams.f_log_scale_alpha) || hparams.f_log_scale_alpha < 0.0f) {
        throw std::runtime_error("inkling2: invalid logarithmic scale");
    }

    hparams.swa_type                = LLAMA_SWA_TYPE_STANDARD;
    hparams.expert_router_f32       = true;
    hparams.expert_weights_norm_eps = 1e-20f;

    // no RoPE, the relative-position bias carries the positions
    hparams.rope_pattern.fill(false);

    // the convolution state keeps K, V, the attention output and the FFN output of the widest layer
    uint64_t n_channels_max = 0;

    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
            hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_expert_used(il) ||
            hparams.n_expert_used(il) > hparams.n_expert || !hparams.n_ff_exp(il)) {
            throw std::runtime_error("inkling2: invalid attention or expert dimensions");
        }

        n_channels_max = std::max<uint64_t>(n_channels_max, 2ull * hparams.n_embd_k_gqa(il) + 2ull * hparams.n_embd);
    }

    if (n_channels_max * (hparams.n_shortconv_l_cache - 1) > UINT32_MAX) {
        throw std::runtime_error("inkling2: convolution state is too large");
    }

    hparams.n_sconv_channels = n_channels_max;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_inkling2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_vocab_unpadded != vocab.n_tokens()) {
        throw std::runtime_error("inkling2: tokenizer size must equal the unpadded vocabulary size");
    }

    const int64_t n_vocab_padded = hparams.n_vocab_padded;
    const int64_t n_rel          = hparams.n_relative;
    const int64_t n_conv         = hparams.n_shortconv_l_cache;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,      "weight"),    {n_embd, n_vocab_padded}, 0);
    tok_norm = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD_NORM, "weight", 0), {n_embd}, 0);

    // output
    output           = create_tensor(tn(LLM_TENSOR_OUTPUT,           "weight"), {n_embd, n_vocab_padded}, 0);
    prec_policy.set_acc(output, GGML_PREC_F32);
    output_norm      = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,      "weight"), {n_embd}, 0);
    output_res_query = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_QUERY, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer_phys(); ++i) {
        auto & layer = layers[i];

        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_head_kv   = hparams.n_head_kv(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_ff_exp    = hparams.n_ff_exp(i);
        const int64_t n_extent    = hparams.is_swa(i) ? hparams.n_swa : hparams.n_relative_extent;

        // attention
        layer.wq        = create_tensor(tn(LLM_TENSOR_ATTN_Q,    "weight", i), {n_embd, n_embd_head*n_head}, 0);
        layer.wk        = create_tensor(tn(LLM_TENSOR_ATTN_K,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wv        = create_tensor(tn(LLM_TENSOR_ATTN_V,    "weight", i), {n_embd, n_embd_head*n_head_kv}, 0);
        layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_embd_head*n_head, n_embd}, 0);
        layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_embd_head*n_head}, 0);

        layer.attn_q_clip    = create_tensor(tn(LLM_TENSOR_ATTN_Q_CLIP,    "weight", i), {n_embd_head, n_head}, 0);
        layer.attn_sink_proj = create_tensor(tn(LLM_TENSOR_ATTN_SINK_PROJ, "weight", i), {n_embd, n_head}, 0);

        // relative-position bias
        layer.attn_rel      = create_tensor(tn(LLM_TENSOR_ATTN_REL,      "weight", i), {n_embd, n_rel*n_head}, 0);
        layer.attn_rel_bias = create_tensor(tn(LLM_TENSOR_ATTN_REL_BIAS, "weight", i), {n_rel, n_extent}, 0);
        layer.attn_rel_clip = create_tensor(tn(LLM_TENSOR_ATTN_REL_CLIP, "weight", i), {n_rel, n_head}, 0);

        // short convolutions
        layer.attn_k_conv    = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_v_conv    = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV,    "weight", i), {n_conv, n_embd_head*n_head_kv}, 0);
        layer.attn_post_conv = create_tensor(tn(LLM_TENSOR_ATTN_POST_CONV, "weight", i), {n_conv, n_embd}, 0);
        layer.ffn_post_conv  = create_tensor(tn(LLM_TENSOR_FFN_POST_CONV,  "weight", i), {n_conv, n_embd}, 0);

        if (hparams.is_swa(i)) {
            layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        } else {
            // full-attention layers: separate Q, K and V input norms, and input-dependent taps for the K and V convolutions
            layer.attn_q_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_K_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_v_inp_norm  = create_tensor(tn(LLM_TENSOR_ATTN_V_INP_NORM,  "weight", i), {n_embd}, 0);
            layer.attn_k_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_K_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
            layer.attn_v_conv_proj = create_tensor(tn(LLM_TENSOR_ATTN_V_CONV_PROJ, "weight", i), {n_embd, n_conv*n_head_kv}, 0);
        }

        // block attention residuals
        if (i > 0) {
            layer.attn_res_query = create_tensor(tn(LLM_TENSOR_ATTN_RES_QUERY, "weight", i), {n_embd, hparams.is_swa(i) ? 1 : 3}, 0);
        }
        layer.ffn_res_query = create_tensor(tn(LLM_TENSOR_FFN_RES_QUERY, "weight", i), {n_embd}, 0);

        // MoE
        layer.ffn_norm         = create_tensor(tn(LLM_TENSOR_FFN_NORM,         "weight", i), {n_embd}, 0);
        layer.ffn_gate_inp     = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,     "weight", i), {n_embd, n_expert}, 0);
        layer.ffn_exp_probs_b  = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,  "bias",   i), {n_expert}, TENSOR_NOT_REQUIRED);
        layer.ffn_gate_up_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_UP_EXPS, "weight", i), {n_embd, 2*n_ff_exp, n_expert}, 0);
        layer.ffn_down_exps    = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,    "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_inkling2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

// scale each row down to an L2 norm of at most threshold, after the optional per-channel scale
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    // x*min(1, t/|x|) = t*x/max(|x|, t), and l2_norm(x, eps) = x/max(|x|, eps)
    return ggml_scale(ctx0, ggml_l2_norm(ctx0, cur, threshold), threshold);
}

// attention over the depth: mix the earlier block outputs and cur with learned queries
// the normalized blocks, newest first, [n_embd, n, n_tokens] and transposed [n, n_embd, n_tokens]
// a block is normalized once per graph, and each block list is stacked once and shared by every mix over it
const llama_model_inkling2::graph::block_stack * llama_model_inkling2::graph::stack(const std::vector<ggml_tensor *> & blocks, int il) {
    if (blocks.empty()) {
        return nullptr;
    }
    // one stack per device, so that a split does not pull another device's stack
    const auto key = std::make_pair(il < 0 ? nullptr : model.dev_layer(il), blocks);
    auto it = block_stacks.find(key);
    if (it != block_stacks.end()) {
        return &it->second;
    }
    const std::vector<ggml_tensor *> prefix(blocks.begin(), blocks.end() - 1);
    const block_stack * prev = stack(prefix, il);

    ggml_tensor * normed = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, blocks.back(), n_embd, 1, n_tokens), 1e-6f);
    ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

    block_stack st;
    st.states   = prev ? ggml_concat(ctx0, normed,   prev->states,   1) : normed;
    st.states_t = prev ? ggml_concat(ctx0, normed_t, prev->states_t, 0) : normed_t;
    return &block_stacks.emplace(key, st).first->second;
}

ggml_tensor * llama_model_inkling2::graph::mix(const std::vector<ggml_tensor *> & blocks, ggml_tensor * cur, ggml_tensor * query, int il) {
    // rows [blocks.back(), ..., blocks.front(), cur], each RMS-normalized; shared by every mix over the same blocks and cur
    const ggml_backend_dev_t dev = il < 0 ? nullptr : model.dev_layer(il);
    auto key = std::make_pair(dev, blocks);
    key.second.push_back(cur);
    block_stack & all = mix_states[key];
    if (!all.states) {
        ggml_tensor * normed   = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, cur, n_embd, 1, n_tokens), 1e-6f);
        ggml_tensor * normed_t = ggml_reshape_3d(ctx0, normed, 1, n_embd, n_tokens);

        const block_stack * st = stack(blocks, il);
        all.states   = st ? ggml_concat(ctx0, st->states,   normed,   1) : normed;
        all.states_t = st ? ggml_concat(ctx0, st->states_t, normed_t, 0) : normed_t;
    }
    ggml_tensor * states   = all.states;
    ggml_tensor * states_t = all.states_t;

    // one token: the query broadcasts as is
    ggml_tensor * queries = n_tokens == 1 ? query :
        ggml_repeat(ctx0, query, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd, query->ne[1], n_tokens));

    ggml_tensor * scores = ggml_mul_mat(ctx0, states, queries);
    scores = ggml_soft_max_ext(ctx0, scores, nullptr, 1.0f/sqrtf(n_embd), 0.0f);

    return ggml_mul_mat(ctx0, scores, states_t); // [n_queries, n_embd, n_tokens]
}

// causal short convolution with a residual; dynamic adds input-dependent taps per head
// the last n_conv - 1 inputs of each sequence live in the recurrent state at channel offset `offset`
ggml_tensor * llama_model_inkling2::graph::convolve(
        ggml_tensor * cur,
        ggml_tensor * weight,
        ggml_tensor * dynamic,
        ggml_tensor * state,
        ggml_tensor * history,
        llm_graph_input_rs * inp,
        int64_t offset) {
    const int64_t n_conv       = hparams.n_shortconv_l_cache;
    const int64_t n_channels   = cur->ne[0];
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    // the inputs first, so that the conv chain from the concat to the scale is consecutive in the graph
    ggml_build_forward_expand(gf, cur);
    // add the taps input first and the result right after the state store, so the conv ops are consecutive and backends can fuse them
    if (dynamic) {
        ggml_build_forward_expand(gf, dynamic);
    }

    ggml_tensor * past = ggml_view_3d(ctx0, history, n_conv - 1, n_channels, n_seqs, history->nb[1], history->nb[2], offset*history->nb[1]);

    ggml_tensor * x = ggml_transpose(ctx0, ggml_reshape_3d(ctx0, cur, n_channels, n_seq_tokens, n_seqs));
    x = ggml_concat(ctx0, past, x, 0);

    build_conv_state_store(inp, state, x, n_conv - 1, offset*(n_conv - 1)*ggml_element_size(state));

    ggml_tensor * out = ggml_ssm_conv(ctx0, x, weight);
    out = ggml_reshape_2d(ctx0, out, n_channels, n_tokens);

    if (dynamic) {
        const int64_t n_heads = dynamic->ne[0]/n_conv;

        dynamic = ggml_reshape_3d(ctx0, dynamic, n_conv, n_heads, n_tokens);

        // the inputs channel-major once, [n_channels, n_conv - 1 + n_seq_tokens, n_seqs]: each lag is a row offset
        ggml_tensor * xt = ggml_cont(ctx0, ggml_transpose(ctx0, x));

        for (int64_t lag = 0; lag < n_conv; ++lag) {
            ggml_tensor * tap    = ggml_view_4d(ctx0, dynamic, 1, n_heads, n_seq_tokens, n_seqs,
                    dynamic->nb[1], dynamic->nb[2], n_seq_tokens*dynamic->nb[2], lag*dynamic->nb[0]);
            ggml_tensor * window = ggml_view_4d(ctx0, xt, n_channels/n_heads, n_heads, n_seq_tokens, n_seqs,
                    (n_channels/n_heads)*xt->nb[0], xt->nb[1], xt->nb[2], (n_conv - 1 - lag)*xt->nb[1]);

            ggml_tensor * term = ggml_reshape_2d(ctx0, ggml_mul(ctx0, window, tap), n_channels, n_tokens);

            out = ggml_add(ctx0, out, term);
        }
    }

    out = ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
    ggml_build_forward_expand(gf, out);

    return out;
}

// mix a cross-pass message into a looped layer input: (1 - forget)*cur + gain*message, both gated on cur and message
// cur_norm and message_norm are cur and message RMS-normalized
ggml_tensor * llama_model_inkling2::graph::control(ggml_tensor * cur, ggml_tensor * cur_norm, ggml_tensor * message, ggml_tensor * message_norm,
        const loop_layer & loop, int ctrl) {
    ggml_tensor * gate = ggml_concat(ctx0, cur_norm, message_norm, 0);
    gate = build_lora_mm(loop.ctrl_down[ctrl], gate);

    ggml_tensor * gain   = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_gain[ctrl],   gate));
    ggml_tensor * forget = ggml_tanh(ctx0, build_lora_mm(loop.ctrl_forget[ctrl], gate));

    return ggml_add(ctx0, ggml_sub(ctx0, cur, ggml_mul(ctx0, forget, cur)), ggml_mul(ctx0, gain, message));
}

bool llama_model_inkling2::graph::use_banded(int il, bool swa) {
    const auto * cache = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
    const int64_t hd = hparams.n_embd_head_k(il);
    // quantized K/V too: the dense fallback materializes an n_kv x n_tokens x n_head bias, far more than the cache saves
    const auto type_ok = [](ggml_type t) {
        return t == GGML_TYPE_F32 || t == GGML_TYPE_F16 || t == GGML_TYPE_BF16 || ggml_is_quantized(t);
    };
    if (!(inp->n_kv_banded[swa] > 0 && (hd == 64 || hd == 128) && hparams.n_embd_head_v(il) == hd &&
            type_ok(cache->type_k()) && type_ok(cache->type_v()) && !cache->get_k_scale(il) && !cache->get_v_scale(il) &&
            (swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3] == 1)) {
        return false;
    }
    ggml_backend_dev_t dev = model.dev_layer(il);
    if (!dev) {
        return false;
    }
    if (decode_banded && strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
        return false;
    }
    for (const auto & [key, ok] : banded_dev_ok) {
        if (key.first == swa && key.second == dev) {
            return ok;
        }
    }
    const int64_t nh  = hparams.n_head(il);
    const int64_t nkv = hparams.n_head_kv(il);
    ggml_tensor * pq = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, hd, 1, nh, 1);
    ggml_tensor * pk = ggml_new_tensor_4d(ctx0, cache->type_k(), hd, 1, nkv, 1);
    ggml_tensor * pv = ggml_new_tensor_4d(ctx0, cache->type_v(), hd, 1, nkv, 1);
    ggml_tensor * pm = ggml_new_tensor_4d(ctx0, GGML_TYPE_F16, 1, 1, 1, 1);
    ggml_tensor * pr = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, nh, 1, 1);
    ggml_tensor * op = ggml_flash_attn_ext_banded(ctx0, pq, pk, pv, pm, pr, 1.0f, 1);
    ggml_flash_attn_ext_add_sinks(op, ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, nh, 1, 1));
    if (inp->self_k_pos[swa]) {
        ggml_flash_attn_ext_banded_set_pos(op, ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1), ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, 1));
    }
    ggml_prec_set_acc(op, GGML_PREC_F32);
    const bool ok = ggml_backend_dev_supports_op(dev, op);
    banded_dev_ok.push_back({ { swa, dev }, ok });
    return ok;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    // banded FA. Ubatches of 2 to n_tokens_banded_min tokens (drafts, prompt tails) span the padded n_kv, which changes
    // only every 256 cells, so that their graph is reused. A decode step stays dense: the FA vec kernel is faster there,
    // and the reserved tg graph (one token over the whole cache) holds its bias, except with a cache type that dense
    // CUDA FA converts to F16 whole every step (it reads F16 / BF16 / q8_0 / q4_0 in place): then the full-attention
    // layers of a single-sequence context take the banded kernel
    const auto * base = attn->mctx->get_base();
    const ggml_type tk = base->type_k();
    const ggml_type tv = base->type_v();
    const bool kv_converted = tk == tv ?
        tk == GGML_TYPE_Q4_1 || tk == GGML_TYPE_Q5_0 || tk == GGML_TYPE_Q5_1 :
        tk == GGML_TYPE_Q8_0 && (tv == GGML_TYPE_Q4_0 || tv == GGML_TYPE_F16);
    decode_banded = n_tokens == 1 && cparams.n_seq_max == 1 && kv_converted;
    // a few-token ubatch (speculative verify, prompt tails) over a short enough cache stays dense too: the FA tile
    // kernel with the indexed bias beats the banded op plus its merge there. n_kv is part of can_reuse(), so a reused
    // graph made the same choice
    const bool small_dense = n_tokens > 1 && n_tokens <= inkling2_dense_n_max() && !kv_converted &&
        (int64_t) base->get_n_kv() <= inkling2_dense_kv_max();
    inp->banded     = cparams.flash_attn && ((n_tokens > 1 && !small_dense) || decode_banded);
    inp->banded_pad = n_tokens <= n_tokens_banded_min;
    // the cells of several sequences, or of a wrapped cache, are not the positions: the banded attention then
    // takes the distances from the positions of the cells and the queries; disabled, those ubatches are dense
    inp->banded_pos = inp->banded ? inkling2_banded_pos_mode() : 0;
    // the positions of the n_kv_banded cells of a cache, filled at set_input
    auto k_pos = [&](int i) {
        inp->self_k_pos[i] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, inp->n_kv_banded[i]);
        ggml_set_input(inp->self_k_pos[i]);
        ggml_set_name(inp->self_k_pos[i], i ? "attn_inp_k_pos_swa" : "attn_inp_k_pos");
    };
    // that decode step keeps the small SWA view dense, in the reserve too
    inp->banded_swa_dense = decode_banded;
    // the causal attention without a mask (below) runs on the CUDA and CPU kernels only
    bool causal_devs = true;
    for (uint32_t il = 0; il < hparams.n_layer() && causal_devs; ++il) {
        if (hparams.is_swa(il)) {
            continue;
        }
        ggml_backend_dev_t dev = model.dev_layer(il);
        const char * reg = dev ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) : "";
        causal_devs = strcmp(reg, "CUDA") == 0 || strcmp(reg, "CPU") == 0;
    }
    const bool causal_ok = inp->banded && cparams.n_seq_max == 1 && cparams.causal_attn && n_tokens <= 65535 &&
        causal_devs && inkling2_banded_causal_enabled();
    inp->banded_causal_ok = causal_ok;
    // the cells of each cache and whether they take the positions, the decision can_reuse() repeats
    bool use_pos[2];
    inp->get_n_kv_banded(attn->mctx, ubatch, inp->n_kv_banded, use_pos);
    for (int i = 0; i < 2; ++i) {
        if (use_pos[i]) {
            k_pos(i);
        }
    }
    // the band of a padded span is shifted by the padding; with the positions the distances need no shift
    for (int i = 0; i < 2 && inp->banded_pad; ++i) {
        if (inp->n_kv_banded[i] > 0 && !inp->self_k_pos[i]) {
            ggml_tensor *& shift = inp->self_rel_shift[i];
            shift = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (i ? hparams.n_swa : hparams.n_relative_extent) + inp->n_rel_pad);
            ggml_set_input(shift);
            ggml_set_name(shift, i ? "attn_inp_rel_shift_swa" : "attn_inp_rel_shift");
        }
    }

    // a cache of one sequence: a cell is visible iff its distance to the query is not negative (empty cells have no
    // position), so the full-attention layers need no mask, an n_kv x n_tokens F16 input (1 GiB at 1M cells and 512
    // tokens, on the host and on the device). Without the positions the distance is a cell difference, which hides
    // nothing before the sequence: it must start at cell 0 (n_kv_banded is then pos_max + 1, the cells before a later
    // start are masked by the mask path); padded spans without the positions measure the distance from the padded tail
    // and keep the mask
    banded_causal = causal_ok && inp->n_kv_banded[0] > 0 &&
        (inp->self_k_pos[0] || (!inp->banded_pad && ubatch.pos != nullptr &&
            inp->n_kv_banded[0] == (uint32_t) ubatch.pos[n_tokens - 1] + 1));

    ggml_tensor * inp_pos = build_inp_pos();
    // the query positions of the banded attention
    q_pos_banded = ggml_view_1d(ctx0, inp_pos, n_tokens, 0);
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);
    // not expanded here: as the first nodes of the graph, before any weight, the scheduler would run this chain on the
    // CPU every ubatch (next to the CPU embedding lookup); expanded at the first full-attention layer it stays on the GPU

    ggml_tensor * cur = build_inp_embd(model.tok_embd);

    // important: do not normalize raw embeddings input (i.e. encoded image embeddings)
    if (ubatch.token) {
        cur = build_norm(cur, model.tok_norm, nullptr, LLM_NORM_RMS, -1);
    }

    return cur;
}

ggml_tensor * llama_model_inkling2::graph::build_layer(int il, int ip, ggml_tensor * cur, std::vector<ggml_tensor *> & blocks,
        ggml_tensor * messages, const loop_layer * loop) {
    GGML_ASSERT(!messages == !loop);

    const auto & layer = model.layers[il];

    const bool    is_swa       = hparams.is_swa(il);
    const int64_t n_head       = hparams.n_head(il);
    const int64_t n_head_kv    = hparams.n_head_kv(il);
    const int64_t n_embd_head  = hparams.n_embd_head_k(il);
    const int64_t n_embd_k_gqa = n_embd_head*n_head_kv;

    // the messages [LOOP_CTRL_COUNT, n_embd, n_tokens] as rows [n_embd, LOOP_CTRL_COUNT, n_tokens], and those rows
    // RMS-normalized once; the message of controller ctrl, [n_embd, n_tokens], is a view of either
    ggml_tensor * messages_rows = messages ? ggml_cont(ctx0, ggml_transpose(ctx0, messages)) : nullptr;
    ggml_tensor * messages_norm = messages_rows ? ggml_rms_norm(ctx0, messages_rows, 1e-6f) : nullptr;
    auto message = [&](ggml_tensor * t, int ctrl) {
        return ggml_view_2d(ctx0, t, n_embd, n_tokens, t->nb[2], ctrl*t->nb[1]);
    };

    ggml_tensor * state   = recr->mctx->get_r_l(il);
    ggml_tensor * history = build_rs(recr, state, hparams.n_embd_r(), ubatch.n_seqs);
    history = ggml_reshape_3d(ctx0, history, hparams.n_shortconv_l_cache - 1, hparams.n_sconv_channels, ubatch.n_seqs);

    ggml_tensor * inpSA = cur;

    // attention inputs: SWA layers mix one, full-attention layers one each for Q, K and V
    ggml_tensor * inputs[3] = {cur, cur, cur};
    if (layer.attn_res_query) {
        ggml_tensor * mixed = mix(blocks, cur, layer.attn_res_query, il);
        if (is_swa) {
            // one query, already contiguous
            inputs[0] = inputs[1] = inputs[2] = ggml_reshape_2d(ctx0, mixed, n_embd, n_tokens);
        } else {
            for (int j = 0; j < 3; ++j) {
                inputs[j] = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_view_3d(ctx0, mixed, 1, n_embd, n_tokens, mixed->nb[1], mixed->nb[2], j*mixed->nb[0])), n_embd, n_tokens);
            }
        }
    }

    if (messages) {
        // the messages make the Q, K and V inputs of a SWA layer differ
        ggml_tensor * inputs_norm[3] = {};
        for (int j = 0; j < 3; ++j) {
            // a SWA layer normalizes its one input once
            inputs_norm[j] = j > 0 && inputs[j] == inputs[0] ? inputs_norm[0] : ggml_rms_norm(ctx0, inputs[j], 1e-6f);
        }
        for (int j = 0; j < 3; ++j) {
            inputs[j] = control(inputs[j], inputs_norm[j], message(messages_rows, LOOP_CTRL_Q + j), message(messages_norm, LOOP_CTRL_Q + j),
                    *loop, LOOP_CTRL_Q + j);
        }
    }

    if (is_swa && messages) {
        for (int j = 0; j < 3; ++j) {
            inputs[j] = build_norm(inputs[j], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        }
    } else if (is_swa) {
        inputs[0] = build_norm(inputs[0], layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = inputs[2] = inputs[0];
    } else {
        inputs[0] = build_norm(inputs[0], layer.attn_q_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[1] = build_norm(inputs[1], layer.attn_k_inp_norm, nullptr, LLM_NORM_RMS, il);
        inputs[2] = build_norm(inputs[2], layer.attn_v_inp_norm, nullptr, LLM_NORM_RMS, il);
    }

    // self-attention
    {
        ggml_tensor * Qcur = build_lora_mm(layer.wq, inputs[0]);
        ggml_tensor * Kcur = convolve(build_lora_mm(layer.wk, inputs[1]), layer.attn_k_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_k_conv_proj, inputs[1]), state, history, recr, 0);
        ggml_build_forward_expand(gf, Kcur);
        ggml_tensor * Vcur = convolve(build_lora_mm(layer.wv, inputs[2]), layer.attn_v_conv,
                is_swa ? nullptr : build_lora_mm(layer.attn_v_conv_proj, inputs[2]), state, history, recr, n_embd_k_gqa);
        ggml_build_forward_expand(gf, Vcur);
        ggml_tensor * Rcur = build_lora_mm(layer.attn_rel, inputs[0]);

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        if (use_banded(il, is_swa)) {
            const auto * cache = is_swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            auto * k_rot = is_swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
            auto * v_rot = is_swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
            if (k_rot) {
                Qcur = llama_mul_mat_hadamard(ctx0, Qcur, k_rot);
                Kcur = llama_mul_mat_hadamard(ctx0, Kcur, k_rot);
            }
            if (v_rot) {
                Vcur = llama_mul_mat_hadamard(ctx0, Vcur, v_rot);
            }
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, cache->cpy_k(ctx0, Kcur, is_swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, cache->cpy_v(ctx0, Vcur, is_swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));

            const int64_t n_kv_b = inp->n_kv_banded[is_swa];
            ggml_tensor * k_all = cache->get_k(ctx0, il);
            ggml_tensor * v_all = cache->get_v(ctx0, il);
            GGML_ASSERT(n_kv_b <= k_all->ne[2]);
            ggml_tensor * q_fa = ggml_permute(ctx0, ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_tokens, 1), 0, 2, 1, 3);
            ggml_tensor * k_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, k_all, k_all->ne[0], k_all->ne[1], n_kv_b, k_all->ne[3],
                    k_all->nb[1], k_all->nb[2], k_all->nb[3], 0), 0, 2, 1, 3);
            ggml_tensor * v_fa;
            if (v_all->nb[1] > v_all->nb[2]) { // transposed V
                v_fa = ggml_view_4d(ctx0, v_all, n_kv_b, v_all->ne[1], v_all->ne[2], v_all->ne[3],
                        v_all->nb[1], v_all->nb[2], v_all->nb[3], 0);
                v_fa = ggml_transpose(ctx0, ggml_permute(ctx0, v_fa, 0, 2, 1, 3));
            } else {
                v_fa = ggml_permute(ctx0, ggml_view_4d(ctx0, v_all, v_all->ne[0], v_all->ne[1], n_kv_b, v_all->ne[3],
                        v_all->nb[1], v_all->nb[2], v_all->nb[3], 0), 0, 2, 1, 3);
            }
            // the causal attention without a mask runs on the CUDA and CPU kernels only
            ggml_backend_dev_t dev = model.dev_layer(il);
            const char * reg = dev ? ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)) : "";
            const bool causal = !is_swa && banded_causal && (strcmp(reg, "CUDA") == 0 || strcmp(reg, "CPU") == 0);
            ggml_tensor * m = nullptr;
            if (!causal) {
                ggml_tensor * mask = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
                // one banded mask per graph and cache, the first n_kv_b cells of each row read in place
                ggml_tensor *& mb = mask_banded[is_swa];
                if (!mb) {
                    mb = ggml_view_4d(ctx0, mask, n_kv_b, mask->ne[1], mask->ne[2], mask->ne[3],
                            mask->nb[1], mask->nb[2], mask->nb[3], 0);
                }
                m = mb;
            }
            // exp(s + b)/(exp(t) + sum(exp(s + b))): the dense mask b - t with a zero sink
            ggml_tensor * rel_bias = layer.attn_rel_bias;
            if (ggml_tensor * shift = inp->self_rel_shift[is_swa]) {
                // padded span: the band shifted by the padding, zero (the padded row) past the extent
                rel_bias = ggml_get_rows(ctx0, ggml_pad(ctx0, rel_bias, 0, 1, 0, 0), shift);
            }
            ggml_tensor * rel_b = ggml_mul_mat(ctx0, rel_bias, Rcur);
            rel_b = ggml_reshape_4d(ctx0, rel_b, rel_b->ne[0], n_head, n_tokens, 1);
            cur = ggml_flash_attn_ext_banded(ctx0, q_fa, k_fa, v_fa, m, rel_b, 1.0f/sqrtf(n_embd_head), rel_b->ne[0]);
            ggml_flash_attn_ext_add_sinks(cur, ggml_reshape_3d(ctx0, sinks, n_head, n_tokens, 1));
            if (inp->self_k_pos[is_swa]) {
                ggml_flash_attn_ext_banded_set_pos(cur, inp->self_k_pos[is_swa], q_pos_banded);
            }
            if (causal) {
                ggml_flash_attn_ext_banded_set_causal(cur, true);
            }
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            res->add_fused_node({LLM_FUSED_OP_FLASH_ATTN, cur, il});
            cur = ggml_reshape_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]);
            cb(cur, "kqv_out", il);
            if (v_rot) {
                cur = llama_mul_mat_hadamard(ctx0, cur, v_rot);
            }
        } else {
            // project before gathering: only the gathered bias is n_kv x n_tokens x n_head
            ggml_tensor * table = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), Rcur);

            // per-query sink t: exp(s - t)/(1 + sum(exp(s - t))) = exp(s)/(exp(t) + sum(exp(s)))
            table = ggml_sub(ctx0, table, ggml_reshape_3d(ctx0, sinks, 1, n_head, n_tokens));

            // zero sinks, filled once per graph and layer device
            ggml_tensor *& zero = clip_limits[{ { model.dev_layer(il), -n_head }, 0.0f }];
            if (!zero) {
                zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
            }
            sinks = zero;

            // the bias is n_kv x n_tokens x n_head wherever it is made dense: chunk query rows past the budget
            const int64_t n_stream     = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
            const int64_t n_kv         = (is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos)->ne[0];
            const int64_t n_seq_tokens = n_tokens/n_stream;
            // sized for a full cache, and the chunk never shrinks as n_tokens grows, so the graph reserved
            // at load (full cache, full ubatch) has the largest chunk any later ubatch builds
            const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());
            // query rows per chunk over all streams: whole 64-row tiles in the budget, at most n_kq_chunk_max chunks
            const int64_t n_rows_all = std::max({ int64_t(64), n_kq_bias_max/(n_kv_max*n_head)/64*64,
                    GGML_PAD((n_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max, 64) });
            const int64_t n_rows_max = std::min(n_seq_tokens, (n_rows_all + n_stream - 1)/n_stream);
            if (n_rows_max == n_seq_tokens) {
                cur = build_attn(inp,
                        nullptr, nullptr, nullptr,
                        Qcur, Kcur, Vcur, table, sinks, 1.0f/sqrtf(n_embd_head), il);
            } else {
                auto rows = [&](ggml_tensor * t, int d, int64_t r0, int64_t nr) {
                    int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                    ne[d] = nr;
                    return ggml_cont(ctx0, ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], r0*t->nb[d]));
                };
                // the chunks write their rows into one output made before their biases: outputs that stayed live until
                // a final concatenation split the region the next chunk's bias reuses, and the buffer grew with each chunk
                ggml_tensor * out_all = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head*n_head, n_seq_tokens, n_stream));
                ggml_build_forward_expand(gf, out_all);
                for (int64_t r0 = 0; r0 < n_seq_tokens; r0 += n_rows_max) {
                    const int64_t nr = std::min(n_rows_max, n_seq_tokens - r0);
                    ggml_tensor * q_c = ggml_reshape_3d(ctx0,
                            rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                            n_embd_head, n_head, nr*n_stream);
                    ggml_tensor * table_c = ggml_reshape_3d(ctx0,
                            rows(ggml_reshape_4d(ctx0, table, table->ne[0], table->ne[1], n_seq_tokens, n_stream), 2, r0, nr),
                            table->ne[0], table->ne[1], nr*n_stream);

                    // only the first chunk stores K and V
                    ggml_tensor * out = build_attn(inp,
                            nullptr, nullptr, nullptr,
                            q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr, table_c, sinks, 1.0f/sqrtf(n_embd_head), il, r0);
                    ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_3d(ctx0, out, out->ne[0], nr, n_stream),
                            ggml_view_3d(ctx0, out_all, out_all->ne[0], nr, n_stream, out_all->nb[1], out_all->nb[2], r0*out_all->nb[1])));
                }
                cur = ggml_reshape_2d(ctx0, out_all, out_all->ne[0], n_tokens);
            }
        }

        cur = ggml_mul(ctx0, cur, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, inputs[0])));
        cur = convolve(build_lora_mm(layer.wo, cur), layer.attn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa);
        ggml_build_forward_expand(gf, cur);
        cb(cur, "attn_out", il);
    }

    // the first layer of each block starts a new residual stream, the earlier one joins the mixed blocks
    if (ip % hparams.attn_res_block_size == 0) {
        blocks.push_back(inpSA);
    } else {
        cur = ggml_add(ctx0, inpSA, cur);
        ggml_build_forward_expand(gf, cur); // right after the conv, so backends can fuse the add into it
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query, il), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, ggml_rms_norm(ctx0, cur, 1e-6f), message(messages_rows, LOOP_CTRL_FFN), message(messages_norm, LOOP_CTRL_FFN),
                *loop, LOOP_CTRL_FFN);
    }
    cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);

    cur = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            nullptr,
            nullptr,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            hparams.n_expert,
            hparams.n_expert_used(il),
            LLM_FFN_SITU, true,
            hparams.expert_weights_scale,
            LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
            il,
            nullptr,
            layer.ffn_gate_up_exps);

    cur = convolve(cur, layer.ffn_post_conv, nullptr, state, history, recr, 2*n_embd_k_gqa + n_embd);
    ggml_build_forward_expand(gf, cur);
    cb(cur, "ffn_out", il);

    cur = ggml_add(ctx0, ffn_inp, cur);
    ggml_build_forward_expand(gf, cur);
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

    // DSpark target features: the last one is the final norm output of every token
    res->t_layer_inp[n_layer] = cur;

    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur);

    if (hparams.f_final_logit_softcapping != 0.0f) {
        const float cap = hparams.f_final_logit_softcapping;

        cur = ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, cur, 1.0f/cap)), cap);
    }

    // drop the padded vocabulary
    cur = ggml_cont(ctx0, ggml_view_2d(ctx0, cur, hparams.n_vocab_unpadded, cur->ne[1], cur->nb[1], 0));

    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

llama_model_inkling2::graph::graph(const llm_graph_params & params) : llm_graph_context(params) {
    ggml_tensor * cur = build_inputs();

    std::vector<ggml_tensor *> blocks;
    for (int il = 0; il < n_layer; ++il) {
        // DSpark target features: the partial block at the layer input
        res->t_layer_inp[il] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
