#include "models.h"
#include "../llama-memory-hybrid-iswa.h"

#include <algorithm>
#include <vector>

// max elements of the dense relative bias per attention call, and max chunks per layer
static constexpr int64_t n_kq_bias_max  = int64_t(1) << 29;
static constexpr int64_t n_kq_chunk_max = 64;
// ubatches up to this many tokens keep the dense path
static constexpr int64_t n_tokens_banded_min = 32;

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
ggml_tensor * llama_model_inkling2::graph::clip(ggml_tensor * cur, ggml_tensor * scale, float threshold, int il) {
    if (scale) {
        cur = ggml_mul(ctx0, cur, scale);
    }

    ggml_tensor * norm = ggml_sqrt(ctx0, ggml_sum_rows(ctx0, ggml_sqr(ctx0, cur)));
    norm = ggml_scale_bias(ctx0, norm, 1.0f, 1e-6f);

    // the threshold tensor only depends on the shape: fill it once per graph and layer device
    ggml_tensor *& limit = clip_limits[{ { model.dev_layer(il), norm->ne[1] }, threshold }];
    if (!limit) {
        limit = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, norm->ne[0], norm->ne[1], norm->ne[2], norm->ne[3]), threshold);
    }

    ggml_tensor * factor = ggml_div(ctx0, limit, norm);

    return ggml_mul(ctx0, cur, ggml_clamp(ctx0, factor, 0.0f, 1.0f));
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

    return ggml_mul_mat(ctx0, states_t, scores);
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

    return ggml_scale(ctx0, ggml_add(ctx0, cur, out), 1.0f/sqrtf((dynamic ? 2 : 1)*n_conv + 1));
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
    ggml_prec_set_acc(op, GGML_PREC_F32);
    const bool ok = ggml_backend_dev_supports_op(dev, op);
    banded_dev_ok.push_back({ { swa, dev }, ok });
    return ok;
}

ggml_tensor * llama_model_inkling2::graph::build_inputs() {
    inp  = build_inp_mem_hybrid_iswa_rel(hparams.n_relative_extent, hparams.n_swa);
    attn = inp->get_attn();
    recr = inp->get_recr();

    // banded FA needs a position-contiguous cache; small ubatches stay dense for graph reuse
    inp->banded = cparams.flash_attn && n_tokens > n_tokens_banded_min;
    for (int i = 0; i < 2 && inp->banded; ++i) {
        // a ubatch of several sequences is dense at runtime, so the reserved worst case (a ubatch
        // of every sequence) must be dense too, else the dense graph outgrows the reserved buffer
        inp->n_kv_banded[i] = ubatch.n_seqs_unq != 1 ? 0 :
            (i ? attn->mctx->get_swa() : attn->mctx->get_base())->get_n_kv_pos_contiguous();
    }
    // the reserve graph: the cache offers the banded path for a ubatch of placeholder positions (all 0),
    // while at runtime it does so only for consecutive ones
    if (inp->n_kv_banded[0] > 0 && ubatch.pos[0] == ubatch.pos[n_tokens - 1]) {
        // once the positions pass the SWA cache it wraps and its layers take the dense bias for every ubatch
        if (attn->mctx->get_swa()->get_size() < attn->mctx->get_base()->get_size()) {
            inp->n_kv_banded[1] = 0;
        }
        n_rows_dense_reserve = std::min<int64_t>(n_tokens, n_tokens_banded_min);
    }

    ggml_tensor * inp_pos = build_inp_pos();
    inp_out_ids = build_inp_out_ids();

    // log-scaled positions: 1 + alpha*log(max(1, (pos + 1)/floor)), multiplies q and r of the full-attention layers
    tau = ggml_scale_bias(ctx0, ggml_cast(ctx0, inp_pos, GGML_TYPE_F32), 1.0f, 1.0f);
    tau = ggml_log(ctx0, ggml_clamp(ctx0, ggml_scale(ctx0, tau, 1.0f/hparams.n_log_scale_floor), 1.0f, INFINITY));
    tau = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, tau, hparams.f_log_scale_alpha, 1.0f), 1, 1, n_tokens);

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

    // the message of controller ctrl, [n_embd, n_tokens], and the messages RMS-normalized once
    ggml_tensor * messages_norm = messages ? ggml_rms_norm(ctx0, messages, 1e-6f) : nullptr;
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
                inputs[j] = ggml_cont(ctx0, ggml_view_2d(ctx0, mixed, n_embd, n_tokens, mixed->nb[2], j*mixed->nb[1]));
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
            inputs[j] = control(inputs[j], inputs_norm[j], message(messages, LOOP_CTRL_Q + j), message(messages_norm, LOOP_CTRL_Q + j),
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

        Qcur = clip(ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens), layer.attn_q_clip, hparams.f_qk_clip, il);
        Kcur = clip(ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens), nullptr,           hparams.f_qk_clip, il);
        Vcur = ggml_rms_norm(ctx0, ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens), hparams.f_norm_rms_eps);
        Rcur = clip(ggml_reshape_3d(ctx0, Rcur, hparams.n_relative, n_head, n_tokens), layer.attn_rel_clip, hparams.f_relative_clip, il);

        if (!is_swa) {
            Qcur = ggml_mul(ctx0, Qcur, tau);
            Rcur = ggml_mul(ctx0, Rcur, tau);
        }

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);
        cb(Rcur, "Rcur", il);

        ggml_tensor * rel_pos = is_swa ? inp->self_rel_pos_swa : inp->self_rel_pos;

        const int64_t n_stream = (is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw)->ne[3];
        const int64_t n_kv     = rel_pos->ne[0];

        ggml_tensor * sinks = build_lora_mm(layer.attn_sink_proj, inputs[0]);

        if (use_banded(il, is_swa) && !is_swa && n_rows_dense_reserve > 0) {
            // reserve only, output unused: the dense bias of the first rows over the full cache, as a small ubatch
            // builds it. Once, before the first banded layer: the banded mask it allocates is not in that ubatch
            const int64_t nr = n_rows_dense_reserve;
            n_rows_dense_reserve = 0;
            ggml_tensor * r_s = ggml_view_3d(ctx0, Rcur, Rcur->ne[0], n_head, nr, Rcur->nb[1], Rcur->nb[2], 0);
            ggml_tensor * t_s = ggml_mul_mat(ctx0, ggml_pad(ctx0, layer.attn_rel_bias, 0, 1, 0, 0), r_s);
            t_s = ggml_cont(ctx0, ggml_permute(ctx0, t_s, 1, 0, 2, 3));
            ggml_tensor * p_s = ggml_view_2d(ctx0, rel_pos, n_kv, nr, rel_pos->nb[1], 0);
            ggml_tensor * b_s = ggml_get_rows(ctx0, t_s, p_s);
            b_s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_4d(ctx0, b_s, n_head, n_kv, nr, 1), 2, 0, 1, 3));
            ggml_tensor * q_s = ggml_cont(ctx0, ggml_view_3d(ctx0, Qcur, n_embd_head, n_head, nr, Qcur->nb[1], Qcur->nb[2], 0));
            ggml_build_forward_expand(gf, build_attn(inp, nullptr, nullptr, nullptr, q_s, nullptr, nullptr, b_s,
                    ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f), 1.0f/sqrtf(n_embd_head), il));
        }
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
            ggml_tensor * mask = is_swa ? attn->self_attn_mask_swa.raw : attn->self_attn_mask.raw;
            // one banded mask per graph and cache: each layer's own view would be a separate scheduler input copy
            ggml_tensor *& m = mask_banded[is_swa];
            if (!m) {
                m = ggml_cont(ctx0, ggml_view_4d(ctx0, mask, n_kv_b, mask->ne[1], mask->ne[2], mask->ne[3],
                        mask->nb[1], mask->nb[2], mask->nb[3], 0));
            }
            // exp(s + b)/(exp(t) + sum(exp(s + b))): the dense mask b - t with a zero sink
            ggml_tensor * rel_b = ggml_mul_mat(ctx0, layer.attn_rel_bias, Rcur);
            rel_b = ggml_reshape_4d(ctx0, rel_b, rel_b->ne[0], n_head, n_tokens, 1);
            cur = ggml_flash_attn_ext_banded(ctx0, q_fa, k_fa, v_fa, m, rel_b, 1.0f/sqrtf(n_embd_head), rel_b->ne[0]);
            ggml_flash_attn_ext_add_sinks(cur, ggml_reshape_3d(ctx0, sinks, n_head, n_tokens, 1));
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
            table = ggml_cont(ctx0, ggml_permute(ctx0, table, 1, 0, 2, 3));

            // zero sinks, filled once per graph and layer device
            ggml_tensor *& zero = clip_limits[{ { model.dev_layer(il), -n_head }, 0.0f }];
            if (!zero) {
                zero = ggml_fill(ctx0, ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, n_head), 0.0f);
            }
            sinks = zero;

            // dense bias is n_kv x n_tokens x n_head: chunk query rows past the budget
            const int64_t n_seq_tokens = n_tokens/n_stream;
            // sized for a full cache, and the chunk never shrinks as n_tokens grows, so the graph reserved
            // at load (full cache, full ubatch) has the largest chunk any later ubatch builds
            const int64_t n_kv_max = std::max<int64_t>(n_kv, (is_swa ? attn->mctx->get_swa() : attn->mctx->get_base())->get_size());
            // query rows per chunk over all streams: whole 64-row tiles in the budget, at most n_kq_chunk_max chunks
            const int64_t n_rows_all = std::max({ int64_t(64), n_kq_bias_max/(n_kv_max*n_head)/64*64,
                    GGML_PAD((n_tokens + n_kq_chunk_max - 1)/n_kq_chunk_max, 64) });
            const int64_t n_rows_max = std::min(n_seq_tokens, (n_rows_all + n_stream - 1)/n_stream);
            auto rows = [&](ggml_tensor * t, int d, int64_t r0, int64_t nr) {
                int64_t ne[4] = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
                ne[d] = nr;
                return ggml_cont(ctx0, ggml_view_4d(ctx0, t, ne[0], ne[1], ne[2], ne[3], t->nb[1], t->nb[2], t->nb[3], r0*t->nb[d]));
            };
            // the chunks write their rows into one output made before their biases: outputs that stayed live until
            // a final concatenation split the region the next chunk's bias reuses, and the buffer grew with each chunk
            ggml_tensor * out_all = nullptr;
            if (n_rows_max < n_seq_tokens) {
                out_all = ggml_cont(ctx0, ggml_reshape_3d(ctx0, Qcur, n_embd_head*n_head, n_seq_tokens, n_stream));
                ggml_build_forward_expand(gf, out_all);
            }
            for (int64_t r0 = 0; r0 < n_seq_tokens; r0 += n_rows_max) {
                const int64_t nr = std::min(n_rows_max, n_seq_tokens - r0);
                const bool all = nr == n_seq_tokens;
                ggml_tensor * q_c = Qcur;
                ggml_tensor * kq_b = nullptr;
                if (all) {
                    kq_b = ggml_get_rows(ctx0, table, rel_pos);
                    kq_b = ggml_reshape_4d(ctx0, kq_b, n_head, n_kv, nr, n_stream);
                    kq_b = ggml_cont(ctx0, ggml_permute(ctx0, kq_b, 2, 0, 1, 3));
                } else {
                    q_c = ggml_reshape_3d(ctx0, rows(ggml_reshape_4d(ctx0, Qcur, n_embd_head, n_head, n_seq_tokens, n_stream), 2, r0, nr),
                            n_embd_head, n_head, nr*n_stream);
                    ggml_tensor * table_c = ggml_reshape_4d(ctx0, table, table->ne[0], table->ne[1], n_seq_tokens, n_stream);
                    table_c = ggml_view_4d(ctx0, table_c, table_c->ne[0], table_c->ne[1], nr, n_stream,
                            table_c->nb[1], table_c->nb[2], table_c->nb[3], r0*table_c->nb[2]);
                    ggml_tensor *& rel_pos_dev = inp->self_rel_pos_dev[is_swa];
                    if (!rel_pos_dev) {
                        rel_pos_dev = ggml_cont(ctx0, rel_pos);
                    }
                    ggml_tensor * rel_pos_c = ggml_view_3d(ctx0, rel_pos_dev, n_kv, nr, n_stream,
                            rel_pos_dev->nb[1], n_seq_tokens*rel_pos_dev->nb[1], r0*rel_pos_dev->nb[1]);
                    kq_b = ggml_cont(ctx0, ggml_permute(ctx0, ggml_get_rows(ctx0, table_c, rel_pos_c), 2, 0, 1, 3));
                }
                cb(kq_b, "kq_b", il);

                // only the first chunk stores K and V
                ggml_tensor * out = build_attn(inp,
                        nullptr, nullptr, nullptr,
                        q_c, r0 == 0 ? Kcur : nullptr, r0 == 0 ? Vcur : nullptr, kq_b, sinks, 1.0f/sqrtf(n_embd_head), il, r0);
                if (all) {
                    cur = out;
                    break;
                }
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, ggml_reshape_3d(ctx0, out, out->ne[0], nr, n_stream),
                        ggml_view_3d(ctx0, out_all, out_all->ne[0], nr, n_stream, out_all->nb[1], out_all->nb[2], r0*out_all->nb[1])));
            }
            if (out_all) {
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
    }

    ggml_tensor * ffn_inp = cur;
    cb(ffn_inp, "ffn_inp", il);

    // MoE
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, layer.ffn_res_query, il), n_embd, n_tokens);
    if (messages) {
        cur = control(cur, ggml_rms_norm(ctx0, cur, 1e-6f), message(messages, LOOP_CTRL_FFN), message(messages_norm, LOOP_CTRL_FFN),
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
    cb(cur, "l_out", il);

    return cur;
}

void llama_model_inkling2::graph::build_output(ggml_tensor * cur, const std::vector<ggml_tensor *> & blocks) {
    cur = ggml_reshape_2d(ctx0, mix(blocks, cur, model.output_res_query), n_embd, n_tokens);
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);

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
        cur = build_layer(il, il, cur, blocks);
    }

    build_output(cur, blocks);
}
