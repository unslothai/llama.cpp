#include "models.h"

#include "llama-memory-recurrent.h"

//
// GLM-5.3-Flash: hybrid KDA (linear) + DSA (nope-only MLA) attention, mHC
// hyper-connections, and a NextN block that is a full DSA decoder layer.
//
// ssm_a holds -exp(A_log), the kimi-k3 convention, so the decay gate reads
// exp(A_log) back as -ssm_a; bailingmoe3 stores +exp(A_log). indistinguishable at
// load time, so conversion/glm5next.py is the only place the sign is checked.
//

// how many positions the indexer keeps: index_topk/index_kpool whole pools, plus the
// always-selected tail pool minus one. below this many cached tokens every position is
// selected and the sparse path is exactly the dense one built here.
//
// asserted rather than measured. an off-by-one here is invisible to every output
// comparison there is: the reference's own seeded off-by-one on this width is
// bit-identical on both the dense and the sparse fixtures. the second form below is the
// independent spelling the parity harness uses, so the two have to agree
static uint32_t glm5next_n_select(const llama_hparams & hparams) {
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k >= hparams.indexer_kpool);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = hparams.indexer_top_k + hparams.indexer_kpool - 1;

    GGML_ASSERT(n_select > hparams.indexer_top_k);
    GGML_ASSERT(n_select == (hparams.indexer_top_k/hparams.indexer_kpool + 1)*hparams.indexer_kpool - 1);

    return n_select;
}

void llama_model_glm5next::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    // indexer k_norm is a LayerNorm with bias; without this key it runs at eps 0
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS,     hparams.f_norm_eps);

    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    GGML_ASSERT(hparams.n_lora_q > 0 && "glm5next requires a q LoRA");
    GGML_ASSERT(hparams.n_rot() == 0 && "glm5next MLA is nope-only");

    // KDA. no GGUF key for linear_num_heads, so the KDA head count is
    // attention.head_count, which also sizes the recurrent state via n_embd_r/s().
    // conversion/glm5next.py refuses a checkpoint where the two differ
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,      hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,         hparams.n_embd_head_kda);
    GGML_ASSERT(hparams.ssm_d_conv > 1);
    GGML_ASSERT(hparams.n_embd_head_kda > 0);
    // required: absent, kimi-k3 selects the softplus branch, a different
    // function, not a missing clamp
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND, hparams.kda_gate_lower_bound);
    GGML_ASSERT(hparams.kda_gate_lower_bound < 0.0f);

    // DSA indexer
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KPOOL,      hparams.indexer_kpool);
    GGML_ASSERT(hparams.indexer_kpool > 0);
    GGML_ASSERT(hparams.indexer_top_k % hparams.indexer_kpool == 0);

    const uint32_t n_select = glm5next_n_select(hparams);
    if (hparams.n_ctx_train > n_select) {
        LLAMA_LOG_WARN("%s: attention is dense above %u cached tokens, but this checkpoint trains to %u. "
                "the sparse selection is not implemented yet\n", __func__, n_select, hparams.n_ctx_train);
    }

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0);

    // trunk residual is hc_mult streams wide (deepseek4); lm_head still sees
    // n_embd, the streams are averaged first
    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    // MoE
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,   hparams.swiglu_clamp_exp,   hparams.n_layer_all, false);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP, hparams.swiglu_clamp_shexp, hparams.n_layer_all, false);

    if (hparams.n_ff_shexp == 0) {
        hparams.n_ff_shexp = hparams.n_ff_exp * std::max(1u, hparams.n_expert_shared);
    }

    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS, hparams.n_layer_nextn, false);
    GGML_ASSERT(hparams.n_layer_nextn < hparams.n_layer_all);

    // n_head_kv == 0 marks a KDA (recurrent) layer, as in kimi-k3 and bailingmoe3.
    // a scalar head_count_kv would make every layer look like DSA, so require both
    uint32_t n_recr = 0;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        hparams.is_recr_impl[il] = hparams.n_head_kv(il) == 0;
        n_recr += hparams.is_recr_impl[il];
    }
    GGML_ASSERT(n_recr > 0 && n_recr < hparams.n_layer() && "glm5next needs a per-layer attention.head_count_kv array");

    // every glm5next indexer is full; glm-dsa gates its indexer on this
    // predicate and the generic loader only zero-fills the array
    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        hparams.is_indexer_full_impl[il] = !hparams.is_recr_impl[il];
    }

    switch (hparams.n_layer()) {
        case 45: type = hparams.n_embd == 4096 && hparams.n_expert == 288 ? LLM_TYPE_313B_A17B : LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_glm5next::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_inner  = head_dim * n_head;
    const int64_t d_conv   = hparams.ssm_d_conv;

    const int64_t q_lora_rank  = hparams.n_lora_q;
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t v_head_dim   = hparams.n_embd_head_v_mla();

    const int64_t n_embd_indexer = hparams.indexer_head_size;
    const int64_t kpool          = hparams.indexer_kpool;

    const int64_t hc_dim     = (int64_t) hparams.dsv4_hc_mult * n_embd;
    const int64_t hc_mix_dim = (2 + (int64_t) hparams.dsv4_hc_mult) * hparams.dsv4_hc_mult;

    // the trunk and the NextN block can be split across two GGUFs in either direction
    const bool mtp_only = (n_layer_nextn > 0) && (ml.get_weight("blk.0.attn_norm.weight") == nullptr);
    const std::string mtp_probe = "blk." + std::to_string(n_layer) + ".nextn.eh_proj.weight";
    const bool trunk_only = (n_layer_nextn > 0) && (ml.get_weight(mtp_probe.c_str()) == nullptr);
    const int trunk_flags = mtp_only   ? TENSOR_NOT_REQUIRED : 0;
    int       mtp_flags   = trunk_only ? TENSOR_NOT_REQUIRED : 0;

    if (!ml.load_mtp) {
        mtp_flags |= TENSOR_SKIP;
    }

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (!output) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    for (int il = 0; il < n_layer_all; ++il) {
        auto & layer = layers[il];
        const int flags = il < n_layer ? trunk_flags : mtp_flags;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", il), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", il), {n_embd}, flags);

        // the NextN block keeps the plain residual, so it has no mHC mixer
        if (il < n_layer) {
            layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", il), {hc_mix_dim}, flags);
            layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", il), {3}, flags);
            layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", il), {hc_dim, hc_mix_dim}, flags);
            layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", il), {hc_mix_dim}, flags);
            layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", il), {3}, flags);
        }

        if (hparams.is_recr(il)) {
            create_tensor_qkv(layer, il, n_embd, d_inner, d_inner, d_inner, flags);

            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", il), {d_conv, 1, d_inner, 1}, flags);
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", il), {d_conv, 1, d_inner, 1}, flags);

            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", il), {head_dim, d_inner}, flags);
            layer.ssm_g_a = create_tensor(tn(LLM_TENSOR_SSM_G_A, "weight", il), {n_embd, head_dim}, flags);
            layer.ssm_g_b = create_tensor(tn(LLM_TENSOR_SSM_G_B, "weight", il), {head_dim, d_inner}, flags);

            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", il), {n_embd, n_head}, flags);
            layer.ssm_a    = create_tensor(tn(LLM_TENSOR_SSM_A,              il), {n_head}, flags);
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT,   "bias",   il), {d_inner}, flags);

            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", il), {head_dim}, flags);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {d_inner, n_embd}, flags);
        } else {
            layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", il), {n_embd, q_lora_rank}, flags);
            layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", il), {q_lora_rank}, flags);
            layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", il), {q_lora_rank, n_head * qk_head_dim}, flags);

            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", il), {n_embd, kv_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", il), {kv_lora_rank}, flags);
            layer.wk_b           = create_tensor(tn(LLM_TENSOR_ATTN_K_B,       "weight", il), {qk_head_dim, kv_lora_rank, n_head}, flags);
            layer.wv_b           = create_tensor(tn(LLM_TENSOR_ATTN_V_B,       "weight", il), {kv_lora_rank, v_head_dim, n_head}, flags);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", il), {n_head * v_head_dim, n_embd}, flags);

            layer.indexer_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", il), {n_embd_indexer}, flags);
            layer.indexer_k_norm_b = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "bias",   il), {n_embd_indexer}, flags);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", il), {n_embd, hparams.indexer_n_head}, flags);
            layer.indexer_attn_k   = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K,   "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", il), {q_lora_rank, hparams.indexer_n_head * n_embd_indexer}, flags);

            // key pooling: DeepSeek-V4 doubles the compressor width, GLM-5.3 does not
            layer.indexer_comp_wgate = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_WGATE, "weight", il), {n_embd, n_embd_indexer}, flags);
            layer.indexer_comp_ape   = create_tensor(tn(LLM_TENSOR_INDEXER_COMPRESSOR_APE,   "weight", il), {n_embd_indexer, kpool}, flags);
        }

        if (il < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", il), {n_embd, n_ff}, flags);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", il), {n_ff, n_embd}, flags);
        } else {
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", il), {n_embd, n_expert}, flags);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   il), {n_expert}, flags);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", il), {n_embd, hparams.n_ff_exp, n_expert}, flags);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", il), {n_embd, hparams.n_ff_exp, n_expert}, flags);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", il), {hparams.n_ff_exp, n_embd, n_expert}, flags);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", il), {n_embd, hparams.n_ff_shexp}, flags);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", il), {hparams.n_ff_shexp, n_embd}, flags);
        }

        if (il >= n_layer) {
            layer.nextn.eh_proj = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ, "weight", il), {2 * n_embd, n_embd}, flags);
            layer.nextn.enorm   = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,   "weight", il), {n_embd}, flags);
            layer.nextn.hnorm   = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,   "weight", il), {n_embd}, flags);

            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", il), {n_embd}, flags);
            // absent in the checkpoint: NextN shares the trunk's embeddings and
            // lm_head. only accepted if an export adds them
            layer.nextn.embed_tokens     = create_tensor(tn(LLM_TENSOR_NEXTN_EMBED_TOKENS,     "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
            layer.nextn.shared_head_head = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_HEAD, "weight", il), {n_embd, n_vocab}, flags | TENSOR_NOT_REQUIRED);
        }
    }
}

//
// KDA layer
//
// one depthwise conv over the concatenated q|k|v channels, as in the reference: it
// leaves the conv state one contiguous block, which is what build_conv_state needs to
// snapshot for recurrent-state rollback. three separate convs would be numerically
// identical but would restate the rollback write three times
//
ggml_tensor * llama_model_glm5next::graph::build_kda_layer(
        const llama_layer & layer,
        llm_graph_input_rs * inp_rs,
        ggml_tensor * cur,
        int il) {
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim * n_head;
    const int64_t d_conv       = hparams.ssm_d_conv;
    const int64_t n_seqs       = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    const auto * mctx_cur = inp_rs->mctx;

    // f, g and beta read the layer input, NOT the convolved q/k/v
    ggml_tensor * inp = cur;

    ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq, inp);
    ggml_tensor * Kcur = ggml_mul_mat(ctx0, layer.wk, inp);
    ggml_tensor * Vcur = ggml_mul_mat(ctx0, layer.wv, inp);

    ggml_tensor * qkv = ggml_concat(ctx0, ggml_concat(ctx0, Qcur, Kcur, 0), Vcur, 0);
    qkv = ggml_reshape_3d(ctx0, qkv, 3*d_inner, n_seq_tokens, n_seqs);
    cb(qkv, "kda_qkv", il);

    ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
    ggml_tensor * conv_in = build_conv_state(inp_rs, conv_states_all, qkv, d_conv, 3*d_inner, il);

    // stored separately (kimi-linear, kimi-k3), stacked back into the single kernel
    ggml_tensor * conv_w = ggml_concat(ctx0,
            ggml_concat(ctx0,
                ggml_reshape_2d(ctx0, layer.ssm_q_conv, d_conv, d_inner),
                ggml_reshape_2d(ctx0, layer.ssm_k_conv, d_conv, d_inner), 1),
            ggml_reshape_2d(ctx0, layer.ssm_v_conv, d_conv, d_inner), 1);

    // SiLU is applied to the conv output, not to the projections
    ggml_tensor * conv_out = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_in, conv_w));
    cb(conv_out, "kda_conv", il);

    const size_t nb_qkv  = ggml_row_size(conv_out->type, 3*d_inner);
    const size_t nb_head = ggml_row_size(conv_out->type, head_dim);

    Qcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, 0);
    Kcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, d_inner));
    Vcur = ggml_view_4d(ctx0, conv_out, head_dim, n_head, n_seq_tokens, n_seqs,
            nb_head, nb_qkv, nb_qkv*n_seq_tokens, ggml_row_size(conv_out->type, 2*d_inner));

    // 1e-6 is the reference's own constant, not the model's norm eps. ggml_l2_norm
    // divides by max(sqrt(sum), eps) where the reference uses sqrt(sum + eps); at
    // head_dim 128 the clamp never binds, so close but not bit-exact
    Qcur = ggml_l2_norm(ctx0, Qcur, 1e-6f);
    Kcur = ggml_l2_norm(ctx0, Kcur, 1e-6f);
    cb(Qcur, "kda_q_norm", il);
    cb(Kcur, "kda_k_norm", il);

    // the 1/sqrt(head_dim) query scale is applied inside build_delta_net, after this norm

    // forget gate. gate_lower_bound is a multiplicative scale, not a clamp:
    //   g = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    // ssm_a holds -exp(A_log), so exp(A_log) * y == -(y * ssm_a)
    ggml_tensor * g = ggml_mul_mat(ctx0, layer.ssm_f_b, ggml_mul_mat(ctx0, layer.ssm_f_a, inp));
    g = ggml_add(ctx0, g, layer.ssm_dt_b);
    g = ggml_reshape_3d(ctx0, g, head_dim, n_head, n_tokens);
    g = ggml_mul(ctx0, g, ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1));
    g = ggml_sigmoid(ctx0, ggml_scale(ctx0, g, -1.0f));
    g = ggml_scale(ctx0, g, hparams.kda_gate_lower_bound);
    g = ggml_reshape_4d(ctx0, g, head_dim, n_head, n_seq_tokens, n_seqs);
    cb(g, "kda_gate", il);

    ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, inp);
    beta = ggml_sigmoid(ctx0, ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs));
    cb(beta, "kda_beta", il);

    ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

    ggml_tensor * out = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g, beta, state, il);

    // the fallbacks return a permuted view, the fused op a contiguous one; cont
    // either way rather than depend on which ran
    ggml_tensor * o = ggml_cont_3d(ctx0, out, head_dim, n_head, n_tokens);
    cb(o, "kda_scan_out", il);

    // low-rank output gate (kimi-k3 has a single full-rank ssm_g instead)
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.ssm_g_b, ggml_mul_mat(ctx0, layer.ssm_g_a, inp));
    gate = ggml_reshape_3d(ctx0, gate, head_dim, n_head, n_tokens);

    // RMS over head_dim only, one weight shared by every head, then a plain sigmoid
    // gate: not the SiLU that FusedRMSNormGated defaults to
    ggml_tensor * normed = build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, gate));
    cb(gated, "kda_normed", il);

    cur = ggml_mul_mat(ctx0, layer.wo, ggml_cont_2d(ctx0, gated, d_inner, n_tokens));
    cb(cur, "kda_out", il);

    return cur;
}

//
// DSA layer, dense
//
// below index_topk + index_kpool - 1 resident positions the indexer selects every
// position, so sparse selection is exactly full attention. this builds that limit:
// correct MLA over the whole cache, no indexer, no kpool, no top-k
//
// the absorbed form is used, as in deepseek2/deepseek32/glm-dsa: q_nope is pushed
// through wk_b so that q.k is taken against the 512-wide latent directly, which is
// what the cache holds (is_mla() suppresses the V allocation and V becomes a view of
// K). the naive form would have to expand the latent back to n_head 256-wide keys and
// values on every step and would need a V cache this memory layout does not have
//
ggml_tensor * llama_model_glm5next::graph::build_dsa_layer(
        const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn,
        ggml_tensor * cur,
        int il) const {
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // nope-only. every other MLA port splits q and k into a nope and a rope half and
    // ropes the second one; here that half is zero-width, so there is no split, no
    // concat and no ggml_rope_ext anywhere in the text tower
    GGML_ASSERT(hparams.n_rot() == 0);

    // the reference scales by qk_head_dim^-0.5, i.e. over the MLA head size, not over
    // n_embd_head_k: after absorption q is kv_lora_rank wide and 1/sqrt(512) would be
    // a different model
    const float kq_scale = 1.0f/sqrtf(float(qk_head_dim));

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_a, cur);
    q = build_norm(q, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(q, "dsa_q_a_norm", il);

    q = ggml_mul_mat(ctx0, layer.wq_b, q);
    q = ggml_reshape_3d(ctx0, q, qk_head_dim, n_head, n_tokens);
    cb(q, "dsa_q_b", il);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv, "dsa_kv_a_norm", il);

    // {qk_head_dim, n_tokens, n_head}
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    // {qk_head_dim, kv_lora_rank, n_head} x {qk_head_dim, n_tokens, n_head}
    q = ggml_mul_mat(ctx0, layer.wk_b, q);

    // {kv_lora_rank, n_head, n_tokens}. deepseek2 gets this contiguous for free out of
    // the concat with the roped half, which does not exist here
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    cb(q, "dsa_q_absorbed", il);

    // absorbed MLA is MQA: one head of keys, and V is the same latent row as K
    ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    cb(k, "dsa_kv_latent", il);

    cur = build_attn(inp_attn,
            layer.wo, nullptr, nullptr,
            q, k, k, nullptr, nullptr, layer.wv_b, kq_scale, il);
    cb(cur, "dsa_out", il);

    return cur;
}

ggml_tensor * llama_model_glm5next::graph::build_layer_attn(
        const llama_model & model,
        llm_graph_input_mem_hybrid_k * inp_mem,
        ggml_tensor * cur,
        int il) {
    if (hparams.is_recr(il)) {
        return build_kda_layer(model.layers[il], inp_mem->get_recr(), cur, il);
    }

    return build_dsa_layer(model.layers[il], inp_mem->get_attn(), cur, il);
}

ggml_tensor * llama_model_glm5next::graph::build_layer_ffn(
        const llama_model & model,
        ggml_tensor * cur,
        int il) const {
    const auto & layer = model.layers[il];

    // the leading dense layers clamp the same way the experts do: the reference
    // routes both through one Glm5NextTextMLP, so swiglu_limit is not MoE-only
    if (il < (int) hparams.n_layer_dense_lead) {
        return build_ffn(cur,
                layer.ffn_up,   nullptr, nullptr,
                layer.ffn_gate, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    }

    // noaux_tc: exp_probs_b biases top-k selection only, the weights are the
    // unbiased sigmoid scores. n_group is 1, so the group mask is a no-op
    ggml_tensor * moe_out = build_moe_ffn(cur,
            layer.ffn_gate_inp,
            layer.ffn_up_exps,
            layer.ffn_gate_exps,
            layer.ffn_down_exps,
            layer.ffn_exp_probs_b,
            n_expert, hparams.n_expert_used,
            LLM_FFN_SILU, hparams.expert_weights_norm,
            hparams.expert_weights_scale,
            (llama_expert_gating_func_type) hparams.expert_gating_func,
            il);

    ggml_tensor * shexp = build_ffn(cur,
            layer.ffn_up_shexp,   nullptr, nullptr,
            layer.ffn_gate_shexp, nullptr, nullptr,
            layer.ffn_down_shexp, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
    cb(shexp, "ffn_shexp", il);

    // shared expert unscaled: routed_scaling_factor is applied inside build_moe_ffn,
    // after norm_topk_prob, to the routed weights only
    return ggml_add(ctx0, moe_out, shexp);
}

llama_model_glm5next::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    ggml_tensor * cur;

    ggml_tensor * inp         = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // MLA absorption leaves a K-only cache holding the kv_lora_rank latent, so the
    // attention half of the hybrid memory is the _k variant, as in bailingmoe3
    llm_graph_input_mem_hybrid_k * inp_mem = build_inp_mem_hybrid_k();

    GGML_ASSERT(ubatch.n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens * ubatch.n_seqs);

    const int64_t hc = hparams.dsv4_hc_mult;

    // hc_mult exact copies of the embedding: no scaling, no one-hot into stream 0
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    for (int il = 0; il < n_layer; ++il) {
        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(ctx0, inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        cur = build_hc_pre(inpL,
                model.layers[il].hc_attn_fn,
                model.layers[il].hc_attn_scale,
                model.layers[il].hc_attn_base,
                &post, &comb, il);
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_layer_attn(model, inp_mem, cur, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        cur = build_hc_pre(inpL,
                model.layers[il].hc_ffn_fn,
                model.layers[il].hc_ffn_scale,
                model.layers[il].hc_ffn_base,
                &post, &comb, il);
        cb(cur, "hc_ffn_pre", il);

        // expand before the sublayer so op offload does not pull the mHC state
        // onto the expert weights' backend, as in deepseek4
        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_layer_ffn(model, cur, il);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(ctx0, inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    if (inp_out_ids) {
        // flattened: get_rows needs one token's streams to be one contiguous row
        ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        inpL = ggml_reshape_3d(ctx0, ggml_get_rows(ctx0, flat, inp_out_ids), n_embd, hc, n_outputs);
    }

    // no hc_head tensor here: unweighted mean, not DeepSeek-V4's learned gated head
    cur = build_hc_mean(ctx0, inpL);
    cb(cur, "hc_mean", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    // llama_init_from_model accepts an MTP context whenever n_layer_nextn > 0,
    // which every glm5next checkpoint has; without this it silently runs the trunk
    GGML_ASSERT(params.gtype != LLM_GRAPH_TYPE_DECODER_MTP && "glm5next NextN graph not implemented yet");

    return std::make_unique<graph>(*this, params);
}
