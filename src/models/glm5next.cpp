#include "models.h"

//
// GLM-5.3-Flash: hybrid KDA (linear) + DSA (nope-only MLA) attention, mHC
// hyper-connections, and a NextN block that is a full DSA decoder layer.
//
// ssm_a holds -exp(A_log), the kimi-k3 convention, so the decay gate reads
// exp(A_log) back as -ssm_a; bailingmoe3 stores +exp(A_log). indistinguishable at
// load time, so conversion/glm5next.py is the only place the sign is checked.
//

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

    // KDA
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,      hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,         hparams.n_embd_head_kda);
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

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0);

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

std::unique_ptr<llm_graph_context> llama_model_glm5next::build_arch_graph(const llm_graph_params & params) const {
    GGML_UNUSED(params);
    throw std::runtime_error("glm5next: graph not implemented yet");
}
