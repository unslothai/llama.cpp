#include "models.h"

#include "llama-memory-recurrent.h"
#include "llama-memory-hybrid.h"
#include "llama-kv-cache-kpool.h"

//
// GLM-5.3-Flash: hybrid KDA (linear) + DSA (nope-only MLA) attention, mHC
// hyper-connections, and a NextN block that is a full DSA decoder layer.
//
// ssm_a holds -exp(A_log), the kimi-k3 convention, so the decay gate reads
// exp(A_log) back as -ssm_a; bailingmoe3 stores +exp(A_log). indistinguishable at
// load time, so conversion/glm5next.py is the only place the sign is checked.
//

// positions the indexer keeps: index_topk/index_kpool whole pools plus the
// always-selected tail pool minus one. below this many cached tokens every position is
// selected, so sparse selection is exactly the dense path built here.
//
// asserted, not measured: an off-by-one here is invisible to output comparison (the
// reference's own off-by-one on this width is bit-identical on both fixtures). the
// second assert is the parity harness's independent spelling, so the two must agree
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
    // The reference HARDCODES nn.LayerNorm(index_head_dim, eps=1e-6); it does not
    // read it from the config, and rms_norm_eps is 1e-5. Warned about rather than
    // asserted, for two reasons. No output comparison can see it: seeding eps=1e-5
    // into the reference leaves the logits BIT-IDENTICAL at 512 tokens and moves
    // the index jaccard by 1.7e-5 at 2048, against a bf16 floor of 0.63. And an
    // assert would abort test-llama-archs, whose synthetic models have no reason
    // to carry a converter contract
    if (hparams.f_norm_eps <= 0.0f || hparams.f_norm_eps > 2e-6f) {
        LLAMA_LOG_WARN("%s: indexer k_norm eps is %g, but the reference hardcodes 1e-6. "
                "this is invisible to every output comparison; check the converter\n",
                __func__, (double) hparams.f_norm_eps);
    }

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

    // below n_select resident positions the indexer selects every visible one, so
    // the dense path is not an approximation of the sparse one, it IS it
    const uint32_t n_select = glm5next_n_select(hparams);
    LLAMA_LOG_INFO("%s: indexer selection width = %u cells (%u pools of %u, plus a %u-wide tail)\n",
            __func__, n_select, hparams.indexer_top_k/hparams.indexer_kpool,
            hparams.indexer_kpool, hparams.indexer_kpool - 1);

    // mHC
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    GGML_ASSERT(hparams.dsv4_hc_mult > 0);

    // trunk residual is hc_mult streams wide (deepseek4); lm_head still sees
    // n_embd, the streams are averaged first -- so n_embd_out stays n_embd.
    //
    // deepseek4 sets this to hc_mult*n_embd, and inheriting that here was a bug:
    // our t_embd is build_norm(build_hc_mean(...)), which is [n_embd, n_tokens],
    // but n_embd_out() would report 4*n_embd, so llama-context.cpp reads
    // n_outputs*n_embd_out floats out of a tensor holding a quarter of that.
    // The assert there sizes the DESTINATION, so nothing catches the short SOURCE.
    // Only --embeddings / llama_get_embeddings* reach it, which is why plain
    // generation never showed it.
    //
    // deepseek4 needs the wide value to size its MTP `h` input; when our NextN
    // graph starts consuming it, give MTP its own width rather than widening this.
    hparams.n_embd_out_impl = 0;

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
// the lightning indexer, pooled
//
// Writes this layer's indexer key and compressor gate into the indexer cache, and
// then - when the cache is large enough for selection to bind - returns the I32
// ATTENTION-cache cell indices this layer's queries select, ready for the scatter in
// build_attn_sparse. `qr` is the shared q LoRA residual, i.e. q_a_norm(q_a_proj(x)),
// which the indexer's own wq_b consumes; `cur` is the layer input.
//
// The store is NOT gated on the sparse path and the scoring is. Gating both the same
// way leaves every cell written below n_select - the first 2051 positions of every
// sequence on the real model - with no indexer state at all, and the first ubatch to
// cross n_select then pools cells that were never written.
//
// Three points where the reference implementations disagree, resolved 2-of-3 by
// reading transformers (modular_glm5_next.py Glm5NextTextIndexer), sglang
// (dsa_indexer_kpool.py) and vLLM (glm5next/nvidia/attention.py):
//
//   * the weights_proj GEMM runs in fp32. sglang gives it params_dtype=fp32 and feeds
//     x.float(); vLLM does the same and says why - bf16 head-gates move logits by
//     ~1e-2, enough to flip near-tie pool rankings on long context. transformers is
//     the outlier and rounds the activation to bf16.
//   * k_norm is a LayerNorm WITH BIAS at eps 1e-6, hardcoded in transformers and in
//     vLLM; sglang leaves torch's 1e-5 default. It is NOT f_norm_rms_eps (1e-5 here)
//     and NOT ggml's 0 default. The value comes from the GGUF and load_arch_hparams
//     warns when it is not 1e-6, because no output comparison can see it.
//   * the ReLU between the QK dot and the head weighting is explicit in transformers
//     and in sglang's non-pooled indexer; in the pooled path both engines hand it to
//     the DeepGEMM MQA-logits kernel, so neither validates it. It is easy to drop and
//     is written out here.
//
// No Hadamard rotation: sglang and vLLM rotate q and k by H128 before their fp8
// kernel, but H is orthogonal so (Hq).(Hk) == q.k. It exists to spread magnitude ahead
// of fp8 quantisation; scoring in f32 here it would only cost accuracy. transformers,
// the semantic reference, has none.
//
ggml_tensor * llama_model_glm5next::graph::build_indexer(
        const llama_layer & layer,
        llm_graph_input_kpool * inp_kp,
        ggml_tensor * cur,
        ggml_tensor * qr,
        bool scoring,
        int il) const {
    const int64_t d_idx   = hparams.indexer_head_size;
    const int64_t n_ihead = hparams.indexer_n_head;
    const int64_t r       = hparams.indexer_kpool;

    const auto * mctx_idx = inp_kp->mctx_idx;

    // a genuine LayerNorm: weight AND bias. glm-dsa runs this same norm at eps 0 today
    GGML_ASSERT(layer.indexer_k_norm_b != nullptr && "the indexer k_norm is a LayerNorm with bias");

    ggml_tensor * ik = build_norm(ggml_mul_mat(ctx0, layer.indexer_attn_k, cur),
            layer.indexer_k_norm, layer.indexer_k_norm_b, LLM_NORM, il);
    cb(ik, "indexer_k", il);

    // the pooling gate is a SECOND, INDEPENDENT projection of the hidden state, not a
    // reuse of the indexer key. it has to be cached beside the key: the compressor
    // mixes a pool's member keys with a softmax over these gates, and a pool is only
    // rebuilt once its members have left the batch
    ggml_tensor * gate = ggml_mul_mat(ctx0, layer.indexer_comp_wgate, cur);
    cb(gate, "indexer_gate", il);

    // {d_idx, 2, n_tokens}: head 0 is the key, head 1 the gate. llama_memory_hybrid
    // allocates the second head exactly for this and only when indexer_kpool > 0
    ggml_tensor * packed = ggml_concat(ctx0,
            ggml_reshape_3d(ctx0, ik,   d_idx, 1, n_tokens),
            ggml_reshape_3d(ctx0, gate, d_idx, 1, n_tokens), 1);
    ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, packed, inp_kp->k_idxs, il));

    if (!scoring) {
        return nullptr;
    }

    // {d_idx, 2, n_kv, n_stream}
    ggml_tensor * kbuf = mctx_idx->get_k(ctx0, il);

    const int64_t n_kv     = kbuf->ne[2];
    const int64_t n_stream = kbuf->ne[3];
    const int64_t n_tps    = n_tokens/n_stream;
    const int64_t n_pools  = inp_kp->pool_cells->ne[0]/r;

    GGML_ASSERT(kbuf->ne[0] == d_idx && kbuf->ne[1] == 2 &&
            "the pooled indexer cache needs a key head and a gate head");
    GGML_ASSERT(kbuf->nb[1] == (size_t) d_idx*kbuf->nb[0] && "key and gate must be adjacent in a cell");
    GGML_ASSERT(n_tokens == n_tps*n_stream);

    // one cell's key and gate as a single row, so that the members of a pool are
    // gathered once rather than twice: {2*d_idx, n_kv, n_stream}
    ggml_tensor * kg_rows = ggml_view_3d(ctx0, kbuf, 2*d_idx, n_kv, n_stream,
            kbuf->nb[2], kbuf->nb[3], 0);

    // gather each pool's members. pool_cells names a cell per (pool, slot); slots that
    // are not resident hold 0 rather than a negative sentinel, because ggml_get_rows
    // has none. The resulting garbage pools are neutralised by pool_bias below, never
    // by a NaN: unlike the reference, no -inf ever enters the compressor softmax.
    // ggml_get_rows always yields F32, so the pooling runs in F32 even though the
    // indexer cache is F16
    ggml_tensor * members = ggml_get_rows(ctx0, kg_rows, inp_kp->pool_cells);
    cb(members, "indexer_pool_members", il);

    const size_t nb_mem = members->nb[1];

    ggml_tensor * mem_k = ggml_view_4d(ctx0, members, d_idx, r, n_pools, n_stream,
            nb_mem, nb_mem*r, members->nb[2], 0);
    ggml_tensor * mem_g = ggml_view_4d(ctx0, members, d_idx, r, n_pools, n_stream,
            nb_mem, nb_mem*r, members->nb[2], d_idx*members->nb[0]);

    // d_idx independent r-way softmaxes over the SLOT axis, so the slot axis has to be
    // dim 0. ape is added PRE-softmax and is indexed by LOGICAL SLOT, so it broadcasts
    // over pools and streams; pool_cells is built in position order, so slot m is
    // position p % kpool and the two agree by construction
    ggml_tensor * keys_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_k, 1, 0, 2, 3));
    ggml_tensor * gate_t = ggml_cont(ctx0, ggml_permute(ctx0, mem_g, 1, 0, 2, 3));

    ggml_tensor * ape = ggml_cont(ctx0, ggml_transpose(ctx0, layer.indexer_comp_ape));
    gate_t = ggml_add(ctx0, gate_t, ggml_reshape_4d(ctx0, ape, r, d_idx, 1, 1));

    ggml_tensor * probs = ggml_soft_max(ctx0, gate_t);
    cb(probs, "indexer_pool_probs", il);

    // per-channel weighted average over the pool's members -> {d_idx, n_pools, 1, n_stream}
    ggml_tensor * pool_k = ggml_sum_rows(ctx0, ggml_mul(ctx0, keys_t, probs));
    pool_k = ggml_reshape_4d(ctx0, pool_k, d_idx, n_pools, 1, n_stream);
    cb(pool_k, "indexer_pool_k", il);

    // {d_idx, n_tps, n_ihead, n_stream}. no rope: n_rot() is 0 for the whole text tower
    ggml_tensor * iq = ggml_mul_mat(ctx0, layer.indexer_attn_q_b, qr);
    iq = ggml_reshape_4d(ctx0, iq, d_idx, n_ihead, n_tps, n_stream);
    iq = ggml_permute(ctx0, iq, 0, 2, 1, 3);
    cb(iq, "indexer_q", il);

    // {n_pools, n_tps, n_ihead, n_stream}: pool_k is MQA and broadcasts over the heads
    ggml_tensor * kq = ggml_mul_mat(ctx0, pool_k, iq);

    // {n_ihead, n_tps, n_pools, n_stream}, contiguous for the relu and the head sum.
    // the ReLU sits BETWEEN the per-head dot product and the head weighting: moving it
    // to either side is a different function, because the head weights are sign-free
    // and the sum is not a convex combination
    kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));
    ggml_tensor * score = ggml_relu(ctx0, kq);
    cb(score, "indexer_score", il);

    // sign-unconstrained head weights: no softmax, no abs, no relu. Both scale
    // constants - the reference's softmax_scale = d_idx^-0.5 and its n_heads^-0.5 head
    // factor - are folded in here, on an {n_ihead, n_tokens} tensor rather than on the
    // {n_pools, n_tps, n_ihead} score tensor. relu is positively homogeneous and both
    // constants are positive, so this is exactly the same function, and it is what the
    // engines and the in-tree glm-dsa both do
    ggml_tensor * w = ggml_mul_mat(ctx0, layer.indexer_proj, cur);
    ggml_mul_mat_set_prec(w, GGML_PREC_F32);
    w = ggml_reshape_4d(ctx0, w, n_ihead, n_tps, 1, n_stream);
    w = ggml_scale(ctx0, w, 1.0f/sqrtf(float(d_idx*n_ihead)));
    cb(w, "indexer_weights", il);

    // {1, n_tps, n_pools, n_stream} -> {n_pools, n_tps, n_stream}
    ggml_tensor * pool_score = ggml_sum_rows(ctx0, ggml_mul(ctx0, score, w));
    pool_score = ggml_cont(ctx0, ggml_permute(ctx0, pool_score, 2, 1, 0, 3));
    pool_score = ggml_reshape_3d(ctx0, pool_score, n_pools, n_tps, n_stream);

    // -INFINITY on every pool the reference's `pool_valid & pool_visible` rejects, the
    // query's own trailing pool included, so that no budget is spent on it
    pool_score = ggml_add(ctx0, pool_score, inp_kp->pool_bias);
    cb(pool_score, "indexer_pool_score", il);

    // Top-k over POOLS at index_topk/index_kpool, then expand each selected pool to its
    // members. This is the reference's own two-step (topk over the pool axis, then
    // selected_indices = pool_indices[batch_idx, selected]) and it is NOT
    // interchangeable with a single top-k of width index_topk over member cells.
    //
    // The cell-level form is the tempting one and the argument for it is false. It says
    // a pool's members carry its score bit-exactly, so the cut must land on a pool
    // boundary. But F.relu drives most pool scores to exactly 0.0, so tie groups SPAN
    // pools, the cut falls inside one, and ggml_top_k - explicitly unordered among
    // equals - takes an arbitrary 1..kpool-1 members of the pool it lands in. A
    // pool-aligned top-k WIDTH does not save it: the ties are not aligned to anything.
    // Measured on TinySparse at 512 tokens, the cell-level form leaves a partial pool
    // on 7.51% of query rows at L3 and 5.93% at L7 (70 and 47 partial pools); this form
    // leaves none. The index-set jaccard does not separate them - it scored the broken
    // form at 0.9958/0.9764 against a bf16 noise floor of 0.9779/0.8385, i.e. ABOVE the
    // floor - which is why scripts/glm5next_pool_integrity.py exists
    const int64_t select_k = llama_kpool_select_k(n_pools, hparams.indexer_top_k, r);
    GGML_ASSERT(select_k > 0 && select_k <= n_pools);

    // {select_k, n_tps, n_stream} of POOL ordinals
    ggml_tensor * sel = ggml_cont(ctx0, ggml_top_k(ctx0, pool_score, (int) select_k));
    cb(sel, "indexer_top_k_pools", il);

    // Expand pools to members: gather whole rows of `kpool` cells out of pool_cells.
    // The query axis folds into the gather's row axis, which is what lets ONE
    // ggml_get_rows serve every query, while the stream axis stays where get_rows wants
    // it (src0 dim 2 is indexed by the index tensor's dim 1)
    ggml_tensor * pc3      = ggml_reshape_3d(ctx0, inp_kp->pool_cells, r, n_pools, n_stream);
    ggml_tensor * sel_flat = ggml_reshape_2d(ctx0, sel, select_k*n_tps, n_stream);

    // {r, select_k*n_tps, n_stream} -> {r*select_k, n_tps, n_stream}
    ggml_tensor * top_k = ggml_get_rows(ctx0, pc3, sel_flat);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32 && "pool_cells is I32, so the gather stays I32");
    top_k = ggml_reshape_3d(ctx0, top_k, r*select_k, n_tps, n_stream);
    cb(top_k, "indexer_top_k", il);

    return top_k;
}

//
// DSA layer. `scoring` false takes the dense limit: full MLA over the whole cache, no
// kpool, no top-k, which is what sparse selection collapses to below
// glm5next_n_select() resident positions
//
// absorbed form, as in deepseek2/deepseek32/glm-dsa: q_nope is pushed through wk_b so
// q.k is taken against the 512-wide latent the cache actually holds (is_mla() drops the
// V allocation and V becomes a view of K). the naive form would re-expand the latent to
// n_head 256-wide k/v every step and needs a V cache this layout does not have
//
ggml_tensor * llama_model_glm5next::graph::build_dsa_layer(
        const llama_layer & layer,
        llm_graph_input_attn_k * inp_attn,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il) const {
    const int64_t qk_head_dim  = hparams.n_embd_head_k_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;

    // nope-only: the rope half is zero-width, so no split, no concat and no rope
    // anywhere in the text tower
    GGML_ASSERT(hparams.n_rot() == 0);

    // scale is over the MLA head size, as in the reference, not over the post-absorption
    // width: 1/sqrt(kv_lora_rank) = 1/sqrt(512) would be a different model
    const float kq_scale = 1.0f/sqrtf(float(qk_head_dim));

    ggml_tensor * qr = ggml_mul_mat(ctx0, layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "dsa_q_a_norm", il);

    // the indexer shares this q LoRA residual with the main MLA path and consumes it
    // with its own wq_b, so it is built here rather than being handed the layer input
    // twice. it also writes the indexer cache, which happens on the dense path too
    ggml_tensor * top_k = inp_kp ? build_indexer(layer, inp_kp, cur, qr, scoring, il) : nullptr;

    ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, qk_head_dim, n_head, n_tokens);
    cb(q, "dsa_q_b", il);

    ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);
    kv = build_norm(kv, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(kv, "dsa_kv_a_norm", il);

    // {qk_head_dim, n_tokens, n_head}
    q = ggml_permute(ctx0, q, 0, 2, 1, 3);

    // {qk_head_dim, kv_lora_rank, n_head} x {qk_head_dim, n_tokens, n_head}
    q = ggml_mul_mat(ctx0, layer.wk_b, q);

    // {kv_lora_rank, n_head, n_tokens}. deepseek2 gets this contiguous for free from the
    // concat with the roped half, which does not exist here
    q = ggml_cont(ctx0, ggml_permute(ctx0, q, 0, 2, 1, 3));
    cb(q, "dsa_q_absorbed", il);

    // absorbed MLA is MQA: one head of keys, and V is the same latent row as K
    ggml_tensor * k = ggml_reshape_3d(ctx0, kv, kv_lora_rank, 1, n_tokens);
    cb(k, "dsa_kv_latent", il);

    if (top_k) {
        cur = build_attn_sparse(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b,
                top_k, inp_kp->sel_mask, inp_kp->cand_mask, kq_scale, il);
    } else {
        cur = build_attn(inp_attn,
                layer.wo, nullptr, nullptr,
                q, k, k, nullptr, nullptr, layer.wv_b, kq_scale, il);
    }
    cb(cur, "dsa_out", il);

    return cur;
}

ggml_tensor * llama_model_glm5next::graph::build_layer_attn(
        const llama_model & model,
        llm_graph_input_mem_hybrid_k * inp_mem,
        llm_graph_input_kpool * inp_kp,
        bool scoring,
        ggml_tensor * cur,
        int il) {
    if (hparams.is_recr(il)) {
        return build_kda_layer(model.layers[il], inp_mem->get_recr(), cur, il);
    }

    return build_dsa_layer(model.layers[il], inp_mem->get_attn(), inp_kp, scoring, cur, il);
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

    // MLA absorption leaves a K-only cache holding the latent, so the attention half of
    // the hybrid memory is the _k variant, as in bailingmoe3
    llm_graph_input_mem_hybrid_k * inp_mem = build_inp_mem_hybrid_k();

    // One pooling map for the whole ubatch. pool_cells, pool_bias, sel_mask and
    // cand_mask depend only on the cells and the ubatch, never on the layer, so
    // rebuilding them per DSA layer would cost O(n_kv * n_tokens) host writes eleven
    // times over - at 128 Ki cells and 512 tokens that dominates prefill on its own.
    //
    // The map is built whenever the model HAS an indexer cache, because the indexer
    // key and gate store runs unconditionally. Only `scoring` is gated: below
    // index_topk + index_kpool - 1 resident positions the reference selects every
    // visible position, so the dense build_attn is not an approximation there, it is
    // the same function.
    //
    // Gated on n_ctx and not on the ubatch's n_kv, even though n_kv is what actually
    // decides whether the budget binds. n_kv grows as the cache fills, so gating on it
    // would flip the graph's topology partway through a run. n_ctx is fixed for the
    // lifetime of the context, so the topology is decided once. The cost is that a
    // context configured larger than n_select runs the indexer even while the cache is
    // still short, where it selects every visible pool: wasted work, never a wrong
    // answer, and llama_kpool_select_k clamps the budget to the pools that exist.
    llm_graph_input_kpool * inp_kp = nullptr;
    bool indexer_scoring = false;
    {
        const auto * mctx_hyb = static_cast<const llama_memory_hybrid_context *>(mctx);

        if (mctx_hyb->get_idx() != nullptr) {
            indexer_scoring = cparams.n_ctx > glm5next_n_select(hparams);

            inp_kp = build_inp_kpool(mctx_hyb,
                    inp_mem->get_attn()->get_kq_mask(), indexer_scoring);
        }
    }

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

        cur = build_layer_attn(model, inp_mem, inp_kp, indexer_scoring, cur, il);

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
