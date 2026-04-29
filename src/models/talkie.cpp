#include "models.h"

// Talkie 1930 13B graph builder.
//
// Mirrors talkie/src/talkie/model.py:
//   - Weightless RMSNorm everywhere (build_norm with mw=NULL).
//   - Pre-attention RMSnorm (talkie line 144).
//   - Post-RoPE Q/K RMSnorm with no learnable weight (talkie line 102).
//   - Per-head HeadGain on Q after Q-RMSnorm (talkie line 103).
//   - Standard SDPA via build_attn.
//   - Per-block ActGain scalars on attn-residual and mlp-residual branches.
//   - Per-block embed_skip: e_x (post-RMSnorm embedding) added to every layer.
//   - Final RMSnorm, lm_head with global lm_head_gain scalar.
//
// The RoPE sign-convention difference between talkie (rotation by -theta)
// and llama.cpp NEOX (rotation by +theta) is absorbed at convert time by
// negating the second half of head_dim of W_q and W_k weights, so this
// graph uses stock NEOX RoPE unchanged.

llm_build_talkie::llm_build_talkie(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    // e_x = RMSnorm(embed(input_ids)) - the same e_x is added to every layer.
    ggml_tensor * e_x = build_norm(inpL, NULL, NULL, LLM_NORM_RMS, -1);
    cb(e_x, "embed_post_norm", -1);

    // The residual stream starts as e_x (talkie/src/talkie/model.py:191).
    inpL = e_x;

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // Pre-attention RMSnorm (weightless).
        cur = build_norm(inpL, NULL, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_pre_norm", il);

        // self-attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur_pre_rope", il);

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            cb(Kcur, "Kcur_pre_rope", il);

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            cb(Vcur, "Vcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            // RoPE - stock NEOX. Sign convention is absorbed in W_q/W_k at
            // conversion time (see TalkieModel.modify_tensors).
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow
            );
            cb(Qcur, "Qcur_post_rope", il);
            cb(Kcur, "Kcur_post_rope", il);

            // Weightless Q/K RMSnorm (talkie line 102).
            Qcur = build_norm(Qcur, NULL, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_post_qknorm", il);
            Kcur = build_norm(Kcur, NULL, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_post_qknorm", il);

            // HeadGain on Q: broadcast [1, n_head, 1] over [head_dim, n_head, n_tokens].
            ggml_tensor * head_gain = ggml_reshape_3d(ctx0, model.layers[il].attn_head_gain, 1, n_head, 1);
            Qcur = ggml_mul(ctx0, Qcur, head_gain);
            cb(Qcur, "Qcur_post_headgain", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, 1.0f / sqrtf((float) n_embd_head), il);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
            e_x   = ggml_get_rows(ctx0, e_x,   inp_out_ids);
        }

        // Apply ActGain on attn branch and add residual.
        cur = ggml_mul(ctx0, cur, model.layers[il].attn_act_gain);
        cb(cur, "attn_branch_scaled", il);
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "after_attn_residual", il);

        ggml_tensor * mlp_in = cur;

        // Pre-MLP RMSnorm (weightless).
        cur = build_norm(cur, NULL, NULL, LLM_NORM_RMS, il);
        cb(cur, "mlp_pre_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "mlp_out", il);

        // Apply ActGain on mlp branch and add residual.
        cur = ggml_mul(ctx0, cur, model.layers[il].ffn_act_gain);
        cb(cur, "mlp_branch_scaled", il);
        cur = ggml_add(ctx0, cur, mlp_in);
        cb(cur, "after_mlp_residual", il);

        // Embedding-skip: cur = cur + embed_skip * e_x.
        ggml_tensor * e_x_scaled = ggml_mul(ctx0, e_x, model.layers[il].embed_skip_scale);
        cb(e_x_scaled, "embed_skip_branch", il);
        cur = ggml_add(ctx0, cur, e_x_scaled);
        cb(cur, "after_embed_skip", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }
    cur = inpL;

    // Final RMSnorm (weightless).
    cur = build_norm(cur, NULL, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head with global gain: matmul(cur, lm_head_gain * output).
    // Reuses the existing build_lora_mm 3-arg form which already handles
    // a per-tensor weight scale.
    cur = build_lora_mm(model.output, cur, model.lm_head_gain);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
