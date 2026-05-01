#include "models.h"

// DeepSeek-V4-Flash forward graph.
//
// PR1: STUB GRAPH. The full mHC + Compressor + dense-Indexer + grouped-output
// attention + sqrt-softplus / hash-routed MoE forward will be implemented in a
// follow-up commit. This stub only embeds + RMSNorms + projects to logits so
// the model loads end-to-end and llama-server boots; output is junk.
//
// Reference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/model.py
llm_build_deepseek_v4::llm_build_deepseek_v4(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // input embedding
    inpL = build_inp_embd(model.tok_embd);

    // final RMSNorm (model.py:787)
    cur = build_norm(inpL, model.output_norm, /*mb=*/nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    // lm_head (model.py:788, 715)
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
