#include "models.h"

// GLM-5.3-Flash reuses the GLM-OCR ViT unchanged: no learned position embeddings, no
// post-conv norm, q/k norms and biases throughout, and an RMSNorm called post_layernorm.
// ref: https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/modeling_glm5_next.py
//
// The one difference is the clamp on the SwiGLU gate and up projections. It applies to
// the per-block MLP and to the merger alike, and both read hparams.ffn_op, so selecting
// FFN_SILU_CLAMP for this projector type is enough to cover the pair.
ggml_cgraph * clip_graph_glm5next::build() {
    GGML_ASSERT(hparams.ffn_op == FFN_SILU_CLAMP);
    GGML_ASSERT(model.norm_embd_w == nullptr);
    GGML_ASSERT(model.position_embeddings == nullptr);

    return clip_graph_glm4v::build();
}
