# Phase 9 — model architecture diff (llama.cpp `mistral3.cpp` vs HF `ministral3`)

## TL;DR

The two implementations are **structurally equivalent** for the inference path
that Mistral-Medium-3.5 actually exercises. SwiGLU MLP, RMSNorm with FP32 cast,
1/sqrt(head_dim) attention scale, pre-norm decoder layers, and YARN-scaled RoPE
all match. The only optional code path is **Llama-4-style attention temperature
scaling**, which is gated on `llama_4_scaling_beta != 0` in HF and on
`hparams.f_attn_temp_scale != 0.0` in llama.cpp — both gates evaluate false for
this model (`llama_4_scaling_beta = 0` per HF config, no
`attn_temperature_scale` key in the GGUF), so the path is skipped on both
sides. **Architecture is not the cause** of the long-context degradation.

## Side-by-side mapping

| concern | HF `Ministral3*` | llama.cpp `mistral3.cpp` |
| --- | --- | --- |
| pre-attention norm | `Ministral3RMSNorm(eps=rms_norm_eps)` cast→fp32 then back | `build_norm(LLM_NORM_RMS)` |
| Q/K/V proj | `nn.Linear(bias=False)` | `build_qkv(...)` |
| RoPE | `apply_rotary_pos_emb` with cached `cos,sin` (yarn) | `ggml_rope_ext` with `n_ctx_orig, freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow` |
| attn temperature scale | `Q *= 1 + beta * log(1 + floor(pos/orig_max))` (only if `beta!=0`) | `Q = ggml_mul(Q, inp_attn_scale)` (only if `f_attn_temp_scale!=0`) |
| attention | `attention_interface(scale=1/sqrt(head_dim), sliding_window=None)` | `build_attn(..., kq_scale = 1/sqrt(n_embd_head))` |
| sliding window | `getattr(config,'sliding_window',None)` → `None` for this model | not configured (correct) |
| O proj | `o_proj = nn.Linear(bias=False)` | `model.layers[il].wo` |
| residual after attn | `hidden = residual + hidden` | `ffn_inp = ggml_add(cur, inpSA)` |
| post-attn norm | `Ministral3RMSNorm` | `build_norm(LLM_NORM_RMS)` (ffn_norm) |
| MLP | SwiGLU: `down(silu(gate(x)) * up(x))` | `build_ffn(LLM_FFN_SILU, LLM_FFN_PAR)` (parallel = silu(gate)*up then down) |
| residual after MLP | `hidden = residual + hidden` | `ggml_add(cur, ffn_inp)` |
| MoE branch | n/a (Mistral-Medium-3.5 is dense) | guarded by `model.layers[il].ffn_gate_inp == nullptr` → dense path |

## RoPE specifically

Both honour every YARN parameter in the GGUF (`yarn_beta_fast=4`,
`yarn_beta_slow=1`, `factor=64`, `original_context_length=4096`,
`freq_base=1e6`, `yarn_log_multiplier=1.0`) — all of which match
`rope_parameters` in `mistral_medium_check/config.json`. RoPE math is correct.

## Llama-4 attn temperature scale

```python
def get_llama_4_attn_scale(positions_ids, beta, max_position_embeddings):
    return (1 + beta * torch.log(1 + torch.floor(positions_ids / max_position_embeddings)))[:, None, :, None]
```

`beta = config.rope_parameters["llama_4_scaling_beta"] = 0`, so the multiplier
collapses to **1.0** at every position. llama.cpp side-steps the entire
multiplication when the GGUF doesn't carry the key. Both safe, both equivalent.

## What this rules out

- ❌ Sliding-window attention mismatch
- ❌ Wrong RoPE dimensions / wrong YARN parameters
- ❌ Missing Llama-4 temperature scaling (it's a no-op for this model)
- ❌ Activation function mismatch (both SwiGLU)
- ❌ Pre-norm vs post-norm placement
- ❌ Attention scale factor

## What it does NOT rule out

- Numerical precision in CUDA kernels (FP16 accumulators in `ggml-cuda` for
  the matmul or attention path).
- Q8_0 quantization rounding error in long-attention contexts.
- KV-cache dtype (default F16 → numerical drift over many tokens).
- Sampler behaviour (min_p default of 0.05 in llama-server vs vLLM not applying
  it).

Phase 7 (rebuild with `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1`) and Phase 6
(F16/BF16/Q8_0 KV-cache experiment) and Phase 8 (matched samplers) cover those.
