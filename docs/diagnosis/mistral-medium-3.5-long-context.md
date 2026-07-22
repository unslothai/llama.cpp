# Mistral-Medium-3.5-128B llama.cpp long-context degradation — diagnosis

## TL;DR

llama-server (Q8_0 GGUF from `bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF`)
collapses into deterministic repetition loops after ~1000–1500 generated tokens,
producing broken Python syntax (e.g. `pygame.display.set mode((800, 600)` —
missing the underscore in `set_mode` and the closing paren). vLLM (FP8 of the
same model) does not exhibit this; it stops naturally at 1496 tokens. **HF
transformers BF16 inference of the same model also degrades** in the same way,
ruling out llama.cpp/Q8_0 as the *unique* culprit.

This matches what unsloth has already published on
[`unsloth/Mistral-Medium-3.5-128B-GGUF`](https://huggingface.co/unsloth/Mistral-Medium-3.5-128B-GGUF):

> Testing shows that this behavior occurs **regardless of who or how** the model
> was converted GGUF. The model initially responds correctly, but over long
> context, does not work properly. **Mistral has now labeled GGUF support as a
> WIP**.

This document records the experiments that were run against vLLM (port 8765,
FP8) and llama-server (port 8766, Q8_0) and what they ruled out. It is
intended as input to a deeper fix.

## Setup

- vLLM 0.20.1rc1.dev127, FP8, `--tensor-parallel-size 2 --port 8765 --tool-call-parser mistral --enable-auto-tool-choice --reasoning-parser mistral --max_num_batched_tokens 16384 --max_num_seqs 128 --gpu_memory_utilization 0.8 --attention-backend FLASH_ATTN`, GPUs 4,5.
- llama-server `b1-aab6821` (unslothai fork) and ggml-org master, Q8_0,
  `--tensor-split 1,1 --port 8766 --jinja --ctx-size 32768 --parallel 1`,
  GPUs 2,3.
- HF transformers 5.7.0, `Mistral3ForConditionalGeneration` BF16,
  `attn_implementation="eager"`, GPUs 6,7.

## Phases run

| phase | topic | result | location |
| --- | --- | --- | --- |
| 1 | Tokenization parity (vLLM / llama-server / mistral-common / HF) | **identical** 434 tokens | `outputs/diagnosis_tokenization.md` |
| 2 | Chat template (GGUF jinja vs unsloth vs mistralai vs HF tokenizer) | semantically equivalent (single trivial diff: a disabled `if false` assertion at line 201) | `outputs/diagnosis_chat_template.md` (and `outputs/template_*.jinja`) |
| 3 | Top-k logits agreement | top-1 token agrees on simple prompts; logit *distributions* differ noticeably (vLLM is less peaked) | `outputs/diagnosis_logits.md` |
| 4 | GGUF metadata vs HF config | match. RoPE: `factor=64`, `freq_base=1e6`, `original_context_length=4096`, `yarn_beta_fast=4`, `yarn_beta_slow=1`, `yarn_log_mul=1.0`. All YARN parameters present in GGUF and equal to `text_config.rope_parameters` | `outputs/diagnosis_config.md`, `outputs/gguf_metadata_full.txt` |
| 5 | HF transformers BF16 ground truth | **also degrades** (recall score 0/4 on the interleaved 11-turn test). T2 Flappy Bird hits 1500-token cap with looping syntax errors in tail | `outputs/hf_groundtruth_*` |
| 6 | KV cache dtype: F16 (default) / BF16 / F32 | **no effect on degradation** | `outputs/diag_single_flappy_*.json` |
| 7 | Rebuild llama.cpp with `-DGGML_CUDA_FORCE_CUBLAS=ON -DGGML_CUDA_FORCE_MMQ=OFF` and runtime `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` | **no effect on degradation** | `outputs/llama_server_compute32f_*.log` |
| 8 | Matched samplers (`temp=0.1, top_p=1, top_k=64, min_p=0.05, seed=42`) | **no effect**; tried `repetition_penalty ∈ {1.0,1.05,1.1,1.2}`, `frequency_penalty ∈ {0,0.1,0.3,0.5}`, `dry_multiplier ∈ {0,0.5,0.8}` — all still loop | `outputs/matched_*` |
| 9 | Architecture diff — `llama.cpp/src/models/mistral3.cpp` vs HF `transformers/models/ministral3/modeling_ministral3.py` | **equivalent**. Same SwiGLU, RMSNorm-fp32, kq_scale=1/√head_dim, optional Llama-4 attn temperature scale (gated and disabled for this model since `llama_4_scaling_beta=0`). YARN `attn_factor` chain in `llama-context.cpp` matches HF's `_compute_yarn_parameters` — both produce `1.0` for `mscale=mscale_all_dim=1.0`. | `outputs/diagnosis_arch.md` |

## Empirical convergence point

Single-turn `Create a Flappy Bird Python game` (greedy, temperature=0):

```
| backend                          | finish | n_out | tail snippet
| -------------------------------- | ------ | ----- | -----------
| vLLM FP8                         | stop   | 1496  | "...Would you like me to explain any specific part?"
| llama-server Q8_0 cuBLAS         | length | 2048  | "if __name__ == \"__main__,\\n    sys.exit()\\n```\\n\\n" (loop)
| llama-server Q8_0 BF16 KV        | length | 2048  | "if pipe.x < 0\\n    self.pipes.remove(pipe)" (loop)
| llama-server Q8_0 F32 KV + no FA | length | 2048  | identical loop
| HF transformers BF16             | length | 1500  | "pipe1 = pipe(50, 100, 50, 100\\n    pipe2 = pipe(50, 100, 50, 100" (loop)
```

For llama-server, output is clean up to ~1000 tokens then degrades:

```
mt=  600: clean
mt= 1000: still clean
mt= 1500: looping
```

Common prefix between vLLM and llama-server greedy outputs is exactly **66
characters**: `# Flappy Bird Game in Python\n\nHere's a complete implementation of `

Then:
- vLLM picks `a Flappy Bird game using Pygame`
- llama-server picks `Flappy Bird using Pygame` (no leading `a`).

## Working hypotheses (still open)

- **H1: Q8_0 quantization error**. Q8_0 gives ~16-bit mantissa precision per
  block of 32. Cumulative error across 88 layers × 1000 decode steps may be
  enough to flip top-1 tokens that compound into loops. *Counter-evidence:* HF
  BF16 with the FP8 safetensors **also** loops, so quantization can't be the
  whole story.
- **H2: Numerical kernel issue specific to Mistral-Medium-3.5's shape (88
  layers, head_dim=128, head_count=96, head_count_kv=8, vocab=131072,
  intermediate=28672, rope_freq_base=1e6 with YARN factor 64)** — common to
  llama.cpp ggml-cuda *and* HF eager. vLLM avoids it because it uses CUTLASS
  FP8 GEMMs with FP32 accumulators and possibly a different attention kernel
  (`FLASH_ATTN` selected by vLLM in our setup).
- **H3: Some prefill artifact** that vLLM mitigates via chunked prefill
  (`--max_num_batched_tokens 16384`). Not yet tested in isolation.

## What is NOT the cause

- Tokenization (4 tokenizers byte-identical).
- Chat template (4 templates render to identical token streams).
- GGUF metadata (all numerical config matches HF, including all YARN params).
- Sampler (no setting of temp/top_p/top_k/min_p/repetition_penalty/freq_penalty/dry that breaks the loop).
- KV cache dtype.
- Flash attention vs default attention in llama.cpp.
- cuBLAS vs MMQ kernels.
- Architecture code (mistral3.cpp implements the same residual/RMSNorm/SwiGLU/RoPE flow as HF).
- Sliding window (None for this model — both honour that).

## Artefacts

All under `workspace_5/outputs/`:

- `diagnosis_tokenization*.{md,txt}` + `tok_ids_*.json`
- `diagnosis_chat_template.md` + `template_*.jinja` + `template_diff_*.txt`
- `diagnosis_config.md` + `gguf_metadata_full.txt`
- `diagnosis_arch.md`
- `diagnosis_logits.md` + `diagnosis_logits_raw.json`
- `hf_groundtruth_alphabet_*.txt` + `hf_groundtruth_interleaved_*.json` + `hf_groundtruth_tok_ids_*.json`
- `recall_interleaved_{vllm,llamacpp}_*.txt`
- `multi_turn_recall_{vllm,llamacpp}_*.txt`
- `diag_single_flappy_*.json` + `.log`
- `matched_*.{json,txt}`
- All llama-server / vllm / HF logs in `workspace_5/logs/`.

## Suggested next steps for a real fix

1. Dump per-layer activations from both vLLM and HF for the same input, and
   compare layer-by-layer to find where they first diverge meaningfully.
2. If that divergence localises to RMSNorm or attention softmax, force FP32
   accumulators in those ops in llama.cpp's CUDA kernels for `mistral3`.
3. Compare against `llama-bench --perplexity` numbers per quant; Q8_0 vs Q6_K
   vs F16-converted-from-FP8 GGUF to determine whether the issue scales with
   precision.
4. Consider writing a reference forward pass in PyTorch using the GGUF weights
   (via `gguf-py`) and the HF arch and comparing token-by-token.
