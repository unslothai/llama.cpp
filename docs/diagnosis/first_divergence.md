# First-divergence pin-down (vLLM vs llama-server, greedy)

Reproducer: `repro_first_divergence.py` in this directory.

Same input ("Create a Flappy Bird Python game" + Mistral-Medium-3.5 SYSTEM_PROMPT
with `reasoning_effort=none`), greedy (temperature=0.0), max_tokens=100, on:

- vLLM 0.20.1rc1.dev127 FP8 + `--attention-backend FLASH_ATTN` (port 8765)
- llama-server (b1-aab68217b unslothai fork, also tested ggml-org master) Q8_0 (port 8766)

The two outputs share an **identical 13-token prefix**:

```
# Flappy Bird Game in Python\n\nHere's a complete implementation of
```

Token 14 is where they first diverge. **The top-2 tokens at this position are
the same in both servers**, but their *relative order* is flipped:

| rank | vLLM | logprob | llama-server | logprob |
| --- | --- | --- | --- | --- |
| 1 | ` a` | **−0.314** | ` Fl` | **−0.289** |
| 2 | ` Fl` | −1.314 | ` a` | −1.430 |
| 3 | ` the` | −7.689 | ` the` | −4.434 |
| 4 | `Fl` | −12.564 | `Fl` | −11.420 |

Both have the same top-2 candidates, but llama-server compresses ` a` from
−0.31 to −1.43 (a Δ of ~1.1 in logprob, i.e. a factor of 3 in probability)
while vLLM compresses ` Fl` from −1.31 to −0.31 (same Δ in the other
direction). On a logprob scale of −0.3 vs −1.4 these tokens are within 1 nat
of each other; greedy-decoding *must* pick exactly one, and the order flip
sends the two backends down different trajectories.

After that single token flip:
- vLLM continues with `... a Flappy Bird game using Pygame. This version includes the core mechanics ...`
- llama-server continues with `... Flappy Bird using Pygame. This version includes all the classic elements ...`

Both trajectories are coherent at this point. The two outputs both produce
clean code for ~600–1000 generated tokens, after which **only the
llama-server trajectory** degenerates into broken syntax and repetition (e.g.
`pygame.display.set mode((800, 600)` — `set_mode` becomes `set mode`).

## Logits progression at fixed prefix

Reproducer: `repro_logits_progression.py`.

When we take vLLM's full greedy output and **feed it as a fixed prefix** at
checkpoints 50/200/500/1000/1400, both servers' top-1 next-token agrees at
*every* checkpoint:

| n_decoded | vLLM top-1 | vLLM logprob | llama-server top-1 | llama-server logprob | match |
| --- | --- | --- | --- | --- | --- |
| 50 | ` ``` ` | −0.0233 | ` ``` ` | −0.0000 | ✓ |
| 200 | `Y` | ~0 | `Y` | ~0 | ✓ |
| 500 | `.rect` | −0.0001 | `.rect` | −0.2432 | ✓ |
| 1000 | `_p` | ~0 | `_p` | −0.0601 | ✓ |
| 1400 | `   ` | ~0 | `   ` | −0.0028 | ✓ |

So the long-context degeneration is **not** caused by the model converging on
different next-token answers given the same prefix. It is caused by a single
~1-nat precision flip near the start, after which vLLM and llama-server walk
different (still individually plausible) decoding paths — and the
llama-server path happens to land in a degenerate attractor.

## Cross-check on Q4_K_M

Repeating the experiment with `bartowski/.../Q4_K_M` (~74 GiB GGUF) on
llama-server: identical degeneration tail. The same wrong top-2 ranking at
token 14 occurs, then the trajectory degenerates *more* than Q8_0 — for
example `pygame.display.set mode sdl hWSIZER, sdl lg2` syntax garbage. So
this is uniform across llama.cpp quants, not a Q8_0-specific bug.

## Implication

The remaining hypothesis is that ggml-cuda's accumulator precision in the
matmul or attention path for this specific model shape (88 layers,
head_count=96, head_count_kv=8, head_dim=128, vocab=131072,
intermediate=28672, rope_freq_base=1e6 with YARN factor=64) is producing
logits that are subtly *flatter* than the reference. The ranking flip on
token 14 is consistent with this: top-2 tokens are within 1 nat in both
servers, but llama-server compresses the gap by ~0.4 nat in a way that puts
` Fl` ahead of ` a` instead of behind.

A targeted next step is to fix the matmul accumulator in
`ggml-cuda/mmq.cu` (or the relevant GEMM path) to FP32 specifically for
`LLM_ARCH_MISTRAL3` and re-test. `GGML_CUDA_FORCE_CUBLAS=1` was already tried
without effect, but only `GGML_CUDA_FORCE_CUBLAS_COMPUTE_32F=1` only helps the
cuBLAS half-precision path; Q8_0 dequant + matmul takes a different path.
