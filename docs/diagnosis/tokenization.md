# Tokenization parity diagnosis (Mistral-Medium-3.5-128B)

Compares token IDs produced by **four** tokenizers for the same multi-turn fixture
(system + 4 user + 3 assistant messages, `reasoning_effort='none'`).

**Sources compared:**
- `vllm`: vLLM `POST /v1/chat/completions/render` (server on `:8765`).
- `llamacpp`: llama-server `POST /apply-template` then `POST /tokenize` with `add_special=true` (server on `:8766`).
- `mistralcommon`: `MistralTokenizer.from_hf_hub('mistralai/Mistral-Medium-3.5-128B').encode_chat_completion(...)` (mistral-common 1.11.1).
- `hf`: `AutoTokenizer.from_pretrained('mistralai/Mistral-Medium-3.5-128B').apply_chat_template(messages, tokenize=True, add_generation_prompt=True, reasoning_effort='none')['input_ids']` (transformers 5.7.0).

## TL;DR

**All four tokenizers produce byte-identical 434-token sequences for this fixture.** Tokenization is NOT the source of the llama-server vs vLLM accuracy gap on long conversations. Special-token boundaries (`<s>`, `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[MODEL_SETTINGS]`, `[/MODEL_SETTINGS]`, `[INST]`, `[/INST]`, `</s>`) all land in the same positions across all four. The `</s>` count is exactly 3 — one per assistant turn — in every source. The next phase should look at chat-template rendering (the small whitespace differences in the textual rendering, see file lengths), top-logits comparison, and KV-cache / kernel-level differences.

## Token-id length per source

| source | total tokens | rendered text length (chars) |
|---|---|---|
| vllm | 434 | 1823 |
| llamacpp | 434 | 1820 |
| mistralcommon | 434 | 1823 |
| hf | 434 | 1823 |

Note: `llamacpp` text length differs by 3 chars vs the other three (1820 vs 1823) — purely the difference between the rendered jinja text returned by `/apply-template` (which omits the BOS string `<s>`) vs the detokenize-of-IDs which renders BOS as a 3-char literal. The IDs themselves are identical.

## Pairwise diff (vs vLLM as reference)

| source | identical to vllm? | first divergence index |
|---|---|---|
| vllm | (reference) | - |
| llamacpp | **YES** | n/a |
| mistralcommon | **YES** | n/a |
| hf | **YES** | n/a |

## Special-token detection (resolved via llama-server `/tokenize`)

| token | id |
|---|---|
| `<s>` | 1 |
| `</s>` | 2 |
| `[SYSTEM_PROMPT]` | 17 |
| `[/SYSTEM_PROMPT]` | 18 |
| `[INST]` | 3 |
| `[/INST]` | 4 |
| `[MODEL_SETTINGS]` | 36 |
| `[/MODEL_SETTINGS]` | 37 |

## Special-token counts per source

| source | `<s>` | `</s>` | `[SYSTEM_PROMPT]` | `[/SYSTEM_PROMPT]` | `[INST]` | `[/INST]` | `[MODEL_SETTINGS]` | `[/MODEL_SETTINGS]` |
|---|---|---|---|---|---|---|---|---|
| vllm | 1 | 3 | 1 | 1 | 4 | 4 | 1 | 1 |
| llamacpp | 1 | 3 | 1 | 1 | 4 | 4 | 1 | 1 |
| mistralcommon | 1 | 3 | 1 | 1 | 4 | 4 | 1 | 1 |
| hf | 1 | 3 | 1 | 1 | 4 | 4 | 1 | 1 |

All four sources show: 1 `<s>`, 3 `</s>` (= one per assistant turn), 1 `[SYSTEM_PROMPT]` / `[/SYSTEM_PROMPT]` pair, 1 `[MODEL_SETTINGS]` / `[/MODEL_SETTINGS]` pair, 4 `[INST]` / `[/INST]` pairs (= one per user turn including the open-ended trailing turn).

## First 16 input ids per source (for visual sanity)

- **vllm**: `[1, 17, 4568, 1584, 42301, 2784, 55668, 1032, 1051, 1046, 1053, 1044, 1261, 43520, 26242, 11512]`
- **llamacpp**: `[1, 17, 4568, 1584, 42301, 2784, 55668, 1032, 1051, 1046, 1053, 1044, 1261, 43520, 26242, 11512]`
- **mistralcommon**: `[1, 17, 4568, 1584, 42301, 2784, 55668, 1032, 1051, 1046, 1053, 1044, 1261, 43520, 26242, 11512]`
- **hf**: `[1, 17, 4568, 1584, 42301, 2784, 55668, 1032, 1051, 1046, 1053, 1044, 1261, 43520, 26242, 11512]`

## Last 16 input ids per source

- **vllm**: `[4176, 24897, 32196, 17616, 5079, 1034, 2, 3, 7493, 1395, 1032, 1050, 1043, 1050, 1063, 4]`
- **llamacpp**: `[4176, 24897, 32196, 17616, 5079, 1034, 2, 3, 7493, 1395, 1032, 1050, 1043, 1050, 1063, 4]`
- **mistralcommon**: `[4176, 24897, 32196, 17616, 5079, 1034, 2, 3, 7493, 1395, 1032, 1050, 1043, 1050, 1063, 4]`
- **hf**: `[4176, 24897, 32196, 17616, 5079, 1034, 2, 3, 7493, 1395, 1032, 1050, 1043, 1050, 1063, 4]`

## Files

- `outputs/diagnosis_tokenization_{vllm,llamacpp,mistralcommon,hf}.txt`
- `outputs/tok_ids_{vllm,llamacpp,mistralcommon,hf}.json`
- `scripts/diagnosis_tokenization.py`

## Conclusion

Tokenization is fully consistent across all four implementations. **Cross off** "tokenization difference" as a hypothesis for the long-context degradation in llama-server / Q8_0 GGUF.

Likely culprits remain: chat-template subtleties (whitespace), kernel-level numerics in llama.cpp (FP16 accumulation, Q8_0 quantization error compounding), KV-cache dtype, sampler defaults, or the GGUF metadata (RoPE scaling, etc.).
