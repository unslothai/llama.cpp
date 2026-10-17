# DeepSeek-V4.1 (`deepseek41`) GGUF contract

Owner: builder A (arch, graph). Builder B (converter) writes exactly this. Shapes are GGUF `ne` order (fastest dim first),
i.e. the reverse of the HF `[out, in]` shape. HF names are the official checkpoint names (`layers.N.*`, `mtp.M.*`).

## Arch
`general.architecture = "deepseek41"` (`gguf.MODEL_ARCH.DEEPSEEK41`, `LLM_ARCH_DEEPSEEK41`).
`block_count` = 40 (backbone only) or 43 with `nextn_predict_layers = 3` when the DSpark blocks `mtp.0..2` are written as
`blk.40..42` (optional, later).

## Tensors
Unchanged from deepseek4 (same names and shapes):
`token_embd`, `output_norm`, `output`; per block `attn_norm`, `attn_sinks` (F32 `{n_head}`), `attn_q_a`, `attn_q_a_norm`,
`attn_q_b`, `attn_kv`, `attn_kv_a_norm`, `attn_output_a`, `attn_output_b`, `hc_{attn,ffn}_{fn,base,scale}` (F32),
`ffn_norm`, `ffn_gate_inp`, `exp_probs_b` (F32 `.bias`), `exp_probs_b_vl` (F32 `.bias`, optional),
`ffn_{gate,up,down}_exps` (MXFP4), `ffn_{gate,up,down}_shexp`.

Removed vs deepseek4: `output_hc_{fn,base,scale}` (no hc_head: the final collapse uses the last FFN pre-mix),
`attn_compressor_ape`, `indexer.compressor_*`, `ffn_gate_tid2eid`, `nextn.*`.

| GGUF name | HF name | ne | dtype | layers |
|---|---|---|---|---|
| `blk.N.attn_compressor_kv.weight` | `layers.N.attn.compressor.wkv.weight` | `{n_embd, head_dim}` | BF16 | kv sources (2,8,14,20) |
| `blk.N.attn_compressor_gate.weight` | `layers.N.attn.compressor.wgate.weight` | `{n_embd, head_dim}` | BF16 | kv sources with ratio > 1 (2,8,14) |
| `blk.N.attn_compressor_norm.weight` | `layers.N.attn.compressor.norm.weight` | `{head_dim}` | F32/BF16 | kv sources |
| `blk.N.indexer.attn_k.weight` | `layers.N.attn.indexer.wk.weight` | `{head_dim, index_head_dim}` | BF16 | kv sources |
| `blk.N.indexer.k_norm.weight` | `layers.N.attn.indexer.k_norm.weight` | `{index_head_dim}` | F32/BF16 | kv sources |
| `blk.N.indexer.attn_q_b.weight` | `layers.N.attn.indexer.wq_b.weight` | `{q_lora_rank, index_n_head*index_head_dim}` | BF16 (FP8 dequant) | index sources (2,8,...,36) |
| `blk.N.indexer.proj.weight` | `layers.N.attn.indexer.weights_proj.weight` | `{n_embd, index_n_head}` | BF16 | index sources |
| `blk.N.engram_embd.weight` | `layers.N.engram.embed.weight` + `.scale` | `{engram_head_dim, num_embeddings[i]}` | MXFP8 (B) / BF16 / F32 | engram layers (1,14) |
| `blk.N.engram_kv.weight` | `layers.N.engram.wkv.weight` | `{n_hash_cols*engram_head_dim, n_embd*(hc+1)}` | BF16 (FP8 dequant) | engram layers |
| `blk.N.engram_gate.weight` | `layers.N.engram.q_weight * layers.N.engram.k_weight` (folded, elementwise) | `{n_embd, hc}` | F32 | engram layers |
| `blk.40.dspark_main_proj.weight` | `mtp.0.main_proj.weight` | `{n_embd*len(target_layers), n_embd}` | BF16 | DSpark stage 0 (optional) |
| `blk.40.dspark_main_norm.weight` | `mtp.0.main_norm.weight` | `{n_embd}` | F32 | DSpark stage 0 |
| `blk.42.dspark_norm.weight` | `mtp.2.norm.weight` | `{n_embd}` | F32 | last DSpark stage |
| `markov_w1.weight` | `mtp.2.markov_head.embed.weight` | `{markov_rank, n_vocab}` | BF16 | DSpark |
| `markov_w2.weight` | `mtp.2.markov_head.head.weight` | `{markov_rank, n_vocab}` | BF16 | DSpark |
| `conf_proj.weight` | `mtp.2.confidence_head.proj.weight` | `{n_embd+markov_rank, 1}` | F32 | DSpark |

Notes
- `n_hash_cols = (engram.max_ngram_size - 1) * engram.head_count` (24). `engram_kv` output rows keep the HF order:
  rows `[0, hc*n_embd)` are the `hc` keys (copy-major), rows `[hc*n_embd, (hc+1)*n_embd)` the shared value.
- `engram_embd` rows are written exactly as stored (no TP padding rows: exactly `engram_num_embeddings[i]` rows). The
  row for hash id r is `float(w[r]).view(8,32) * 2^(scale[r]-127)`. Never repack, never offload: the graph gathers it on
  the host with `ggml_get_rows`. Any type with get_rows on CPU works for the graph.
- FP8 linear weights use 32x32 `ue8m0` blocks (derive the block from the scale shape, do not hard-code 128).
- Keep F32 in every output file (checked by gguf_build `eval/verify_types.py`): `ffn_gate_inp` (router), `exp_probs_b`,
  `exp_probs_b_vl`, every `*norm.weight` (incl. `attn_compressor_norm`, `indexer.k_norm`), `hc_*`, `attn_sinks`, `engram_gate`.
- Every layer `0..39` has all backbone attention/HC/MoE tensors; DSpark blocks have 128 experts (`dspark.expert_count`).

## Keys (`deepseek41.` prefix)
Same as deepseek4: `context_length`, `embedding_length`, `block_count`, `feed_forward_length` (unused),
`expert_feed_forward_length`, `expert_count` (384), `expert_used_count` (6), `expert_shared_count`, `expert_weights_scale`
(1.5), `expert_weights_norm` (true), `expert_gating_func` (`gguf.ExpertGatingFuncType.SQRTSOFTPLUS` = 4),
`swiglu_clamp_exp` / `swiglu_clamp_shexp` (f32[block_count], 10.0), `attention.head_count` (64), `attention.head_count_kv` (1),
`attention.key_length` = `attention.value_length` (512), `rope.dimension_count` (64), `attention.q_lora_rank`,
`attention.sliding_window` (128), `attention.layer_norm_rms_epsilon` (1e-20), `attention.output_group_count`,
`attention.output_lora_rank`, `attention.compress_ratios` (u32[>= block_count], all 43 entries ok),
`attention.compress_rope_freq_base` (160000), `rope.freq_base` (10000), `rope.scaling.type` yarn, `rope.scaling.factor` 16,
`rope.scaling.original_context_length` 65536, `rope.scaling.yarn_beta_fast` 32, `rope.scaling.yarn_beta_slow` 1,
`attention.indexer.head_count` (32), `attention.indexer.key_length` (128), `attention.indexer.top_k` (512),
`hyper_connection.count` (4), `hyper_connection.sinkhorn_iterations` (20), `hyper_connection.epsilon` (1e-6),
`embedding_length_out` = n_embd*hc (as deepseek4).
Not written: `hash_layer_count`.

New:
| key | type | value (V4.1-Flash-1017) |
|---|---|---|
| `attention.kv_source_layers` | u32[] | [2, 8, 14, 20] |
| `attention.indexer.source_layers` | u32[] | [2, 8, 14, 20, 24, 28, 32, 36] |
| `attention.indexer.candidate_source_layer` | i32 | 20 (-1 = off) |
| `attention.indexer.candidate_top_k_blocks` | u32 | 2048 |
| `attention.indexer.candidate_block_size` | u32 | 8 |
| `engram.layers` | u32[] | [1, 14] |
| `engram.num_embeddings` | u32[] | [384006168, 384016682] |
| `engram.max_ngram_size` | u32 | 4 |
| `engram.vocab_size` | u32 | 16000000 (prime search start, informational) |
| `engram.head_count` | u32 | 8 |
| `engram.head_dim` | u32 | 256 |
| `engram.pad_token_id` | u32 | 2 (raw id; the graph maps it through token_map) |
| `engram.compressed_vocab_size` | u32 | 99092 |
| `engram.token_map` | i32[n_vocab] | `build_compressed_token_map(tokenizer)` from inference/engram.py |
| `engram.multipliers` | u64[n_engram*max_ngram] | layer-major, `compute_hash_multipliers` |
| `engram.primes` | u32[n_engram*(max_ngram-1)*head_count] | layer, n-gram order (2..max), head; `EngramLayout.primes` |
| `dspark.block_size` | u32 | 5 |
| `dspark.noise_token_id` | u32 | 128799 |
| `dspark.target_layers` | u32[] | [37, 38, 39] |
| `dspark.markov_rank` | u32 | 256 |
| `dspark.expert_count` | u32 | 128 |
| `dspark.expert_used_count` | u32 | 3 |

Engram offsets are not stored: per layer they are the exclusive cumulative sum of that layer's 24 primes (order-major,
head-minor), exactly `NgramHashState.offsets`.

gguf-py writer helpers: `add_attention_kv_source_layers`, `add_indexer_source_layers`, `add_indexer_candidate_source_layer`,
`add_indexer_candidate_top_k_blocks`, `add_indexer_candidate_block_size`, `add_engram_{layers,num_embeddings,max_ngram_size,
vocab_size,head_count,head_dim,pad_token_id,compressed_vocab_size,token_map,multipliers,primes}`,
`add_dspark_{block_size,noise_token_id,target_layers,markov_rank,expert_count,expert_used_count}`.
