# Builder A status (DeepSeek-V4.1-Flash-1017 llama.cpp port)

Branch `dsv41` in `$W/wt_dsv41` (off llama.cpp_pinset fd55cb98f). Contract for builder B: `TENSORS.md`.

## Porting workflow (unslothai/scripts #649 bundle, `temp/us_gguf`: gguf_build_workflow.md A1-A5, PORTING.md)
| step | state | evidence |
|---|---|---|
| A1 recon + reuse map + fixture + converter + graph | done | reuse map in deepseek41.cpp header (V4 HC/MLA/MoE/raw cache reused; new: Engram, lagged mHC, ratio 1/2 + sharing + candidates). Two fixtures: `tiny_fixture` (small dims, fast, seeded-bug harness) and `tiny_fixture_realdim` (A1 rule: REAL dims 5120 / 64x512 heads / rope 64 / q 1280 / o 8x1024 / expert 2304 / index 32x128 / Engram 8x256 n-gram 4 / window 128 / block 8 / YaRN 65536x16; only counts shrunk: 10 layers, 8 experts top-2, Engram bucket start 1009, index top-k 64, 16 candidate blocks). Rare-path thresholds: window 128, ratio-2 top-k binds > 128 tokens, candidate mask binds > 128 ratio-1 rows; prompt wiki1100 exceeds all. Reference = official model.py (no transformers implementation of deepseek_v41 exists in 4.57 / 5.x, so ref_logits.py --generic cannot load it); the fixture writes the ref_logits.py npz schema + full logits. Fixture generates under the reference (prefill vs decode max abs 2.7e-5 at real dims). |
| A2 parity (bundled tools/dump_logits + port/compare_ref.py, F32 gate --max-kl-mean 1e-6 --min-top1 1.0 --min-cos 0.99999) | done (CPU), CUDA calibrated | see "A2 numbers" below; 13/13 seeded bugs FAIL on the small fixture, 5/5 on real dims |
| A3 template + V4.1 DSML tool parser | done (1d51c092f) | template `models/templates/deepseek-ai-DeepSeek-V4.1-Flash-1017.jinja` (= `$W/models/dsv41/chat_template.jinja`, 5/5 byte parity with encoding.py). V4.1 tags carry a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`); detection in common/chat.cpp keys on `｜DSML｜ calls>` (the template never names `dsml_token`), parser = the V4 parser with the spaced tags. test-chat: the V4 parser suite ported to V4.1 markup (content, thinking, single / parallel / mixed-type calls, content before calls, continuation, reconstruction) all pass; negative control (V4.1 tag switch off) aborts on the V4.1 block. |
| A4 never break other models | done | test-chat all passed; test-llama-archs CPU all 137 passed; CUDA: full suite exceeded its 2 h box (killed, no verdict), so the archs sharing touched code were run: deepseek4/41, deepseek2/32, glm-dsa, glm5-next, maple, qwen3moe, qwen35moe, qwen4exp on CUDA + CPU + Meta: 28/28 passed; deepseek41 on CPU / CUDA / Meta OK; test-backend-ops for used ops pass; V4 logits bitwise identical base vs head; no vocab change (vocab differential N/A) |
| A5 pin PR / pinset sim | local branch `dsv41-pin` (worktree `wt_dsv41_pin`), not pushed | see "A5" below |

### A2 numbers
- CPU, F32 KV, FA off, single-shot / ub 7 / ub 1, swa_full 1 and 0: small fixture KL mean <= 4.9e-12, top-1 1.0, router top-k
  exact on all 10 MoE layers; real-dims fixture (incl. wiki1100) KL mean <= 2.7e-11, max 3.0e-10, top-1 1.0, router exact
  40/40 (prompt x layer), l_out rel err <= 4e-6 on every layer.
- Seeded bugs (scripts/dsv41/seeded_bugs.py, every one must FAIL): Engram signed-sqrt dropped, Engram history order,
  Engram ungated, mHC not lagged, output collapse with wrong pre, V4 per-head q norm, compressor softmax axis, index-K rope
  dims, compressed-KV rope dims, indexer ReLU dropped, candidate pin dropped, candidate mask not applied, top-k reuse
  replaced by dense: 13/13 FAIL (KL 4e-3 .. 2). On real dims: candidate pin / candidate mask / reuse / ReLU / Engram
  history: 5/5 FAIL. Free oracle: below 16 visible ratio-1 rows the candidate bugs are inert (short16 passes), as the
  architecture guarantees.
- CUDA (B200), `NVIDIA_TF32_OVERRIDE=0` alone: FAILS (KL 1e-3..7e-2): the override does not reach the mmf tf32 path for
  F32 weights (PORTING.md Phase 5). With `GGML_CUDA_F32_EXACT=1` (debug knob in mmf.cu, default off) as well: small
  fixture passes every cell (ub 0/7/1, graphs on and off); real dims pass everything except wiki1100 at ub 0/7 (KL mean
  2.3e-6, top-1 1.0, max 2.4e-3 at t=1035). Chased: one indexer near-tie at layer 2, t=1035, k-th vs (k+1)-th CPU score
  gap 4.4e-5 relative while CUDA batched idx_score differs from CPU by 1.7e-5..2.6e-5 relative (score = signed sum of
  relu terms, ~10x cancellation); the flip then propagates to that token's selections in later layers. Decode (ub 1) passes.
  Classified as precision, not a defect; CUDA gate = CPU gate except one near-tie token.
- CUDA graphs on vs off (B10 A/B): identical results, no hang under the 1200 s watchdog (timeout -k 15 => SIGKILL).
- FA on (F16 attention) is an acceptance config, not a parity config (see older numbers below).

## Gate on agent C's fixtures (the parity oracle from now on; temp/dsv41_fixtures, refs/<p>/matched.npy)
Build: `temp/wt_dsv41_gate` = dsv41 + B's dsv41-mxfp8 (MXFP8 Engram), tools `temp/dsv41_tools/gate_cpu/`. GGUF: B's
`<fx>-F32.gguf` with MXFP4 experts dequantized to F32 (`models/dsv41/c_fx/<fx>-F32X.gguf`, scripts/dsv41/experts_to_f32.py;
with MXFP4 experts the CPU matmul quantizes activations to q8_0 and the gate fails at ~1e-3, as C found). Runner
`scripts/dsv41/gate_c.sh`. CPU, F32 KV, FA off. Gate KL(matched || llama) mean < 1e-3: PASS on every fixture.
| fixture | prompts | KL mean (range) | worst p99 | top-1 min |
|---|---|---|---|---|
| fx_base (Engram 1, 6; MXFP8 tables) | t100..t3001, c1600, sp1600 | 9e-12 .. 2.3e-5 | 8.5e-5 | 1.0000 |
| fx_noengram | same | 9e-12 .. 1.3e-5 | 8.9e-5 | 0.9987 |
| fx_engram0 (Engram on layer 0) | same | 9e-12 .. 2.7e-5 | 1.6e-4 | 0.9987 |
| fx_long (real 2048-block pool) | t2001 / sp20001 (ub 4096) | 1.2e-5 / 5.1e-4 | 1.1e-2 | 0.9861 |
| fx_base ub 7 / ub 1 | t129, t1027, sp1600 / t129, t700 | identical to one-shot | | 1.0000 |
llama.cpp vs matched is the same size as matched_f32 vs matched_f64 (the fp32 floor). sp20001 is the long-context
near-tie regime characterised above.
Selection sets vs C's matched_traces.pt (`scripts/dsv41/sel_vs_traces.py`, fx_base): t1027 L2-L7 0 rows, L8 2, L12 4,
pool 0; t3001 L2 5 .. L12 121 rows, pool 21 / 1977. Every mismatch is 1 swap; the earliest divergences (t=646, 877, 973,
998) have score gaps 7e-7..3.3e-6 of the row top score (near-ties); later ones follow earlier flips. Waiting on C's
f32-vs-f64 selection floor to close this gate.
CUDA (`GGML_CUDA_F32_EXACT=1`, gate build `build-cuda`): fx_base t129 / t1027 / t3001
KL 8.6e-12 / 4.5e-9 / 5.8e-5, top-1 1.0 / 1.0 / 0.999: PASS.
Full model: `scripts/dsv41/full_dump.sh` (3 B200 x 125 GB, ~45 min, window requested from main) dumps the run-of-record
pkg/BF16 GGUF on C's fx_base token files + real-text wiki1100 / long17k, for the TP8 MATCHED oracle.
First full-model results (GPUs 0,2,3, ub 512, fa=0, F32 KV; load + 6 prompts in ~10 min): all finite; real-text
self-perplexity wiki1100 2.11 (next-token top-1 0.826), long17k 2.0 / 1.28 / 1.80 / 2.08 / 3.03 on [0,1k) / [1k,4k) /
[4k,8k) / [8k,16384) / [16384,17k); 256-token bands before the pool threshold already range 1.6..3.0 (text-dependent, no
jump at 16,392). All phases done (outputs/dsv41_full/bf16_ub512/{ub512,ub7,ub1,fa1_ub512}), all finite; self-ppl
wiki1100 2.105 / 2.102 / 2.103 / 2.086 and long17k 1.862 / 1.862 / - / 1.858 for ub512 / ub7 / ub1 / fa1.
CUDA batch-split is not bitwise on the full model: ub7 vs ub512 KL mean 9.6e-3 (wiki1100), 3.4e-3 (long17k), 6e-2 on
the random-token t1025; fa1 vs ub512 7.4e-3 / 3.2e-3. Attribution: the full model on CPU (ngl 0, 12 wiki tokens,
~5 min) is bitwise identical at ub 512 and ub 7, and CPU vs each CUDA arm shows the same per-position profile (KL
3e-6 at t0, 0.08-0.2 at t5-t6) whichever ubatch / FA is used: those positions are numerically sensitive and CUDA's
kernels (MMQ / MMVQ, q8_1 activations) differ from CPU there. Released V4-Flash (UD-IQ1_S) on the same text: ub7 vs
ub512 KL mean 3.4e-2, 0.4 at t6. So the graph is batch-invariant; the CUDA spread is kernel numerics. Which side is
closer to the truth is for C's full-model MATCHED oracle (prompts: outputs/dsv41_full/bf16_ub512/tokens).

## Critique fold-in (plans/dsv41_support.md "Critique fold-in")
1. No per-head q RMS norm: the graph has none; "V4 per-head q norm" is seeded bug #6 and FAILS (KL 4e-3..2).
2. MATCHED oracle: the fixture ref/ IS the matched reference (shim act_quant / fp4_act_quant = identity); every gate above is
   against it. NATIVE (in-graph FP8/FP4 activation quant emulated on CPU, `scripts/dsv41/native_gap.py`) vs MATCHED:
   small KL 0.24..0.60, real dims 0.69..1.47 (random weights are chaotic); llama vs MATCHED 1.7e-11..2.7e-11 on the same
   prompts. The gap belongs to the oracle, not to the port; on the real model gate against MATCHED (KL < 1e-3 + p99).
3. Harness uses full logits: `ref_forward` asserts `[1, T, vocab]` (ParallelHead full_logits path), dumps are [T, V].
4. Epsilons kept separate: norms 1e-20 (as `RMSNorm.eps` in model.py), hc_eps 1e-6, router weight sum +1e-20 (2f4055f00;
   V4 keeps its 6.1e-5 clamp); norm statistics in F32.
5. Boundaries, `tiny_fixture_budget` (real budgets on small dims: window 128, index top-k 512, candidate pool 2048 x 8 =
   16384 rows, YaRN 65536 x 16; one 17,000-token prompt, so every t is its own boundary: window 127/128/129, top-k
   511/512/513 at both ratios, pool 16383/16384/16385; 616 rows with real exclusion, up to 77 blocks excluded):
   - candidate invariants (reference): newest partial block kept on every row, kept-block count = min(visible, 2048) on
     every row, real exclusion present: PASS.
   - CPU llama.cpp, single shot / ub 4096 / ub 1025 (odd ratio-2 groups split across ubatches): logits and every
     selected set bitwise identical across the three splits.
   - vs the fp32 reference: top-1 99.75%, KL median ~1e-7, mean 3.0e-4 (p99 3.9e-4, max 0.42), i.e. heavy-tailed, and 0
     below t=1100. Selection: L2 (first index layer) differs on 87 / 17000 rows, every one a single swap at rank 512/513;
     llama-vs-reference index-score error (max over a row / row top score) median 1.3e-5, p99 4.7e-5; the reference's own
     gap for each swapped pair is <= 2.4e-5 (median 2.3e-6), below the error p99 on 87 / 87 rows. Later layers inherit
     the flips (L4 349, L6 1291, L8 2422 rows; picks outside the reference pool only on rows whose pool itself flipped).
     The score error grows linearly with position (median 1.2e-6 at t < 1100 -> 2.5e-5 at t > 12000): float32 rope
     angle rounding (|angle| up to ~1.7e4 rad, ~2e-4 rad rounding at t = 16000 on both sides, rounded differently:
     ggml multiplies theta by theta_scale per pair, model.py multiplies t by a rounded freq).
     Attribution (`scripts/dsv41/ref_precision_floor.py`, reference variants, same weights):
     | reference variant | KL(variant ‖ ref32) | L2 / L4 / L6 / L8 rows differing from ref32 | llama vs variant: KL, L2 / L4 / L6 / L8 |
     |---|---|---|---|
     | float64 upcasts (fp32 angles) | 1.6e-4 | 8 / 53 / 616 / 1267 | 4.0e-4, 93 / 388 / 1721 / 3154 |
     | exact (float64) rope angles | 6.6e-4 | 139 / 623 / 2519 / 4200 | 7.5e-4, 150 / 687 / 2781 / 4675 |
     | ggml-rounded rope angles | 3.7e-4 | 87 / 338 / 1371 / 2648 | 1.1e-4, 8 / 40 / 274 / 701 |
     With the reference rounding its rope angles the way ggml does, llama.cpp's L2 differences fall from 87 to 8 rows,
     the same count as the reference's own fp32-vs-fp64 noise; with exact angles the fp32 reference is itself further
     from the truth than llama.cpp is from it. Classified as precision (rope-angle rounding + fp32 arithmetic), no defect.
   - CUDA (`GGML_CUDA_F32_EXACT=1`): same shape, more flips (L2 1539 rows, 97% single swaps; the >= 3-swap rows all sit
     in the 128-token window after one L0 router near-tie at t=16017). KL mean 3.2e-3, top-1 98.2%; decode (ub 1, 17,000
     steps) gives the same numbers (KL 3.1e-3). CUDA rope rounds theta differently again (`pos*powf(theta_scale, i)`).
   The 1e-6 F32 gate holds wherever the top-k does not bind on a near-tie (all of small / real-dims fixtures, and
   t < 1100 here); at the real budgets the right gate is selection parity with gap evidence, as above.
6. Slot reuse / prefix restore (`scripts/dsv41/state_test.cpp`): save seq 0 after n1 tokens, clear memory, restore into
   seq 0 or 1, continue: PASS (n1 201 / 17 / 164). Real defect found and fixed here: the Engram n-gram token history was
   not in the saved state (2f4055f00, `has_cell_ext()`); negative control with the fix off FAILS (KL 0.115).
   Compressor state rides the existing DSV4 snapshot path.
7. V4-Flash regression on the real model: unsloth/DeepSeek-V4-Flash-0731-GGUF UD-IQ1_S (77 GB, downloaded for this
   check), CUDA, 600 tokens, F32 KV, FA off: logits bitwise identical base (fd55cb98f) vs head at ub 512 and ub 7; head
   A/A also bitwise. Plus the random V4 model (bitwise, ub 256 / 7).

## Flash attention (fa=1) and multi-sequence / rollback (coordinator items 1, 2)
- FA graph is exact: with K/V kept F32 inside the FA op (debug-only patch, not committed) fa=1 matches the reference at
  KL 5.0e-12 / 5.0e-12 / 4.6e-12 (short / wiki330 / rep300): mask concat (window + top-k -inf), attn_sinks, padding and
  the compressed-row concat are right. FA accumulation is already GGML_PREC_F32 for every arch (llama-graph.cpp).
- The fa=1 excess is the F16 kernel: llama vs the reference with q/k/v rounded to F16 is still 1.9e-2 on short16 (floor
  7.9e-7, error from position 1 on, attn_raw-0 rel err 3e-4..2e-3 per row). The CPU single-chunk kernel also rounds Q to
  F16 and accumulates V*softmax in an F16 VKQ16 buffer, which input rounding alone does not model; the small fixture
  then amplifies it (routing flips). Not a V4.1 defect.
- What users see, on agent C's non-chaotic fx_base (F16 KV, fa=1, vs MATCHED): CPU KL 6.5e-8 (t129), 4.1e-5 (t1027),
  1.2e-4 (c1600), 3.4e-4 (t3001), top-1 >= 0.992; CUDA (default precision) 4.0e-4 / 3.0e-4 / 4.4e-4 / 9.3e-4, top-1
  >= 0.983. All < 1e-3 and 5-80x below C's bf16 floor (5.1e-3).
- Multi-sequence (`scripts/dsv41/seq_test.cpp multi`, n_seq_max 2, every ubatch interleaves wiki330 on seq 0 and rep300
  on seq 1, ub 7 and 64): each sequence vs its own single-sequence reference KL 4.6e-12..4.9e-12 (CPU), 1.6e-11..2.8e-11
  (CUDA exact), top-1 1.0.
- Rollback snapshots (`seq_test rollback`, n_rs_seq R; after each true chunk decode a draft whose tail is garbage tokens,
  seq_rm the rejected tail, continue; odd chunk sizes so ratio-2 groups straddle the rollback point; garbage passes
  through Engram history, compressor state and KV): R4/D4/acc1 (23 rollbacks), R4/D4/acc3, R2/D5/acc3: KL 4.9e-12 CPU,
  2.8e-11 CUDA, top-1 1.0. Negative control (keep one rejected token): KL 1.9, so the gate sees pollution.
- Slot reuse / prefix restore: state_test above (PASS).

## Milestones
| # | state | commit |
|---|---|---|
| 1 gguf-py constants, arch enums, keys, TENSORS.md | done | 7976d362d |
| 2 tiny fixture from official model.py + reference logits + fixture-only converter | done | scripts in `$W/scripts/dsv41/` |
| 3 graph + KV-cache generalisation, CPU + CUDA parity | done | bae963fb2 |
| 4 test-llama-archs (CPU, CUDA, Meta split), test-backend-ops, V4 regression | done | d29889d8f |
| 5 full-model handoff (below) | written, waiting for B's GGUF | |
| 6 DSpark / MTP graph | done (DFLASH draft) | 1c9a39a25, see "DSpark" below |

## What the graph implements (`src/models/deepseek41.cpp`)
Engram (host n-gram hash from the raw-cache token history, CPU get_rows on `engram_embd`, gate as model.py), single-pass mHC
(no hc_head), ratio-1/2 compressors (V4 HCA-style state, no ape/overlap), index K = k_norm(wk(latent)) stored next to the
compressed KV in one cache row, compressed KV / index K read from the kv source layer, top-k computed on index sources and
reused in between, two-level candidate blocks (pool_1d max per block, pin newest block, top blocks, mask later index layers),
no per-head q norm. KV cache: `llama-kv-cache-dsv4` slots are ratio-parametrised (V4: CSA 4 overlap + LID, HCA 128;
V4.1: CSA slot = ratio 2, HCA slot = ratio 1, only kv-source layers own rows, no LID).

## Tiny fixtures (`$W/models/dsv41/tiny_fixture` small dims; `tiny_fixture_realdim` + `tiny_gguf/realdim-f32.gguf` real dims, 24.6 GB F32). Small fixture (`$W/models/dsv41/tiny_fixture`, GGUF `$W/models/dsv41/tiny_gguf/tiny-f32.gguf`)
10 layers, dim 128, real tokenizer + real 99,092-id Engram token map, Engram on L1/L4, ratios [0,0,2,2,2,2,1,1,1,1],
kv sources 2/4/6, index sources 2/4/6/8, candidate source 6 (4 blocks x 4), index_topk 8, window 16, YaRN orig 64.
Checkpoint in official naming and storage formats (FP8 32x32 ue8m0, FP4 experts, Engram FP8 + per-row-32 scales).
Reference = official model.py on CPU fp32 with activation fake-quant off (`scripts/dsv41/shim/kernel.py`).
Regenerate: `python scripts/dsv41/make_fixture.py`; convert: `python scripts/dsv41/fixture_to_gguf.py --out X.gguf [--experts mxfp4]`;
sweep: `bash scripts/dsv41/run_parity.sh <build> <gguf> <outdir> <ngl> "fa:ub ..."`; ablations: `python scripts/dsv41/ablate.py --got <dump>`.

## Fixture parity (KL(ref||llama) mean / max, top-1; prompts wiki330, rep300, short16)
- CPU F32, fa=0, ub 512 / 7 / 1: KL mean <= 4.9e-12, max <= 1.5e-11, top-1 100% on all three (ub 7 differs from ub 512 by 5.7e-6 abs, still passes).
- CUDA (B200) F32, fa=0, `GGML_CUDA_F32_EXACT=1 NVIDIA_TF32_OVERRIDE=0`, ub 512 / 7 / 1: KL mean <= 2.8e-11, max <= 3.0e-10, top-1 100%.
- CUDA default precision fails the 1e-6 gate (F32 GEMMs run on TF32: hc_mixes rel err 9e-4 at layer 0); with the knob the
  per-tensor CPU-vs-CUDA error is <= 1.3e-6 everywhere.
- fa=1 (CPU and CUDA): F16 flash attention. short16 KL 1e-6..2e-6; long prompts 4e-3..1e-1, the same scale as the reference
  itself with q/k/v rounded to F16 (4.5e-3 wiki, 1.7e-2 rep300): the random fixture's top-k / routing flips amplify F16 noise.
- Mechanisms are live: KL(ref || ref without X) = Engram 3.1, candidate mask 0.17, top-k reuse (dense instead) 1.2,
  single-pass mHC (standard instead) 0.6; llama.cpp matches the full reference, not any ablation.
- MXFP4 experts (bit-exact repack verified): the 2.8e-4 short16 figure was from an older fixture and is stale. Current
  small fixture: short 2.0e-2, wiki330 1.4e-1 (chaotic fixture). On C's fx_base (B's MXFP4-expert -F32 GGUF, CPU) vs
  MATCHED: t129 3.2e-4, t1027 9.1e-4, t3001 2.1e-3 (top-1 0.965), vs the F32-expert copy at <= 2.3e-5: the CPU MXFP4
  matmul quantizes activations to q8_0 (C localised it: with q8_0 applied in the reference the expert outputs agree to
  5e-8). Kernel precision shared by every MXFP4 model, below the bf16 floor (5e-3), not a port defect.

## Other checks
- test-llama-archs deepseek41: CPU OK, CUDA NMSE 3.3e-6, Meta (tensor split) OK, roundtrip OK. deepseek4: still OK.
- V4 regression: random V4 model (ratios 4/128, 240 tokens) logits bitwise identical base vs head build, ub 256 and 7.
- test-backend-ops CUDA0: POOL_1D, TOP_K, SET_ROWS, CONCAT, FILL, GET_ROWS, SGN, ROPE, SOFT_MAX, SUM_ROWS all pass;
  MUL_MAT type_a=f32 with GGML_CUDA_F32_EXACT=1 275/275.

## For builder B
- Keys/tensors exactly as `TENSORS.md`. The loader derives sharing from `attention.kv_source_layers` /
  `attention.indexer.source_layers` and validates ratios; Engram offsets are derived from `engram.primes`.
- `engram_embd` is placed like token_embd (host buffer); any type works if CPU `get_rows` supports it (MXFP8 must
  implement CPU get_rows -> F32). Do not repack it. `engram_gate` must be F32 `q_weight*k_weight`.
- DSpark blocks may be written as blk.40-42 (loaded as optional, unused) or left out.
- Smoke test for a converted fixture: `python scripts/dsv41/fixture_to_gguf.py` is the reference converter; B's output on
  `$W/models/dsv41/tiny_fixture` should give the same parity numbers (compare with `run_parity.sh`).

## Full-model parity run (once B's reference GGUF exists)
Sizes: experts 289 GB (MXFP4), dense ~14 GB BF16 (FP8 dequant), embed+head 2.7 GB, Engram 203 GB (MXFP8, host).
1. Engram tables stay on the host automatically (input-layer placement); host RAM 2 TB is enough. mmap first-touch is
   random reads (48 rows per token): pre-warm the file (`vmtouch -t` or one `cat > /dev/null`) before timing.
2. GPU: ~306 GB weights -> 2 B200 with ~50 GB headroom (3 to be safe). Lease all cards at once:
   `python $W/gpu_pool_lease.py take --owner $$ --key $$:dsv41 --gpus A,B --gb 178 --est-s 3600`, run with
   `CUDA_VISIBLE_DEVICES=A,B`, then `gpu_pool_lease.py drop --key $$:dsv41 --gpus A,B`.
3. Logits: `$W/wt_dsv41/build-cuda/bin/llama_dump_logits -m DSV41.gguf -o OUT -c 8192 -ub 512 -ngl 99 -fa 0 --kv-f32
   prompts/*.tokens.bin` (the Phase-1 oracle writes the same int32 token files); compare with
   `python $W/scripts/dsv41/compare.py --ref ORACLE --got OUT --kl-max 1e-3`. Repeat with `-ub 1` and `-ub 7`
   (batch-split invariance) and once with fa=1 (expected higher KL, F16 attention).
4. Long-context gate: a prompt > 16,384 + 8 tokens (candidate pool) and > 512 compressed rows per ratio, so the
   candidate mask, top-k and reuse all bind.
5. Oracle caveat: model.py on GPU quantises activations (FP8 before every FP8/FP4 GEMM, FP8 window KV, FP4 index/compressed
   KV); the GGUF port does not, so expect KL above the fixture's 1e-11 even when correct. Measure the oracle with and without
   fake-quant on a few prompts (`DSV41_FAKE_QUANT=1` in the shim emulates it on CPU).

## Image-span Engram (embedding inputs)
model.py: image tokens are hashed DEAD (no n-gram spans them) and the Engram gate is 0 there; routing uses bias_vl.
llama.cpp: embd tokens hash DEAD, KV cells of embd tokens record LLAMA_TOKEN_NULL (dsv41_tok_hist), and embd ubatches
get an `engram_keep` input (0 for embd tokens) multiplied into the gate; text-only graphs are unchanged (no extra op).
Test (`scripts/dsv41/image_ref.py` + `embd_test.cpp`, small fixture, 48 wiki tokens, [16,24) fed as an embedding
ubatch of the same token rows, reference with token_types image mask and bias_vl): KL 4.4e-12 CPU, 5.2e-12 CUDA,
top-1 1.0; without the gate fix KL 0.97; image vs text reference differ by KL 1.0 (the test can fail). Text parity
unchanged (4.6e-12..4.9e-12 at ub 7), test-llama-archs CPU 137/137.

## Engram placement guard
`llama-model-loader.cpp`: an `engram_embd` placement on a non-host buffer (from `-ot`, or any split mode) is refused with a
warning and the table goes to the CPU buffer. Test `scripts/dsv41/engram_placement.cpp` (all layers offloaded,
override engram_embd -> CUDA0): with the guard, a warning per table, CUDA0 model buffer 78.90 MiB (= no override) and
logits bitwise equal to the run without the override; negative control (build without the guard): CUDA0 buffer
80.53 MiB, i.e. the tables were uploaded.

## Not proven / open
- A5 (pin PR / pinset sim) not started; full-model parity waits for B's GGUF and the oracle.
- The F32 KL gate cannot be met at the real top-k budgets on any implementation (the fp32 reference misses its own
  float64-rope version by KL 6.6e-4); long-context acceptance = selection parity with near-tie gap evidence.
- Vision tower not ported (the text side of image spans is done, see below).
- No ubatch shares a token across sequences (asserts, as the PLE path).
- DSpark drafting graph not implemented.

## A2 additions (2026-10-17)
### CPU full model vs official MATCHED (`scripts/dsv41/score_arms.py <prompt>`, dumps `outputs/dsv41_full/cpu_wiki1100`)
CPU ngl 0, ub 512, F32 KV, fa 0, 96 threads (`scripts/dsv41/dump_logits_mt.cpp`, DUMP_THREADS), ~220 s per prompt.
KL(matched || arm) mean: wiki1100 CPU 6.21e-3, CUDA ub512 6.11e-3, ub7 9.9e-3, ub1 6.2e-3, fa1 6.3e-3, native 2.6e-2;
at the 30 CPU-vs-CUDA sensitive positions (KL > 0.05) CPU 7.4e-2 vs CUDA 8.1e-2, CPU closer on 60%: no systematic
winner. t1025 (random tokens): CPU 3.95e-2, CUDA 2.26e-2, native 9.4e-2; CPU is already off at t0 (KL 0.84 vs CUDA
0.058), i.e. per-token kernel numerics amplified by a chaotic input. Both arms sit ~6x over 1e-3, at C's bf16 floor.

### A5 (pin PR = upstream + this change; local only)
Branch `dsv41-pin` = dsv41 minus STATUS_A.md / TENSORS.md / the GGML_CUDA_F32_EXACT debug knob, plus
`git merge --no-ff b11707` ("Fix merge conflicts": test-llama-archs gating switch, upstream added MELLUM, DEEPSEEK41 added
to the DEEPSEEK4 case), plus later dsv41 merges. Upstream #30670 (Beam) added a jinja object builtin `list` that shadows
the template's `namespace(list=...)` attribute ("Function is not a bool value" in test-chat): fixed in the template
(attribute renamed to msgs, cd1714ad9; 5/5 encoding.py parity re-verified; `models/dsv41/chat_template.jinja` updated,
old copy kept as `.pre_msgs_rename`). GGUFs built before this embed the old template and break on the nightly (which
pins #30670). Converter template path fixed (`deepseek-ai-DeepSeek-V4.1.jinja` was deleted in 47dd9807b, so the
converter silently kept the V4 template).
b11714 + pin (CPU): test-llama-archs 140/140, test-chat all pass, DSpark gate PASS (below).
`llama_pinset.py sim --tags window` with the pin appended (placeholder `danielhanchen#99999`, `temp/dsv41_pinset/pr-set.json`):
REFUSED on every tag b11707..b11714 on `ggml/include/ggml.h` + `gguf-py/gguf/constants.py` against unslothai#61
(IQ1_XS/XXS/XXXS at 64..66, GGML_TYPE_COUNT 67) vs MXFP8 = 43 / COUNT 44: both pins edit the GGML_TYPE_COUNT line, not
additive. Needs a decision (fold MXFP8 into the pinset next to #61, or order + a resolved type list). Baseline without
the pin already FAILs: ggml-org#30670 REFUSED (common/parsers/*, + llama-arch/model on b11714), pre-existing.

### DSpark (MTP) draft
DeepSeek-V4.1 DSpark = upstream's DFLASH DSpark draft (V4 `graph_dsv4`, driver `--spec-type draft-dspark`) with
`dflash.hyper_connection.single_pass = true` selecting `llama_model_dflash::graph_dsv41`: single-pass mHC (identity copy 0
for stage 0, FFN uses the attention pre-mix), no hc_head (collapse with the last FFN pre-mix), no per-head q norm, plain
rope; target_layers = the V4.1 ids (attention INPUT of layers 37-39, after Engram, as model.py). Exact ring window: every
block query sees the 128-position main-KV ring plus the block (get_dspark_topk_idxs); the graph gives all block queries
the mask row of the first one and n_swa = window + 1. Converter: `convert_hf_to_gguf.py <V4.1 dir> --dspark` ->
`DeepseekV41DSparkModel` (fixture output value-identical, 78/78 tensors, to `scripts/dsv41/dspark_fixture_gguf.py`).
Reference: `scripts/dsv41/dspark_ref.py` (C's ref_run.build matched, forward_spec at temperature 0, fx_base t700, draft
points s = 64 / 127 / 128 / 300 / 699) -> `outputs/dsv41_dspark/ref`. Gate `scripts/dsv41/dspark_gate.py` + harness
`dspark_test.cpp` (target prefill with layer-input extraction, embd injection, [anchor, noise x4] block, logits incl.
markov bias, sigmoid confidence):
| build | KL mean (all points) | chain top-1 | conf abs err |
|---|---|---|---|
| CPU F32 | 9.6e-13 .. 2.3e-11 | 25/25 | <= 6.5e-7 |
| CUDA F32 exact | 1.0e-12 .. 4.3e-12 | 25/25 | <= 5.2e-7 |
| b11714 + pin, CPU | same as CPU | 25/25 | |
| converter GGUF (MXFP4 experts), CPU | 2.8e-5 .. 3.0e-5 (q8_0 activations) | 10/10 (s64, s300) | 1.2e-3 |
Negative control: before the block-window mask (standard SWA 128) s >= 127 FAILED (KL 3e-3 .. 5e-3, chain 3/5 .. 4/5).
Server loop (`scripts/dsv41/dspark_server_smoke.sh`, llama-server, greedy 300 tokens): with F32 KV + fa off, output with
the draft is identical to no draft (881 drafted, 4 accepted: random fixture weights). With server defaults (F16 KV,
FA auto) both draft-dspark and draft-simple (self-draft, 97% accepted) diverge at the same token, a 0.009-logit near-tie:
F16 attention numerics of the batched verify. Target rollback via the API (seq_test rollback on fx_base, D 4, R 3,
acc 1..4) is exact: max abs logit diff 9.5e-6 vs one-shot.
Not proven: real-model DSpark acceptance (needs the 7.9 GB mtp.* converted + a GPU window); multi-sequence DSpark
blocks; KV eviction of ring cells in a long running loop with swa_full off (the smoke ran 300 tokens past the window).

### MXFP8 renumber to fork-local 67 (main's decision, da47a97a9)
GGML_TYPE_MXFP8 = 67, GGML_TYPE_COUNT = 68 (42..63 reserved for upstream, as #61). No other literal 43 / COUNT-sized
table depends on it (type_traits designated initialisers; type loops already skip NULL names / blck_size 0).
New build rejects old (43) files and old builds reject 67 files (`invalid ggml type`), both checked.
Header patcher `scripts/dsv41/gguf_header_patch.py` (dry run default; --apply backs up header bytes; pwrite inside the
header only): type 43 -> 67 for `blk.N.engram_embd.weight` (2-D, ne0 % 32, size checked), template list -> msgs
(same length; patched template == the repo jinja byte for byte). On copies (temp/dsv41_patch_test): fx_base, fx_engram0,
pkg shard-1 copy, pkg shard-8 header + 256 MiB: data sha256 unchanged; cmp vs original = 126 bytes, all below
data_start. Dry run on pkg/BF16 (read only): shard 1 template, shards 8 and 9 one Engram table each, mtp-*.gguf template.
NOT applied to pkg/* (main applies after D's grid). verify_all.py accepts type 67 (or 43) for Engram MXFP8.
Fixture gate on the patched fx_base with the new build: unchanged (t129 9.2e-12, t3001 2.30e-5); DSpark gate unchanged;
CUDA test-backend-ops GET_ROWS 225/225 (MXFP8 incl.).
Pin: `dsv41-pin` (f8cfdaedb) = upstream + change (MXFP8 67) still REFUSED after #61 (both edit the COUNT line).
`dsv41-pin-after61` (71752e6e0) = dsv41-pin + #61 merged, COUNT pre-resolved to 68: merges clean on b11707..b11714;
pin_contract FAILs only unslothai#61 (`GGML_TYPE_COUNT = 67` not kept: replaced by 68) -> needs the fold (#241/#253
precedent). Mix tree b11714 + pins (temp/dsv41_mix_b11714): test-llama-archs 142/142, test-chat pass, fx_base gate PASS
(t3001 2.44e-5), DSpark gate PASS. ggml-org#30670 REFUSED on every tag before us: its squash 01fc3aae8 is already in
the aged base b11707, so the pin re-applies upstream code (pr-set doc: delete the entry once a base tag carries it).
Real DSpark draft: models/dsv41/gguf/dspark/DeepSeek-V4.1-Flash-1017-DSpark-DFLASH-BF16.gguf (8.5 GB, 78 tensors,
target_layers 37/38/39, 128 experts top-3, 13 s). Acceptance run: scripts/dsv41/dspark_accept_real.sh with the MXFP8=43
build temp/wt_dsv41_43 (7a582830e) because pkg is not patched yet; gpu_queue --gb 36 (--cpu-moe target).
