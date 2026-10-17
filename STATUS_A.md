# Builder A status (DeepSeek-V4.1-Flash-1017 llama.cpp port)

Branch `dsv41` in `$W/wt_dsv41` (off llama.cpp_pinset fd55cb98f). Contract for builder B: `TENSORS.md`.

## Porting workflow (unslothai/scripts #649 bundle, `temp/us_gguf`: gguf_build_workflow.md A1-A5, PORTING.md)
| step | state | evidence |
|---|---|---|
| A1 recon + reuse map + fixture + converter + graph | done | reuse map in deepseek41.cpp header (V4 HC/MLA/MoE/raw cache reused; new: Engram, lagged mHC, ratio 1/2 + sharing + candidates). Two fixtures: `tiny_fixture` (small dims, fast, seeded-bug harness) and `tiny_fixture_realdim` (A1 rule: REAL dims 5120 / 64x512 heads / rope 64 / q 1280 / o 8x1024 / expert 2304 / index 32x128 / Engram 8x256 n-gram 4 / window 128 / block 8 / YaRN 65536x16; only counts shrunk: 10 layers, 8 experts top-2, Engram bucket start 1009, index top-k 64, 16 candidate blocks). Rare-path thresholds: window 128, ratio-2 top-k binds > 128 tokens, candidate mask binds > 128 ratio-1 rows; prompt wiki1100 exceeds all. Reference = official model.py (no transformers implementation of deepseek_v41 exists in 4.57 / 5.x, so ref_logits.py --generic cannot load it); the fixture writes the ref_logits.py npz schema + full logits. Fixture generates under the reference (prefill vs decode max abs 2.7e-5 at real dims). |
| A2 parity (bundled tools/dump_logits + port/compare_ref.py, F32 gate --max-kl-mean 1e-6 --min-top1 1.0 --min-cos 0.99999) | done (CPU), CUDA calibrated | see "A2 numbers" below; 13/13 seeded bugs FAIL on the small fixture, 5/5 on real dims |
| A3 template + V4.1 DSML tool parser | done (1d51c092f) | template `models/templates/deepseek-ai-DeepSeek-V4.1-Flash-1017.jinja` (= `$W/models/dsv41/chat_template.jinja`, 5/5 byte parity with encoding.py). V4.1 tags carry a leading space (`<｜DSML｜ calls>`, `<｜DSML｜ invoke>`, `<｜DSML｜ parameter>`); detection in common/chat.cpp keys on `｜DSML｜ calls>` (the template never names `dsml_token`), parser = the V4 parser with the spaced tags. test-chat: the V4 parser suite ported to V4.1 markup (content, thinking, single / parallel / mixed-type calls, content before calls, continuation, reconstruction) all pass; negative control (V4.1 tag switch off) aborts on the V4.1 block. |
| A4 never break other models | done | test-chat all passed; test-llama-archs CPU all 137 passed; CUDA: full suite exceeded its 2 h box (killed, no verdict), so the archs sharing touched code were run: deepseek4/41, deepseek2/32, glm-dsa, glm5-next, maple, qwen3moe, qwen35moe, qwen4exp on CUDA + CPU + Meta: 28/28 passed; deepseek41 on CPU / CUDA / Meta OK; test-backend-ops for used ops pass; V4 logits bitwise identical base vs head; no vocab change (vocab differential N/A) |
| A5 pin PR / pinset sim | not started (after A3 and B's GGUF) | |

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
near-tie regime characterised above. CUDA (`GGML_CUDA_F32_EXACT=1`, gate build `build-cuda`): fx_base t129 / t1027 / t3001
KL 8.6e-12 / 4.5e-9 / 5.8e-5, top-1 1.0 / 1.0 / 0.999: PASS.
Full model: `scripts/dsv41/full_dump.sh` (3 B200 x 125 GB, ~45 min, window requested from main) dumps the run-of-record
pkg/BF16 GGUF on C's fx_base token files + real-text wiki1100 / long17k, for the TP8 MATCHED oracle.

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

## Milestones
| # | state | commit |
|---|---|---|
| 1 gguf-py constants, arch enums, keys, TENSORS.md | done | 7976d362d |
| 2 tiny fixture from official model.py + reference logits + fixture-only converter | done | scripts in `$W/scripts/dsv41/` |
| 3 graph + KV-cache generalisation, CPU + CUDA parity | done | bae963fb2 |
| 4 test-llama-archs (CPU, CUDA, Meta split), test-backend-ops, V4 regression | done | d29889d8f |
| 5 full-model handoff (below) | written, waiting for B's GGUF | |
| 6 DSpark / MTP graph | not started (tensors load as optional, graph throws on the MTP context) | |

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
- MXFP4 experts (bit-exact repack verified): short16 KL 2.8e-4 from the Q8 activation quantisation of the MXFP4 matmul.

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

## Not proven / open
- A5 (pin PR / pinset sim) not started; full-model parity waits for B's GGUF and the oracle.
- The F32 KL gate cannot be met at the real top-k budgets on any implementation (the fp32 reference misses its own
  float64-rope version by KL 6.6e-4); long-context acceptance = selection parity with near-tie gap evidence.
- Image tokens: Engram DEAD / gate-off for embd inputs is hashed as DEAD but the gate is not forced to 0 yet; vision not ported.
- Multi-sequence batches (n_seq_max > 1) and rollback snapshots (n_rs_seq > 0) are wired like V4 but not parity-tested.
- No ubatch shares a token across sequences (asserts, as the PLE path).
- DSpark drafting graph not implemented.
