# Builder A status (DeepSeek-V4.1-Flash-1017 llama.cpp port)

Branch `dsv41` in `$W/wt_dsv41` (off llama.cpp_pinset fd55cb98f). Contract for builder B: `TENSORS.md`.

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

## Tiny fixture (`$W/models/dsv41/tiny_fixture`, GGUF `$W/models/dsv41/tiny_gguf/tiny-f32.gguf`)
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
- Image tokens: Engram DEAD / gate-off for embd inputs is hashed as DEAD but the gate is not forced to 0 yet; vision not ported.
- Multi-sequence batches (n_seq_max > 1) and rollback snapshots (n_rs_seq > 0) are wired like V4 but not parity-tested.
- No ubatch shares a token across sequences (asserts, as the PLE path).
- DSpark drafting graph not implemented.
