# Exact concurrency experiment p

Opt in before loading the model with `LLAMA_EXACT_CONCURRENCY=1`. This also forces
`GGML_CUDA_BATCH_INVARIANT=2` with no column limit, including during prefill.

The experimental policy supports unified, offloaded F16 K/V, causal flash attention,
256-dimensional K and V heads, no attention soft cap, and no sliding window.
Shared-weight matmuls over multiple sequence planes are normalized to one plane
before the inherited selective column dispatcher. Without this, the recurrent
output projection bypasses batch invariance during concurrent prefill.
It is measured on text prompts with Qwen3.5-4B on one B200. Context shifting,
position division, cross-sequence prefix copies, shared-prefix input tokens, and
whole-context state loading are unsupported. Per-sequence state save and restore
is supported. Unsupported cache transformations assert instead of silently
violating the page invariant.

The allocator owns pages of 256 cells on behalf of one (sequence, position/256).
Position modulo 256 fixes the cell offset. Empty pages remain in the unified pool
and can be allocated by any sequence. The metadata is derived from live cells so
allocation rollback, tail removal, and sequence removal do not need another
transaction log. Restoring a sequence allocates free pages from this same pool.
The cost is up to 255 reserved cells per active sequence tail, plus holes introduced
by partial range removal.

Attention receives an I32 page table in source 5: `[count, physical page IDs...]`
for each query, sorted by logical position. The physical K/V view and mask span
the pool, but the attention loop only visits the query's logical pages. The
final page is padded to 256 cells using the existing causal mask. Wholly future
pages in a prefill ubatch are excluded from the query's table.

The vector attention specialization runs one query per block, four warps, and
`parallel_blocks=1`. It reads K/V directly from the physical pages, with two
128-cell softmax iterations per page, in logical page order. There is no K/V
gather and no split-K combine. The default vector specialization has no page
lookup. The ordinary path allocates no page metadata and launches no extra kernels.

`FATTN_KQ_STRIDE=256` is a mask-scan stride, not the actual MMA rescaling tile.
For K/V head size 256, the Ampere-or-newer MMA configurations use 64 KV rows at
8 query/head columns and 32 KV rows at 16/32/64 columns. The retained vector path
has a 128-cell iteration. Both divide the 256-cell placement page.

The probe is adapted from the existing batch-invariant harness and rejects
nonfinite logits and attention. `PROBE_B_REVERSE=1` fills neighbours before P0;
`PROBE_RESTORE=1` parks P0, releases a neighbour, restores P0, then rebuilds the
neighbour. Compute rows can be compared across this relocation; physical cache
views and index tensors must not be mistaken for sequence-0 compute outputs.

`divergence.py --reference FILE` compares against an existing unparked solo token
reference. `bench.py --modes 0,1 --pairs 3` measures default off against exact mode
on, with 256 predicted tokens. Set `UNSLOTH_WORKSPACE` to the model parent workspace
and `LD_LIBRARY_PATH` to this build's bin directory. The harness uses GPU 3;
select a port in 9601-9610 explicitly.
