# Exact concurrency experiment p

Opt in before loading the model with `LLAMA_EXACT_CONCURRENCY=1`. This also forces
`GGML_CUDA_BATCH_INVARIANT=2` and gives `GGML_CUDA_BATCH_INVARIANT_MAX_COLS` a
default. The column policy only has to cover the widest ubatch a decode step can
build, one column per slot times one plus the draft length, because a prompt
ubatch is kept to one sequence and gets its exactness from that instead. Tools
built on `common` report that width, so the default is `--parallel` times one plus
`--spec-draft-n-max`, and an explicitly set `GGML_CUDA_BATCH_INVARIANT_MAX_COLS`
smaller than it is refused at startup. Nothing reported a width, the default is 16
and the dispatcher warns once the first time a `MUL_MAT` or `MUL_MAT_ID` above the
bound is left unsplit. Set the variable to `0` for no bound; above the bound the
column policy does not fire, including during prefill.

The experimental policy supports unified, offloaded F16 K/V, causal flash attention,
256-dimensional K and V heads, no attention soft cap, and no sliding window.
Shared-weight matmuls over multiple sequence planes are normalized to one plane
before the inherited selective column dispatcher. Without this, the recurrent
output projection bypasses batch invariance during concurrent prefill.
It is measured on text prompts with Qwen3.5-4B on one B200. Every KV layer has to
be on the CUDA backend, since no other backend reads the page table; a partial or
absent offload fails the load naming the layer. The V-less attention layouts have
no page table either, and a model on one of those is refused at context creation.
Context shifting, position division, cross-sequence prefix copies, shared-prefix
input tokens, and whole-context state loading are unsupported. Per-sequence state
save and restore is supported. Unsupported cache transformations are refused with
a logged error and leave the cells untouched; `--context-shift` and
`--cache-reuse` are reported as unsupported at load and disabled there.

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

A concurrent request that fails now fails the run instead of being dropped from
the result, and every run records the environment the server actually inherited,
including `LLAMA_EXACT_CONCURRENCY`, under `env` in the JSON and in the server log
header, so a run labelled as the mode-off reference can be checked rather than
trusted. Teardown is POSIX only.
