# llama-speculative-tree

Tree-structured speculative decoding: the draft model proposes a **tree** of continuations instead
of a single chain, and the target model verifies the whole tree in one forward pass.

**Read the results section before using this.** With a separate draft model, tree drafting is
correct and lossless-equivalent to the existing linear draft, but it is **not faster** than a
well-tuned linear draft. This example exists to make the encoding reusable, to provide a self-test
for it, and to record the measurement so the experiment does not have to be repeated.

## Why no new kernel is needed

Tree attention normally needs a custom attention mask. It does not here. The mask predicate in
`llama_kv_cache::set_input_kq_mask` is membership AND causality:

```
keep(query i, cell j) = !cells.is_empty(j)
                     && cells.seq_has(j, ubatch->seq_id[i][0])
                     && p0 <= p1
```

and a KV cell can belong to many sequences at once (`llama-kv-cells.h`, `std::bitset<LLAMA_MAX_SEQ>`).
So a tree can be encoded directly:

- one `seq_id` per root-to-leaf path
- each node carries the seq_ids of every leaf in its subtree
- `pos = committed_prefix_len + depth`

A node then sees exactly its ancestors: they are on its paths and at a smaller position, while
siblings and cousins carry disjoint path sets. Using `seq_id[i][0]` is safe because all paths
through a node share the same ancestors.

`examples/lookahead` and `tools/perplexity` already rely on the same primitive.

## Requirements and limits

- **Unified KV cache is mandatory.** The example sets `kv_unified = true`; without it the sequential
  ubatch split rejects coupled sequences and `llama_decode` fails.
- **`LLAMA_MAX_SEQ` (256) caps leaves, not nodes.** A full top-3 depth-8 tree would need 6561 leaves,
  so trees are budgeted.
- **Hybrid and recurrent models are not supported** (qwen35 / Qwen3.5 / Qwen3.6, qwen3next,
  nemotron-h). Recurrent state cannot be branched by masking, so those architectures need
  parent-and-depth driven state kernels rather than this encoding.
- Not wired into the server, and not routed through `common/speculative.*` (which asserts
  `n_seq_id == 1`).

## Usage

Tree shape is set with environment variables so that no shared argument parsing has to change.

| variable | default | meaning |
|---|---|---|
| `LLAMA_TREE_TOP_K` | 2 | children proposed per frontier node |
| `LLAMA_TREE_DEPTH` | 8 | maximum tree depth; **0 disables drafting** (no-speculation control) |
| `LLAMA_TREE_MAX_NODES` | 16 | total node budget, admitted best-first by cumulative draft score |
| `LLAMA_TREE_AUDIT` | 0 | print the top-2 target logit margin for every generated token |
| `LLAMA_TREE_SELFTEST` | 0 | verify the tree encoding against per-path replay (see below) |
| `LLAMA_TREE_SELFTEST_STEPS` | 4 | how many steps to self-test |

```bash
LLAMA_TREE_TOP_K=4 LLAMA_TREE_DEPTH=2 LLAMA_TREE_MAX_NODES=12 \
./build/bin/llama-speculative-tree \
    -m  model.gguf \
    -md draft.gguf \
    -ngl 99 -ngld 99 -c 4096 --temp 0 --seed 1234 -n 128 \
    -p "Explain what a hash table is and when you would use one."
```

`LLAMA_TREE_TOP_K=1` degenerates to a linear draft and is the control arm.
`LLAMA_TREE_DEPTH=0` disables drafting entirely and is the no-speculation baseline, produced by the
same binary and the same code path.

## Verifying the encoding

`LLAMA_TREE_SELFTEST=1` replays every node's root-to-node path on its own in a scratch sequence and
diffs the resulting logits against what the tree batch produced:

```
SELFTEST nodes = 80, max|logit diff| = 6.6694e-01, argmax mismatches = 1
```

Interpret this against the `LLAMA_TREE_TOP_K=1` control, which is a plain chain with no branching at
all. Measured on Qwen2.5-1.5B-Instruct-Q8_0 with a Qwen2.5-0.5B-Instruct-Q8_0 draft, CUDA:

| arm | nodes | max &#124;logit diff&#124; |
|---|---|---|
| top_k 1 (chain, no branching) | 28 | 0.65 |
| top_k 2 | 80 | 0.67 |
| top_k 3 | 80 | 0.64 |
| top_k 4 | 80 | 0.77 |

The control shows the same deviation as the trees, because the probe replays at a different batch
shape. Tree attention adds no error beyond re-batching.

## Accuracy

Greedy speculative decoding is lossless in exact arithmetic, but CUDA kernels are not batch-shape
invariant, so neither tree nor linear speculative decoding reproduces batch-1 greedy output
bit-for-bit. Over 10 prompts the linear arm matched batch-1 on 2 of 10 and trees on 0 to 4 of 10;
where the linear arm diverged, every tree shape diverged at the same token index.

`LLAMA_TREE_AUDIT=1` shows why. At one divergence:

```
idx 19   batch-1: token 1558  margin 0.1047
idx 19   linear : token  374  margin 0.0094
idx 19   tree   : token  374  margin 0.0300
```

Typical margins elsewhere in the same run are 2 to 9. Divergences occur only where the top-2 logit
gap is around two orders of magnitude tighter than normal.

## Results

Qwen3-32B-Q4_K_M target, Qwen3-0.6B-Q8_0 draft, single B200, batch of one, 3 prompts by 6 repeats,
median:

| config | t/s | speedup | accepted/step | ms/token |
|---|---|---|---|---|
| top_k 1, depth 2, 3 nodes (linear) | 91.64 | 1.25x | 2.46 | 10.93 |
| top_k 4, depth 2, 12 nodes (tree) | 74.22 | 1.01x | 2.72 | 13.51 |
| depth 0 (no speculation) | 73.39 | 1.00x | 1.00 | 13.62 |
| top_k 2, depth 2, 6 nodes (tree) | 58.67 | 0.80x | 2.43 | 17.03 |

Ten shapes were swept and no tree configuration beat the best linear one. Trees do raise acceptance
(2.46 to roughly 2.7-3.1) but not enough to pay for what they cost. Per-phase timings show where it
goes: verifying 12 nodes costs 22.4 ms against 13.1 ms for a single token, and building a tree with a
separate draft model still needs one sequential draft forward per level, the same count as a linear
draft but with wider batches and more host-side work.

This is the structural reason the technique pays off elsewhere and not here. When the draft comes
from a multi-token-prediction head, all candidates are produced in effectively one pass, so the tree
costs almost nothing to build and only the wider verification remains. With a separate draft model
that amortisation is not available.

One measurement is unexplained: top_k 2 depth 2 with 6 nodes is reproducibly slower than top_k 4
depth 2 with 12 nodes, despite having half the nodes. It was rerun interleaved to rule out clock
drift and persists.

## When to revisit

Once MTP draft heads are available in GGUF for a non-hybrid architecture, the verification side of
this example is the part worth reusing, because the draft-build cost that sinks it here largely
disappears.
