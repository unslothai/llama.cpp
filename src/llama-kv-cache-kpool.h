#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;

//
// GLM-5-Next indexer pooling
//
// GLM's lightning indexer scores pools of `kpool` consecutive *positions*, while its
// top-k budget is counted in tokens (indexer_top_k), i.e. indexer_top_k/kpool whole
// pools. The reference requires indexer_top_k % kpool == 0, so the division is exact.
//
// Pools are defined on positions, but everything the graph indexes is addressed by
// *cell*, and a cell index is whatever llama_kv_cache::find_slot handed out: cells of
// one pool are not adjacent, not ordered, and under a unified cache not even owned by
// the same sequence. The mapping therefore cannot be derived in the graph. It is built
// here, host side, from the cache's cells, and passed in as plain input tensors, as
// qwen4exp's QSA does for its compression blocks.
//
// A position only names a pool inside ONE sequence, so the map is per SEQUENCE. A
// non-unified cache gives every sequence its own stream and its own cells array, and
// the two are the same thing. A unified cache puts every sequence of the ubatch in one
// stream and one cells array, so the stream's pool table is PARTITIONED: each sequence
// gets a contiguous run of slots and rebases inside it. `pool_bias` is -INFINITY outside
// the query's own run, which is what stops a query from selecting a foreign pool.
//
// The runs are packed, not one full-width table per sequence. A full-width table per
// sequence would multiply the pool axis by the sequence count, and the indexer scores
// every pool against every query, so llama-embedding (which asks for n_seq_max 256 with
// a unified cache) reserved 286 GB of compute buffer that way.
//
// Nothing here may emit a negative index: ggml_set_rows asserts i1 >= 0 and
// ggml_get_rows has no sentinel, so unpopulated entries are clamped into range and
// neutralised by the additive masks instead.
//

// number of pool slots the graph must allocate for `n_kv` cells shared by `n_seqs`
// sequences.
//
// pool ordinals are position-derived, then rebased on each sequence's lowest resident
// pool so sequences not starting at position 0 still land inside the array. rebasing
// can cost one slot at each end, hence 2 per sequence.
//
// n_kv/kpool is the budget the sequences share, which is exact while their cells are
// disjoint: a cell belongs to one pool of one sequence. llama_memory_seq_cp breaks that
// - a shared prefix cell is pooled by every sequence holding it - and then the runs are
// cut to the newest pools rather than the table being grown, see [TAG_KPOOL_PACK].
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// how many POOLS ggml_top_k selects.
//
// The reference (modular_glm5_next.py, Glm5NextTextIndexer.forward) scores the pool
// axis and takes select_k = min(index_topk // index_kpool, n_pools), then expands each
// selected pool to its members. This is that number, and top-k over pools rather than
// over cells is not an optimisation, it is the only correct spelling.
//
// The tempting alternative - broadcast a pool's score onto its kpool member cells and
// run one top-k of width indexer_top_k over cells - is WRONG, for a reason a
// set-similarity metric cannot see. Its argument is that a pool's members carry its
// score bit-exactly, so the cut must land on a pool boundary; that holds only if tie
// groups never span pools. They do: F.relu drives most pool scores to exactly 0.0 (on
// TinySparse, 92 cells tie at 0.0 for query 386 at layer 3), so the cut falls inside a
// tie group that straddles pools, and ggml_top_k - explicitly unordered among equals;
// the CPU op deliberately swaps the first two results, the CUDA op declares
// determinism::not_guaranteed - takes an arbitrary 1..kpool-1 members of the pool it
// cuts. Measured on TinySparse at 512 tokens the cell-level form leaves a partial pool
// on 7.51% of query rows at L3 and 5.93% at L7 (70 and 47 partial pools); the
// pool-level form leaves none. A pool-aligned top-k WIDTH does not help, because the
// ties are not aligned to anything.
//
// Selecting whole pools also removes the CPU/CUDA tie-break divergence structurally
// rather than making it unlikely: the backends may disagree about WHICH pools come out
// of a tie group, never about whether a pool is taken whole.
//
// NOT indexer_top_k + kpool - 1 either: that is the width of the reference's *output*
// buffer, tail included. The always-selected tail is forced in via `sel_mask`.
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// Fill the host-side inputs of the pooled indexer.
//
// `cell_pool` and `bias` are the per-CELL view of the same information and may be
// nullptr. They are kept because they are the independent spelling
// tests/test-glm5next-memory.cpp checks `pool_bias` and `cand_mask` against - but the
// graph selects at POOL granularity (see llama_kpool_select_k), so it passes nullptr
// for both. An input tensor with no consumer is never backed by the allocator, so
// building them anyway would write through a null buffer.
//
//   cell_pool  I32 [n_kv, n_stream]                    OPTIONAL
//       cell -> its pool slot, or 0 when it has no usable pool.
//
//   pool_cells I32 [kpool*n_pools, n_stream]
//       pool slot, member -> the cell holding that position, or 0 when not resident.
//       One stream's slots are cut into one contiguous run per sequence.
//       Two consumers: the compressor gathers a pool's member keys and gates with it,
//       and the top-k expands a selected POOL ordinal back into its kpool member CELLS
//       with it - the reference's `pool_indices[batch_idx, selected]`.
//
//   bias       F32 [n_kv, n_tokens/n_stream, n_stream] OPTIONAL
//       additive per-(cell, query) bias on a per-cell score: 0.0f in a complete pool
//       whose last member the query can see, else -INFINITY, the trailing incomplete
//       pool included.
//
//   pool_bias  F32 [n_pools, n_tokens/n_stream, n_stream]
//       the same predicate per POOL, which is where the reference applies it: 0.0f when
//       pool p is completely resident and its LAST member is visible to query q
//       (`pool_valid & pool_visible`), else -INFINITY, the query's own trailing pool
//       included so no budget is spent on it. Every slot outside the query's own
//       sequence run is -INFINITY, so a query never selects another sequence's pool.
//
//       Derived here rather than in the graph on purpose. Gathering `bias` at
//       each pool's last member looks equivalent and is not: an incomplete or
//       entirely absent pool has no resident last member, `pool_cells` points
//       that slot at cell 0 to keep the gather in range, and the pool would then
//       inherit cell 0's validity and compete for budget with a finite score.
//
//   sel_mask   F16/F32 [n_kv, n_batch, 1, n_stream]  (KQ mask shape, may be padded)
//       what the top-k scatter starts from, replacing the ggml_fill(kq_mask, -INFINITY)
//       opening the DSA mask build in llm_graph_context::build_attn: 0.0f for the
//       query's own incomplete trailing pool, which GLM always attends to
//       (index_kpool_always_select_tail), else -INFINITY, padding rows included.
//       Forcing the tail in here rather than through the score keeps the budget a whole
//       number of pools.
//
//   cand_mask  F16/F32 [n_kv, n_batch, 1, n_stream]  (KQ mask shape, may be padded)
//       max(bias, sel_mask): the reference's candidate set, i.e. every cell it could
//       return for this query. 0.0f in a complete visible pool OR in the query's own
//       tail, else -INFINITY, padding rows included.
//
//       ggml_top_k returns `select_k` pool ordinals even when fewer pools carry a
//       finite score, and during prefill that is the NORMAL state, not a corner case:
//       query q has only ~q/kpool complete visible pools against a budget of
//       index_topk/kpool (on TinySparse, 20 of 20 query rows are under budget). The
//       spilled ordinals are arbitrary among the -INFINITY ties and expanding them
//       unmasks cells. Adding the causal KQ mask kills the spills that are empty,
//       foreign or in the future. What it does not kill is a resident, causally visible
//       cell in an INCOMPLETE pool below the tail - unreachable while positions are
//       contiguous, reachable the moment a partial seq_rm leaves a hole. cand_mask
//       kills exactly those, for one store per element in a loop that already computes
//       both operands.
//
//       (The other spill, an unfilled pool slot pointing at cell 0, is a provable
//       no-op: the budget can only overflow once every finitely-scored pool is already
//       selected, so cell 0 is either already selected through its own pool, or masked
//       for the same reason it would have been anyway.)
//
// `kv` must be the ATTENTION (MLA) cache: its cells define the pools and are what the
// top-k indices are ultimately read against. llama_memory_hybrid gives the indexer
// cache the attention cache's slot layout, so the two agree cell for cell.
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// One pooling map per ubatch, shared by every indexer layer: all four tensors, and
// k_idxs, depend only on the cells and the ubatch, never on the layer. Rebuilding them
// per layer costs O(n_kv * n_tokens) host writes each time (at 128 Ki cells, 512 tokens
// and 11 DSA layers, ~11 x 67M float stores per ubatch, which dominates prefill).
//
// Sharing `pool_bias`, `sel_mask` and `cand_mask` is correct only while every indexer
// layer sees the same candidate set. That holds for glm5next, whose indexer_types are
// all "full"; a model mixing in windowed indexers would need one map per window.
//
// `k_idxs` is present whenever the model has an indexer cache; the rest only when the
// graph actually scores. The indexer key and gate STORE is not gated on the sparse path
// - below n_select the selection is a no-op but the cells still have to be written, or
// the first ubatch to cross n_select would pool cells that were never filled.
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * pool_bias  = nullptr;   // F32 [n_pools, n_tps, n_stream]

    // pool_bias in the shape and type the fused lightning indexer wants for its mask.
    // A ggml_cast node, not an input: built once per graph rather than once per DSA
    // layer, since every indexer layer shares the same mask. The cast is exact -
    // pool_bias only ever holds 0.0f or -INFINITY. nullptr when the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
