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
// GLM's lightning indexer scores pools of `kpool` consecutive *positions* rather
// than single tokens, and its top-k budget is counted in tokens (indexer_top_k),
// i.e. indexer_top_k/kpool whole pools. The reference requires
// indexer_top_k % kpool == 0, so that division is exact.
//
// A pool is defined on positions, but everything the graph indexes - the indexer
// key cache, the MLA KQ mask - is addressed by *cell*, and a cell index is
// whatever llama_kv_cache::find_slot happened to hand out. Cells of one pool are
// not adjacent, not ordered, and with a unified cache not even owned by the same
// sequence. The mapping between the two therefore cannot be derived in the
// graph; it is built here, host side, from the cache's own cells, and handed to
// the graph as plain input tensors. This mirrors what qwen4exp's QSA does for
// its compression blocks.
//
// Nothing here ever emits a negative index: ggml_set_rows asserts i1 >= 0 and
// ggml_get_rows has no sentinel either, so unpopulated entries are clamped into
// range and neutralised by the additive masks instead.
//

// number of pool slots the graph has to allocate for `n_kv` cells.
//
// pool ordinals are position-derived and then rebased on the lowest resident
// pool of each stream, so a sequence whose positions do not start at 0 still
// lands inside the array. rebasing can cost one slot at each end, hence the +2.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool);

// width to run ggml_top_k at.
//
// NOT indexer_top_k + kpool - 1, which is the width of the reference's *output*
// buffer, tail included. ggml_top_k is explicitly unordered among equals
// (ggml/src/ggml-cpu/ops.cpp uses std::partial_sort and then swaps the first two
// results to make the point; the CUDA op declares determinism::not_guaranteed),
// so a budget that does not end on a pool boundary picks an arbitrary 1..kpool-1
// members out of the pool it cuts, and picks differently on CPU and on CUDA.
// With the tail biased to -INFINITY the budget is spent only on whole pools, and
// indexer_top_k is a multiple of kpool by construction, so the cut is exact. The
// tail is forced back in through `sel_mask` instead of through the budget.
uint32_t llama_kpool_top_k_width(uint32_t n_kv, uint32_t indexer_top_k, uint32_t kpool);

// Fill the host-side inputs of the pooled indexer.
//
//   cell_pool  I32 [n_kv, n_stream]
//       for cell j of stream s: the pool slot it belongs to, or 0 when it has no
//       usable pool. Used to broadcast a pool's score back onto its member cells
//       with ggml_get_rows, so that ggml_top_k over the replicated per-cell
//       scores yields CELL indices directly and the cut still lands on a pool
//       boundary (a pool's members tie bit-exactly).
//
//   pool_cells I32 [kpool*n_pools, n_stream]
//       for pool slot p, member m of stream s: the cell holding that member's
//       position, or 0 when it is not resident. Used to gather a pool's member
//       keys and gates before the learned compressor mixes them.
//
//   bias       F32 [n_kv, n_tokens/n_stream, n_stream]
//       additive per-(cell, query) bias on the indexer SCORE:
//         0.0f       the cell is in a complete pool whose last member the query
//                    can see, which is the reference's `pool_valid &
//                    pool_visible`
//         -INFINITY  everything else, the trailing incomplete pool included, so
//                    that no part of the top-k budget is spent on it
//
//   sel_mask   F32 [n_kv, n_batch, 1, n_stream]  (KQ mask shape, may be padded)
//       the tensor the top-k scatter starts from, in place of the
//       ggml_fill(kq_mask, -INFINITY) that opens the DSA mask build in
//       llm_graph_context::build_attn. Forcing the tail in here rather than
//       through the score is what lets the budget stay pool-aligned:
//         0.0f       the cell is in the query's own incomplete trailing pool,
//                    which GLM always attends to
//                    (index_kpool_always_select_tail)
//         -INFINITY  everything else, including the padding rows
//
// `kv` must be the ATTENTION (MLA) cache, since the cells that define the pools
// are the ones the top-k indices are ultimately read against. The indexer cache
// is given the attention cache's slot layout by llama_memory_hybrid, so the two
// agree cell for cell.
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * sel_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// One pooling map per ubatch, shared by every indexer layer.
//
// All four tensors depend only on the cells and the ubatch, never on the layer,
// and so does the indexer cache's k_idxs. Rebuilding them per layer costs
// O(n_kv * n_tokens) host writes each time: at 128 Ki cells, 512 tokens and 11
// DSA layers that is ~11 x 67M float stores per ubatch, which dominates prefill.
// The model graph creates one of these before the layer loop and reads the same
// tensors in every layer.
//
// Sharing `bias` and `sel_mask` too is only correct while every indexer layer
// sees the same candidate set. That holds for glm5next, whose indexer_types are
// all "full"; an architecture that mixed windowed indexers into the same model
// would need one bias per window.
class llm_graph_input_kpool : public llm_graph_input_i {
public:
    llm_graph_input_kpool(
            const llama_kv_cache_context * mctx_attn,
            const llama_kv_cache_context * mctx_idx,
            uint32_t kpool) : mctx_attn(mctx_attn), mctx_idx(mctx_idx), kpool(kpool) {}

    ~llm_graph_input_kpool() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_idxs     = nullptr;   // I32 [n_tokens]
    ggml_tensor * cell_pool  = nullptr;   // I32 [n_kv, n_stream]
    ggml_tensor * pool_cells = nullptr;   // I32 [kpool*n_pools, n_stream]
    ggml_tensor * bias       = nullptr;   // F32 [n_kv, n_tps, n_stream]
    ggml_tensor * sel_mask   = nullptr;   // F32 [n_kv, n_batch, 1, n_stream]

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
