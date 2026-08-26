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
// Nothing here may emit a negative index: ggml_set_rows asserts i1 >= 0 and
// ggml_get_rows has no sentinel, so unpopulated entries are clamped into range and
// neutralised by the additive masks instead.
//

// number of pool slots the graph must allocate for `n_kv` cells.
//
// pool ordinals are position-derived, then rebased on each stream's lowest resident
// pool so sequences not starting at position 0 still land inside the array. rebasing
// can cost one slot at each end, hence the +2.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool);

// width to run ggml_top_k at.
//
// NOT indexer_top_k + kpool - 1: that is the width of the reference's *output* buffer,
// not a top-k width. ggml_top_k is explicitly unordered among equals (the CPU op
// deliberately swaps the first two results; the CUDA op declares
// determinism::not_guaranteed), so a budget that does not end on a pool boundary would
// take an arbitrary, CPU/CUDA-divergent subset of the pool it cuts. Biasing the tail to
// -INFINITY keeps the budget on whole pools; the tail is forced back in via `sel_mask`.
// See the PR body for the full argument.
uint32_t llama_kpool_top_k_width(uint32_t n_kv, uint32_t indexer_top_k, uint32_t kpool);

// Fill the host-side inputs of the pooled indexer.
//
//   cell_pool  I32 [n_kv, n_stream]
//       cell -> its pool slot, or 0 when it has no usable pool. ggml_get_rows
//       broadcasts a pool's score onto its members, so ggml_top_k over the replicated
//       scores yields CELL indices directly and still cuts on a pool boundary (members
//       tie bit-exactly).
//
//   pool_cells I32 [kpool*n_pools, n_stream]
//       pool slot, member -> the cell holding that position, or 0 when not resident.
//       gathers a pool's member keys and gates for the compressor.
//
//   bias       F32 [n_kv, n_tokens/n_stream, n_stream]
//       additive per-(cell, query) bias on the indexer SCORE: 0.0f in a complete pool
//       whose last member the query can see (the reference's `pool_valid &
//       pool_visible`), else -INFINITY, the trailing incomplete pool included so no
//       budget is spent on it.
//
//   sel_mask   F32 [n_kv, n_batch, 1, n_stream]  (KQ mask shape, may be padded)
//       what the top-k scatter starts from, replacing the ggml_fill(kq_mask, -INFINITY)
//       opening the DSA mask build in llm_graph_context::build_attn: 0.0f for the
//       query's own incomplete trailing pool, which GLM always attends to
//       (index_kpool_always_select_tail), else -INFINITY, padding rows included.
//       forcing the tail in here rather than through the score keeps the budget
//       pool-aligned.
//
// `kv` must be the ATTENTION (MLA) cache: its cells define the pools and are what the
// top-k indices are ultimately read against. llama_memory_hybrid gives the indexer
// cache the attention cache's slot layout, so the two agree cell for cell.
void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * sel_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool);

// One pooling map per ubatch, shared by every indexer layer: all four tensors, and
// k_idxs, depend only on the cells and the ubatch, never on the layer. Rebuilding them
// per layer costs O(n_kv * n_tokens) host writes each time (at 128 Ki cells, 512 tokens
// and 11 DSA layers, ~11 x 67M float stores per ubatch, which dominates prefill).
//
// Sharing `bias` and `sel_mask` is correct only while every indexer layer sees the same
// candidate set. That holds for glm5next, whose indexer_types are all "full"; a model
// mixing in windowed indexers would need one bias per window.
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
