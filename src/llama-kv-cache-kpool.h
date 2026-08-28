#pragma once

#include "ggml.h"
#include "llama.h"
#include "llama-graph.h"

#include <cstdint>

struct llama_ubatch;
class llama_kv_cache;
class llama_kv_cache_context;

// GLM-5-Next indexer pooling. the position -> cell map is built host side because
// find_slot's cell order is arbitrary. no input may hold a negative index: ggml_set_rows
// asserts i1 >= 0 and ggml_get_rows has no sentinel, so unusable entries are clamped into
// range and neutralised by the additive masks instead.

// pool slots for `n_kv` cells shared by `n_seqs` sequences: n_kv/kpool, exact only while
// the sequences' cells are disjoint, plus 2 per sequence for rebasing.
uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs = 1);

// select_k of modular_glm5_next.py, Glm5NextTextIndexer.forward. must run over POOLS, not
// cells: relu ties span pool boundaries, so a cell-level cut takes partial pools.
uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool);

// `kv` must be the ATTENTION (MLA) cache; the indexer cache shares its slot layout.
//   cell_pool  I32 [n_kv, n_stream]                 per-cell view, optional, unused here
//   pool_cells I32 [kpool*n_pools, n_stream]        pool member -> cell, 0 if not resident
//   bias       F32 [n_kv, n_tps, n_stream]          per-cell view, optional, unused here
//   pool_bias  F32 [n_pools, n_tps, n_stream]       pool_valid & pool_visible, -INFINITY
//       outside the query's own sequence run; computed, not gathered from `bias` at the
//       last member, which an incomplete pool lacks and would inherit cell 0's validity
//   sel_mask   F16/F32 [n_kv, n_batch, 1, n_stream] 0.0f on the always-selected tail only
//   cand_mask  F16/F32 [n_kv, n_batch, 1, n_stream] max(bias, sel_mask); bounds the top-k
//       spills that a partial seq_rm would otherwise let escape the candidate set
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

// One pooling map per ubatch; rebuilding it per indexer layer costs O(n_kv * n_tokens)
// host writes and dominates prefill. sharing is valid only while every indexer layer sees
// the same candidate set - true for glm5next (indexer_types all "full"), not for windowed.
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

    // exact, since pool_bias only holds 0.0f or -INFINITY. nullptr if the fused path is off
    ggml_tensor * pool_bias_f16 = nullptr; // F16 [n_pools, n_tps, 1, n_stream]

    ggml_tensor * sel_mask   = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]
    ggml_tensor * cand_mask  = nullptr;   // F16 [n_kv, n_batch, 1, n_stream]

    const llama_kv_cache_context * mctx_attn;
    const llama_kv_cache_context * mctx_idx;

    const uint32_t kpool;
};
