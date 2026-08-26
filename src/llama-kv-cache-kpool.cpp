#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cmath>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);

    return n_kv/kpool + 2;
}

uint32_t llama_kpool_top_k_width(uint32_t n_kv, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    return std::min(n_kv, indexer_top_k);
}

void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * sel_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(bias      ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask  ->buffer));

    // sel_mask is KQ-mask shaped, and every KQ mask in the tree is f16 under
    // flash attention. writing floats into one would overrun the allocation by
    // 2x, so refuse rather than trust the caller
    GGML_ASSERT(cell_pool ->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(bias      ->type == GGML_TYPE_F32);
    GGML_ASSERT(sel_mask  ->type == GGML_TYPE_F32 && "sel_mask must be f32 even when the KQ mask is f16");

    // everything below is written through raw strides
    GGML_ASSERT(ggml_is_contiguous(cell_pool));
    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(bias));
    GGML_ASSERT(ggml_is_contiguous(sel_mask));

    const int64_t n_kv     = cell_pool->ne[0];
    const int64_t n_ns     = cell_pool->ne[1];          // streams in this ubatch
    const int64_t r        = kpool;
    const int64_t n_pools  = pool_cells->ne[0]/r;
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[2] == n_ns);
    GGML_ASSERT(sel_mask->ne[0] == n_kv && sel_mask->ne[2] == 1 && sel_mask->ne[3] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;               // tokens per stream
    const int64_t n_padq = sel_mask->ne[1];             // KQ mask rows, >= n_tps

    GGML_ASSERT(bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

    int32_t * dst_cell_pool  = (int32_t *) cell_pool ->data;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = (float   *) bias      ->data;
    float   * dst_sel_mask   = (float   *) sel_mask  ->data;

    // -1 marks a cell with no usable pool. Kept host side only: it is never
    // copied into cell_pool, which ggml_get_rows would read as an index
    std::vector<int32_t>   pool_of(n_kv);
    std::vector<int32_t>   filled(n_pools);
    std::vector<llama_pos> pos_at;

    // [TAG_KPOOL_NEEDS_ONE_SEQ_PER_STREAM]
    // A pool is a set of cells grouped by position, and cell positions are only
    // unambiguous within one sequence. Under a unified cache all sequences share
    // one cells array, so two sequences holding the same position would collide
    // in pool_cells and silently pool each other's keys. The pooled indexer
    // therefore needs one sequence per stream: either a non-unified cache
    // (n_stream == n_seq_max, the default) or a single sequence in flight.
    // qwen4exp's set_input_qsa has the same requirement and no check.
    GGML_ASSERT((int64_t) ubatch->n_seqs_unq == n_ns &&
            "the pooled indexer needs one sequence per stream; use a non-unified KV cache");

    for (int64_t s = 0; s < n_ns; ++s) {
        // the token at ubatch index s*n_tps belongs to this stream; ask the cache
        // which cells array that sequence actually uses. same convention as
        // llama_kv_cache::set_input_kq_mask
        const llama_seq_id seq_of_stream = ubatch->seq_id[s*n_tps][0];
        const auto & cells = kv->get_cells(seq_of_stream);

        int32_t * cur_cell_pool  = dst_cell_pool  + s*n_kv;
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);

        std::fill(pool_of.begin(), pool_of.end(), -1);
        std::fill(filled.begin(),  filled.end(),   0);
        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);

        // hoist occupancy and sequence membership out of the O(n_kv * n_tokens)
        // loop below: neither depends on the query, and this stream holds exactly
        // one sequence. -1 means the cell holds nothing this stream may pool or
        // attend to. under a unified cache `cells` is shared with the sequences
        // that are not in this ubatch, and seq_has is what keeps their keys out
        pos_at.resize(n_kv);
        for (int64_t j = 0; j < n_kv; ++j) {
            pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_stream) ? -1 : cells.pos_get(j);
        }

        // Pools are cut out of the position line, so a pool ordinal is p/kpool -
        // an absolute number that can be far larger than n_kv/kpool. The array
        // slot is a separate thing, so rebase on the lowest resident pool of this
        // stream. Grouping is untouched: every member of a pool shifts together.
        //
        // Not the reference's anchor. HF pools from the first *resident* key
        // (valid_keys.argmax(-1)), which for a left-padded batch differs from
        // p/kpool; vLLM and SGLang both anchor at p/kpool exactly as here, and
        // that is the only choice that keeps a pool's identity stable between the
        // prefill that built it and the decode steps that read it.
        //
        // The window is n_pools wide. Positions are contiguous in any batch the
        // model actually sees, so the whole resident range fits and the base is
        // just the lowest pool. seq_rm can leave a hole large enough that it does
        // not, and then the newest pools are the ones worth keeping.
        int64_t b_base = 0;
        {
            int64_t b_min = 0;
            int64_t b_max = 0;
            bool    found = false;

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }
                const int64_t b = pos_at[j]/r;
                b_min = found ? std::min(b_min, b) : b;
                b_max = found ? std::max(b_max, b) : b;
                found = true;
            }

            b_base = std::max(b_min, b_max - (n_pools - 1));
        }

        for (int64_t j = 0; j < n_kv; ++j) {
            if (pos_at[j] < 0) {
                continue;
            }

            const llama_pos p  = pos_at[j];
            const int64_t   bo = p/r - b_base;

            if (bo < 0 || bo >= n_pools) {
                continue;
            }

            pool_of[j] = (int32_t) bo;
            cur_pool_cells[bo*r + (p%r)] = (int32_t) j;
            filled[bo]++;
        }

        // a pool that is not completely resident cannot be pooled: the learned
        // compressor consumes all r member keys, and the reference demands
        // pool_valid = grouped_valid_keys.all(-1). Those cells are the tail of
        // the sequence, which sel_mask forces in below whatever score they carry,
        // so they are pointed at pool slot 0 only to keep the gather in range
        for (int64_t j = 0; j < n_kv; ++j) {
            // != rather than <: two cells claiming one position would overwrite
            // each other in pool_cells, so such a pool is not usable either
            if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                pool_of[j] = -1;
            }
            cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
        }

        float * cur_sel_mask = dst_sel_mask + s*(n_padq*n_kv);

        // the rows below n_tps are written in full by the loop; only the KQ mask's
        // padding rows need clearing
        std::fill(cur_sel_mask + n_tps*n_kv, cur_sel_mask + n_padq*n_kv, -INFINITY);

        for (int64_t ii = 0; ii < n_tps; ++ii) {
            const int64_t   i = s*n_tps + ii;
            const llama_pos q = ubatch->pos[i];

            // q >= 0 is what makes the unsigned range test below a range test
            GGML_ASSERT(q >= 0 && ubatch->seq_id[i][0] == seq_of_stream);

            // everything from here on is inside the query's own incomplete pool
            // and is always attended to (index_kpool_always_select_tail), which
            // is what makes the selection land on pool boundaries. (q + 1) % r
            // cells, the query's own token included
            const llama_pos tail_start = (q + 1)/r*r;

            // the reference tests visibility at a pool's LAST member, so a pool
            // that straddles the query is dropped whole rather than partially
            // masked. Pools are position-aligned here, so pool b's last member is
            // position b*r + r - 1 and the test collapses to b*r < tail_start
            const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

            float * cur_bias = dst_bias     + i*n_kv;
            float * cur_sel  = cur_sel_mask + ii*n_kv;

            // the unsigned compares fold "empty or another sequence" (pos_at -1)
            // and "no usable pool" (pool_of -1) into the same branch as the range
            // test, which is what lets this vectorise
            for (int64_t j = 0; j < n_kv; ++j) {
                const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
                const bool tail   = pos_at[j] >= tail_start;

                cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                cur_sel [j] = vis && tail   ? 0.0f : -INFINITY;
            }
        }
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(), cell_pool, pool_cells, bias, sel_mask, ubatch, kpool);
}
