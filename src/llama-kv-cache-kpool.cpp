#include "llama-kv-cache-kpool.h"

#include "llama-batch.h"
#include "llama-kv-cache.h"
#include "llama-kv-cells.h"

#include <algorithm>
#include <cmath>
#include <vector>

uint32_t llama_kpool_n_pools(uint32_t n_kv, uint32_t kpool, uint32_t n_seqs) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_seqs > 0);

    return n_kv/kpool + 2*n_seqs;
}

uint32_t llama_kpool_select_k(uint32_t n_pools, uint32_t indexer_top_k, uint32_t kpool) {
    GGML_ASSERT(kpool > 0);
    GGML_ASSERT(n_pools > 0);
    GGML_ASSERT(indexer_top_k % kpool == 0 && "indexer_top_k must be a whole number of pools");

    // min(index_topk // index_kpool, n_pools), exactly the reference's select_k
    return std::min(n_pools, indexer_top_k/kpool);
}

void llama_kv_cache_set_input_kpool(
        const llama_kv_cache * kv,
              ggml_tensor    * cell_pool,
              ggml_tensor    * pool_cells,
              ggml_tensor    * bias,
              ggml_tensor    * pool_bias,
              ggml_tensor    * sel_mask,
              ggml_tensor    * cand_mask,
        const llama_ubatch   * ubatch,
              uint32_t         kpool) {
    GGML_ASSERT(kv != nullptr);
    GGML_ASSERT(kpool > 0);

    // the per-CELL view is optional: the pooled graph does not consume it, and an
    // input tensor with no consumer is never backed by the allocator
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_cells->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(pool_bias ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(sel_mask  ->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(cand_mask ->buffer));

    // sel_mask and cand_mask are KQ-mask shaped, and KQ masks are f16 under flash
    // attention; writing floats into one would overrun the allocation 2x, so check
    // rather than trust
    GGML_ASSERT(pool_cells->type == GGML_TYPE_I32);
    GGML_ASSERT(pool_bias ->type == GGML_TYPE_F32);
    GGML_ASSERT(sel_mask  ->type == GGML_TYPE_F32 && "sel_mask must be f32 even when the KQ mask is f16");
    GGML_ASSERT(cand_mask ->type == GGML_TYPE_F32 && "cand_mask must be f32 even when the KQ mask is f16");

    // everything below is written through raw strides
    GGML_ASSERT(ggml_is_contiguous(pool_cells));
    GGML_ASSERT(ggml_is_contiguous(pool_bias));
    GGML_ASSERT(ggml_is_contiguous(sel_mask));
    GGML_ASSERT(ggml_is_contiguous(cand_mask));

    const int64_t n_kv     = sel_mask->ne[0];
    const int64_t n_ns     = sel_mask->ne[3];           // streams in this ubatch
    const int64_t r        = kpool;
    const int64_t n_tokens = ubatch->n_tokens;

    // [TAG_KPOOL_SEQ_PARTITION]
    // positions are unambiguous only within one sequence, so one pool map per SEQUENCE,
    // not per stream. a non-unified cache gives each stream its own cells array and one
    // sequence (n_ps == 1, the layout this file had before), a unified cache gives one
    // stream carrying every sequence in the ubatch
    GGML_ASSERT(n_ns == 1 || (int64_t) ubatch->n_seqs_unq == n_ns);

    const int64_t n_ps    = (int64_t) ubatch->n_seqs_unq/n_ns;  // sequences per stream
    const int64_t n_pools = pool_cells->ne[0]/r;                // pool slots per stream

    GGML_ASSERT(n_ps > 0 && (int64_t) ubatch->n_seqs_unq == n_ns*n_ps);
    GGML_ASSERT(pool_cells->ne[0] % r == 0);
    GGML_ASSERT(n_pools >= 2*n_ps);
    GGML_ASSERT(pool_cells->ne[1] == n_ns);
    GGML_ASSERT(sel_mask->ne[2] == 1);
    GGML_ASSERT(ggml_are_same_shape(cand_mask, sel_mask));
    GGML_ASSERT(pool_bias->ne[0] == n_pools && pool_bias->ne[2] == n_ns);
    GGML_ASSERT(n_tokens % n_ns == 0);

    const int64_t n_tps  = n_tokens/n_ns;               // tokens per stream
    const int64_t n_padq = sel_mask->ne[1];             // KQ mask rows, >= n_tps

    GGML_ASSERT(pool_bias->ne[1] == n_tps);
    GGML_ASSERT(n_padq >= n_tps);

    if (cell_pool) {
        GGML_ASSERT(ggml_backend_buffer_is_host(cell_pool->buffer));
        GGML_ASSERT(cell_pool->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(cell_pool));
        GGML_ASSERT(cell_pool->ne[0] == n_kv && cell_pool->ne[1] == n_ns);

        // one row per stream, so a cell that two sequences of one stream share has
        // nowhere to put its second pool. the graph never asks for this view
        GGML_ASSERT(n_ps == 1 && "the per-cell pool view needs one sequence per stream");
    }

    if (bias) {
        GGML_ASSERT(ggml_backend_buffer_is_host(bias->buffer));
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(bias->ne[0] == n_kv && bias->ne[1] == n_tps && bias->ne[2] == n_ns);
    }

    int32_t * dst_cell_pool  = cell_pool ? (int32_t *) cell_pool->data : nullptr;
    int32_t * dst_pool_cells = (int32_t *) pool_cells->data;
    float   * dst_bias       = bias ? (float *) bias->data : nullptr;
    float   * dst_pool_bias  = (float   *) pool_bias ->data;
    float   * dst_sel_mask   = (float   *) sel_mask  ->data;
    float   * dst_cand_mask  = (float   *) cand_mask ->data;

    // -1 marks a cell with no usable pool. host side only: never copied into cell_pool,
    // where ggml_get_rows would read it as an index
    std::vector<int32_t>   pool_of(n_kv);
    std::vector<int32_t>   filled(n_pools);
    std::vector<llama_pos> pos_at;

    // one contiguous run of pool slots per sequence of the stream
    std::vector<int64_t> run_off(n_ps);
    std::vector<int64_t> run_len(n_ps);

    // which cells array a sequence of this stream uses; same convention as
    // llama_kv_cache::set_input_kq_mask. with one sequence per stream the ubatch's
    // unique list and the stream's own sequence are the same thing
    auto seq_of = [&](int64_t s, int64_t ps) {
        return n_ps == 1 ? ubatch->seq_id[s*n_tps][0] : ubatch->seq_id_unq[ps];
    };

    for (int64_t s = 0; s < n_ns; ++s) {
        int32_t * cur_pool_cells = dst_pool_cells + s*(r*n_pools);
        float   * cur_sel_mask   = dst_sel_mask   + s*(n_padq*n_kv);
        float   * cur_cand_mask  = dst_cand_mask  + s*(n_padq*n_kv);
        float   * cur_pool_bias  = dst_pool_bias  + s*(n_tps*n_pools);

        // slots of a pool that is not resident, and slots outside the query's own
        // sequence run, are cleared once per stream here. the per-sequence pass below
        // only writes what it owns
        std::fill(cur_pool_cells, cur_pool_cells + r*n_pools, 0);
        std::fill(cur_pool_bias,  cur_pool_bias  + n_tps*n_pools, -INFINITY);

        // the token loop writes rows < n_tps in full; only the padding rows need clearing
        std::fill(cur_sel_mask  + n_tps*n_kv, cur_sel_mask  + n_padq*n_kv, -INFINITY);
        std::fill(cur_cand_mask + n_tps*n_kv, cur_cand_mask + n_padq*n_kv, -INFINITY);

        // [TAG_KPOOL_PACK]
        // cut the stream's pool table into one run per sequence, sized on the pool range
        // that sequence actually holds. the table is n_kv/kpool shared plus 2 per sequence
        // for rebasing, which covers it whenever the sequences' cells are disjoint - every
        // case but a prefix shared through llama_memory_seq_cp. that one can ask for more
        // slots than exist, and then a sequence keeps its newest pools, the same cut a
        // large hole already forces.
        //
        // NOT one full-width table per sequence: the indexer scores every slot against
        // every query, so a full-width table would multiply the score tensor by the
        // sequence count, and llama-embedding asks for n_seq_max 256
        {
            int64_t n_want = 0;

            for (int64_t ps = 0; ps < n_ps; ++ps) {
                const llama_seq_id seq = seq_of(s, ps);
                const auto & cells = kv->get_cells(seq);

                int64_t b_min = 0;
                int64_t b_max = 0;
                bool    found = false;

                for (int64_t j = 0; j < n_kv; ++j) {
                    if (cells.is_empty(j) || !cells.seq_has(j, seq)) {
                        continue;
                    }
                    const int64_t b = cells.pos_get(j)/r;
                    b_min = found ? std::min(b_min, b) : b;
                    b_max = found ? std::max(b_max, b) : b;
                    found = true;
                }

                run_len[ps] = found ? b_max - b_min + 1 : 0;
                n_want += run_len[ps];
            }

            if (n_want > n_pools) {
                int64_t rem = n_pools;

                for (int64_t ps = 0; ps < n_ps; ++ps) {
                    run_len[ps] = std::min(run_len[ps], rem/(n_ps - ps));
                    rem -= run_len[ps];
                }
            }

            int64_t off = 0;
            for (int64_t ps = 0; ps < n_ps; ++ps) {
                run_off[ps] = off;
                off += run_len[ps];
            }

            GGML_ASSERT(off <= n_pools);
        }

        int64_t n_done = 0;

        for (int64_t ps = 0; ps < n_ps; ++ps) {
            const llama_seq_id seq_of_pool = seq_of(s, ps);
            const auto & cells = kv->get_cells(seq_of_pool);

            const int64_t n_run = run_len[ps];

            int32_t * cur_cell_pool   = dst_cell_pool ? dst_cell_pool + s*n_kv : nullptr;
            int32_t * part_pool_cells = cur_pool_cells + run_off[ps]*r;

            std::fill(pool_of.begin(), pool_of.end(), -1);
            std::fill(filled.begin(),  filled.end(),   0);

            // hoist occupancy and sequence membership out of the O(n_kv * n_tokens) loop
            // below; neither depends on the query. -1 means the cell holds nothing this
            // sequence may pool or attend to. under a unified cache `cells` is shared with
            // every other sequence, and seq_has is what keeps their keys out
            pos_at.resize(n_kv);
            for (int64_t j = 0; j < n_kv; ++j) {
                pos_at[j] = cells.is_empty(j) || !cells.seq_has(j, seq_of_pool) ? -1 : cells.pos_get(j);
            }

            // a pool ordinal is the absolute p/kpool, which can far exceed n_kv/kpool, so
            // rebase on this sequence's lowest resident pool. grouping is untouched, every
            // member shifts together.
            //
            // anchoring at p/kpool follows vLLM and SGLang, not HF (which pools from the
            // first *resident* key, valid_keys.argmax(-1), differing under left padding).
            // it is the only anchor that keeps a pool's identity stable between the prefill
            // that built it and the decodes that read it.
            //
            // the window is this sequence's run; positions are contiguous in any real
            // batch so the resident range fits. seq_rm can leave a hole large enough that
            // it does not, and then the newest pools are the ones worth keeping.
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

                b_base = std::max(b_min, b_max - (n_run - 1));
            }

            for (int64_t j = 0; j < n_kv; ++j) {
                if (pos_at[j] < 0) {
                    continue;
                }

                const llama_pos p  = pos_at[j];
                const int64_t   bo = p/r - b_base;

                if (bo < 0 || bo >= n_run) {
                    continue;
                }

                pool_of[j] = (int32_t) bo;
                part_pool_cells[bo*r + (p%r)] = (int32_t) j;
                filled[bo]++;
            }

            // an incompletely resident pool cannot be pooled: the compressor consumes all r
            // member keys, and the reference demands pool_valid = grouped_valid_keys.all(-1).
            // such cells are the sequence tail, which sel_mask forces in below regardless of
            // score, so they point at pool slot 0 purely to keep the gather in range
            for (int64_t j = 0; j < n_kv; ++j) {
                // != rather than <: two cells claiming one position overwrite each other in
                // pool_cells, so an over-filled pool is not usable either
                if (pool_of[j] >= 0 && filled[pool_of[j]] != (int32_t) r) {
                    pool_of[j] = -1;
                }
                if (cur_cell_pool) {
                    cur_cell_pool[j] = pool_of[j] < 0 ? 0 : pool_of[j];
                }
            }

            for (int64_t ii = 0; ii < n_tps; ++ii) {
                const int64_t   i = s*n_tps + ii;

                // a query is pooled by the sequence it is being written to. with several
                // sequences in one stream the other partitions own the rest of the rows
                if (ubatch->seq_id[i][0] != seq_of_pool) {
                    continue;
                }

                const llama_pos q = ubatch->pos[i];

                // q >= 0 is what makes the unsigned range test below a range test
                GGML_ASSERT(q >= 0);

                n_done++;

                // the query's own incomplete pool ((q + 1) % r cells, its own token
                // included) is always attended to (index_kpool_always_select_tail), which is
                // what makes the selection land on pool boundaries
                const llama_pos tail_start = (q + 1)/r*r;

                // the reference tests visibility at a pool's LAST member, so a pool
                // straddling the query is dropped whole. pools are position-aligned here, so
                // that test collapses to b*r < tail_start
                const int64_t bo_vis = std::max<int64_t>(0, tail_start/r - b_base);

                float * cur_bias = dst_bias ? dst_bias + i*n_kv : nullptr;
                float * cur_sel  = cur_sel_mask  + ii*n_kv;
                float * cur_cand = cur_cand_mask + ii*n_kv;

                // the unsigned compares fold "empty or another sequence" (pos_at -1) and "no
                // usable pool" (pool_of -1) into the range test, which lets this vectorise
                for (int64_t j = 0; j < n_kv; ++j) {
                    const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                    const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;
                    const bool tail   = pos_at[j] >= tail_start;

                    cur_sel [j] = vis && tail   ? 0.0f : -INFINITY;
                    // max(bias, sel_mask): the reference's candidate set, which the
                    // top-k budget may overrun but must never escape
                    cur_cand[j] = vis && (pooled || tail) ? 0.0f : -INFINITY;
                }

                if (cur_bias) {
                    for (int64_t j = 0; j < n_kv; ++j) {
                        const bool vis    = (uint32_t) pos_at [j] <= (uint32_t) q;
                        const bool pooled = (uint32_t) pool_of[j] <  (uint32_t) bo_vis;

                        cur_bias[j] = vis && pooled ? 0.0f : -INFINITY;
                    }
                }

                // The same predicate, per POOL, which is where the reference applies
                // it: pool_valid (completely resident) & pool_visible (its LAST
                // member is visible, so a pool the query straddles is dropped whole).
                // Pools are position-aligned here, so pool bo's last member is at
                // position (b_base + bo)*r + r - 1 and "last member visible" collapses
                // to bo < bo_vis.
                //
                // NOT gathered from `bias` at the last member cell: an incomplete or
                // absent pool has no resident last member, pool_cells points that slot
                // at cell 0, and the pool would inherit cell 0's validity.
                //
                // the query's own sequence run only. every other slot stays at the
                // -INFINITY the per-stream fill above left, which is what keeps a foreign
                // pool out of the budget
                float * q_pool_bias = cur_pool_bias + ii*n_pools + run_off[ps];

                for (int64_t p = 0; p < n_run; ++p) {
                    const bool valid   = filled[p] == (int32_t) r;
                    const bool visible = p < bo_vis;

                    q_pool_bias[p] = valid && visible ? 0.0f : -INFINITY;
                }
            }
        }

        // every row of sel_mask, cand_mask and pool_bias must have been written by
        // exactly one partition, or a query is left reading another sequence's pools
        GGML_ASSERT(n_done == n_tps && "every query must belong to a sequence of the ubatch");
    }
}

void llm_graph_input_kpool::set_input(const llama_ubatch * ubatch) {
    // unconditional: the indexer key and gate STORE runs on the dense path too,
    // and k_idxs is what tells cpy_k where to put them. Gating the store the way
    // the scoring is gated would leave every cell written below n_select - the
    // first 2051 positions of every sequence on the real model - with no indexer
    // state, and the first ubatch to cross n_select would pool cells that were
    // never written
    mctx_idx->set_input_k_idxs(k_idxs, ubatch);

    // the rest exists only when the graph scores. below n_select the indexer
    // would select every visible position, so build_inp_kpool does not allocate
    // these at all and there is nothing to fill
    if (pool_cells == nullptr) {
        return;
    }

    llama_kv_cache_set_input_kpool(
            mctx_attn->get_kv(),
            /* cell_pool */ nullptr, pool_cells, /* bias */ nullptr, pool_bias,
            sel_mask, cand_mask, ubatch, kpool);
}
