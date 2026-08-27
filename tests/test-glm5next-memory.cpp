// The GLM-5-Next hybrid memory: the memory object holds all three halves (KDA
// recurrent + conv state, MLA latent KV, pooled indexer key cache) at the sizes the
// reference implies, one ggml graph reaches every one of them, and the pooled top-k
// cuts on a pool boundary and agrees between CPU and CUDA. The indexer graph itself is
// not built here.
//
// Run as:  test-glm5next-memory <tiny-glm5next.gguf>, as written by
// tests/glm5next_make_tiny_gguf.py

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-model.h"
#include "../src/llama-hparams.h"
#include "../src/llama-cparams.h"
#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-recurrent.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cache-kpool.h"
#include "../src/llama-batch.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <set>
#include <vector>

static int n_fail = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            printf("FAIL %s:%d: ", __func__, __LINE__);      \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
            n_fail++;                                        \
        } else {                                             \
            printf("ok   ");                                 \
            printf(__VA_ARGS__);                             \
            printf("\n");                                    \
        }                                                    \
    } while (0)

//
// numeric evidence for WHERE the top-k runs, independent of any model
//
// The claim under test is the one that decides the whole design: a top-k over
// CELLS cannot be made pool-aligned by choosing its width, and a top-k over
// POOLS is pool-aligned by construction.
//
// The tempting argument for the cell-level form is that a pool's members carry
// its score bit-exactly, so an intra-pool tie is harmless and a budget that is a
// whole multiple of kpool must cut on a pool boundary. That argument silently
// assumes tie groups never SPAN pools. F.relu drives most pool scores to exactly
// 0.0, so they do, and ggml_top_k - explicitly unordered among equals - then
// takes an arbitrary 1..kpool-1 members of the pool it lands in. This
// reproduces that with no model at all: most pools at exactly 0.0, a minority
// positive, fewer positive pools than the budget.
//

static std::vector<int32_t> run_top_k(
        ggml_backend_t backend,
        const std::vector<float> & scores,
        int64_t n_kv,
        int64_t n_rows,
        int64_t width) {
    ggml_init_params gparams = {
        /* .mem_size   */ ggml_tensor_overhead()*8 + ggml_graph_overhead(),
        /* .mem_buffer */ nullptr,
        /* .no_alloc   */ true,
    };

    ggml_context * ctx = ggml_init(gparams);

    ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_kv, n_rows);
    ggml_set_input(src);

    ggml_tensor * dst = ggml_top_k(ctx, src, (int) width);
    ggml_set_output(dst);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, dst);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<int32_t> out(width*n_rows);

    if (buf) {
        ggml_backend_tensor_set(src, scores.data(), 0, scores.size()*sizeof(float));

        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            ggml_backend_tensor_get(dst, out.data(), 0, out.size()*sizeof(int32_t));
        } else {
            out.clear();
        }

        ggml_backend_buffer_free(buf);
    } else {
        out.clear();
    }

    ggml_free(ctx);

    return out;
}

static int64_t n_partial_pools(const std::vector<int32_t> & sel, int64_t off, int64_t width, int64_t kpool) {
    std::map<int32_t, int32_t> cnt;
    for (int64_t i = 0; i < width; ++i) {
        cnt[sel[off + i]/(int32_t) kpool]++;
    }

    int64_t n = 0;
    for (const auto & kv : cnt) {
        n += kv.second != (int32_t) kpool;
    }

    return n;
}

static void test_top_k_boundary() {
    printf("\n--- top-k granularity, standalone ---\n");

    const int64_t kpool   = 4;
    const int64_t top_k   = 2048;        // the reference requires top_k %% kpool == 0
    const int64_t n_kv    = 8192;
    const int64_t n_rows  = 4;           // queries
    const int64_t n_pools = n_kv/kpool;

    // the shape ReLU actually produces: a minority of pools with a positive score,
    // every other pool at EXACTLY 0.0. Fewer positive pools than the budget, which
    // during prefill is the normal state and not a corner case
    const int64_t n_pos = 300;

    std::vector<float> pool_score(n_pools, 0.0f);
    for (int64_t p = 0; p < n_pos; ++p) {
        // deterministic, distinct, strictly positive
        pool_score[(p*7919) % n_pools] = 1.0f + std::fabs(std::sin(0.7f*(float) p))*1000.0f;
    }

    const int64_t select_k = llama_kpool_select_k((uint32_t) n_pools, (uint32_t) top_k, (uint32_t) kpool);
    CHECK(select_k == top_k/kpool, "llama_kpool_select_k is index_topk/index_kpool (%d pools)", (int) select_k);

    ggml_backend_t backend_cpu = ggml_backend_cpu_init();
    ggml_backend_t backend_gpu = nullptr;
    {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev) {
            backend_gpu = ggml_backend_dev_init(dev, nullptr);
        }
    }

    printf("%s  GPU backend for the top-k comparison: %s\n",
            backend_gpu ? "ok  " : "note", backend_gpu ? ggml_backend_name(backend_gpu) : "none, CPU only");

    // t = (q + 1) %% kpool cells of trailing incomplete pool, biased out of the
    // budget. swept so that the per-stream tail residue is covered, not just t == 0
    for (int64_t t = 0; t < kpool; ++t) {
        const int64_t n_tail = t;
        const int64_t n_full = n_kv - n_tail;   // cells belonging to complete pools

        // ---- A: the WRONG design. one score per cell, top-k of width index_topk --
        std::vector<float> s_cell((size_t) n_kv*n_rows);
        // ---- B: what ships. one score per pool, top-k of width index_topk/kpool --
        std::vector<float> s_pool((size_t) n_pools*n_rows);

        for (int64_t r = 0; r < n_rows; ++r) {
            for (int64_t j = 0; j < n_kv; ++j) {
                s_cell[r*n_kv + j] = j >= n_full ? -INFINITY : pool_score[j/kpool];
            }
            for (int64_t p = 0; p < n_pools; ++p) {
                // a pool that the tail bites into is not a candidate at all
                s_pool[r*n_pools + p] = (p + 1)*kpool > n_full ? -INFINITY : pool_score[p];
            }
        }

        const auto sel_cell = run_top_k(backend_cpu, s_cell, n_kv,    n_rows, top_k);
        const auto sel_pool = run_top_k(backend_cpu, s_pool, n_pools, n_rows, select_k);

        // 1. the cell-level form, at a budget that IS a whole number of pools,
        //    still splits pools. this is the measurement the design turns on
        int64_t partial_cell = 0;
        for (int64_t r = 0; r < n_rows; ++r) {
            partial_cell += n_partial_pools(sel_cell, r*top_k, top_k, kpool);
        }
        CHECK(partial_cell > 0,
                "t=%d: a CELL-level top-k at the pool-aligned width %d still leaves %d partial pool(s); "
                "a pool-aligned WIDTH does not make a pool-aligned CUT",
                (int) t, (int) top_k, (int) partial_cell);

        // 2. the pool-level form: expand each selected pool ordinal to its kpool
        //    member cells, exactly as the graph does through pool_cells
        std::vector<int32_t> expanded((size_t) select_k*kpool*n_rows);
        for (int64_t r = 0; r < n_rows; ++r) {
            for (int64_t i = 0; i < select_k; ++i) {
                const int32_t p = sel_pool[r*select_k + i];
                for (int64_t m = 0; m < kpool; ++m) {
                    expanded[r*select_k*kpool + i*kpool + m] = (int32_t) (p*kpool + m);
                }
            }
        }

        int64_t partial_pool = 0;
        int64_t tail_in_pool = 0;
        for (int64_t r = 0; r < n_rows; ++r) {
            partial_pool += n_partial_pools(expanded, r*select_k*kpool, select_k*kpool, kpool);
            for (int64_t i = 0; i < select_k*kpool; ++i) {
                tail_in_pool += expanded[r*select_k*kpool + i] >= n_full;
            }
        }
        CHECK(partial_pool == 0,
                "t=%d: a POOL-level top-k of %d pools expands to only whole pools (%d partial)",
                (int) t, (int) select_k, (int) partial_pool);
        CHECK(tail_in_pool == 0, "t=%d: no tail cell consumes budget (%d did)", (int) t, (int) tail_in_pool);

        // 3. CPU vs CUDA on the same input
        if (backend_gpu) {
            const auto gpu_cell = run_top_k(backend_gpu, s_cell, n_kv,    n_rows, top_k);
            const auto gpu_pool = run_top_k(backend_gpu, s_pool, n_pools, n_rows, select_k);

            // the pool SET may legitimately differ between backends when the cut
            // falls inside a tie group. What may not differ is pool integrity,
            // and that is what is asserted
            int64_t partial_gpu = 0;
            bool ok_gpu = gpu_pool.size() == sel_pool.size();
            for (int64_t r = 0; ok_gpu && r < n_rows; ++r) {
                for (int64_t i = 0; i < select_k; ++i) {
                    const int32_t p = gpu_pool[r*select_k + i];
                    for (int64_t m = 0; m < kpool; ++m) {
                        expanded[r*select_k*kpool + i*kpool + m] = (int32_t) (p*kpool + m);
                    }
                }
                partial_gpu += n_partial_pools(expanded, r*select_k*kpool, select_k*kpool, kpool);
            }
            CHECK(ok_gpu && partial_gpu == 0,
                    "t=%d: %s also expands to only whole pools (%d partial)",
                    (int) t, ggml_backend_name(backend_gpu), (int) partial_gpu);

            bool same_pool = gpu_pool.size() == sel_pool.size();
            for (int64_t r = 0; same_pool && r < n_rows; ++r) {
                std::set<int32_t> a(sel_pool.begin() + r*select_k, sel_pool.begin() + (r + 1)*select_k);
                std::set<int32_t> b(gpu_pool.begin() + r*select_k, gpu_pool.begin() + (r + 1)*select_k);
                same_pool = a == b;
            }
            int64_t partial_gpu_cell = 0;
            for (int64_t r = 0; r < n_rows; ++r) {
                partial_gpu_cell += n_partial_pools(gpu_cell, r*top_k, top_k, kpool);
            }
            // reported, not asserted: which members of a cut pool each backend's
            // partial sort happens to keep is not contractual
            printf("note t=%d: CPU and %s %s on the selected POOL set; the cell-level form leaves "
                   "%d partial pool(s) there too\n",
                    (int) t, ggml_backend_name(backend_gpu),
                    same_pool ? "agree" : "DISAGREE", (int) partial_gpu_cell);
        }
    }

    if (backend_gpu) {
        ggml_backend_free(backend_gpu);
    }
    ggml_backend_free(backend_cpu);
}

//
// the memory object itself
//

struct kpool_tensors {
    ggml_tensor * cell_pool  = nullptr;
    ggml_tensor * pool_cells = nullptr;
    ggml_tensor * bias       = nullptr;
    ggml_tensor * pool_bias  = nullptr;
    ggml_tensor * sel_mask   = nullptr;
    ggml_tensor * cand_mask  = nullptr;
};

// The test asks for all six, including the two the pooled graph does not consume.
// cell_pool and bias are the per-CELL spelling of the same predicate, computed by a
// different loop, and they are what pool_bias and cand_mask are checked against here
static kpool_tensors alloc_kpool_tensors(
        ggml_context * ctx,
        int64_t n_kv,
        int64_t n_tps,
        int64_t n_padq,
        int64_t n_stream,
        int64_t kpool,
        int64_t n_pools) {
    kpool_tensors t;

    t.cell_pool  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv, n_stream);
    t.pool_cells = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, kpool*n_pools, n_stream);
    t.bias       = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_kv, n_tps, n_stream);
    t.pool_bias  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_pools, n_tps, n_stream);
    t.sel_mask   = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_padq, 1, n_stream);
    t.cand_mask  = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, n_kv, n_padq, 1, n_stream);

    ggml_set_input(t.cell_pool);
    ggml_set_input(t.pool_cells);
    ggml_set_input(t.bias);
    ggml_set_input(t.pool_bias);
    ggml_set_input(t.sel_mask);
    ggml_set_input(t.cand_mask);

    return t;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("usage: %s <tiny-glm5next.gguf>\n", argv[0]);
        return 1;
    }

    setvbuf(stdout, nullptr, _IOLBF, 0);

    llama_backend_init();

    test_top_k_boundary();

    printf("\n--- model ---\n");

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (model == nullptr) {
        printf("FAIL: could not load %s\n", argv[1]);
        return 1;
    }

    const auto & hparams = model->hparams;

    printf("n_layer          = %u (+%u nextn)\n", hparams.n_layer(), hparams.n_layer_nextn);
    printf("n_head           = %u\n", hparams.n_head());
    printf("n_embd_head_kda  = %u\n", hparams.n_embd_head_kda);
    printf("ssm_d_conv       = %u\n", hparams.ssm_d_conv);
    printf("n_lora_kv        = %u\n", hparams.n_lora_kv);
    printf("indexer_head_size= %u\n", hparams.indexer_head_size);
    printf("indexer_kpool    = %u\n", hparams.indexer_kpool);
    printf("indexer_top_k    = %u\n", hparams.indexer_top_k);
    printf("n_embd_r()       = %u\n", hparams.n_embd_r());
    printf("n_embd_s()       = %u\n", hparams.n_embd_s());

    // ---- sizing ------------------------------------------------------------
    {
        const uint32_t d_inner = hparams.n_head()*hparams.n_embd_head_kda;

        CHECK(hparams.n_embd_r() == 3*(hparams.ssm_d_conv - 1)*d_inner,
                "n_embd_r == 3 conv states of (d_conv-1)*n_head*head_dim (%u)", hparams.n_embd_r());
        CHECK(hparams.n_embd_s() == hparams.n_embd_head_kda*hparams.n_embd_head_kda*hparams.n_head(),
                "n_embd_s == head_dim*head_dim per head (%u)", hparams.n_embd_s());
        CHECK(hparams.indexer_kpool > 0 && hparams.indexer_top_k % hparams.indexer_kpool == 0,
                "indexer_top_k (%u) is a whole number of pools of %u", hparams.indexer_top_k, hparams.indexer_kpool);
    }

    // ---- a multi-sequence unified cache pools per SEQUENCE ------------------
    //
    // [TAG_KPOOL_SEQ_PARTITION]. pools group cells by position, and a position only
    // names a pool inside one sequence. a unified cache shares one cells array, so the
    // stream's pool table is cut into one run per sequence instead of the cache being
    // refused. -kvu with --parallel is reachable (llama-perplexity forces it for
    // hellaswag / winogrande / multiple-choice, and llama-embedding turns -np 1 into
    // -kvu with n_seq_max 256), so this has to work rather than fail at startup.
    {
        llama_cparams cp = {};
        cp.n_ctx       = 256;
        cp.n_ctx_seq   = 256;
        cp.n_batch     = 32;
        cp.n_ubatch    = 32;
        cp.n_seq_max   = 2;
        cp.kv_unified  = true;
        cp.causal_attn = true;

        llama_memory_params mp = {};
        mp.type_k   = GGML_TYPE_F16;
        mp.type_v   = GGML_TYPE_F16;
        mp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;

        llama_memory_i * raw = nullptr;
        try {
            raw = model->create_memory(mp, cp);
        } catch (const std::exception & e) {
            printf("note create_memory threw: %s\n", e.what());
            raw = nullptr;
        }
        CHECK(raw != nullptr, "-kvu with n_seq_max 2 is accepted: the pool map is per sequence");

        const uint32_t kpool = hparams.indexer_kpool;

        // one shared n_kv/kpool budget plus the rebasing slack per sequence, NOT one
        // full-width table each: the indexer scores every slot against every query, so a
        // full-width table per sequence multiplies the score tensor by the sequence count
        CHECK(llama_kpool_n_pools(256, kpool, 1) == 256/kpool + 2 &&
              llama_kpool_n_pools(256, kpool, 2) == 256/kpool + 4,
                "llama_kpool_n_pools is n_kv/kpool + 2 per sequence (%u slots for 2 sequences)",
                llama_kpool_n_pools(256, kpool, 2));

        auto * m2 = dynamic_cast<llama_memory_hybrid *>(raw);
        CHECK(m2 != nullptr && m2->get_mem_idx() != nullptr, "-kvu builds an indexer cache too");

        // drive one ubatch holding BOTH sequences through the map. 16 positions each, so
        // every sequence owns several complete pools and the runs have to be told apart
        if (m2) {
            const int64_t n_tok = 16;

            std::vector<llama_token>    tok;
            std::vector<llama_pos>      pos;
            std::vector<llama_seq_id>   sid;
            std::vector<llama_seq_id *> sptr;
            std::vector<int32_t>        nsid;
            std::vector<int8_t>         lg;

            for (llama_seq_id s = 0; s < 2; ++s) {
                for (int64_t i = 0; i < n_tok; ++i) {
                    tok.push_back(0);
                    pos.push_back((llama_pos) i);
                    sid.push_back(s);
                    nsid.push_back(1);
                    lg.push_back(1);
                }
            }
            for (size_t i = 0; i < sid.size(); ++i) {
                sptr.push_back(&sid[i]);
            }

            llama_batch b = {};
            b.n_tokens = (int32_t) tok.size();
            b.token    = tok.data();
            b.pos      = pos.data();
            b.seq_id   = sptr.data();
            b.n_seq_id = nsid.data();
            b.logits   = lg.data();

            llama_batch_allocr ba(hparams.n_pos_per_embd());
            if (ba.init(b, model->vocab, nullptr, hparams.n_embd_inp(), cp.n_seq_max, true)) {
                auto c  = m2->init_batch(ba, cp.n_ubatch, false);
                auto * mc = dynamic_cast<llama_memory_hybrid_context *>(c.get());

                CHECK(mc && mc->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
                        "both sequences fit one unified ubatch");

                if (mc && mc->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
                    mc->apply();

                    const llama_ubatch & ub = mc->get_ubatch();

                    const int64_t n_kv     = mc->get_attn()->get_n_kv();
                    const int64_t n_stream = mc->get_attn()->get_n_stream();
                    const int64_t n_tps    = ub.n_tokens/n_stream;
                    const int64_t n_pools  = llama_kpool_n_pools((uint32_t) n_kv, kpool, ub.n_seqs_unq);

                    CHECK(n_stream == 1 && ub.n_seqs_unq == 2,
                            "a unified cache carries both sequences in one stream (n_seqs_unq %u)",
                            ub.n_seqs_unq);

                    ggml_init_params gp = { ggml_tensor_overhead()*16, nullptr, true };
                    ggml_context * ctx = ggml_init(gp);

                    kpool_tensors kt = alloc_kpool_tensors(ctx, n_kv, n_tps, n_tps, n_stream, kpool, n_pools);

                    ggml_backend_t        backend = ggml_backend_cpu_init();
                    ggml_backend_buffer_t buf     = ggml_backend_alloc_ctx_tensors(ctx, backend);

                    if (buf) {
                        // no cell_pool: the per-cell view has one row per stream, and a
                        // cell two sequences share has nowhere to put its second pool
                        llama_kv_cache_set_input_kpool(m2->get_mem_attn(),
                                /* cell_pool */ nullptr, kt.pool_cells, kt.bias, kt.pool_bias,
                                kt.sel_mask, kt.cand_mask, &ub, kpool);

                        const int32_t * pc = (const int32_t *) kt.pool_cells->data;
                        const float   * pb = (const float   *) kt.pool_bias->data;

                        const auto & cells = m2->get_mem_attn()->get_cells(0);

                        // the invariant the partitioning exists for: a pool the query may
                        // spend budget on holds only that query's own visible cells. this
                        // is what a shared cells array breaks if the map is per stream
                        bool     own      = true;
                        int64_t  n_finite = 0;
                        std::map<int64_t, int> seq_mask_of_slot;

                        for (int64_t ii = 0; ii < n_tps; ++ii) {
                            const llama_seq_id sq = ub.seq_id[ii][0];
                            const llama_pos    q  = ub.pos[ii];

                            for (int64_t p = 0; p < n_pools; ++p) {
                                if (pb[ii*n_pools + p] != 0.0f) {
                                    continue;
                                }

                                n_finite++;
                                seq_mask_of_slot[p] |= 1 << sq;

                                for (int64_t m = 0; m < (int64_t) kpool; ++m) {
                                    const int32_t c = pc[p*kpool + m];
                                    own &= c >= 0 && c < n_kv && !cells.is_empty(c) &&
                                           cells.seq_has(c, sq) && cells.pos_get(c) <= q;
                                }
                            }
                        }

                        CHECK(own && n_finite > 0,
                                "every pool a query may select holds only that query's own visible cells (%d selectable (query, pool) pairs)",
                                (int) n_finite);

                        int mask_all = 0;
                        bool disjoint = true;
                        for (const auto & kv : seq_mask_of_slot) {
                            mask_all |= kv.second;
                            disjoint &= (kv.second & (kv.second - 1)) == 0;
                        }

                        CHECK(disjoint, "the two sequences get disjoint pool runs (%d slots in use of %d)",
                                (int) seq_mask_of_slot.size(), (int) n_pools);
                        CHECK(mask_all == 0x3, "both sequences own selectable pools, so neither run is empty");

                        ggml_backend_buffer_free(buf);
                    }

                    ggml_backend_free(backend);
                    ggml_free(ctx);
                }
            }
        }

        delete raw;

        // one sequence in flight stays the simple case: the shared cells array holds
        // only its keys, and the map filters on seq_has anyway
        cp.n_seq_max = 1;
        llama_memory_i * raw1 = nullptr;
        try {
            raw1 = model->create_memory(mp, cp);
        } catch (const std::exception &) {
            raw1 = nullptr;
        }
        CHECK(raw1 != nullptr, "-kvu with a single sequence is still allowed");
        delete raw1;
    }

    // ---- the indexer cache keeps its own dtype ------------------------------
    {
        llama_cparams cp = {};
        cp.n_ctx       = 256;
        cp.n_ctx_seq   = 256;
        cp.n_batch     = 32;
        cp.n_ubatch    = 32;
        cp.n_seq_max   = 1;
        cp.kv_unified  = false;
        cp.causal_attn = true;

        llama_memory_params mp = {};
        mp.type_k   = GGML_TYPE_Q8_0;
        mp.type_v   = GGML_TYPE_Q8_0;
        mp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;

        llama_memory_i * raw = model->create_memory(mp, cp);
        auto * m = dynamic_cast<llama_memory_hybrid *>(raw);

        CHECK(m != nullptr && m->get_mem_idx() != nullptr, "-ctk q8_0 still builds an indexer cache");
        if (m && m->get_mem_idx()) {
            CHECK(!ggml_is_quantized(m->get_mem_idx()->type_k()),
                    "indexer cache stays %s under -ctk q8_0: it also holds the compressor gates",
                    ggml_type_name(m->get_mem_idx()->type_k()));
            CHECK(ggml_is_quantized(m->get_mem_attn()->type_k()),
                    "the MLA cache still honours -ctk q8_0 (%s)", ggml_type_name(m->get_mem_attn()->type_k()));
        }

        delete raw;
    }

    // ---- create the memory -------------------------------------------------
    llama_cparams cparams = {};
    cparams.n_ctx        = 512;
    cparams.n_ctx_seq    = 512;
    cparams.n_batch      = 32;
    cparams.n_ubatch     = 32;
    cparams.n_seq_max    = 2;
    cparams.n_rs_seq     = 0;
    cparams.kv_unified   = false;   // one sequence per stream, the n_ps == 1 layout
    cparams.offload_kqv  = false;
    cparams.flash_attn   = false;
    cparams.causal_attn  = true;

    llama_memory_params mparams_mem = {};
    mparams_mem.type_k    = GGML_TYPE_F16;
    mparams_mem.type_v    = GGML_TYPE_F16;
    mparams_mem.ctx_type  = LLAMA_CONTEXT_TYPE_DEFAULT;
    mparams_mem.swa_full  = false;

    llama_memory_i * mem_raw = model->create_memory(mparams_mem, cparams);
    CHECK(mem_raw != nullptr, "create_memory returned a memory object");
    if (!mem_raw) {
        return 1;
    }

    auto * mem = dynamic_cast<llama_memory_hybrid *>(mem_raw);
    CHECK(mem != nullptr, "the memory is a llama_memory_hybrid");
    if (!mem) {
        return 1;
    }

    llama_kv_cache          * kv_attn = mem->get_mem_attn();
    llama_memory_recurrent  * rs      = mem->get_mem_recr();
    llama_kv_cache          * kv_idx  = mem->get_mem_idx();

    CHECK(kv_attn != nullptr, "hybrid holds an attention (MLA) cache");
    CHECK(rs      != nullptr, "hybrid holds a recurrent (KDA) cache");
    CHECK(kv_idx  != nullptr, "hybrid holds an indexer key cache");
    if (!kv_attn || !rs || !kv_idx) {
        return 1;
    }

    // ---- layer partition ---------------------------------------------------
    {
        const auto ids_attn = kv_attn->get_layer_ids();
        const auto ids_idx  = kv_idx ->get_layer_ids();

        uint32_t n_dsa = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            n_dsa += !hparams.is_recr(il);
        }

        CHECK(ids_attn.size() == n_dsa, "MLA cache holds the %u DSA trunk layers (%zu)", n_dsa, ids_attn.size());
        CHECK(ids_idx.size()  == n_dsa, "indexer cache holds the same %u layers (%zu)",   n_dsa, ids_idx.size());

        bool same = ids_attn.size() == ids_idx.size();
        for (size_t i = 0; same && i < ids_attn.size(); ++i) {
            same = ids_attn[i] == ids_idx[i];
        }
        CHECK(same, "MLA and indexer caches cover exactly the same layers");

        for (uint32_t il : ids_attn) {
            CHECK(!hparams.is_recr(il), "cached attention layer %u is not recurrent", il);
        }
    }

    // ---- cache geometry ----------------------------------------------------
    {
        const uint32_t il_dsa = kv_attn->get_layer_ids().front();

        ggml_tensor * k_mla = kv_attn->get_k_storage(il_dsa);
        ggml_tensor * k_idx = kv_idx ->get_k_storage(il_dsa);

        CHECK(k_mla != nullptr && k_idx != nullptr, "both caches expose K storage for layer %u", il_dsa);

        // nope-only MLA: the latent row is kv_lora_rank + qk_rope_head_dim
        CHECK(k_mla->ne[0] == (int64_t) hparams.n_embd_head_k(il_dsa),
                "MLA latent row = %d (n_embd_head_k = %u)", (int) k_mla->ne[0], hparams.n_embd_head_k(il_dsa));

        // the pooling indexer caches the key AND the compressor gate score
        CHECK(k_idx->ne[0] == (int64_t) (2*hparams.indexer_head_size),
                "indexer row = %d (expected 2 x indexer_head_size = %u)",
                (int) k_idx->ne[0], 2*hparams.indexer_head_size);

        CHECK(k_mla->ne[1] == k_idx->ne[1], "MLA and indexer caches have the same cell count (%d)", (int) k_mla->ne[1]);
        CHECK(k_mla->ne[2] == k_idx->ne[2], "MLA and indexer caches have the same stream count (%d)", (int) k_mla->ne[2]);

        // is_mla() stays true for the indexer hparams copy, so no V is allocated
        {
            size_t bytes = 0;
            for (const auto & b : kv_idx->memory_breakdown()) {
                bytes += b.second;
            }

            const size_t k_only = (size_t) ggml_nbytes(k_idx)*kv_idx->get_layer_ids().size();

            // the byte count is the observable: K only, no V
            CHECK(hparams.is_mla() && bytes == k_only,
                    "indexer cache allocates K and no V: %zu bytes for %zu layers",
                    bytes, kv_idx->get_layer_ids().size());
        }
    }

    // ---- recurrent geometry ------------------------------------------------
    {
        uint32_t n_kda = 0;
        for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
            n_kda += hparams.is_recr(il);
        }

        CHECK(rs->size >= cparams.n_seq_max, "recurrent cache has >= n_seq_max slots (%u)", rs->size);
        CHECK(n_kda > 0, "the model has %u KDA layers", n_kda);
    }

    // ---- drive a batch through it and reach all three halves in one graph ---
    {
        const int32_t n_tokens = 10;   // pos 9 leaves (9+1) %% kpool = 2 tail cells

        std::vector<llama_token> tokens(n_tokens*cparams.n_seq_max, 0);
        std::vector<llama_pos>   pos;
        std::vector<llama_seq_id> seqs;
        for (uint32_t s = 0; s < cparams.n_seq_max; ++s) {
            for (int32_t i = 0; i < n_tokens; ++i) {
                pos.push_back(i);
                seqs.push_back((llama_seq_id) s);
            }
        }

        llama_batch batch = {};
        batch.n_tokens = n_tokens*(int32_t) cparams.n_seq_max;
        batch.token    = tokens.data();
        batch.pos      = pos.data();

        std::vector<llama_seq_id *> seq_ptrs(batch.n_tokens);
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            seq_ptrs[i] = &seqs[i];
        }
        std::vector<int32_t> n_seq_id(batch.n_tokens, 1);
        std::vector<int8_t>  logits(batch.n_tokens, 0);
        logits.back() = 1;

        batch.seq_id   = seq_ptrs.data();
        batch.n_seq_id = n_seq_id.data();
        batch.logits   = logits.data();

        llama_batch_allocr balloc(hparams.n_pos_per_embd());
        const bool ok = balloc.init(batch, model->vocab, nullptr, hparams.n_embd_inp(),
                                    cparams.n_seq_max, true);
        CHECK(ok, "batch allocr accepted a %d-token, %u-sequence batch", batch.n_tokens, cparams.n_seq_max);

        auto mctx_ptr = mem->init_batch(balloc, cparams.n_ubatch, false);
        CHECK(mctx_ptr != nullptr && mctx_ptr->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
                "init_batch produced a usable memory context");

        auto * mctx = dynamic_cast<llama_memory_hybrid_context *>(mctx_ptr.get());
        CHECK(mctx != nullptr, "the context is a llama_memory_hybrid_context");

        if (mctx && mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
            const auto * ctx_attn = mctx->get_attn();
            const auto * ctx_recr = mctx->get_recr();
            const auto * ctx_idx  = mctx->get_idx();

            CHECK(ctx_attn != nullptr, "context exposes the attention half");
            CHECK(ctx_recr != nullptr, "context exposes the recurrent half");
            CHECK(ctx_idx  != nullptr, "context exposes the indexer half");

            // n_kv is only valid once applied; apply() also asserts the caches agree
            mctx->apply();

            // apply() asserts both itself, so reaching this line is the result; printed
            // rather than CHECKed so the count stays honest
            printf("note indexer and attention caches agree on n_kv (%u) and n_stream (%u)\n",
                    ctx_idx->get_n_kv(), ctx_idx->get_n_stream());
            CHECK(ctx_idx && ctx_idx->get_kv() == kv_idx, "the indexer context is a view of the indexer cache");

            ggml_init_params gparams = {
                /* .mem_size   */ ggml_tensor_overhead()*1024 + ggml_graph_overhead(),
                /* .mem_buffer */ nullptr,
                /* .no_alloc   */ true,
            };

            ggml_context * ctx0 = ggml_init(gparams);
            ggml_cgraph  * gf   = ggml_new_graph(ctx0);

            const uint32_t il_dsa = kv_attn->get_layer_ids().front();

            uint32_t il_kda = 0;
            while (il_kda < hparams.n_layer() && !hparams.is_recr(il_kda)) {
                il_kda++;
            }

            const int64_t n_kv     = ctx_attn->get_n_kv();
            const int64_t n_stream = ctx_attn->get_n_stream();
            const int64_t n_tps    = mctx->get_ubatch().n_tokens/n_stream;

            // 1. MLA latent K
            ggml_tensor * k_mla = ctx_attn->get_k(ctx0, il_dsa);
            CHECK(k_mla != nullptr, "graph reaches the MLA latent cache: [%d, %d, %d, %d]",
                    (int) k_mla->ne[0], (int) k_mla->ne[1], (int) k_mla->ne[2], (int) k_mla->ne[3]);

            // 2. indexer keys, split into the key half and the gate half
            ggml_tensor * k_idx_all = ctx_idx->get_k(ctx0, il_dsa);
            CHECK(k_idx_all->ne[1] == 2, "indexer cache view has 2 heads (key | compressor gate)");

            const int64_t d_idx = hparams.indexer_head_size;

            ggml_tensor * k_idx_v = ggml_view_3d(ctx0, k_idx_all, d_idx, n_kv, n_stream,
                    k_idx_all->nb[2], k_idx_all->nb[3], 0);
            ggml_tensor * g_idx_v = ggml_view_3d(ctx0, k_idx_all, d_idx, n_kv, n_stream,
                    k_idx_all->nb[2], k_idx_all->nb[3], k_idx_all->nb[1]);

            // 3. KDA recurrent + conv state
            ggml_tensor * r_kda = ctx_recr->get_r_l(il_kda);
            ggml_tensor * s_kda = ctx_recr->get_s_l(il_kda);
            CHECK(r_kda != nullptr && s_kda != nullptr,
                    "graph reaches the KDA conv state [%d x %d] and recurrent state [%d x %d]",
                    (int) r_kda->ne[0], (int) r_kda->ne[1], (int) s_kda->ne[0], (int) s_kda->ne[1]);

            // 4. host-side pool map
            const int64_t kpool   = hparams.indexer_kpool;
            const int64_t n_pools = llama_kpool_n_pools((uint32_t) n_kv, (uint32_t) kpool);
            const int64_t n_padq  = n_tps;   // this tree does not pad the KQ mask

            kpool_tensors kt = alloc_kpool_tensors(ctx0, n_kv, n_tps, n_padq, n_stream, kpool, n_pools);

            // shape of the real thing: gather pool members, mix, score, broadcast the
            // pool score back onto its member cells, top-k
            ggml_tensor * members = ggml_get_rows(ctx0, k_idx_v, kt.pool_cells);
            ggml_tensor * gates   = ggml_get_rows(ctx0, g_idx_v, kt.pool_cells);
            members = ggml_reshape_4d(ctx0, members, d_idx, kpool, n_pools, n_stream);
            gates   = ggml_reshape_4d(ctx0, gates,   d_idx, kpool, n_pools, n_stream);

            // stand-in for softmax(gate + ape) * member, summed over kpool
            ggml_tensor * pooled = ggml_mul(ctx0, members, gates);
            pooled = ggml_reshape_3d(ctx0, pooled, d_idx*kpool, n_pools, n_stream);
            // stand-in for the q . k_pool score: one number per (pool, query)
            ggml_tensor * score = ggml_mul_mat(ctx0,
                    ggml_cont(ctx0, ggml_view_3d(ctx0, pooled, d_idx, n_pools, n_stream,
                            pooled->nb[1], pooled->nb[2], 0)),
                    ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, d_idx, n_tps, n_stream));

            // mask the pools the reference rejects, then top-k over POOLS
            score = ggml_add(ctx0, ggml_cont(ctx0, score), kt.pool_bias);

            const int64_t select_k = llama_kpool_select_k((uint32_t) n_pools, hparams.indexer_top_k, (uint32_t) kpool);
            ggml_tensor * sel = ggml_cont(ctx0, ggml_top_k(ctx0, score, (int) select_k));

            // expand each selected POOL ordinal into its kpool member CELLS, which is
            // the reference's selected_indices = pool_indices[batch_idx, selected].
            // the query axis folds into the gather's row axis so that one get_rows
            // serves every query, while the stream axis stays where get_rows wants it
            ggml_tensor * pc3      = ggml_reshape_3d(ctx0, kt.pool_cells, kpool, n_pools, n_stream);
            ggml_tensor * sel_flat = ggml_reshape_2d(ctx0, sel, select_k*n_tps, n_stream);

            const int64_t width = kpool*select_k;
            ggml_tensor * top_k = ggml_reshape_3d(ctx0,
                    ggml_get_rows(ctx0, pc3, sel_flat), width, n_tps, n_stream);

            CHECK(top_k->type == GGML_TYPE_I32, "the pool -> cell expansion stays I32 (pool_cells is I32)");
            printf("note top-k over %d POOL scores expands to %d I32 CELL indices per query\n",
                    (int) select_k, (int) width);

            // 5. the scatter the mask is built from, starting at sel_mask rather than
            //    an all -INFINITY fill, which is what forces the tail in
            ggml_tensor * top_k_4d = ggml_reshape_4d(ctx0, top_k, width, n_tps, 1, n_stream);
            ggml_tensor * base = ggml_view_4d(ctx0, kt.sel_mask, 1, n_kv, n_padq, n_stream,
                    kt.sel_mask->nb[0], kt.sel_mask->nb[1], kt.sel_mask->nb[2], 0);
            ggml_tensor * idxs = ggml_view_4d(ctx0, top_k_4d, width, n_tps, n_stream, 1,
                    top_k_4d->nb[1], top_k_4d->nb[2], n_stream*top_k_4d->nb[3], 0);
            ggml_tensor * zeros = ggml_fill(ctx0,
                    ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, width, n_tps, n_stream), 0.0f);
            ggml_tensor * mask = ggml_set_rows(ctx0, base, zeros, idxs);

            ggml_build_forward_expand(gf, k_mla);
            ggml_build_forward_expand(gf, r_kda);
            ggml_build_forward_expand(gf, s_kda);
            ggml_build_forward_expand(gf, mask);

            printf("note one graph reaches all three halves in %d nodes\n", ggml_graph_n_nodes(gf));

            // 6. fill the pool map on a real allocated buffer
            ggml_backend_t         backend = ggml_backend_cpu_init();
            ggml_backend_buffer_t  buf     = ggml_backend_alloc_ctx_tensors(ctx0, backend);
            CHECK(buf != nullptr, "allocated the graph's input tensors on the CPU backend");

            if (buf) {
                const llama_ubatch & ub = mctx->get_ubatch();
                llama_kv_cache_set_input_kpool(kv_attn, kt.cell_pool, kt.pool_cells, kt.bias, kt.pool_bias, kt.sel_mask, kt.cand_mask,
                        &ub, (uint32_t) kpool);

                const int32_t * cp = (const int32_t *) kt.cell_pool->data;
                const int32_t * pc = (const int32_t *) kt.pool_cells->data;
                const float   * bi = (const float   *) kt.bias->data;
                const float   * sm = (const float   *) kt.sel_mask->data;

                bool in_range = true;
                for (int64_t i = 0; i < ggml_nelements(kt.cell_pool); ++i) {
                    in_range &= cp[i] >= 0 && cp[i] < n_pools;
                }
                for (int64_t i = 0; i < ggml_nelements(kt.pool_cells); ++i) {
                    in_range &= pc[i] >= 0 && pc[i] < n_kv;
                }
                CHECK(in_range, "every emitted index is non-negative and in range (ggml_set_rows asserts i1 >= 0)");

                bool finite = true;
                for (int64_t i = 0; i < ggml_nelements(kt.bias); ++i) {
                    finite &= bi[i] == 0.0f || bi[i] == -INFINITY;
                }
                for (int64_t i = 0; i < ggml_nelements(kt.sel_mask); ++i) {
                    finite &= sm[i] == 0.0f || sm[i] == -INFINITY;
                }
                for (int64_t i = 0; i < ggml_nelements(kt.pool_bias); ++i) {
                    const float * pb = (const float *) kt.pool_bias->data;
                    finite &= pb[i] == 0.0f || pb[i] == -INFINITY;
                }
                for (int64_t i = 0; i < ggml_nelements(kt.cand_mask); ++i) {
                    const float * cm = (const float *) kt.cand_mask->data;
                    finite &= cm[i] == 0.0f || cm[i] == -INFINITY;
                }
                CHECK(finite, "bias, pool_bias, sel_mask and cand_mask hold only 0 and -INFINITY: "
                        "no +1e9 to meet a -inf");

                // cand_mask is exactly max(bias, sel_mask) lifted to KQ shape, and it
                // is what stops an over-budget top-k from escaping the reference's
                // candidate set.
                //
                // ggml_top_k always returns select_k pool ordinals even when fewer
                // than select_k pools are finite. During prefill the query at
                // position q has only ~q/kpool complete visible pools against a
                // budget of index_topk/kpool, so the budget spills into -INFINITY
                // pools and picks among them arbitrarily. Adding the causal mask
                // kills the expansions that are empty or in the future. What it does
                // not kill is a resident, causally visible cell that sits in an
                // INCOMPLETE pool below the tail: the reference never selects it
                // (pool_valid = grouped_valid_keys.all(-1)), the graph would.
                // Unreachable while positions are contiguous, reachable the moment a
                // partial seq_rm leaves a hole.
                {
                    const float * cm = (const float *) kt.cand_mask->data;

                    bool    is_union = true;
                    int64_t n_spill  = 0;   // candidate pools strictly fewer than the budget

                    const int64_t select_k =
                        llama_kpool_select_k((uint32_t) n_pools, hparams.indexer_top_k, (uint32_t) kpool);

                    for (int64_t s = 0; s < n_stream; ++s) {
                        for (int64_t ii = 0; ii < n_tps; ++ii) {
                            const float * rb = bi + (s*n_tps  + ii)*n_kv;
                            const float * rs = sm + (s*n_padq + ii)*n_kv;
                            const float * rc = cm + (s*n_padq + ii)*n_kv;

                            for (int64_t j = 0; j < n_kv; ++j) {
                                is_union &= rc[j] == std::max(rb[j], rs[j]);
                            }

                            const float * rp = (const float *) kt.pool_bias->data + (s*n_tps + ii)*n_pools;
                            int64_t n_cand = 0;
                            for (int64_t p = 0; p < n_pools; ++p) {
                                n_cand += rp[p] == 0.0f;
                            }
                            n_spill += n_cand < select_k;
                        }
                    }

                    CHECK(is_union, "cand_mask == max(bias, sel_mask): the reference's candidate set");

                    // not a failure: it is the normal prefill state, and the whole
                    // reason the gate has to exist. Printed so that a future change
                    // making it zero cannot quietly turn the check above vacuous
                    printf("note %d of %d (query, stream) rows have fewer candidate pools than the\n"
                           "     top-k budget of %d, so the budget spills and cand_mask is load bearing\n",
                            (int) n_spill, (int) (n_stream*n_tps), (int) select_k);
                }

                // pool_bias is the same predicate as bias, evaluated where the
                // reference evaluates it. For a COMPLETE pool the two must agree
                // cell for cell; for an incomplete or absent pool bias has no cell
                // to speak for it, which is exactly why pool_bias is computed here
                // rather than gathered from bias at each pool's last member
                {
                    const float * pb = (const float *) kt.pool_bias->data;

                    bool agree = true;
                    bool exact = true;

                    for (int64_t s = 0; s < n_stream; ++s) {
                        for (int64_t ii = 0; ii < n_tps; ++ii) {
                            const float * rb = bi + (s*n_tps + ii)*n_kv;
                            const float * rp = pb + (s*n_tps + ii)*n_pools;

                            std::set<int32_t> pools_of_scored_cells;

                            for (int64_t j = 0; j < n_kv; ++j) {
                                if (rb[j] == 0.0f) {
                                    // a scored cell's pool must be a candidate pool
                                    agree &= rp[cp[s*n_kv + j]] == 0.0f;
                                    pools_of_scored_cells.insert(cp[s*n_kv + j]);
                                }
                            }

                            int64_t n_cand = 0;
                            for (int64_t p = 0; p < n_pools; ++p) {
                                n_cand += rp[p] == 0.0f;
                            }

                            // and nothing else may be one. an entirely absent pool
                            // would pass the first check vacuously; this is what
                            // catches it, and it is exactly the failure mode of
                            // gathering pool_bias from bias at each pool's last
                            // member instead of computing it
                            exact &= n_cand == (int64_t) pools_of_scored_cells.size();
                        }
                    }

                    CHECK(agree, "every cell bias scores lies in a pool pool_bias also accepts");
                    CHECK(exact, "and pool_bias accepts no pool that has no scored cell "
                            "(an absent pool must not become a candidate)");
                }

                // per query: (q+1) %% kpool cells sit in its own incomplete pool and
                // must be forced in, the q+1-minus-that below must be scored, and
                // nothing else either. both streams, to cover the per-stream strides
                {
                    const llama_ubatch & u = mctx->get_ubatch();

                    for (int64_t s = 0; s < n_stream; ++s) {
                        bool ok_tail   = true;
                        bool ok_scored = true;

                        for (int64_t ii = 0; ii < n_tps; ++ii) {
                            const llama_pos q = u.pos[s*n_tps + ii];
                            const int64_t   t = (q + 1) % kpool;

                            const float * row_s = sm + (s*n_padq + ii)*n_kv;
                            const float * row_b = bi + (s*n_tps  + ii)*n_kv;

                            int64_t n_forced = 0;
                            int64_t n_scored = 0;
                            for (int64_t j = 0; j < n_kv; ++j) {
                                n_forced += row_s[j] == 0.0f;
                                n_scored += row_b[j] == 0.0f;
                            }

                            ok_tail   &= n_forced == t;
                            ok_scored &= n_scored == (q + 1) - t;
                        }

                        CHECK(ok_tail,   "stream %d: every query forces in exactly its (q+1) %% %d tail cells",
                                (int) s, (int) kpool);
                        CHECK(ok_scored, "stream %d: and scores exactly the cells below that tail",
                                (int) s);
                    }
                }

                // while the window starts at position 0, pos/kpool and the reference's
                // first-resident-key anchor are the same grouping, so the map must
                // reproduce it. incomplete pools carry slot 0 and are masked instead
                {
                    const llama_ubatch & u = mctx->get_ubatch();
                    bool agree = true;
                    for (int64_t s = 0; s < n_stream; ++s) {
                        const llama_seq_id seq = u.seq_id[s*n_tps][0];
                        const auto & cells = kv_attn->get_cells(seq);

                        std::map<int64_t, int> members;
                        llama_pos p_min = -1;
                        for (int64_t j = 0; j < n_kv; ++j) {
                            if (cells.is_empty(j)) {
                                continue;
                            }
                            members[cells.pos_get(j)/kpool]++;
                            if (p_min < 0 || cells.pos_get(j) < p_min) {
                                p_min = cells.pos_get(j);
                            }
                        }
                        agree &= p_min == 0;   // HF's first_key would also be 0

                        for (int64_t j = 0; j < n_kv && agree; ++j) {
                            if (cells.is_empty(j)) {
                                continue;
                            }
                            const int64_t b = cells.pos_get(j)/kpool;
                            agree &= cp[s*n_kv + j] == (int32_t) (members[b] == kpool ? b : 0);
                        }
                    }
                    CHECK(agree, "pool ordinals match the reference anchor while the window starts at position 0");
                }

                {
                    ggml_status st = ggml_backend_graph_compute(backend, gf);
                    CHECK(st == GGML_STATUS_SUCCESS, "the pooled-indexer-shaped graph computes (%d)", (int) st);
                }

                ggml_backend_buffer_free(buf);
            }

            ggml_backend_free(backend);
            ggml_free(ctx0);
        }
    }

    // ---- pool origin across a front eviction --------------------------------
    //
    // the reference anchors a pool at the first *resident* key (valid_keys.argmax(-1));
    // this port anchors at pos/kpool, as vLLM and SGLang do. same grouping until the
    // window front is dropped by a non-multiple of kpool, when the reference regroups
    // every surviving key and this port does not. a cache cannot afford regrouping: the
    // pooled key a decode step scores was built during prefill.
    {
        printf("\n--- pool origin ---\n");

        // seq 0 holds positions 0..9. drop 0..1, not a multiple of kpool = 4
        const llama_pos n_drop = 2;
        mem->seq_rm(0, 0, n_drop);

        std::vector<llama_token>    tok(1, 0);
        std::vector<llama_pos>      pos(1, 10);
        std::vector<llama_seq_id>   sid(1, 0);
        std::vector<llama_seq_id *> sptr(1, sid.data());
        std::vector<int32_t>        nsid(1, 1);
        std::vector<int8_t>         lg(1, 1);

        llama_batch b = {};
        b.n_tokens = 1;
        b.token    = tok.data();
        b.pos      = pos.data();
        b.seq_id   = sptr.data();
        b.n_seq_id = nsid.data();
        b.logits   = lg.data();

        llama_batch_allocr ba(hparams.n_pos_per_embd());
        if (ba.init(b, model->vocab, nullptr, hparams.n_embd_inp(), cparams.n_seq_max, true)) {
            auto c = mem->init_batch(ba, cparams.n_ubatch, false);
            auto * mc = dynamic_cast<llama_memory_hybrid_context *>(c.get());

            CHECK(mc && mc->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
                    "one more token fits after the eviction");

            if (mc && mc->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
                mc->apply();

                const int64_t n_kv    = mc->get_attn()->get_n_kv();
                const int64_t kpool   = hparams.indexer_kpool;
                const int64_t n_pools = llama_kpool_n_pools((uint32_t) n_kv, (uint32_t) kpool);

                ggml_init_params gp = { ggml_tensor_overhead()*16, nullptr, true };
                ggml_context * ctx = ggml_init(gp);

                kpool_tensors kt = alloc_kpool_tensors(ctx, n_kv, 1, 1, 1, kpool, n_pools);

                ggml_backend_t        backend = ggml_backend_cpu_init();
                ggml_backend_buffer_t buf     = ggml_backend_alloc_ctx_tensors(ctx, backend);

                if (buf) {
                    llama_kv_cache_set_input_kpool(kv_attn, kt.cell_pool, kt.pool_cells, kt.bias, kt.pool_bias, kt.sel_mask, kt.cand_mask,
                            &mc->get_ubatch(), (uint32_t) kpool);

                    const int32_t * cp = (const int32_t *) kt.cell_pool->data;
                    const auto & cells = kv_attn->get_cells(0);

                    // positions 4..7 must stay one pool, or the pooled key built during
                    // prefill no longer describes the cells it is scored against
                    std::map<llama_pos, int32_t> slot_of;
                    for (int64_t j = 0; j < n_kv; ++j) {
                        if (!cells.is_empty(j) && cells.seq_has(j, 0)) {
                            slot_of[cells.pos_get(j)] = cp[j];
                        }
                    }

                    const bool grouped =
                        slot_of.count(4) && slot_of.count(5) && slot_of.count(6) && slot_of.count(7) &&
                        slot_of[4] == slot_of[5] && slot_of[5] == slot_of[6] && slot_of[6] == slot_of[7];
                    CHECK(grouped, "positions 4..7 stay one pool after the eviction (slot %d)",
                            grouped ? (int) slot_of[4] : -1);

                    // the reference's anchor would make 2..5 the first pool; this port
                    // deliberately differs, so the difference is pinned rather than fixed
                    const bool differs = !slot_of.count(2) || slot_of[2] != slot_of[4];
                    CHECK(differs, "positions 2 and 4 are NOT pooled together, where the reference anchor would");

                    // the leading remnant 2..3 is an incomplete pool and not in the
                    // query's tail, the one case where a visible cell is neither
                    const float * bi = (const float *) kt.bias->data;
                    const float * sm = (const float *) kt.sel_mask->data;
                    int64_t n_orphan = 0;
                    for (int64_t j = 0; j < n_kv; ++j) {
                        if (cells.is_empty(j) || !cells.seq_has(j, 0)) {
                            continue;
                        }
                        n_orphan += bi[j] == -INFINITY && sm[j] == -INFINITY;
                    }
                    CHECK(n_orphan == 2, "the %d-cell leading remnant is neither pooled nor tail (%d)",
                            (int) n_drop, (int) n_orphan);

                    ggml_backend_buffer_free(buf);
                }

                ggml_backend_free(backend);
                ggml_free(ctx);
            }
        }
    }

    delete mem_raw;

    // ---- the shared input: one fill per ubatch, not one per DSA layer -------
    {
        printf("\n--- shared input ---\n");

        llama_cparams cp = {};
        cp.n_ctx       = 16384;
        cp.n_ctx_seq   = 16384;
        cp.n_batch     = 512;
        cp.n_ubatch    = 512;
        cp.n_seq_max   = 1;
        cp.kv_unified  = false;
        cp.causal_attn = true;

        llama_memory_params mp = {};
        mp.type_k   = GGML_TYPE_F16;
        mp.type_v   = GGML_TYPE_F16;
        mp.ctx_type = LLAMA_CONTEXT_TYPE_DEFAULT;

        llama_memory_i * raw = model->create_memory(mp, cp);
        auto * m = dynamic_cast<llama_memory_hybrid *>(raw);

        if (m) {
            llama_kv_cache * kv = m->get_mem_attn();

            const int64_t n_step   = cp.n_ubatch;
            const int64_t n_fill   = cp.n_ctx_seq - n_step;
            const uint32_t n_dsa   = (uint32_t) kv->get_layer_ids().size();
            const int64_t  kpool   = hparams.indexer_kpool;

            std::vector<llama_token>    tok(n_step, 0);
            std::vector<llama_pos>      pos(n_step);
            std::vector<llama_seq_id>   sid(n_step, 0);
            std::vector<llama_seq_id *> sptr(n_step);
            std::vector<int32_t>        nsid(n_step, 1);
            std::vector<int8_t>         lg(n_step, 0);

            for (int64_t i = 0; i < n_step; ++i) {
                sptr[i] = &sid[i];
            }
            lg.back() = 1;

            llama_memory_context_ptr keep;

            for (int64_t base = 0; base <= n_fill; base += n_step) {
                for (int64_t i = 0; i < n_step; ++i) {
                    pos[i] = (llama_pos) (base + i);
                }

                llama_batch b = {};
                b.n_tokens = (int32_t) n_step;
                b.token    = tok.data();
                b.pos      = pos.data();
                b.seq_id   = sptr.data();
                b.n_seq_id = nsid.data();
                b.logits   = lg.data();

                llama_batch_allocr ba(hparams.n_pos_per_embd());
                if (!ba.init(b, model->vocab, nullptr, hparams.n_embd_inp(), cp.n_seq_max, true)) {
                    break;
                }

                auto c = m->init_batch(ba, cp.n_ubatch, false);
                if (!c || c->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
                    break;
                }
                c->apply();
                keep = std::move(c);
            }

            auto * mctx = dynamic_cast<llama_memory_hybrid_context *>(keep.get());
            CHECK(mctx && mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS,
                    "filled a %d-cell cache in %d-token ubatches", (int) cp.n_ctx_seq, (int) cp.n_ubatch);
            if (mctx && mctx->get_status() == LLAMA_MEMORY_STATUS_SUCCESS) {
                const int64_t n_kv    = mctx->get_attn()->get_n_kv();
                const int64_t n_tps   = mctx->get_ubatch().n_tokens;
                const int64_t n_padq  = GGML_PAD(n_tps, 8) + 8;   // exercise the padded-mask path
                const int64_t n_pools = llama_kpool_n_pools((uint32_t) n_kv, (uint32_t) kpool);

                ggml_init_params gp = {
                    /* .mem_size   */ ggml_tensor_overhead()*16,
                    /* .mem_buffer */ nullptr,
                    /* .no_alloc   */ true,
                };

                ggml_context * ctx = ggml_init(gp);
                kpool_tensors kt = alloc_kpool_tensors(ctx, n_kv, n_tps, n_padq, 1, kpool, n_pools);

                ggml_backend_t        backend = ggml_backend_cpu_init();
                ggml_backend_buffer_t buf     = ggml_backend_alloc_ctx_tensors(ctx, backend);

                if (buf) {
                    const llama_ubatch & ub = mctx->get_ubatch();

                    // first pass faults in 60+ MiB of fresh pages; time the steady
                    // state, which is what a prefill pays per ubatch
                    double ms = 1e9;
                    for (int rep = 0; rep < 4; ++rep) {
                        const auto t0 = std::chrono::steady_clock::now();
                        llama_kv_cache_set_input_kpool(kv, kt.cell_pool, kt.pool_cells, kt.bias, kt.pool_bias, kt.sel_mask, kt.cand_mask,
                                &ub, (uint32_t) kpool);
                        const auto t1 = std::chrono::steady_clock::now();

                        if (rep > 0) {
                            ms = std::min(ms, std::chrono::duration<double, std::milli>(t1 - t0).count());
                        }
                    }

                    printf("note n_kv = %d, n_tokens = %d, %u DSA layers\n", (int) n_kv, (int) n_tps, n_dsa);
                    printf("note one fill = %.2f ms; shared = %.2f ms/ubatch, per layer = %.2f ms/ubatch\n",
                            ms, ms, ms*n_dsa);
                    printf("note extrapolated to 128Ki x 512 x 11 layers: %.0f ms shared, %.0f ms per layer\n",
                            ms*(131072.0/(double) n_kv)*(512.0/(double) n_tps),
                            ms*(131072.0/(double) n_kv)*(512.0/(double) n_tps)*11.0);

                    CHECK(n_dsa > 1, "the model has %u indexer layers, so sharing one fill saves %ux the host writes",
                            n_dsa, n_dsa);

                    {
                        const float * sm = (const float *) kt.sel_mask ->data;
                        const float * cm = (const float *) kt.cand_mask->data;

                        bool pad_masked = true;
                        for (int64_t ii = n_tps; ii < n_padq; ++ii) {
                            for (int64_t j = 0; j < n_kv; ++j) {
                                pad_masked &= sm[ii*n_kv + j] == -INFINITY;
                                pad_masked &= cm[ii*n_kv + j] == -INFINITY;
                            }
                        }
                        CHECK(pad_masked, "the %d padding rows of a wider sel_mask/cand_mask stay -INFINITY",
                                (int) (n_padq - n_tps));
                    }

                    // the shared object refills the same tensors deterministically.
                    // it fills only what the pooled graph consumes - cell_pool and
                    // bias have no consumer there, so it passes nullptr for both and
                    // pool_cells is what gets compared
                    std::vector<int32_t> first((size_t) ggml_nelements(kt.pool_cells));
                    memcpy(first.data(), kt.pool_cells->data, first.size()*sizeof(int32_t));

                    llm_graph_input_kpool inp(mctx->get_attn(), mctx->get_idx(), (uint32_t) kpool);
                    inp.k_idxs     = mctx->get_idx()->build_input_k_idxs(ctx, ub);
                    inp.pool_cells = kt.pool_cells;
                    inp.pool_bias  = kt.pool_bias;
                    inp.sel_mask   = kt.sel_mask;
                    inp.cand_mask  = kt.cand_mask;

                    ggml_backend_buffer_t buf2 = ggml_backend_alloc_ctx_tensors(ctx, backend);
                    if (buf2) {
                        inp.set_input(&ub);

                        CHECK(memcmp(first.data(), kt.pool_cells->data, first.size()*sizeof(int32_t)) == 0,
                                "llm_graph_input_kpool::set_input rebuilds the identical map, so one\n"
                                "     object can back every indexer layer");

                        ggml_backend_buffer_free(buf2);
                    }

                    ggml_backend_buffer_free(buf);
                }

                ggml_backend_free(backend);
                ggml_free(ctx);
            } else {
                printf("note could not fill a %d-cell cache; skipping the cost measurement\n", (int) cp.n_ctx_seq);
            }

            keep.reset();
        }

        delete raw;
    }

    llama_model_free(model);
    llama_backend_free();

    printf("\n%s: %d failure(s)\n", argv[0], n_fail);
    return n_fail == 0 ? 0 : 1;
}
