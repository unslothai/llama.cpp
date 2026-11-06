// Tree-structured speculative decoding.
//
// The draft model proposes a TREE of continuations instead of a single chain, and the target model
// verifies the whole tree in one forward pass. Tree attention needs no new kernel or mask: a KV cell
// can belong to many sequences (llama-kv-cells.h, std::bitset<LLAMA_MAX_SEQ>) and the mask predicate
// is membership AND causality, so encoding one seq_id per root-to-leaf path and setting
// pos = base_pos + depth makes every node see exactly its ancestors.
//
// Requires a unified KV cache: coupled sequences are rejected by the sequential ubatch split.
//
// Tree shape is set with LLAMA_TREE_TOP_K / LLAMA_TREE_DEPTH / LLAMA_TREE_MAX_NODES so that no
// shared argument-parsing code has to change. LLAMA_TREE_TOP_K=1 degenerates to a linear draft and
// is the control arm.

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

struct tree_node {
    llama_token tok   = 0;
    int         parent = -1;   // index into nodes, -1 for the root
    int         depth  = 0;    // root = 0
    float       logp   = 0.0f; // cumulative draft log-probability

    int i_batch_dft = -1;      // logits index in the draft batch that produced this node's children
    int i_batch_tgt = -1;      // logits index in the target verification batch

    llama_seq_id seq_dft = 0;  // draft-side sequence this node lives in
    std::vector<int> children;
    std::vector<llama_seq_id> paths; // target-side leaf paths passing through this node
};

static int env_int(const char * name, int fallback) {
    const char * v = getenv(name);
    if (v == nullptr || *v == '\0') {
        return fallback;
    }
    return atoi(v);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    // note: unlike examples/speculative we do NOT raise sampling.n_probs. The draft candidates come
    // straight from llama_get_logits_ith, and forcing the sampler to materialise a 128-entry
    // probability list costs full-vocabulary work on every accepted token.

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    const int tree_top_k     = std::max(1, env_int("LLAMA_TREE_TOP_K",     2));
    // depth 0 disables drafting entirely: the tree is just the root, so the target runs plain
    // autoregressive decoding with a batch of one. This is the reference arm for the accuracy gate.
    const int tree_depth     = std::max(0, env_int("LLAMA_TREE_DEPTH",     8));
    const int tree_max_nodes = std::max(2, env_int("LLAMA_TREE_MAX_NODES", 16));

    // one sequence per leaf, plus headroom; the tree can never have more leaves than nodes
    const int n_paths_max = std::min(tree_max_nodes, 192);

    // paths use ids [0, n_paths_max); the last two are scratch sequences for the self-test
    const llama_seq_id seq_prefix = n_paths_max;
    const llama_seq_id seq_probe  = n_paths_max + 1;

    params.n_parallel  = n_paths_max + 2;
    params.kv_unified  = true;  // mandatory: the sequential split rejects coupled sequences

    // every node of the verification batch needs logits
    params.n_outputs_max         = (tree_max_nodes + 8) * 2;
    params.n_outputs_max_per_seq = tree_max_nodes + 8;

    LOG_INF("%s: tree top_k = %d, depth = %d, max_nodes = %d, paths = %d\n",
            __func__, tree_top_k, tree_depth, tree_max_nodes, n_paths_max);

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init_tgt = common_init_from_params(params);

    llama_model   * model_tgt = llama_init_tgt->model();
    llama_context * ctx_tgt   = llama_init_tgt->context();

    // load the draft model with the same sequence capacity
    common_params params_dft = params;
    params_dft.devices           = params.speculative.draft.devices;
    params_dft.model             = params.speculative.draft.mparams;
    params_dft.n_gpu_layers      = params.speculative.draft.n_gpu_layers;
    params_dft.n_ctx             = params.n_ctx;
    params_dft.kv_unified        = true;
    params_dft.n_parallel        = n_paths_max + 1;
    params_dft.n_outputs_max     = (tree_max_nodes + 8) * 2;
    params_dft.n_outputs_max_per_seq = tree_max_nodes + 8;
    params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

    auto llama_init_dft = common_init_from_params(params_dft);

    llama_model   * model_dft = llama_init_dft->model();
    llama_context * ctx_dft   = llama_init_dft->context();

    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_type(vocab_tgt) != llama_vocab_type(vocab_dft)) {
        LOG_ERR("%s: draft and target vocab types differ\n", __func__);
        return 1;
    }

    llama_memory_t mem_tgt = llama_get_memory(ctx_tgt);
    llama_memory_t mem_dft = llama_get_memory(ctx_dft);

    const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
    const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);

    const bool audit          = env_int("LLAMA_TREE_AUDIT", 0) != 0;
    const bool selftest       = env_int("LLAMA_TREE_SELFTEST", 0) != 0;
    const int  selftest_steps = env_int("LLAMA_TREE_SELFTEST_STEPS", 4);

    double  selftest_max_abs         = 0.0;
    int64_t selftest_nodes           = 0;
    int64_t selftest_argmax_mismatch = 0;

    // tokenize the prompt and prefill both models
    const llama_tokens inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if ((int) inp.size() < 2) {
        LOG_ERR("%s: prompt must be at least 2 tokens\n", __func__);
        return 1;
    }

    {
        llama_batch b = llama_batch_init(inp.size(), 0, params.n_parallel);
        for (size_t i = 0; i + 1 < inp.size(); ++i) {
            common_batch_add(b, inp[i], i, { 0 }, false);
        }
        if (llama_decode(ctx_tgt, b) != 0) { LOG_ERR("%s: target prefill failed\n", __func__); return 1; }
        if (llama_decode(ctx_dft, b) != 0) { LOG_ERR("%s: draft prefill failed\n",  __func__); return 1; }
        llama_batch_free(b);
    }

    llama_token id_last = inp.back();
    int         n_past  = (int) inp.size() - 1;   // tokens committed to KV; id_last is NOT among them

    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));
    for (size_t i = 0; i + 1 < inp.size(); ++i) {
        common_sampler_accept(smpl.get(), inp[i], false);
    }

    llama_batch batch_dft = llama_batch_init(tree_max_nodes + 8, 0, params.n_parallel);
    llama_batch batch_tgt = llama_batch_init(tree_max_nodes + 8, 0, params.n_parallel);

    // statistics, including the coverage counters that stop a dead code path from passing as a win
    int64_t n_predict = 0, n_drafted = 0, n_accepted = 0, n_steps = 0;
    int64_t t_draft_us = 0, t_verify_us = 0, t_accept_us = 0, t_roll_us = 0;
    int64_t n_branching_steps = 0;   // steps where the tree actually had more than one leaf
    int64_t n_sibling_wins    = 0;   // steps where a non-leftmost child was accepted
    int64_t n_nodes_total     = 0;
    std::vector<int64_t> acc_hist(tree_depth + 2, 0);

    std::vector<llama_token> out_tokens;

    const int64_t t_start = ggml_time_us();

    bool has_eos = false;

    while (!has_eos && (params.n_predict < 0 || n_predict < params.n_predict)) {
        // ---------------------------------------------------------------------------------------
        // 1. build the draft tree, one level per draft forward
        // ---------------------------------------------------------------------------------------
        const int64_t t_step0 = ggml_time_us();

        std::vector<tree_node> nodes;
        nodes.push_back({}); // root carries id_last
        nodes[0].tok     = id_last;
        nodes[0].parent  = -1;
        nodes[0].depth   = 0;
        nodes[0].logp    = 0.0f;
        nodes[0].seq_dft = 0;

        // room left in the target context for this tree
        const int n_ctx_room = (int) llama_n_ctx(ctx_tgt) - n_past - 2;
        const int depth_max  = std::min(tree_depth, std::max(0, n_ctx_room - 1));

        llama_seq_id next_seq_dft = 1;

        std::vector<int> frontier = { 0 };

        // decode the root in the draft so we have logits for its children. Skipped entirely at
        // depth 0 so that arm is a clean no-speculation control.
        if (depth_max > 0) {
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, nodes[0].tok, n_past, { nodes[0].seq_dft }, true);
            if (llama_decode(ctx_dft, batch_dft) != 0) { LOG_ERR("%s: draft root decode failed\n", __func__); break; }
            nodes[0].i_batch_dft = 0;
        }

        for (int d = 1; d <= depth_max && (int) nodes.size() < tree_max_nodes; ++d) {
            // collect candidate children of every frontier node, best cumulative logp first
            struct cand { int parent; llama_token tok; float logp; };
            std::vector<cand> cands;
            std::vector<std::pair<int, float>> top;

            for (int fi : frontier) {
                const float * logits = llama_get_logits_ith(ctx_dft, nodes[fi].i_batch_dft);
                if (logits == nullptr) {
                    continue;
                }

                // Single pass: track the running max and the top-k by raw logit. k is small (2-4),
                // so an insertion scan beats sorting an index vector over the whole vocabulary.
                //
                // The admission score is the cumulative (logit - max_logit), i.e. log-probability up
                // to the per-node partition function. Computing the true softmax would cost an expf
                // over the whole vocabulary per frontier node, which on a 151k-token vocabulary is
                // more expensive than the model forwards this is meant to save. The omitted term
                // only shifts scores within a node, so it does not change that node's own ranking.
                const int k = std::min(tree_top_k, n_vocab_dft);

                top.assign(k, { -1, -INFINITY });
                float max_l = -INFINITY;

                for (int t = 0; t < n_vocab_dft; ++t) {
                    const float l = logits[t];
                    max_l = std::max(max_l, l);

                    if (l > top[k - 1].second) {
                        int j = k - 1;
                        while (j > 0 && l > top[j - 1].second) {
                            top[j] = top[j - 1];
                            --j;
                        }
                        top[j] = { t, l };
                    }
                }

                for (int j = 0; j < k; ++j) {
                    if (top[j].first < 0) {
                        continue;
                    }
                    cands.push_back({ fi, top[j].first, nodes[fi].logp + (top[j].second - max_l) });
                }
            }

            if (cands.empty()) {
                break;
            }

            std::stable_sort(cands.begin(), cands.end(),
                    [](const cand & a, const cand & b) { return a.logp > b.logp; });

            // admit candidates under the node budget; a parent's first admitted child inherits its
            // draft sequence, later ones fork a fresh sequence cloned from the parent
            common_batch_clear(batch_dft);
            std::vector<int> next_frontier;

            for (const auto & c : cands) {
                if ((int) nodes.size() >= tree_max_nodes) {
                    break;
                }

                llama_seq_id seq;
                if (nodes[c.parent].children.empty()) {
                    seq = nodes[c.parent].seq_dft;
                } else {
                    if (next_seq_dft > n_paths_max) {
                        continue;
                    }
                    seq = next_seq_dft++;
                    llama_memory_seq_rm(mem_dft, seq, -1, -1);
                    llama_memory_seq_cp(mem_dft, nodes[c.parent].seq_dft, seq, -1, -1);
                }

                tree_node n;
                n.tok     = c.tok;
                n.parent  = c.parent;
                n.depth   = d;
                n.logp    = c.logp;
                n.seq_dft = seq;

                const int id = (int) nodes.size();
                nodes.push_back(n);
                nodes[c.parent].children.push_back(id);
                next_frontier.push_back(id);

                nodes[id].i_batch_dft = batch_dft.n_tokens;
                common_batch_add(batch_dft, c.tok, n_past + d, { seq }, true);
            }

            if (batch_dft.n_tokens == 0) {
                break;
            }

            if (llama_decode(ctx_dft, batch_dft) != 0) {
                LOG_ERR("%s: draft level decode failed\n", __func__);
                break;
            }

            frontier = std::move(next_frontier);
        }

        const int n_nodes = (int) nodes.size();
        n_nodes_total += n_nodes;

        t_draft_us += ggml_time_us() - t_step0;
        const int64_t t_verify0 = ggml_time_us();

        // ---------------------------------------------------------------------------------------
        // 2. assign one target sequence per leaf, and give every node its subtree's leaves
        // ---------------------------------------------------------------------------------------
        std::vector<int> leaves;
        for (int i = 0; i < n_nodes; ++i) {
            if (nodes[i].children.empty()) {
                leaves.push_back(i);
            }
        }

        for (size_t l = 0; l < leaves.size(); ++l) {
            nodes[leaves[l]].paths.push_back((llama_seq_id) l);
        }

        // nodes are created in increasing index order, so a child always has a larger index than its
        // parent; walking backwards propagates leaf sets up the tree in one pass
        for (int i = n_nodes - 1; i >= 1; --i) {
            auto & pp = nodes[nodes[i].parent].paths;
            pp.insert(pp.end(), nodes[i].paths.begin(), nodes[i].paths.end());
        }
        for (int i = 0; i < n_nodes; ++i) {
            auto & p = nodes[i].paths;
            std::sort(p.begin(), p.end());
            p.erase(std::unique(p.begin(), p.end()), p.end());
        }

        const int n_leaves = (int) leaves.size();
        if (n_leaves > 1) {
            n_branching_steps++;
        }

        // ---------------------------------------------------------------------------------------
        // 3. clone the committed prefix into every path sequence, then verify the whole tree in one
        //    target forward
        // ---------------------------------------------------------------------------------------
        // stash a pristine copy of the committed prefix for the self-test to replay paths against
        if (selftest) {
            llama_memory_seq_rm(mem_tgt, seq_prefix, -1, -1);
            llama_memory_seq_cp(mem_tgt, 0, seq_prefix, -1, -1);
        }

        for (int l = 1; l < n_leaves; ++l) {
            llama_memory_seq_rm(mem_tgt, (llama_seq_id) l, -1, -1);
            llama_memory_seq_cp(mem_tgt, 0, (llama_seq_id) l, -1, -1);
        }

        common_batch_clear(batch_tgt);
        for (int i = 0; i < n_nodes; ++i) { // node order is already breadth-first
            nodes[i].i_batch_tgt = batch_tgt.n_tokens;
            common_batch_add(batch_tgt, nodes[i].tok, n_past + nodes[i].depth, nodes[i].paths, true);
        }

        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: target verification decode failed\n", __func__);
            break;
        }

        // Self-test of the tree encoding itself, independent of which token gets sampled.
        // For every node, decode its root-to-node path on its own in a scratch sequence and compare
        // the resulting logits against the ones the tree batch produced. If the seq-id/depth
        // encoding gives correct tree attention these agree to floating-point noise; if a node can
        // see a sibling or the wrong ancestor they diverge grossly.
        if (selftest && n_steps < selftest_steps) {
            // snapshot every node's tree logits first: the probe decodes below overwrite them
            std::vector<float> tree_logits((size_t) n_nodes * n_vocab_tgt);
            bool snapshot_ok = true;
            for (int i = 0; i < n_nodes; ++i) {
                const float * lt = llama_get_logits_ith(ctx_tgt, nodes[i].i_batch_tgt);
                if (lt == nullptr) {
                    LOG_ERR("%s: selftest: no logits for node %d (i_batch=%d) - raise n_outputs_max\n",
                            __func__, i, nodes[i].i_batch_tgt);
                    snapshot_ok = false;
                    break;
                }
                std::copy(lt, lt + n_vocab_tgt, tree_logits.begin() + (size_t) i * n_vocab_tgt);
            }

            for (int i = 0; snapshot_ok && i < n_nodes; ++i) {
                std::vector<int> chain;
                for (int c = i; c >= 0; c = nodes[c].parent) {
                    chain.push_back(c);
                }
                std::reverse(chain.begin(), chain.end());

                const float * lt = tree_logits.data() + (size_t) i * n_vocab_tgt;

                llama_memory_seq_rm(mem_tgt, seq_probe, -1, -1);
                llama_memory_seq_cp(mem_tgt, seq_prefix, seq_probe, -1, -1);

                common_batch_clear(batch_tgt);
                for (size_t c = 0; c < chain.size(); ++c) {
                    common_batch_add(batch_tgt, nodes[chain[c]].tok, n_past + (int) c, { seq_probe },
                                     c + 1 == chain.size());
                }
                if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                    LOG_ERR("%s: selftest decode failed\n", __func__);
                    break;
                }

                const float * ll = llama_get_logits_ith(ctx_tgt, batch_tgt.n_tokens - 1);
                if (ll == nullptr) {
                    LOG_ERR("%s: selftest: no logits from the linear replay of node %d\n", __func__, i);
                    break;
                }

                double max_abs = 0.0;
                int    arg_t = 0, arg_l = 0;
                for (int t = 0; t < n_vocab_tgt; ++t) {
                    max_abs = std::max(max_abs, (double) fabsf(lt[t] - ll[t]));
                    if (lt[t] > lt[arg_t]) arg_t = t;
                    if (ll[t] > ll[arg_l]) arg_l = t;
                }

                selftest_max_abs = std::max(selftest_max_abs, max_abs);
                selftest_nodes++;
                if (arg_t != arg_l) {
                    selftest_argmax_mismatch++;
                }
                LOG_DBG("%s: selftest node %d depth %d max|d|=%.3e argmax %s\n",
                        __func__, i, nodes[i].depth, max_abs, arg_t == arg_l ? "same" : "DIFF");
            }

            llama_memory_seq_rm(mem_tgt, seq_probe, -1, -1);

            // rebuild the verification batch state that the accept walk depends on
            common_batch_clear(batch_tgt);
            for (int i = 0; i < n_nodes; ++i) {
                nodes[i].i_batch_tgt = batch_tgt.n_tokens;
                common_batch_add(batch_tgt, nodes[i].tok, n_past + nodes[i].depth, nodes[i].paths, true);
            }
            for (int l = 0; l < n_leaves; ++l) {
                llama_memory_seq_rm(mem_tgt, (llama_seq_id) l, -1, -1);
                llama_memory_seq_cp(mem_tgt, seq_prefix, (llama_seq_id) l, -1, -1);
            }
            if (llama_decode(ctx_tgt, batch_tgt) != 0) {
                LOG_ERR("%s: selftest re-decode failed\n", __func__);
                break;
            }
        }

        // ---------------------------------------------------------------------------------------
        // 4. walk the tree: sample at a node, follow the child that matches, stop at the first miss
        // ---------------------------------------------------------------------------------------
        t_verify_us += ggml_time_us() - t_verify0;
        const int64_t t_accept0 = ggml_time_us();

        int cur = 0;
        std::vector<llama_token> accepted;
        std::vector<int> path_nodes = { 0 };

        while (true) {
            const llama_token id = common_sampler_sample(smpl.get(), ctx_tgt, nodes[cur].i_batch_tgt);
            common_sampler_accept(smpl.get(), id, true);
            accepted.push_back(id);

            // the gap between the best and second-best target logit at this position. A divergence
            // between two decoding arms at a position with a tiny gap is floating-point reduction
            // order, not a masking or position error.
            if (audit) {
                const float * lg = llama_get_logits_ith(ctx_tgt, nodes[cur].i_batch_tgt);
                float t1 = -INFINITY, t2 = -INFINITY;
                for (int t = 0; t < n_vocab_tgt; ++t) {
                    if (lg[t] > t1) { t2 = t1; t1 = lg[t]; }
                    else if (lg[t] > t2) { t2 = lg[t]; }
                }
                LOG("AUDIT %zu %d %.6g\n", out_tokens.size() + accepted.size() - 1, id, t1 - t2);
            }

            int match = -1;
            for (size_t c = 0; c < nodes[cur].children.size(); ++c) {
                if (nodes[nodes[cur].children[c]].tok == id) {
                    match = nodes[cur].children[c];
                    if (c > 0) {
                        n_sibling_wins++;
                    }
                    break;
                }
            }

            if (match < 0) {
                break; // id is the bonus token; it is not yet in the KV cache
            }

            cur = match;
            path_nodes.push_back(cur);
        }

        // accepted.size() - 1 draft nodes matched, plus one bonus token
        const int n_acc_nodes = (int) accepted.size() - 1;

        n_steps++;
        n_drafted  += n_nodes - 1;
        n_accepted += n_acc_nodes;
        acc_hist[std::min<size_t>(accepted.size(), acc_hist.size() - 1)]++;

        // ---------------------------------------------------------------------------------------
        // 5. keep the accepted path, drop every other branch
        // ---------------------------------------------------------------------------------------
        t_accept_us += ggml_time_us() - t_accept0;
        const int64_t t_roll0 = ggml_time_us();

        // any leaf under the last accepted node runs through the whole accepted path
        const llama_seq_id winner = nodes[path_nodes.back()].paths.front();

        // the root and every matched node are now committed, so the surviving range is
        // [0, n_past + n_acc_nodes] and anything deeper on the winning branch was rejected
        llama_memory_seq_keep(mem_tgt, winner);                               // drop all other branches
        llama_memory_seq_rm  (mem_tgt, winner, n_past + n_acc_nodes + 1, -1); // trim rejected tail
        if (winner != 0) {
            llama_memory_seq_cp(mem_tgt, winner, 0, -1, -1);
            llama_memory_seq_keep(mem_tgt, 0);
        }

        LOG_DBG("%s: step %" PRId64 " n_past=%d n_nodes=%d n_leaves=%d acc=%d winner=%d tgt[%d,%d]\n",
                __func__, n_steps, n_past, n_nodes, n_leaves, n_acc_nodes, winner,
                llama_memory_seq_pos_min(mem_tgt, 0), llama_memory_seq_pos_max(mem_tgt, 0));

        // the draft cache is rebuilt from the committed prefix each step: wipe every tree node from
        // every draft sequence, then replay only the accepted ones into sequence 0
        llama_memory_seq_rm(mem_dft, -1, n_past, -1);
        for (llama_seq_id s = 1; s < next_seq_dft; ++s) {
            llama_memory_seq_rm(mem_dft, s, -1, -1);
        }

        // replay the root plus every matched node, i.e. all of path_nodes
        {
            common_batch_clear(batch_dft);
            for (size_t i = 0; i < path_nodes.size(); ++i) {
                common_batch_add(batch_dft, nodes[path_nodes[i]].tok, n_past + (int) i, { 0 }, false);
            }
            if (llama_decode(ctx_dft, batch_dft) != 0) {
                LOG_ERR("%s: draft replay failed\n", __func__);
                break;
            }
        }

        // ---------------------------------------------------------------------------------------
        // 6. commit
        // ---------------------------------------------------------------------------------------
        // `accepted` is exactly the newly generated text: the matched child tokens followed by the
        // bonus token. The root is id_last, which the previous step already emitted.
        for (llama_token t : accepted) {
            out_tokens.push_back(t);
        }

        for (llama_token t : accepted) {
            if (llama_vocab_is_eog(vocab_tgt, t)) {
                has_eos = true;
            }
        }

        t_roll_us += ggml_time_us() - t_roll0;

        // the root (id_last) is committed too, hence the +1
        n_past   += n_acc_nodes + 1;
        n_predict += (int) accepted.size();
        id_last   = accepted.back();

        if (n_past + 4 >= (int) llama_n_ctx(ctx_tgt)) {
            break;
        }
    }

    const int64_t t_end = ggml_time_us();
    const double  t_s   = (t_end - t_start) / 1e6;

    // emit the generated token ids on a single line so the harness can diff them exactly
    {
        std::string ids;
        for (size_t i = 0; i < out_tokens.size(); ++i) {
            ids += (i ? "," : "") + std::to_string(out_tokens[i]);
        }
        LOG("\nGENERATED_IDS: %s\n", ids.c_str());
        LOG("GENERATED_TEXT: %s\n", common_detokenize(ctx_tgt, out_tokens, true).c_str());
    }

    LOG("\n");
    LOG("tree       : top_k = %d, depth = %d, max_nodes = %d\n", tree_top_k, tree_depth, tree_max_nodes);
    LOG("generated  : %" PRId64 " tokens in %.3f s = %.3f t/s\n", n_predict, t_s, n_predict / t_s);
    LOG("steps      : %" PRId64 "\n", n_steps);
    LOG("nodes/step : %.3f\n", n_steps ? (double) n_nodes_total / n_steps : 0.0);
    LOG("drafted    : %" PRId64 "\n", n_drafted);
    LOG("accepted   : %" PRId64 " (%.2f%% of drafted)\n", n_accepted, n_drafted ? 100.0 * n_accepted / n_drafted : 0.0);
    LOG("mean accept: %.3f tokens/step\n", n_steps ? (double) (n_accepted + n_steps) / n_steps : 0.0);
    if (selftest) {
        LOG("SELFTEST nodes = %" PRId64 ", max|logit diff| = %.4e, argmax mismatches = %" PRId64 "\n",
                selftest_nodes, selftest_max_abs, selftest_argmax_mismatch);
    }
    LOG("phases ms/step: draft %.2f  verify %.2f  accept %.2f  rollback %.2f\n",
            n_steps ? t_draft_us  / 1e3 / n_steps : 0.0,
            n_steps ? t_verify_us / 1e3 / n_steps : 0.0,
            n_steps ? t_accept_us / 1e3 / n_steps : 0.0,
            n_steps ? t_roll_us   / 1e3 / n_steps : 0.0);
    LOG("COVERAGE branching_steps = %" PRId64 " / %" PRId64 "\n", n_branching_steps, n_steps);
    LOG("COVERAGE sibling_wins    = %" PRId64 "\n", n_sibling_wins);
    LOG("accept_hist:");
    for (size_t i = 1; i < acc_hist.size(); ++i) {
        LOG(" %zu:%" PRId64, i, acc_hist[i]);
    }
    LOG("\n");

    llama_batch_free(batch_dft);
    llama_batch_free(batch_tgt);

    llama_backend_free();

    return 0;
}
