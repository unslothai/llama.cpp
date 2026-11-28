#include "models.h"

int llama_model_inkling2_loop::layer_phys(int il) const {
    return il < n_phys ? il : core_start + (il - n_phys) % (core_end - core_start);
}

// the earlier layout (arch inkling2) stores one block per visit: the prelude, every pass over the core, then the coda;
// fold the per-visit parameters back to the physical layers, the slot expansion below recreates the passes
void llama_model_inkling2_loop::load_legacy_hparams() {
    const uint64_t n_visit = hparams.n_layer();
    const uint64_t n_core  = core_end > core_start ? core_end - core_start : 0;
    const uint64_t n_extra = n_passes > 1 ? (uint64_t) (n_passes - 1)*n_core : 0;
    if (!n_extra || core_end + n_extra > n_visit || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2: invalid looped layout, core [%u, %u) with %u passes for %u blocks",
                core_start, core_end, n_passes, (uint32_t) n_visit));
    }

    auto fold = [&](auto & arr) {
        for (uint64_t v = core_start; v < n_visit; ++v) {
            const uint64_t p = v < core_end + n_extra ? core_start + (v - core_start) % n_core : v - n_extra;
            if (v < core_end + n_extra) {
                if (arr[v] != arr[p]) {
                    throw std::runtime_error(format("inkling2: looped layout block %u differs from its core layer %u",
                            (uint32_t) v, (uint32_t) p));
                }
            } else {
                arr[p] = arr[v];
            }
        }
    };
    fold(hparams.n_head_arr);
    fold(hparams.n_head_kv_arr);
    fold(hparams.n_ff_arr);
    fold(hparams.n_ff_exp_arr);
    fold(hparams.n_expert_used_arr);
    fold(hparams.is_swa_impl);
    fold(hparams.is_recr_impl);

    hparams.n_layer_all -= n_extra;
}

void llama_model_inkling2_loop::load_arch_hparams(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_hparams(ml);

    legacy = ml.get_arch() == LLM_ARCH_INKLING2;
    if (legacy) {
        ml.get_key(LLM_KV_LOOP_CORE_START, core_start);
        ml.get_key(LLM_KV_LOOP_CORE_END,   core_end);
        ml.get_key(LLM_KV_NUM_LOOPS,       n_passes);
        ml.get_key(LLM_KV_LOOP_GATE_RANK,  gate_rank);
        load_legacy_hparams();
    } else {
        ml.get_key(LLM_KV_LOOP_START,     core_start);
        ml.get_key(LLM_KV_LOOP_END,       core_end);
        ml.get_key(LLM_KV_LOOP_COUNT,     n_passes);
        ml.get_key(LLM_KV_LOOP_GATE_RANK, gate_rank);
        ml.get_key(LLM_KV_LOOP_VARYING,   varying, false);
    }

    n_phys = hparams.n_layer();
    const uint32_t block = hparams.attn_res_block_size;
    // a fixed loop needs at least 2 passes, a variable loop at least 1
    if (n_passes < (varying ? 1u : 2u) || !gate_rank || !core_start || core_start >= core_end || core_end > (uint32_t) n_phys ||
        core_start % block || core_end % block || hparams.n_layer_nextn) {
        throw std::runtime_error(format("inkling2_loop: invalid loop core [%u, %u) with block size %u, %u passes or gate rank %u for %d layers",
                core_start, core_end, block, n_passes, gate_rank, n_phys));
    }
    const size_t n_slot = n_phys + (size_t) (n_passes - 1)*(core_end - core_start);
    if (n_slot > LLAMA_MAX_LAYERS) {
        throw std::runtime_error(format("inkling2_loop: %zu cache slots exceed the maximum of %d", n_slot, LLAMA_MAX_LAYERS));
    }

    // expand to the cache slots before load_tensors() allocates the layers and the memory
    hparams.n_layer_all = n_slot;
    for (int il = n_phys; il < (int) n_slot; ++il) {
        const int p = layer_phys(il);
        hparams.n_head_arr[il]        = hparams.n_head_arr[p];
        hparams.n_head_kv_arr[il]     = hparams.n_head_kv_arr[p];
        hparams.n_ff_arr[il]          = hparams.n_ff_arr[p];
        hparams.n_ff_exp_arr[il]      = hparams.n_ff_exp_arr[p];
        hparams.n_expert_used_arr[il] = hparams.n_expert_used_arr[p];
        hparams.is_swa_impl[il]       = hparams.is_swa_impl[p];
        hparams.is_recr_impl[il]      = hparams.is_recr_impl[p];
    }
}

void llama_model_inkling2_loop::load_arch_tensors(llama_model_loader & ml) {
    llama_model_inkling2::load_arch_tensors(ml);

    const int64_t n_embd = hparams.n_embd;
    const int64_t rank   = gate_rank;
    const llm_tensor down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_DOWN,   LLM_TENSOR_LOOP_K_CTRL_DOWN,   LLM_TENSOR_LOOP_V_CTRL_DOWN,   LLM_TENSOR_LOOP_FFN_CTRL_DOWN   };
    const llm_tensor gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_Q_CTRL_GAIN,   LLM_TENSOR_LOOP_K_CTRL_GAIN,   LLM_TENSOR_LOOP_V_CTRL_GAIN,   LLM_TENSOR_LOOP_FFN_CTRL_GAIN   };
    const llm_tensor forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_Q_CTRL_FORGET, LLM_TENSOR_LOOP_K_CTRL_FORGET, LLM_TENSOR_LOOP_V_CTRL_FORGET, LLM_TENSOR_LOOP_FFN_CTRL_FORGET };

    // the legacy layout names the q and ffn controls after attn and mlp, and keeps all message queries in two tensors
    const llm_tensor legacy_down[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_ATTN_DOWN,   LLM_TENSOR_LOOP_K_DOWN,   LLM_TENSOR_LOOP_V_DOWN,   LLM_TENSOR_LOOP_MLP_DOWN   };
    const llm_tensor legacy_gain[LOOP_CTRL_COUNT]   = { LLM_TENSOR_LOOP_ATTN_GAIN,   LLM_TENSOR_LOOP_K_GAIN,   LLM_TENSOR_LOOP_V_GAIN,   LLM_TENSOR_LOOP_MLP_GAIN   };
    const llm_tensor legacy_forget[LOOP_CTRL_COUNT] = { LLM_TENSOR_LOOP_ATTN_FORGET, LLM_TENSOR_LOOP_K_FORGET, LLM_TENSOR_LOOP_V_FORGET, LLM_TENSOR_LOOP_MLP_FORGET };

    const int64_t n_core = core_end - core_start;
    if (legacy) {
        legacy_res_query    = create_tensor(tn(LLM_TENSOR_LOOP_RES_QUERY,    "weight"), {n_embd, 2*n_core}, TENSOR_NO_SCALE);
        legacy_kv_res_query = create_tensor(tn(LLM_TENSOR_LOOP_KV_RES_QUERY, "weight"), {n_embd, 2*n_core}, TENSOR_NO_SCALE);
    }

    loop_layers.resize(n_core);
    for (uint32_t il = core_start; il < core_end; ++il) {
        auto & l = loop_layers[il - core_start];
        // the q, k, v and ffn message queries, built in the graph for the legacy layout
        if (!legacy) {
            l.msg_query = create_tensor(tn(LLM_TENSOR_LOOP_MSG_QUERY, "weight", il), {n_embd, LOOP_CTRL_COUNT}, TENSOR_NO_SCALE);
        }
        for (int c = 0; c < LOOP_CTRL_COUNT; ++c) {
            l.ctrl_down[c]   = create_tensor(tn(legacy ? legacy_down[c]   : down[c],   "weight", il), {2*n_embd, rank}, 0);
            l.ctrl_gain[c]   = create_tensor(tn(legacy ? legacy_gain[c]   : gain[c],   "weight", il), {rank, n_embd}, 0);
            l.ctrl_forget[c] = create_tensor(tn(legacy ? legacy_forget[c] : forget[c], "weight", il), {rank, n_embd}, 0);
        }
    }
}

void llama_model_inkling2_loop::post_arch_tensors(llama_model_loader &) {
    // share the physical weights, each cache slot keeps its own KV and conv state
    for (int il = n_phys; il < (int) hparams.n_layer(); ++il) {
        layers[il] = layers[layer_phys(il)];
    }
}

// legacy layout: the message queries of core layer i are rows [res[2i], kv[2i], kv[2i+1], res[2i+1]] (q, k, v, ffn)
ggml_tensor * llama_model_inkling2_loop::graph::legacy_query(int i) const {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    auto row = [&](ggml_tensor * t, int r, int n) {
        return ggml_view_2d(ctx0, t, t->ne[0], n, t->nb[1], r*t->nb[1]);
    };
    auto * q = ggml_concat(ctx0, row(m.legacy_res_query, 2*i, 1), row(m.legacy_kv_res_query, 2*i, 2), 1);
    return ggml_concat(ctx0, q, row(m.legacy_res_query, 2*i + 1, 1), 1);
}

std::unique_ptr<llm_graph_context> llama_model_inkling2_loop::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(params);
}

llama_model_inkling2_loop::graph::graph(const llm_graph_params & params) : llama_model_inkling2::graph(params, no_build{}) {
    const auto & m = static_cast<const llama_model_inkling2_loop &>(model);
    const int start = m.core_start;
    const int end   = m.core_end;

    auto * cur = build_inputs();
    std::vector<ggml_tensor *> blocks;

    // DSpark target features are indexed by layer visit, not by cache slot
    int visit = 0;

    for (int il = 0; il < start; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }

    // every pass starts from the prelude output
    auto * prelude = cur;
    const auto prelude_blocks = blocks;
    for (uint32_t pass = 0; pass < m.n_passes; ++pass) {
        // the bank of the previous pass: its blocks after the first core block, and its output
        std::vector<ggml_tensor *> bank;
        ggml_tensor * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }
        for (int ip = start; ip < end; ++ip) {
            const int il = pass == 0 ? ip : m.n_phys + (pass - 1)*(end - start) + (ip - start);
            res->t_layer_inp[visit++] = cur;
            if (pass == 0) {
                cur = build_layer(il, ip, cur, blocks);
                continue;
            }
            const auto & loop = m.loop_layers[ip - start];
            cur = build_layer(il, ip, cur, blocks, mix(bank, bank_out, loop.msg_query ? loop.msg_query : legacy_query(ip - start)), &loop);
        }
    }

    for (int il = end; il < m.n_phys; ++il) {
        res->t_layer_inp[visit++] = cur;
        cur = build_layer(il, il, cur, blocks);
    }
    GGML_ASSERT(visit == n_layer);
    build_output(cur, blocks);
}
