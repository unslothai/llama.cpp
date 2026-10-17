#include "llama-hparams.h"
#include "models.h"

#include "llama-kv-cache-dsv4.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

// DeepSeek-V4.1 (ref: deepseek-ai/DeepSeek-V4.1-Flash-1017 inference/model.py, engram.py)
//
// Differences to DeepSeek-V4 handled here:
//  - Engram on a few layers: hashed 2..N-gram rows gathered from host-resident tables, mixed into every HC copy
//  - single-pass mHC: a sublayer collapses the HC copies with the pre-mix computed by the PREVIOUS sublayer
//    (layer 0 attention: copy 0), and there is no hc_head (the output collapses with the last FFN pre-mix)
//  - compress ratios 1 and 2 (plain softmax pooling, no ape / overlap), compressed KV and index K produced only by
//    the kv source layers and read by the layers after them; index K = k_norm(wk(latent)), stored next to the KV row
//  - indexer top-k computed only on index source layers, reused by the layers in between
//  - two-level candidate selection: the candidate source layer picks blocks, later index layers score inside them
//  - no per-head q norm after wq_b, no hash-routed (tid2eid) layers

static float dsv41_rope_attn_factor(float freq_scale, float ext_factor) {
    if (ext_factor == 0.0f) {
        return 1.0f;
    }

    return 1.0f / (1.0f + 0.1f*logf(1.0f/freq_scale));
}

static constexpr int64_t DSV41_ENGRAM_DEAD = -1;

//
// loading
//

void llama_model_deepseek41::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);

    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm);
    ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_EXP,     hparams.swiglu_clamp_exp,   hparams.n_layer_all);
    if (!ml.get_key_or_arr(LLM_KV_SWIGLU_CLAMP_SHEXP,   hparams.swiglu_clamp_shexp, hparams.n_layer_all, 0)) {
        hparams.swiglu_clamp_shexp = hparams.swiglu_clamp_exp;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT, hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH, hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,      hparams.indexer_top_k);

    ml.get_key(LLM_KV_ATTENTION_OUTPUT_GROUP_COUNT,         hparams.dsv4_o_group_count);
    ml.get_key(LLM_KV_ATTENTION_OUTPUT_LORA_RANK,           hparams.dsv4_o_lora_rank);
    ml.get_key(LLM_KV_ATTENTION_COMPRESS_ROPE_FREQ_BASE,    hparams.dsv4_compress_rope_base);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,               hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_SINKHORN_ITERATIONS, hparams.dsv4_hc_sinkhorn_iters);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,             hparams.dsv4_hc_eps);
    hparams.dsv4_hash_layer_count = 0;

    hparams.n_embd_out_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    uint32_t n_compress_ratios = 0;
    ml.get_arr_n(LLM_KV_ATTENTION_COMPRESS_RATIOS, n_compress_ratios);
    if (n_compress_ratios < hparams.n_layer_all) {
        throw std::runtime_error("DeepSeek-V4.1 compress_ratios is shorter than block_count");
    }
    GGML_ASSERT(n_compress_ratios <= LLAMA_MAX_LAYERS);
    ml.get_arr(LLM_KV_ATTENTION_COMPRESS_RATIOS, hparams.dsv4_compress_ratios);

    ml.get_key(LLM_KV_EXPERT_GATING_FUNC, hparams.expert_gating_func);
    if (hparams.expert_gating_func != LLAMA_EXPERT_GATING_FUNC_TYPE_SQRT_SOFTPLUS) {
        throw std::runtime_error("DeepSeek-V4.1 loader expects sqrtsoftplus MoE scoring");
    }

    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.set_swa_pattern(0);
    hparams.non_causal_type = LLAMA_NON_CAUSAL_TYPE_SWA_FULL;
    for (uint32_t il = hparams.n_layer(); il < hparams.n_layer_all; ++il) {
        hparams.is_swa_impl[il] = true;
    }

    // cross-layer sharing: every compressed layer reads the latest kv source at or before it, and attends with the
    // top-k of the latest index source at or before it (SharedAttentionRuntime in the reference)
    std::vector<uint32_t> kv_src;
    std::vector<uint32_t> idx_src;
    ml.get_arr(LLM_KV_ATTENTION_KV_SOURCE_LAYERS,      kv_src);
    ml.get_arr(LLM_KV_ATTENTION_INDEXER_SOURCE_LAYERS, idx_src);

    const uint32_t n_layer = hparams.n_layer();
    int32_t last_kv  = -1;
    int32_t last_idx = -1;
    for (uint32_t il = 0; il < LLAMA_MAX_LAYERS; ++il) {
        hparams.dsv41_kv_src[il]   = -1;
        hparams.dsv41_topk_src[il] = -1;
    }
    for (uint32_t il = 0; il < n_layer; ++il) {
        const uint32_t ratio = hparams.dsv4_compress_ratios[il];
        if (std::find(kv_src.begin(), kv_src.end(), il) != kv_src.end()) {
            if (ratio == 0) {
                throw std::runtime_error(format("DeepSeek-V4.1 kv source layer %u has compress ratio 0", il));
            }
            last_kv = (int32_t) il;
        }
        if (std::find(idx_src.begin(), idx_src.end(), il) != idx_src.end()) {
            if (ratio == 0) {
                throw std::runtime_error(format("DeepSeek-V4.1 index source layer %u has compress ratio 0", il));
            }
            last_idx = (int32_t) il;
        }
        if (ratio == 0) {
            continue;
        }
        if (ratio != 1 && ratio != 2) {
            throw std::runtime_error(format("DeepSeek-V4.1 layer %u: compress ratio %u (only 0, 1, 2 supported)", il, ratio));
        }
        if (last_kv < 0 || last_idx < 0 ||
                hparams.dsv4_compress_ratios[last_kv] != ratio || hparams.dsv4_compress_ratios[last_idx] != ratio ||
                last_idx < last_kv) {
            throw std::runtime_error(format("DeepSeek-V4.1 layer %u: no matching kv/index source before it", il));
        }
        hparams.dsv41_kv_src[il]   = last_kv;
        hparams.dsv41_topk_src[il] = last_idx;
    }

    hparams.dsv41_cand_src         = -1;
    hparams.dsv41_cand_topk_blocks = 0;
    hparams.dsv41_cand_block_size  = 0;
    ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_SOURCE_LAYER, hparams.dsv41_cand_src, false);
    if (hparams.dsv41_cand_src >= 0) {
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_TOP_K_BLOCKS, hparams.dsv41_cand_topk_blocks);
        ml.get_key(LLM_KV_ATTENTION_INDEXER_CANDIDATE_BLOCK_SIZE,   hparams.dsv41_cand_block_size);
        const int32_t cs = hparams.dsv41_cand_src;
        if ((uint32_t) cs >= n_layer || hparams.dsv41_topk_src[cs] != cs || hparams.dsv41_cand_block_size == 0 ||
                hparams.dsv41_cand_topk_blocks == 0) {
            throw std::runtime_error("DeepSeek-V4.1 candidate source layer must be an index source");
        }
        for (uint32_t il = cs + 1; il < n_layer; ++il) {
            if (hparams.dsv41_topk_src[il] == (int32_t) il && hparams.dsv4_compress_ratios[il] != hparams.dsv4_compress_ratios[cs]) {
                throw std::runtime_error("DeepSeek-V4.1 candidate consumers must share the source's compress ratio");
            }
        }
    }

    // Engram
    std::vector<uint32_t> e_layers;
    ml.get_arr(LLM_KV_ENGRAM_LAYERS, e_layers, false);
    if (!e_layers.empty()) {
        uint32_t pad_id = 0;
        std::vector<uint32_t> primes;
        ml.get_arr(LLM_KV_ENGRAM_NUM_EMBEDDINGS, engram_n_rows);
        ml.get_key(LLM_KV_ENGRAM_MAX_NGRAM_SIZE, engram_max_ngram);
        ml.get_key(LLM_KV_ENGRAM_HEAD_COUNT,     engram_n_head);
        ml.get_key(LLM_KV_ENGRAM_HEAD_DIM,       engram_head_dim);
        ml.get_key(LLM_KV_ENGRAM_PAD_TOKEN_ID,   pad_id);
        ml.get_arr(LLM_KV_ENGRAM_TOKEN_MAP,      engram_token_map);
        ml.get_arr(LLM_KV_ENGRAM_MULTIPLIERS,    engram_mult);
        ml.get_arr(LLM_KV_ENGRAM_PRIMES,         primes);

        const size_t n_e    = e_layers.size();
        const size_t n_cols = (size_t) (engram_max_ngram - 1)*engram_n_head;
        if (engram_max_ngram < 2 || engram_n_head == 0 || engram_n_rows.size() != n_e ||
                engram_mult.size() != n_e*engram_max_ngram || primes.size() != n_e*n_cols ||
                pad_id >= engram_token_map.size()) {
            throw std::runtime_error("DeepSeek-V4.1 engram metadata is inconsistent");
        }
        engram_layers.assign(e_layers.begin(), e_layers.end());
        hparams.dsv41_tok_hist = true; // state save/restore must carry the n-gram history
        engram_pad_cid = engram_token_map[pad_id];
        engram_primes.assign(primes.begin(), primes.end());
        engram_offsets.resize(primes.size());
        for (size_t e = 0; e < n_e; ++e) {
            uint64_t off = 0;
            for (size_t c = 0; c < n_cols; ++c) {
                engram_offsets[e*n_cols + c] = off;
                off += engram_primes[e*n_cols + c];
            }
            if (off > engram_n_rows[e] || off > (uint64_t) INT32_MAX) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram layer %u: buckets (%llu) exceed the table rows (%u)",
                        e_layers[e], (unsigned long long) off, engram_n_rows[e]));
            }
        }
        for (uint32_t il : e_layers) {
            if (il >= n_layer) {
                throw std::runtime_error(format("DeepSeek-V4.1 engram layer %u is out of range", il));
            }
        }
    }

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_deepseek41::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t q_lora_rank     = hparams.n_lora_q;
    const int64_t n_ff_exp        = hparams.n_ff_exp();
    const int64_t n_expert_shared = hparams.n_expert_shared;

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t o_groups    = hparams.dsv4_o_group_count;
    const int64_t o_lora_rank = hparams.dsv4_o_lora_rank;
    const int64_t hc_mult     = hparams.dsv4_hc_mult;
    const int64_t hc_dim      = hc_mult * n_embd;
    const int64_t hc_mix_dim  = (2 + hc_mult) * hc_mult;
    const int64_t n_idx_embd  = hparams.indexer_head_size;
    const int64_t n_idx_head  = hparams.indexer_n_head;

    // DSpark stages (blk.n_layer..) are loaded as optional tensors; the graph does not use them yet
    uint32_t n_expert_dspark = n_expert;
    ml.get_key(LLM_KV_DSPARK_EXPERT_COUNT, n_expert_dspark, false);

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const bool mtp   = i >= n_layer;
        const int  flags = mtp ? TENSOR_NOT_REQUIRED : 0;
        const int64_t n_exp = mtp ? (int64_t) n_expert_dspark : n_expert;

        layer.attn_norm     = create_tensor(tn(LLM_TENSOR_ATTN_NORM,     "weight", i), {n_embd}, flags);
        layer.attn_sinks    = create_tensor(tn(LLM_TENSOR_ATTN_SINKS,    "weight", i), {n_head}, flags);
        layer.wq_a          = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
        layer.attn_q_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, flags);
        layer.wq_b          = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head * n_embd_head}, flags);
        layer.wkv           = create_tensor(tn(LLM_TENSOR_ATTN_KV,       "weight", i), {n_embd, n_embd_head}, flags);
        layer.attn_kv_norm  = create_tensor(tn(LLM_TENSOR_ATTN_KV_NORM,  "weight", i), {n_embd_head}, flags);
        layer.wo_a          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_A,    "weight", i), {n_head * n_embd_head / o_groups, o_lora_rank, o_groups}, flags | TENSOR_ALLOW_RESHAPE);
        layer.wo_b          = create_tensor(tn(LLM_TENSOR_ATTN_OUT_B,    "weight", i), {o_groups * o_lora_rank, n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_scale = create_tensor(tn(LLM_TENSOR_HC_ATTN_SCALE, "weight", i), {3}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc_dim, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_scale  = create_tensor(tn(LLM_TENSOR_HC_FFN_SCALE,  "weight", i), {3}, flags);

        if (!mtp && hparams.dsv41_kv_src[i] == i) {
            const uint32_t ratio = hparams.dsv4_compress_ratios[i];
            layer.attn_comp_wkv  = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WKV,  "weight", i), {n_embd, n_embd_head}, 0);
            layer.attn_comp_norm = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_NORM, "weight", i), {n_embd_head}, 0);
            if (ratio > 1) {
                layer.attn_comp_wgate = create_tensor(tn(LLM_TENSOR_ATTN_COMPRESSOR_WGATE, "weight", i), {n_embd, n_embd_head}, 0);
            }
            layer.indexer_attn_k = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_K, "weight", i), {n_embd_head, n_idx_embd}, 0);
            layer.indexer_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {n_idx_embd}, 0);
        }
        if (!mtp && hparams.dsv41_topk_src[i] == i) {
            layer.indexer_attn_q_b = create_tensor(tn(LLM_TENSOR_INDEXER_ATTN_Q_B, "weight", i), {q_lora_rank, n_idx_head * n_idx_embd}, 0);
            layer.indexer_proj     = create_tensor(tn(LLM_TENSOR_INDEXER_PROJ,     "weight", i), {n_embd, n_idx_head}, 0);
        }

        if (!mtp) {
            const int e = engram_index(i);
            if (e >= 0) {
                const int64_t n_cols = engram_n_cols();
                layer.engram_embd = create_tensor(tn(LLM_TENSOR_ENGRAM_EMBD, "weight", i), {(int64_t) engram_head_dim, (int64_t) engram_n_rows[e]}, 0);
                layer.engram_kv   = create_tensor(tn(LLM_TENSOR_ENGRAM_KV,   "weight", i), {n_cols * engram_head_dim, n_embd * (hc_mult + 1)}, 0);
                layer.engram_gate = create_tensor(tn(LLM_TENSOR_ENGRAM_GATE, "weight", i), {n_embd, hc_mult}, 0);
            }
        }

        layer.ffn_gate_inp       = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,       "weight", i), {n_embd, n_exp}, flags);
        layer.ffn_exp_probs_b    = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B,    "bias",   i), {n_exp}, flags);
        layer.ffn_exp_probs_b_vl = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B_VL, "bias",   i), {n_exp}, TENSOR_NOT_REQUIRED);
        layer.ffn_norm           = create_tensor(tn(LLM_TENSOR_FFN_NORM,           "weight", i), {n_embd}, flags);

        layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd,   n_ff_exp, n_exp}, flags);
        layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd,   n_exp}, flags);
        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_exp}, flags);

        layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);
        layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_exp * n_expert_shared, n_embd                    }, flags);
        layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd,                     n_ff_exp * n_expert_shared}, flags);

        if (mtp) {
            layer.dspark_main_proj = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_PROJ, "weight", i), {n_embd * 3, n_embd}, TENSOR_NOT_REQUIRED);
            layer.dspark_main_norm = create_tensor(tn(LLM_TENSOR_DSPARK_MAIN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
            layer.dspark_norm      = create_tensor(tn(LLM_TENSOR_DSPARK_NORM,      "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_deepseek41::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        throw std::runtime_error("DeepSeek-V4.1 DSpark drafting is not implemented yet");
    }
    return std::make_unique<graph>(*this, params);
}

//
// Engram n-gram hash (engram.py NgramHashState.forward)
//

void llama_model_deepseek41::engram_hash(int e, const llama_ubatch & ubatch, const std::vector<llama_token> & prev, std::vector<int32_t> & out) const {
    const int64_t n_tokens = ubatch.n_tokens;
    const int64_t n_gram   = engram_max_ngram;
    const int64_t n_prev   = n_gram - 1;
    const int64_t n_head   = engram_n_head;
    const int64_t n_cols   = engram_n_cols();
    const int64_t n_map    = (int64_t) engram_token_map.size();

    const uint64_t * mult   = engram_mult.data()    + (size_t) e*n_gram;
    const uint64_t * primes = engram_primes.data()  + (size_t) e*n_cols;
    const uint64_t * offs   = engram_offsets.data() + (size_t) e*n_cols;

    out.resize(n_tokens*n_cols);

    // compressed id of a raw token; embeddings (image spans) and unknown history are DEAD
    const auto cid = [&](llama_token t) -> int64_t {
        if (t < 0 || t >= n_map) {
            return DSV41_ENGRAM_DEAD;
        }
        return engram_token_map[t];
    };

    std::vector<uint64_t> toks(n_gram);
    for (int64_t i = 0; i < n_tokens; ++i) {
        const bool is_embd = !ubatch.token || (ubatch.is_mixed() && ubatch.type[i]);
        const llama_pos pos = ubatch.pos[i];

        // shift 0 is the token itself; once a shift is blocked (sequence start or dead token) every longer one is too
        int64_t c = is_embd ? DSV41_ENGRAM_DEAD : cid(ubatch.token[i]);
        bool blocked = c == DSV41_ENGRAM_DEAD;
        toks[0] = blocked ? engram_pad_cid : c;
        for (int64_t s = 1; s < n_gram; ++s) {
            const llama_token t = prev[i*n_prev + (n_prev - s)];
            c = (pos - s < 0) ? DSV41_ENGRAM_DEAD : cid(t);
            blocked = blocked || c == DSV41_ENGRAM_DEAD;
            toks[s] = blocked ? engram_pad_cid : c;
        }

        uint64_t rolling = toks[0]*mult[0];
        for (int64_t o = 1; o < n_gram; ++o) {
            rolling ^= toks[o]*mult[o];
            for (int64_t h = 0; h < n_head; ++h) {
                const int64_t col = (o - 1)*n_head + h;
                out[i*n_cols + col] = (int32_t) (rolling % primes[col] + offs[col]);
            }
        }
    }
}

// Host-side inputs: Engram row ids per engram layer, and the candidate-block pin (+1e30 on the block holding each
// query's newest visible compressed row: select_candidate_blocks pins it so a partly filled block always survives).
class llm_graph_input_dsv41 : public llm_graph_input_i {
public:
    llm_graph_input_dsv41(const llama_model_deepseek41 & model, const llama_kv_cache_dsv4_context * mctx) :
        model(model), mctx(mctx) {}
    virtual ~llm_graph_input_dsv41() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::vector<ggml_tensor *> engram_ids; // per engram layer: I32 [n_cols*n_tokens]
    ggml_tensor * engram_keep = nullptr;    // F32 [1, 1, n_tokens], 0 for embedding (image) tokens; only in embd ubatches
    ggml_tensor * cand_pin = nullptr;       // F32 [n_blk, n_tokens/n_stream, 1, n_stream]
    bool cand_csa = false;                  // candidate source reads the CSA (ratio 2) slot, else HCA (ratio 1)

    const llama_model_deepseek41 & model;
    const llama_kv_cache_dsv4_context * mctx;

    std::vector<llama_token> prev;
    std::vector<int32_t> ids;
    std::vector<float> pin;
    std::vector<float> keep;

private:
    const llama_kv_cache_dsv4_context::comp_plan & cand_plan(const llama_ubatch & ubatch) const {
        return cand_csa ? mctx->get_csa_plan(ubatch) : mctx->get_hca_plan(ubatch);
    }
};

void llm_graph_input_dsv41::set_input(const llama_ubatch * ubatch) {
    if (!engram_ids.empty()) {
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            GGML_ASSERT(ubatch->n_seq_id[i] == 1 && "DeepSeek-V4.1 Engram n-grams do not support tokens shared by multiple sequences");
        }
        // predecessors come from the raw window cells, which already hold this ubatch
        mctx->get_raw()->get_prev_tokens(*ubatch, model.engram_max_ngram - 1, prev);
        for (size_t e = 0; e < engram_ids.size(); ++e) {
            if (!engram_ids[e]->buffer) {
                continue;
            }
            model.engram_hash((int) e, *ubatch, prev, ids);
            ggml_backend_tensor_set(engram_ids[e], ids.data(), 0, ids.size()*sizeof(int32_t));
        }
    }

    // image tokens take no part in an n-gram (hashed DEAD above) and get no Engram contribution (gate 0), as model.py
    if (engram_keep && engram_keep->buffer) {
        keep.resize(ubatch->n_tokens);
        for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
            const bool is_embd = !ubatch->token || (ubatch->is_mixed() && ubatch->type[i]);
            keep[i] = is_embd ? 0.0f : 1.0f;
        }
        ggml_backend_tensor_set(engram_keep, keep.data(), 0, keep.size()*sizeof(float));
    }

    if (cand_pin && cand_pin->buffer) {
        const auto & plan = cand_plan(*ubatch);
        const int64_t n_blk = cand_pin->ne[0];
        const int64_t bs    = model.hparams.dsv41_cand_block_size;
        GGML_ASSERT((int64_t) plan.n_visible.size() == (int64_t) ubatch->n_tokens);
        pin.assign(n_blk*ubatch->n_tokens, 0.0f);
        for (int64_t i = 0; i < (int64_t) ubatch->n_tokens; ++i) {
            const int64_t nv = plan.n_visible[i];
            if (nv > 0) {
                const int64_t b = (nv - 1)/bs;
                GGML_ASSERT(b < n_blk);
                pin[i*n_blk + b] = 1e30f;
            }
        }
        ggml_backend_tensor_set(cand_pin, pin.data(), 0, pin.size()*sizeof(float));
    }
}

bool llm_graph_input_dsv41::can_reuse(const llm_graph_params & params) {
    mctx = static_cast<const llama_kv_cache_dsv4_context *>(params.mctx);

    bool res = true;
    for (auto * t : engram_ids) {
        res &= t->ne[0] == (int64_t) model.engram_n_cols()*params.ubatch.n_tokens;
    }
    res &= (engram_keep != nullptr) == (!engram_ids.empty() && params.ubatch.embd != nullptr);
    if (engram_keep) {
        res &= engram_keep->ne[2] == (int64_t) params.ubatch.n_tokens;
    }
    if (cand_pin) {
        const auto & plan = cand_plan(params.ubatch);
        const int64_t bs = model.hparams.dsv41_cand_block_size;
        res &= plan.n_kv > 0 && cand_pin->ne[0] == (plan.n_kv + bs - 1)/bs;
        res &= cand_pin->ne[1]*cand_pin->ne[3] == (int64_t) params.ubatch.n_tokens;
    }
    return res;
}

//
// graph helpers
//

static ggml_tensor * dsv41_view_2d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t ne1, int64_t i0) {
    return ggml_view_2d(ctx, t, ne0, ne1, t->nb[1], ggml_row_size(t->type, i0));
}

static ggml_tensor * dsv41_view_1d(ggml_context * ctx, ggml_tensor * t, int64_t ne0, int64_t i0) {
    return ggml_view_1d(ctx, t, ne0, ggml_row_size(t->type, i0));
}

ggml_tensor * llama_model_deepseek41::graph::build_hc_mixes(
        ggml_tensor * x,
        ggml_tensor * hc_fn,
        ggml_tensor * hc_scale,
        ggml_tensor * hc_base,
        ggml_tensor ** post,
        ggml_tensor ** comb,
        int il) const {
    const int64_t hc         = hparams.dsv4_hc_mult;
    const int64_t hc_dim     = hc*n_embd;
    const int64_t hc_mix_dim = (2 + hc)*hc;
    const int64_t nt         = x->ne[2];

    GGML_ASSERT(hc == 4);
    GGML_ASSERT(hc_fn->ne[1] == hc_mix_dim);

    // normalized over the whole flattened hc*d stream, one statistic per token
    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, hc_dim, nt);
    ggml_tensor * flat_norm = ggml_rms_norm(ctx0, flat, norm_rms_eps);
    ggml_tensor * mixes = ggml_mul_mat(ctx0, hc_fn, flat_norm);
    cb(mixes, "hc_mixes", il);

    ggml_tensor * pre = dsv41_view_2d(ctx0, mixes, hc, nt, 0);
    pre = ggml_mul(ctx0, pre, dsv41_view_1d(ctx0, hc_scale, 1, 0));
    pre = ggml_add(ctx0, pre, dsv41_view_1d(ctx0, hc_base, hc, 0));
    pre = ggml_sigmoid(ctx0, pre);
    pre = ggml_scale_bias(ctx0, pre, 1.0f, hparams.dsv4_hc_eps);
    cb(pre, "hc_pre", il);

    *post = dsv41_view_2d(ctx0, mixes, hc, nt, hc);
    *post = ggml_mul(ctx0, *post, dsv41_view_1d(ctx0, hc_scale, 1, 1));
    *post = ggml_add(ctx0, *post, dsv41_view_1d(ctx0, hc_base, hc, hc));
    *post = ggml_sigmoid(ctx0, *post);
    *post = ggml_scale(ctx0, *post, 2.0f);
    cb(*post, "hc_post", il);

    if (cparams.fused_dsv4_hc_comb) {
        *comb = ggml_dsv4_hc_comb(ctx0, mixes, hc_scale, hc_base, hparams.dsv4_hc_eps,
                (int32_t) hparams.dsv4_hc_sinkhorn_iters);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_COMB, *comb, il});
    } else {
        *comb = dsv41_view_2d(ctx0, mixes, hc*hc, nt, 2*hc);
        *comb = ggml_mul(ctx0, *comb, dsv41_view_1d(ctx0, hc_scale, 1, 2));
        *comb = ggml_add(ctx0, *comb, dsv41_view_1d(ctx0, hc_base, hc*hc, 2*hc));
        *comb = ggml_reshape_3d(ctx0, *comb, hc, hc, nt);
        *comb = build_hc_sinkhorn(*comb, il);
    }
    cb(*comb, "hc_comb", il);

    return pre;
}

// Engram.forward: x [n_embd, hc, n_tokens], ids I32 [n_cols*n_tokens]
ggml_tensor * llama_model_deepseek41::graph::build_engram(
        const llama_model_deepseek41 & model,
        ggml_tensor * x,
        ggml_tensor * ids,
        ggml_tensor * keep,
        int il) const {
    const auto & layer = model.layers[il];
    const int64_t hc     = hparams.dsv4_hc_mult;
    const int64_t nt     = x->ne[2];
    const int64_t n_cols = model.engram_n_cols();
    const int64_t hd     = model.engram_head_dim;

    ggml_tensor * rows = ggml_get_rows(ctx0, layer.engram_embd, ids); // [hd, n_cols*nt]
    if (rows->type != GGML_TYPE_F32) {
        rows = ggml_cast(ctx0, rows, GGML_TYPE_F32);
    }
    rows = ggml_reshape_2d(ctx0, rows, hd*n_cols, nt);
    cb(rows, "engram_rows", il);

    ggml_tensor * kv = build_lora_mm(layer.engram_kv, rows); // [n_embd*(hc+1), nt]
    cb(kv, "engram_kv", il);

    ggml_tensor * key   = ggml_view_3d(ctx0, kv, n_embd, hc, nt, ggml_row_size(kv->type, n_embd), kv->nb[1], 0);
    ggml_tensor * value = ggml_view_3d(ctx0, kv, n_embd, 1,  nt, ggml_row_size(kv->type, n_embd), kv->nb[1], ggml_row_size(kv->type, hc*n_embd));
    key   = ggml_cont(ctx0, key);
    value = ggml_cont(ctx0, value);

    // dot_c = sum_d (x * rstd_x) * (key * rstd_k) * w / sqrt(n_embd), normalized per (token, copy)
    ggml_tensor * xn = ggml_rms_norm(ctx0, x,   norm_rms_eps);
    ggml_tensor * kn = ggml_rms_norm(ctx0, key, norm_rms_eps);
    ggml_tensor * dot = ggml_mul(ctx0, ggml_mul(ctx0, xn, kn), layer.engram_gate);
    dot = ggml_sum_rows(ctx0, dot); // [1, hc, nt]
    dot = ggml_scale(ctx0, dot, 1.0f/sqrtf((float) n_embd));
    cb(dot, "engram_dot", il);

    // gate = sigmoid(sign(dot) * sqrt(max(|dot|, 1e-6)))
    ggml_tensor * mag = ggml_sqrt(ctx0, ggml_clamp(ctx0, ggml_abs(ctx0, dot), 1e-6f, INFINITY));
    ggml_tensor * gate = ggml_sigmoid(ctx0, ggml_mul(ctx0, mag, ggml_sgn(ctx0, dot)));
    if (keep) {
        gate = ggml_mul(ctx0, gate, keep); // [1, hc, nt] * [1, 1, nt]: image tokens pass through untouched
    }
    cb(gate, "engram_gate", il);

    ggml_tensor * upd = ggml_mul(ctx0, ggml_repeat_4d(ctx0, value, n_embd, hc, nt, 1), gate);
    return ggml_add(ctx0, x, upd);
}

static void dsv41_state_restore(
        ggml_context * ctx,
        const llm_graph_input_dsv4::comp_input & inp,
        const llama_dsv4_comp_state * state,
        int32_t il,
        ggml_tensor ** kv,
        ggml_tensor ** score) {
    *kv    = state->get_kv_all(ctx, il);
    *score = state->get_score_all(ctx, il);

    if (inp.state_restore_src_idxs == nullptr || inp.state_restore_dst_idxs == nullptr) {
        return;
    }

    *kv    = state->cpy_kv   (ctx, ggml_get_rows(ctx, *kv,    inp.state_restore_src_idxs), inp.state_restore_dst_idxs, il);
    *score = state->cpy_score(ctx, ggml_get_rows(ctx, *score, inp.state_restore_src_idxs), inp.state_restore_dst_idxs, il);
}

ggml_tensor * llama_model_deepseek41::graph::build_attention41(
        const llama_model_deepseek41 & model,
        llm_graph_input_dsv4 * inp_dsv4,
        ggml_tensor * cur,
        ggml_tensor * inp_pos,
        ggml_tensor * cand_pin,
        int il) {
    const auto & layer = model.layers[il];
    llm_graph_input_dsv4_raw * inp_attn = inp_dsv4->get_raw();

    const int64_t n_embd_head      = hparams.n_embd_head_k();
    const int64_t n_embd_head_rope = hparams.n_rot();
    const int64_t n_embd_head_nope = n_embd_head - n_embd_head_rope;
    const int64_t n_groups         = hparams.dsv4_o_group_count;
    const int64_t n_heads_group    = n_head / n_groups;
    const int64_t o_lora_rank      = hparams.dsv4_o_lora_rank;
    const int64_t o_group_dim      = n_heads_group*n_embd_head;
    const int64_t n_idx_embd       = hparams.indexer_head_size;
    const int64_t n_idx_head       = hparams.indexer_n_head;
    const int64_t nt               = cur->ne[1];
    const float   kq_scale         = 1.0f/sqrtf(float(n_embd_head));

    GGML_ASSERT(n_embd_head == n_embd_head_v);
    GGML_ASSERT(n_head % n_groups == 0);

    const int64_t ratio = hparams.dsv4_compress_ratios[il];

    // layers with compressed KV use the compress theta + YaRN, window-only layers plain rope
    const bool use_compress_rope = ratio != 0;
    const float freq_base_l      = use_compress_rope ? hparams.dsv4_compress_rope_base : freq_base;
    const float freq_scale_l     = use_compress_rope ? freq_scale : 1.0f;
    const float ext_factor_l     = use_compress_rope ? ext_factor : 0.0f;
    const float attn_factor_l    = dsv41_rope_attn_factor(freq_scale_l, ext_factor_l);
    const float beta_fast_l      = use_compress_rope ? beta_fast : 0.0f;
    const float beta_slow_l      = use_compress_rope ? beta_slow : 0.0f;
    const int32_t n_ctx_orig_l   = use_compress_rope ? n_ctx_orig : 0;

    const auto rope = [&](ggml_tensor * t, ggml_tensor * pos, int64_t n_nope) {
        t = ggml_rope_ext(ctx0, t, pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
                freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
        return ggml_rope_set_offset(t, n_nope);
    };

    ggml_tensor * qr = build_lora_mm(layer.wq_a, cur);
    qr = build_norm(qr, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
    cb(qr, "qr_norm", il);

    ggml_tensor * q = build_lora_mm(layer.wq_b, qr);
    q = ggml_reshape_3d(ctx0, q, n_embd_head, n_head, nt);
    q = rope(q, inp_pos, n_embd_head_nope);
    cb(q, "q", il);

    ggml_tensor * kv = build_lora_mm(layer.wkv, cur);
    kv = build_norm(kv, layer.attn_kv_norm, nullptr, LLM_NORM_RMS, il);
    kv = ggml_reshape_3d(ctx0, kv, n_embd_head, 1, nt);
    kv = rope(kv, inp_pos, n_embd_head_nope);
    cb(kv, "kv", il);

    ggml_tensor * out = nullptr;

    if (ratio == 0) {
        out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
    } else {
        GGML_ASSERT(ratio == 1 || ratio == 2);
        const bool use_csa = ratio == 2;
        const auto & ci = use_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        const auto * comp_ctx   = use_csa ? inp_dsv4->mctx->get_csa() : inp_dsv4->mctx->get_hca();
        const auto * comp_state = use_csa ? inp_dsv4->mctx->get_csa_state() : inp_dsv4->mctx->get_hca_state();
        const int32_t src = hparams.dsv41_kv_src[il];
        GGML_ASSERT(src >= 0 && (int64_t) comp_state->get_ratio() == ratio);

        // ---- compressor (kv sources only): pool ratio tokens, norm, index K, rope, write [kv | index K]
        if (src == il && ci.state_pos) {
            ggml_tensor * state_kv = build_lora_mm(layer.attn_comp_wkv, cur);
            cb(state_kv, "comp_state_kv", il);
            ggml_tensor * state_score = ratio > 1 ? build_lora_mm(layer.attn_comp_wgate, cur) : ggml_scale(ctx0, state_kv, 0.0f);
            cb(state_score, "comp_state_score", il);

            ggml_tensor * restored_kv    = nullptr;
            ggml_tensor * restored_score = nullptr;
            dsv41_state_restore(ctx0, ci, comp_state, il, &restored_kv, &restored_score);

            if (ci.state_write_idxs) {
                const int64_t n_blocks = ci.state_write_pos->ne[0];
                GGML_ASSERT(ci.state_read_idxs->ne[0] == ratio*n_blocks);

                ggml_tensor * src_kv = ggml_concat(ctx0,
                        dsv41_view_2d(ctx0, restored_kv, restored_kv->ne[0], comp_state->get_n_rows(), 0), state_kv, 1);
                ggml_tensor * src_score = ggml_concat(ctx0,
                        dsv41_view_2d(ctx0, restored_score, restored_score->ne[0], comp_state->get_n_rows(), 0), state_score, 1);

                ggml_tensor * v = ggml_get_rows(ctx0, src_kv, ci.state_read_idxs);
                ggml_tensor * s = ggml_get_rows(ctx0, src_score, ci.state_read_idxs);
                v = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, v, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));
                s = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, s, n_embd_head, ratio, n_blocks), 1, 0, 2, 3));

                // per-channel softmax over the ratio slots of a group
                ggml_tensor * comp = ggml_sum_rows(ctx0, ggml_mul(ctx0, v, ggml_soft_max(ctx0, s))); // [1, hd, nb]
                comp = ggml_cont(ctx0, ggml_permute(ctx0, comp, 1, 0, 2, 3));                         // [hd, 1, nb]
                comp = build_norm(comp, layer.attn_comp_norm, nullptr, LLM_NORM_RMS, il);
                cb(comp, "comp_latent", il);

                // index K from the RoPE-free latent; a latent stands for the first token of its group
                ggml_tensor * ik = build_lora_mm(layer.indexer_attn_k, comp); // [n_idx_embd, 1, nb]
                ik = build_norm(ik, layer.indexer_k_norm, nullptr, LLM_NORM_RMS, il);
                ik = rope(ik, ci.state_write_pos, n_idx_embd - n_embd_head_rope);
                cb(ik, "comp_index_k", il);

                comp = rope(comp, ci.state_write_pos, n_embd_head_nope);
                cb(comp, "comp_kv", il);

                ggml_tensor * row = ggml_concat(ctx0, comp, ik, 0); // [hd + n_idx_embd, 1, nb]
                ggml_build_forward_expand(gf, comp_ctx->cpy_k(ctx0, row, ci.state_write_idxs, il));
            }

            // rollback snapshots and the persistent partial-group state (as V4 HCA)
            if (ci.state_snapshot_src_idxs && ci.state_snapshot_dst_idxs) {
                ggml_tensor * snap_kv    = ggml_concat(ctx0, restored_kv,    state_kv,    1);
                ggml_tensor * snap_score = ggml_concat(ctx0, restored_score, state_score, 1);
                ggml_build_forward_expand(gf, comp_state->cpy_kv(ctx0,
                            ggml_get_rows(ctx0, snap_kv, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
                ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0,
                            ggml_get_rows(ctx0, snap_score, ci.state_snapshot_src_idxs), ci.state_snapshot_dst_idxs, il));
            }

            ggml_build_forward_expand(gf, comp_state->cpy_kv(ctx0,
                        ggml_get_rows(ctx0, state_kv, ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
            ggml_build_forward_expand(gf, comp_state->cpy_score(ctx0,
                        ggml_get_rows(ctx0, state_score, ci.state_persist_src_idxs), ci.state_persist_dst_idxs, il));
        }

        if (!ci.kq_mask) {
            // no compressed row visible to any token of this ubatch: window only
            out = build_raw_attention(inp_attn, q, kv, layer.attn_sinks, kq_scale, il);
        } else {
            const int64_t n_comp = ci.kq_mask->ne[0];

            ggml_tensor * comp_all = comp_ctx->get_k(ctx0, src); // [hd + n_idx_embd, 1, n_cells, n_stream]
            GGML_ASSERT(comp_all->ne[0] == n_embd_head + n_idx_embd);
            GGML_ASSERT(n_comp <= comp_all->ne[2]);
            const int64_t n_stream = comp_all->ne[3];

            ggml_tensor * comp_mask = ci.kq_mask;
            if (comp_mask->type != GGML_TYPE_F32) {
                comp_mask = ggml_cast(ctx0, comp_mask, GGML_TYPE_F32);
            }

            // ---- top-k: own indexer on index sources, reused otherwise
            const int32_t topk_src = hparams.dsv41_topk_src[il];
            ggml_tensor * top_k = nullptr;
            if (topk_src == il) {
                ggml_tensor * ik = ggml_view_4d(ctx0, comp_all, n_idx_embd, 1, n_comp, n_stream,
                        comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], ggml_row_size(comp_all->type, n_embd_head));
                ik = ggml_cont(ctx0, ik);
                cb(ik, "idx_k", il);

                ggml_tensor * iq = build_lora_mm(layer.indexer_attn_q_b, qr);
                iq = ggml_reshape_3d(ctx0, iq, n_idx_embd, n_idx_head, nt);
                iq = rope(iq, inp_pos, n_idx_embd - n_embd_head_rope);
                cb(iq, "idx_q", il);

                ggml_tensor * iw = build_lora_mm(layer.indexer_proj, cur);
                iw = ggml_scale(ctx0, iw, 1.0f/sqrtf(float(n_idx_embd*n_idx_head)));
                cb(iw, "idx_w", il);

                iq = ggml_view_4d(ctx0, iq, iq->ne[0], iq->ne[1], nt/n_stream, n_stream,
                        iq->nb[1], iq->nb[2], iq->nb[2]*(nt/n_stream), 0);
                iw = ggml_view_4d(ctx0, iw, iw->ne[0], nt/n_stream, 1, n_stream,
                        iw->nb[1], iw->nb[1]*(nt/n_stream), iw->nb[1]*(nt/n_stream), 0);

                // score[j, t] = sum_h relu(q_h . k_j) * w_h
                ggml_tensor * kq = ggml_mul_mat(ctx0,
                        ggml_permute(ctx0, ik, 0, 2, 1, 3),  // [d, n_comp, 1, ns]
                        ggml_permute(ctx0, iq, 0, 2, 1, 3)); // [d, nt/ns, n_ih, ns] -> [n_comp, nt/ns, n_ih, ns]
                kq = ggml_cont(ctx0, ggml_permute(ctx0, kq, 2, 1, 0, 3));        // [n_ih, nt/ns, n_comp, ns]
                ggml_tensor * score = ggml_sum_rows(ctx0, ggml_mul(ctx0, ggml_relu(ctx0, kq), iw)); // [1, nt/ns, n_comp, ns]
                score = ggml_cont(ctx0, ggml_permute(ctx0, score, 2, 1, 0, 3));  // [n_comp, nt/ns, 1, ns]
                score = ggml_add(ctx0, score, comp_mask);
                cb(score, "idx_score", il);

                const int32_t cand_src = hparams.dsv41_cand_src;
                if (cand_src == il && cand_pin) {
                    // level one: best position per block, pin the newest block, keep the top blocks
                    const int64_t bs    = hparams.dsv41_cand_block_size;
                    const int64_t n_blk = (n_comp + bs - 1)/bs;
                    GGML_ASSERT(cand_pin->ne[0] == n_blk);

                    ggml_tensor * s2 = ggml_reshape_2d(ctx0, score, n_comp, nt);
                    if (n_blk*bs != n_comp) {
                        ggml_tensor * pad = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_blk*bs - n_comp, nt);
                        s2 = ggml_concat(ctx0, s2, ggml_fill(ctx0, pad, -INFINITY), 0);
                    }
                    ggml_tensor * bmax = ggml_pool_1d(ctx0, s2, GGML_OP_POOL_MAX, bs, bs, 0); // [n_blk, nt]
                    bmax = ggml_add(ctx0, bmax, ggml_reshape_2d(ctx0, cand_pin, n_blk, nt));
                    cb(bmax, "cand_block_score", il);

                    const int64_t k_b = std::min<int64_t>(hparams.dsv41_cand_topk_blocks, n_blk);
                    ggml_tensor * bsel = ggml_top_k(ctx0, bmax, k_b); // [k_b, nt]

                    ggml_tensor * bm = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, n_blk, nt, 1), -INFINITY);
                    ggml_tensor * zeros = ggml_fill(ctx0, ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 1, k_b, nt, 1), 0.0f);
                    bm = ggml_set_rows(ctx0, bm, zeros, ggml_reshape_3d(ctx0, bsel, k_b, nt, 1)); // [1, n_blk, nt, 1]
                    bm = ggml_repeat_4d(ctx0, bm, bs, n_blk, nt, 1);
                    bm = ggml_reshape_2d(ctx0, bm, bs*n_blk, nt);
                    bm = ggml_cont(ctx0, ggml_view_2d(ctx0, bm, n_comp, nt, bm->nb[1], 0));
                    cand_mask = ggml_reshape_4d(ctx0, bm, n_comp, nt/n_stream, 1, n_stream);
                    cb(cand_mask, "cand_mask", il);
                } else if (cand_src >= 0 && cand_src < il && cand_mask) {
                    // level two: own scores, restricted to the source's candidate blocks
                    GGML_ASSERT(cand_mask->ne[0] == n_comp);
                    score = ggml_add(ctx0, score, cand_mask);
                    cb(score, "idx_score_cand", il);
                }

                const int64_t k = std::min<int64_t>(n_comp, hparams.indexer_top_k);
                top_k = ggml_cont(ctx0, ggml_top_k(ctx0, score, k));
                cb(top_k, "idx_top_k", il);
                topk_of[il] = top_k;
            } else {
                GGML_ASSERT(topk_src >= 0);
                top_k = topk_of[topk_src];
                GGML_ASSERT(top_k && "DeepSeek-V4.1: top-k source layer produced no top-k");
            }

            {
                // per-layer alias of the top-k this layer attends with (reused on consumers), for selection parity
                ggml_tensor * sel = ggml_view_2d(ctx0, top_k, top_k->ne[0], top_k->ne[1]*top_k->ne[2]*top_k->ne[3], top_k->nb[1], 0);
                cb(sel, "sel_top_k", il);
                ggml_build_forward_expand(gf, sel);
            }

            // ---- sparse attention over [window raw KV | selected compressed rows]
            ggml_tensor * k_rot = inp_attn->self_k_rot;
            if (k_rot) {
                q  = llama_mul_mat_hadamard(ctx0, q, k_rot);
                kv = llama_mul_mat_hadamard(ctx0, kv, k_rot);
            }

            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, kv);

            const llama_kv_cache_dsv4_raw_context * mctx_raw = inp_attn->mctx;
            ggml_build_forward_expand(gf, mctx_raw->cpy_k(ctx0, kv, inp_attn->get_k_idxs(), il));

            ggml_tensor * raw_k = mctx_raw->get_k(ctx0, il);

            ggml_tensor * comp_k = ggml_view_4d(ctx0, comp_all, n_embd_head, 1, n_comp, n_stream,
                    comp_all->nb[1], comp_all->nb[2], comp_all->nb[3], 0);
            if (comp_k->type != raw_k->type) {
                comp_k = ggml_cast(ctx0, comp_k, raw_k->type);
            } else {
                comp_k = ggml_cont(ctx0, comp_k);
            }
            if (k_rot) {
                comp_k = llama_mul_mat_hadamard(ctx0, comp_k, k_rot);
            }
            cb(comp_k, "comp_k", il);

            ggml_tensor * k_all = ggml_concat(ctx0, raw_k, comp_k, 2);

            ggml_tensor * raw_mask = inp_attn->get_kq_mask();
            ggml_tensor * sel_mask = build_top_k_mask(ci.kq_mask, top_k, "comp_top_k_mask", il);
            ggml_tensor * kq_mask  = ggml_concat(ctx0, raw_mask, sel_mask, 0);
            cb(kq_mask, "comp_kq_mask", il);

            out = build_attn_mha(q, k_all, k_all, nullptr, kq_mask, layer.attn_sinks, nullptr, 0, kq_scale, il);
            if (k_rot) {
                out = llama_mul_mat_hadamard(ctx0, out, k_rot);
            }
            cb(out, "attn_comp", il);
        }
    }

    // undo the query rotation on the rope dims, then the grouped low-rank output projection
    out = ggml_reshape_3d(ctx0, out, n_embd_head, n_head, nt);
    out = ggml_rope_ext_back(ctx0, out, inp_pos, nullptr, n_embd_head_rope, rope_type, n_ctx_orig_l,
            freq_base_l, freq_scale_l, ext_factor_l, attn_factor_l, beta_fast_l, beta_slow_l);
    out = ggml_rope_set_offset(out, n_embd_head_nope);
    cb(out, "attn_derope", il);

    out = ggml_reshape_3d(ctx0, out, o_group_dim, n_groups, nt);
    out = ggml_permute(ctx0, out, 0, 2, 1, 3);
    ggml_tensor * oa = ggml_mul_mat(ctx0, layer.wo_a, out);
    cb(oa, "attn_wo_a", il);
    oa = ggml_permute(ctx0, oa, 0, 2, 1, 3);
    oa = ggml_cont_2d(ctx0, oa, o_lora_rank*n_groups, nt);

    out = build_lora_mm(layer.wo_b, oa);
    cb(out, "attn_out", il);

    return out;
}

llama_model_deepseek41::graph::graph(const llama_model & model_base, const llm_graph_params & params) :
    llama_model_deepseek4::graph(params) {
    const auto & model = static_cast<const llama_model_deepseek41 &>(model_base);

    topk_of.assign(n_layer, nullptr);

    ggml_tensor * cur;

    ggml_tensor * inp = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos = build_inp_pos();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    llm_graph_input_dsv4 * inp_dsv4 = build_inp_dsv4();
    ggml_build_forward_expand(gf, inp_dsv4->get_raw()->self_kq_mask);

    // host-side inputs
    auto inp41 = std::make_unique<llm_graph_input_dsv41>(model, inp_dsv4->mctx);
    std::vector<ggml_tensor *> engram_ids(model.engram_layers.size(), nullptr);
    for (size_t e = 0; e < engram_ids.size(); ++e) {
        engram_ids[e] = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, (int64_t) model.engram_n_cols()*n_tokens);
        ggml_set_input(engram_ids[e]);
        ggml_set_name(engram_ids[e], ("engram_ids-" + std::to_string(model.engram_layers[e])).c_str());
    }
    inp41->engram_ids = engram_ids;
    ggml_tensor * engram_keep = nullptr;
    if (!engram_ids.empty() && ubatch.embd) {
        engram_keep = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, 1, 1, n_tokens);
        ggml_set_input(engram_keep);
        ggml_set_name(engram_keep, "engram_keep");
        inp41->engram_keep = engram_keep;
    }
    ggml_tensor * cand_pin = nullptr;
    if (hparams.dsv41_cand_src >= 0) {
        const int32_t cs = hparams.dsv41_cand_src;
        inp41->cand_csa = hparams.dsv4_compress_ratios[cs] == 2;
        const auto & ci = inp41->cand_csa ? inp_dsv4->get_csa() : inp_dsv4->get_hca();
        if (ci.kq_mask) {
            const int64_t bs = hparams.dsv41_cand_block_size;
            cand_pin = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, (ci.kq_mask->ne[0] + bs - 1)/bs,
                    ci.kq_mask->ne[1], 1, ci.kq_mask->ne[3]);
            ggml_set_input(cand_pin);
            ggml_set_name(cand_pin, "cand_pin");
            inp41->cand_pin = cand_pin;
        }
    }
    res->add_input(std::move(inp41));

    const int64_t hc = hparams.dsv4_hc_mult;
    ggml_tensor * inpL = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inpL = ggml_repeat_4d(ctx0, inpL, n_embd, hc, n_tokens, 1);
    cb(inpL, "hc_init", -1);

    // single-pass mHC: the pre-mix used to collapse the stream lags one sublayer
    ggml_tensor * pre_prev = nullptr;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int e = model.engram_index(il);
        if (e >= 0) {
            inpL = build_engram(model, inpL, engram_ids[e], engram_keep, il);
            cb(inpL, "engram_out", il);
        }

        if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) {
            res->t_layer_inp[il] = build_hc_mean(inpL);
            cb(res->t_layer_inp[il], "layer_inp", il);
            ggml_build_forward_expand(gf, res->t_layer_inp[il]);
        }

        ggml_tensor * residual = inpL;
        ggml_tensor * post = nullptr;
        ggml_tensor * comb = nullptr;

        ggml_tensor * attn_pre = build_hc_mixes(inpL, layer.hc_attn_fn, layer.hc_attn_scale, layer.hc_attn_base, &post, &comb, il);

        if (pre_prev == nullptr) {
            // identity pre-mix of the first layer: copy 0
            cur = ggml_cont(ctx0, ggml_view_2d(ctx0, inpL, n_embd, n_tokens, inpL->nb[2], 0));
        } else {
            cur = build_hc_pre(inpL, pre_prev, il);
        }
        cb(cur, "hc_attn_pre", il);

        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        cur = build_attention41(model, inp_dsv4, cur, inp_pos, cand_pin, il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        cb(inpL, "hc_attn_post", il);

        residual = inpL;
        ggml_tensor * ffn_pre = build_hc_mixes(inpL, layer.hc_ffn_fn, layer.hc_ffn_scale, layer.hc_ffn_base, &post, &comb, il);

        cur = build_hc_pre(inpL, attn_pre, il);
        cb(cur, "hc_ffn_pre", il);

        ggml_build_forward_expand(gf, residual);
        ggml_build_forward_expand(gf, post);
        ggml_build_forward_expand(gf, comb);

        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * exp_probs_b = layer.ffn_exp_probs_b;
        ASSERT_EMBD_OR_TOKEN(ubatch);
        if (ubatch.embd != nullptr && layer.ffn_exp_probs_b_vl) {
            exp_probs_b = layer.ffn_exp_probs_b_vl;
        }

        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                exp_probs_b,
                n_expert, hparams.n_expert_used(),
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
        cb(moe_out, "ffn_moe_out", il);

        ggml_tensor * ffn_shexp = build_ffn(cur,
                layer.ffn_up_shexp, nullptr, nullptr,
                layer.ffn_gate_shexp, nullptr, nullptr,
                layer.ffn_down_shexp, nullptr, nullptr,
                nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(ffn_shexp, "ffn_shexp", il);

        cur = ggml_add(ctx0, moe_out, ffn_shexp);
        cb(cur, "ffn_out", il);

        inpL = build_hc_post(cur, residual, post, comb, il);
        inpL = build_cvec(inpL, il);
        cb(inpL, "l_last", il);

        // flat [n_embd*hc, n_tokens] alias for parity tooling (compare_ref.py reads l_out-<L>)
        ggml_tensor * l_out = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
        cb(l_out, "l_out", il);
        ggml_build_forward_expand(gf, l_out);

        pre_prev = ffn_pre;
    }

    if ((size_t) n_layer < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[n_layer]) {
        res->t_layer_inp[n_layer] = build_hc_mean(inpL);
        cb(res->t_layer_inp[n_layer], "layer_inp", n_layer);
        ggml_build_forward_expand(gf, res->t_layer_inp[n_layer]);
    }

    ggml_tensor * flat = ggml_reshape_2d(ctx0, inpL, n_embd*hc, n_tokens);
    if (inp_out_ids) {
        flat     = ggml_get_rows(ctx0, flat, inp_out_ids);
        pre_prev = ggml_get_rows(ctx0, pre_prev, inp_out_ids);
    }
    res->t_h_nextn = flat;

    // no hc_head: collapse with the last FFN's pre-mix
    inpL = ggml_reshape_3d(ctx0, flat, n_embd, hc, flat->ne[1]);
    cur = build_hc_pre(inpL, pre_prev, -1);
    cb(cur, "hc_out", -1);

    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
