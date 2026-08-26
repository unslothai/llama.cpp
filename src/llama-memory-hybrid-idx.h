#pragma once

#include "llama-memory-hybrid.h"

#include <memory>
#include <unordered_map>
#include <vector>

//
// llama_memory_hybrid_idx
//

// llama_memory_hybrid plus a third cache holding one indexer key per token, for hybrid
// architectures whose attention layers are block-sparse (qwen4exp QSA).
//
// this is a separate llama_memory type rather than an option on llama_memory_hybrid so that
// nothing in the hybrid path used by the other architectures changes. it duplicates
// llama_memory_hybrid::init_batch because the indexer cache must be given the attention
// cache's slot layout instead of finding its own, and that layout is not observable from
// outside the returned context.
//
// the indexer cache is a side buffer addressed by the attention cache's cells: same size,
// same padding, same stream count, same slots, so cell j means the same token in both.
// everything that depends on that layout is computed host-side in set_input_qsa; the model
// graph only gathers, pools and scores.

class llama_memory_hybrid_idx : public llama_memory_hybrid {
public:
    llama_memory_hybrid_idx(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                            /* the indexer cache exists only if this is given */
    const layer_filter_cb & filter_idx);

    ~llama_memory_hybrid_idx() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0)       override;

    //
    // llama_memory_hybrid_idx specific API
    //

    llama_kv_cache * get_mem_idx() const;   // nullptr when the model carries no indexer

    // [TAG_PLE_HISTORY]
    // The qwen4exp PLE hash of a token mixes in the ple_ngram_size - 1 tokens before it.
    // A decode ubatch does not carry them, so they are remembered here (vLLM's
    // ngram_context).
    //
    // This lives on the memory rather than on llama_model because it is per-context
    // per-sequence state, not a model weight: a llama_model is shared by every context
    // that loads it, so a map on the model keyed only by llama_seq_id let two contexts
    // (two server instances on one model, or a draft/target pair) overwrite each other's
    // history. It is also state that has to survive save/restore and to follow seq_rm,
    // seq_cp, seq_keep, seq_add and seq_div, and this class already does exactly that
    // bookkeeping for the caches next to it.
    struct ple_history {
        // position the next token of this sequence must have. -1 means "unknown": the
        // window is not trusted and the hash falls back to EOS padding.
        llama_pos next_pos = -1;

        // the tokens at positions [next_pos - toks.size(), next_pos), oldest first.
        // never longer than ple_ngram_size - 1, and may be shorter near a sequence start
        // or after a rewind - callers pad the missing front with EOS.
        std::vector<llama_token> toks;
    };

    // history for seq_id, default-constructed (and therefore untrusted) on first use.
    // const + mutable because it is read and updated from set_input, which runs off a
    // const memory context.
    ple_history & ple_hist_get(llama_seq_id seq_id) const;

private:
    // the indexer cache stores only one key head per layer, so it needs its own hparams
    // instance: llama_kv_cache keeps a reference to whatever it is given
    llama_hparams hparams_idx;

    const std::unique_ptr<llama_kv_cache> mem_idx;

    // [TAG_PLE_HISTORY] empty for every architecture but qwen4exp, which is the only one
    // whose graph asks for a history
    mutable std::unordered_map<llama_seq_id, ple_history> ple_hist;

    // the seq_* halves of the history bookkeeping, one per llama_memory_i operation
    void ple_hist_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1);
    void ple_hist_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1);
    void ple_hist_keep(llama_seq_id seq_id);
    void ple_hist_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift);
    void ple_hist_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1);

    void ple_hist_state_write(llama_io_write_i & io, llama_seq_id seq_id) const;
    void ple_hist_state_read (llama_io_read_i  & io, llama_seq_id seq_id);
};

class llama_memory_hybrid_idx_context : public llama_memory_hybrid_context {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    explicit llama_memory_hybrid_idx_context(llama_memory_status status);

    // used to create a full-cache context
    explicit llama_memory_hybrid_idx_context(llama_memory_hybrid_idx * mem);

    // used to create an update context
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                      llama_context * lctx,
                               bool   optimize);

    // used to create a batch processing context from a batch
    llama_memory_hybrid_idx_context(
            llama_memory_hybrid_idx * mem,
                    slot_info_vec_t   sinfos_attn,
                    slot_info_vec_t   sinfos_idx,
          std::vector<llama_ubatch>   ubatches);

    ~llama_memory_hybrid_idx_context() = default;

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    //
    // llama_memory_hybrid_idx_context specific API
    //

    // nullptr when the model carries no indexer, and for the full and update contexts,
    // which do not drive the sparse-attention graph
    const llama_kv_cache_context * get_idx() const;

    // streams in the current slot info, matching get_k/get_v's `ns`. 1 if unified.
    uint32_t get_n_stream() const;

    // [TAG_PLE_HISTORY] the owning memory's per-sequence n-gram history, for set_input
    llama_memory_hybrid_idx::ple_history & get_ple_hist(llama_seq_id seq_id) const;

    // block-compressed sparse attention (qwen4exp QSA) over the indexer cache's cells.
    // blocks cut the *position* line, not the cell array, so nothing assumes a contiguous
    // layout:
    //   cell_blk  I32 [n_kv, ns]           block each cell belongs to
    //   blk_cells I32 [ratio*n_blocks, ns] cells making up each block
    //   blk_pos   I32 [4*n_blocks*ns]      mrope position rows of each block's first token
    //   bias      F32 [n_kv, n_tokens/ns, ns] -inf where invisible, large where always visible
    void set_input_qsa(ggml_tensor * cell_blk, ggml_tensor * blk_cells, ggml_tensor * blk_pos,
                       ggml_tensor * bias, const llama_ubatch * ubatch, uint32_t ratio) const;

private:
    const llama_memory_hybrid_idx * mem = nullptr;

    // streams per ubatch, taken from the slot infos before they are handed to ctx_idx.
    // declared first so that it is initialised while sinfos_idx is still intact
    const std::vector<uint32_t> ns_ubatch;

    // null unless the model has an indexer and this is a batch context
    const llama_memory_context_ptr ctx_idx;

    // mirrors the base class's ubatch cursor, which is private there
    size_t i_cur = 0;
};
