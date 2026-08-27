// fault injection for the qwen4exp indexer cache restore.
//
// the indexer cache is addressed by the cells of the attention cache, so cell j must hold the
// same token in both. a restore that fails partway is the case that breaks it: the attention
// cache keeps what it read and the indexer keeps what it had. nothing here can be reached by a
// blob that state_write produced, so the faults are injected: blobs are truncated, and the
// mirrored layout the indexer is handed is corrupted by hand.
//
// PART 1 uses only the public state API and compiles against the unfixed tree as well, so it
// can be run either side of the fix to show it fails without it.
// PART 2 calls llama_kv_cache::state_read_sinfo directly and needs the fix.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include "../src/llama-io.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-kv-cells.h"
#include "../src/llama-memory-hybrid-idx.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#ifndef FAULTS_HAVE_SINFO
#define FAULTS_HAVE_SINFO 1
#endif

static int g_failures = 0;
static int g_checks   = 0;

static void check(bool ok, const std::string & what) {
    ++g_checks;
    if (!ok) {
        fprintf(stderr, "  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// host io over a plain buffer, as llama-context.cpp has for the public API
// ---------------------------------------------------------------------------

class io_write_buf : public llama_io_write_i {
public:
    void write(const void * src, size_t size) override {
        const uint8_t * p = (const uint8_t *) src;
        buf.insert(buf.end(), p, p + size);
        n += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        const size_t pos = buf.size();
        buf.resize(pos + size);
        ggml_backend_tensor_get(tensor, buf.data() + pos, offset, size);
        n += size;
    }

    size_t n_bytes() override { return n; }

    std::vector<uint8_t> buf;

private:
    size_t n = 0;
};

class io_read_buf : public llama_io_read_i {
public:
    io_read_buf(const uint8_t * p, size_t len) : ptr(p), left(len) {}

    ~io_read_buf() {
        for (const auto & r : pending) {
            ggml_backend_tensor_set(r.tensor, r.ptr, r.offset, r.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > left) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size; left -= size; n += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > left) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        pending.push_back({tensor, ptr, size, offset});
        ptr += size; left -= size; n += size;
    }

    size_t n_bytes() override { return n; }

private:
    struct rinfo { ggml_tensor * tensor; const uint8_t * ptr; size_t size; size_t offset; };

    const uint8_t * ptr;
    size_t left;
    size_t n = 0;
    std::vector<rinfo> pending;
};

// ---------------------------------------------------------------------------
// the property the whole change exists to protect
// ---------------------------------------------------------------------------

// the two caches must agree cell for cell: same occupancy, same position, same owner
static bool caches_in_step(const llama_memory_hybrid_idx * mem, uint32_t n_seq_max, std::string & why) {
    const llama_kv_cache * attn = mem->get_mem_attn();
    const llama_kv_cache * idx  = mem->get_mem_idx();

    for (uint32_t s = 0; s < n_seq_max; ++s) {
        const llama_kv_cells & ca = attn->get_cells(s);
        const llama_kv_cells & ci = idx ->get_cells(s);

        if (ca.size() != ci.size()) {
            why = "cache sizes differ";
            return false;
        }

        for (uint32_t i = 0; i < ca.size(); ++i) {
            if (ca.is_empty(i) != ci.is_empty(i)) {
                why = "cell " + std::to_string(i) + " of stream " + std::to_string(s) +
                      (ca.is_empty(i) ? " is set in the indexer only" : " is set in the attention cache only");
                return false;
            }

            if (ca.is_empty(i)) {
                continue;
            }

            if (ca.pos_get(i) != ci.pos_get(i)) {
                why = "cell " + std::to_string(i) + " holds different positions";
                return false;
            }

            for (uint32_t q = 0; q < n_seq_max; ++q) {
                if (ca.seq_has(i, q) != ci.seq_has(i, q)) {
                    why = "cell " + std::to_string(i) + " has different owners";
                    return false;
                }
            }
        }

        if (attn->get_n_stream() == 1) {
            break;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------

static constexpr int N_SEQ = 4;

struct fixture {
    llama_model         * model = nullptr;
    llama_context       * ctx   = nullptr;
    llama_memory_hybrid_idx * mem = nullptr;
    int n_vocab = 0;
    llama_context_params cparams = {};
};

static llama_token tok_of(int seq, int pos, int n_vocab) {
    return (llama_token) ((13*(unsigned) pos + 101*(unsigned) seq + 3) % (unsigned) n_vocab);
}

static bool decode_seq(llama_context * ctx, int seq, int n_vocab, int pos0, int n_tokens) {
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; ++i) {
        common_batch_add(batch, tok_of(seq, pos0 + i, n_vocab), pos0 + i, { seq }, i + 1 == n_tokens);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// fresh context with seq 0 (16 tokens) and seq 1 (20 tokens) live
static bool fixture_reset(fixture & fx) {
    if (fx.ctx) {
        llama_free(fx.ctx);
        fx.ctx = nullptr;
    }

    fx.ctx = llama_init_from_model(fx.model, fx.cparams);
    if (fx.ctx == nullptr) {
        return false;
    }

    fx.mem = dynamic_cast<llama_memory_hybrid_idx *>((llama_memory_i *) llama_get_memory(fx.ctx));
    if (fx.mem == nullptr || fx.mem->get_mem_idx() == nullptr) {
        fprintf(stderr, "  this model has no indexer cache, nothing to inject into\n");
        return false;
    }

    return decode_seq(fx.ctx, 0, fx.n_vocab, 0, 16) && decode_seq(fx.ctx, 1, fx.n_vocab, 0, 20);
}

// ---------------------------------------------------------------------------
// PART 1: truncate a good blob at many lengths. a truncation that lands past the
// attention section leaves the attention cache restored and the indexer not.
// ---------------------------------------------------------------------------

static void part1_truncation_sweep(fixture & fx) {
    fprintf(stderr, "\n== part 1: truncated blob (public API), %s ==\n", fx.cparams.kv_unified ? "kv-unified" : "kv-per-seq");

    if (!fixture_reset(fx)) {
        fprintf(stderr, "  FAIL: could not set up the fixture\n");
        ++g_failures;
        return;
    }

    std::vector<uint8_t> blob(llama_state_seq_get_size(fx.ctx, 0));
    const size_t n_blob = llama_state_seq_get_data(fx.ctx, blob.data(), blob.size(), 0);
    if (n_blob == 0) {
        fprintf(stderr, "  FAIL: could not save seq 0\n");
        ++g_failures;
        return;
    }
    fprintf(stderr, "  seq 0 blob is %zu bytes\n", n_blob);

    int n_rejected = 0;
    int n_drifted  = 0;

    // 32 cut points across the blob, plus a few right after the head where the
    // attention section is still being read
    std::vector<size_t> cuts;
    for (int i = 1; i < 32; ++i) {
        cuts.push_back(n_blob*i/32);
    }
    for (size_t c : {(size_t) 4, (size_t) 8, (size_t) 16, n_blob - 1}) {
        cuts.push_back(c);
    }

    for (size_t cut : cuts) {
        if (cut == 0 || cut >= n_blob) {
            continue;
        }

        if (!fixture_reset(fx)) {
            fprintf(stderr, "  FAIL: could not reset the fixture\n");
            ++g_failures;
            return;
        }

        const size_t n_set = llama_state_seq_set_data(fx.ctx, blob.data(), cut, 0);
        if (n_set != 0) {
            continue; // the cut did not actually break the read
        }

        ++n_rejected;

        std::string why;
        if (!caches_in_step(fx.mem, N_SEQ, why)) {
            fprintf(stderr, "  cut at %6zu/%zu bytes left the caches out of step: %s\n", cut, n_blob, why.c_str());
            ++n_drifted;
        }

        // seq 1 was never the target of the restore and must survive it
        llama_memory_t m = llama_get_memory(fx.ctx);
        check(llama_memory_seq_pos_max(m, 1) == 19,
                "cut at " + std::to_string(cut) + ": the failed restore of seq 0 damaged seq 1");
    }

    fprintf(stderr, "  %d of %zu cuts were rejected by the reader\n", n_rejected, cuts.size());
    fprintf(stderr, "  %d of those left the two caches out of step\n", n_drifted);

    check(n_rejected >= 8, "too few truncations were rejected for this to test anything");
    check(n_drifted == 0, std::to_string(n_drifted) + " truncated restores left the indexer out of step with the attention cache");
}

#if FAULTS_HAVE_SINFO

// ---------------------------------------------------------------------------
// PART 2: hand the indexer a mirrored layout that is wrong in each of the ways
// the new code rejects. none of these can come out of state_write.
// ---------------------------------------------------------------------------

// the cells seq_id occupies in a cache, in position order, as find_slot would have produced
static llama_kv_cache::slot_info_vec_t layout_of(const llama_kv_cache * kv, llama_seq_id seq_id, uint32_t n_stream) {
    llama_kv_cache::slot_info_vec_t out(n_stream);

    for (uint32_t s = 0; s < n_stream; ++s) {
        const llama_kv_cells & c = kv->get_cells(n_stream == 1 ? 0 : (llama_seq_id) s);

        std::vector<uint32_t> idxs;
        for (uint32_t i = 0; i < c.size(); ++i) {
            if (!c.is_empty(i) && c.seq_has(i, seq_id)) {
                idxs.push_back(i);
            }
        }

        if (idxs.empty()) {
            continue;
        }

        out[s].s0 = s;
        out[s].s1 = s;
        out[s].resize(1);
        out[s].strm[0] = s;
        out[s].idxs[0] = idxs;
    }

    return out;
}

// save the indexer section on its own, so it can be fed back with a doctored layout
static std::vector<uint8_t> save_idx(fixture & fx, llama_seq_id seq_id) {
    io_write_buf io;
    fx.mem->get_mem_idx()->state_write(io, seq_id, 0);
    return io.buf;
}

// returns true if the read was rejected
static bool read_idx(fixture & fx, llama_seq_id seq_id, const std::vector<uint8_t> & blob,
        const llama_kv_cache::slot_info_vec_t * sinfos_in, std::string & err) {
    io_read_buf io(blob.data(), blob.size());
    try {
        fx.mem->get_mem_idx()->state_read_sinfo(io, seq_id, 0, nullptr, sinfos_in);
    } catch (const std::exception & e) {
        err = e.what();
        return true;
    }
    err.clear();
    return false;
}

static void part2_bad_layouts(fixture & fx) {
    fprintf(stderr, "\n== part 2: corrupt mirrored layouts (state_read_sinfo) ==\n");

    if (!fixture_reset(fx)) {
        fprintf(stderr, "  FAIL: could not set up the fixture\n");
        ++g_failures;
        return;
    }

    llama_kv_cache * idx = fx.mem->get_mem_idx();
    const uint32_t n_stream = idx->get_n_stream();

    const std::vector<uint8_t> blob0 = save_idx(fx, 0);
    const llama_kv_cache::slot_info_vec_t good0 = layout_of(idx, 0, n_stream);
    const llama_kv_cache::slot_info_vec_t good1 = layout_of(idx, 1, n_stream);

    fprintf(stderr, "  indexer blob for seq 0 is %zu bytes, n_stream = %u\n", blob0.size(), n_stream);

    std::string err;

    // -- the good layout must be accepted, or nothing below means anything
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        idx = fx.mem->get_mem_idx();
        const llama_kv_cache::slot_info_vec_t g = layout_of(idx, 0, n_stream);
        const bool rejected = read_idx(fx, 0, blob0, &g, err);
        check(!rejected, "the layout the indexer already had was rejected: " + err);
    }

    // -- wrong stream count
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        llama_kv_cache::slot_info_vec_t bad = layout_of(fx.mem->get_mem_idx(), 0, n_stream);
        bad.push_back(llama_kv_cache::slot_info{});
        const bool rejected = read_idx(fx, 0, blob0, &bad, err);
        fprintf(stderr, "  wrong stream count      -> %s\n", rejected ? err.c_str() : "ACCEPTED");
        check(rejected && err.find("stream count") != std::string::npos, "a layout with the wrong stream count was accepted");

        // the check fires before anything is read, so the cache must be untouched
        check(llama_memory_seq_pos_max(llama_get_memory(fx.ctx), 0) == 15,
                "the stream count check dropped seq 0 before it read anything");
    }

    // -- empty layout, blob has cells
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        llama_kv_cache::slot_info_vec_t bad(n_stream);
        const bool rejected = read_idx(fx, 0, blob0, &bad, err);
        fprintf(stderr, "  empty layout            -> %s\n", rejected ? err.c_str() : "ACCEPTED");
        check(rejected, "an empty mirrored layout was accepted for a blob that holds cells");
    }

    // -- layout with the wrong number of cells (seq 1 has 20, seq 0's blob has 16)
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        const llama_kv_cache::slot_info_vec_t bad = layout_of(fx.mem->get_mem_idx(), 1, n_stream);
        const bool rejected = read_idx(fx, 0, blob0, &bad, err);
        fprintf(stderr, "  wrong cell count        -> %s\n", rejected ? err.c_str() : "ACCEPTED");
        check(rejected, "a mirrored layout of the wrong length was accepted");
    }

    // -- layout that points at cells another sequence holds
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        llama_kv_cache::slot_info_vec_t bad = layout_of(fx.mem->get_mem_idx(), 1, n_stream);

        // trim seq 1's cells down to seq 0's cell count, so only the occupancy check can fire
        bool trimmed = false;
        for (uint32_t s = 0; s < n_stream; ++s) {
            if (!bad[s].empty() && bad[s].idxs[0].size() > 16) {
                bad[s].idxs[0].resize(16);
                trimmed = true;
            }
        }

        if (trimmed) {
            const bool rejected = read_idx(fx, 0, blob0, &bad, err);
            fprintf(stderr, "  layout over live cells  -> %s\n", rejected ? err.c_str() : "ACCEPTED");
            check(rejected, "a mirrored layout pointing at another sequence's live cells was accepted");
        } else {
            fprintf(stderr, "  layout over live cells  -> could not build the case for n_stream = %u\n", n_stream);
        }
    }

    // -- layout out of range of the cache
    {
        if (!fixture_reset(fx)) { ++g_failures; return; }
        llama_kv_cache::slot_info_vec_t bad = layout_of(fx.mem->get_mem_idx(), 0, n_stream);
        for (uint32_t s = 0; s < n_stream; ++s) {
            if (!bad[s].empty()) {
                bad[s].idxs[0].back() = 1u << 30;
            }
        }
        const bool rejected = read_idx(fx, 0, blob0, &bad, err);
        fprintf(stderr, "  layout out of range     -> %s\n", rejected ? err.c_str() : "ACCEPTED");
        check(rejected, "a mirrored layout pointing outside the cache was accepted");
    }
}

#endif // FAULTS_HAVE_SINFO

// ---------------------------------------------------------------------------
// PART 3: the same, for a whole-context blob. this one restores every stream, so
// it is where a per-stream reset of the whole cache shows up.
// ---------------------------------------------------------------------------

static void part3_whole_context(fixture & fx) {
    fprintf(stderr, "\n== part 3: whole-context blob, %s ==\n", fx.cparams.kv_unified ? "kv-unified" : "kv-per-seq");

    if (!fixture_reset(fx)) {
        fprintf(stderr, "  FAIL: could not set up the fixture\n");
        ++g_failures;
        return;
    }

    // a third sequence, so that more than two streams carry cells
    if (!decode_seq(fx.ctx, 2, fx.n_vocab, 0, 24)) {
        fprintf(stderr, "  FAIL: could not fill seq 2\n");
        ++g_failures;
        return;
    }

    std::vector<uint8_t> blob(llama_state_get_size(fx.ctx));
    const size_t n_blob = llama_state_get_data(fx.ctx, blob.data(), blob.size());
    blob.resize(n_blob);
    fprintf(stderr, "  whole-context blob is %zu bytes\n", n_blob);

    // a clean round trip must bring every sequence back
    {
        llama_free(fx.ctx);
        fx.ctx = llama_init_from_model(fx.model, fx.cparams);
        fx.mem = dynamic_cast<llama_memory_hybrid_idx *>((llama_memory_i *) llama_get_memory(fx.ctx));

        const size_t n_set = llama_state_set_data(fx.ctx, blob.data(), blob.size());
        check(n_set == blob.size(), "the whole-context restore did not consume the blob");

        llama_memory_t m = llama_get_memory(fx.ctx);
        const llama_pos want[3] = { 15, 19, 23 };
        for (int s = 0; s < 3; ++s) {
            check(llama_memory_seq_pos_max(m, s) == want[s],
                    "whole-context restore lost seq " + std::to_string(s) +
                    " (pos_max " + std::to_string(llama_memory_seq_pos_max(m, s)) + ")");
        }

        std::string why;
        check(caches_in_step(fx.mem, N_SEQ, why), "whole-context restore left the caches out of step: " + why);
    }

    // and a broken one must leave the caches agreeing with each other
    int n_rejected = 0;
    int n_drifted  = 0;

    for (int i = 1; i < 24; ++i) {
        const size_t cut = n_blob*i/24;
        if (cut == 0 || cut >= n_blob) {
            continue;
        }

        llama_free(fx.ctx);
        fx.ctx = llama_init_from_model(fx.model, fx.cparams);
        fx.mem = dynamic_cast<llama_memory_hybrid_idx *>((llama_memory_i *) llama_get_memory(fx.ctx));
        if (!decode_seq(fx.ctx, 0, fx.n_vocab, 0, 16) || !decode_seq(fx.ctx, 1, fx.n_vocab, 0, 20)) {
            ++g_failures;
            return;
        }

        if (llama_state_set_data(fx.ctx, blob.data(), cut) != 0) {
            continue;
        }

        ++n_rejected;

        std::string why;
        if (!caches_in_step(fx.mem, N_SEQ, why)) {
            fprintf(stderr, "  cut at %6zu/%zu bytes left the caches out of step: %s\n", cut, n_blob, why.c_str());
            ++n_drifted;
        }
    }

    fprintf(stderr, "  %d cuts rejected, %d left the caches out of step\n", n_rejected, n_drifted);
    check(n_rejected >= 4, "too few whole-context truncations were rejected for this to test anything");
    check(n_drifted == 0, std::to_string(n_drifted) + " failed whole-context restores left the caches out of step");
}

int main(int argc, char ** argv) {
    common_params params;

    params.n_ctx      = 1024;
    params.n_parallel = N_SEQ;
    params.n_predict  = 0;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    common_init();
    llama_backend_init();
    ggml_backend_load_all();

    fixture fx;

    llama_model_params mparams = common_model_params_to_llama(params);
    fx.model = llama_model_load_from_file(params.model.path.c_str(), mparams);
    if (fx.model == nullptr) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    fx.n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(fx.model));

    fx.cparams = common_context_params_to_llama(params);
    fx.cparams.n_seq_max  = N_SEQ;
    fx.cparams.n_ctx      = 256*N_SEQ;
    fx.cparams.n_batch    = 256;
    fx.cparams.n_ubatch   = 256;
    fx.cparams.kv_unified = true;

    part1_truncation_sweep(fx);

    fx.cparams.kv_unified = false;
    part1_truncation_sweep(fx);
    fx.cparams.kv_unified = true;

#if FAULTS_HAVE_SINFO
    part2_bad_layouts(fx);
#endif

    part3_whole_context(fx);

    fx.cparams.kv_unified = false;
    part3_whole_context(fx);
    fx.cparams.kv_unified = true;

    if (fx.ctx) {
        llama_free(fx.ctx);
    }
    llama_model_free(fx.model);
    llama_backend_free();

    fprintf(stderr, "\n%d checks, %d failed\n", g_checks, g_failures);

    return g_failures > 0 ? 1 : 0;
}
