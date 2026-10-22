// [TAG_EXACT_CONCURRENCY] tests the paged KV pool removal and restore paths, with LLAMA_KV_CACHE_DEBUG=1 cross-checking page ownership every ubatch.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const uint32_t PAGE = 256;

static bool decode_range(llama_context * ctx, llama_seq_id seq, llama_pos first, llama_pos last) {
    common_batch batch(ctx);

    bool ok = true;

    for (llama_pos p = first; p <= last && ok; ) {
        batch.clear();

        for (int i = 0; i < 64 && p <= last; ++i, ++p) {
            batch.add(1, p, seq, i == 63 || p == last);
        }

        ok = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    }

    return ok;
}

static void set_env_default(const char * name, const char * value) {
    if (getenv(name)) {
        return;
    }
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 0);
#endif
}

int main(int argc, char ** argv) {
    set_env_default("LLAMA_EXACT_CONCURRENCY", "1");
    set_env_default("LLAMA_KV_CACHE_DEBUG",    "1");

    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified    = true;
    params.n_parallel    = 2;
    params.n_ctx         = 2*4*PAGE;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    params.n_gpu_layers = 999;

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (llama_init->model() == nullptr) {
        fprintf(stderr, "%s : the model did not load\n", __func__);
        return 1;
    }

    // the test runs only when a model was named for it, so a context that cannot be made is a failure, never a pass
    if (ctx == nullptr) {
        fprintf(stderr, "%s : FAIL, no exact-concurrency context for this build and model (needs CUDA and 256-wide K/V heads)\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    const uint32_t gran = llama_memory_alloc_granularity(mem);
    if (gran != PAGE) {
        fprintf(stderr, "%s : allocation granularity is %u, expected %u\n", __func__, gran, PAGE);
        return 1;
    }

    {
        // a negative position is an invalid batch: decode refuses it, it never reaches the page allocator
        common_batch batch(ctx);
        batch.add(1, -1, 1, true);

        const int32_t ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get());

        if (ret >= 0 || llama_memory_seq_pos_max(mem, 1) != -1) {
            fprintf(stderr, "%s : a token at position -1 was not refused as an invalid batch (decode returned %d)\n", __func__, ret);
            return 1;
        }
    }

    if (!decode_range(ctx, 0, 0, 599)) {
        fprintf(stderr, "%s : failed to fill sequence 0\n", __func__);
        return 1;
    }

    llama_memory_seq_cp(mem, 0, 1, -1, -1);

    if (llama_memory_seq_pos_max(mem, 1) != -1 || llama_memory_seq_pos_max(mem, 0) != 599) {
        fprintf(stderr, "%s : a refused copy left sequence 1 at %d and sequence 0 at %d\n", __func__,
                llama_memory_seq_pos_max(mem, 1), llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    if (!llama_memory_seq_rm(mem, 0, 600, -1) || llama_memory_seq_pos_max(mem, 0) != 599) {
        fprintf(stderr, "%s : a removal past the tail changed the sequence, its end is %d\n",
                __func__, llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    if (!decode_range(ctx, 0, 600, 655)) {
        fprintf(stderr, "%s : failed to continue sequence 0 after a removal that removed nothing\n", __func__);
        return 1;
    }

    {
        // 656 tokens hold 3 pages; the occupancy-only count (block skip) must agree with the count that also sizes reclaim
        const llama_seq_id s0[] = { 0, 0 };
        int32_t reclaim = -1;
        const int32_t n_fast = llama_memory_n_cells_used(mem, nullptr, 0, nullptr);
        const int32_t n_full = llama_memory_n_cells_used(mem, s0, 2, &reclaim);
        if (n_fast != 3*(int32_t) PAGE || n_full != n_fast || reclaim != n_fast) {
            fprintf(stderr, "%s : cells used %d (occupancy only) / %d (with reclaim %d), expected %u for 656 tokens\n",
                    __func__, n_fast, n_full, reclaim, 3*PAGE);
            return 1;
        }
    }

    const bool holes = llama_memory_seq_rm(mem, 0, 1, 511);

    if (holes && llama_memory_seq_pos_max(mem, 0) != 655) {
        fprintf(stderr, "%s : a partial removal changed the end of the sequence: %d\n", __func__,
                llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    printf("%s : interior removal %s\n", __func__, holes ? "left holes" : "was refused, skipping the hole case");

    if (!decode_range(ctx, 1, 0, 5*PAGE - 1)) {
        fprintf(stderr, "%s : failed to fill the pages sequence 0 does not hold\n", __func__);
        return 1;
    }

    if (decode_range(ctx, 1, 5*PAGE, 5*PAGE)) {
        fprintf(stderr, "%s : the pool allocated a page that sequence 0 still holds\n", __func__);
        return 1;
    }

    if (!llama_memory_seq_rm(mem, 0, -1, -1) || llama_memory_seq_pos_max(mem, 0) != -1) {
        fprintf(stderr, "%s : sequence 0 is still in the pool after a full removal\n", __func__);
        return 1;
    }

    if (!decode_range(ctx, 1, 5*PAGE, 8*PAGE - 1)) {
        fprintf(stderr, "%s : the three pages of the removed sequence were not reusable\n", __func__);
        return 1;
    }

    if (decode_range(ctx, 1, 8*PAGE, 8*PAGE)) {
        fprintf(stderr, "%s : the pool allocated a ninth page\n", __func__);
        return 1;
    }

    {
        std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 1));
        state.resize(llama_state_seq_get_data(ctx, state.data(), state.size(), 1));

        const size_t off_pos = 4*sizeof(uint32_t);

        uint32_t head[4] = {};
        int32_t  pos0    = -1;
        if (state.size() >= off_pos + sizeof(pos0)) {
            memcpy(head,  state.data(),           sizeof(head));
            memcpy(&pos0, state.data() + off_pos, sizeof(pos0));
        }

        if (head[1] != 1 || head[2] != 1 || head[3] != 8*PAGE || pos0 < 0 || pos0 >= (int32_t) (8*PAGE)) {
            fprintf(stderr, "%s : unexpected state layout: sequence %u, %u streams, %u cells, first position %d\n",
                    __func__, head[1], head[2], head[3], pos0);
            return 1;
        }

        std::vector<uint8_t> bad = state;
        const int32_t neg = -1;
        memcpy(bad.data() + off_pos, &neg, sizeof(neg));

        if (llama_state_seq_set_data(ctx, bad.data(), bad.size(), 1) != 0) {
            fprintf(stderr, "%s : a state with a negative position was restored\n", __func__);
            return 1;
        }

        if (llama_state_seq_set_data(ctx, state.data(), state.size(), 1) == 0 ||
            llama_memory_seq_pos_max(mem, 1) != (llama_pos) (8*PAGE - 1)) {
            fprintf(stderr, "%s : the unpatched state did not restore, sequence 1 ends at %d\n",
                    __func__, llama_memory_seq_pos_max(mem, 1));
            return 1;
        }

        std::vector<uint8_t> whole(state.size() + 4096);

        if (llama_state_get_size(ctx) != 0 || llama_state_get_data(ctx, whole.data(), whole.size()) != 0) {
            fprintf(stderr, "%s : a whole-context state was saved\n", __func__);
            return 1;
        }

        const char * session = "test-exact-pages.session";
        std::remove(session);

        const llama_token tok = 1;
        FILE * f = nullptr;
        if (llama_state_save_file(ctx, session, &tok, 1) || (f = fopen(session, "rb")) != nullptr) {
            if (f) {
                fclose(f);
            }
            std::remove(session);
            fprintf(stderr, "%s : a whole-context session file was written\n", __func__);
            return 1;
        }

        if (llama_state_set_data(ctx, state.data(), state.size()) != 0) {
            fprintf(stderr, "%s : a whole-context state was restored\n", __func__);
            return 1;
        }
    }

    printf("%s : ok, page ownership survived a no-op removal, holes and a full removal, and a negative restored position and a whole-context save and restore were refused\n", __func__);

    return 0;
}
