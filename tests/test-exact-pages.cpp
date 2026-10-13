// [TAG_EXACT_CONCURRENCY] drives the removal paths of the paged KV pool - a removal that empties nothing, one that leaves holes, the pages those holes keep reserved, and a restore whose position has no page - while LLAMA_KV_CACHE_DEBUG=1 makes the pool cross-check page ownership against the live cells every ubatch.
// Needs a CUDA build with 256-wide heads and a fully offloaded F16 cache; without one the test reports what it skipped and passes.

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
            batch.add(1, p, seq, false);
        }

        // every decode asks for one set of logits, so none of them is a batch with no output
        batch.set_output(batch.size() - 1, true);

        ok = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0;
    }

    return ok;
}

// Windows has no setenv
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
    // read before the model is loaded: both are latched on first use
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

    // after the parser, which requires the default here
    params.n_gpu_layers = 999;

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (llama_init->model() == nullptr || ctx == nullptr) {
        printf("%s : skipped, this build and model cannot run exact concurrency\n", __func__);
        return 0;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    const uint32_t gran = llama_memory_alloc_granularity(mem);
    if (gran != PAGE) {
        fprintf(stderr, "%s : allocation granularity is %u, expected %u\n", __func__, gran, PAGE);
        return 1;
    }

    // positions 0..599 of sequence 0: three pages, the last one part full
    if (!decode_range(ctx, 0, 0, 599)) {
        fprintf(stderr, "%s : failed to fill sequence 0\n", __func__);
        return 1;
    }

    // a page belongs to one sequence, so a cross-sequence copy is refused whole rather than half
    // applied: the pool logs the refusal and leaves the destination empty and the source as it was
    llama_memory_seq_cp(mem, 0, 1, -1, -1);

    if (llama_memory_seq_pos_max(mem, 1) != -1 || llama_memory_seq_pos_max(mem, 0) != 599) {
        fprintf(stderr, "%s : a refused copy left sequence 1 at %d and sequence 0 at %d\n", __func__,
                llama_memory_seq_pos_max(mem, 1), llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    // the removal every accepted speculative step makes: a rejected tail that is not there. It
    // must leave the pool alone, ownership included
    if (!llama_memory_seq_rm(mem, 0, 600, -1) || llama_memory_seq_pos_max(mem, 0) != 599) {
        fprintf(stderr, "%s : a removal past the tail changed the sequence, its end is %d\n",
                __func__, llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    if (!decode_range(ctx, 0, 600, 655)) {
        fprintf(stderr, "%s : failed to continue sequence 0 after a removal that removed nothing\n", __func__);
        return 1;
    }

    // holes: positions 1 to 510 go, 0 and 511 to 655 stay, so the first two pages each keep a live
    // cell and neither is free for another sequence. A hybrid memory refuses to remove the middle
    // of a sequence, and then there is nothing to check here
    const bool holes = llama_memory_seq_rm(mem, 0, 1, 511);

    if (holes && llama_memory_seq_pos_max(mem, 0) != 655) {
        fprintf(stderr, "%s : a partial removal changed the end of the sequence: %d\n", __func__,
                llama_memory_seq_pos_max(mem, 0));
        return 1;
    }

    printf("%s : interior removal %s\n", __func__, holes ? "left holes" : "was refused, skipping the hole case");

    // sequence 1 fills what is left of the pool. The pool holds 8 pages and sequence 0 holds 3 of
    // them, holes and a part full tail page included, so 5 remain
    if (!decode_range(ctx, 1, 0, 5*PAGE - 1)) {
        fprintf(stderr, "%s : failed to fill the pages sequence 0 does not hold\n", __func__);
        return 1;
    }

    // one page more than the pool has left: it has to refuse rather than take a page that still
    // has a live cell in it
    if (decode_range(ctx, 1, 5*PAGE, 5*PAGE)) {
        fprintf(stderr, "%s : the pool allocated a page that sequence 0 still holds\n", __func__);
        return 1;
    }

    // a whole sequence goes back to the pool as whole pages, holes included
    if (!llama_memory_seq_rm(mem, 0, -1, -1) || llama_memory_seq_pos_max(mem, 0) != -1) {
        fprintf(stderr, "%s : sequence 0 is still in the pool after a full removal\n", __func__);
        return 1;
    }

    if (!decode_range(ctx, 1, 5*PAGE, 8*PAGE - 1)) {
        fprintf(stderr, "%s : the three pages of the removed sequence were not reusable\n", __func__);
        return 1;
    }

    // the pool is full again
    if (decode_range(ctx, 1, 8*PAGE, 8*PAGE)) {
        fprintf(stderr, "%s : the pool allocated a ninth page\n", __func__);
        return 1;
    }

    // a restored cell with a negative position has no page: the restore fails as a bad state does,
    // and the same state unpatched still restores. After the magic and the sequence id, the
    // attention cache comes first, as n_stream, cell_count, then the first cell's position
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

        // a whole-context restore is refused up front in this mode, through the same error return
        std::vector<uint8_t> whole(llama_state_get_size(ctx));
        whole.resize(llama_state_get_data(ctx, whole.data(), whole.size()));

        if (whole.empty() || llama_state_set_data(ctx, whole.data(), whole.size()) != 0) {
            fprintf(stderr, "%s : a whole-context state was %s\n", __func__, whole.empty() ? "not saved" : "restored");
            return 1;
        }
    }

    printf("%s : ok, page ownership survived a no-op removal, holes and a full removal, and a negative restored position and a whole-context restore were refused\n", __func__);

    return 0;
}
