// [TAG_STATE_ASYNC] guards on the asynchronous per-sequence state transfer; skipped where it is unavailable.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

#define CHECK(cond)                                                          \
    do {                                                                     \
        if (!(cond)) {                                                       \
            fprintf(stderr, "%s : FAILED at line %d: %s\n", __func__,        \
                    __LINE__, #cond);                                        \
            return 1;                                                        \
        }                                                                    \
    } while (0)

int main(int argc, char ** argv) {
    common_params params;

    params.sampling.seed = 1234;
    params.kv_unified    = true;
    params.n_parallel    = 2;
    params.n_ctx         = 256;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();

    if (llama_init->model() == nullptr || ctx == nullptr) {
        fprintf(stderr, "%s : failed to init\n", __func__);
        return 1;
    }

    // two interleaved sequences, so each one's cells are a comb rather than one block
    std::vector<llama_token> tokens(60, 1);

    common_batch batch(ctx);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            batch.add(tokens[i], i, s, false);
        }
    }
    batch.set_output(batch.size() - 1, true);

    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        return 1;
    }

    llama_state_seq_copy * cpy = llama_state_seq_copy_init(ctx);

    if (cpy == nullptr) {
        fprintf(stderr, "%s : this backend cannot copy sequence states asynchronously, skipping\n", __func__);
        return 0;
    }

    const int          seq_id = 1;
    const size_t       size   = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);

    CHECK(size > 0);

    CHECK(llama_state_seq_copy_buf_is_pinned(cpy) == false);
    CHECK(llama_state_seq_copy_buf(cpy) == nullptr);

    CHECK(llama_state_seq_copy_buf_resize(cpy, size) != nullptr);
    CHECK(llama_state_seq_copy_buf_size(cpy) == size);

    fprintf(stderr, "%s : seq %d state is %zu bytes, %s host memory (backend offers %s)\n",
            __func__, seq_id, size,
            llama_state_seq_copy_buf_is_pinned(cpy) ? "pinned" : "pageable",
            llama_state_seq_copy_buf_can_pin(cpy)   ? "pinned" : "pageable");

    CHECK(llama_state_seq_copy_get(cpy, size + 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, size + 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);

    CHECK(llama_state_seq_copy_get(cpy, 0, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, 0, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);

    CHECK(llama_state_seq_copy_get(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0);

    CHECK(llama_state_seq_copy_done(cpy));

    fprintf(stderr, "%s : oversized, empty and ON_DEVICE transfers are all refused\n", __func__);

    // a transfer that fails part way must post nothing, so the caller can reuse the buffer at once
    CHECK(llama_state_seq_copy_get(cpy, size - 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_n_copies(cpy) == 0);
    CHECK(llama_state_seq_copy_done(cpy));

    fprintf(stderr, "%s : a transfer one byte short is refused and posts no copies\n", __func__);

    std::vector<uint8_t> before(llama_state_seq_get_size(ctx, seq_id));
    CHECK(llama_state_seq_get_data(ctx, before.data(), before.size(), seq_id) == before.size());

    CHECK(llama_state_seq_copy_get(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
    llama_state_seq_copy_wait(cpy);

    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);

    CHECK(llama_state_seq_copy_set(cpy, size - 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_n_copies(cpy) == 0);
    CHECK(llama_state_seq_copy_done(cpy));

    CHECK(llama_state_seq_copy_set(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
    llama_state_seq_copy_wait(cpy);

    std::vector<uint8_t> after(llama_state_seq_get_size(ctx, seq_id));
    CHECK(after.size() == before.size());
    CHECK(llama_state_seq_get_data(ctx, after.data(), after.size(), seq_id) == after.size());
    CHECK(before == after);

    fprintf(stderr, "%s : a transfer at the buffer's own size round-trips seq %d byte-for-byte\n",
            __func__, seq_id);

    llama_state_seq_copy_buf_free(cpy);
    CHECK(llama_state_seq_copy_buf_is_pinned(cpy) == false);
    CHECK(llama_state_seq_copy_buf_capacity(cpy) == 0);

    llama_state_seq_copy_free(cpy);

    fprintf(stderr, "%s : SUCCESS\n", __func__);

    return 0;
}
