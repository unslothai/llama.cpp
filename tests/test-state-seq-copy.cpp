// [TAG_STATE_ASYNC] guards on the asynchronous per-sequence state transfer
//
// llama_state_seq_copy_get / _set take a size and a flags word from the caller and hand both
// to an io object that validates everything else against them. The buffer belongs to the
// transfer, so a size larger than it is refused rather than believed, and
// LLAMA_STATE_SEQ_FLAGS_ON_DEVICE is refused because these copies serialise through host
// memory. This also checks that the buffer reports itself as page-locked only while it holds
// memory that is.
//
// Skipped, not failed, on a backend that cannot copy asynchronously: there is no transfer to
// make and the synchronous calls are what a caller uses there.

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

    // put something in the cache to copy: two sequences interleaved, so the cells of each
    // are a comb rather than one block, which is what the transfer is built for
    std::vector<llama_token> tokens(60, 1);

    llama_batch batch = llama_batch_init(params.n_parallel*tokens.size(), 0, 1);
    for (size_t i = 0; i < tokens.size(); i++) {
        for (int s = 0; s < params.n_parallel; ++s) {
            common_batch_add(batch, tokens[i], i, {s}, false);
        }
    }
    batch.logits[batch.n_tokens - 1] = true;

    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "%s : failed to decode\n", __func__);
        llama_batch_free(batch);
        return 1;
    }

    llama_batch_free(batch);

    llama_state_seq_copy * cpy = llama_state_seq_copy_init(ctx);

    if (cpy == nullptr) {
        fprintf(stderr, "%s : this backend cannot copy sequence states asynchronously, skipping\n", __func__);
        return 0;
    }

    const int          seq_id = 1;
    const size_t       size   = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE);

    CHECK(size > 0);

    // nothing is allocated yet, so nothing is page-locked yet, whatever the backend offers
    CHECK(llama_state_seq_copy_buf_is_pinned(cpy) == false);
    CHECK(llama_state_seq_copy_buf(cpy) == nullptr);

    CHECK(llama_state_seq_copy_buf_resize(cpy, size) != nullptr);
    CHECK(llama_state_seq_copy_buf_size(cpy) == size);

    fprintf(stderr, "%s : seq %d state is %zu bytes, %s host memory (backend offers %s)\n",
            __func__, seq_id, size,
            llama_state_seq_copy_buf_is_pinned(cpy) ? "pinned" : "pageable",
            llama_state_seq_copy_buf_can_pin(cpy)   ? "pinned" : "pageable");

    // a size beyond the buffer the transfer owns is refused, on both directions
    CHECK(llama_state_seq_copy_get(cpy, size + 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, size + 1, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);

    // and so is an empty one, which cannot even hold the header
    CHECK(llama_state_seq_copy_get(cpy, 0, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, 0, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == 0);

    // ON_DEVICE keeps the tensor data off the host, which is where these copies go
    CHECK(llama_state_seq_copy_get(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0);
    CHECK(llama_state_seq_copy_set(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) == 0);

    // none of that may have posted anything
    CHECK(llama_state_seq_copy_done(cpy));

    fprintf(stderr, "%s : oversized, empty and ON_DEVICE transfers are all refused\n", __func__);

    // the same call at the size the transfer does own still works, and round-trips
    std::vector<uint8_t> before(llama_state_seq_get_size(ctx, seq_id));
    CHECK(llama_state_seq_get_data(ctx, before.data(), before.size(), seq_id) == before.size());

    CHECK(llama_state_seq_copy_get(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
    llama_state_seq_copy_wait(cpy);

    llama_memory_seq_rm(llama_get_memory(ctx), seq_id, -1, -1);

    CHECK(llama_state_seq_copy_set(cpy, size, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE) == size);
    llama_state_seq_copy_wait(cpy);

    std::vector<uint8_t> after(llama_state_seq_get_size(ctx, seq_id));
    CHECK(after.size() == before.size());
    CHECK(llama_state_seq_get_data(ctx, after.data(), after.size(), seq_id) == after.size());
    CHECK(before == after);

    fprintf(stderr, "%s : a transfer at the buffer's own size round-trips seq %d byte-for-byte\n",
            __func__, seq_id);

    // giving the memory back leaves nothing page-locked to report
    llama_state_seq_copy_buf_free(cpy);
    CHECK(llama_state_seq_copy_buf_is_pinned(cpy) == false);
    CHECK(llama_state_seq_copy_buf_capacity(cpy) == 0);

    llama_state_seq_copy_free(cpy);

    fprintf(stderr, "%s : SUCCESS\n", __func__);

    return 0;
}
