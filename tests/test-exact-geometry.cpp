// [TAG_EXACT_CONCURRENCY] the batch shape a prefill needs to be split into the ubatches it would
// get alone: the server adds a prompt in whole ubatches, so the batch has to hold one of those
// beside a decode step of every slot, or the prompt is left the shorter remainder.

#include "common.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

int main() {
    int n_min = 0;

    // the reported minimum is the ubatch plus the decode step, whether or not the batch reaches it
    assert(common_exact_batch_geometry(2048, 512, 4, &n_min));
    assert(n_min == 516);

    // the case that used to warn and carry on: one decoder beside the prompt leaves it 511 tokens
    assert(!common_exact_batch_geometry(512, 512, 1, &n_min));
    assert(n_min == 513);

    assert(!common_exact_batch_geometry(512, 512, 2, &n_min));
    assert(n_min == 514);

    // exactly enough, and one short of it
    assert(common_exact_batch_geometry(514, 512, 2, &n_min) && n_min == 514);
    assert(!common_exact_batch_geometry(513, 512, 2, &n_min) && n_min == 514);

    // a single slot with no draft still needs room for its own decoded token
    assert(!common_exact_batch_geometry(512, 512, 1));
    assert(common_exact_batch_geometry(1024, 512, 1));

    // an unset ubatch is the whole batch, which then cannot hold a decode step as well
    assert(!common_exact_batch_geometry(2048, 0, 4, &n_min));
    assert(n_min == 2052);

    // a ubatch larger than the batch is clamped to it, so it cannot pass either
    assert(!common_exact_batch_geometry(512, 4096, 1, &n_min));
    assert(n_min == 513);

    // the shape a context settles on when its size clamps the batch: n_batch becomes min(n_ctx, -b)
    // and n_ubatch min(n_batch, -ub), so a context of 256 cells leaves the two equal and no column
    // for a decode step, whatever -b and -ub asked for
    assert(!common_exact_batch_geometry(256, 256, 2, &n_min));
    assert(n_min == 258);

    // no slot decoding at all: the prompt has the batch to itself
    assert(common_exact_batch_geometry(512, 512, 0, &n_min));
    assert(n_min == 512);

    printf("%s: all tests passed\n", __func__);

    return 0;
}
