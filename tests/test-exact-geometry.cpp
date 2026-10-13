// [TAG_EXACT_CONCURRENCY] the batch must hold a whole ubatch beside a decode step of every slot, or a prefill gets a shorter remainder.

#include "common.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

int main() {
    int n_min = 0;

    assert(common_exact_batch_geometry(2048, 512, 4, &n_min));
    assert(n_min == 516);

    // One decoder beside the prompt used to leave it 511 tokens with only a warning.
    assert(!common_exact_batch_geometry(512, 512, 1, &n_min));
    assert(n_min == 513);

    assert(!common_exact_batch_geometry(512, 512, 2, &n_min));
    assert(n_min == 514);

    assert(common_exact_batch_geometry(514, 512, 2, &n_min) && n_min == 514);
    assert(!common_exact_batch_geometry(513, 512, 2, &n_min) && n_min == 514);

    assert(!common_exact_batch_geometry(512, 512, 1));
    assert(common_exact_batch_geometry(1024, 512, 1));

    assert(!common_exact_batch_geometry(2048, 0, 4, &n_min));
    assert(n_min == 2052);

    assert(!common_exact_batch_geometry(512, 4096, 1, &n_min));
    assert(n_min == 513);

    // A context of 256 cells clamps n_batch and n_ubatch to equal sizes, leaving no column for a decode step.
    assert(!common_exact_batch_geometry(256, 256, 2, &n_min));
    assert(n_min == 258);

    assert(common_exact_batch_geometry(512, 512, 0, &n_min));
    assert(n_min == 512);

    printf("%s: all tests passed\n", __func__);

    return 0;
}
