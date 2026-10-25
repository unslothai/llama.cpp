#pragma once

#include <cstdint>

// A speculative verification reads all of a draft's logits from one decode.
// Keep its contiguous [first, last] indices together when retrying a batch.
// The retry limit can become smaller than a single draft: in that case decode
// the whole draft, so the existing n_batch == 1 exhaustion path still terminates.
inline int32_t server_speculative_batch_end(int32_t off, int32_t end, int32_t first, int32_t last) {
    if (first < end && end <= last) {
        return first > off ? first : last + 1;
    }
    return end;
}
