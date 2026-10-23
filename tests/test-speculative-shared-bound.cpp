// [TAG_PREEMPT] DSpark and DFlash2 decode every drafting sequence's block in one batch and split it into equal blocks, so the sequences
// share one draft bound: the configured n_max lowered to the tightest per-call bound. Different bounds (here 1 and 2 with n_max 3)
// once built blocks of different lengths, which the DSpark markov head asserts against.

#include "speculative.h"

#include <cstdio>
#include <vector>

static int check(const char * what, int got, int want) {
    printf("%-58s %3d (want %d)\n", what, got, want);
    return got != want;
}

int main() {
    std::vector<common_speculative_draft_params> dp(3);

    int bad = 0;

    bad += check("no sequence drafting", common_speculative_n_draft_shared(dp, 3), 3);

    dp[0].drafting = true; dp[0].n_max = 1;
    dp[1].drafting = true; dp[1].n_max = 2;
    bad += check("drafting with per-call bounds 1 and 2, n_max 3", common_speculative_n_draft_shared(dp, 3), 1);

    dp[0].n_max = -1;
    bad += check("one unbounded, one bounded at 2", common_speculative_n_draft_shared(dp, 3), 2);

    dp[1].drafting = false;
    dp[2].drafting = false; dp[2].n_max = 1;
    bad += check("bounds of sequences not drafting are ignored", common_speculative_n_draft_shared(dp, 3), 3);

    dp[0].n_max = 5;
    dp[1].drafting = true; dp[1].n_max = 4;
    bad += check("per-call bounds above n_max leave n_max", common_speculative_n_draft_shared(dp, 3), 3);

    printf("%s\n", bad == 0 ? "ok" : "FAIL");
    return bad == 0 ? 0 : 1;
}
