#include "../tools/server/server-speculative-batch.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static void require(bool value) {
    if (!value) {
        std::abort();
    }
}

int main() {
    // The old halving scheme cuts a three-token verification at index 2.
    require(server_speculative_batch_end(0, 2, 0, 2) == 3);
    require(server_speculative_batch_end(0, 4, 3, 5) == 3);
    require(server_speculative_batch_end(3, 4, 3, 5) == 6);

    int cases = 0;
    for (int a = 1; a <= 16; ++a) {
        for (int b = 1; b <= 16; ++b) {
            for (int plain = 0; plain <= 4; ++plain) {
                const int prefix = plain; // non-speculative generation before a draft
                const std::vector<std::pair<int, int>> groups = {
                    {prefix, prefix + a - 1}, {prefix + a + plain, prefix + a + plain + b - 1}};
                const int total = prefix + a + plain + b + 7; // pending prompt after generation
                for (int limit = 1; limit <= total + 1; ++limit) {
                    for (bool reverse : {false, true}) {
                        auto ordered = groups;
                        if (reverse) {
                            std::reverse(ordered.begin(), ordered.end());
                        }
                        int views = 0;
                        for (int off = 0; off < total;) {
                            int end = std::min(off + limit, total);
                            for (const auto & group : ordered) {
                                end = server_speculative_batch_end(off, end, group.first, group.second);
                            }
                            require(end > off && end <= total);
                            for (const auto & group : groups) {
                                if (group.first < end && group.second >= off) {
                                    require(group.first >= off && group.second < end);
                                    for (int i = group.first; i <= group.second; ++i) {
                                        require(i - off >= 0 && i - off < end - off);
                                    }
                                }
                            }
                            off = end;
                            require(++views <= total); // also tests retry-limit 1 terminates
                        }
                        ++cases;
                    }
                }
            }
        }
    }
    std::printf("PASS: %d speculative batch partitions\n", cases);
}
