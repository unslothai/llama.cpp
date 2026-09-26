#include "server-queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_set>

// recv() can be called with ids that are not in the waiting list: a cancel or a cleanup drops
// them between the caller posting and the caller arriving. That has to park this one connection
// and nothing else, which is what the unbounded wait did before the per-waiter queues.
//
// It must not assert. GGML_ASSERT is GGML_ABORT, and recv() runs on the HTTP thread, so a
// single dropped request would take the whole server down for every other client. The parent
// commit dies inside recv() here, in well under the 2.5 s this waits.
int main() {
    server_response res;

    std::atomic<bool> returned{false};

    std::thread parked([&] {
        server_task_result_ptr r = res.recv(std::unordered_set<int>{4242});
        (void) r;
        returned.store(true);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(2500));

    if (returned.load()) {
        fprintf(stderr, "FAIL: recv() returned for ids that are not in the waiting list\n");
        return 1;
    }

    // the timeout form already tolerated this, and must keep doing so promptly
    const auto t0 = std::chrono::steady_clock::now();
    server_task_result_ptr none = res.recv_with_timeout(std::unordered_set<int>{4243}, 1);
    const auto waited = std::chrono::steady_clock::now() - t0;

    if (none != nullptr) {
        fprintf(stderr, "FAIL: recv_with_timeout() invented a result\n");
        return 1;
    }
    if (waited > std::chrono::seconds(5)) {
        fprintf(stderr, "FAIL: recv_with_timeout() did not honour its timeout\n");
        return 1;
    }

    printf("OK: a dropped request parks its own caller and leaves the server up\n");

    // parked is still inside recv() by design: terminate() would make it std::terminate(),
    // which is the documented behaviour for an HTTP caller, so leave without joining it.
    parked.detach();
    fflush(stdout);
    _Exit(0);
}
