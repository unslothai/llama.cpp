// Unit test for server_response, the result queue between the decode loop and the HTTP threads.
//
// It exercises the public API directly, so the awkward cases are injected rather than waited
// for: a reader whose ids were dropped between posting and receiving, a send that races a
// cancel, per id and bulk teardown, broadcast, and concurrent registration and removal.
//
// Run with "leak <n>" to measure what the queue retains over n reader lifecycles, each leaving
// one result queued at teardown, which is what a client disconnect during generation does.

#include "server-queue.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

static int g_fail = 0;

static void check(bool ok, const char * name, const char * detail = "") {
    printf("%-46s %s %s\n", name, ok ? "PASS" : "FAIL", detail);
    if (!ok) { g_fail++; }
}

// minimal concrete result so we can put things on the queue without a model
struct fake_result : server_task_result {
    int payload = 0;
    bool stop = false;
    fake_result(int id_, int payload_, bool stop_) : payload(payload_), stop(stop_) { id = id_; }
    bool is_stop() override { return stop; }
    json to_json() override { return json{{"payload", payload}}; }
    server_task_result * clone() const override { return new fake_result(*this); }
};

static server_task_result_ptr mk(int id, int payload, bool stop = true) {
    return server_task_result_ptr(new fake_result(id, payload, stop));
}

static int payload_of(const server_task_result_ptr & p) {
    return p ? static_cast<fake_result *>(p.get())->payload : -1;
}

using ms = std::chrono::milliseconds;

// ---------------------------------------------------------------------------

// a result for an id that has left the waiting list is dropped, silently, no crash
static void t_send_to_absent_id() {
    server_response res;
    res.send(mk(1, 100));                       // never registered
    res.add_waiting_task_id(2);
    res.remove_waiting_task_id(2);
    res.send(mk(2, 200));                       // registered then removed
    res.add_waiting_task_id(3);
    res.send(mk(3, 300));
    auto got = res.recv_with_timeout({3}, 1);
    check(got != nullptr && payload_of(got) == 300, "send to absent id is dropped, live id still delivered");
}

// FIFO order per reader is what scanning the shared vector from the front used to give
static void t_fifo_order() {
    server_response res;
    res.add_waiting_task_id(7);
    for (int i = 0; i < 32; i++) { res.send(mk(7, i, i == 31)); }
    bool ok = true;
    for (int i = 0; i < 32; i++) {
        auto r = res.recv_with_timeout({7}, 1);
        if (payload_of(r) != i) { ok = false; break; }
    }
    check(ok, "FIFO order preserved for a single reader");
}

// two independent readers must not see each other's results
static void t_reader_isolation() {
    server_response res;
    res.add_waiting_task_ids({10, 11});
    res.add_waiting_task_ids({20, 21});
    res.send(mk(20, 2000));
    res.send(mk(10, 1000));
    res.send(mk(21, 2100));
    res.send(mk(11, 1100));

    std::vector<int> a, b;
    for (int i = 0; i < 2; i++) { a.push_back(payload_of(res.recv_with_timeout({10, 11}, 1))); }
    for (int i = 0; i < 2; i++) { b.push_back(payload_of(res.recv_with_timeout({20, 21}, 1))); }
    const bool ok = a.size() == 2 && b.size() == 2 &&
        (a[0] == 1000 && a[1] == 1100) && (b[0] == 2000 && b[1] == 2100);
    char d[128];
    snprintf(d, sizeof(d), "a=[%d,%d] b=[%d,%d]", a[0], a[1], b[0], b[1]);
    check(ok, "two readers do not steal each other's results", d);
    check(res.recv_with_timeout({10, 11}, 1) == nullptr, "reader A drained, no extra result");
}

// per-id removal must drop only that id's results
static void t_partial_removal() {
    server_response res;
    res.add_waiting_task_ids({30, 31});
    res.send(mk(30, 3000));
    res.send(mk(31, 3100));
    res.remove_waiting_task_id(30);
    auto r = res.recv_with_timeout({31}, 1);
    check(r != nullptr && payload_of(r) == 3100, "removing one id keeps the sibling's result");
    check(res.recv_with_timeout({31}, 1) == nullptr, "the removed id's result is gone");
}

// bulk removal then a late send: nothing delivered, nothing leaked, no use after free
static void t_bulk_removal() {
    server_response res;
    res.add_waiting_task_ids({40, 41, 42});
    res.send(mk(40, 4000));
    res.remove_waiting_task_ids({40, 41, 42});
    res.send(mk(41, 4100));
    check(res.recv_with_timeout({40, 41, 42}, 1) == nullptr, "bulk removal drops queued and late results");
}

// broadcast: one copy per registered id, id overridden
static void t_broadcast() {
    server_response res;
    res.add_waiting_task_ids({50, 51});
    res.add_waiting_task_id(60);
    res.broadcast(mk(-1, 9999));
    auto a1 = res.recv_with_timeout({50, 51}, 1);
    auto a2 = res.recv_with_timeout({50, 51}, 1);
    auto a3 = res.recv_with_timeout({50, 51}, 1);
    auto b1 = res.recv_with_timeout({60}, 1);
    auto b2 = res.recv_with_timeout({60}, 1);
    const bool ok = a1 && a2 && !a3 && b1 && !b2 &&
                    payload_of(a1) == 9999 && payload_of(b1) == 9999;
    check(ok, "broadcast delivers one copy per registered id");
    const bool ids_ok = a1 && a2 && (a1->id == 50 || a1->id == 51) && (a2->id == 50 || a2->id == 51) && a1->id != a2->id && b1 && b1->id == 60;
    check(ids_ok, "broadcast overrides the result id per target");
}

// lost wakeup: park a reader on ids that do not exist yet, then create them and send.
// Both arms must deliver; the head's condition_gone poll bounds the delay.
static void t_late_registration() {
    server_response res;
    std::atomic<int> got{-2};
    std::atomic<bool> done{false};
    std::thread th([&] {
        for (int i = 0; i < 60; i++) {
            auto r = res.recv_with_timeout({70}, 1);
            if (r) { got.store(payload_of(r)); break; }
        }
        done.store(true);
    });
    std::this_thread::sleep_for(ms(300));
    res.add_waiting_task_id(70);
    res.send(mk(70, 7000));
    const auto t0 = std::chrono::steady_clock::now();
    while (!done.load() && std::chrono::steady_clock::now() - t0 < std::chrono::seconds(10)) {
        std::this_thread::sleep_for(ms(10));
    }
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();
    th.join();
    char d[64]; snprintf(d, sizeof(d), "%lldms", (long long) waited);
    check(got.load() == 7000, "id registered after the reader parked is still served", d);
}

// a reader whose ids were dropped parks itself and nothing else.
// It must not assert: GGML_ASSERT is GGML_ABORT, and recv() runs on the HTTP thread, so one
// dropped request would take the whole server down for every other client.
static void t_parked_reader_does_not_abort() {
    static server_response res;   // static: the parked thread outlives this function
    std::atomic<bool> returned{false};
    std::thread parked([&] {
        auto r = res.recv(std::unordered_set<int>{4242});
        (void) r;
        returned.store(true);
    });
    std::this_thread::sleep_for(ms(2500));
    check(!returned.load(), "recv() on dropped ids parks instead of returning garbage");

    // and the rest of the queue keeps working while that thread is parked
    res.add_waiting_task_id(80);
    res.send(mk(80, 8000));
    auto r = res.recv_with_timeout({80}, 2);
    check(r != nullptr && payload_of(r) == 8000, "other readers unaffected by a parked reader");
    parked.detach();
}

// recv_with_timeout honours its timeout when the waiter exists but is empty
static void t_timeout_honoured() {
    server_response res;
    res.add_waiting_task_id(90);
    const auto t0 = std::chrono::steady_clock::now();
    auto r = res.recv_with_timeout({90}, 1);
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();
    char d[64]; snprintf(d, sizeof(d), "%lldms", (long long) waited);
    check(r == nullptr && waited >= 900 && waited < 5000, "recv_with_timeout honours its timeout", d);
}

// concurrent churn: producers, consumers, registration and teardown all at once.
// This is the case ThreadSanitizer is pointed at.
static void t_stress() {
    server_response res;
    const int n_readers = 16;
    const int n_msgs    = 200;
    std::atomic<int> received{0};
    std::atomic<bool> go{false};

    std::vector<std::thread> readers;
    for (int r = 0; r < n_readers; r++) {
        readers.emplace_back([&, r] {
            while (!go.load()) { std::this_thread::yield(); }
            const int base_id = 1000 + r * 10;
            std::unordered_set<int> ids{base_id, base_id + 1};
            res.add_waiting_task_ids(ids);
            int seen = 0;
            while (seen < n_msgs) {
                auto p = res.recv_with_timeout(ids, 1);
                if (!p) { break; }
                seen++;
                received.fetch_add(1);
            }
            res.remove_waiting_task_ids(ids);
        });
    }

    std::vector<std::thread> writers;
    for (int w = 0; w < 4; w++) {
        writers.emplace_back([&, w] {
            while (!go.load()) { std::this_thread::yield(); }
            for (int i = w; i < n_msgs * n_readers; i += 4) {
                const int r = (i / n_msgs) % n_readers;
                res.send(mk(1000 + r * 10 + (i % 2), i, false));
                if ((i & 63) == 0) { std::this_thread::sleep_for(ms(1)); }
            }
        });
    }

    // a churn thread that registers and drops ids nobody waits for
    std::thread churn([&] {
        while (!go.load()) { std::this_thread::yield(); }
        for (int i = 0; i < 2000; i++) {
            res.add_waiting_task_id(500000 + i);
            res.send(mk(500000 + i, i, false));
            res.remove_waiting_task_id(500000 + i);
        }
    });

    go.store(true);
    for (auto & t : writers) { t.join(); }
    for (auto & t : readers) { t.join(); }
    churn.join();
    char d[64]; snprintf(d, sizeof(d), "received=%d", received.load());
    check(received.load() > 0, "concurrent send/recv/register/remove churn survives", d);
}

// API hazard probe: recv() with a strict subset of the ids that share one waiter.
// Not reachable from server_response_reader (it always passes the whole set), reported
// as a latent sharp edge rather than a defect.
static void t_subset_recv_probe() {
    server_response res;
    res.add_waiting_task_ids({200, 201});
    res.send(mk(201, 2010));
    auto r = res.recv_with_timeout({200}, 1);
    printf("%-46s %s (id=%d)\n", "PROBE recv() with a subset of a shared waiter",
           r == nullptr ? "returns nullptr (id filtered)" : "RETURNS THE SIBLING'S RESULT",
           r ? r->id : -1);
}

static long rss_kb() {
    FILE * f = fopen("/proc/self/status", "r");
    if (!f) { return -1; }
    char line[256];
    long v = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) { sscanf(line + 6, "%ld", &v); break; }
    }
    fclose(f);
    return v;
}

// Isolated measurement of what the result queue itself retains. One reader lifecycle per
// iteration: register two ids, receive one result, leave one result queued (which is what a
// disconnect during generation does), then tear the reader down the way stop() does.
static void t_leak(long n) {
    server_response res;
    const long rss0 = rss_kb();
    for (long i = 0; i < n; i++) {
        const int a = 100000 + (int) (i * 2);
        const int b = a + 1;
        res.add_waiting_task_ids({a, b});
        res.send(mk(a, 1));
        res.send(mk(b, 2));                 // left unconsumed on purpose
        auto got = res.recv_with_timeout({a, b}, 1);
        (void) got;
        res.remove_waiting_task_ids({a, b}); // exactly what server_response_reader::stop() does
    }
    const long rss1 = rss_kb();
    printf("LEAK n=%ld rss_start=%ld kB rss_end=%ld kB growth=%ld kB (%.3f kB per reader)\n",
           n, rss0, rss1, rss1 - rss0, (double) (rss1 - rss0) / (double) n);
}

int main(int argc, char ** argv) {
    if (argc > 1 && strcmp(argv[1], "leak") == 0) {
        t_leak(argc > 2 ? atol(argv[2]) : 200000);
        fflush(stdout);
        _Exit(0);
    }
    t_send_to_absent_id();
    t_fifo_order();
    t_reader_isolation();
    t_partial_removal();
    t_bulk_removal();
    t_broadcast();
    t_late_registration();
    t_timeout_honoured();
    t_stress();
    t_subset_recv_probe();
    t_parked_reader_does_not_abort();

    printf("\nRESULT queue failures=%d\n", g_fail);
    fflush(stdout);
    _Exit(g_fail == 0 ? 0 : 1);   // a thread is parked in recv() by design
}
