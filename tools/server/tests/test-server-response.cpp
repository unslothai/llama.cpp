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

#ifndef _WIN32
#  include <sys/wait.h>
#  include <unistd.h>
#endif

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
// The parent commit of the head asserted here, which is GGML_ABORT on the HTTP thread.
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


// A single timed receive that starts before the ids exist must still return a result that
// arrives during the call. The shared queue used to notify one condition on every send, so a
// parked receiver woke; per waiter queues have to notify registration explicitly or the caller
// sleeps out its whole timeout and reports a spurious nullptr.
static void t_single_timed_recv_before_registration() {
    server_response res;
    std::atomic<bool> started{false};
    std::thread producer([&] {
        while (!started.load()) { std::this_thread::yield(); }
        std::this_thread::sleep_for(ms(200));
        res.add_waiting_task_id(300);
        res.send(mk(300, 3000));
    });
    started.store(true);
    const auto t0 = std::chrono::steady_clock::now();
    auto r = res.recv_with_timeout({300}, 5);          // ONE call, not a retry loop
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();
    producer.join();
    char d[80]; snprintf(d, sizeof(d), "%lldms, %s", (long long) waited, r ? "got result" : "nullptr");
    check(r != nullptr && payload_of(r) == 3000 && waited < 4000,
          "one timed recv sees a result that arrives while it waits", d);
}

// terminate() has to be honoured even when the caller is parked on ids that are not registered.
// recv_with_timeout() promises std::terminate() there, because the caller is HTTP code that
// cannot return. Run in a child: the correct outcome is that the child dies.
static void t_terminate_while_parked_on_absent_ids() {
#ifdef _WIN32
    // needs fork(): the correct outcome is that the caller terminates, which cannot be asserted
    // in-process. The behaviour itself is not platform specific.
    printf("%-46s SKIP (needs fork())\n", "terminate() is honoured while parked on absent ids");
#else
    fflush(stdout);
    pid_t pid = fork();
    if (pid == 0) {
        auto * res = new server_response();
        std::thread killer([res] {
            std::this_thread::sleep_for(ms(300));
            res->terminate();
        });
        auto r = res->recv_with_timeout({9999}, 5);
        killer.join();
        // reaching here at all means terminate() was ignored
        _Exit(r == nullptr ? 20 : 21);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const bool died = WIFSIGNALED(status);
    char d[96];
    if (died) { snprintf(d, sizeof(d), "child died on signal %d", WTERMSIG(status)); }
    else      { snprintf(d, sizeof(d), "child returned %d, terminate() ignored", WEXITSTATUS(status)); }
    check(died, "terminate() is honoured while parked on absent ids", d);
#endif
}

// A result for a sibling id must not be handed to a caller that did not ask for it.
static void t_subset_recv_is_filtered() {
    server_response res;
    res.add_waiting_task_ids({200, 201});
    res.send(mk(201, 2010));
    auto r = res.recv_with_timeout({200}, 1);
    char d[64]; snprintf(d, sizeof(d), "id=%d", r ? r->id : -1);
    check(r == nullptr, "recv() does not return a result for an id it was not asked for", d);
    // and the sibling's result is still there for the caller that does ask
    auto r2 = res.recv_with_timeout({200, 201}, 1);
    check(r2 != nullptr && r2->id == 201, "the sibling's result is still delivered to its own reader");
}


// Two readers taking disjoint subsets of one registration share a waiter, so waking only one of
// them can wake the wrong one: it finds nothing matching, sleeps again, and the reader whose
// result is actually queued sits there until its timeout. The shared condition used to
// notify_all(), so every subset receiver got a look.
static void t_subset_receivers_are_all_woken() {
    server_response res;
    res.add_waiting_task_ids({400, 401});

    std::atomic<int> parked{0};
    std::atomic<int> got_a{-1};
    std::vector<std::thread> others;

    // four readers waiting on the sibling id park first, so a single notify picks one of them
    for (int i = 0; i < 4; i++) {
        others.emplace_back([&] {
            parked.fetch_add(1);
            auto r = res.recv_with_timeout({401}, 3);
            (void) r;
        });
    }
    while (parked.load() < 4) { std::this_thread::yield(); }
    std::this_thread::sleep_for(ms(200));

    std::thread reader_a([&] {
        auto r = res.recv_with_timeout({400}, 3);
        got_a.store(payload_of(r));
    });
    std::this_thread::sleep_for(ms(200));

    const auto t0 = std::chrono::steady_clock::now();
    res.send(mk(400, 4000));
    reader_a.join();
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();
    for (auto & t : others) { t.join(); }

    char d[80]; snprintf(d, sizeof(d), "%lldms, payload=%d", (long long) waited, got_a.load());
    check(got_a.load() == 4000 && waited < 2500,
          "a subset receiver is woken even when siblings wait too", d);
}


// Ids registered by separate calls belong to separate waiters, so no single condition covers a
// receive that names both. The receiver must be woken by a result for either of them, whichever
// waiter the lookup happened to pick, so both directions are driven.
static void t_ids_spanning_two_waiters_one(int base_id, int send_to, const char * label) {
    server_response res;
    res.add_waiting_task_id(base_id);       // two separate registrations, so two waiters
    res.add_waiting_task_id(base_id + 1);

    std::atomic<int> got{-1};
    std::thread reader([&] {
        auto r = res.recv_with_timeout({base_id, base_id + 1}, 3);
        got.store(payload_of(r));
    });
    std::this_thread::sleep_for(ms(300));

    const auto t0 = std::chrono::steady_clock::now();
    res.send(mk(send_to, 5000 + send_to));
    reader.join();
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();

    char d[96]; snprintf(d, sizeof(d), "%lldms, payload=%d", (long long) waited, got.load());
    check(got.load() == 5000 + send_to && waited < 2500, label, d);
}

static void t_ids_spanning_two_waiters() {
    t_ids_spanning_two_waiters_one(500, 500, "a receive over two waiters is woken by the first id");
    t_ids_spanning_two_waiters_one(600, 601, "a receive over two waiters is woken by the second id");
}


// Results must come back in arrival order even when the ids live in different waiters. The
// shared vector scanned from the front, so it did. Both orders are driven, because which waiter
// the lookup reaches first depends on the set's iteration order.
static void t_fifo_across_waiters_one(int first, int second, const char * label) {
    server_response res;
    res.add_waiting_task_id(first);     // separate registrations, so separate waiters
    res.add_waiting_task_id(second);

    res.send(mk(first,  7000 + first));
    res.send(mk(second, 7000 + second));

    auto r1 = res.recv_with_timeout({first, second}, 1);
    auto r2 = res.recv_with_timeout({first, second}, 1);

    char d[96];
    snprintf(d, sizeof(d), "got %d then %d, wanted %d then %d",
             r1 ? r1->id : -1, r2 ? r2->id : -1, first, second);
    check(r1 != nullptr && r2 != nullptr && r1->id == first && r2->id == second, label, d);
}

static void t_fifo_across_waiters() {
    t_fifo_across_waiters_one(700, 701, "arrival order kept across waiters, low id first");
    t_fifo_across_waiters_one(711, 710, "arrival order kept across waiters, high id first");
}

// A reader already parked on a waiter has to be woken when that waiter is discarded, or it will
// wait out its deadline on a condition nothing will ever fire again while its id is re-registered
// and served on a brand new waiter.
static void t_waiter_replaced_under_a_parked_reader() {
    server_response res;
    res.add_waiting_task_id(800);

    std::atomic<int> got{-1};
    std::thread reader([&] {
        auto r = res.recv_with_timeout({800}, 3);
        got.store(payload_of(r));
    });
    std::this_thread::sleep_for(ms(300));   // let the reader select the current waiter and park

    const auto t0 = std::chrono::steady_clock::now();
    res.remove_waiting_task_id(800);        // discards the waiter the reader is parked on
    res.add_waiting_task_id(800);           // a brand new waiter
    res.send(mk(800, 8800));
    reader.join();
    const auto waited = std::chrono::duration_cast<ms>(std::chrono::steady_clock::now() - t0).count();

    char d[80]; snprintf(d, sizeof(d), "%lldms, payload=%d", (long long) waited, got.load());
    check(got.load() == 8800 && waited < 2500,
          "a parked reader is woken when its waiter is replaced", d);
}

static long rss_kb() {
    // Linux only; returns -1 elsewhere, and only the optional "leak" mode uses it
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
    t_single_timed_recv_before_registration();
    t_terminate_while_parked_on_absent_ids();
    t_subset_recv_is_filtered();
    t_subset_receivers_are_all_woken();
    t_ids_spanning_two_waiters();
    t_fifo_across_waiters();
    t_waiter_replaced_under_a_parked_reader();
    t_parked_reader_does_not_abort();

    printf("\nRESULT queue failures=%d\n", g_fail);
    fflush(stdout);
    _Exit(g_fail == 0 ? 0 : 1);   // a thread is parked in recv() by design
}
