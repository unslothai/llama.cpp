// pp-stage: one stage of a layer-split pipeline, served over a minimal binary TCP protocol.
//
// Stage A (LLAMA_PP_IL_END=k) takes token ids and returns the raw residual stream entering
// layer k. Stage B (LLAMA_PP_IL_BEG=k) takes that residual and returns a greedily sampled
// token id. Each stage owns only its own layers' weights and KV cache.
//
// The protocol is deliberately not OpenAI-shaped: the payload is a float tensor, and wrapping
// n_embd floats per token in JSON would dominate the measurement it exists to make.
//
//   request : magic 'PPS1' | op | seq_id | n_tokens | n_embd | pos[n_tokens] | payload
//   response: magic 'PPS1' | status | n_out_floats | payload
//
//   op = 0 RESET   forget seq_id's KV
//   op = 1 FORWARD payload = n_tokens int32 token ids   (stage A)
//                          | n_tokens*n_embd float32    (stage B)
//        response payload = n_tokens*n_embd float32 hidden states (stage A)
//                          | 1 int32 sampled token      (stage B)
//
// CONTINUOUS BATCHING
// -------------------
// A stage accepts many concurrent connections (one per sequence) and runs a single engine
// thread that drains all currently-queued requests into ONE llama_decode. This is the whole
// point: a decode step is weight-bandwidth-bound, so folding N sequences' single tokens into
// one batch moves the stage's weights ONCE instead of N times. Serving one request per
// decode -- the original design -- made a two-stage split 0.35x a properly batched single
// node at concurrency 8, because the split paid N weight passes where one node paid 1.
//
// Large requests are deliberately NOT batched together. Coalescing prefills would force
// stage A to finish every sequence before stage B could start any, destroying the
// cross-sequence overlap that gives this design its 1.70x prefill win. So the gather loop
// accumulates requests only while the running token total stays under PP_BATCH_TOK (default
// one ubatch); a prefill larger than that is always dispatched alone. Decodes are 1 token
// each and so always coalesce.

#include "llama.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct req_hdr {
    uint32_t magic;
    int32_t  op;
    int32_t  seq_id;
    int32_t  n_tokens;
    int32_t  n_embd;
};

static const uint32_t PPS_MAGIC = 0x31535050; // 'PPS1'

static bool read_all(int fd, void * buf, size_t n) {
    char * p = (char *) buf;
    while (n) {
        const ssize_t r = recv(fd, p, n, 0);
        if (r <= 0) return false;
        p += r; n -= (size_t) r;
    }
    return true;
}

static bool write_all(int fd, const void * buf, size_t n) {
    const char * p = (const char *) buf;
    while (n) {
        const ssize_t w = send(fd, p, n, 0);
        if (w <= 0) return false;
        p += w; n -= (size_t) w;
    }
    return true;
}

// One unit of work handed from a connection thread to the engine thread. The connection
// thread owns the socket and does all the I/O; the engine only ever touches memory, so a
// slow or stalled client can never hold the GPU.
struct pending {
    int32_t op      = 0;
    int32_t seq_id  = 0;
    int32_t n_tokens = 0;

    std::vector<int32_t> pos;
    std::vector<int32_t> toks;   // first stage: input token ids
    std::vector<float>   embd;   // non-first stage: input hidden states

    std::vector<float>   out_f;  // non-final stage: emitted hidden states
    int32_t              out_tok = 0;   // final stage: sampled token

    bool ok   = false;
    bool done = false;
    std::mutex              m;
    std::condition_variable cv;
};

using pending_ptr = std::shared_ptr<pending>;

// Evidence that batching is real. A "fix" that silently stopped coalescing would post a
// perfectly good-looking speedup, so the stage reports how many requests it actually folded
// into each llama_decode. mean_group ~= concurrency means the batching is working; a mean
// pinned at 1.0 means it is not, whatever the throughput says.
static std::atomic<uint64_t> g_n_decode{0};   // llama_decode calls
static std::atomic<uint64_t> g_n_req{0};      // FORWARD requests served
static std::atomic<uint64_t> g_n_tok{0};      // tokens pushed through
static std::atomic<uint64_t> g_max_group{0};

// Connections that have issued at least one FORWARD and not yet disconnected. This is the
// number of requests a full batch should contain, and it is what the engine lingers for.
static std::atomic<int> g_active{0};

static std::deque<pending_ptr> g_queue;
static std::mutex              g_qmtx;
static std::condition_variable g_qcv;

static void finish(const pending_ptr & p, bool ok) {
    std::lock_guard<std::mutex> lk(p->m);
    p->ok = ok;
    p->done = true;
    p->cv.notify_one();
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <port> [n_seq_max] [n_ctx]\n", argv[0]);
        fprintf(stderr, "  set LLAMA_PP_IL_BEG / LLAMA_PP_IL_END to select this stage's layers\n");
        return 1;
    }
    const std::string model_path = argv[1];
    const int port      = atoi(argv[2]);
    const int n_seq_max = argc > 3 ? atoi(argv[3]) : 8;
    const int n_ctx     = argc > 4 ? atoi(argv[4]) : 4096;

    const char * env_beg = getenv("LLAMA_PP_IL_BEG");
    const char * env_end = getenv("LLAMA_PP_IL_END");
    const bool is_first  = !env_beg || atoi(env_beg) == 0;
    const bool is_last   = !env_end || atoi(env_end) == 0;

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 999;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    const enum llama_rope_type rt = llama_model_rope_type(model);
    const int n_pos_per_embd = (rt == LLAMA_ROPE_TYPE_MROPE || rt == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx     = n_ctx;
    cp.n_batch   = n_ctx;
    cp.n_ubatch  = getenv("UBATCH") ? atoi(getenv("UBATCH")) : 512;
    cp.n_seq_max = n_seq_max;
    if (!is_last) {
        // a non-final stage must emit per-token hidden states rather than logits
        cp.embeddings   = true;
        cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    }

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_memory_t mem = llama_get_memory(ctx);

    // Token budget for coalescing. A request bigger than this is dispatched on its own, so
    // prefill keeps its cross-sequence pipelining while decode steps always batch.
    const int batch_tok = getenv("PP_BATCH_TOK") ? atoi(getenv("PP_BATCH_TOK")) : (int) cp.n_ubatch;

    // TEST-ONLY. Batched decode on CUDA is not bitwise-invariant to which sequences happen
    // to share a llama_decode -- verified on a SINGLE unsplit node, so it is a property of
    // llama.cpp, not of the layer split. That makes "batched vs alone" useless as an
    // equivalence check. PP_BATCH_WAIT_N pins the composition: the engine waits for exactly
    // N queued requests before decoding, so a split run and an unsplit run batch IDENTICALLY
    // and their outputs must then match token for token. Deadlocks if fewer than N sequences
    // are ever in flight, which is why it is a test knob and not a serving policy.
    const int wait_n = getenv("PP_BATCH_WAIT_N") ? atoi(getenv("PP_BATCH_WAIT_N")) : 1;

    // How long the engine may wait for the stragglers of a batch. Bounded, so it degrades to
    // "decode what we have" rather than hanging. 0 disables lingering entirely.
    const int linger_us = getenv("PP_BATCH_LINGER_US") ? atoi(getenv("PP_BATCH_LINGER_US")) : 3000;

    fprintf(stderr, "pp-stage: %s stage, n_embd=%d n_pos_per_embd=%d, n_seq_max=%d "
                    "batch_tok=%d, listening on port %d\n",
            is_first ? (is_last ? "single" : "FIRST") : (is_last ? "LAST" : "middle"),
            n_embd, n_pos_per_embd, n_seq_max, batch_tok, port);

    // ---------------- engine thread: the only thing that touches the context -------------
    std::thread engine([&] {
        std::vector<pending_ptr> group;
        std::vector<llama_pos>     pos;
        std::vector<int32_t>       n_seq_id;
        std::vector<llama_seq_id>  seq_ids;
        std::vector<llama_seq_id*> seq_ptr;
        std::vector<int8_t>        logits;
        std::vector<llama_token>   toks;
        std::vector<float>         embd;

        while (true) {
            group.clear();
            int total = 0;
            {
                std::unique_lock<std::mutex> lk(g_qmtx);
                g_qcv.wait(lk, [&] { return (int) g_queue.size() >= wait_n && !g_queue.empty(); });

                // LINGER. Having something to decode is not a reason to decode it yet. A
                // client is a round trip behind the engine: it cannot send token n+1 until it
                // has been given token n, so after a decode completes its requests trickle
                // back in over tens of microseconds. Firing on the first arrival shreds the
                // batch -- measured mean_group ~2.0 at 8 concurrent sequences, which threw
                // away most of the batching win this file exists to get.
                //
                // So wait until every client that could contribute has contributed, i.e. the
                // queue holds one request per active connection. Two properties make this
                // safe rather than a latency tax:
                //   * it is bounded by PP_BATCH_LINGER_US, so a client that goes quiet (or
                //     disconnects mid-flight) delays one decode by that much and no more --
                //     unlike PP_BATCH_WAIT_N, this cannot deadlock;
                //   * the predicate is already satisfied at concurrency 1, so a single stream
                //     never waits at all and its TPOT is untouched. That was a hard
                //     requirement: single-stream decode must not be traded for throughput.
                // A decode step costs ~80 ms per stage, so even the full linger is under 5%
                // of one step, and it buys a 4x larger batch for the same weight pass.
                if (linger_us > 0 && wait_n <= 1) {
                    const auto deadline = std::chrono::steady_clock::now() +
                                          std::chrono::microseconds(linger_us);
                    g_qcv.wait_until(lk, deadline, [&] {
                        return (int) g_queue.size() >= g_active.load();
                    });
                }

                while (!g_queue.empty()) {
                    pending_ptr p = g_queue.front();

                    if (p->op == 0) {
                        // RESET. Ordering per sequence is guaranteed by the client waiting
                        // for each reply, and resets of other sequences are independent, so
                        // it is safe to service this immediately.
                        g_queue.pop_front();
                        lk.unlock();
                        llama_memory_seq_rm(mem, p->seq_id, -1, -1);
                        finish(p, true);
                        lk.lock();
                        continue;
                    }

                    // never let two requests for the same sequence share one decode
                    bool dup = false;
                    for (const auto & q : group) if (q->seq_id == p->seq_id) { dup = true; break; }
                    if (dup) break;

                    // the first request always goes in, however large; after that respect
                    // the token budget so a big prefill is never glued to anything else
                    if (!group.empty() && total + p->n_tokens > batch_tok) break;

                    g_queue.pop_front();
                    group.push_back(p);
                    total += p->n_tokens;
                }
            }
            if (group.empty()) continue;

            // Lay the batch out in seq_id order, not arrival order. Without this a stage's
            // output depends on which connection thread happened to win the race into the
            // queue: the row order inside a llama_batch changes the CUDA reduction order,
            // and with it the low bits of every logit. Measured on a SINGLE UNSPLIT node,
            // three identical concurrent runs produced three different greedy token streams
            // -- so this is a property of batching in llama.cpp, not of the layer split.
            //
            // That matters here beyond tidiness. Two stages on two machines each decide
            // their own batch layout; if both sort by seq_id they agree by construction,
            // whereas two independent races do not. This is the same "independent per-node
            // decisions about data that must match" hazard as the KV bookkeeping, and it is
            // closed the same way: make the decision a function of the request set alone.
            // It also makes a served response reproducible, which is worth having on its own.
            std::stable_sort(group.begin(), group.end(),
                             [](const pending_ptr & a, const pending_ptr & b) {
                                 return a->seq_id < b->seq_id;
                             });

            const int T = total;

            pos.assign((size_t) T * n_pos_per_embd, 0);
            n_seq_id.assign(T, 1);
            seq_ids.assign(T, 0);
            seq_ptr.assign(T, nullptr);
            logits.assign(T, 0);

            llama_batch batch = {};
            batch.n_tokens = T;

            int off = 0;
            if (is_first) {
                toks.assign(T, 0);
                for (const auto & p : group) {
                    for (int i = 0; i < p->n_tokens; ++i) {
                        const int gi = off + i;
                        toks[gi] = p->toks[i];
                        seq_ids[gi] = p->seq_id;
                        seq_ptr[gi] = &seq_ids[gi];
                        logits[gi] = is_last ? (i == p->n_tokens - 1) : 1;
                    }
                    off += p->n_tokens;
                }
                batch.token = toks.data();
            } else {
                embd.assign((size_t) T * n_embd, 0.0f);
                for (const auto & p : group) {
                    memcpy(embd.data() + (size_t) off * n_embd, p->embd.data(),
                           (size_t) p->n_tokens * n_embd * sizeof(float));
                    for (int i = 0; i < p->n_tokens; ++i) {
                        const int gi = off + i;
                        seq_ids[gi] = p->seq_id;
                        seq_ptr[gi] = &seq_ids[gi];
                        logits[gi] = is_last ? (i == p->n_tokens - 1) : 1;
                    }
                    off += p->n_tokens;
                }
                batch.embd = embd.data();
            }

            // M-RoPE: the embd path reads positions as n_pos_per_embd contiguous sections of
            // n_tokens, where n_tokens is the size of the WHOLE batch (src/llama-batch.cpp).
            // Writing them per request would silently misplace every sequence but the first.
            off = 0;
            for (const auto & p : group) {
                for (int i = 0; i < p->n_tokens; ++i) {
                    for (int j = 0; j < n_pos_per_embd; ++j) {
                        pos[(size_t) j * T + off + i] = p->pos[i];
                    }
                }
                off += p->n_tokens;
            }

            batch.pos      = pos.data();
            batch.n_seq_id = n_seq_id.data();
            batch.seq_id   = seq_ptr.data();
            batch.logits   = logits.data();

            g_n_decode++;
            g_n_req += group.size();
            g_n_tok += (uint64_t) T;
            { uint64_t m = g_max_group.load();
              while (group.size() > m && !g_max_group.compare_exchange_weak(m, group.size())) {} }
            if (getenv("PP_STATS") && (g_n_decode % 100 == 0)) {
                fprintf(stderr, "pp-stage: decodes=%llu reqs=%llu tokens=%llu "
                                "mean_group=%.2f max_group=%llu\n",
                        (unsigned long long) g_n_decode.load(),
                        (unsigned long long) g_n_req.load(),
                        (unsigned long long) g_n_tok.load(),
                        (double) g_n_req.load() / (double) g_n_decode.load(),
                        (unsigned long long) g_max_group.load());
            }

            const int rc = llama_decode(ctx, batch);
            if (rc != 0) {
                fprintf(stderr, "decode failed rc=%d (batch %d tok, %zu seq)\n",
                        rc, T, group.size());
                for (const auto & p : group) finish(p, false);
                continue;
            }

            off = 0;
            for (const auto & p : group) {
                if (is_last) {
                    const int gi = off + p->n_tokens - 1;
                    const float * lg = llama_get_logits_ith(ctx, gi);
                    if (!lg) { finish(p, false); off += p->n_tokens; continue; }
                    int best = 0;
                    for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
                    p->out_tok = best;
                    finish(p, true);
                } else {
                    p->out_f.resize((size_t) p->n_tokens * n_embd);
                    bool ok = true;
                    for (int i = 0; i < p->n_tokens; ++i) {
                        const float * e = llama_get_embeddings_ith(ctx, off + i);
                        if (!e) { ok = false; break; }
                        memcpy(p->out_f.data() + (size_t) i * n_embd, e, n_embd * sizeof(float));
                    }
                    finish(p, ok);
                }
                off += p->n_tokens;
            }
        }
    });

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    listen(srv, 64);

    std::atomic<int> n_clients{0};

    // One thread per connection. Concurrent connections are what let the engine see more
    // than one request at a time, which is the prerequisite for batching at all.
    while (true) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        fprintf(stderr, "pp-stage: client connected (%d)\n", ++n_clients);

        std::thread([fd, is_first, n_embd, &n_clients] {
            std::vector<int32_t> pos_buf;
            bool counted = false;
            while (true) {
                req_hdr h;
                if (!read_all(fd, &h, sizeof(h))) break;
                if (h.magic != PPS_MAGIC) { fprintf(stderr, "bad magic\n"); break; }

                auto p = std::make_shared<pending>();
                p->op = h.op;
                p->seq_id = h.seq_id;
                p->n_tokens = h.n_tokens;

                if (h.op == 2) {
                    // STATS: batching evidence, read back by the driver
                    uint32_t resp[3] = { PPS_MAGIC, 0, 4 };
                    const double mean = g_n_decode ? (double) g_n_req / (double) g_n_decode : 0.0;
                    const float st[4] = { (float) g_n_decode.load(), (float) g_n_req.load(),
                                          (float) g_n_tok.load(),   (float) mean };
                    if (!write_all(fd, resp, sizeof(resp))) break;
                    if (!write_all(fd, st, sizeof(st))) break;
                    continue;
                }
                if (h.op != 0) {
                    const int nt = h.n_tokens;
                    p->pos.resize(nt);
                    if (!read_all(fd, p->pos.data(), nt * sizeof(int32_t))) break;
                    if (is_first) {
                        p->toks.resize(nt);
                        if (!read_all(fd, p->toks.data(), nt * sizeof(int32_t))) break;
                    } else {
                        p->embd.resize((size_t) nt * n_embd);
                        if (!read_all(fd, p->embd.data(), p->embd.size() * sizeof(float))) break;
                    }
                }

                // Count this connection as active from its first FORWARD. Done here, not at
                // accept(), because a socket that has connected but not yet asked for
                // anything must not make the engine linger for it.
                if (h.op != 0 && !counted) { counted = true; g_active++; }

                {
                    std::lock_guard<std::mutex> lk(g_qmtx);
                    g_queue.push_back(p);
                }
                // notify_all, not notify_one: the engine may be parked in the linger
                // wait_until re-checking its predicate, and a lost wakeup there would stall
                // this request for the whole linger window.
                g_qcv.notify_all();

                {
                    std::unique_lock<std::mutex> lk(p->m);
                    p->cv.wait(lk, [&] { return p->done; });
                }
                if (!p->ok) { fprintf(stderr, "request failed\n"); break; }

                uint32_t resp[3] = { PPS_MAGIC, 0, 0 };
                if (h.op == 0) {
                    if (!write_all(fd, resp, sizeof(resp))) break;
                } else if (!p->out_f.empty()) {
                    resp[2] = (uint32_t) p->out_f.size();
                    if (!write_all(fd, resp, sizeof(resp))) break;
                    if (!write_all(fd, p->out_f.data(), p->out_f.size() * sizeof(float))) break;
                } else {
                    resp[2] = 1;
                    if (!write_all(fd, resp, sizeof(resp))) break;
                    if (!write_all(fd, &p->out_tok, sizeof(p->out_tok))) break;
                }
            }
            close(fd);
            // Drop out of the active set and wake the engine: if it is lingering for a
            // request this connection will now never send, it should stop waiting.
            if (counted) { g_active--; g_qcv.notify_all(); }
            fprintf(stderr, "pp-stage: client disconnected (%d)\n", --n_clients);
        }).detach();
    }

    engine.join();
}
