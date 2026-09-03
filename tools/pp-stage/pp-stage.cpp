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

#include "llama.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
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

    fprintf(stderr, "pp-stage: %s stage, n_embd=%d n_pos_per_embd=%d, listening on port %d\n",
            is_first ? (is_last ? "single" : "FIRST") : (is_last ? "LAST" : "middle"),
            n_embd, n_pos_per_embd, port);

    int srv = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr = {};
    addr.sin_family = AF_INET; addr.sin_addr.s_addr = INADDR_ANY; addr.sin_port = htons(port);
    if (bind(srv, (sockaddr *) &addr, sizeof(addr)) != 0) { perror("bind"); return 1; }
    listen(srv, 8);

    std::vector<int32_t> pos_buf;
    std::vector<float>   fbuf;
    std::vector<int32_t> ibuf;
    std::vector<float>   out;

    // one client at a time: a stage is one GPU, and the point is to keep it busy, not to
    // interleave two conversations on the same device
    while (true) {
        int fd = accept(srv, nullptr, nullptr);
        if (fd < 0) continue;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        fprintf(stderr, "pp-stage: client connected\n");

        while (true) {
            req_hdr h;
            if (!read_all(fd, &h, sizeof(h))) break;
            if (h.magic != PPS_MAGIC) { fprintf(stderr, "bad magic\n"); break; }

            if (h.op == 0) {
                llama_memory_seq_rm(mem, h.seq_id, -1, -1);
                const uint32_t ok[3] = { PPS_MAGIC, 0, 0 };
                if (!write_all(fd, ok, sizeof(ok))) break;
                continue;
            }

            const int nt = h.n_tokens;
            pos_buf.resize(nt);
            if (!read_all(fd, pos_buf.data(), nt * sizeof(int32_t))) break;

            llama_batch batch;
            if (is_first) {
                ibuf.resize(nt);
                if (!read_all(fd, ibuf.data(), nt * sizeof(int32_t))) break;
                batch = llama_batch_init(nt, 0, 1);
                for (int i = 0; i < nt; ++i) {
                    batch.token[i] = ibuf[i];
                    batch.pos[i] = pos_buf[i];
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = h.seq_id;
                    batch.logits[i] = is_last ? (i == nt - 1) : 1;
                }
            } else {
                fbuf.resize((size_t) nt * n_embd);
                if (!read_all(fd, fbuf.data(), fbuf.size() * sizeof(float))) break;
                // M-RoPE: the embd path expects n_pos_per_embd sections of n_tokens positions
                // (src/llama-batch.cpp:785). Over-allocate so pos[] has room for all of them.
                batch = llama_batch_init(nt * n_pos_per_embd, n_embd, 1);
                memcpy(batch.embd, fbuf.data(), fbuf.size() * sizeof(float));
                for (int j = 0; j < n_pos_per_embd; ++j) {
                    for (int i = 0; i < nt; ++i) batch.pos[j * nt + i] = pos_buf[i];
                }
                for (int i = 0; i < nt; ++i) {
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = h.seq_id;
                    batch.logits[i] = is_last ? (i == nt - 1) : 1;
                }
            }
            batch.n_tokens = nt;

            const int rc = llama_decode(ctx, batch);
            llama_batch_free(batch);
            if (rc != 0) { fprintf(stderr, "decode failed rc=%d\n", rc); break; }

            uint32_t resp[3] = { PPS_MAGIC, 0, 0 };
            if (is_last) {
                const float * lg = llama_get_logits_ith(ctx, nt - 1);
                int best = 0;
                for (int i = 1; i < n_vocab; ++i) if (lg[i] > lg[best]) best = i;
                resp[2] = 1;
                if (!write_all(fd, resp, sizeof(resp))) break;
                const int32_t tok = best;
                if (!write_all(fd, &tok, sizeof(tok))) break;
            } else {
                const float * e = llama_get_embeddings(ctx);
                if (!e) { fprintf(stderr, "no embeddings\n"); break; }
                const size_t n = (size_t) nt * n_embd;
                resp[2] = (uint32_t) n;
                if (!write_all(fd, resp, sizeof(resp))) break;
                if (!write_all(fd, e, n * sizeof(float))) break;
            }
        }
        close(fd);
        fprintf(stderr, "pp-stage: client disconnected\n");
    }
}
