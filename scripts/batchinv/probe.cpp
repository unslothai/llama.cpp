// Locate the first graph op whose sequence-0 output changes when the decode batch
// holds four sequences instead of one. Prompt KV for seq 0 is built identically in
// both phases, so the only difference is the width of the final decode ubatch.
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct node_rec {
    std::string name;
    std::string op;
    std::string tname;
    int64_t ne[4];
    int64_t gdn_tokens = 0, gdn_seqs = 0;
    size_t esize = 0;          // bytes per element, 0 = not byte comparable
    std::vector<uint8_t> data; // empty when skipped
    bool contiguous = false;
    uint64_t hash = 0;
};

static bool g_record = false;
static std::vector<node_rec> * g_sink = nullptr;

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ULL; }
    return h;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * /*ud*/) {
    if (!g_record) return false;
    if (ask)       return true;

    node_rec r;
    r.name = ggml_get_name(t);
    r.tname = ggml_type_name(t->type);
    r.op   = t->op == GGML_OP_NONE ? "LEAF" : ggml_op_name(t->op);
    if (t->op == GGML_OP_UNARY)  r.op = std::string("UNARY_")  + ggml_unary_op_name(ggml_get_unary_op(t));
    if (t->op == GGML_OP_GLU)    r.op = std::string("GLU_")    + ggml_glu_op_name(ggml_get_glu_op(t));
    for (int i = 0; i < 4; ++i) r.ne[i] = t->ne[i];
    r.contiguous = ggml_is_contiguous(t);
    if (r.name == "linear_attn_out-0") {
        fprintf(stderr, "linear_attn_out-0: weight=%s input=[%lld,%lld,%lld,%lld]\n",
            ggml_type_name(t->src[0]->type), (long long)t->src[1]->ne[0],
            (long long)t->src[1]->ne[1], (long long)t->src[1]->ne[2], (long long)t->src[1]->ne[3]);
    }
    if (t->op == GGML_OP_GATED_DELTA_NET) {
        r.gdn_tokens = t->src[2]->ne[2];
        r.gdn_seqs = t->src[2]->ne[3];
    }

    const size_t nbytes = ggml_nbytes(t);
    if (r.contiguous && ggml_blck_size(t->type) == 1 && nbytes <= (256u << 20)) {
        r.esize = ggml_type_size(t->type);
        r.data.resize(nbytes);
        ggml_backend_tensor_get(t, r.data.data(), 0, nbytes);
        if (t->op == GGML_OP_FLASH_ATTN_EXT) {
            for (size_t i = 0; i < nbytes/sizeof(float); ++i) {
                float v; memcpy(&v, r.data.data() + i*sizeof(float), sizeof(float));
                if (!std::isfinite(v)) { fprintf(stderr, "nonfinite attention: %s\n", t->name); exit(5); }
            }
        }
        r.hash = fnv1a(r.data.data(), nbytes);
        if (nbytes > (64u << 20)) { r.data.clear(); } // keep the hash only for the big ones
    }
    g_sink->push_back(std::move(r));
    return true;
}

static std::string slurp(const char * path) {
    std::ifstream f(path);
    std::stringstream ss; ss << f.rdbuf();
    std::string s = ss.str();
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
    return s;
}

static std::vector<llama_token> tokenize(const llama_vocab * v, const std::string & s) {
    std::vector<llama_token> out(s.size() + 16);
    int n = llama_tokenize(v, s.c_str(), (int) s.size(), out.data(), (int) out.size(), true, false);
    if (n < 0) { out.resize(-n); n = llama_tokenize(v, s.c_str(), (int) s.size(), out.data(), (int) out.size(), true, false); }
    out.resize(n);
    return out;
}

struct batch_holder {
    std::vector<llama_token>   tok;
    std::vector<llama_pos>     pos;
    std::vector<int32_t>       nsid;
    std::vector<llama_seq_id>  sid;
    std::vector<llama_seq_id*> sidp;
    std::vector<int8_t>        out;
    llama_batch get() {
        sidp.resize(tok.size());
        for (size_t i = 0; i < tok.size(); ++i) sidp[i] = &sid[i];
        llama_batch b{};
        b.n_tokens = (int32_t) tok.size();
        b.token = tok.data(); b.pos = pos.data(); b.n_seq_id = nsid.data();
        b.seq_id = sidp.data(); b.logits = out.data();
        return b;
    }
};

static llama_token greedy(llama_context * ctx, int32_t i, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, i);
    for (int k = 0; k < n_vocab; ++k) {
        if (!std::isfinite(l[k])) { fprintf(stderr, "nonfinite logits at %d\n", k); exit(4); }
    }
    int best = 0;
    for (int k = 1; k < n_vocab; ++k) if (l[k] > l[best]) best = k;
    return best;
}

// Feed a prompt as one decode call for one sequence, return the greedy next token.
static llama_token feed(llama_context * ctx, const std::vector<llama_token> & p, llama_seq_id seq, int n_vocab) {
    batch_holder h;
    for (size_t i = 0; i < p.size(); ++i) {
        h.tok.push_back(p[i]); h.pos.push_back((llama_pos) i);
        h.nsid.push_back(1);   h.sid.push_back(seq);
        h.out.push_back(i + 1 == p.size());
    }
    llama_batch b = h.get();
    if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed\n"); exit(1); }
    return greedy(ctx, (int32_t) p.size() - 1, n_vocab);
}

int main(int argc, char ** argv) {
    const bool prefill = getenv("PROBE_PREFILL") != nullptr;
    const char * model_path = argv[1];
    const int  n_seqs       = argc > 2 ? atoi(argv[2]) : 4;    // width of the probed decode batch
    const char * out_path   = argc > 3 ? argv[3] : nullptr;
    std::vector<std::string> prompts;
    for (int i = 4; i < argc; ++i) prompts.push_back(slurp(argv[i]));

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    std::vector<std::vector<llama_token>> ptok;
    for (auto & s : prompts) ptok.push_back(tokenize(vocab, s));
    for (size_t i = 0; i < ptok.size(); ++i) fprintf(stderr, "prompt %zu: %zu tokens\n", i, ptok[i].size());

    auto make_ctx = [&]() {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = 2048; cp.n_ubatch = 512;
        if (prefill) { cp.n_ubatch = 2048; }
        cp.n_seq_max = 4; cp.kv_unified = true;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.cb_eval = eval_cb; cp.cb_eval_user_data = nullptr;
        cp.no_perf = true;
        return llama_init_from_model(model, cp);
    };

    std::vector<node_rec> rec_a, rec_b;
    llama_token first_tok[4] = {0, 0, 0, 0};

    // Phase A: decode ubatch width 1. PROBE_A_FILL controls how many sequences are
    // already in the shared KV cache, which is what sets K->ne[1] for attention.
    const int a_fill = getenv("PROBE_A_FILL") ? atoi(getenv("PROBE_A_FILL")) : 1;
    // PROBE_A_PERM reorders which prompt goes into which sequence in phase A. With the same
    // multiset of prompts the cache keeps its length but the masked cells hold different data.
    // Phase B decodes prompt 0's first token on sequence 0, so the permutation may only move
    // the neighbours: sequence 0 keeps prompt 0, or the two phases would compare different
    // sequences.
    int a_perm[4] = {0, 1, 2, 3};
    if (const char * perm = getenv("PROBE_A_PERM")) {
        for (int k = 0; k < 4 && perm[2*k]; ++k) a_perm[k] = perm[2*k] - '0';
        if (a_perm[0] != 0) { fprintf(stderr, "PROBE_A_PERM must keep prompt 0 on sequence 0\n"); return 1; }
    }
    {
        llama_context * ctx = make_ctx();
        g_sink = &rec_a; g_record = prefill;
        for (int s = 0; s < a_fill; ++s) {
            const llama_token t = feed(ctx, ptok[a_perm[s]], s, n_vocab);
            if (a_perm[s] == 0) first_tok[0] = t;
        }
        batch_holder h;
        h.tok = {first_tok[0]}; h.pos = {(llama_pos) ptok[0].size()};
        h.nsid = {1}; h.sid = {0}; h.out = {1};
        llama_batch b = h.get();
        g_sink = &rec_a; g_record = !prefill;
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "A decode failed\n"); return 1; }
        g_record = false;
        llama_free(ctx);
    }

    // Phase B: same seq-0 prompt KV, then a decode ubatch holding n_seqs tokens.
    {
        llama_context * ctx = make_ctx();
        if (prefill) {
            batch_holder h;
            for (int seq = 0; seq < n_seqs; ++seq) {
                for (size_t i = 0; i < ptok[0].size(); ++i) {
                    h.tok.push_back(ptok[seq][i%ptok[seq].size()]); h.pos.push_back(i);
                    h.nsid.push_back(1); h.sid.push_back(seq); h.out.push_back(i+1 == ptok[0].size());
                }
            }
            auto b = h.get();
            g_sink = &rec_b; g_record = true;
            if (llama_decode(ctx, b) != 0) { return 6; }
            g_record = false;
            for (int seq = 0; seq < n_seqs; ++seq) {
                first_tok[seq] = greedy(ctx, (seq+1)*ptok[0].size()-1, n_vocab);
            }
        } else for (int k = 0; k < n_seqs; ++k) {
            const int s = getenv("PROBE_B_REVERSE") ? n_seqs - 1 - k : k;
            first_tok[s] = feed(ctx, ptok[s], s, n_vocab);
        }
        if (getenv("PROBE_RESTORE")) {
            std::vector<uint8_t> state(llama_state_seq_get_size(ctx, 0));
            if (llama_state_seq_get_data(ctx, state.data(), state.size(), 0) != state.size()) { return 2; }
            llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
            llama_memory_seq_rm(llama_get_memory(ctx), 1, -1, -1);
            if (llama_state_seq_set_data(ctx, state.data(), state.size(), 0) != state.size()) { return 3; }
            first_tok[1] = feed(ctx, ptok[1], 1, n_vocab);
        }
        if (first_tok[0] != 0 && rec_a.size()) {}
        batch_holder h;
        for (int s = 0; s < n_seqs; ++s) {
            h.tok.push_back(first_tok[s]); h.pos.push_back((llama_pos) ptok[s].size());
            h.nsid.push_back(1); h.sid.push_back(s); h.out.push_back(1);
        }
        llama_batch b = h.get();
        g_sink = &rec_b; g_record = !prefill;
        if (!prefill && llama_decode(ctx, b) != 0) { fprintf(stderr, "B decode failed\n"); return 1; }
        g_record = false;
        llama_free(ctx);
    }

    // Optional: keep decoding and report the first step at which seq 0's token differs.
    const int n_steps = getenv("PROBE_STEPS") ? atoi(getenv("PROBE_STEPS")) : 0;
    int first_bad_step = -1;
    if (n_steps > 0) {
        std::vector<llama_token> tok_a, tok_b;
        for (int phase = 0; phase < 2; ++phase) {
            const int fill = phase == 0 ? a_fill : n_seqs;
            const int width = phase == 0 ? 1 : n_seqs;
            std::vector<llama_token> & out = phase == 0 ? tok_a : tok_b;
            llama_context * ctx = make_ctx();
            std::vector<llama_token> next(4, 0);
            std::vector<llama_pos>   pos(4, 0);
            for (int s = 0; s < fill; ++s) {
                const int p = phase == 0 ? a_perm[s] : s;
                next[s] = feed(ctx, ptok[p], s, n_vocab);
                pos[s]  = (llama_pos) ptok[p].size();
            }
            for (int step = 0; step < n_steps; ++step) {
                batch_holder h;
                for (int s = 0; s < width; ++s) {
                    h.tok.push_back(next[s]); h.pos.push_back(pos[s]);
                    h.nsid.push_back(1); h.sid.push_back(s); h.out.push_back(1);
                }
                llama_batch b = h.get();
                if (llama_decode(ctx, b) != 0) { fprintf(stderr, "step decode failed\n"); exit(1); }
                out.push_back(next[0]);
                for (int s = 0; s < width; ++s) { next[s] = greedy(ctx, s, n_vocab); pos[s] += 1; }
            }
            llama_free(ctx);
        }
        for (int i = 0; i < n_steps; ++i) {
            if (tok_a[i] != tok_b[i]) { first_bad_step = i; break; }
        }
        fprintf(stderr, "steps: %d  first differing step: %d\n", n_steps, first_bad_step);
    }

    fprintf(stderr, "nodes: A=%zu B=%zu  first tokens: %d %d %d %d\n",
            rec_a.size(), rec_b.size(), first_tok[0], first_tok[1], first_tok[2], first_tok[3]);

    // Walk both node lists in order and compare seq 0's slice.
    FILE * out = out_path ? fopen(out_path, "w") : stdout;
    fprintf(out, "{\"n_seqs\":%d,\"first_bad_step\":%d,\"nodes_a\":%zu,\"nodes_b\":%zu,\"diffs\":[", n_seqs, first_bad_step, rec_a.size(), rec_b.size());
    size_t n = rec_a.size() < rec_b.size() ? rec_a.size() : rec_b.size();
    int emitted = 0;
    for (size_t i = 0; i < n; ++i) {
        const node_rec & A = rec_a[i];
        const node_rec & B = rec_b[i];
        const char * verdict = nullptr;
        double max_abs = 0.0;
        size_t ndiff = 0, ncmp = 0;

        if (A.name != B.name || A.op != B.op) {
            verdict = "misaligned";
        } else if (A.op == "GATED_DELTA_NET" && A.gdn_tokens == B.gdn_tokens &&
                !A.data.empty() && !B.data.empty()) {
            // Packed GDN outputs put ALL token outputs before ALL sequence states.
            // Sequence 0's state therefore moves when the number of sequences changes.
            const size_t output = A.ne[0]*A.gdn_tokens;
            const size_t state = A.ne[0]*A.ne[1]/A.gdn_seqs - output;
            for (size_t k = 0; k < output + state; ++k) {
                const size_t ia = k < output ? k : A.gdn_seqs*output + k-output;
                const size_t ib = k < output ? k : B.gdn_seqs*output + k-output;
                float va, vb;
                memcpy(&va, A.data.data()+ia*4, 4); memcpy(&vb, B.data.data()+ib*4, 4);
                ++ncmp;
                if (memcmp(&va, &vb, 4)) {
                    ++ndiff;
                    if (std::abs(double(va)-vb) > max_abs) { max_abs = std::abs(double(va)-vb); }
                }
            }
            verdict = ndiff ? "row-differs" : nullptr;
        } else if (A.esize == 0 || B.esize == 0 || A.esize != B.esize) {
            verdict = "skipped";
        } else {
            int tdim = -1; bool same = true;
            for (int d = 0; d < 4; ++d) {
                if (A.ne[d] == B.ne[d]) continue;
                same = false;
                if (B.ne[d] == n_seqs*A.ne[d] && tdim < 0) tdim = d; else { tdim = -2; break; }
            }
            if (tdim == -2) {
                verdict = "shape-incomparable";
            } else if (same) {
                verdict = (A.hash == B.hash) ? nullptr : "whole-tensor-differs";
            } else if (A.data.empty() || B.data.empty()) {
                verdict = "too-large";
            } else {
                // compare element (.., i_tdim = 0, ..) across all other indices
                int64_t st[4] = {1, A.ne[0], A.ne[0]*A.ne[1], A.ne[0]*A.ne[1]*A.ne[2]};
                int64_t stb[4] = {1, B.ne[0], B.ne[0]*B.ne[1], B.ne[0]*B.ne[1]*B.ne[2]};
                for (int64_t i3 = 0; i3 < A.ne[3]; ++i3)
                for (int64_t i2 = 0; i2 < A.ne[2]; ++i2)
                for (int64_t i1 = 0; i1 < A.ne[1]; ++i1)
                for (int64_t i0 = 0; i0 < A.ne[0]; ++i0) {
                    int64_t idx[4] = {i0, i1, i2, i3};

                    size_t oa = 0, ob = 0;
                    for (int d = 0; d < 4; ++d) { oa += idx[d]*st[d]; ob += idx[d]*stb[d]; }
                    ncmp++;
                    const uint8_t * pa = A.data.data() + oa*A.esize;
                    const uint8_t * pb = B.data.data() + ob*B.esize;
                    if (memcmp(pa, pb, A.esize) != 0) {
                        ndiff++;
                        if (A.esize == 4) {
                            float fa, fb; memcpy(&fa, pa, 4); memcpy(&fb, pb, 4);
                            double d2 = fa - fb; if (d2 < 0) d2 = -d2;
                            if (d2 > max_abs) max_abs = d2;
                        }
                    }
                }
                verdict = ndiff ? "row-differs" : nullptr;
            }
        }
        {
            if (emitted++) fprintf(out, ",");
            fprintf(out, "\n{\"i\":%zu,\"name\":\"%s\",\"op\":\"%s\",\"ne_a\":[%lld,%lld,%lld,%lld],"
                         "\"ne_b\":[%lld,%lld,%lld,%lld],\"type\":\"%s\",\"verdict\":\"%s\",\"ndiff\":%zu,\"ncmp\":%zu,\"max_abs\":%.6g}",
                    i, A.name.c_str(), A.op.c_str(),
                    (long long)A.ne[0],(long long)A.ne[1],(long long)A.ne[2],(long long)A.ne[3],
                    (long long)B.ne[0],(long long)B.ne[1],(long long)B.ne[2],(long long)B.ne[3],
                    A.tname.c_str(), verdict ? verdict : "same", ndiff, ncmp, max_abs);
        }
    }
    fprintf(out, "\n]}\n");
    if (out_path) fclose(out);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
