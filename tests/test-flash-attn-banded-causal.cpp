// ggml_flash_attn_ext_banded_set_causal against an explicit causal mask: the same op, inputs and backend, once with an
// F16 mask that hides the cells after each query (and the empty cells) and once with no mask and the causal flag.
// The two must agree on every backend that runs the causal op, with cell distances and with cell positions.
// Also checks that a backend refuses a causal op it cannot run instead of aborting.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

struct causal_config {
    int64_t   d;
    int64_t   n_head;
    int64_t   n_head_kv;
    int64_t   n_q;
    int64_t   n_kv;
    int64_t   extent;
    ggml_type kv_type;
    bool      sinks;
    bool      positions; // cell positions with empty cells and a gap; else the cells are the positions, queries at the tail
};

struct causal_data {
    std::vector<float>   q, k, v, rel, sinks;
    std::vector<int32_t> k_pos, q_pos;
};

static causal_data make_data(const causal_config & c, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-0.25f, 0.25f);
    std::uniform_real_distribution<float> ur(-1.0f, 1.0f);
    std::uniform_real_distribution<float> us(-4.0f, 4.0f);
    causal_data data;
    data.q.resize(c.d*c.n_q*c.n_head);
    data.k.resize(c.d*c.n_kv*c.n_head_kv);
    data.v.resize(c.d*c.n_kv*c.n_head_kv);
    data.rel.resize(c.extent*c.n_head*c.n_q);
    data.sinks.resize(c.n_head*c.n_q);
    for (auto & x : data.q) { x = u(rng); }
    for (auto & x : data.k) { x = u(rng); }
    for (auto & x : data.v) { x = u(rng); }
    for (auto & x : data.rel) { x = ur(rng); }
    for (auto & x : data.sinks) { x = us(rng); }

    data.k_pos.assign(c.n_kv, -1);
    data.q_pos.resize(c.n_q);
    if (c.positions) {
        // as the causal test-backend-ops case: queries at cells [q0, q0 + n_q), a gap of 9 positions from n_kv/2 on,
        // every 13th cell before the queries empty, two cells of later positions after them, then empty cells
        const int64_t q0 = c.n_kv - c.n_q - 4;
        auto pos_of = [&](int64_t j) { return int32_t(j + (j >= c.n_kv/2 ? 9 : 0)); };
        for (int64_t j = 0; j < q0 + c.n_q; ++j) {
            if (j % 13 != 7 || j >= q0) {
                data.k_pos[j] = pos_of(j);
            }
        }
        data.k_pos[q0 + c.n_q + 0] = pos_of(q0 + c.n_q) + 3;
        data.k_pos[q0 + c.n_q + 1] = pos_of(q0 + c.n_q) + 4;
        for (int64_t i = 0; i < c.n_q; ++i) {
            data.q_pos[i] = pos_of(q0 + i);
        }
    } else {
        // the cells are the positions, the queries the last n_q
        for (int64_t j = 0; j < c.n_kv; ++j) {
            data.k_pos[j] = int32_t(j);
        }
        for (int64_t i = 0; i < c.n_q; ++i) {
            data.q_pos[i] = int32_t(c.n_kv - c.n_q + i);
        }
    }
    return data;
}

static void set_typed(ggml_tensor * t, const std::vector<float> & src) {
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_set(t, src.data(), 0, src.size()*sizeof(float));
        return;
    }
    const int64_t n_per_row = t->ne[0];
    const int64_t nrows     = (int64_t) src.size()/n_per_row;
    std::vector<uint8_t> buf(ggml_row_size(t->type, n_per_row)*nrows);
    if (t->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row(src.data(), (ggml_fp16_t *) buf.data(), (int64_t) src.size());
    } else {
        ggml_quantize_chunk(t->type, src.data(), buf.data(), 0, nrows, n_per_row, nullptr);
    }
    ggml_backend_tensor_set(t, buf.data(), 0, buf.size());
}

// runs the op with the explicit mask (causal = false) or with the causal flag; returns false if the backend refuses it
static bool run(ggml_backend_t backend, const causal_config & c, const causal_data & data, bool causal, std::vector<float> & out_data) {
    ggml_init_params params = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));

    ggml_tensor * q  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.d, c.n_q, c.n_head, 1);
    ggml_tensor * k  = ggml_new_tensor_4d(ctx.get(), c.kv_type, c.d, c.n_kv, c.n_head_kv, 1);
    ggml_tensor * v  = ggml_new_tensor_4d(ctx.get(), c.kv_type, c.d, c.n_kv, c.n_head_kv, 1);
    ggml_tensor * r  = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, c.extent, c.n_head, c.n_q, 1);
    ggml_tensor * m  = causal ? nullptr : ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, c.n_kv, c.n_q, 1, 1);
    ggml_tensor * s  = c.sinks ? ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, c.n_head, c.n_q) : nullptr;
    ggml_tensor * kp = c.positions ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.n_kv) : nullptr;
    ggml_tensor * qp = c.positions ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, c.n_q) : nullptr;

    ggml_tensor * out = ggml_flash_attn_ext_banded(ctx.get(), q, k, v, m, r, 1.0f/float(c.d), c.extent);
    if (s) {
        ggml_flash_attn_ext_add_sinks(out, s);
    }
    if (kp) {
        ggml_flash_attn_ext_banded_set_pos(out, kp, qp);
    }
    if (causal) {
        ggml_flash_attn_ext_banded_set_causal(out, true);
    }
    ggml_prec_set_acc(out, GGML_PREC_F32);

    if (!ggml_backend_supports_op(backend, out)) {
        return false;
    }

    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buf);

    ggml_backend_tensor_set(q, data.q.data(), 0, data.q.size()*sizeof(float));
    set_typed(k, data.k);
    set_typed(v, data.v);
    ggml_backend_tensor_set(r, data.rel.data(), 0, data.rel.size()*sizeof(float));
    if (s) {
        ggml_backend_tensor_set(s, data.sinks.data(), 0, data.sinks.size()*sizeof(float));
    }
    if (kp) {
        ggml_backend_tensor_set(kp, data.k_pos.data(), 0, data.k_pos.size()*sizeof(int32_t));
        ggml_backend_tensor_set(qp, data.q_pos.data(), 0, data.q_pos.size()*sizeof(int32_t));
    }
    if (m) {
        // visible iff the cell has a position and it is not after the query's
        std::vector<ggml_fp16_t> mask(c.n_kv*c.n_q);
        for (int64_t iq = 0; iq < c.n_q; ++iq) {
            for (int64_t ik = 0; ik < c.n_kv; ++ik) {
                const bool visible = data.k_pos[ik] >= 0 && data.k_pos[ik] <= data.q_pos[iq];
                mask[iq*c.n_kv + ik] = ggml_fp32_to_fp16(visible ? 0.0f : -INFINITY);
            }
        }
        ggml_backend_tensor_set(m, mask.data(), 0, mask.size()*sizeof(ggml_fp16_t));
    }

    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, out);
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    out_data.resize(ggml_nelements(out));
    ggml_backend_tensor_get(out, out_data.data(), 0, out_data.size()*sizeof(float));
    return true;
}

static double nmse(const std::vector<float> & a, const std::vector<float> & b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
            return INFINITY;
        }
        num += double(a[i] - b[i])*double(a[i] - b[i]);
        den += double(b[i])*double(b[i]);
    }
    return den > 0.0 ? num/den : num;
}

// a causal op whose queries exceed what the backend can run must be refused by supports_op, not abort in compute
static bool check_large_refused(ggml_backend_t backend, const char * name) {
    ggml_init_params params = { 16*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx(ggml_init(params));
    const int64_t d = 128, n_head = 8, n_q = 300, n_seq = 219; // n_q*n_seq = 65700 > 65535
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, n_q, n_head, n_seq);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, d, 512, 2, n_seq);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, d, 512, 2, n_seq);
    ggml_tensor * r = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 64, n_head, n_q, 1);
    ggml_tensor * out = ggml_flash_attn_ext_banded(ctx.get(), q, k, v, nullptr, r, 0.125f, 64);
    ggml_flash_attn_ext_banded_set_causal(out, true);
    ggml_prec_set_acc(out, GGML_PREC_F32);
    const bool supported = ggml_backend_supports_op(backend, out);
    // the CPU kernel has no such limit; a GPU backend that cannot run it must say so
    const bool ok = strcmp(name, "CPU") == 0 || !supported;
    printf("  %-6s causal n_q %lld x %lld streams: supported %d -> %s\n", name, (long long) n_q, (long long) n_seq, supported, ok ? "ok" : "FAIL");
    return ok;
}

int main() {
    ggml_backend_load_all();

    const std::vector<causal_config> configs = {
        // d, heads, kv heads, n_q, n_kv, extent, K/V, sinks, positions
        { 128, 24, 8,   1, 1000, 1280, GGML_TYPE_F16,  true,  false },
        { 128, 24, 8,  17, 1000, 1280, GGML_TYPE_F16,  true,  false },
        { 128, 24, 8, 512, 2048, 1280, GGML_TYPE_F16,  true,  false },
        { 128, 24, 8, 512, 2048, 1280, GGML_TYPE_Q8_0, true,  false },
        { 128, 24, 8,  17, 1000, 1280, GGML_TYPE_Q4_0, true,  false },
        {  64,  8, 2,  77, 1000,    8, GGML_TYPE_F16,  false, false },
        { 128,  8, 2,  65,  333,   32, GGML_TYPE_F32,  false, false },
        { 128, 24, 8,   1, 1000, 1280, GGML_TYPE_Q8_0, true,  true  },
        { 128, 24, 8,  17, 1000, 1280, GGML_TYPE_F16,  true,  true  },
        { 128, 24, 8, 512, 2048, 1280, GGML_TYPE_Q8_0, true,  true  },
        { 128, 24, 8,  17, 1000, 1280, GGML_TYPE_Q4_0, true,  true  },
        { 128, 24, 8,  17, 1000, 1280, GGML_TYPE_F32,  true,  true  },
        {  64,  8, 2,  17,  700,   64, GGML_TYPE_F16,  false, true  },
    };
    const double max_nmse = 1e-5;

    bool all_ok = true;
    int  n_run  = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        const char * name = ggml_backend_dev_name(dev);
        printf("%s (%s)\n", name, ggml_backend_dev_description(dev));
        for (const auto & c : configs) {
            const causal_data data = make_data(c, 1234);
            std::vector<float> masked, causal;
            const bool ok_m = run(backend.get(), c, data, false, masked);
            const bool ok_c = run(backend.get(), c, data, true,  causal);
            char desc[160];
            snprintf(desc, sizeof(desc), "d %lld n_q %lld n_kv %lld %s sinks %d pos %d", (long long) c.d, (long long) c.n_q,
                    (long long) c.n_kv, ggml_type_name(c.kv_type), c.sinks, c.positions);
            if (!ok_c) {
                printf("  %-60s causal not supported, skipped\n", desc);
                continue;
            }
            if (!ok_m) {
                printf("  %-60s masked not supported, skipped\n", desc);
                continue;
            }
            const double err = nmse(causal, masked);
            const bool ok = err <= max_nmse;
            all_ok &= ok;
            ++n_run;
            printf("  %-60s nmse %.3e %s\n", desc, err, ok ? "ok" : "FAIL");
        }
        all_ok &= check_large_refused(backend.get(), name);
    }
    if (n_run == 0) {
        printf("no backend ran the causal op\n");
        return 1;
    }
    printf("%s\n", all_ok ? "OK" : "FAILED");
    return all_ok ? 0 : 1;
}
