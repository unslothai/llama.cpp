#include "conv-chain.cuh"
#include "ggml-impl.h"

#include <cstring>

#define CONV_CHAIN_BLOCK_SIZE 128

// the longest per-sequence ubatch fused: decode, batched decode and speculative verify
#define CONV_CHAIN_MAX_TOKENS 32

struct conv_chain_params {
    const float * cur;     // [C, T*ns] contiguous
    const float * past;    // [n_conv - 1, C, ns], strides in floats
    int64_t       past_nb1;
    int64_t       past_nb2;
    const float * w;       // [n_conv, C], row stride in floats
    int64_t       w_nb1;
    const float * dyn;     // [n_conv, n_head, T*ns] contiguous, or null
    int           ch_per_head;
    int           n_head;
    float       * dst;     // [C, T*ns] contiguous, may be cur
    float         scale;
    float         bias;
    float         conv_bias; // 0: the bias add of ssm_conv_f32
    char        * st;      // state window of slot 0, column 0, channel 0, sequence 0
    int64_t       st_slot; // bytes
    int64_t       st_ch;
    int64_t       st_seq;
    int           n_win;   // windows stored: start columns T - n_win + 1 .. T go to slots n_win - 1 .. 0
    int           C;
    int           T;
};

template <int n_conv, bool has_dyn>
static __global__ void conv_chain_f32(const conv_chain_params p) {
    ggml_cuda_pdl_lc();
    const int c = blockIdx.y*CONV_CHAIN_BLOCK_SIZE + threadIdx.x;
    const int s = blockIdx.x;

    ggml_cuda_pdl_sync();
    if (c >= p.C) {
        return;
    }

    float w[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv; ++j) {
        w[j] = p.w[c*p.w_nb1 + j];
    }

    // xr[j] = column t + j of x; every input is read before the output of its token is written, so dst may be cur, and all
    // of past is read here before the first store, so the stored cell may be past
    float xr[n_conv];
#pragma unroll
    for (int j = 0; j < n_conv - 1; ++j) {
        xr[j] = p.past[j + c*p.past_nb1 + s*p.past_nb2];
    }

    const int64_t tok0 = int64_t(s)*p.T;
    const float * cur  = p.cur + tok0*p.C + c;
    float       * dst  = p.dst + tok0*p.C + c;
    char        * st   = p.st + c*p.st_ch + s*p.st_seq;
    const int     h    = has_dyn ? c / p.ch_per_head : 0;

    auto store = [&](int slot) {
        float * d = (float *) (st + slot*p.st_slot);
#pragma unroll
        for (int j = 0; j < n_conv - 1; ++j) {
            d[j] = xr[j];
        }
    };

    for (int t = 0; t < p.T; ++t) {
        xr[n_conv - 1] = cur[int64_t(t)*p.C];

        // the window starting at column t
        if (t >= p.T - p.n_win + 1) {
            store(p.T - t);
        }

        // SSM_CONV as ssm_conv_f32: contracted products in tap order, then the (zero) bias
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < n_conv; ++j) {
            sumf = __fmaf_rn(xr[j], w[j], sumf);
        }
        sumf = __fadd_rn(sumf, p.conv_bias);

        if (has_dyn) {
            // MUL, ADD per lag: window lag starts n_conv - 1 - lag columns into x
            const float * tap = p.dyn + ((tok0 + t)*p.n_head + h)*n_conv;
#pragma unroll
            for (int lag = 0; lag < n_conv; ++lag) {
                sumf = __fadd_rn(sumf, __fmul_rn(xr[n_conv - 1 - lag], tap[lag]));
            }
        }

        // ADD cur + out, then SCALE as scale_f32 (contracted)
        dst[int64_t(t)*p.C] = __fmaf_rn(p.scale, __fadd_rn(xr[n_conv - 1], sumf), p.bias);

#pragma unroll
        for (int j = 0; j < n_conv - 1; ++j) {
            xr[j] = xr[j + 1];
        }
    }
    // the last window, the next tokens' history
    store(0);
}

template <int n_conv>
static void conv_chain_launch(const conv_chain_params & p, int ns, bool has_dyn, cudaStream_t stream) {
    const dim3 blocks(ns, (p.C + CONV_CHAIN_BLOCK_SIZE - 1)/CONV_CHAIN_BLOCK_SIZE, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(blocks, dim3(CONV_CHAIN_BLOCK_SIZE, 1, 1), 0, stream);
    if (has_dyn) {
        ggml_cuda_kernel_launch(conv_chain_f32<n_conv, true>, lp, p);
    } else {
        ggml_cuda_kernel_launch(conv_chain_f32<n_conv, false>, lp, p);
    }
}

static bool conv_chain_is_view(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_PERMUTE;
}

// the tensor a chain of RESHAPE nodes is computed from
static const ggml_tensor * conv_chain_unreshape(const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        t = t->src[0];
    }
    return t;
}

static bool conv_chain_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) {
        return false;
    }
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static int conv_chain_decline(int line) {
    static const bool dbg = getenv("GGML_CUDA_CONV_CHAIN_DEBUG") != nullptr;
    if (dbg) {
        static int n = 0;
        if (n++ < 64) {
            GGML_LOG_WARN("ggml_cuda_try_fuse_conv_chain: declined at conv-chain.cu:%d\n", line);
        }
    }
    return 0;
}

int ggml_cuda_try_fuse_conv_chain(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    const ggml_tensor * x = cgraph->nodes[i];
    if (x->op != GGML_OP_CONCAT || x->type != GGML_TYPE_F32 || ggml_get_op_params_i32(x, 0) != 0 || !ggml_is_contiguous(x)) {
        return 0;
    }

    // the compute nodes after x in order, views skipped: store, SSM_CONV, [CONT, (MUL, ADD) x n_conv], ADD, SCALE
    const int n_scan_max = 64;
    const ggml_tensor * store = nullptr; // CPY or SET_ROWS
    const ggml_tensor * arange = nullptr, * slot_scale = nullptr, * slot_cast = nullptr;
    const ggml_tensor * ssm = nullptr, * xt = nullptr;
    const ggml_tensor * muls[16] = {}, * adds[16] = {};
    int n_mul = 0, n_add = 0;
    const ggml_tensor * scl = nullptr;
    int j = i + 1;
    for (; j < cgraph->n_nodes && j - i < n_scan_max && !scl; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (conv_chain_is_view(n)) {
            continue;
        }
        if (!store) {
            if (n->op == GGML_OP_ARANGE && !arange) { arange = n; continue; }
            if (n->op == GGML_OP_SCALE && arange && !slot_scale) { slot_scale = n; continue; }
            if (n->op == GGML_OP_CPY && slot_scale && !slot_cast) { slot_cast = n; continue; } // ggml_cast
            if ((n->op == GGML_OP_CPY && !arange) || (n->op == GGML_OP_SET_ROWS && slot_cast)) { store = n; continue; }
            return conv_chain_decline(__LINE__);
        }
        if (!ssm) {
            if (n->op != GGML_OP_SSM_CONV) {
                return conv_chain_decline(__LINE__);
            }
            ssm = n;
            continue;
        }
        if (n->op == GGML_OP_CONT && !xt && n_add == 0) { xt = n; continue; }
        if (n->op == GGML_OP_MUL && xt && n_mul == n_add && n_mul < 16) { muls[n_mul++] = n; continue; }
        if (n->op == GGML_OP_ADD && n_add < 16 && n_add <= n_mul) { adds[n_add++] = n; continue; }
        if (n->op == GGML_OP_SCALE && n_add == n_mul + 1) { scl = n; continue; }
        return conv_chain_decline(__LINE__);
    }
    if (!scl || !store || !ssm) {
        return conv_chain_decline(__LINE__);
    }
    const int i_last = j - 1;

    // SSM_CONV(x, w)
    const ggml_tensor * w = ssm->src[1];
    const int64_t nc = w->ne[0];
    const int64_t C  = x->ne[1];
    const int64_t ns = x->ne[2];
    const int64_t T  = x->ne[0] - (nc - 1);
    if (ssm->src[0] != x || ssm->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || w->nb[0] != sizeof(float) ||
        w->ne[1] != C || T < 1 || T > CONV_CHAIN_MAX_TOKENS || !(nc >= 2 && nc <= 5) || x->ne[3] != 1) {
        return conv_chain_decline(__LINE__);
    }

    // x = concat(past, transpose(cur)): cur is [C, T*ns] contiguous
    const ggml_tensor * past = x->src[0];
    const ggml_tensor * xT   = x->src[1];
    const ggml_tensor * add_res = adds[n_add - 1];
    if (scl->src[0] != add_res || add_res->type != GGML_TYPE_F32 || scl->type != GGML_TYPE_F32) {
        return conv_chain_decline(__LINE__);
    }
    const bool dyn_chain = xt != nullptr;
    if (dyn_chain ? n_mul != nc : n_mul != 0) {
        return conv_chain_decline(__LINE__);
    }
    const ggml_tensor * cur = add_res->src[0];
    const ggml_tensor * out = conv_chain_unreshape(add_res->src[1]);
    if (out != (dyn_chain ? adds[n_add - 2] : ssm)) {
        return conv_chain_decline(__LINE__);
    }
    if (cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) || cur->ne[0] != C || ggml_nelements(cur) != C*T*ns ||
        !ggml_are_same_shape(cur, add_res) || !ggml_is_contiguous(add_res) || !ggml_are_same_shape(scl, add_res) ||
        !ggml_is_contiguous(scl)) {
        return conv_chain_decline(__LINE__);
    }
    if (past->type != GGML_TYPE_F32 || past->nb[0] != sizeof(float) || past->ne[0] != nc - 1 || past->ne[1] != C ||
        past->ne[2] != ns || past->ne[3] != 1) {
        return conv_chain_decline(__LINE__);
    }
    if (xT->type != GGML_TYPE_F32 || xT->data != cur->data || xT->ne[0] != T || xT->ne[1] != C || xT->ne[2] != ns ||
        xT->nb[0] != (size_t) C*sizeof(float) || xT->nb[1] != sizeof(float) || (ns > 1 && xT->nb[2] != (size_t) (C*T)*sizeof(float))) {
        return conv_chain_decline(__LINE__);
    }

    // the dynamic taps: out_l = out_{l-1} + window(x, n_conv - 1 - l)*tap(l)
    const ggml_tensor * dyn = nullptr;
    int64_t n_head = 0, ch_per_head = 0;
    if (dyn_chain) {
        if (xt->src[0]->op != GGML_OP_TRANSPOSE || xt->src[0]->src[0] != x || xt->type != GGML_TYPE_F32 ||
            xt->ne[0] != C || xt->ne[1] != x->ne[0] || xt->ne[2] != ns || !ggml_is_contiguous(xt)) {
            return conv_chain_decline(__LINE__);
        }
        const ggml_tensor * prev = ssm;
        for (int l = 0; l < nc; ++l) {
            const ggml_tensor * m = muls[l];
            const ggml_tensor * a = adds[l];
            if (conv_chain_unreshape(a->src[0]) != prev || conv_chain_unreshape(a->src[1]) != m || a->type != GGML_TYPE_F32 ||
                m->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || ggml_nelements(a) != C*T*ns) {
                return conv_chain_decline(__LINE__);
            }
            const ggml_tensor * win = m->src[0];
            const ggml_tensor * tap = m->src[1];
            if (l == 0) {
                n_head      = win->ne[1];
                ch_per_head = win->ne[0];
                dyn         = tap->view_src;
                if (!dyn || dyn->type != GGML_TYPE_F32 || !ggml_is_contiguous(dyn) || n_head*ch_per_head != C ||
                    ggml_nelements(dyn) != nc*n_head*T*ns) {
                    return conv_chain_decline(__LINE__);
                }
            }
            if (win->op != GGML_OP_VIEW || win->view_src != xt || win->view_offs != (size_t) (nc - 1 - l)*xt->nb[1] ||
                win->ne[0] != ch_per_head || win->ne[1] != n_head || win->ne[2] != T || win->ne[3] != ns ||
                win->nb[1] != (size_t) ch_per_head*sizeof(float) || win->nb[2] != xt->nb[1] || win->nb[3] != xt->nb[2]) {
                return conv_chain_decline(__LINE__);
            }
            if (tap->op != GGML_OP_VIEW || tap->view_src != dyn || tap->view_offs != (size_t) l*sizeof(float) ||
                tap->ne[0] != 1 || tap->ne[1] != n_head || tap->ne[2] != T || tap->ne[3] != ns ||
                tap->nb[1] != (size_t) nc*sizeof(float) || tap->nb[2] != (size_t) (nc*n_head)*sizeof(float) ||
                tap->nb[3] != (size_t) (nc*n_head*T)*sizeof(float)) {
                return conv_chain_decline(__LINE__);
            }
            prev = a;
        }
    }

    // the state store
    conv_chain_params p = {};
    const ggml_tensor * st_dst = nullptr;
    if (store->op == GGML_OP_CPY) {
        // the last n_conv - 1 columns of x into [(n_conv - 1)*C, ns] rows
        const ggml_tensor * last = store->src[0];
        const ggml_tensor * cell = store->src[1];
        if (last->view_src != x || last->view_offs != (size_t) T*sizeof(float) || last->ne[0] != nc - 1 || last->ne[1] != C ||
            last->ne[2] != ns || last->nb[1] != x->nb[1] || last->nb[2] != x->nb[2] ||
            cell->type != GGML_TYPE_F32 || cell->ne[0] != (nc - 1)*C || cell->ne[1] != ns || cell->ne[2] != 1 ||
            cell->nb[0] != sizeof(float)) {
            return conv_chain_decline(__LINE__);
        }
        st_dst    = cell;
        p.st      = (char *) cell->data;
        p.st_slot = 0;
        p.st_ch   = (nc - 1)*sizeof(float);
        p.st_seq  = cell->nb[1];
        p.n_win   = 1;
    } else {
        // SET_ROWS(slots, windows, slot_order): window k starts at column T - (n_win - 1) + k and goes to slot n_win - 1 - k
        const ggml_tensor * wins  = store->src[0];
        const ggml_tensor * order = store->src[1];
        const ggml_tensor * slots = store->src[2];
        const int64_t n_win = wins->ne[1];
        const int64_t K     = arange->ne[0];
        float ar[3];
        memcpy(ar, arange->op_params, sizeof(ar));
        float sb[2];
        memcpy(sb, slot_scale->op_params, sizeof(sb));
        if (ar[0] != 0.0f || ar[2] != 1.0f || arange->type != GGML_TYPE_F32 || K < 1 || ar[1] != (float) K ||
            slot_scale->src[0] != arange || sb[0] != -1.0f || sb[1] != (float) (K - 1) ||
            slot_cast->src[0] != slot_scale || slot_cast->type != GGML_TYPE_I32 || (slot_cast->src[1] && slot_cast->src[1] != slot_cast) ||
            !order || order->view_src != slot_cast || order->ne[0] != n_win || order->ne[1] != 1 ||
            order->view_offs != (size_t) (K - n_win)*sizeof(int32_t) || n_win < 1 || n_win > K || n_win > T) {
            return conv_chain_decline(__LINE__);
        }
        if (wins->view_src != x || wins->ne[0] != nc - 1 || wins->ne[2] != C || wins->ne[3] != ns ||
            wins->nb[1] != sizeof(float) || wins->nb[2] != x->nb[1] || wins->nb[3] != x->nb[2] ||
            wins->view_offs != (size_t) (T - (n_win - 1))*sizeof(float)) {
            return conv_chain_decline(__LINE__);
        }
        if (slots->type != GGML_TYPE_F32 || slots->nb[0] != sizeof(float) || slots->ne[0] != nc - 1 || slots->ne[1] != K ||
            slots->ne[2] != C || slots->ne[3] != ns || store->type != GGML_TYPE_F32) {
            return conv_chain_decline(__LINE__);
        }
        st_dst    = slots;
        p.st      = (char *) slots->data;
        p.st_slot = slots->nb[1];
        p.st_ch   = slots->nb[2];
        p.st_seq  = slots->nb[3];
        p.n_win   = (int) n_win;
    }

    // every intermediate is read only inside the chain; the store and the final SCALE are the outputs
    for (int k = i; k <= i_last; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            return conv_chain_decline(__LINE__);
        }
        if (n == scl || n == store) {
            continue;
        }
        if (n->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return conv_chain_decline(__LINE__);
        }
        int uses = 0;
        for (int m = k; m <= i_last; ++m) { // from k: a cast reads itself as src[1]
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[m]->src[s] == n;
            }
        }
        if (uses != ggml_node_get_use_count(cgraph, k)) {
            return conv_chain_decline(__LINE__);
        }
    }
    if (conv_chain_overlap(scl, w) || conv_chain_overlap(st_dst, w)) {
        return conv_chain_decline(__LINE__);
    }
    // with the recurrent states read in place, past is the cell the state is stored into: a thread reads all of the past
    // of its channel before it stores the state there, so past may be the stored cell itself with the same strides
    const bool past_in_place = store->op == GGML_OP_CPY && past->data == st_dst->data &&
        past->nb[1] == (size_t) p.st_ch && past->nb[2] == (size_t) p.st_seq;
    // dst may be cur (in place), nothing else may alias an output
    if ((scl->data != cur->data && conv_chain_overlap(scl, cur)) || conv_chain_overlap(scl, past) || conv_chain_overlap(scl, dyn) ||
        conv_chain_overlap(scl, st_dst) || conv_chain_overlap(st_dst, cur) || (!past_in_place && conv_chain_overlap(st_dst, past)) ||
        conv_chain_overlap(st_dst, dyn)) {
        return conv_chain_decline(__LINE__);
    }

    float sc[2];
    memcpy(sc, scl->op_params, sizeof(sc));

    p.cur         = (const float *) cur->data;
    p.past        = (const float *) past->data;
    p.past_nb1    = past->nb[1]/sizeof(float);
    p.past_nb2    = past->nb[2]/sizeof(float);
    p.w           = (const float *) w->data;
    p.w_nb1       = w->nb[1]/sizeof(float);
    p.dyn         = dyn ? (const float *) dyn->data : nullptr;
    p.ch_per_head = (int) ch_per_head;
    p.n_head      = (int) n_head;
    p.dst         = (float *) scl->data;
    p.scale       = sc[0];
    p.bias        = sc[1];
    p.conv_bias   = 0.0f;
    p.C           = (int) C;
    p.T           = (int) T;
    if (past->nb[1] % sizeof(float) || past->nb[2] % sizeof(float) || w->nb[1] % sizeof(float)) {
        return conv_chain_decline(__LINE__);
    }

    // GGML_CUDA_CONV_CHAIN_DEBUG=N logs the first N fused chains
    static const int n_debug = getenv("GGML_CUDA_CONV_CHAIN_DEBUG") ? atoi(getenv("GGML_CUDA_CONV_CHAIN_DEBUG")) : 0;
    if (n_debug > 0) {
        static int n_logged = 0;
        if (n_logged++ < n_debug) {
            GGML_LOG_WARN("%s: fused %d nodes at %d: C %lld T %lld ns %lld n_conv %lld dyn %d store %s n_win %d\n", __func__,
                i_last - i + 1, i, (long long) C, (long long) T, (long long) ns, (long long) nc, dyn_chain ? 1 : 0,
                ggml_op_name(store->op), p.n_win);
        }
    }

    cudaStream_t stream = ctx.stream();
    switch (nc) {
        case 2: conv_chain_launch<2>(p, (int) ns, dyn_chain, stream); break;
        case 3: conv_chain_launch<3>(p, (int) ns, dyn_chain, stream); break;
        case 4: conv_chain_launch<4>(p, (int) ns, dyn_chain, stream); break;
        case 5: conv_chain_launch<5>(p, (int) ns, dyn_chain, stream); break;
        default: GGML_ABORT("unreachable");
    }
    return i_last - i;
}
