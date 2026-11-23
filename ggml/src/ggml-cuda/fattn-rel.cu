#include "fattn-rel.cuh"
#include "fattn-vec.cuh"
#include "fattn.cuh"
#include "ggml-impl.h"

#include <cinttypes>
#include <vector>

static bool ggml_cuda_flash_attn_ext_vec_rel(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * rel_mask, const fattn_rel_bias & rel, const bool dry_run) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

#define FATTN_VEC_REL_CASE(D, tK, tV)                                                         \
    if (Q->ne[0] == (D) && K->type == (tK) && V->type == (tV)) {                              \
        if (!dry_run) {                                                                        \
            ggml_cuda_flash_attn_ext_vec_rel_case<D, tK, tV>(ctx, dst, rel_mask, rel);         \
        }                                                                                      \
        return true;                                                                           \
    }

    FATTN_VEC_REL_CASE( 64, GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_REL_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_REL_CASE(256, GGML_TYPE_F16,  GGML_TYPE_F16)
    FATTN_VEC_REL_CASE( 64, GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_REL_CASE(128, GGML_TYPE_BF16, GGML_TYPE_BF16)
    FATTN_VEC_REL_CASE(256, GGML_TYPE_BF16, GGML_TYPE_BF16)
#undef FATTN_VEC_REL_CASE

    return false;
}

static int32_t ggml_cuda_rel_use_count(const ggml_cgraph * cgraph, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&cgraph->visited_hash_set, t);
    if (pos == GGML_HASHSET_FULL || !ggml_bitset_get(cgraph->visited_hash_set.used, pos)) {
        return 0;
    }
    return cgraph->use_counts[pos];
}

// the views from t up to its source src are only read by the next step and src is not an output
static bool ggml_cuda_rel_private_chain(const ggml_cgraph * cgraph, const ggml_tensor * t, const ggml_tensor * src) {
    for (; t != src; t = t->src[0]) {
        if (!t || t->view_src != src || ggml_cuda_rel_use_count(cgraph, t) != 1 || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
    }
    return ggml_cuda_rel_use_count(cgraph, src) == 1 && !(src->flags & GGML_TENSOR_FLAG_OUTPUT);
}

static bool ggml_cuda_rel_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

int ggml_cuda_try_fuse_rel_bias_fa(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i,
        bool (*compute)(ggml_backend_cuda_context & ctx, ggml_tensor * dst)) {
    const ggml_tensor * node = cgraph->nodes[i];
    if (node->op != GGML_OP_SUB && node->op != GGML_OP_CONT && node->op != GGML_OP_GET_ROWS) {
        return 0;
    }

    // the flash attention reading the chain, with only views and chain nodes in between
    int f = -1;
    for (int k = i + 1; k < cgraph->n_nodes && k <= i + 32; ++k) {
        if (cgraph->nodes[k]->op == GGML_OP_FLASH_ATTN_EXT) {
            f = k;
            break;
        }
    }
    if (f < 0) {
        return 0;
    }
    ggml_tensor * fa = cgraph->nodes[f];

    const ggml_tensor * add = fa->src[3];
    if (!add || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F16 || fa->src[5] != nullptr) {
        return 0;
    }
    float params[3];
    memcpy(params, fa->op_params, sizeof(params));
    if (params[1] != 0.0f || params[2] != 0.0f) {
        return 0; // ALiBi, logit softcap
    }

    const ggml_tensor * cpy  = add->src[0];
    const ggml_tensor * mask = add->src[1];
    if (!cpy || cpy->op != GGML_OP_CPY || cpy->type != GGML_TYPE_F16 || !ggml_is_contiguous(cpy) || !ggml_is_contiguous(add) ||
            !ggml_are_same_shape(cpy, add) || mask->type != GGML_TYPE_F16 || mask->nb[0] != sizeof(half) ||
            mask->ne[0] != add->ne[0] || mask->ne[1] != add->ne[1] || mask->ne[2] != 1 || mask->ne[3] != add->ne[3] ||
            mask->nb[1] != add->nb[1]) {
        return 0;
    }

    const ggml_tensor * Q = fa->src[0];
    const ggml_tensor * K = fa->src[1];
    const int64_t n_kv   = add->ne[0];
    const int64_t nr     = add->ne[1];
    const int64_t n_head = add->ne[2];
    const int64_t ns     = add->ne[3];
    if (K->ne[1] != n_kv || Q->ne[1] != nr || Q->ne[2] != n_head || Q->ne[3] != ns) {
        return 0;
    }

    // the F32 bias [n_kv, nr, n_head, ns]: a permuted view of the gathered rows G = [n_head, n_kv, nr*ns]
    const ggml_tensor * bias = cpy->src[0];
    const ggml_tensor * G    = bias->view_src;
    if (bias->type != GGML_TYPE_F32 || !ggml_are_same_shape(bias, add) || !G || G->op != GGML_OP_GET_ROWS ||
            G->type != GGML_TYPE_F32 || !ggml_is_contiguous(G) || bias->data != G->data ||
            G->ne[0] != n_head || G->ne[1] != n_kv || G->ne[2] != nr*ns || G->ne[3] != 1 ||
            bias->nb[0] != G->nb[1] || bias->nb[1] != G->nb[2] || bias->nb[2] != G->nb[0] || bias->nb[3] != nr*G->nb[2] ||
            !ggml_cuda_rel_private_chain(cgraph, bias, G)) {
        return 0;
    }
    if (ggml_cuda_rel_use_count(cgraph, cpy) != 1 || ggml_cuda_rel_use_count(cgraph, add) != 1 ||
            (cpy->flags & GGML_TENSOR_FLAG_OUTPUT) || (add->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    // the CPY is not run, so its destination must not be seen anywhere else: not an output, not a view of another
    // tensor, and read by nothing but the CPY (a view of it read elsewhere counts as a use as well)
    const ggml_tensor * cpy_dst = cpy->src[1];
    if (cpy_dst != cpy && (!cpy_dst || cpy_dst->view_src || (cpy_dst->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_cuda_rel_use_count(cgraph, cpy_dst) != 1)) {
        return 0;
    }

    // G[h, kv, t] = T[h, rp[kv, t], t]
    const ggml_tensor * T  = G->src[0];
    const ggml_tensor * rp = G->src[1];
    if (T->type != GGML_TYPE_F32 || T->ne[0] != n_head || T->ne[2] != nr*ns || T->ne[3] != 1 ||
            rp->type != GGML_TYPE_I32 || rp->ne[0] != n_kv || rp->ne[1] != nr*ns || rp->ne[2] != 1 || rp->ne[3] != 1) {
        return 0;
    }

    fattn_rel_bias rel = {};
    rel.rp    = (const char *) rp->data;
    rel.rp_kv = rp->nb[0];
    rel.rp_q  = rp->nb[1];
    rel.rp_s  = nr*rp->nb[1];

    const ggml_tensor * tab_src = T;
    const ggml_tensor * view    = T;
    const ggml_tensor * cont    = nullptr;
    const ggml_tensor * sub     = nullptr;
    // T = CONT(view of S): read S through the view
    if (T->op == GGML_OP_CONT && T->src[0]->type == GGML_TYPE_F32 && T->src[0]->view_src &&
            ggml_are_same_shape(T, T->src[0]) && ggml_cuda_rel_private_chain(cgraph, T->src[0], T->src[0]->view_src) &&
            ggml_cuda_rel_use_count(cgraph, T) == 1 && !(T->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        cont    = T;
        view    = T->src[0];
        tab_src = view->view_src;
        // S = A - B, B broadcast over the rows of A
        const ggml_tensor * S = tab_src;
        if (S->op == GGML_OP_SUB && ggml_is_contiguous(S) && ggml_is_contiguous(S->src[0]) && ggml_are_same_shape(S, S->src[0]) &&
                S->src[0]->type == GGML_TYPE_F32 && S->src[1]->type == GGML_TYPE_F32 && S->ne[3] == 1 &&
                S->src[1]->ne[0] == 1 && S->src[1]->ne[1] == S->ne[1] && S->src[1]->ne[2] == S->ne[2] && S->src[1]->ne[3] == 1 &&
                ggml_cuda_rel_use_count(cgraph, S) == 1 && !(S->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            // B is indexed by (head, token) of the view, so the view must be exactly the [head, rel row, token]
            // permutation of all of S; a slice or another stride order would subtract the wrong entries of B
            if (view->data != S->data || view->ne[0] != S->ne[1] || view->ne[1] != S->ne[0] || view->ne[2] != S->ne[2] ||
                    view->nb[0] != S->nb[1] || view->nb[1] != S->nb[0] || view->nb[2] != S->nb[2]) {
                return 0;
            }
            sub = S;
        }
    }

    const int64_t view_offs = (const char *) view->data - (const char *) tab_src->data;
    const ggml_tensor * tab = sub ? sub->src[0] : tab_src;
    rel.tab   = (const char *) tab->data + view_offs;
    rel.tab_h = view->nb[0];
    rel.tab_r = view->nb[1];
    rel.tab_q = view->nb[2];
    rel.tab_s = nr*view->nb[2];
    if (sub) {
        const ggml_tensor * B = sub->src[1];
        rel.sub   = (const char *) B->data;
        rel.sub_h = B->nb[1];
        rel.sub_q = B->nb[2];
        rel.sub_s = nr*B->nb[2];
    }

    // the chain starts at node i, and every other node up to the flash attention is a view or a chain node
    const ggml_tensor * start = sub ? (const ggml_tensor *) sub : cont ? cont : G;
    if (node != start) {
        return 0;
    }
    const ggml_tensor * chain[] = { sub, cont, G, cpy, add };
    auto in_chain = [&](const ggml_tensor * t) {
        for (const ggml_tensor * c : chain) {
            if (t && c && (t == c || t->view_src == c)) {
                return true;
            }
        }
        return false;
    };
    std::vector<ggml_tensor *> others; // computed before the fused kernel, in graph order
    for (int k = i + 1; k < f; ++k) {
        ggml_tensor * t = cgraph->nodes[k];
        if (t == cont || t == G || t == cpy || t == add || ggml_is_empty(t) ||
                t->op == GGML_OP_NONE || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW ||
                t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE) {
            continue;
        }
        // an independent node: it must not read the chain, and the chain must not read it
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE) || t->op == GGML_OP_FLASH_ATTN_EXT || others.size() >= 4) {
            if (getenv("GGML_CUDA_REL_BIAS_FA_DEBUG")) {
                fprintf(stderr, "%s: %s has %s (%s) in between, not fused\n", __func__, fa->name, t->name, ggml_op_name(t->op));
            }
            return 0;
        }
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (in_chain(t->src[j])) {
                return 0;
            }
        }
        for (const ggml_tensor * c : chain) {
            for (int j = 0; c && j < GGML_MAX_SRC; ++j) {
                if (c->src[j] && (c->src[j] == t || c->src[j]->view_src == t)) {
                    return 0;
                }
            }
        }
        others.push_back(t);
    }

    // nothing written before or by the fused kernel may overlap what it reads (the skipped nodes do not write): the
    // independent nodes run first here, but the allocator may have placed them over a table input that it freed after
    // the chain's first node, and the attention output may share memory with the inputs as well
    for (const ggml_tensor * t : { tab, (const ggml_tensor *) (sub ? sub->src[1] : nullptr), rp, mask }) {
        if (!t) {
            continue;
        }
        for (const ggml_tensor * w : others) {
            if (ggml_cuda_rel_overlap(w, t)) {
                if (getenv("GGML_CUDA_REL_BIAS_FA_DEBUG")) {
                    fprintf(stderr, "%s: %s (in between) overlaps %s, not fused\n", __func__, w->name, t->name);
                }
                return 0;
            }
        }
        if (ggml_cuda_rel_overlap(fa, t)) {
            if (getenv("GGML_CUDA_REL_BIAS_FA_DEBUG")) {
                fprintf(stderr, "%s: %s overlaps %s, not fused\n", __func__, fa->name, t->name);
            }
            return 0;
        }
    }

    if (!ggml_cuda_flash_attn_ext_uses_vec(ctx.device, fa) || !ggml_cuda_flash_attn_ext_vec_rel(ctx, fa, mask, rel, true)) {
        return 0;
    }

    static const bool debug = getenv("GGML_CUDA_REL_BIAS_FA_DEBUG") != nullptr;
    if (debug) {
        fprintf(stderr, "%s: fused %d nodes into %s (sub %d, cont %d, n_kv %" PRId64 ", rows %" PRId64 ", streams %" PRId64 ")\n",
            __func__, f - i + 1, fa->name, sub != nullptr, cont != nullptr, n_kv, nr, ns);
    }

    for (ggml_tensor * t : others) {
        GGML_ASSERT(compute(ctx, t));
    }
    ggml_cuda_set_device(ctx.device);
    ggml_cuda_flash_attn_ext_vec_rel(ctx, fa, mask, rel, false);
    return f - i;
}
