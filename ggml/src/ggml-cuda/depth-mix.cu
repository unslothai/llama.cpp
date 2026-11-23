#include "depth-mix.cuh"
#include "mmvf.cuh"
#include "fwht.cuh"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

#define DEPTH_MIX_BLOCK 1024
#define DEPTH_MIX_MAX_N 32
#define DEPTH_MIX_MAX_NQ 3
#define DEPTH_MIX_MAX_NCOLS 8192

struct depth_mix_params {
    const float * normed; int64_t normed_s2;                // [ncols, 1, T], the RMS_NORM output (computed by its own kernel)
    const float * prev;   int64_t prev_s1,   prev_s2;       // [ncols, n - 1, T]
    const float * prev_t; int64_t prev_t_s1, prev_t_s2;     // [n - 1, ncols, T]
    const float * q;      int64_t q_s1,      q_s2;          // [ncols, nq, T], q_s2 = 0 for a repeated query
    float * states;       int64_t states_s1, states_s2;     // CONCAT outputs, if used outside the chain
    float * states_t;     int64_t states_t_s1, states_t_s2;
    float * scores;       int64_t scores_s1, scores_s2;     // SOFT_MAX output, if used outside the chain
    float * out;          int64_t out_s1,    out_s2;        // the second MUL_MAT, if fused
    int   ncols;
    int   n;                                                // rows of the mix, the earlier blocks + x
    float scale;
    int   mm_block;                                         // the block size mul_mat_vec_f picks for ncols
    int   rows_per_block;                                   // rows of out (and of the written intermediates) per block
};

template <int nq>
static __global__ void __launch_bounds__(DEPTH_MIX_BLOCK, 1) depth_mix_f32(const depth_mix_params p) {
    extern __shared__ float4 depth_mix_smem[];
    float * normed = (float *) depth_mix_smem; // [ncols]
    __shared__ float red_mm[nq][DEPTH_MIX_BLOCK / WARP_SIZE];
    __shared__ float raw[nq * DEPTH_MIX_MAX_N];
    __shared__ float sc[nq * DEPTH_MIX_MAX_N];

    const int     tid   = threadIdx.x;
    const int     warp  = tid / WARP_SIZE;
    const int     lane  = tid % WARP_SIZE;
    const int64_t t     = blockIdx.y;
    const int     ncols = p.ncols;
    const int     n     = p.n;

    ggml_cuda_pdl_sync();
    for (int col = tid; col < ncols; col += DEPTH_MIX_BLOCK) {
        normed[col] = p.normed[col + t*p.normed_s2];
    }
    ggml_cuda_pdl_lc();
    __syncthreads();

    // scores = states^T q as mul_mat_vec_f<float, float, nq, mm_block>: one group of mm_block threads per row
    {
        const int g_size  = p.mm_block;
        const int ngroups = DEPTH_MIX_BLOCK / g_size;
        const int g       = tid / g_size;
        const int gt      = tid % g_size;
        const int ncols2  = ncols / 2;
        const float2 * q2 = (const float2 *) (p.q + t*p.q_s2);
        const int64_t q_s1_2 = p.q_s1 / 2;

        for (int k0 = 0; k0 < n; k0 += ngroups) {
            const int k = k0 + g;
            float sumf[nq] = {0.0f};
            if (g < ngroups && k < n) {
                const float2 * x2 = (const float2 *) (k < n - 1 ? p.prev + k*p.prev_s1 + t*p.prev_s2 : normed);
                for (int col2 = gt; col2 < ncols2; col2 += g_size) {
                    const float2 tmpx = x2[col2];
#pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        const float2 tmpy = q2[j*q_s1_2 + col2];
                        ggml_cuda_mad(sumf[j], tmpx.x, tmpy.x);
                        ggml_cuda_mad(sumf[j], tmpx.y, tmpy.y);
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < nq; ++j) {
                sumf[j] = warp_reduce_sum(sumf[j]);
            }
            if (g_size > WARP_SIZE) {
                if (lane == 0) {
#pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        red_mm[j][warp] = sumf[j];
                    }
                }
                __syncthreads();
                if (g < ngroups && gt < WARP_SIZE) {
                    const int nw = g_size / WARP_SIZE;
#pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        float v = 0.0f;
                        if (lane < nw) {
                            v = red_mm[j][g*nw + lane];
                        }
                        sumf[j] = warp_reduce_sum(v);
                    }
                }
                __syncthreads();
            }
            if (g < ngroups && k < n && gt == 0) {
#pragma unroll
                for (int j = 0; j < nq; ++j) {
                    raw[j*n + k] = sumf[j];
                }
            }
        }
    }
    __syncthreads();

    // SOFT_MAX_EXT without mask and sinks as soft_max_f32 with one warp per row (n <= WARP_SIZE)
    if (warp < nq) {
        const int col = lane;
        float max_val = -INFINITY;
        float val     = 0.0f;
        if (col < n) {
            val = raw[warp*n + col]*p.scale + 0.0f;
            max_val = max(max_val, val);
        }
        max_val = warp_reduce_max(max_val);
        float tmp = 0.0f;
        if (col < n) {
            val = expf(val - max_val);
            tmp += val;
        }
        tmp = warp_reduce_sum(tmp);
        const float inv_sum = 1.0f / tmp;
        if (col < n) {
            const float s = val * inv_sum;
            sc[warp*n + col] = s;
            if (p.scores && blockIdx.x == 0) {
                p.scores[col + warp*p.scores_s1 + t*p.scores_s2] = s;
            }
        }
    }
    __syncthreads();

    const int e0 = blockIdx.x*p.rows_per_block;
    const int e1 = min(ncols, e0 + p.rows_per_block);

    // out = states_t^T scores as mul_mat_vec_f<float, float, nq, WARP_SIZE> computes it with one warp per row: lane l
    // accumulates columns 2l and 2l + 1 (n <= 32, one iteration), then warp_reduce_sum adds lane l^o to lane l for
    // o = 16, 8, 4, 2, 1. Here one thread per row forms the same 32 lane partials (zero past n/2) and the same additions
    // that end in lane 0. Floating-point addition is commutative, so a + b and b + a round alike.
    if (p.out) {
        const int nh = n / 2;
        for (int e = e0 + tid; e < e1; e += DEPTH_MIX_BLOCK) {
            // the earlier rows transposed: a CONCAT of their own, or the plain rows read across (the same values)
            const float * xr  = p.prev_t ? p.prev_t + e*p.prev_t_s1 + t*p.prev_t_s2 : p.prev + e + t*p.prev_s2;
            const int64_t xs  = p.prev_t ? 1 : p.prev_s1;
            const float   xn  = normed[e];
            float v[nq][WARP_SIZE/2];
#pragma unroll
            for (int l = 0; l < WARP_SIZE/2; ++l) {
#pragma unroll
                for (int j = 0; j < nq; ++j) {
                    v[j][l] = 0.0f;
                }
                if (l < nh) {
                    const int k = 2*l;
                    const float x0 = k     < n - 1 ? xr[k*xs]       : xn;
                    const float x1 = k + 1 < n - 1 ? xr[(k + 1)*xs] : xn;
#pragma unroll
                    for (int j = 0; j < nq; ++j) {
                        float sumf = 0.0f;
                        ggml_cuda_mad(sumf, x0, sc[j*n + k]);
                        ggml_cuda_mad(sumf, x1, sc[j*n + k + 1]);
                        v[j][l] = sumf;
                    }
                }
            }
#pragma unroll
            for (int j = 0; j < nq; ++j) {
                // o = 16: lanes 16..31 hold 0
#pragma unroll
                for (int l = 0; l < WARP_SIZE/2; ++l) {
                    v[j][l] = __fadd_rn(v[j][l], 0.0f);
                }
#pragma unroll
                for (int o = WARP_SIZE/4; o > 0; o >>= 1) {
#pragma unroll
                    for (int l = 0; l < o; ++l) {
                        v[j][l] = __fadd_rn(v[j][l], v[j][l + o]);
                    }
                }
                p.out[e + j*p.out_s1 + t*p.out_s2] = v[j][0];
            }
        }
    }

    // intermediates used outside the chain
    if (p.states || p.states_t) {
        for (int e = e0 + tid; e < e1; e += DEPTH_MIX_BLOCK) {
            const float v = normed[e];
            if (p.states) {
                for (int k = 0; k < n; ++k) {
                    p.states[e + k*p.states_s1 + t*p.states_s2] = k < n - 1 ? p.prev[e + k*p.prev_s1 + t*p.prev_s2] : v;
                }
            }
            if (p.states_t) {
                for (int k = 0; k < n; ++k) {
                    p.states_t[k + e*p.states_t_s1 + t*p.states_t_s2] = k >= n - 1 ? v :
                        p.prev_t ? p.prev_t[k + e*p.prev_t_s1 + t*p.prev_t_s2] : p.prev[e + k*p.prev_s1 + t*p.prev_s2];
                }
            }
        }
    }
}

// GGML_CUDA_DEPTH_MIX_SHADOW=1 (debug only, with GGML_CUDA_DISABLE_GRAPHS=1): run the fused kernel into scratch
// buffers, let the separate ops run, and compare the bytes of every output of the chain before the next node runs
struct depth_mix_shadow_entry {
    const ggml_cgraph * graph;
    int end;
    const ggml_tensor * real;
    float * scratch;
    char desc[128];
};

static bool depth_mix_shadow() {
    static const bool shadow = getenv("GGML_CUDA_DEPTH_MIX_SHADOW") != nullptr;
    return shadow;
}

static std::vector<depth_mix_shadow_entry> & depth_mix_shadow_list() {
    static std::vector<depth_mix_shadow_entry> pending;
    return pending;
}

static void depth_mix_shadow_check(const ggml_cgraph * cgraph, int i, cudaStream_t stream) {
    std::vector<depth_mix_shadow_entry> & pending = depth_mix_shadow_list();
    std::vector<depth_mix_shadow_entry> keep;
    for (auto & e : pending) {
        if (e.graph == cgraph && e.end >= i) {
            keep.push_back(e);
            continue;
        }
        if (e.graph == cgraph) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const size_t nb = ggml_nbytes(e.real);
            std::vector<float> a(nb/sizeof(float)), b(nb/sizeof(float));
            CUDA_CHECK(cudaMemcpy(a.data(), e.real->data, nb, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(b.data(), e.scratch, nb, cudaMemcpyDeviceToHost));
            size_t ndiff = 0;
            for (size_t k = 0; k < a.size(); ++k) {
                ndiff += memcmp(&a[k], &b[k], sizeof(float)) != 0;
            }
            fprintf(stderr, "depth_mix_shadow: %s %s: %zu/%zu differ\n", e.desc, e.real->name, ndiff, a.size());
        }
        CUDA_CHECK(cudaFree(e.scratch));
    }
    pending = keep;
}

static void depth_mix_shadow_add(const ggml_cgraph * cgraph, int end, const ggml_tensor * real, float *& ptr, const char * desc) {
    depth_mix_shadow_entry e;
    e.graph = cgraph;
    e.end   = end;
    e.real  = real;
    CUDA_CHECK(cudaMalloc(&e.scratch, ggml_nbytes(real)));
    snprintf(e.desc, sizeof(e.desc), "%s", desc);
    ptr = e.scratch;
    depth_mix_shadow_list().push_back(e);
}

static bool depth_mix_debug() {
    static const bool debug = getenv("GGML_CUDA_DEPTH_MIX_DEBUG") != nullptr;
    return debug;
}

#define DEPTH_MIX_DECLINE(reason) do { if (depth_mix_debug()) { fprintf(stderr, "%s: node %d (%s): %s\n", __func__, i, cgraph->nodes[i]->name, reason); } return 0; } while (0)

static bool depth_mix_rows_ok(const ggml_tensor * t) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[3] == 1;
}

int ggml_cuda_try_fuse_depth_mix(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i, depth_mix_memory_check_t memory_ok) {
    if (depth_mix_shadow()) {
        depth_mix_shadow_check(cgraph, i, ctx->stream());
    }

    // the chain starts after the RMS_NORM of the current state, which stays its own op, in one of two forms:
    //   (a) CONCAT (prev_t, RESHAPE(normed), dim 0), CONCAT (prev, normed, dim 1), [REPEAT,] MUL_MAT, SOFT_MAX, [MUL_MAT]
    //   (b) CONCAT (prev, normed, dim 1), [REPEAT,] MUL_MAT, SOFT_MAX, [PERMUTE, CONT, MUL_MAT]: the transposed rows are a
    //       copy of the plain ones made right before the product, in any order the graph puts these nodes
    ggml_tensor * first = cgraph->nodes[i];
    const bool form_b = first->op == GGML_OP_CONCAT && ggml_get_op_params_i32(first, 0) == 1 &&
        first->src[1]->op == GGML_OP_RMS_NORM;
    ggml_tensor * cat_t    = nullptr; // form (a): the transposed CONCAT; form (b): the CONT of the permuted rows
    ggml_tensor * perm     = nullptr; // form (b) only
    const ggml_tensor * normed_t = nullptr;
    const ggml_tensor * rms      = nullptr;
    if (form_b) {
        rms = first->src[1];
    } else {
        cat_t = first;
        if (cat_t->op != GGML_OP_CONCAT || ggml_get_op_params_i32(cat_t, 0) != 0 || cat_t->src[1]->op != GGML_OP_RESHAPE ||
                cat_t->src[1]->src[0]->op != GGML_OP_RMS_NORM) {
            return 0;
        }
        normed_t = cat_t->src[1];
        rms      = normed_t->src[0];
    }
    if (!depth_mix_rows_ok(rms) || !ggml_is_contiguous(rms) || rms->ne[1] != 1 || rms->ne[3] != 1) {
        return 0;
    }
    const int64_t ncols = rms->ne[0];
    const int64_t T     = rms->ne[2];

    ggml_tensor * cat_s = form_b ? first : nullptr;
    ggml_tensor * rep   = nullptr;
    ggml_tensor * mm1   = nullptr;
    ggml_tensor * sm    = nullptr;
    ggml_tensor * mm2   = nullptr;
    int end = i;
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 8; ++j) {
        ggml_tensor * node = cgraph->nodes[j];
        bool ok = false;
        if (!cat_s && node->op == GGML_OP_CONCAT && node->src[1] == rms && ggml_get_op_params_i32(node, 0) == 1) {
            cat_s = node; ok = true;
        } else if (cat_s && !rep && !mm1 && node->op == GGML_OP_REPEAT) {
            rep = node; ok = true;
        } else if (cat_s && !mm1 && node->op == GGML_OP_MUL_MAT && node->src[0] == cat_s && (!rep || node->src[1] == rep)) {
            mm1 = node; ok = true;
        } else if (mm1 && !sm && node->op == GGML_OP_SOFT_MAX && node->src[0] == mm1) {
            sm = node; ok = true;
        } else if (form_b && !perm && node->op == GGML_OP_PERMUTE && node->src[0] == cat_s &&
                node->op_params[0] == 1 && node->op_params[1] == 0 && node->op_params[2] == 2 && node->op_params[3] == 3) {
            perm = node; ok = true;
        } else if (form_b && perm && !cat_t && node->op == GGML_OP_CONT && node->src[0] == perm) {
            cat_t = node; ok = true;
        } else if (sm && cat_t && node->op == GGML_OP_MUL_MAT && node->src[0] == cat_t && node->src[1] == sm) {
            mm2 = node; ok = true;
        }
        if (!ok) {
            break;
        }
        end = j;
        if (mm2) {
            break;
        }
    }
    if (!cat_s || !mm1 || !sm) {
        return 0;
    }
    if (form_b && !mm2) {
        // no product to fuse: keep the chain up to the SOFT_MAX, unless the copy of the rows sits inside it
        cat_t = nullptr;
        perm  = nullptr;
    }

    const ggml_tensor * prev   = cat_s->src[0];
    const ggml_tensor * prev_t = form_b ? nullptr : cat_t->src[0];
    const int64_t n  = cat_s->ne[1];
    const int64_t nq = mm1->ne[1];

    if (!depth_mix_rows_ok(prev) || !depth_mix_rows_ok(cat_s) || !depth_mix_rows_ok(mm1) || !depth_mix_rows_ok(sm) ||
        (mm2 && !depth_mix_rows_ok(mm2)) ||
        prev->ne[0] != ncols || prev->ne[1] != n - 1 || prev->ne[2] != T ||
        (cat_t && (!depth_mix_rows_ok(cat_t) || cat_t->ne[0] != n || cat_t->ne[1] != ncols || cat_t->ne[2] != T)) ||
        (form_b && cat_t && !ggml_is_contiguous(cat_t)) ||
        (!form_b && (!depth_mix_rows_ok(prev_t) || prev_t->ne[0] != n - 1 || prev_t->ne[1] != ncols || prev_t->ne[2] != T ||
            normed_t->ne[0] != 1 || normed_t->ne[1] != ncols || normed_t->ne[2] != T))) {
        DEPTH_MIX_DECLINE("shapes");
    }
    // the scores read the earlier rows of the stack as float2: an F32 view at an odd float offset or with an odd row or
    // token stride is left to the separate kernels (CONCAT copies it into an aligned stack first)
    if ((uintptr_t) prev->data % (2*sizeof(float)) != 0 || prev->nb[1] % (2*sizeof(float)) != 0 ||
        prev->nb[2] % (2*sizeof(float)) != 0) {
        DEPTH_MIX_DECLINE("unaligned stack rows");
    }
    if (n < 2 || n > DEPTH_MIX_MAX_N || nq < 1 || nq > DEPTH_MIX_MAX_NQ || ncols % 2 != 0 || ncols > DEPTH_MIX_MAX_NCOLS ||
        mm1->ne[0] != n || mm1->ne[2] != T) {
        DEPTH_MIX_DECLINE("sizes");
    }

    // the query: a REPEAT of [ncols, nq, 1] or a tensor [ncols, nq, T]
    const ggml_tensor * q = rep ? rep->src[0] : mm1->src[1];
    if (!depth_mix_rows_ok(q) || q->ne[0] != ncols || q->ne[1] != nq || (rep ? q->ne[2] != 1 : q->ne[2] != T) ||
        q->nb[1] % (2*sizeof(float)) != 0 || (uintptr_t) q->data % (2*sizeof(float)) != 0 ||
        (rep && (rep->ne[0] != ncols || rep->ne[1] != nq || rep->ne[2] != T || rep->ne[3] != 1))) {
        DEPTH_MIX_DECLINE("query");
    }

    // SOFT_MAX_EXT: no mask, no sinks, no ALiBi
    float sm_params[2];
    memcpy(sm_params, sm->op_params, sizeof(sm_params));
    if (sm->src[1] || sm->src[2] || sm_params[1] != 0.0f || !ggml_is_contiguous(mm1) || !ggml_is_contiguous(sm)) {
        DEPTH_MIX_DECLINE("softmax");
    }

    // both products must run as MMVF in ggml_cuda_mul_mat
    const int device    = ctx->device;
    const int cc        = ggml_cuda_info().devices[device].cc;
    const int warp_size = ggml_cuda_info().devices[device].warp_size;
    if (warp_size != WARP_SIZE) {
        DEPTH_MIX_DECLINE("warp size");
    }
    // decode-size mixes (up to 8 tokens) before Ada (cc < 8.9) keep the separate kernels: measured ~1% slower fused
    // at batch-1 decode on A100 (cc 8.0), within noise on T4 (cc 7.5), while prefill gains everywhere (5-12%)
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_ADA_LOVELACE && T <= 8) {
        DEPTH_MIX_DECLINE("decode-size mix before Ada");
    }
    auto takes_mmvf = [&](const ggml_tensor * mm) {
        const ggml_tensor * src0 = mm->src[0];
        const ggml_tensor * src1 = mm->src[1];
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
        return !bad_padding_clear && src1->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
            !ggml_cuda_op_mul_mat_use_fwht(mm) && ggml_cuda_should_use_mmvf(src0, cc, warp_size, src1->ne[1]);
    };
    if (!takes_mmvf(mm1)) {
        DEPTH_MIX_DECLINE("first product not MMVF");
    }
    if (mm2 && !(takes_mmvf(mm2) && n % 2 == 0 && sm->nb[1] == n*sizeof(float))) {
        // the second product stays a separate op
        mm2 = nullptr;
        if (form_b) {
            cat_t = nullptr;
            perm  = nullptr;
        }
    }
    if (!mm2) {
        // end the chain at the SOFT_MAX; in form (b) the copy of the rows must then come after it
        int k_sm = -1;
        for (int j = i; j <= end; ++j) {
            if (cgraph->nodes[j] == sm) {
                k_sm = j;
                break;
            }
        }
        for (int j = i; j < k_sm; ++j) {
            const ggml_tensor * node = cgraph->nodes[j];
            if (node != cat_s && node != rep && node != mm1 && node != first) {
                DEPTH_MIX_DECLINE("rows copy inside the chain without its product");
            }
        }
        end = k_sm;
    }

    // the nodes of the chain, and which of them are used outside it
    const int count = end - i + 1;
    ggml_op ops[16];
    int     outs[16];
    int     n_outs = 0;
    for (int k = 0; k < count; ++k) {
        const ggml_tensor * node = cgraph->nodes[i + k];
        ops[k] = node->op;
        int uses = 0;
        for (int l = k + 1; l < count; ++l) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[i + l]->src[s] == node;
            }
        }
        if (k == count - 1 || uses != ggml_node_get_use_count(cgraph, i + k) || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            outs[n_outs++] = i + k;
        }
    }
    bool write_cat_s = false;
    bool write_cat_t = false;
    bool write_sm    = mm2 == nullptr;
    for (int k = 0; k < n_outs; ++k) {
        const ggml_tensor * node = cgraph->nodes[outs[k]];
        if (node == cat_s) {
            write_cat_s = true;
        } else if (node == cat_t) {
            write_cat_t = true;
        } else if (node == sm) {
            write_sm = true;
        } else if (node == perm) {
            DEPTH_MIX_DECLINE("permuted rows used outside the chain");
        } else if (node != mm2) {
            DEPTH_MIX_DECLINE("intermediate used outside the chain");
        }
    }
    if (!ggml_can_fuse_subgraph(cgraph, i, count, ops, outs, n_outs)) {
        DEPTH_MIX_DECLINE("subgraph");
    }
    // an output may reuse the memory of the RMS_NORM output, which the separate CONCATs have consumed by then;
    // every block loads its token's normed row before it writes, so one block per token is safe when only this token's
    // rows can overlap
    bool one_block = false;
    if (!memory_ok(cgraph, i, count, outs, n_outs)) {
        auto overlap = [](const ggml_tensor * a, const ggml_tensor * b) {
            const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
            const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
            return a0 < b1 && b0 < a1;
        };
        const ggml_tensor * srcs[4] = { prev, prev_t ? prev_t : prev, q, rms };
        bool rms_overlap_ok = true;
        for (int k = 0; k < n_outs && rms_overlap_ok; ++k) {
            const ggml_tensor * o = cgraph->nodes[outs[k]];
            for (const ggml_tensor * src : srcs) {
                if (!overlap(o, src)) {
                    continue;
                }
                // a single token, or the mix output in the same per-token layout as the normed rows
                rms_overlap_ok = src == rms && (T == 1 || (o == mm2 && nq == 1 && o->data == rms->data && o->nb[2] == rms->nb[2]));
            }
        }
        if (!rms_overlap_ok) {
            DEPTH_MIX_DECLINE("memory ranges");
        }
        one_block = true;
    }

    depth_mix_params p = {};
    p.normed = (const float *) rms->data;         p.normed_s2 = rms->nb[2] / sizeof(float);
    p.prev = (const float *) prev->data;          p.prev_s1 = prev->nb[1] / sizeof(float);     p.prev_s2 = prev->nb[2] / sizeof(float);
    if (prev_t) {
        p.prev_t = (const float *) prev_t->data;  p.prev_t_s1 = prev_t->nb[1] / sizeof(float); p.prev_t_s2 = prev_t->nb[2] / sizeof(float);
    }
    p.q = (const float *) q->data;                p.q_s1 = q->nb[1] / sizeof(float);           p.q_s2 = rep ? 0 : q->nb[2] / sizeof(float);
    if (write_cat_s) {
        p.states = (float *) cat_s->data;         p.states_s1 = cat_s->nb[1] / sizeof(float);   p.states_s2 = cat_s->nb[2] / sizeof(float);
    }
    if (write_cat_t) {
        p.states_t = (float *) cat_t->data;       p.states_t_s1 = cat_t->nb[1] / sizeof(float); p.states_t_s2 = cat_t->nb[2] / sizeof(float);
    }
    if (write_sm) {
        p.scores = (float *) sm->data;            p.scores_s1 = sm->nb[1] / sizeof(float);      p.scores_s2 = sm->nb[2] / sizeof(float);
    }
    if (mm2) {
        p.out = (float *) mm2->data;              p.out_s1 = mm2->nb[1] / sizeof(float);        p.out_s2 = mm2->nb[2] / sizeof(float);
    }
    p.ncols = ncols;
    p.n     = n;
    p.scale = sm_params[0];

    // the block size of launch_mul_mat_vec_f_cuda for the first product
    {
        int64_t block_size_best = warp_size;
        int64_t niter_best      = (ncols + 2*warp_size - 1) / (2*warp_size);
        int64_t max_block_size  = 256;
        if (cc > GGML_CUDA_CC_OFFSET_AMD && cc < GGML_CUDA_CC_RDNA1) {
            max_block_size = 128;
        }
        for (int64_t block_size = 2*warp_size; block_size <= max_block_size; block_size += warp_size) {
            const int64_t niter = (ncols + 2*block_size - 1) / (2*block_size);
            if (niter < niter_best) {
                niter_best      = niter;
                block_size_best = block_size;
            }
        }
        p.mm_block = block_size_best;
    }

    // with work per row of the output, spread each token over several blocks
    int nsplit = 1;
    if (!one_block && (mm2 || write_cat_s || write_cat_t)) {
        const int nsm = ggml_cuda_info().devices[device].nsm;
        nsplit = std::max<int64_t>(1, std::min<int64_t>(4, (nsm + T - 1) / T));
    }
    p.rows_per_block = (ncols + nsplit - 1) / nsplit;

    const bool shadow = depth_mix_shadow();
    if (shadow) {
        char desc[128];
        snprintf(desc, sizeof(desc), "node %d n %d nq %d T %d out %d writes %d%d%d", i, (int) n, (int) nq, (int) T, mm2 != nullptr,
            write_cat_s, write_cat_t, write_sm);
        if (p.out)      { depth_mix_shadow_add(cgraph, end, mm2,   p.out,      desc); }
        if (p.scores)   { depth_mix_shadow_add(cgraph, end, sm,    p.scores,   desc); }
        if (p.states)   { depth_mix_shadow_add(cgraph, end, cat_s, p.states,   desc); }
        if (p.states_t) { depth_mix_shadow_add(cgraph, end, cat_t, p.states_t, desc); }
    }

    const dim3 grid(nsplit, T, 1);
    const size_t smem = ncols*sizeof(float);
    const ggml_cuda_kernel_launch_params launch_params(grid, dim3(DEPTH_MIX_BLOCK, 1, 1), smem, ctx->stream());
    switch (nq) {
        case 1: ggml_cuda_kernel_launch(depth_mix_f32<1>, launch_params, p); break;
        case 2: ggml_cuda_kernel_launch(depth_mix_f32<2>, launch_params, p); break;
        case 3: ggml_cuda_kernel_launch(depth_mix_f32<3>, launch_params, p); break;
        default: GGML_ABORT("fatal error");
    }

    if (depth_mix_debug()) {
        fprintf(stderr, "%s: node %d (%s): fused %d nodes, ncols %d, n %d, nq %d, T %d, out %d, writes %d%d%d, mm block %d, blocks %d\n",
            __func__, i, cgraph->nodes[i]->name, count, (int) ncols, (int) n, (int) nq, (int) T, mm2 != nullptr,
            write_cat_s, write_cat_t, write_sm, p.mm_block, nsplit);
    }
    return shadow ? 0 : count - 1;
}
