#include "depth-mix.cuh"
#include "mmvf.cuh"
#include "fwht.cuh"
#include "ggml-backend-impl.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

// vendors/hip.h maps only cudaStreamCaptureModeRelaxed of the capture API (vendors/musa.h maps these three as well)
#if defined(GGML_USE_HIP) && !defined(cudaStreamIsCapturing)
#define cudaStreamCaptureStatus     hipStreamCaptureStatus
#define cudaStreamCaptureStatusNone hipStreamCaptureStatusNone
#define cudaStreamIsCapturing       hipStreamIsCapturing
#endif

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
    float * out;          int64_t out_se, out_sj, out_s2;   // the second MUL_MAT, if fused: element (e, j) of token t at e*out_se + j*out_sj + t*out_s2 ([ncols, nq] or [nq, ncols])
    int   ncols;
    int   n;                                                // rows of the mix, the earlier blocks + x
    float scale;
    int   mm_block;                                         // the block size mul_mat_vec_f picks for ncols
    int   rows_per_block;                                   // rows of out (and of the written intermediates) per block

    // the norm of cur in the kernel (pre_mode 1: RMS_NORM of pre_x, 2: RMS_NORM of the ADD pre_x + pre_y, its sum written to pre_sum), and the norm of the mix output (post_mode 1: RMS_NORM, 2: RMS_NORM, MUL by post_w): both as the vector path of rms_norm_f32 with one float4 per thread computes them
    int   pre_mode;
    const float * pre_x;  int64_t pre_x_s2;
    const float * pre_y;  int64_t pre_y_s2;
    float * pre_sum;      int64_t pre_sum_s2;
    float * pre_normed;                                     // the RMS_NORM output, if used outside (stride normed_s2)
    float   pre_eps;
    int   post_mode;
    const float * post_w;
    float * post_norm;    int64_t post_norm_s2;             // the RMS_NORM output, if used outside or the last node
    float * post_out;     int64_t post_out_s2;              // the MUL output
    float   post_eps;
};

template <int nq>
static __global__ void __launch_bounds__(DEPTH_MIX_BLOCK, 1) depth_mix_f32(const depth_mix_params p) {
    extern __shared__ float4 depth_mix_smem[];
    float * normed = (float *) depth_mix_smem; // [ncols]
    float * mixed  = normed + p.ncols;         // [ncols], the mix output when its norm is fused
    __shared__ float red_mm[nq][DEPTH_MIX_BLOCK / WARP_SIZE];
    __shared__ float red_norm[DEPTH_MIX_BLOCK / WARP_SIZE];
    __shared__ float raw[nq * DEPTH_MIX_MAX_N];
    __shared__ float sc[nq * DEPTH_MIX_MAX_N];

    const int     tid   = threadIdx.x;
    const int     warp  = tid / WARP_SIZE;
    const int     lane  = tid % WARP_SIZE;
    const int64_t t     = blockIdx.y;
    const int     ncols = p.ncols;
    const int     n     = p.n;

    ggml_cuda_pdl_sync();
    if (p.pre_mode == 0) {
        for (int col = tid; col < ncols; col += DEPTH_MIX_BLOCK) {
            normed[col] = p.normed[col + t*p.normed_s2];
        }
    } else {
        // add_rms_norm_f32 / rms_norm_f32 (vector path, one float4 per thread, ncols/4 <= DEPTH_MIX_BLOCK): the threads past ncols/4 add +0 to the block reduction, which leaves the sum of a smaller block unchanged
        const int ncols4 = ncols / 4;
        float4 x   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float  tmp = 0.0f;
        if (tid < ncols4) {
            const float4 xv = ((const float4 *) (p.pre_x + t*p.pre_x_s2))[tid];
            if (p.pre_mode == 2) {
                const float4 yv = ((const float4 *) (p.pre_y + t*p.pre_y_s2))[tid];
                x = make_float4(__fadd_rn(xv.x, yv.x), __fadd_rn(xv.y, yv.y), __fadd_rn(xv.z, yv.z), __fadd_rn(xv.w, yv.w));
                ((float4 *) (p.pre_sum + t*p.pre_sum_s2))[tid] = x;
            } else {
                x = xv;
            }
            tmp += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
        }
        tmp = block_reduce<block_reduce_method::SUM, DEPTH_MIX_BLOCK>(tmp, red_norm);
        const float scale = rsqrtf(__fadd_rn(tmp / ncols, p.pre_eps));
        if (tid < ncols4) {
            const float4 r = make_float4(scale*x.x, scale*x.y, scale*x.z, scale*x.w);
            ((float4 *) normed)[tid] = r;
            if (p.pre_normed) {
                ((float4 *) (p.pre_normed + t*p.normed_s2))[tid] = r;
            }
        }
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

    // out = states_t^T scores as mul_mat_vec_f<float, float, nq, WARP_SIZE> computes it with one warp per row: lane l accumulates columns 2l and 2l + 1 (n <= 32, one iteration), then warp_reduce_sum adds lane l^o to lane l for o = 16, 8, 4, 2, 1.
    // Here one thread per row forms the same 32 lane partials (zero past n/2) and the same additions that end in lane 0. Floating-point addition is commutative, so a + b and b + a round alike.
    if (p.out || p.post_mode) {
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
                if (p.out) {
                    p.out[e*p.out_se + j*p.out_sj + t*p.out_s2] = v[j][0];
                }
                if (p.post_mode && j == 0) {
                    mixed[e] = v[j][0];
                }
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

    // the norm of the mix output (one query, one block per token) as rms_norm_f32 (vector path, one float4 per thread): dst = scale*x [* w], the scale rounded like that kernel's mean and eps add
    if (p.post_mode) {
        __syncthreads();
        const int ncols4 = ncols / 4;
        float4 x   = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float  tmp = 0.0f;
        if (tid < ncols4) {
            x = ((const float4 *) mixed)[tid];
            tmp += x.x*x.x + x.y*x.y + x.z*x.z + x.w*x.w;
        }
        tmp = block_reduce<block_reduce_method::SUM, DEPTH_MIX_BLOCK>(tmp, red_norm);
        const float scale = rsqrtf(__fadd_rn(tmp / ncols, p.post_eps));
        if (tid < ncols4) {
            float4 r = make_float4(scale*x.x, scale*x.y, scale*x.z, scale*x.w);
            if (p.post_norm) {
                ((float4 *) (p.post_norm + t*p.post_norm_s2))[tid] = r;
            }
            if (p.post_mode == 2) {
                const float4 m = ((const float4 *) p.post_w)[tid];
                r.x = __fmul_rn(r.x, m.x);
                r.y = __fmul_rn(r.y, m.y);
                r.z = __fmul_rn(r.z, m.z);
                r.w = __fmul_rn(r.w, m.w);
                ((float4 *) (p.post_out + t*p.post_out_s2))[tid] = r;
            }
        }
    }
}

// GGML_CUDA_DEPTH_MIX_SHADOW=1 (debug only, with GGML_CUDA_DISABLE_GRAPHS=1): run the fused kernel into scratch buffers, let the separate ops run, and compare the bytes of every output of the chain before the next node runs
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

void ggml_cuda_depth_mix_shadow_flush(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph) {
    // a chain that ends at the last node has no later node to trigger its compare: compare and free it at the end of the graph
    if (depth_mix_shadow() && !depth_mix_shadow_list().empty()) {
        depth_mix_shadow_check(cgraph, cgraph->n_nodes, ctx->stream());
    }
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

static bool depth_mix_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a0 = (uintptr_t) a->data, a1 = a0 + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);
    const uintptr_t b0 = (uintptr_t) b->data, b1 = b0 + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);
    return a0 < b1 && b0 < a1;
}

// rows of [ncols, 1, T] F32 that the vector path of rms_norm_f32 reads or writes as float4: contiguous rows, 16-byte aligned data and strides
static bool depth_mix_rows4_ok(const ggml_tensor * t, int64_t ncols, int64_t T) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && t->ne[0] == ncols && ggml_nrows(t) == T &&
        (t->ne[1] == 1 || t->ne[2] == 1) && t->ne[3] == 1 &&
        (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

// the token stride of rows [ncols, 1, T] or [ncols, T]
static int64_t depth_mix_row_stride(const ggml_tensor * t) {
    return (t->ne[1] == 1 ? t->nb[2] : t->nb[1]) / sizeof(float);
}

static int depth_mix_try(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i, depth_mix_memory_check_t memory_ok, bool ext) {
    // ext: the RMS_NORM of cur (after an ADD, [RESHAPE,] or alone) and the RMS_NORM[, MUL] of a one-query mix output after [RESHAPE,] are taken by the kernel as well: [ADD, [RESHAPE,]] RMS_NORM, the chain, [[RESHAPE,] RMS_NORM[, MUL]]
    const int i0 = i;
    ggml_tensor * pre_add = nullptr;
    ggml_tensor * pre_rs  = nullptr;
    ggml_tensor * pre_rms = nullptr;
    if (ext) {
        ggml_tensor * n0 = cgraph->nodes[i];
        int k = i;
        if (n0->op == GGML_OP_ADD) {
            k = i + 1;
            if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_RESHAPE && cgraph->nodes[k]->src[0] == n0) {
                pre_rs = cgraph->nodes[k++];
            }
            if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_RMS_NORM && cgraph->nodes[k]->src[0] == (pre_rs ? pre_rs : n0)) {
                pre_add = n0;
                pre_rms = cgraph->nodes[k];
            } else {
                return 0;
            }
        } else if (n0->op == GGML_OP_RMS_NORM) {
            pre_rms = n0;
        }
        if (pre_rms) {
            // the views of the norm that the chain reads (the transposed row of form (a)) come between it and the chain
            i = k + 1;
            while (i < cgraph->n_nodes && cgraph->nodes[i]->op == GGML_OP_RESHAPE && cgraph->nodes[i]->src[0] == pre_rms) {
                ++i;
            }
            if (i >= cgraph->n_nodes) {
                return 0;
            }
        }
    }

    // the chain starts after the RMS_NORM of the current state, which stays its own op, in one of two forms:
    //   (a) CONCAT (prev_t, RESHAPE(normed), dim 0), CONCAT (prev, normed, dim 1), [REPEAT,] MUL_MAT, SOFT_MAX, [MUL_MAT]
    //   (b) CONCAT (prev, normed, dim 1), [REPEAT,] MUL_MAT, SOFT_MAX, [PERMUTE, CONT, MUL_MAT]: the transposed rows are a copy of the plain ones made right before the product, in any order the graph puts these nodes
    // In (b) the transposed rows may also be their own CONCAT (prev_t, RESHAPE(normed), dim 0) placed after the plain one, with the RESHAPE right before it (the graph order when the last product reads the scores first).
    // The last MUL_MAT is either mul_mat(states_t, scores), out [ncols, nq, T], or mul_mat(scores, states_t) with the scores as src0, out [nq, ncols, T]: the same products and sums, only the layout of the output differs
    ggml_tensor * first = cgraph->nodes[i];
    const bool form_b = first->op == GGML_OP_CONCAT && ggml_get_op_params_i32(first, 0) == 1 &&
        first->src[1]->op == GGML_OP_RMS_NORM;
    ggml_tensor * cat_t    = nullptr; // form (a): the transposed CONCAT; form (b): the CONT of the permuted rows
    ggml_tensor * perm     = nullptr; // form (b) only
    bool cat_t_concat      = !form_b; // the transposed rows are a CONCAT of their own (form (a), or (b) with a later CONCAT)
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
    if (pre_rms && rms != pre_rms) {
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
        } else if (form_b && !perm && !cat_t && !normed_t && node->op == GGML_OP_RESHAPE && node->src[0] == rms) {
            normed_t = node; ok = true; // the view the later CONCAT of the transposed rows reads
        } else if (form_b && !perm && !cat_t && node->op == GGML_OP_CONCAT && ggml_get_op_params_i32(node, 0) == 0 &&
                node->src[1]->op == GGML_OP_RESHAPE && node->src[1]->src[0] == rms) {
            cat_t = node; cat_t_concat = true; normed_t = node->src[1]; ok = true;
        } else if (sm && cat_t && node->op == GGML_OP_MUL_MAT &&
                ((node->src[0] == cat_t && node->src[1] == sm) || (node->src[0] == sm && node->src[1] == cat_t))) {
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
        cat_t_concat = false;
    }

    const ggml_tensor * prev   = cat_s->src[0];
    const ggml_tensor * prev_t = cat_t_concat ? cat_t->src[0] : nullptr;
    const int64_t n  = cat_s->ne[1];
    const int64_t nq = mm1->ne[1];

    if (!depth_mix_rows_ok(prev) || !depth_mix_rows_ok(cat_s) || !depth_mix_rows_ok(mm1) || !depth_mix_rows_ok(sm) ||
        (mm2 && !depth_mix_rows_ok(mm2)) ||
        prev->ne[0] != ncols || prev->ne[1] != n - 1 || prev->ne[2] != T ||
        (cat_t && (!depth_mix_rows_ok(cat_t) || cat_t->ne[0] != n || cat_t->ne[1] != ncols || cat_t->ne[2] != T)) ||
        (form_b && cat_t && !ggml_is_contiguous(cat_t)) ||
        (cat_t_concat && (!depth_mix_rows_ok(prev_t) || prev_t->ne[0] != n - 1 || prev_t->ne[1] != ncols || prev_t->ne[2] != T ||
            normed_t->ne[0] != 1 || normed_t->ne[1] != ncols || normed_t->ne[2] != T))) {
        DEPTH_MIX_DECLINE("shapes");
    }
    // the scores read the earlier rows of the stack as float2: an F32 view at an odd float offset or with an odd row or token stride is left to the separate kernels (CONCAT copies it into an aligned stack first)
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
    // decode-size mixes (up to 8 tokens) before Ada (cc < 8.9) keep the separate kernels: measured ~1% slower fused at batch-1 decode on A100 (cc 8.0), within noise on T4 (cc 7.5), while prefill gains everywhere (5-12%)
    if (GGML_CUDA_CC_IS_NVIDIA(cc) && cc < GGML_CUDA_CC_ADA_LOVELACE && T <= 8) {
        DEPTH_MIX_DECLINE("decode-size mix before Ada");
    }
    auto takes_mmvf = [&](const ggml_tensor * mm) {
        const ggml_tensor * src0 = mm->src[0];
        const ggml_tensor * src1 = mm->src[1];
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
            && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
        return !bad_padding_clear && src1->type == GGML_TYPE_F32 && mm->type == GGML_TYPE_F32 &&
            !ggml_cuda_op_mul_mat_use_fwht(mm) && ggml_cuda_should_use_mmvf(src0, src1->type, cc, warp_size, src1->ne[1], /*mul_mat_id =*/ false);
    };
    if (!takes_mmvf(mm1)) {
        DEPTH_MIX_DECLINE("first product not MMVF");
    }
    // scores as src0: the separate product is not a MMVF (src1 has ncols columns); fused, it is computed as MMVF computes the states_t-first order.
    // The separate scores-first product sums in another order, so this form matches it to rounding (within the 1e-6 NMSE of the tests), not bit for bit, and GGML_CUDA_DEPTH_MIX_SHADOW reports byte differences for it.
    const bool mm2_flipped = mm2 && mm2->src[0] == sm;
    if (mm2 && !((mm2_flipped || takes_mmvf(mm2)) && n % 2 == 0 && sm->nb[1] == n*sizeof(float))) {
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

    // the norm of a one-query mix output after the chain
    ggml_tensor * post_rs  = nullptr;
    ggml_tensor * post_rms = nullptr;
    ggml_tensor * post_mul = nullptr;
    int end_ext = end;
    if (ext && mm2 && nq == 1) {
        int k = end + 1;
        if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_RESHAPE && cgraph->nodes[k]->src[0] == mm2) {
            post_rs = cgraph->nodes[k++];
        }
        if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_RMS_NORM && cgraph->nodes[k]->src[0] == (post_rs ? post_rs : mm2)) {
            post_rms = cgraph->nodes[k++];
            if (k < cgraph->n_nodes && cgraph->nodes[k]->op == GGML_OP_MUL && cgraph->nodes[k]->src[0] == post_rms) {
                const ggml_tensor * w = cgraph->nodes[k]->src[1];
                // a weight row broadcast over the tokens, read as float4 like rms_norm_f32's fused multiply reads it
                if (w->type == GGML_TYPE_F32 && w->ne[0] == ncols && ggml_nelements(w) == ncols && ggml_is_contiguous(w) &&
                        (uintptr_t) w->data % 16 == 0 && ggml_are_same_shape(cgraph->nodes[k], post_rms) &&
                        depth_mix_rows4_ok(cgraph->nodes[k], ncols, T)) {
                    post_mul = cgraph->nodes[k];
                    end_ext  = k;
                }
            }
        }
        // only the weighted norm (build_norm): a plain RMS_NORM of the mix output starts the loop controller's chain
        if (!post_mul) {
            post_rs  = nullptr;
            post_rms = nullptr;
        }
    }
    if (ext) {
        if (!pre_rms && !post_rms) {
            return 0;
        }
        // both norms as the vector path of rms_norm_f32 with one float4 per thread computes them, where that is the path the separate kernels take: 16-byte aligned rows and ncols/4 <= the block
        if (ncols % 4 != 0 || ncols/4 > DEPTH_MIX_BLOCK) {
            DEPTH_MIX_DECLINE("norm: row length");
        }
        if (pre_rms) {
            const ggml_tensor * x = pre_rs ? pre_rs : pre_add ? pre_add : pre_rms->src[0];
            if (!depth_mix_rows4_ok(pre_rms, ncols, T) || !depth_mix_rows4_ok(x, ncols, T)) {
                DEPTH_MIX_DECLINE("norm of cur: layout");
            }
            if (pre_add) {
                const ggml_tensor * a = pre_add->src[0];
                const ggml_tensor * b = pre_add->src[1];
                if (!ggml_are_same_shape(a, pre_add) || !ggml_are_same_shape(b, pre_add) || !ggml_is_contiguous(a) ||
                        !ggml_is_contiguous(b) || !ggml_is_contiguous(pre_add) ||
                        (uintptr_t) a->data % 16 != 0 || (uintptr_t) b->data % 16 != 0 || (uintptr_t) pre_add->data % 16 != 0 ||
                        a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || pre_add->type != GGML_TYPE_F32 ||
                        ggml_nelements(pre_add) != ncols*T) {
                    DEPTH_MIX_DECLINE("norm of cur: add");
                }
            }
            float eps;
            memcpy(&eps, pre_rms->op_params, sizeof(float));
            if (!(eps >= 0.0f)) {
                DEPTH_MIX_DECLINE("norm of cur: eps");
            }
        }
        if (post_rms) {
            const ggml_tensor * x = post_rs ? post_rs : mm2;
            if (!depth_mix_rows4_ok(post_rms, ncols, T) || !depth_mix_rows4_ok(x, ncols, T) || !ggml_is_contiguous(mm2)) {
                DEPTH_MIX_DECLINE("norm of the mix: layout");
            }
        }
    }

    // the nodes of the chain, and which of them are used outside it. A RESHAPE of the normed rows inside the chain (form (b) with a later CONCAT of the transposed rows) is a view of a tensor computed before the chain: it is skipped with the chain but left out of the fused set, since it computes nothing. With the norm of cur in the kernel the RESHAPE is a view of a fused node and stays in the set.
    const int count = end_ext - i0 + 1;
    ggml_op ops[24];
    int     idxs[24];
    int     outs[24];
    int     n_idx  = 0;
    int     n_outs = 0;
    if (count > 24) {
        DEPTH_MIX_DECLINE("chain length");
    }
    for (int k = 0; k < count; ++k) {
        const ggml_tensor * node = cgraph->nodes[i0 + k];
        if (node == normed_t && node->op == GGML_OP_RESHAPE && node != first && !pre_rms) {
            continue;
        }
        idxs[n_idx] = i0 + k;
        ops[n_idx++] = node->op;
        int uses = 0;
        for (int l = k + 1; l < count; ++l) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                uses += cgraph->nodes[i0 + l]->src[s] == node;
            }
        }
        if (k == count - 1 || uses != ggml_node_get_use_count(cgraph, i0 + k) || (node->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            outs[n_outs++] = i0 + k;
        }
    }
    bool write_cat_s = false;
    bool write_cat_t = false;
    bool write_sm    = mm2 == nullptr;
    bool write_mm2   = !ext || !post_rms;
    bool write_pre   = false;
    bool write_post  = post_rms && !post_mul;
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
        } else if (node == mm2 || (node == post_rs && post_rs)) {
            write_mm2 = true;
        } else if (node == pre_add || (node == pre_rs && pre_rs)) {
            // the sum is always written
        } else if (pre_rms && (node == pre_rms || (node->op == GGML_OP_RESHAPE && node->src[0] == pre_rms))) {
            write_pre = true;
        } else if (node == post_rms && post_rms) {
            write_post = true;
        } else if (node == post_mul && post_mul) {
        } else {
            DEPTH_MIX_DECLINE("intermediate used outside the chain");
        }
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, n_idx, ops, outs, n_outs)) {
        DEPTH_MIX_DECLINE("subgraph");
    }
    bool one_block = false;
    if (ext) {
        // nothing written may overlap an input of the chain, except an input of the norm of cur: each block (one per token) reads it into registers or shared memory before a barrier and writes only after that, so an output over it is safe for one token, or when it covers the same rows of the same token (the sum over an addend, the mix output over a dead addend)
        std::vector<const ggml_tensor *> written;
        if (pre_add)            { written.push_back(pre_add); }
        if (write_pre)          { written.push_back(pre_rms); }
        if (write_cat_s)        { written.push_back(cat_s); }
        if (write_cat_t)        { written.push_back(cat_t); }
        if (write_sm)           { written.push_back(sm); }
        if (write_mm2 && mm2)   { written.push_back(mm2); }
        if (write_post)         { written.push_back(post_rms); }
        if (post_mul)           { written.push_back(post_mul); }
        std::vector<const ggml_tensor *> inputs = { cat_s->src[0], q };
        if (prev_t) {
            inputs.push_back(prev_t);
        }
        if (post_mul) {
            inputs.push_back(post_mul->src[1]);
        }
        std::vector<const ggml_tensor *> norm_inputs;
        if (pre_add) {
            norm_inputs.push_back(pre_add->src[0]);
            norm_inputs.push_back(pre_add->src[1]);
        } else if (pre_rms) {
            norm_inputs.push_back(pre_rms->src[0]);
        } else {
            norm_inputs.push_back(rms);
        }
        for (const ggml_tensor * o : written) {
            for (const ggml_tensor * src : inputs) {
                if (depth_mix_overlap(o, src)) {
                    if (depth_mix_debug()) {
                        fprintf(stderr, "depth_mix: overlap out %s (%s) src %s (%s)\n", o->name, ggml_op_name(o->op), src->name, ggml_op_name(src->op));
                    }
                    DEPTH_MIX_DECLINE("norm: memory ranges");
                }
            }
            for (const ggml_tensor * src : norm_inputs) {
                if (depth_mix_overlap(o, src) && T > 1 && !(o->data == src->data && depth_mix_row_stride(o) == depth_mix_row_stride(src) &&
                        o->nb[0] == src->nb[0] && ggml_nrows(o) == ggml_nrows(src) && o->ne[0] == src->ne[0])) {
                    DEPTH_MIX_DECLINE("norm: memory ranges of the norm input");
                }
            }
        }
        for (size_t k = 0; k < written.size(); ++k) {
            for (size_t l = k + 1; l < written.size(); ++l) {
                if (depth_mix_overlap(written[k], written[l])) {
                    DEPTH_MIX_DECLINE("norm: outputs overlap");
                }
            }
        }
        one_block = true;
    } else if (!memory_ok(cgraph, i, count, outs, n_outs)) {
        // an output may reuse the memory of the RMS_NORM output, which the separate CONCATs have consumed by then; every block loads its token's normed row before it writes, so one block per token is safe when only this token's rows can overlap
        const ggml_tensor * srcs[4] = { prev, prev_t ? prev_t : prev, q, rms };
        bool rms_overlap_ok = true;
        for (int k = 0; k < n_outs && rms_overlap_ok; ++k) {
            const ggml_tensor * o = cgraph->nodes[outs[k]];
            for (const ggml_tensor * src : srcs) {
                if (!depth_mix_overlap(o, src)) {
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
    if (mm2 && write_mm2) {
        p.out = (float *) mm2->data;              p.out_s2 = mm2->nb[2] / sizeof(float);
        p.out_se = mm2_flipped ? mm2->nb[1] / sizeof(float) : 1;
        p.out_sj = mm2_flipped ? 1 : mm2->nb[1] / sizeof(float);
    }
    if (pre_rms) {
        p.pre_mode = pre_add ? 2 : 1;
        const ggml_tensor * x = pre_add ? pre_add->src[0] : pre_rms->src[0];
        p.pre_x = (const float *) x->data;        p.pre_x_s2 = ncols;
        if (pre_add) {
            p.pre_y   = (const float *) pre_add->src[1]->data; p.pre_y_s2 = ncols;
            p.pre_sum = (float *) pre_add->data;               p.pre_sum_s2 = ncols;
        } else {
            p.pre_x_s2 = depth_mix_row_stride(x);
        }
        if (write_pre) {
            p.pre_normed = (float *) pre_rms->data;
        }
        memcpy(&p.pre_eps, pre_rms->op_params, sizeof(float));
    }
    if (post_rms) {
        p.post_mode = post_mul ? 2 : 1;
        if (write_post) {
            p.post_norm = (float *) post_rms->data; p.post_norm_s2 = depth_mix_row_stride(post_rms);
        }
        if (post_mul) {
            p.post_w   = (const float *) post_mul->src[1]->data;
            p.post_out = (float *) post_mul->data;  p.post_out_s2 = depth_mix_row_stride(post_mul);
        }
        memcpy(&p.post_eps, post_rms->op_params, sizeof(float));
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
        if (p.out)      { depth_mix_shadow_add(cgraph, end_ext, mm2,   p.out,      desc); }
        if (p.scores)   { depth_mix_shadow_add(cgraph, end_ext, sm,    p.scores,   desc); }
        if (p.states)   { depth_mix_shadow_add(cgraph, end_ext, cat_s, p.states,   desc); }
        if (p.states_t) { depth_mix_shadow_add(cgraph, end_ext, cat_t, p.states_t, desc); }
        if (p.pre_sum)    { depth_mix_shadow_add(cgraph, end_ext, pre_add,  p.pre_sum,    "pre sum"); }
        if (p.pre_normed) { depth_mix_shadow_add(cgraph, end_ext, pre_rms,  p.pre_normed, "pre normed"); }
        if (p.post_norm)  { depth_mix_shadow_add(cgraph, end_ext, post_rms, p.post_norm,  "post norm"); }
        if (p.post_out)   { depth_mix_shadow_add(cgraph, end_ext, post_mul, p.post_out,   "post out"); }
    }

    const dim3 grid(nsplit, T, 1);
    const size_t smem = (post_rms ? 2 : 1)*ncols*sizeof(float);
    const ggml_cuda_kernel_launch_params launch_params(grid, dim3(DEPTH_MIX_BLOCK, 1, 1), smem, ctx->stream());
    switch (nq) {
        case 1: ggml_cuda_kernel_launch(depth_mix_f32<1>, launch_params, p); break;
        case 2: ggml_cuda_kernel_launch(depth_mix_f32<2>, launch_params, p); break;
        case 3: ggml_cuda_kernel_launch(depth_mix_f32<3>, launch_params, p); break;
        default: GGML_ABORT("fatal error");
    }

    if (depth_mix_debug()) {
        fprintf(stderr, "%s: node %d (%s): fused %d nodes, ncols %d, n %d, nq %d, T %d, out %d, writes %d%d%d, mm block %d, blocks %d, pre %d post %d\n",
            __func__, i0, cgraph->nodes[i0]->name, count, (int) ncols, (int) n, (int) nq, (int) T, mm2 != nullptr,
            write_cat_s, write_cat_t, write_sm, p.mm_block, nsplit, p.pre_mode, p.post_mode);
    }
    return shadow ? 0 : count - 1;
}

int ggml_cuda_try_fuse_depth_mix(ggml_backend_cuda_context * ctx, ggml_cgraph * cgraph, int i, depth_mix_memory_check_t memory_ok) {
    if (depth_mix_shadow()) {
        // a CUDA graph capture can neither wait on the stream nor keep the scratch buffers alive for its replays: leave the chain to the separate ops
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx->stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return 0;
        }
        depth_mix_shadow_check(cgraph, i, ctx->stream());
    }
    // the norms around the mix in the kernel, on NVIDIA cc >= 10.0 only: the add + rms_norm rounding is replicated as measured there (B200, RTX PRO 6000)
    static const bool norm_disabled = getenv("GGML_CUDA_DISABLE_DEPTH_MIX_NORM_FUSION") != nullptr;
    const int cc = ggml_cuda_info().devices[ctx->device].cc;
    if (!norm_disabled && GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= 1000) {
        const int n_fused = depth_mix_try(ctx, cgraph, i, memory_ok, true);
        if (n_fused > 0) {
            return n_fused;
        }
    }
    return depth_mix_try(ctx, cgraph, i, memory_ok, false);
}
