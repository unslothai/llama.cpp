#include "attn-res-mix.cuh"

#include <cmath>
#include <cstdint>

static constexpr int ATTN_RES_MAX_STATES = GGML_MAX_SRC - 1; // 9

struct attn_res_mix_states {
    const float * ptr[ATTN_RES_MAX_STATES];
    int64_t       nb1[ATTN_RES_MAX_STATES]; // row stride in bytes
};

template <typename Tq>
static __device__ __forceinline__ float load_query_elem(const Tq * p) {
    if constexpr (std::is_same_v<Tq, float>) {
        return *p;
    } else if constexpr (std::is_same_v<Tq, half>) {
        return __half2float(*p);
    } else {
        return __bfloat162float(*p);
    }
}

template <typename Tq>
static __device__ __forceinline__ float4 load_query_f4(const Tq * p) {
    if constexpr (std::is_same_v<Tq, float>) {
        return *reinterpret_cast<const float4 *>(p);
    } else if constexpr (std::is_same_v<Tq, half>) {
        const half2 * h2 = reinterpret_cast<const half2 *>(p);
        const float2 a = __half22float2(h2[0]);
        const float2 b = __half22float2(h2[1]);
        return make_float4(a.x, a.y, b.x, b.y);
    } else {
        const nv_bfloat162 * b2 = reinterpret_cast<const nv_bfloat162 *>(p);
        const float2 a = __bfloat1622float2(b2[0]);
        const float2 b = __bfloat1622float2(b2[1]);
        return make_float4(a.x, a.y, b.x, b.y);
    }
}

// Fast single-pass register-cached kernel for n_embd % 4 == 0 && n_embd <= 4096 (e.g. D=2304 -> 576 threads = 18 warps)
template <typename Tq, int NS, int NQ>
static __global__ void attn_res_mix_vec_f32(
        const Tq * __restrict__ query,
        const int64_t q_nb1,
        const attn_res_mix_states states,
        float * __restrict__ dst,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int n_embd,
        const float eps,
        const float scale) {
    const int tok  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;
    const int nwarps = blockDim.x >> 5;
    const int n4   = n_embd >> 2;

    float4 xv[NS];
    float sq[NS];
    float dot[NQ][NS];

#pragma unroll
    for (int s = 0; s < NS; ++s) {
        sq[s] = 0.0f;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            dot[q][s] = 0.0f;
        }
    }

    if (tid < n4) {
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            const float * s_row = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(states.ptr[s]) + int64_t(tok) * states.nb1[s]);
            const float4 v = reinterpret_cast<const float4 *>(s_row)[tid];
            xv[s] = v;
            sq[s] = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const Tq * q_row = reinterpret_cast<const Tq *>(
                reinterpret_cast<const char *>(query) + int64_t(q) * q_nb1);
            const float4 qv = load_query_f4(q_row + tid * 4);
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                dot[q][s] = xv[s].x * qv.x + xv[s].y * qv.y + xv[s].z * qv.z + xv[s].w * qv.w;
            }
        }
    }

    // Warp reductions
    __shared__ float s_sq[32][NS];
    __shared__ float s_dot[32][NQ][NS];

#pragma unroll
    for (int s = 0; s < NS; ++s) {
        const float ws = warp_reduce_sum(sq[s]);
        if (lane == 0) {
            s_sq[warp][s] = ws;
        }
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            const float wd = warp_reduce_sum(dot[q][s]);
            if (lane == 0) {
                s_dot[warp][q][s] = wd;
            }
        }
    }
    __syncthreads();

    __shared__ float s_eff_w[NQ][NS];
    if (warp == 0) {
        float inv_rms[NS];
        const float inv_d = 1.0f / float(n_embd);
#pragma unroll
        for (int s = 0; s < NS; ++s) {
            float v = (lane < nwarps) ? s_sq[lane][s] : 0.0f;
            v = warp_reduce_sum(v);
            inv_rms[s] = rsqrtf(v * inv_d + eps);
        }

#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float logits[NS];
            float m = -INFINITY;
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                float v = (lane < nwarps) ? s_dot[lane][q][s] : 0.0f;
                v = warp_reduce_sum(v);
                logits[s] = v * inv_rms[s] * scale;
                m = fmaxf(m, logits[s]);
            }
            if (lane == 0) {
                float sum_e = 0.0f;
                float exps[NS];
#pragma unroll
                for (int s = 0; s < NS; ++s) {
                    exps[s] = expf(logits[s] - m);
                    sum_e += exps[s];
                }
                const float inv_e = 1.0f / sum_e;
#pragma unroll
                for (int s = 0; s < NS; ++s) {
                    s_eff_w[q][s] = (exps[s] * inv_e) * inv_rms[s];
                }
            }
        }
    }
    __syncthreads();

    if (tid < n4) {
        char * dst_tok = reinterpret_cast<char *>(dst) + int64_t(tok) * dst_nb2;
#pragma unroll
        for (int q = 0; q < NQ; ++q) {
            float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int s = 0; s < NS; ++s) {
                const float w = s_eff_w[q][s];
                acc.x += w * xv[s].x;
                acc.y += w * xv[s].y;
                acc.z += w * xv[s].z;
                acc.w += w * xv[s].w;
            }
            float4 * dst_row = reinterpret_cast<float4 *>(dst_tok + int64_t(q) * dst_nb1);
            dst_row[tid] = acc;
        }
    }
}

// Generic 2-pass streaming kernel for arbitrary n_embd, n_states in [1..9], n_q >= 1
template <typename Tq>
static __global__ void attn_res_mix_generic_f32(
        const Tq * __restrict__ query,
        const int64_t q_nb1,
        const attn_res_mix_states states,
        const int n_states,
        const int n_q,
        float * __restrict__ dst,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int n_embd,
        const float eps,
        const float scale) {
    constexpr int NWARPS = 8; // 256 threads
    const int tok  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;

    __shared__ float s_red[NWARPS];
    __shared__ float s_inv_rms[ATTN_RES_MAX_STATES];
    __shared__ float s_eff_w[ATTN_RES_MAX_STATES];

    // Pass 1: inv_rms per state
    const float inv_d = 1.0f / float(n_embd);
    for (int s = 0; s < n_states; ++s) {
        const float * s_row = reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(states.ptr[s]) + int64_t(tok) * states.nb1[s]);
        float partial = 0.0f;
        for (int d = tid; d < n_embd; d += blockDim.x) {
            const float v = s_row[d];
            partial += v * v;
        }
        partial = warp_reduce_sum(partial);
        if (lane == 0) {
            s_red[warp] = partial;
        }
        __syncthreads();
        if (warp == 0) {
            float total = (lane < NWARPS) ? s_red[lane] : 0.0f;
            total = warp_reduce_sum(total);
            if (lane == 0) {
                s_inv_rms[s] = rsqrtf(total * inv_d + eps);
            }
        }
        __syncthreads();
    }

    char * dst_tok = reinterpret_cast<char *>(dst) + int64_t(tok) * dst_nb2;

    // Pass 2: per query row, compute dots -> softmax -> weighted sum
    for (int q = 0; q < n_q; ++q) {
        const Tq * q_row = reinterpret_cast<const Tq *>(
            reinterpret_cast<const char *>(query) + int64_t(q) * q_nb1);
        float logits[ATTN_RES_MAX_STATES];
        for (int s = 0; s < n_states; ++s) {
            const float * s_row = reinterpret_cast<const float *>(
                reinterpret_cast<const char *>(states.ptr[s]) + int64_t(tok) * states.nb1[s]);
            float partial = 0.0f;
            for (int d = tid; d < n_embd; d += blockDim.x) {
                partial += s_row[d] * load_query_elem(q_row + d);
            }
            partial = warp_reduce_sum(partial);
            if (lane == 0) {
                s_red[warp] = partial;
            }
            __syncthreads();
            if (warp == 0) {
                float total = (lane < NWARPS) ? s_red[lane] : 0.0f;
                total = warp_reduce_sum(total);
                if (lane == 0) {
                    s_eff_w[s] = total * s_inv_rms[s] * scale;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            float m = -INFINITY;
            for (int s = 0; s < n_states; ++s) {
                logits[s] = s_eff_w[s];
                m = fmaxf(m, logits[s]);
            }
            float sum_e = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                logits[s] = expf(logits[s] - m);
                sum_e += logits[s];
            }
            const float inv_e = 1.0f / sum_e;
            for (int s = 0; s < n_states; ++s) {
                s_eff_w[s] = (logits[s] * inv_e) * s_inv_rms[s];
            }
        }
        __syncthreads();

        float * dst_row = reinterpret_cast<float *>(dst_tok + int64_t(q) * dst_nb1);
        for (int d = tid; d < n_embd; d += blockDim.x) {
            float acc = 0.0f;
            for (int s = 0; s < n_states; ++s) {
                const float * s_row = reinterpret_cast<const float *>(
                    reinterpret_cast<const char *>(states.ptr[s]) + int64_t(tok) * states.nb1[s]);
                acc += s_eff_w[s] * s_row[d];
            }
            dst_row[d] = acc;
        }
        __syncthreads();
    }
}

template <typename Tq>
static void launch_attn_res_mix_typed(
        cudaStream_t stream,
        const Tq * query,
        const int64_t q_nb1,
        const attn_res_mix_states & states,
        const int n_states,
        const int n_q,
        const int n_tokens,
        float * dst,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int n_embd,
        const float eps,
        const float scale) {
    bool aligned = ((uintptr_t) query % 16 == 0) && (q_nb1 % 16 == 0) &&
                   ((uintptr_t) dst % 16 == 0) && (dst_nb1 % 16 == 0) && (dst_nb2 % 16 == 0);
    for (int s = 0; s < n_states && aligned; ++s) {
        aligned &= ((uintptr_t) states.ptr[s] % 16 == 0) && (states.nb1[s] % 16 == 0);
    }

    if (aligned && (n_embd % 4 == 0) && n_embd <= 4096 && n_states >= 1 && n_states <= 5 &&
        (n_q == 1 || n_q == 3 || n_q == 4)) {
        const int n4 = n_embd / 4;
        const int nthreads = ((n4 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
        const dim3 grid(n_tokens, 1, 1);
        const dim3 block(nthreads, 1, 1);

#define LAUNCH_VEC(NS, NQ) \
        attn_res_mix_vec_f32<Tq, NS, NQ><<<grid, block, 0, stream>>>( \
            query, q_nb1, states, dst, dst_nb1, dst_nb2, n_embd, eps, scale)

#define DISPATCH_NS(NQ) \
        switch (n_states) { \
            case 1: LAUNCH_VEC(1, NQ); return; \
            case 2: LAUNCH_VEC(2, NQ); return; \
            case 3: LAUNCH_VEC(3, NQ); return; \
            case 4: LAUNCH_VEC(4, NQ); return; \
            case 5: LAUNCH_VEC(5, NQ); return; \
            default: break; \
        }

        switch (n_q) {
            case 1: DISPATCH_NS(1); break;
            case 3: DISPATCH_NS(3); break;
            case 4: DISPATCH_NS(4); break;
            default: break;
        }
#undef DISPATCH_NS
#undef LAUNCH_VEC
    }

    attn_res_mix_generic_f32<Tq><<<dim3(n_tokens, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        query, q_nb1, states, n_states, n_q, dst, dst_nb1, dst_nb2, n_embd, eps, scale);
}

void ggml_cuda_op_attn_res_mix(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * query = dst->src[0];
    float eps = 1e-6f;
    float scale = 1.0f;
    int32_t n_states = 0;
    memcpy(&eps,      (const char *) dst->op_params + 0, sizeof(float));
    memcpy(&scale,    (const char *) dst->op_params + 4, sizeof(float));
    memcpy(&n_states, (const char *) dst->op_params + 8, sizeof(int32_t));

    GGML_ASSERT(n_states >= 1 && n_states <= ATTN_RES_MAX_STATES);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    attn_res_mix_states states = {};
    for (int s = 0; s < n_states; ++s) {
        const ggml_tensor * st = dst->src[1 + s];
        GGML_ASSERT(st != nullptr && st->type == GGML_TYPE_F32);
        states.ptr[s] = (const float *) st->data;
        states.nb1[s] = st->nb[1];
    }

    const int n_embd   = (int) dst->ne[0];
    const int n_q      = (int) dst->ne[1];
    const int n_tokens = (int) dst->ne[2];
    cudaStream_t stream = ctx.stream();

    switch (query->type) {
        case GGML_TYPE_F32:
            launch_attn_res_mix_typed<float>(
                stream, (const float *) query->data, query->nb[1], states, n_states, n_q, n_tokens,
                (float *) dst->data, dst->nb[1], dst->nb[2], n_embd, eps, scale);
            break;
        case GGML_TYPE_F16:
            launch_attn_res_mix_typed<half>(
                stream, (const half *) query->data, query->nb[1], states, n_states, n_q, n_tokens,
                (float *) dst->data, dst->nb[1], dst->nb[2], n_embd, eps, scale);
            break;
        case GGML_TYPE_BF16:
            launch_attn_res_mix_typed<nv_bfloat16>(
                stream, (const nv_bfloat16 *) query->data, query->nb[1], states, n_states, n_q, n_tokens,
                (float *) dst->data, dst->nb[1], dst->nb[2], n_embd, eps, scale);
            break;
        default:
            GGML_ABORT("ggml_cuda_op_attn_res_mix: unsupported query type");
    }
}
