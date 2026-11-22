#include "inkling2-fuse.cuh"

#include <cmath>
#include <cstdint>

// ---------------------------------------------------------------------------
// 1. Fused control() post-gate: (cur - tanh(forget_raw)*cur) + tanh(gain_raw)*msg
// ---------------------------------------------------------------------------

static __global__ void inkling_control_gate_f32_vec4(
        const float4 * __restrict__ cur,
        const float4 * __restrict__ msg,
        const float4 * __restrict__ forget_raw,
        const float4 * __restrict__ gain_raw,
        float4 * __restrict__ dst,
        const int64_t n4) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    const float4 c = cur[i];
    const float4 m = msg[i];
    const float4 f = forget_raw[i];
    const float4 g = gain_raw[i];
    float4 o;
    o.x = (c.x - tanhf(f.x) * c.x) + tanhf(g.x) * m.x;
    o.y = (c.y - tanhf(f.y) * c.y) + tanhf(g.y) * m.y;
    o.z = (c.z - tanhf(f.z) * c.z) + tanhf(g.z) * m.z;
    o.w = (c.w - tanhf(f.w) * c.w) + tanhf(g.w) * m.w;
    dst[i] = o;
}

static __global__ void inkling_control_gate_f32(
        const float * __restrict__ cur,
        const float * __restrict__ msg,
        const float * __restrict__ forget_raw,
        const float * __restrict__ gain_raw,
        float * __restrict__ dst,
        const int64_t n) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const float c = cur[i];
    dst[i] = (c - tanhf(forget_raw[i]) * c) + tanhf(gain_raw[i]) * msg[i];
}

void ggml_cuda_op_inkling_control_gate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur,
        const ggml_tensor * msg,
        const ggml_tensor * forget_raw,
        const ggml_tensor * gain_raw,
        ggml_tensor * dst) {
    const int64_t n = ggml_nelements(dst);
    const float * c_d = (const float *) cur->data;
    const float * m_d = (const float *) msg->data;
    const float * f_d = (const float *) forget_raw->data;
    const float * g_d = (const float *) gain_raw->data;
    float * d_d = (float *) dst->data;
    const bool aligned = (n % 4 == 0) &&
                         ((uintptr_t) c_d % 16 == 0) && ((uintptr_t) m_d % 16 == 0) &&
                         ((uintptr_t) f_d % 16 == 0) && ((uintptr_t) g_d % 16 == 0) &&
                         ((uintptr_t) d_d % 16 == 0);
    constexpr int threads = 256;
    if (aligned) {
        const int64_t n4 = n / 4;
        const int blocks = (int) ((n4 + threads - 1) / threads);
        inkling_control_gate_f32_vec4<<<blocks, threads, 0, ctx.stream()>>>(
            (const float4 *) c_d, (const float4 *) m_d, (const float4 *) f_d, (const float4 *) g_d, (float4 *) d_d, n4);
    } else {
        const int blocks = (int) ((n + threads - 1) / threads);
        inkling_control_gate_f32<<<blocks, threads, 0, ctx.stream()>>>(
            c_d, m_d, f_d, g_d, d_d, n);
    }
}

// ---------------------------------------------------------------------------
// 2. Fused ADD + SCALE: dst = (a + b) * scale + bias
// ---------------------------------------------------------------------------

static __global__ void inkling_add_scale_f32_vec4(
        const float4 * __restrict__ a,
        const float4 * __restrict__ b,
        float4 * __restrict__ dst,
        const int64_t n4,
        const float scale,
        const float bias) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n4) {
        return;
    }
    const float4 av = a[i];
    const float4 bv = b[i];
    float4 o;
    o.x = (av.x + bv.x) * scale + bias;
    o.y = (av.y + bv.y) * scale + bias;
    o.z = (av.z + bv.z) * scale + bias;
    o.w = (av.w + bv.w) * scale + bias;
    dst[i] = o;
}

static __global__ void inkling_add_scale_f32(
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ dst,
        const int64_t n,
        const float scale,
        const float bias) {
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    dst[i] = (a[i] + b[i]) * scale + bias;
}

void ggml_cuda_op_add_scale(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * a,
        const ggml_tensor * b,
        ggml_tensor * dst,
        const float scale,
        const float bias) {
    const int64_t n = ggml_nelements(dst);
    const float * a_d = (const float *) a->data;
    const float * b_d = (const float *) b->data;
    float * d_d = (float *) dst->data;
    const bool aligned = (n % 4 == 0) &&
                         ((uintptr_t) a_d % 16 == 0) && ((uintptr_t) b_d % 16 == 0) && ((uintptr_t) d_d % 16 == 0);
    constexpr int threads = 256;
    if (aligned) {
        const int64_t n4 = n / 4;
        const int blocks = (int) ((n4 + threads - 1) / threads);
        inkling_add_scale_f32_vec4<<<blocks, threads, 0, ctx.stream()>>>(
            (const float4 *) a_d, (const float4 *) b_d, (float4 *) d_d, n4, scale, bias);
    } else {
        const int blocks = (int) ((n + threads - 1) / threads);
        inkling_add_scale_f32<<<blocks, threads, 0, ctx.stream()>>>(
            a_d, b_d, d_d, n, scale, bias);
    }
}

// ---------------------------------------------------------------------------
// 3. Fused loop control down-projection on pre-normalized inputs:
//      dst[r] = dot(w_down[r, 0..D-1], a) + dot(w_down[r, D..2D-1], b)
//      Eliminates intermediate CONCAT(cur_norm, msg_norm) for n_tokens == 1.
// ---------------------------------------------------------------------------

static __global__ void inkling_loop_down_concat_f32(
        const float * __restrict__ w_down,
        const float * __restrict__ a,
        const float * __restrict__ b,
        float * __restrict__ dst,
        const int64_t D,
        const int64_t w_nb1_elems) {
    constexpr int NWARPS = 8; // 256 threads
    const int r    = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & (WARP_SIZE - 1);
    const int warp = tid >> 5;
    const int n4   = (int) (D >> 2);

    const float4 * wa4 = reinterpret_cast<const float4 *>(w_down + int64_t(r) * w_nb1_elems);
    const float4 * wb4 = reinterpret_cast<const float4 *>(w_down + int64_t(r) * w_nb1_elems + D);
    const float4 * a4  = reinterpret_cast<const float4 *>(a);
    const float4 * b4  = reinterpret_cast<const float4 *>(b);

    float dot = 0.0f;
    for (int i = tid; i < n4; i += blockDim.x) {
        const float4 av   = a4[i];
        const float4 bv   = b4[i];
        const float4 wv_a = wa4[i];
        const float4 wv_b = wb4[i];
        dot += av.x * wv_a.x + av.y * wv_a.y + av.z * wv_a.z + av.w * wv_a.w
             + bv.x * wv_b.x + bv.y * wv_b.y + bv.z * wv_b.z + bv.w * wv_b.w;
    }

    dot = warp_reduce_sum(dot);
    __shared__ float s_red[NWARPS];
    if (lane == 0) {
        s_red[warp] = dot;
    }
    __syncthreads();

    if (warp == 0) {
        float tot = (lane < NWARPS) ? s_red[lane] : 0.0f;
        tot = warp_reduce_sum(tot);
        if (lane == 0) {
            dst[r] = tot;
        }
    }
}

void ggml_cuda_op_inkling_loop_down_concat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_down,
        const ggml_tensor * a,
        const ggml_tensor * b,
        ggml_tensor * dst) {
    const int rank = (int) w_down->ne[1];
    const int64_t D = a->ne[0];
    const int64_t w_nb1_elems = w_down->nb[1] / sizeof(float);
    inkling_loop_down_concat_f32<<<dim3(rank, 1, 1), dim3(256, 1, 1), 0, ctx.stream()>>>(
        (const float *) w_down->data, (const float *) a->data, (const float *) b->data,
        (float *) dst->data, D, w_nb1_elems);
}

// ---------------------------------------------------------------------------
// 4. Fused loop control up-projection + tanh gate blend:
//      f_raw = w_forget * gate;  g_raw = w_gain * gate;
//      dst[r] = (cur[r] - tanh(f_raw)*cur[r]) + tanh(g_raw)*msg[r]
//      Specialized for rank == 64 (16 lanes x float4 per row = 2 rows/warp, 16 rows/block)
// ---------------------------------------------------------------------------

static __global__ void inkling_loop_up_gate_r64_f32(
        const float * __restrict__ w_gain,
        const float * __restrict__ w_forget,
        const float * __restrict__ gate,
        const float * __restrict__ cur,
        const float * __restrict__ msg,
        float * __restrict__ dst,
        const int64_t n_embd,
        const int64_t wg_nb1_elems,
        const int64_t wf_nb1_elems) {
    __shared__ float4 s_gate[16];
    if (threadIdx.x < 16) {
        s_gate[threadIdx.x] = reinterpret_cast<const float4 *>(gate)[threadIdx.x];
    }
    __syncthreads();

    const int grp  = threadIdx.x >> 4;
    const int lane = threadIdx.x & 15;
    const int64_t row = int64_t(blockIdx.x) * 16 + grp;
    if (row >= n_embd) {
        return;
    }

    const float4 gv = s_gate[lane];
    const float4 wg = reinterpret_cast<const float4 *>(w_gain   + row * wg_nb1_elems)[lane];
    const float4 wf = reinterpret_cast<const float4 *>(w_forget + row * wf_nb1_elems)[lane];

    float dg = wg.x * gv.x + wg.y * gv.y + wg.z * gv.z + wg.w * gv.w;
    float df = wf.x * gv.x + wf.y * gv.y + wf.z * gv.z + wf.w * gv.w;

    dg += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, dg, 8, 16);
    df += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, df, 8, 16);
    dg += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, dg, 4, 16);
    df += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, df, 4, 16);
    dg += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, dg, 2, 16);
    df += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, df, 2, 16);
    dg += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, dg, 1, 16);
    df += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, df, 1, 16);

    if (lane == 0) {
        const float c = cur[row];
        dst[row] = (c - tanhf(df) * c) + tanhf(dg) * msg[row];
    }
}

void ggml_cuda_op_inkling_loop_up_gate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_gain,
        const ggml_tensor * w_forget,
        const ggml_tensor * gate,
        const ggml_tensor * cur,
        const ggml_tensor * msg,
        ggml_tensor * dst) {
    const int64_t n_embd = w_gain->ne[1];
    const int blocks = (int) ((n_embd + 15) / 16);
    inkling_loop_up_gate_r64_f32<<<dim3(blocks, 1, 1), dim3(256, 1, 1), 0, ctx.stream()>>>(
        (const float *) w_gain->data, (const float *) w_forget->data,
        (const float *) gate->data, (const float *) cur->data, (const float *) msg->data,
        (float *) dst->data, n_embd, w_gain->nb[1] / sizeof(float), w_forget->nb[1] / sizeof(float));
}

// ---------------------------------------------------------------------------
// 5. Fused single-token shortconv step (n_conv == 4, n_seq_tokens == 1, n_seqs == 1):
//      Reads past[3, C] + cur_in[C], updates state_cell[3*C] in place,
//      computes dst[c] = (cur_res[c] + sum_j w[c,j]*x[c,j] [+ dyn]) * scale + bias (+ outer_res[c])
// ---------------------------------------------------------------------------

template <bool has_outer, bool has_dyn>
static __global__ void inkling_shortconv_step_k4_f32(
        const char * __restrict__ past,
        const int64_t past_nb1,
        const float * __restrict__ cur_in,
        const char * __restrict__ weight,
        const int64_t weight_nb1,
        const float * __restrict__ dynamic,
        const float * __restrict__ cur_res,
        const float * __restrict__ outer_res,
        float * __restrict__ state_cell,
        float * __restrict__ dst,
        const int64_t n_channels,
        const int64_t head_dim,
        const float scale,
        const float bias) {
    const int64_t ch = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ch >= n_channels) {
        return;
    }
    const float * p = reinterpret_cast<const float *>(past + ch * past_nb1);
    const float p0 = p[0];
    const float p1 = p[1];
    const float p2 = p[2];
    const float c  = cur_in[ch];

    const float4 w = *reinterpret_cast<const float4 *>(weight + ch * weight_nb1);
    float conv_val = p0 * w.x + p1 * w.y + p2 * w.z + c * w.w;

    if constexpr (has_dyn) {
        const int64_t h = ch / head_dim;
        const float4 d = reinterpret_cast<const float4 *>(dynamic)[h];
        conv_val += c * d.x + p2 * d.y + p1 * d.z + p0 * d.w;
    }

    float * st = state_cell + ch * 3;
    st[0] = p1;
    st[1] = p2;
    st[2] = c;

    float out_val = (cur_res[ch] + conv_val) * scale + bias;
    if constexpr (has_outer) {
        out_val += outer_res[ch];
    }
    dst[ch] = out_val;
}

void ggml_cuda_op_inkling_shortconv_step(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * past,
        const ggml_tensor * cur_in,
        const ggml_tensor * weight,
        const ggml_tensor * cur_res,
        const ggml_tensor * outer_res,
        ggml_tensor * state_cell,
        ggml_tensor * dst,
        const float scale,
        const float bias) {
    const int64_t n_channels = weight->ne[1];
    constexpr int threads = 128;
    const int blocks = (int) ((n_channels + threads - 1) / threads);
    if (outer_res) {
        inkling_shortconv_step_k4_f32<true, false><<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) past->data, past->nb[1], (const float *) cur_in->data,
            (const char *) weight->data, weight->nb[1], nullptr, (const float *) cur_res->data,
            (const float *) outer_res->data, (float *) state_cell->data, (float *) dst->data,
            n_channels, 1, scale, bias);
    } else {
        inkling_shortconv_step_k4_f32<false, false><<<blocks, threads, 0, ctx.stream()>>>(
            (const char *) past->data, past->nb[1], (const float *) cur_in->data,
            (const char *) weight->data, weight->nb[1], nullptr, (const float *) cur_res->data,
            nullptr, (float *) state_cell->data, (float *) dst->data,
            n_channels, 1, scale, bias);
    }
}

void ggml_cuda_op_inkling_shortconv_step_dyn(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * past,
        const ggml_tensor * cur_in,
        const ggml_tensor * weight,
        const ggml_tensor * dynamic,
        const ggml_tensor * cur_res,
        ggml_tensor * state_cell,
        ggml_tensor * dst,
        const int64_t n_heads,
        const float scale,
        const float bias) {
    const int64_t n_channels = weight->ne[1];
    const int64_t head_dim   = n_channels / n_heads;
    constexpr int threads = 128;
    const int blocks = (int) ((n_channels + threads - 1) / threads);
    inkling_shortconv_step_k4_f32<false, true><<<blocks, threads, 0, ctx.stream()>>>(
        (const char *) past->data, past->nb[1], (const float *) cur_in->data,
        (const char *) weight->data, weight->nb[1], (const float *) dynamic->data,
        (const float *) cur_res->data, nullptr, (float *) state_cell->data, (float *) dst->data,
        n_channels, head_dim, scale, bias);
}

// ---------------------------------------------------------------------------
// 6. Fused Inkling 2 single-token dense kq_b bias builder (9 ops -> 1 kernel):
//      dst[h * n_kv + ik] = -sinks[h] + (idx < n_extent ? dot(rel_bias[idx], Rcur[h]) : 0)
//      Specialized for n_rel == 16 (4 x float4 in registers, coalesced writes along ik)
// ---------------------------------------------------------------------------

static __global__ void inkling_dense_kq_bias_r16_f32(
        const float4 * __restrict__ rel_bias4,
        const float4 * __restrict__ r_cur4,
        const float  * __restrict__ sinks,
        const int32_t * __restrict__ rel_pos,
        float * __restrict__ dst,
        const int64_t n_kv,
        const int32_t n_extent) {
    const int h  = (int) blockIdx.y;
    const int64_t ik = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (ik >= n_kv) {
        return;
    }

    const float4 r0 = r_cur4[h * 4 + 0];
    const float4 r1 = r_cur4[h * 4 + 1];
    const float4 r2 = r_cur4[h * 4 + 2];
    const float4 r3 = r_cur4[h * 4 + 3];
    float val = -sinks[h];

    const int32_t idx = rel_pos[ik];
    if ((uint32_t) idx < (uint32_t) n_extent) {
        const float4 * b = rel_bias4 + int64_t(idx) * 4;
        const float4 b0 = b[0];
        const float4 b1 = b[1];
        const float4 b2 = b[2];
        const float4 b3 = b[3];
        val += b0.x * r0.x + b0.y * r0.y + b0.z * r0.z + b0.w * r0.w
             + b1.x * r1.x + b1.y * r1.y + b1.z * r1.z + b1.w * r1.w
             + b2.x * r2.x + b2.y * r2.y + b2.z * r2.z + b2.w * r2.w
             + b3.x * r3.x + b3.y * r3.y + b3.z * r3.z + b3.w * r3.w;
    }
    dst[int64_t(h) * n_kv + ik] = val;
}

void ggml_cuda_op_inkling_dense_kq_bias(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * rel_bias,
        const ggml_tensor * r_cur,
        const ggml_tensor * sinks,
        const ggml_tensor * rel_pos,
        ggml_tensor * dst) {
    const int64_t n_kv     = dst->ne[0];
    const int64_t n_head   = dst->ne[2];
    const int32_t n_extent = (int32_t) rel_bias->ne[1];
    constexpr int threads  = 256;
    const dim3 grid((unsigned int) ((n_kv + threads - 1) / threads), (unsigned int) n_head, 1);
    inkling_dense_kq_bias_r16_f32<<<grid, dim3(threads, 1, 1), 0, ctx.stream()>>>(
        (const float4 *) rel_bias->data,
        (const float4 *) r_cur->data,
        (const float  *) sinks->data,
        (const int32_t *) rel_pos->data,
        (float *) dst->data,
        n_kv, n_extent);
}
