#include "common.cuh"
#include "convert.cuh"
#include "fattn-banded.cuh"
#include "fattn.cuh"

#include <algorithm>
#include <cstdint>

static __device__ __forceinline__ float fattn_banded_load(
        const char * ptr, const int type) {
    switch (type) {
        case GGML_TYPE_F32:
            return *(const float *) ptr;
        case GGML_TYPE_F16:
            return __half2float(*(const half *) ptr);
        case GGML_TYPE_BF16: {
            // Read BF16 as raw bits so this kernel needs no native BF16 support; all math stays FP32.
            const uint32_t bits = uint32_t(*(const uint16_t *) ptr) << 16;
            return __uint_as_float(bits);
        }
        default:
            return 0.0f;
    }
}

// distance of query iq to cell ik: the cell distance, or q_pos - k_pos when the cells are not the positions
static __device__ __forceinline__ int64_t fattn_banded_dist(
        const int32_t * __restrict__ k_pos, const int32_t * __restrict__ q_pos, const int64_t iq, const int64_t ik,
        const int64_t q_offset) {
    if (k_pos) {
        const int32_t kp = k_pos[ik];
        return kp < 0 ? -1 : int64_t(q_pos[iq]) - kp;
    }
    const int64_t rel_dist = iq + q_offset - ik;
    return rel_dist;
}

template<int D, int WARPS_PER_BLOCK>
static __global__ void flash_attn_ext_banded_f32(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ dst,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_batch,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        int64_t kpos_s1,
        int64_t qpos_s1,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    static_assert(D % WARP_SIZE == 0, "head dimension must be divisible by warp size");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t iq = int64_t(blockIdx.x) * WARPS_PER_BLOCK + warp;
    const int64_t ih = blockIdx.y;
    const int64_t ib = blockIdx.z;

    if (iq >= n_q || ih >= n_head_q || ib >= n_batch) {
        return;
    }

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;
    const int32_t * kp = k_pos ? k_pos + ib*kpos_s1 : nullptr;
    const int32_t * qp = q_pos ? q_pos + ib*qpos_s1 : nullptr;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;

    for (int64_t ik = 0; ik < n_kv; ++ik) {
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = fattn_banded_dist(kp, qp, iq, ik, q_offset);
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = ggml_cuda_shfl(score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    // attention sink: an extra logit with a zero value
    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + ((ib*n_q + iq)*n_head_q + ih)*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        dst_row[d] = out[j]*inv_sum;
    }
}

template<int D, int NW>
static __global__ void flash_attn_ext_banded_f32_split(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        float      * __restrict__ part,
        float scale,
        int type_k,
        int type_v,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb0,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb0,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        int64_t kpos_s1,
        int64_t qpos_s1,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        const int causal) {
    constexpr int values_per_lane = D / WARP_SIZE;
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int64_t ic = blockIdx.x;
    const int64_t ih = blockIdx.y;
    const int64_t iq = int64_t(blockIdx.z) % n_q;
    const int64_t ib = int64_t(blockIdx.z) / n_q;

    const int64_t ih_kv = ih / (n_head_q / n_head_kv);
    const char * q_row = q + uint64_t(iq)*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3;
    const int32_t * kp = k_pos ? k_pos + ib*kpos_s1 : nullptr;
    const int32_t * qp = q_pos ? q_pos + ib*qpos_s1 : nullptr;

    float q_reg[values_per_lane];
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = lane + j*WARP_SIZE;
        q_reg[j] = *(const float *)(q_row + uint64_t(d)*sizeof(float));
        out[j] = 0.0f;
    }

    float row_max = -INFINITY;
    float row_sum = 0.0f;
    const int64_t q_offset = n_kv - n_q;
    const int64_t ik0 = ic*chunk;
    const int64_t ik1 = ik0 + chunk < n_kv ? ik0 + chunk : n_kv;

    for (int64_t ik = ik0 + warp; ik < ik1; ik += NW) {
        // causal without a mask: the cells after the query (and the empty ones, with positions) are masked
        if (causal && fattn_banded_dist(kp, qp, iq, ik, q_offset) < 0) {
            continue;
        }
        if (mask && __half2float(*(const half *) (mask + uint64_t(ik)*sizeof(half) +
                uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                uint64_t(ib % mask_ne3)*m_nb3)) == -INFINITY) {
            continue;
        }

        const char * k_row = k + uint64_t(ik)*k_nb1 + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            dot += q_reg[j] * fattn_banded_load(k_row + uint64_t(d)*k_nb0, type_k);
        }
        dot = warp_reduce_sum(dot);

        float score = dot * scale;
        if (lane == 0) {
            const int64_t rel_dist = fattn_banded_dist(kp, qp, iq, ik, q_offset);
            if (rel_dist >= 0 && rel_dist < rel_extent) {
                const char * rel_value = rel +
                    uint64_t(rel_dist)*r_nb0 + uint64_t(ih)*r_nb1 +
                    uint64_t(iq)*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
                score += fattn_banded_load(rel_value, type_rel);
            }
            if (mask) {
                const char * mask_value = mask + uint64_t(ik)*sizeof(half) +
                    uint64_t(iq)*m_nb1 + uint64_t(ih % mask_ne2)*m_nb2 +
                    uint64_t(ib % mask_ne3)*m_nb3;
                score += __half2float(*(const half *) mask_value);
            }
        }
        score = ggml_cuda_shfl(score, 0, WARP_SIZE);

        if (score == -INFINITY) {
            continue;
        }

        float old_scale = 1.0f;
        float value_scale = 1.0f;
        if (score > row_max) {
            old_scale = expf(row_max - score);
            row_max = score;
        } else {
            value_scale = expf(score - row_max);
        }

        const char * v_row = v + uint64_t(ik)*v_nb1 + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = lane + j*WARP_SIZE;
            const float vv = fattn_banded_load(v_row + uint64_t(d)*v_nb0, type_v);
            out[j] = out[j]*old_scale + vv*value_scale;
        }
        row_sum = row_sum*old_scale + value_scale;
    }

    __shared__ float s_max[NW];
    __shared__ float s_sum[NW];
    __shared__ float s_out[NW][D];
    if (lane == 0) {
        s_max[warp] = row_max;
        s_sum[warp] = row_sum;
    }
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        s_out[warp][lane + j*WARP_SIZE] = out[j];
    }
    __syncthreads();

    if (warp == 0) {
        float m = -INFINITY;
#pragma unroll
        for (int w = 0; w < NW; ++w) {
            m = fmaxf(m, s_max[w]);
        }
        float acc[values_per_lane];
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            acc[j] = 0.0f;
        }
        float sum = 0.0f;
        if (m != -INFINITY) {
#pragma unroll
            for (int w = 0; w < NW; ++w) {
                if (s_max[w] == -INFINITY) {
                    continue;
                }
                const float f = expf(s_max[w] - m);
                sum += s_sum[w]*f;
#pragma unroll
                for (int j = 0; j < values_per_lane; ++j) {
                    acc[j] += s_out[w][lane + j*WARP_SIZE]*f;
                }
            }
        }
        float * p = part + ((((ib*n_q + iq)*n_head_q + ih)*n_chunks) + ic)*(D + 2);
        if (lane == 0) {
            p[0] = m;
            p[1] = sum;
        }
#pragma unroll
        for (int j = 0; j < values_per_lane; ++j) {
            p[2 + lane + j*WARP_SIZE] = acc[j];
        }
    }
}

template<int D>
static __global__ void flash_attn_ext_banded_f32_merge(
        const float * __restrict__ part,
        float       * __restrict__ dst,
        int64_t n_rows,
        int64_t n_head_q,
        int64_t n_q,
        int64_t n_chunks,
        const float * __restrict__ sinks,
        int64_t sinks_s1,
        int64_t sinks_s2) {
    constexpr int values_per_lane = D / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int64_t row = int64_t(blockIdx.x)*(blockDim.x/WARP_SIZE) + threadIdx.x/WARP_SIZE;
    if (row >= n_rows) {
        return;
    }
    const int64_t ih = row % n_head_q;
    const int64_t iq = (row / n_head_q) % n_q;
    const int64_t ib = row / (n_head_q*n_q);
    const float * p = part + row*n_chunks*(D + 2);

    float m = -INFINITY;
    for (int64_t c = 0; c < n_chunks; ++c) {
        m = fmaxf(m, p[c*(D + 2)]);
    }
    float out[values_per_lane];
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        out[j] = 0.0f;
    }
    float row_sum = 0.0f;
    if (m != -INFINITY) {
        for (int64_t c = 0; c < n_chunks; ++c) {
            const float * pc = p + c*(D + 2);
            if (pc[0] == -INFINITY) {
                continue;
            }
            const float f = expf(pc[0] - m);
            row_sum += pc[1]*f;
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] += pc[2 + lane + j*WARP_SIZE]*f;
            }
        }
    }
    float row_max = m;

    if (sinks) {
        const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        if (sink > row_max) {
            const float old_scale = expf(row_max - sink);
#pragma unroll
            for (int j = 0; j < values_per_lane; ++j) {
                out[j] *= old_scale;
            }
            row_sum = row_sum*old_scale + 1.0f;
            row_max = sink;
        } else {
            row_sum += expf(sink - row_max);
        }
    }

    const float inv_sum = row_sum == 0.0f ? 0.0f : 1.0f/row_sum;
    float * dst_row = dst + row*D;
#pragma unroll
    for (int j = 0; j < values_per_lane; ++j) {
        dst_row[lane + j*WARP_SIZE] = out[j]*inv_sum;
    }
}

// decode: few queries. A block takes one KV head, a chunk of cells and NR (query, Q head) rows of that KV head (the
// GQA groups of up to 8 / G queries), so K and V are read once per group. Groups of D/8 lanes take one cell at a
// time, every lane holding 8 dimensions of Q, K, V and the output. Per tile, each group first issues the K, V and
// mask loads of all its cells, then reduces K.Q in the group; one warp per row takes the softmax of the tile, and
// each group accumulates P.V of its cells. The groups are merged at the end.
static constexpr int fattn_banded_vec_threads = 128;
static constexpr int fattn_banded_vec_cpg     = 8;   // cells per group and tile

// 8 consecutive K/V elements [d0, d0 + 8) of a row, loaded raw and converted on use
template<ggml_type type> struct fattn_banded_kv8;

template<> struct fattn_banded_kv8<GGML_TYPE_F16> {
    int4 raw;
    __device__ __forceinline__ void load(const char * row, const int d0) { raw = *(const int4 *) (row + d0*2); }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const half2 * h = (const half2 *) &raw;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = __half22float2(h[j]);
            x[2*j + 0] = f.x;
            x[2*j + 1] = f.y;
        }
    }
};

template<> struct fattn_banded_kv8<GGML_TYPE_BF16> {
    int4 raw;
    __device__ __forceinline__ void load(const char * row, const int d0) { raw = *(const int4 *) (row + d0*2); }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const uint32_t * w = (const uint32_t *) &raw;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            x[2*j + 0] = __uint_as_float(w[j] << 16);
            x[2*j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
        }
    }
};

template<> struct fattn_banded_kv8<GGML_TYPE_Q8_0> {
    uint32_t q[2];
    half     d;
    __device__ __forceinline__ void load(const char * row, const int d0) {
        const block_q8_0 * b = (const block_q8_0 *) row + d0/QK8_0;
        const uint16_t * qs = (const uint16_t *) (b->qs + d0 % QK8_0); // 2-byte aligned in the 34-byte block
        q[0] = uint32_t(qs[0]) | (uint32_t(qs[1]) << 16);
        q[1] = uint32_t(qs[2]) | (uint32_t(qs[3]) << 16);
        d    = b->d;
    }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const float scale = __half2float(d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = scale*float(int8_t((q[j/4] >> (8*(j % 4))) & 0xFF));
        }
    }
};

template<> struct fattn_banded_kv8<GGML_TYPE_Q4_0> {
    uint32_t q[2];
    half     d;
    __device__ __forceinline__ void load(const char * row, const int d0) {
        // elements 0..15 are the low nibbles of qs[0..15], 16..31 the high ones
        const block_q4_0 * b = (const block_q4_0 *) row + d0/QK4_0;
        const uint16_t * qs = (const uint16_t *) (b->qs + d0 % (QK4_0/2));
        const int shift = d0 % QK4_0 >= QK4_0/2 ? 4 : 0;
        q[0] = ((uint32_t(qs[0]) | (uint32_t(qs[1]) << 16)) >> shift) & 0x0F0F0F0Fu;
        q[1] = ((uint32_t(qs[2]) | (uint32_t(qs[3]) << 16)) >> shift) & 0x0F0F0F0Fu;
        d    = b->d;
    }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const float scale = __half2float(d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = scale*float(int((q[j/4] >> (8*(j % 4))) & 0xFF) - 8);
        }
    }
};

// 8 nibbles of a q4_x / q5_x block (elements e0..e0+7 of 32, e0 % 8 == 0) as bytes, 2-byte loads
static __device__ __forceinline__ void fattn_banded_nib8(const uint8_t * qs, const int e0, uint32_t (&q)[2]) {
    const uint16_t * p = (const uint16_t *) (qs + e0 % 16);
    const int shift = e0 >= 16 ? 4 : 0;
    q[0] = ((uint32_t(p[0]) | (uint32_t(p[1]) << 16)) >> shift) & 0x0F0F0F0Fu;
    q[1] = ((uint32_t(p[2]) | (uint32_t(p[3]) << 16)) >> shift) & 0x0F0F0F0Fu;
}

// bits e0..e0+7 of the 32 high bits as bit 4 of each byte
static __device__ __forceinline__ void fattn_banded_qh8(const uint8_t * qh, const int e0, uint32_t (&q)[2]) {
    const uint32_t h = qh[e0/8];
    // bit i of a nibble to bit 0 of byte i: the shifted copies do not overlap, so no carries
    q[0] |= (((h & 0xF) * 0x00204081u) & 0x01010101u) << 4;
    q[1] |= (((h >> 4)  * 0x00204081u) & 0x01010101u) << 4;
}

template<> struct fattn_banded_kv8<GGML_TYPE_Q4_1> {
    uint32_t q[2];
    half     d, m;
    __device__ __forceinline__ void load(const char * row, const int d0) {
        const block_q4_1 * b = (const block_q4_1 *) row + d0/QK4_1;
        fattn_banded_nib8(b->qs, d0 % QK4_1, q);
        const half * dm = (const half *) &b->dm; // 2-byte loads: rows are only 2-byte aligned
        d = dm[0];
        m = dm[1];
    }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const float fd = __half2float(d);
        const float fm = __half2float(m);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = fd*float((q[j/4] >> (8*(j % 4))) & 0xFF) + fm;
        }
    }
};

template<> struct fattn_banded_kv8<GGML_TYPE_Q5_0> {
    uint32_t q[2];
    half     d;
    __device__ __forceinline__ void load(const char * row, const int d0) {
        const block_q5_0 * b = (const block_q5_0 *) row + d0/QK5_0;
        fattn_banded_nib8(b->qs, d0 % QK5_0, q);
        fattn_banded_qh8(b->qh, d0 % QK5_0, q);
        d = b->d;
    }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const float scale = __half2float(d);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = scale*float(int((q[j/4] >> (8*(j % 4))) & 0xFF) - 16);
        }
    }
};

template<> struct fattn_banded_kv8<GGML_TYPE_Q5_1> {
    uint32_t q[2];
    half     d, m;
    __device__ __forceinline__ void load(const char * row, const int d0) {
        const block_q5_1 * b = (const block_q5_1 *) row + d0/QK5_1;
        fattn_banded_nib8(b->qs, d0 % QK5_1, q);
        fattn_banded_qh8(b->qh, d0 % QK5_1, q);
        const half * dm = (const half *) &b->dm;
        d = dm[0];
        m = dm[1];
    }
    __device__ __forceinline__ void get(float (&x)[8]) const {
        const float fd = __half2float(d);
        const float fm = __half2float(m);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            x[j] = fd*float((q[j/4] >> (8*(j % 4))) & 0xFF) + fm;
        }
    }
};

template<int D, int NR, ggml_type type_k, ggml_type type_v>
__launch_bounds__(fattn_banded_vec_threads)
static __global__ void flash_attn_ext_banded_vec(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const char * __restrict__ rel,
        const float * __restrict__ sinks,
        float      * __restrict__ dst,
        float      * __restrict__ part,
        const float scale,
        const int type_rel,
        const int n_q,
        const int64_t n_kv,
        const int n_head_q,
        const int n_head_kv,
        const int64_t n_chunks,
        const int64_t chunk,
        const int64_t rel_extent,
        const int64_t mask_ne2,
        const int64_t mask_ne3,
        const uint64_t q_nb1, const uint64_t q_nb2, const uint64_t q_nb3,
        const uint64_t k_nb1, const uint64_t k_nb2, const uint64_t k_nb3,
        const uint64_t v_nb1, const uint64_t v_nb2, const uint64_t v_nb3,
        const uint64_t m_nb1, const uint64_t m_nb2, const uint64_t m_nb3,
        const uint64_t r_nb0, const uint64_t r_nb1, const uint64_t r_nb2, const uint64_t r_nb3,
        const int64_t rel_ne3,
        const int64_t sinks_s1,
        const int64_t sinks_s2,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos_d,
        const int64_t kpos_s1,
        const int64_t qpos_s1,
        const int causal) {
    constexpr int NT  = fattn_banded_vec_threads;
    constexpr int E   = 8;                 // dimensions per lane
    constexpr int T   = D / E;             // lanes per cell
    constexpr int NG  = NT / T;            // cell groups
    constexpr int CPG = fattn_banded_vec_cpg;
    constexpr int TC  = NG*CPG;            // cells per tile
    static_assert(WARP_SIZE % T == 0 && T >= NR, "banded vec tiling");

    __shared__ float S[TC][NR + 1];        // scores, then probabilities
    __shared__ float alpha_s[NR];
    __shared__ float max_s[NR];
    __shared__ float sum_s[NR];
    __shared__ float o_s[NG][NR][D + 1];

    const int tid = threadIdx.x;
    const int t   = tid % T;
    const int grp = tid / T;

    // the KV heads of a chunk are neighbours in the launch order: they share the DRAM pages of the cells
    const int     ih_kv = blockIdx.x;
    const int64_t ic    = blockIdx.y;
    const int     G     = n_head_q / n_head_kv;
    const int     n_rb  = (n_q*G + NR - 1)/NR;
    const int     rb    = blockIdx.z % n_rb;
    const int     ib    = blockIdx.z / n_rb;
    const int     n_rows = min(NR, n_q*G - rb*NR);

    // row r: (query, Q head) number rb*NR + r of this KV head; rows past n_rows repeat the last one
    int iq_r[NR];
    int ih_r[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int f = rb*NR + min(r, n_rows - 1);
        iq_r[r] = f / G;
        ih_r[r] = ih_kv*G + f % G;
    }

    float qv[NR][E];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const float * qr = (const float *) (q + uint64_t(iq_r[r])*q_nb1 + uint64_t(ih_r[r])*q_nb2 + uint64_t(ib)*q_nb3) + t*E;
        const float4 x0 = *(const float4 *) (qr + 0);
        const float4 x1 = *(const float4 *) (qr + 4);
        qv[r][0] = x0.x*scale; qv[r][1] = x0.y*scale; qv[r][2] = x0.z*scale; qv[r][3] = x0.w*scale;
        qv[r][4] = x1.x*scale; qv[r][5] = x1.y*scale; qv[r][6] = x1.z*scale; qv[r][7] = x1.w*scale;
    }
    // lane t < NR of a group reads the mask and the bias of row t
    const int rt = min(t, NR - 1);
    const char * m_row = mask ? mask + uint64_t(iq_r[rt])*m_nb1 + uint64_t(ih_r[rt] % mask_ne2)*m_nb2 + uint64_t(ib % mask_ne3)*m_nb3 : nullptr;
    const char * r_row = rel + uint64_t(ih_r[rt])*r_nb1 + uint64_t(iq_r[rt])*r_nb2 + uint64_t(ib % rel_ne3)*r_nb3;
    // the cell the row's query is aligned to, or its position when the cells are not the positions
    const int32_t * kp = k_pos ? k_pos + ib*kpos_s1 : nullptr;
    const int64_t q_pos = kp ? int64_t(q_pos_d[ib*qpos_s1 + iq_r[rt]]) : iq_r[rt] + (n_kv - n_q);

    if (tid < NR) {
        max_s[tid] = -INFINITY;
        sum_s[tid] = 0.0f;
    }
    float o[NR][E];
#pragma unroll
    for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            o[r][e] = 0.0f;
        }
    }
    __syncthreads();

    const int64_t c_begin = ic*chunk;
    const int64_t c_end   = min(c_begin + chunk, n_kv);
    const char * k_base = k + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
    const char * v_base = v + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;

    for (int64_t c0 = c_begin; c0 < c_end; c0 += TC) {
        // every load of the tile first: group grp takes cells grp, grp + NG, ...
        fattn_banded_kv8<type_k> kr[CPG];
        fattn_banded_kv8<type_v> vr[CPG];
        float bm[CPG]; // lane t < NR: mask + bias of row t, -inf when masked
#pragma unroll
        for (int j = 0; j < CPG; ++j) {
            const int64_t c  = c0 + grp + j*NG;
            const int64_t cc = c < c_end ? c : c_end - 1;
            kr[j].load(k_base + uint64_t(cc)*k_nb1, t*E);
            vr[j].load(v_base + uint64_t(cc)*v_nb1, t*E);
            bm[j] = -INFINITY;
            if (t < NR && c < c_end) {
                bm[j] = m_row ? __half2float(*(const half *) (m_row + uint64_t(c)*sizeof(half))) : 0.0f;
                const int64_t dist = kp ? (kp[c] < 0 ? -1 : q_pos - kp[c]) : q_pos - c;
                if (causal && dist < 0) {
                    bm[j] = -INFINITY;
                }
                if (bm[j] != -INFINITY && dist >= 0 && dist < rel_extent) {
                    bm[j] += fattn_banded_load(r_row + uint64_t(dist)*r_nb0, type_rel);
                }
            }
        }
        // K.Q
#pragma unroll
        for (int j = 0; j < CPG; ++j) {
            float kx[E];
            kr[j].get(kx);
            float sr[NR];
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                float s = 0.0f;
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    s += qv[r][e]*kx[e];
                }
                sr[r] = warp_reduce_sum<T>(s);
            }
            // lane t writes row t
            float s_t = sr[0];
#pragma unroll
            for (int r = 1; r < NR; ++r) {
                s_t = t == r ? sr[r] : s_t;
            }
            if (t < NR) {
                S[grp + j*NG][t] = bm[j] == -INFINITY ? -INFINITY : s_t + bm[j];
            }
        }
        __syncthreads();

        // online softmax: warp w takes rows w, w + 4, ...
        {
            const int lane = tid % WARP_SIZE;
            const int warp = tid / WARP_SIZE;
            for (int r = warp; r < n_rows; r += NT/WARP_SIZE) {
                float m = -INFINITY;
#pragma unroll
                for (int j = lane; j < TC; j += WARP_SIZE) {
                    m = fmaxf(m, S[j][r]);
                }
                m = warp_reduce_max(m);
                const float m_old = max_s[r];
                const float m_new = fmaxf(m_old, m);
                const float m_sub = m_new == -INFINITY ? 0.0f : m_new;
                float sum = 0.0f;
#pragma unroll
                for (int j = lane; j < TC; j += WARP_SIZE) {
                    const float p = expf(S[j][r] - m_sub);
                    S[j][r] = p;
                    sum += p;
                }
                sum = warp_reduce_sum(sum);
                if (lane == 0) {
                    const float a = expf(m_old - m_sub);
                    alpha_s[r] = a;
                    sum_s[r]   = sum_s[r]*a + sum;
                    max_s[r]   = m_new;
                }
            }
        }
        __syncthreads();

        // P.V of the group's cells
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float a = alpha_s[r];
#pragma unroll
            for (int e = 0; e < E; ++e) {
                o[r][e] *= a;
            }
        }
#pragma unroll
        for (int j = 0; j < CPG; ++j) {
            float vx[E];
            vr[j].get(vx);
#pragma unroll
            for (int r = 0; r < NR; ++r) {
                const float p = S[grp + j*NG][r];
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    o[r][e] += p*vx[e];
                }
            }
        }
        __syncthreads();
    }

    // merge the groups
#pragma unroll
    for (int r = 0; r < NR; ++r) {
#pragma unroll
        for (int e = 0; e < E; ++e) {
            o_s[grp][r][t*E + e] = o[r][e];
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n_rows*D; idx += NT) {
        const int r = idx / D;
        const int d = idx % D;
        float acc = 0.0f;
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            acc += o_s[g][r][d];
        }
        const float m  = max_s[r];
        float       l  = sum_s[r];
        const int f   = rb*NR + r;
        const int iqr = f / G;
        const int ihr = ih_kv*G + f % G;
        const int64_t row = (int64_t(ib)*n_q + iqr)*n_head_q + ihr;
        if (n_chunks == 1) {
            float fs = 1.0f;
            if (sinks) {
                const float sink  = sinks[ihr + iqr*sinks_s1 + ib*sinks_s2];
                const float m_new = fmaxf(m, sink);
                fs = expf(m - m_new);
                l  = l*fs + expf(sink - m_new);
            }
            dst[row*D + d] = l == 0.0f ? 0.0f : acc*fs/l;
        } else {
            float * p = part + (row*n_chunks + ic)*(D + 2);
            if (d == 0) {
                p[0] = m;
                p[1] = l;
            }
            p[2 + d] = acc;
        }
    }
}

// merge of the chunk partials: a block per row, the chunk weights in parallel, then the dimensions
template<int D>
static __global__ void flash_attn_ext_banded_vec_merge(
        const float * __restrict__ part,
        float       * __restrict__ dst,
        const int64_t n_head_q,
        const int64_t n_q,
        const int     n_chunks,
        const float * __restrict__ sinks,
        const int64_t sinks_s1,
        const int64_t sinks_s2) {
    constexpr int NT = 128;
    extern __shared__ float w_s[]; // [n_chunks]
    __shared__ float red[NT/WARP_SIZE];
    __shared__ float m_all;
    __shared__ float l_all;
    const int64_t row = blockIdx.x;
    const int     tid = threadIdx.x;
    const float * p   = part + row*n_chunks*(D + 2);

    float m = -INFINITY;
    for (int c = tid; c < n_chunks; c += NT) {
        m = fmaxf(m, p[c*(D + 2)]);
    }
    m = warp_reduce_max(m);
    if (tid % WARP_SIZE == 0) {
        red[tid / WARP_SIZE] = m;
    }
    __syncthreads();
    if (tid == 0) {
        float mm = red[0];
        for (int w = 1; w < NT/WARP_SIZE; ++w) {
            mm = fmaxf(mm, red[w]);
        }
        m_all = mm;
    }
    __syncthreads();
    m = m_all;
    float l = 0.0f;
    for (int c = tid; c < n_chunks; c += NT) {
        const float mc = p[c*(D + 2)];
        const float f  = mc == -INFINITY || m == -INFINITY ? 0.0f : expf(mc - m);
        w_s[c] = f;
        l += f*p[c*(D + 2) + 1];
    }
    l = warp_reduce_sum(l);
    __syncthreads();
    if (tid % WARP_SIZE == 0) {
        red[tid / WARP_SIZE] = l;
    }
    __syncthreads();
    if (tid == 0) {
        float ll = 0.0f;
        for (int w = 0; w < NT/WARP_SIZE; ++w) {
            ll += red[w];
        }
        l_all = ll;
    }
    __syncthreads();
    l = l_all;

    const int64_t ih = row % n_head_q;
    const int64_t iq = (row / n_head_q) % n_q;
    const int64_t ib = row / (n_head_q*n_q);
    float f = 1.0f;
    if (sinks) {
        const float sink  = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
        const float m_new = fmaxf(m, sink);
        f = m == -INFINITY ? 0.0f : expf(m - m_new);
        l = l*f + expf(sink - m_new);
    }
    const float inv = l == 0.0f ? 0.0f : f/l;
    for (int d = tid; d < D; d += NT) {
        float acc = 0.0f;
#pragma unroll 8
        for (int c = 0; c < n_chunks; ++c) {
            acc += w_s[c]*p[c*(D + 2) + 2 + d];
        }
        dst[row*D + d] = acc*inv;
    }
}

// the decode kernel reads K and V once per block of rows: past 8 queries the MMA kernel is faster
// K/V the MMA kernel reads directly (F16 / BF16, matching q8_0 / q4_0 tiles) take it from 9 queries on,
// the others stay here up to 32 queries instead of an F16 copy of the whole cache
static constexpr int64_t fattn_banded_vec_max_q      = 8;
static constexpr int64_t fattn_banded_vec_max_q_conv = 32;

static bool fattn_banded_vec_mma_direct(const ggml_type tk, const ggml_type tv) {
    return tk == tv && (tk == GGML_TYPE_F16 || tk == GGML_TYPE_BF16 || tk == GGML_TYPE_Q8_0 || tk == GGML_TYPE_Q4_0);
}

// K/V type pairs with a decode kernel instance: q4_1 / q5_x and the mixed pairs only for D = 128
static bool fattn_banded_vec_pair(const ggml_type tk, const ggml_type tv, const int64_t D) {
    if (fattn_banded_vec_mma_direct(tk, tv)) {
        return true;
    }
    if (D != 128) {
        return false;
    }
    if (tk == tv) {
        return tk == GGML_TYPE_Q4_1 || tk == GGML_TYPE_Q5_0 || tk == GGML_TYPE_Q5_1;
    }
    return tk == GGML_TYPE_Q8_0 && (tv == GGML_TYPE_Q4_0 || tv == GGML_TYPE_F16);
}

static bool fattn_banded_vec_usable(const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    if (!fattn_banded_vec_pair(k->type, v->type, q->ne[0])) {
        return false;
    }
    if (q->ne[1] > (fattn_banded_vec_mma_direct(k->type, v->type) ? fattn_banded_vec_max_q : fattn_banded_vec_max_q_conv)) {
        return false;
    }
    // 16-byte F16 / BF16 and 2-byte quantized loads, 16-byte Q loads
    for (const ggml_tensor * t : {k, v}) {
        const size_t align = t->type == GGML_TYPE_F16 || t->type == GGML_TYPE_BF16 ? 16 : 2;
        if (t->nb[1] % align != 0 || t->nb[2] % align != 0 || t->nb[3] % align != 0 || (uintptr_t) t->data % align != 0) {
            return false;
        }
    }
    return q->nb[1] % 16 == 0 && q->nb[2] % 16 == 0 && q->nb[3] % 16 == 0 && (uintptr_t) q->data % 16 == 0;
}

static void fattn_banded_vec_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale, const int64_t rel_extent) {
    // ggml_flash_attn_ext_banded_set_causal: no mask, the kernels mask the cells after each query
    const int causal = ggml_flash_attn_ext_banded_is_causal(dst);
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * m     = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];
    cudaStream_t stream = ctx.stream();

    const int64_t D    = q->ne[0];
    const int64_t n_q  = q->ne[1];
    const int64_t n_kv = k->ne[1];
    const int64_t G    = q->ne[2]/k->ne[2];
    // rows per block: the GQA groups of as many queries as fit 8 rows (one query: its group alone), else 8
    // (Q heads of a KV head in steps of 8)
    const int     NR   = G > 8 ? 8 : int(G*std::min<int64_t>(n_q, 8/G));
    const int64_t n_rb = (n_q*G + NR - 1)/NR;

    // split the cells so that the grid fills the GPU
    constexpr int64_t tile      = fattn_banded_vec_threads/16*fattn_banded_vec_cpg; // D = 128
    constexpr int64_t min_chunk = 4*tile;
    const int64_t base_blocks = k->ne[2]*n_rb*q->ne[3];
    // one wave of resident blocks, 3 per SM
    const int64_t target      = int64_t(ggml_cuda_info().devices[ctx.device].nsm)*3;
    int64_t n_chunks = std::max<int64_t>(1, std::min<int64_t>(target/base_blocks, (n_kv + min_chunk - 1)/min_chunk));
    n_chunks = std::min<int64_t>(n_chunks, 4096); // merge weights in shared memory
    const int64_t chunk = GGML_PAD((n_kv + n_chunks - 1)/n_chunks, tile);
    n_chunks = (n_kv + chunk - 1)/chunk;

    const int64_t n_rows = n_q*q->ne[2]*q->ne[3];
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (n_chunks > 1) {
        part.alloc(n_rows*n_chunks*(D + 2));
    }
    const float * sinks_d  = sinks ? (const float *) sinks->data : nullptr;
    const int64_t sinks_s1 = sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0;
    const int64_t sinks_s2 = sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0;

    const ggml_tensor * k_pos = dst->src[7];
    const ggml_tensor * q_pos = dst->src[8];
    const int32_t * k_pos_d = k_pos ? (const int32_t *) k_pos->data : nullptr;
    const int32_t * q_pos_d = q_pos ? (const int32_t *) q_pos->data : nullptr;
    const int64_t kpos_s1 = k_pos && k_pos->ne[1] > 1 ? k_pos->ne[0] : 0;
    const int64_t qpos_s1 = q_pos && q_pos->ne[1] > 1 ? q_pos->ne[0] : 0;

    const dim3 blocks(k->ne[2], n_chunks, n_rb*q->ne[3]);
#define LAUNCH_BANDED_VEC_T(DD, NRR, TK, TV) \
    flash_attn_ext_banded_vec<DD, NRR, TK, TV><<<blocks, fattn_banded_vec_threads, 0, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, m ? (const char *) m->data : nullptr, \
        (const char *) rel->data, n_chunks == 1 ? sinks_d : nullptr, (float *) dst->data, part.get(), \
        scale, rel->type, int(n_q), n_kv, int(q->ne[2]), int(k->ne[2]), n_chunks, chunk, rel_extent, \
        m ? m->ne[2] : 1, m ? m->ne[3] : 1, q->nb[1], q->nb[2], q->nb[3], k->nb[1], k->nb[2], k->nb[3], \
        v->nb[1], v->nb[2], v->nb[3], m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], sinks_s1, sinks_s2, \
        k_pos_d, q_pos_d, kpos_s1, qpos_s1, causal)
#define LAUNCH_BANDED_VEC_NR(DD, TK, TV) \
    switch (NR) { \
        case 1:  LAUNCH_BANDED_VEC_T(DD, 1, TK, TV); break; \
        case 2:  LAUNCH_BANDED_VEC_T(DD, 2, TK, TV); break; \
        case 3:  LAUNCH_BANDED_VEC_T(DD, 3, TK, TV); break; \
        case 4:  LAUNCH_BANDED_VEC_T(DD, 4, TK, TV); break; \
        case 5:  LAUNCH_BANDED_VEC_T(DD, 5, TK, TV); break; \
        case 6:  LAUNCH_BANDED_VEC_T(DD, 6, TK, TV); break; \
        case 7:  LAUNCH_BANDED_VEC_T(DD, 7, TK, TV); break; \
        default: LAUNCH_BANDED_VEC_T(DD, 8, TK, TV); break; \
    }
#define LAUNCH_BANDED_VEC_SAME(DD) \
    switch (k->type) { \
        case GGML_TYPE_F16:  LAUNCH_BANDED_VEC_NR(DD, GGML_TYPE_F16,  GGML_TYPE_F16);  break; \
        case GGML_TYPE_BF16: LAUNCH_BANDED_VEC_NR(DD, GGML_TYPE_BF16, GGML_TYPE_BF16); break; \
        case GGML_TYPE_Q8_0: LAUNCH_BANDED_VEC_NR(DD, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0); break; \
        default:             LAUNCH_BANDED_VEC_NR(DD, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0); break; \
    }
    if (D == 64) {
        LAUNCH_BANDED_VEC_SAME(64);
    } else if (fattn_banded_vec_mma_direct(k->type, v->type)) {
        LAUNCH_BANDED_VEC_SAME(128);
    } else if (k->type == v->type) {
        switch (k->type) {
            case GGML_TYPE_Q4_1: LAUNCH_BANDED_VEC_NR(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1); break;
            case GGML_TYPE_Q5_0: LAUNCH_BANDED_VEC_NR(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0); break;
            default:             LAUNCH_BANDED_VEC_NR(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1); break;
        }
    } else if (v->type == GGML_TYPE_Q4_0) {
        LAUNCH_BANDED_VEC_NR(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0);
    } else {
        LAUNCH_BANDED_VEC_NR(128, GGML_TYPE_Q8_0, GGML_TYPE_F16);
    }
#undef LAUNCH_BANDED_VEC_NR
#undef LAUNCH_BANDED_VEC_SAME
#undef LAUNCH_BANDED_VEC_T
    CUDA_CHECK(cudaGetLastError());

    if (n_chunks > 1) {
        const size_t smem = n_chunks*sizeof(float);
        if (D == 64) {
            flash_attn_ext_banded_vec_merge<64><<<n_rows, 128, smem, stream>>>(
                part.get(), (float *) dst->data, q->ne[2], n_q, int(n_chunks), sinks_d, sinks_s1, sinks_s2);
        } else {
            flash_attn_ext_banded_vec_merge<128><<<n_rows, 128, smem, stream>>>(
                part.get(), (float *) dst->data, q->ne[2], n_q, int(n_chunks), sinks_d, sinks_s1, sinks_s2);
        }
    }
}

#if defined(TURING_MMA_AVAILABLE)
#define FATTN_BANDED_MMA_AVAILABLE
#endif

static constexpr int fattn_banded_mma_bk = 64;

#ifdef FATTN_BANDED_MMA_AVAILABLE
static __device__ __forceinline__ void fattn_banded_mma_16816(
        float (&c)[4], const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
        const uint32_t b0, const uint32_t b1) {
#ifdef AMPERE_MMA_AVAILABLE
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a2), "r"(a3), "r"(b1));
#endif // AMPERE_MMA_AVAILABLE
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ void fattn_banded_ldmatrix_x4_trans(uint32_t (&r)[4], const half * p) {
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

static __device__ __forceinline__ uint32_t fattn_banded_pack_h2(const float lo, const float hi) {
    const half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

// q8_0 / q4_0 K/V rows are dequantized to F16 per tile, so the cache is never copied to F16 as a whole
template<int D, int NW, ggml_type type_kv>
static __device__ __forceinline__ void fattn_banded_mma_load_tile_q(
        half * __restrict__ tile, const char * __restrict__ src, const uint64_t nb1, const int64_t kv0, const int64_t n_kv) {
    static_assert(type_kv == GGML_TYPE_Q8_0 || type_kv == GGML_TYPE_Q4_0, "unsupported banded K/V tile type");
    constexpr int stride   = D + 8;
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
#pragma unroll
    for (int c0 = 0; c0 < fattn_banded_mma_bk*chunks; c0 += nthreads) {
        const int c   = c0 + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        half2 h[4];
        if (kv0 + row < n_kv) {
            const char * s = src + uint64_t(kv0 + row)*nb1;
            if constexpr (type_kv == GGML_TYPE_Q8_0) {
                const block_q8_0 * b = (const block_q8_0 *) s + col/QK8_0;
                const float d = __half2float(b->d);
                const int   o = col % QK8_0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    h[i] = __floats2half2_rn(float(b->qs[o + 2*i + 0])*d, float(b->qs[o + 2*i + 1])*d);
                }
            } else {
                const block_q4_0 * b = (const block_q4_0 *) s + col/QK4_0;
                const float d     = __half2float(b->d);
                const int   o     = col % QK4_0;
                const int   shift = o < QK4_0/2 ? 0 : 4;
                const int   j0    = o % (QK4_0/2);
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int q0 = ((b->qs[j0 + 2*i + 0] >> shift) & 0x0F) - 8;
                    const int q1 = ((b->qs[j0 + 2*i + 1] >> shift) & 0x0F) - 8;
                    h[i] = __floats2half2_rn(float(q0)*d, float(q1)*d);
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                h[i] = make_half2(0.0f, 0.0f);
            }
        }
        *(int4 *) (tile + row*stride + col) = *(const int4 *) h;
    }
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;"); // keeps the wait_group counts of the F16 pipeline
#endif // CP_ASYNC_AVAILABLE
}

// The quantized tiles of the main loop in two steps, so that the global loads overlap the MMAs as cp.async does for F16:
// fattn_banded_mma_issue_tile_q only loads the quants and scales of this thread's chunks into registers, fattn_banded_mma_store_tile_q later
// dequantizes them into the F16 tile with the values fattn_banded_mma_load_tile_q gives, bit for bit.
// Rows past n_kv get zero quants and a zero scale, which store as +0 like the zero fill of the one-step load.
template<int D, int NW>
struct fattn_banded_mma_qregs {
    static constexpr int n = fattn_banded_mma_bk*(D/8)/(NW*WARP_SIZE);
    static_assert(n*NW*WARP_SIZE == fattn_banded_mma_bk*(D/8), "the tile chunks must split evenly over the threads");
    uint2  q[n];
    half   d[n];
};

template<int D, int NW, ggml_type type_kv>
static __device__ __forceinline__ void fattn_banded_mma_issue_tile_q(
        fattn_banded_mma_qregs<D, NW> & r, const char * __restrict__ src, const uint64_t nb1, const int64_t kv0, const int64_t n_kv) {
    static_assert(type_kv == GGML_TYPE_Q8_0 || type_kv == GGML_TYPE_Q4_0, "unsupported banded K/V tile type");
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
#pragma unroll
    for (int i = 0; i < fattn_banded_mma_qregs<D, NW>::n; ++i) {
        const int c   = i*nthreads + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        r.q[i] = make_uint2(0, 0);
        r.d[i] = __float2half(0.0f);
        if (kv0 + row < n_kv) {
            const char * s = src + uint64_t(kv0 + row)*nb1;
            // the quants of a block start 2 bytes in: 16-bit loads
            const uint16_t * q16;
            if constexpr (type_kv == GGML_TYPE_Q8_0) {
                const block_q8_0 * b = (const block_q8_0 *) s + col/QK8_0;
                r.d[i] = b->d;
                q16 = (const uint16_t *) (b->qs + col % QK8_0);
            } else {
                const block_q4_0 * b = (const block_q4_0 *) s + col/QK4_0;
                r.d[i] = b->d;
                q16 = (const uint16_t *) (b->qs + (col % QK4_0) % (QK4_0/2));
            }
            r.q[i].x = q16[0] | ((uint32_t) q16[1] << 16);
            r.q[i].y = q16[2] | ((uint32_t) q16[3] << 16);
        }
    }
}

template<int D, int NW, ggml_type type_kv>
static __device__ __forceinline__ void fattn_banded_mma_store_tile_q(half * __restrict__ tile, const fattn_banded_mma_qregs<D, NW> & r) {
    constexpr int stride   = D + 8;
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
    // a quant u in [0, 255] as the half 1024 + u (exponent byte 0x64), minus 1152 (q8_0, u = q + 128) or 1032 (q4_0, u = q + 8), is exact.
    // q*d is exact in F32, so one half multiply rounds it as the F32 product rounded to half does.
    const half2 off = __float2half2_rn(type_kv == GGML_TYPE_Q8_0 ? 1152.0f : 1032.0f);
#pragma unroll
    for (int i = 0; i < fattn_banded_mma_qregs<D, NW>::n; ++i) {
        const int c   = i*nthreads + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        const half2 d2 = __half2half2(r.d[i]);
        const uint32_t w[2] = { r.q[i].x, r.q[i].y };
        half2 h[4];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            uint32_t u;
            if constexpr (type_kv == GGML_TYPE_Q8_0) {
                u = w[k] ^ 0x80808080u;
            } else {
                u = (w[k] >> (col % QK4_0 < QK4_0/2 ? 0 : 4)) & 0x0F0F0F0Fu;
            }
            const uint32_t lo = __byte_perm(u, 0x64646464u, 0x4140);
            const uint32_t hi = __byte_perm(u, 0x64646464u, 0x4342);
            h[2*k + 0] = __hmul2(__hsub2(*(const half2 *) &lo, off), d2);
            h[2*k + 1] = __hmul2(__hsub2(*(const half2 *) &hi, off), d2);
        }
        *(int4 *) (tile + row*stride + col) = *(const int4 *) h;
    }
}

template<int D, int NW, ggml_type type_kv = GGML_TYPE_F16>
static __device__ __forceinline__ void fattn_banded_mma_load_tile(
        half * __restrict__ tile, const char * __restrict__ src, const uint64_t nb1, const int64_t kv0, const int64_t n_kv) {
    if constexpr (type_kv != GGML_TYPE_F16) {
        fattn_banded_mma_load_tile_q<D, NW, type_kv>(tile, src, nb1, kv0, n_kv);
        return;
    }
    constexpr int stride   = D + 8;
    constexpr int chunks   = D/8;
    constexpr int nthreads = NW*WARP_SIZE;
#pragma unroll
    for (int c0 = 0; c0 < fattn_banded_mma_bk*chunks; c0 += nthreads) {
        const int c   = c0 + threadIdx.x;
        const int row = c / chunks;
        const int col = (c % chunks)*8;
        half * dst = tile + row*stride + col;
        if (kv0 + row < n_kv) {
            const char * s = src + uint64_t(kv0 + row)*nb1 + uint64_t(col)*sizeof(half);
#ifdef CP_ASYNC_AVAILABLE
            const uint32_t d = (uint32_t) __cvta_generic_to_shared(dst);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" : : "r"(d), "l"(s));
#else
            *(int4 *) dst = *(const int4 *) s;
#endif // CP_ASYNC_AVAILABLE
        } else {
            *(int4 *) dst = make_int4(0, 0, 0, 0);
        }
    }
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.commit_group;");
#endif // CP_ASYNC_AVAILABLE
}

template<int n>
static __device__ __forceinline__ void fattn_banded_mma_wait() {
#ifdef CP_ASYNC_AVAILABLE
    asm volatile("cp.async.wait_group %0;" : : "n"(n));
#endif // CP_ASYNC_AVAILABLE
}
#endif // FATTN_BANDED_MMA_AVAILABLE

static __global__ void flash_attn_ext_banded_mma_tile_flags(
        const char * __restrict__ mask,
        uint8_t    * __restrict__ flags,
        int64_t n_q,
        int64_t n_kv,
        int bq,
        int64_t mask_ne2,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3) {
    const int64_t kt   = blockIdx.x;
    const int64_t qt   = blockIdx.y;
    const int64_t z    = blockIdx.z;
    const int64_t z2   = z % mask_ne2;
    const int64_t z3   = z / mask_ne2;
    const char  * mrow = mask + z2*m_nb2 + z3*m_nb3;

    bool any_visible = false;
    bool any_nonzero = false;
    for (int e = threadIdx.x; e < bq*fattn_banded_mma_bk; e += blockDim.x) {
        const int64_t iq = qt*bq + e / fattn_banded_mma_bk;
        const int64_t ik = kt*fattn_banded_mma_bk + e % fattn_banded_mma_bk;
        if (iq < n_q && ik < n_kv) {
            const float m = __half2float(*(const half *) (mrow + uint64_t(iq)*m_nb1 + uint64_t(ik)*sizeof(half)));
            any_visible |= m != -INFINITY;
            any_nonzero |= m != 0.0f;
        }
    }
    any_visible = __syncthreads_or(any_visible);
    any_nonzero = __syncthreads_or(any_nonzero);
    if (threadIdx.x == 0) {
        const int64_t n_kt = gridDim.x;
        const int64_t n_qt = gridDim.y;
        flags[(z*n_qt + qt)*n_kt + kt] = !any_visible ? 0 : (any_nonzero ? 2 : 1);
    }
}

template<int D, int NW, ggml_type type_kv>
__launch_bounds__(NW*WARP_SIZE, 1)
static __global__ void flash_attn_ext_banded_mma(
        const char * __restrict__ q,
        const char * __restrict__ k,
        const char * __restrict__ v,
        const char * __restrict__ mask,
        const uint8_t * __restrict__ flags,
        const char * __restrict__ rel,
        const float * __restrict__ sinks,
        float      * __restrict__ dst,
        float      * __restrict__ part,
        float scale,
        int type_rel,
        int64_t n_q,
        int64_t n_kv,
        int64_t n_head_q,
        int64_t n_head_kv,
        int64_t n_chunks,
        int64_t tiles_per_chunk,
        int64_t rel_extent,
        int64_t mask_ne2,
        int64_t mask_ne3,
        uint64_t q_nb1,
        uint64_t q_nb2,
        uint64_t q_nb3,
        uint64_t k_nb1,
        uint64_t k_nb2,
        uint64_t k_nb3,
        uint64_t v_nb1,
        uint64_t v_nb2,
        uint64_t v_nb3,
        uint64_t m_nb1,
        uint64_t m_nb2,
        uint64_t m_nb3,
        uint64_t r_nb0,
        uint64_t r_nb1,
        uint64_t r_nb2,
        uint64_t r_nb3,
        int64_t rel_ne3,
        int64_t sinks_s1,
        int64_t sinks_s2,
        const int32_t * __restrict__ k_pos,
        const int32_t * __restrict__ q_pos,
        int64_t kpos_s1,
        int64_t qpos_s1,
        const int causal) {
#ifdef FATTN_BANDED_MMA_AVAILABLE
    static_assert(D == 64 || D == 128, "banded FA supports head dimensions 64 and 128");
    constexpr int BK     = fattn_banded_mma_bk;
    constexpr int BQ     = NW*16;
    constexpr int stride = D + 8; // padding keeps ldmatrix bank-conflict free
    constexpr int ND     = D/8;
    constexpr int NK     = D/16;

    __shared__ __align__(16) half K_s[BK*stride];
    __shared__ __align__(16) half V_s[BK*stride];

    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int g    = lane / 4;
    const int t    = lane % 4;

    const int64_t n_qt = (n_q + BQ - 1)/BQ;
    const int64_t qt   = int64_t(blockIdx.x) / n_chunks;
    const int64_t ic   = int64_t(blockIdx.x) % n_chunks;
    const int64_t ih   = blockIdx.y;
    const int64_t ib   = blockIdx.z;
    const int64_t ih_kv = ih / (n_head_q / n_head_kv);

    const int64_t n_kt     = (n_kv + BK - 1)/BK;
    const int64_t kt_begin = ic*tiles_per_chunk;
    const int64_t kt_end   = kt_begin + tiles_per_chunk < n_kt ? kt_begin + tiles_per_chunk : n_kt;

    const int64_t iq_w = qt*BQ + warp*16;
    const int64_t iq_r[2] = {iq_w + g, iq_w + g + 8};
    const int64_t q_offset = n_kv - n_q;

    const char * k_base = k + uint64_t(ih_kv)*k_nb2 + uint64_t(ib)*k_nb3;
    const char * v_base = v + uint64_t(ih_kv)*v_nb2 + uint64_t(ib)*v_nb3;
    const char * m_base = mask ? mask + uint64_t(ih % mask_ne2)*m_nb2 + uint64_t(ib % mask_ne3)*m_nb3 : nullptr;
    const uint8_t * f_base = flags ? flags + ((ih % mask_ne2) + mask_ne2*(ib % mask_ne3))*n_qt*n_kt + qt*n_kt : nullptr;
    const char * r_base = rel + uint64_t(ih)*r_nb1 + uint64_t(ib % rel_ne3)*r_nb3;
    const int32_t * kp = k_pos ? k_pos + ib*kpos_s1 : nullptr;
    const int32_t * qp = q_pos ? q_pos + ib*qpos_s1 : nullptr;

    uint32_t Q_a[NK][4];
#pragma unroll
    for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float lo0 = 0.0f, hi0 = 0.0f, lo1 = 0.0f, hi1 = 0.0f;
            if (iq_r[r] < n_q) {
                const float * qr = (const float *) (q + uint64_t(iq_r[r])*q_nb1 + uint64_t(ih)*q_nb2 + uint64_t(ib)*q_nb3);
                lo0 = qr[kk*16 + 2*t + 0];
                hi0 = qr[kk*16 + 2*t + 1];
                lo1 = qr[kk*16 + 2*t + 8];
                hi1 = qr[kk*16 + 2*t + 9];
            }
            Q_a[kk][r + 0] = fattn_banded_pack_h2(lo0, hi0);
            Q_a[kk][r + 2] = fattn_banded_pack_h2(lo1, hi1);
        }
    }

    float O[ND][4];
#pragma unroll
    for (int n = 0; n < ND; ++n) {
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            O[n][e] = 0.0f;
        }
    }
    float row_max[2] = {-INFINITY, -INFINITY};
    float row_sum[2] = {0.0f, 0.0f};

    auto next_tile = [&](int64_t kt) {
        if (f_base) {
            while (kt < kt_end && f_base[kt] == 0) {
                ++kt;
            }
        }
        return kt;
    };

    int64_t kt = next_tile(kt_begin);
    if (kt < kt_end) {
        fattn_banded_mma_load_tile<D, NW, type_kv>(K_s, k_base, k_nb1, kt*BK, n_kv);
        fattn_banded_mma_load_tile<D, NW, type_kv>(V_s, v_base, v_nb1, kt*BK, n_kv);
    }
    // quantized K/V: the tiles after the first are loaded into registers one step ahead (see fattn_banded_mma_issue_tile_q)
    fattn_banded_mma_qregs<D, NW> kq;
    fattn_banded_mma_qregs<D, NW> vq;
    bool v_pending = false;
    GGML_UNUSED(kq);
    GGML_UNUSED(vq);
    GGML_UNUSED(v_pending);

    while (kt < kt_end) {
        const int64_t kv0  = kt*BK;
        const int     flag = f_base ? f_base[kt] : 1;

        fattn_banded_mma_wait<1>(); // K tile landed, V may still be in flight
        __syncthreads();

        float S[BK/8][4];
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                S[j][e] = 0.0f;
            }
        }
#pragma unroll
        for (int kk = 0; kk < NK; ++kk) {
#pragma unroll
            for (int j = 0; j < BK/8; j += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4(b, K_s + ((j + mi/2)*8 + lane % 8)*stride + kk*16 + (mi % 2)*8);
                fattn_banded_mma_16816(S[j + 0], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[0], b[1]);
                fattn_banded_mma_16816(S[j + 1], Q_a[kk][0], Q_a[kk][1], Q_a[kk][2], Q_a[kk][3], b[2], b[3]);
            }
        }

        const int64_t dist_hi = iq_w + 15 + q_offset - kv0;
        const int64_t dist_lo = iq_w + q_offset - (kv0 + BK - 1);
        // with positions the band is not a cell range: test every element
        const bool in_band = kp || (dist_hi >= 0 && dist_lo < rel_extent);

#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int     r  = e / 2;
                const int64_t ik = kv0 + j*8 + 2*t + (e % 2);
                float s = S[j][e]*scale;
                if (ik >= n_kv) {
                    s = -INFINITY;
                } else if (iq_r[r] < n_q) {
                    if (in_band || causal) {
                        const int64_t dist = fattn_banded_dist(kp, qp, iq_r[r], ik, q_offset);
                        if (in_band && dist >= 0 && dist < rel_extent) {
                            s += fattn_banded_load(r_base + uint64_t(dist)*r_nb0 + uint64_t(iq_r[r])*r_nb2, type_rel);
                        }
                        // causal without a mask: the cells after the query (and the empty ones, with positions)
                        if (causal && dist < 0) {
                            s = -INFINITY;
                        }
                    }
                    if (flag == 2) {
                        s += __half2float(*(const half *) (m_base + uint64_t(iq_r[r])*m_nb1 + uint64_t(ik)*sizeof(half)));
                    }
                }
                S[j][e] = s;
            }
        }

        float alpha[2];
        float msub[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float m = row_max[r];
#pragma unroll
            for (int j = 0; j < BK/8; ++j) {
                m = fmaxf(m, fmaxf(S[j][2*r + 0], S[j][2*r + 1]));
            }
            m = fmaxf(m, __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, m, 1, WARP_SIZE));
            m = fmaxf(m, __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, m, 2, WARP_SIZE));
            msub[r]    = m == -INFINITY ? 0.0f : m;
            alpha[r]   = expf(row_max[r] - msub[r]);
            row_max[r] = m;
        }
        uint32_t P_a[BK/8][2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            row_sum[r] *= alpha[r];
        }
#pragma unroll
        for (int j = 0; j < BK/8; ++j) {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float p0 = expf(S[j][2*r + 0] - msub[r]);
                const float p1 = expf(S[j][2*r + 1] - msub[r]);
                row_sum[r] += p0 + p1;
                P_a[j][r] = fattn_banded_pack_h2(p0, p1);
            }
        }
#pragma unroll
        for (int n = 0; n < ND; ++n) {
            O[n][0] *= alpha[0];
            O[n][1] *= alpha[0];
            O[n][2] *= alpha[1];
            O[n][3] *= alpha[1];
        }

        if constexpr (type_kv != GGML_TYPE_F16) {
            // the V tile of this iteration, loaded during the QK MMAs; the PV MMAs of the previous one are done (barrier after them)
            if (v_pending) {
                fattn_banded_mma_store_tile_q<D, NW, type_kv>(V_s, vq);
            }
        }
        __syncthreads();
        const int64_t kt_next = next_tile(kt + 1);
        if constexpr (type_kv != GGML_TYPE_F16) {
            if (kt_next < kt_end) {
                fattn_banded_mma_issue_tile_q<D, NW, type_kv>(kq, k_base, k_nb1, kt_next*BK, n_kv);
            }
        } else if (kt_next < kt_end) {
            fattn_banded_mma_load_tile<D, NW, type_kv>(K_s, k_base, k_nb1, kt_next*BK, n_kv);
            fattn_banded_mma_wait<1>(); // V tile of this iteration landed
        } else {
            fattn_banded_mma_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int s = 0; s < BK/16; ++s) {
#pragma unroll
            for (int n = 0; n < ND; n += 2) {
                const int mi = lane / 8;
                uint32_t b[4];
                fattn_banded_ldmatrix_x4_trans(b, V_s + (s*16 + (mi % 2)*8 + lane % 8)*stride + (n + mi/2)*8);
                fattn_banded_mma_16816(O[n + 0], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[0], b[1]);
                fattn_banded_mma_16816(O[n + 1], P_a[2*s][0], P_a[2*s][1], P_a[2*s + 1][0], P_a[2*s + 1][1], b[2], b[3]);
            }
        }

        if constexpr (type_kv != GGML_TYPE_F16) {
            // the next K tile, loaded during the PV MMAs (which read only V_s; every warp left the QK MMAs before the barrier above)
            if (kt_next < kt_end) {
                fattn_banded_mma_store_tile_q<D, NW, type_kv>(K_s, kq);
            }
        }
        __syncthreads();
        if constexpr (type_kv != GGML_TYPE_F16) {
            v_pending = kt_next < kt_end;
            if (v_pending) {
                fattn_banded_mma_issue_tile_q<D, NW, type_kv>(vq, v_base, v_nb1, kt_next*BK, n_kv);
            }
        } else if (kt_next < kt_end) {
            fattn_banded_mma_load_tile<D, NW, type_kv>(V_s, v_base, v_nb1, kt_next*BK, n_kv);
        }
        kt = kt_next;
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        row_sum[r] += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, row_sum[r], 1, WARP_SIZE);
        row_sum[r] += __shfl_xor_sync(GGML_CUDA_FULL_THREAD_MASK, row_sum[r], 2, WARP_SIZE);
    }

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int64_t iq = iq_r[r];
        if (iq >= n_q) {
            continue;
        }
        const int64_t row = (ib*n_q + iq)*n_head_q + ih;
        if (n_chunks == 1) {
            float m   = row_max[r];
            float sum = row_sum[r];
            float f   = 1.0f;
            if (sinks) {
                const float sink = sinks[ih + iq*sinks_s1 + ib*sinks_s2];
                const float m_new = fmaxf(m, sink);
                f   = expf(m - m_new);
                sum = sum*f + expf(sink - m_new);
            }
            const float inv = sum == 0.0f ? 0.0f : f/sum;
            float2 * d = (float2 *) (dst + row*D);
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                d[n*4 + t] = make_float2(O[n][2*r + 0]*inv, O[n][2*r + 1]*inv);
            }
        } else {
            float * p = part + (row*n_chunks + ic)*(D + 2);
            if (t == 0) {
                p[0] = row_max[r];
                p[1] = row_sum[r];
            }
#pragma unroll
            for (int n = 0; n < ND; ++n) {
                p[2 + n*8 + 2*t + 0] = O[n][2*r + 0];
                p[2 + n*8 + 2*t + 1] = O[n][2*r + 1];
            }
        }
    }
#else
    GGML_UNUSED_VARS(q, k, v, mask, flags, rel, sinks, dst, part, scale, type_rel, n_q, n_kv, n_head_q, n_head_kv,
        n_chunks, tiles_per_chunk, rel_extent, mask_ne2, mask_ne3, q_nb1, q_nb2, q_nb3, k_nb1, k_nb2, k_nb3,
        v_nb1, v_nb2, v_nb3, m_nb1, m_nb2, m_nb3, r_nb0, r_nb1, r_nb2, r_nb3, rel_ne3, sinks_s1, sinks_s2,
        k_pos, q_pos, kpos_s1, qpos_s1);
    NO_DEVICE_CODE;
#endif // FATTN_BANDED_MMA_AVAILABLE
}

static bool fattn_banded_type_supported(ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16;
}

// quantized K/V: the MMA kernel dequantizes q8_0 / q4_0 tiles itself, other types are converted to F16 first
static bool fattn_banded_kv_f16_convertible(ggml_type type) {
    return ggml_is_quantized(type) && ggml_get_to_fp16_nc_cuda(type) != nullptr;
}

static bool fattn_banded_mma_tile_type(ggml_type type) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_BANDED_NO_QUANT_TILES");
        return env != nullptr && atoi(env) != 0;
    }();
    return !disabled && (type == GGML_TYPE_Q8_0 || type == GGML_TYPE_Q4_0);
}

bool ggml_cuda_flash_attn_ext_banded_supported(int device, const ggml_tensor * dst) {
    GGML_UNUSED(device);
#if defined(GGML_USE_MUSA)
    GGML_UNUSED(dst);
    return false;
#else
    if (dst->op != GGML_OP_FLASH_ATTN_EXT_BANDED) {
        return false;
    }

    // quantized K/V are read as F16: check the op with F16 K/V of the same shape
    if (dst->src[1] && dst->src[2] &&
        (fattn_banded_kv_f16_convertible(dst->src[1]->type) || fattn_banded_kv_f16_convertible(dst->src[2]->type))) {
        ggml_tensor op  = *dst;
        ggml_tensor k16 = *dst->src[1];
        ggml_tensor v16 = *dst->src[2];
        for (ggml_tensor * t : {&k16, &v16}) {
            if (fattn_banded_kv_f16_convertible(t->type)) {
                if (t->nb[0] != ggml_type_size(t->type) || t->ne[0] % ggml_blck_size(t->type) != 0) {
                    return false;
                }
                t->type  = GGML_TYPE_F16;
                t->nb[0] = sizeof(half);
            }
        }
        op.src[1] = &k16;
        op.src[2] = &v16;
        return ggml_cuda_flash_attn_ext_banded_supported(device, &op);
    }

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    if (!q || !k || !v || !rel || q->type != GGML_TYPE_F32) {
        return false;
    }
    // quantized K/V: only the delegated FA kernel reads them, it converts them to F16
    const bool kv_quant_ok = (ggml_is_quantized(k->type) || ggml_is_quantized(v->type)) &&
        ggml_cuda_flash_attn_ext_banded_use_mma(device, dst);
    if (!(fattn_banded_type_supported(k->type) || (kv_quant_ok && ggml_is_quantized(k->type))) ||
        !(fattn_banded_type_supported(v->type) || (kv_quant_ok && ggml_is_quantized(v->type))) ||
        !fattn_banded_type_supported(rel->type)) {
        return false;
    }
    if ((q->ne[0] != 64 && q->ne[0] != 128) || v->ne[0] != q->ne[0] || k->ne[0] != q->ne[0]) {
        return false;
    }
    if (q->ne[2] % k->ne[2] != 0 || q->ne[2] % v->ne[2] != 0 || k->ne[2] != v->ne[2]) {
        return false;
    }
    if (q->ne[3] != k->ne[3] || q->ne[3] != v->ne[3]) {
        return false;
    }
    if (q->nb[0] != sizeof(float) || k->nb[0] != ggml_type_size(k->type) ||
        v->nb[0] != ggml_type_size(v->type) || rel->nb[0] != ggml_type_size(rel->type)) {
        return false;
    }
    if (rel->ne[0] <= 0 || rel->ne[1] != q->ne[2] || rel->ne[2] != q->ne[1] ||
        (rel->ne[3] != 1 && rel->ne[3] != q->ne[3])) {
        return false;
    }
    const ggml_tensor * sinks = dst->src[4];
    if (sinks && (sinks->type != GGML_TYPE_F32 || sinks->nb[0] != sizeof(float))) {
        return false;
    }
    const ggml_tensor * k_pos = dst->src[7];
    const ggml_tensor * q_pos = dst->src[8];
    if ((k_pos == nullptr) != (q_pos == nullptr)) {
        return false;
    }
    if (k_pos && (k_pos->type != GGML_TYPE_I32 || q_pos->type != GGML_TYPE_I32 ||
            !ggml_is_contiguous(k_pos) || !ggml_is_contiguous(q_pos) ||
            k_pos->ne[0] != k->ne[1] || q_pos->ne[0] != q->ne[1] ||
            (k_pos->ne[1] != 1 && k_pos->ne[1] != q->ne[3]) || (q_pos->ne[1] != 1 && q_pos->ne[1] != q->ne[3]))) {
        return false;
    }
    if (m && m->type == GGML_TYPE_F16 && m->nb[0] == sizeof(half) && !ggml_is_contiguous(m)) {
        // the first n_kv cells of a wider mask, as the model views it: the rows are read with their stride
        return q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0;
    }
    return !m || (m->type == GGML_TYPE_F16 && ggml_is_contiguous(m) &&
        q->ne[2] % m->ne[2] == 0 && q->ne[3] % m->ne[3] == 0);
#endif
}

// below this the decode kernel or, without it, the MMA kernel beats the scalar kernels
static constexpr int64_t fattn_banded_mma_min_q = 1;

static bool fattn_banded_mma_aligned(const ggml_tensor * t) {
    return (uintptr_t) t->data % 16 == 0 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0;
}

static bool fattn_banded_mma_usable(const int device, const ggml_tensor * dst) {
    if (ggml_cuda_banded_mma_disabled()) {
        return false;
    }
    if (!turing_mma_available(ggml_cuda_info().devices[device].cc)) {
        return false;
    }
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    if (q->ne[1] < fattn_banded_mma_min_q) {
        return false;
    }
    for (const ggml_tensor * t : {k, v}) {
        if (t->type == GGML_TYPE_F16 ? !fattn_banded_mma_aligned(t) : (t->type != GGML_TYPE_BF16 && !ggml_is_quantized(t->type))) {
            return false;
        }
    }
    return true;
}

// K/V the kernels cannot read directly are converted to a contiguous F16 copy in the pool
static void fattn_banded_kv_to_f16(ggml_backend_cuda_context & ctx, const ggml_tensor * t, ggml_cuda_pool_alloc<half> & buf,
        const char *& data, ggml_type & type, uint64_t (&nb)[4]) {
    const size_t ts = ggml_type_size(t->type); // strides in blocks
    buf.alloc(ggml_nelements(t));
    ggml_get_to_fp16_nc_cuda(t->type)(t->data, buf.get(), t->ne[0], t->ne[1], t->ne[2], t->ne[3],
        t->nb[1]/ts, t->nb[2]/ts, t->nb[3]/ts, ctx.stream());
    data  = (const char *) buf.get();
    type  = GGML_TYPE_F16;
    nb[0] = sizeof(half);
    nb[1] = t->ne[0]*nb[0];
    nb[2] = t->ne[1]*nb[1];
    nb[3] = t->ne[2]*nb[2];
}

static const ggml_tensor * fattn_banded_kv_to_f16_tensor(ggml_backend_cuda_context & ctx, const ggml_tensor * t,
        ggml_cuda_pool_alloc<half> & buf, ggml_tensor & t_f16) {
    const char * data = nullptr;
    ggml_type    type = t->type;
    uint64_t     nb[4];
    fattn_banded_kv_to_f16(ctx, t, buf, data, type, nb);
    t_f16      = *t;
    t_f16.data = const_cast<char *>(data);
    t_f16.type = type;
    for (int i = 0; i < 4; ++i) {
        t_f16.nb[i] = nb[i];
    }
    return &t_f16;
}

static void fattn_banded_mma_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const float scale, const int64_t rel_extent) {
    // ggml_flash_attn_ext_banded_set_causal: no mask, the kernels mask the cells after each query
    const int causal = ggml_flash_attn_ext_banded_is_causal(dst);
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * m     = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    const ggml_tensor * rel   = dst->src[6];
    cudaStream_t stream = ctx.stream();

    const int64_t D    = q->ne[0];
    const int64_t n_q  = q->ne[1];
    const int64_t n_kv = k->ne[1];

    const char * k_data = (const char *) k->data;
    const char * v_data = (const char *) v->data;
    uint64_t k_nb[4] = {k->nb[0], k->nb[1], k->nb[2], k->nb[3]};
    uint64_t v_nb[4] = {v->nb[0], v->nb[1], v->nb[2], v->nb[3]};
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool());
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool());
    // matching q8_0 / q4_0 K and V are dequantized per tile in the kernel
    ggml_type type_kv = k->type == v->type && fattn_banded_mma_tile_type(k->type) ? k->type : GGML_TYPE_F16;
    if (type_kv == GGML_TYPE_F16) {
        ggml_type type_k = k->type;
        ggml_type type_v = v->type;
        if (type_k != GGML_TYPE_F16) {
            fattn_banded_kv_to_f16(ctx, k, k_f16, k_data, type_k, k_nb);
        }
        if (type_v != GGML_TYPE_F16) {
            fattn_banded_kv_to_f16(ctx, v, v_f16, v_data, type_v, v_nb);
        }
    }

    const int nw = n_q >= 64 ? 4 : (n_q >= 32 ? 2 : 1);
    const int bq = nw*16;
    const int64_t n_qt = (n_q + bq - 1)/bq;
    const int64_t n_kt = (n_kv + fattn_banded_mma_bk - 1)/fattn_banded_mma_bk;

    ggml_cuda_pool_alloc<uint8_t> flags(ctx.pool());
    if (m) {
        const int64_t n_z = m->ne[2]*m->ne[3];
        flags.alloc(n_z*n_qt*n_kt);
        flash_attn_ext_banded_mma_tile_flags<<<dim3(n_kt, n_qt, n_z), 256, 0, stream>>>(
            (const char *) m->data, flags.get(), n_q, n_kv, bq, m->ne[2], m->nb[1], m->nb[2], m->nb[3]);
    }

    int occ = 0;
#define OCC_BANDED_MMA_T(DD, NW, T) CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, flash_attn_ext_banded_mma<DD, NW, T>, NW*WARP_SIZE, 0))
#define OCC_BANDED_MMA(DD, NW) \
    switch (type_kv) { \
        case GGML_TYPE_Q8_0: OCC_BANDED_MMA_T(DD, NW, GGML_TYPE_Q8_0); break; \
        case GGML_TYPE_Q4_0: OCC_BANDED_MMA_T(DD, NW, GGML_TYPE_Q4_0); break; \
        default:             OCC_BANDED_MMA_T(DD, NW, GGML_TYPE_F16);  break; \
    }
    if (D == 64) {
        if (nw == 4) { OCC_BANDED_MMA(64, 4); } else if (nw == 2) { OCC_BANDED_MMA(64, 2); } else { OCC_BANDED_MMA(64, 1); }
    } else {
        if (nw == 4) { OCC_BANDED_MMA(128, 4); } else if (nw == 2) { OCC_BANDED_MMA(128, 2); } else { OCC_BANDED_MMA(128, 1); }
    }
#undef OCC_BANDED_MMA
#undef OCC_BANDED_MMA_T
    const int64_t resident    = int64_t(std::max(occ, 1))*ggml_cuda_info().devices[ctx.device].nsm;
    const int64_t base_blocks = n_qt*q->ne[2]*q->ne[3];
    constexpr int64_t min_tiles_per_chunk = 4;
    const int64_t max_chunks  = std::max<int64_t>(1, std::min<int64_t>(n_kt/min_tiles_per_chunk, (4*resident + base_blocks - 1)/base_blocks));
    int64_t n_chunks = 1;
    double  best_eff = 0.0;
    for (int64_t c = 1; c <= max_chunks; ++c) {
        const int64_t blocks = base_blocks*c;
        const double  eff    = double(blocks) / double(((blocks + resident - 1)/resident)*resident);
        if (eff > best_eff + 1e-3) {
            best_eff = eff;
            n_chunks = c;
        }
    }
    const int64_t tiles_per_chunk = (n_kt + n_chunks - 1)/n_chunks;
    n_chunks = (n_kt + tiles_per_chunk - 1)/tiles_per_chunk;

    const int64_t n_rows = n_q*q->ne[2]*q->ne[3];
    ggml_cuda_pool_alloc<float> part(ctx.pool());
    if (n_chunks > 1) {
        part.alloc(n_rows*n_chunks*(D + 2));
    }

    const float * sinks_d  = (const float *) sinks->data;
    const int64_t sinks_s1 = sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0;
    const int64_t sinks_s2 = sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0;

    const ggml_tensor * k_pos = dst->src[7];
    const ggml_tensor * q_pos = dst->src[8];
    const int32_t * k_pos_d = k_pos ? (const int32_t *) k_pos->data : nullptr;
    const int32_t * q_pos_d = q_pos ? (const int32_t *) q_pos->data : nullptr;
    const int64_t kpos_s1 = k_pos && k_pos->ne[1] > 1 ? k_pos->ne[0] : 0;
    const int64_t qpos_s1 = q_pos && q_pos->ne[1] > 1 ? q_pos->ne[0] : 0;

    const dim3 blocks(n_qt*n_chunks, q->ne[2], q->ne[3]);
#define LAUNCH_BANDED_MMA_T(DD, NW, T) \
    flash_attn_ext_banded_mma<DD, NW, T><<<blocks, NW*WARP_SIZE, 0, stream>>>( \
        (const char *) q->data, k_data, v_data, m ? (const char *) m->data : nullptr, m ? flags.get() : nullptr, \
        (const char *) rel->data, n_chunks == 1 ? sinks_d : nullptr, (float *) dst->data, part.get(), \
        scale, rel->type, n_q, n_kv, q->ne[2], k->ne[2], n_chunks, tiles_per_chunk, rel_extent, \
        m ? m->ne[2] : 1, m ? m->ne[3] : 1, q->nb[1], q->nb[2], q->nb[3], \
        k_nb[1], k_nb[2], k_nb[3], v_nb[1], v_nb[2], v_nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], sinks_s1, sinks_s2, \
        k_pos_d, q_pos_d, kpos_s1, qpos_s1, causal)
#define LAUNCH_BANDED_MMA(DD, NW) \
    switch (type_kv) { \
        case GGML_TYPE_Q8_0: LAUNCH_BANDED_MMA_T(DD, NW, GGML_TYPE_Q8_0); break; \
        case GGML_TYPE_Q4_0: LAUNCH_BANDED_MMA_T(DD, NW, GGML_TYPE_Q4_0); break; \
        default:             LAUNCH_BANDED_MMA_T(DD, NW, GGML_TYPE_F16);  break; \
    }
#define LAUNCH_BANDED_MMA_NW(DD) \
    switch (nw) { \
        case 4:  LAUNCH_BANDED_MMA(DD, 4); break; \
        case 2:  LAUNCH_BANDED_MMA(DD, 2); break; \
        default: LAUNCH_BANDED_MMA(DD, 1); break; \
    }
    if (D == 64) {
        LAUNCH_BANDED_MMA_NW(64);
    } else {
        LAUNCH_BANDED_MMA_NW(128);
    }
#undef LAUNCH_BANDED_MMA_NW
#undef LAUNCH_BANDED_MMA
#undef LAUNCH_BANDED_MMA_T
    CUDA_CHECK(cudaGetLastError());

    if (n_chunks > 1) {
        constexpr int merge_warps = 4;
        const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
        const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
        if (D == 64) {
            flash_attn_ext_banded_f32_merge<64><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        } else {
            flash_attn_ext_banded_f32_merge<128><<<mblocks, mthreads, 0, stream>>>(
                part.get(), (float *) dst->data, n_rows, q->ne[2], n_q, n_chunks, sinks_d, sinks_s1, sinks_s2);
        }
    }
}

void ggml_cuda_flash_attn_ext_banded(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // ggml_flash_attn_ext_banded_set_causal: no mask, the kernels mask the cells after each query
    const int causal = ggml_flash_attn_ext_banded_is_causal(dst);
    GGML_ASSERT(ggml_cuda_flash_attn_ext_banded_supported(ctx.device, dst));

    const ggml_tensor * q   = dst->src[0];
    const ggml_tensor * k   = dst->src[1];
    const ggml_tensor * v   = dst->src[2];
    const ggml_tensor * m   = dst->src[3];
    const ggml_tensor * rel = dst->src[6];
    const ggml_tensor * sinks = dst->src[4];

    // only the MMA kernel reads src[6]; delegating otherwise drops the bias
    if (ggml_cuda_flash_attn_ext_banded_use_mma(ctx.device, dst)) {
        ggml_cuda_flash_attn_ext(ctx, dst);
        return;
    }

    float scale;
    memcpy(&scale, dst->op_params, sizeof(scale));
    // the tensor extent (not op_params) is authoritative after graph cloning
    const int64_t rel_extent = rel->ne[0];

    // decode: K and V read once per GQA group, quantized K/V read in place
    static const bool no_vec = [] {
        const char * env = getenv("GGML_CUDA_BANDED_NO_VEC");
        return env != nullptr && atoi(env) != 0;
    }();
    if (!no_vec && fattn_banded_vec_usable(dst)) {
        fattn_banded_vec_launch(ctx, dst, scale, rel_extent);
        return;
    }

    if (sinks && fattn_banded_mma_usable(ctx.device, dst)) {
        fattn_banded_mma_launch(ctx, dst, scale, rel_extent);
        return;
    }

    const ggml_tensor * k_pos = dst->src[7];
    const ggml_tensor * q_pos = dst->src[8];
    const int32_t * k_pos_d = k_pos ? (const int32_t *) k_pos->data : nullptr;
    const int32_t * q_pos_d = q_pos ? (const int32_t *) q_pos->data : nullptr;
    const int64_t kpos_s1 = k_pos && k_pos->ne[1] > 1 ? k_pos->ne[0] : 0;
    const int64_t qpos_s1 = q_pos && q_pos->ne[1] > 1 ? q_pos->ne[0] : 0;

    // the scalar kernels read F32 / F16 / BF16 elements: other K/V types go through an F16 copy
    ggml_cuda_pool_alloc<half> k_f16(ctx.pool());
    ggml_cuda_pool_alloc<half> v_f16(ctx.pool());
    ggml_tensor k_f16_t;
    ggml_tensor v_f16_t;
    if (!fattn_banded_type_supported(k->type)) {
        k = fattn_banded_kv_to_f16_tensor(ctx, k, k_f16, k_f16_t);
    }
    if (!fattn_banded_type_supported(v->type)) {
        v = fattn_banded_kv_to_f16_tensor(ctx, v, v_f16, v_f16_t);
    }

    {
        constexpr int split_warps   = 4;
        constexpr int min_chunk     = 64;
        const int64_t n_rows        = q->ne[1]*q->ne[2]*q->ne[3];
        const int64_t n_kv          = k->ne[1];
        const int64_t target_blocks = int64_t(ggml_cuda_info().devices[ctx.device].nsm)*16;
        int64_t n_chunks = n_rows < target_blocks ? (target_blocks + n_rows - 1)/n_rows : 1;
        n_chunks = std::min<int64_t>(n_chunks, (n_kv + min_chunk - 1)/min_chunk);
        // the causal attention without a mask is masked in the split kernel, even in one chunk
        if ((n_chunks > 1 || causal) && q->ne[1]*q->ne[3] <= 65535) {
            const int64_t chunk = (n_kv + n_chunks - 1)/n_chunks;
            n_chunks = (n_kv + chunk - 1)/chunk;
            const int64_t D = q->ne[0];
            ggml_cuda_pool_alloc<float> part(ctx.pool(), n_rows*n_chunks*(D + 2));
            const dim3 sblocks(n_chunks, q->ne[2], q->ne[1]*q->ne[3]);
            const dim3 sthreads(split_warps*WARP_SIZE, 1, 1);
            constexpr int merge_warps = 4;
            const dim3 mblocks((n_rows + merge_warps - 1)/merge_warps, 1, 1);
            const dim3 mthreads(merge_warps*WARP_SIZE, 1, 1);
            cudaStream_t stream = ctx.stream();
#define LAUNCH_SPLIT(DD) \
            flash_attn_ext_banded_f32_split<DD, split_warps><<<sblocks, sthreads, 0, stream>>>( \
                (const char *) q->data, (const char *) k->data, (const char *) v->data, \
                m ? (const char *) m->data : nullptr, (const char *) rel->data, part.get(), \
                scale, k->type, v->type, rel->type, q->ne[1], n_kv, q->ne[2], k->ne[2], n_chunks, chunk, \
                rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
                q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
                v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
                m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
                k_pos_d, q_pos_d, kpos_s1, qpos_s1, \
                rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3], causal); \
            flash_attn_ext_banded_f32_merge<DD><<<mblocks, mthreads, 0, stream>>>( \
                part.get(), (float *) dst->data, n_rows, q->ne[2], q->ne[1], n_chunks, \
                sinks ? (const float *) sinks->data : nullptr, \
                sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
                sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0)
            if (D == 64) {
                LAUNCH_SPLIT(64);
            } else {
                LAUNCH_SPLIT(128);
            }
#undef LAUNCH_SPLIT
            return;
        }
    }

    GGML_ASSERT(!causal && "the causal banded attention without a mask needs the split kernel");
    constexpr int warps_per_block = 4;
    const dim3 blocks((q->ne[1] + warps_per_block - 1) / warps_per_block, q->ne[2], q->ne[3]);
    const dim3 threads(warps_per_block * WARP_SIZE, 1, 1);
    cudaStream_t stream = ctx.stream();

#define LAUNCH_BANDED(D) \
    flash_attn_ext_banded_f32<D, warps_per_block><<<blocks, threads, 0, stream>>>( \
        (const char *) q->data, (const char *) k->data, (const char *) v->data, \
        m ? (const char *) m->data : nullptr, (const char *) rel->data, (float *) dst->data, \
        scale, k->type, v->type, rel->type, q->ne[1], k->ne[1], q->ne[2], k->ne[2], q->ne[3], \
        rel_extent, m ? m->ne[2] : 1, m ? m->ne[3] : 1, \
        q->nb[1], q->nb[2], q->nb[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3], \
        v->nb[0], v->nb[1], v->nb[2], v->nb[3], \
        m ? m->nb[1] : 0, m ? m->nb[2] : 0, m ? m->nb[3] : 0, \
        sinks ? (const float *) sinks->data : nullptr, \
        sinks && sinks->ne[1] > 1 ? int64_t(sinks->nb[1] / sizeof(float)) : 0, \
        sinks && sinks->ne[2] > 1 ? int64_t(sinks->nb[2] / sizeof(float)) : 0, \
        k_pos_d, q_pos_d, kpos_s1, qpos_s1, \
        rel->nb[0], rel->nb[1], rel->nb[2], rel->nb[3], rel->ne[3])

    if (q->ne[0] == 64) {
        LAUNCH_BANDED(64);
    } else {
        LAUNCH_BANDED(128);
    }
#undef LAUNCH_BANDED
}
