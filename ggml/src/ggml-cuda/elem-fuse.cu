#include "elem-fuse.cuh"
#include "reduce_rows.cuh"

#include <cstring>

// ---------------------------------------------------------------------------------------------------------------------
// y = c -/+ tanh(a)*b
// each value is rounded as in the separate TANH (tanhf), MUL and SUB/ADD kernels: the product and the sum are not
// contracted into an FMA

template <bool sub>
static __global__ void tanh_mul_addsub_f32(
        const char * a_ptr, const char * b_ptr, const char * c_ptr, float * dst_ptr,
        const int ne0, const int ne1, const int ne2,
        const int64_t a1, const int64_t a2, const int64_t a3,
        const int64_t b1, const int64_t b2, const int64_t b3,
        const int64_t c1, const int64_t c2, const int64_t c3) {
    const char * GGML_CUDA_RESTRICT a   = a_ptr;
    const char * GGML_CUDA_RESTRICT b   = b_ptr;
    const char * GGML_CUDA_RESTRICT c   = c_ptr;
    float      * GGML_CUDA_RESTRICT dst = dst_ptr;
    ggml_cuda_pdl_lc();
    const int i0 = blockIdx.x*blockDim.x + threadIdx.x;
    const int i1 = blockIdx.y;
    const int i2 = blockIdx.z % ne2;
    const int i3 = blockIdx.z / ne2;

    ggml_cuda_pdl_sync();
    if (i0 >= ne0) {
        return;
    }

    const float av = *(const float *) (a + i1*a1 + i2*a2 + i3*a3 + i0*sizeof(float));
    const float bv = *(const float *) (b + i1*b1 + i2*b2 + i3*b3 + i0*sizeof(float));
    const float cv = *(const float *) (c + i1*c1 + i2*c2 + i3*c3 + i0*sizeof(float));

    const float m = __fmul_rn(tanhf(av), bv);

    dst[((int64_t(i3)*ne2 + i2)*ne1 + i1)*ne0 + i0] = sub ? __fsub_rn(cv, m) : __fadd_rn(cv, m);
}

bool ggml_cuda_elem_overlap_ok(const ggml_tensor * dst, const ggml_tensor * src) {
    const char * d0 = (const char *) dst->data;
    const char * d1 = d0 + ggml_nbytes(dst);
    const char * s0 = (const char *) src->data;
    const char * s1 = s0 + ggml_nbytes(src);
    if (d1 <= s0 || s1 <= d0) {
        return true;
    }
    return d0 == s0 && ggml_is_contiguous(dst) && ggml_is_contiguous(src) && ggml_nelements(dst) == ggml_nelements(src) &&
        dst->type == src->type;
}

static bool elem_is_view_op(const ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

static bool elem_f32_rows_ok(const ggml_tensor * t, const ggml_tensor * shape) {
    return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float) && ggml_are_same_shape(t, shape);
}

int ggml_cuda_match_tanh_mul_addsub(const ggml_cgraph * cgraph, int i, ggml_cuda_tanh_mul_addsub_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    // the views of b (no kernels) may sit between TANH and MUL
    int j = i + 1;
    while (j < i + 3 && j < cgraph->n_nodes && elem_is_view_op(cgraph->nodes[j]->op)) {
        ++j;
    }
    if (j + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * t = cgraph->nodes[i];
    const ggml_tensor * m = cgraph->nodes[j];
    ggml_tensor       * y = cgraph->nodes[j + 1];
    if (t->op != GGML_OP_UNARY || ggml_get_unary_op(t) != GGML_UNARY_OP_TANH || m->op != GGML_OP_MUL ||
        (y->op != GGML_OP_SUB && y->op != GGML_OP_ADD)) {
        return 0;
    }

    const ggml_tensor * b = nullptr;
    if (m->src[0] == t) {
        b = m->src[1];
    } else if (m->src[1] == t) {
        b = m->src[0];
    } else {
        return 0;
    }

    const ggml_tensor * c = nullptr;
    bool sub = false;
    if (y->op == GGML_OP_SUB) {
        if (y->src[1] != m || y->src[0] == m) {
            return 0;
        }
        c   = y->src[0];
        sub = true;
    } else {
        if (y->src[0] == m && y->src[1] != m) {
            c = y->src[1];
        } else if (y->src[1] == m && y->src[0] != m) {
            c = y->src[0];
        } else {
            return 0;
        }
    }

    const ggml_tensor * a = t->src[0];
    if (!elem_f32_rows_ok(a, y) || !elem_f32_rows_ok(b, y) || !elem_f32_rows_ok(c, y) || !elem_f32_rows_ok(t, y) ||
        !elem_f32_rows_ok(m, y) || !ggml_is_contiguous(y) || !ggml_is_contiguous(t) || b == t || c == t) {
        return 0;
    }
    if (y->ne[1] > 65535 || y->ne[2]*y->ne[3] > 65535 || y->ne[0] > INT_MAX) {
        return 0;
    }

    // the skipped views must not be views of an elided node
    for (int k = i + 1; k < j; ++k) {
        if (cgraph->nodes[k]->view_src == t || cgraph->nodes[k]->view_src == m) {
            return 0;
        }
    }

    const ggml_op ops[3]  = { GGML_OP_UNARY, GGML_OP_MUL, y->op };
    const int     idxs[3] = { i, j, j + 1 };
    const int     outs[1] = { j + 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs, 3, ops, outs, 1)) {
        return 0;
    }

    if (!ggml_cuda_elem_overlap_ok(y, a) || !ggml_cuda_elem_overlap_ok(y, b) || !ggml_cuda_elem_overlap_ok(y, c)) {
        return 0;
    }

    match.a   = a;
    match.b   = b;
    match.c   = c;
    match.dst = y;
    match.sub = sub;
    return j + 2 - i;
}

void ggml_cuda_op_tanh_mul_addsub(ggml_backend_cuda_context & ctx, const ggml_cuda_tanh_mul_addsub_match & match) {
    const ggml_tensor * a = match.a;
    const ggml_tensor * b = match.b;
    const ggml_tensor * c = match.c;
    ggml_tensor       * y = match.dst;

    const int block = 256;
    const dim3 grid((y->ne[0] + block - 1)/block, y->ne[1], y->ne[2]*y->ne[3]);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid, dim3(block, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params,
            (const char *) a->data, (const char *) b->data, (const char *) c->data, (float *) y->data,
            (int) y->ne[0], (int) y->ne[1], (int) y->ne[2],
            (int64_t) a->nb[1], (int64_t) a->nb[2], (int64_t) a->nb[3],
            (int64_t) b->nb[1], (int64_t) b->nb[2], (int64_t) b->nb[3],
            (int64_t) c->nb[1], (int64_t) c->nb[2], (int64_t) c->nb[3]);
    };
    if (match.sub) {
        launch(tanh_mul_addsub_f32<true>);
    } else {
        launch(tanh_mul_addsub_f32<false>);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// x / f(sum_rows(x)) [* scale + bias], f = SCALE (x*s + b) or CLAMP
// one block per row with the block size of sum_rows_f32_cuda, so the row sum is reduce_row_f32's for the same shape;
// SCALE as scale_f32's FMA, the division as op_div's a / b

template <bool clamp, bool has_scale>
static __global__ void row_norm_div_f32(const float * x_ptr, float * dst_ptr, const int ncols,
        const float p0, const float p1, const float out_scale, const float out_bias) {
    const float * GGML_CUDA_RESTRICT x   = x_ptr;
    float       * GGML_CUDA_RESTRICT dst = dst_ptr;
    const int64_t row = blockIdx.x;

    const float * xr = x + row*ncols;
    const float sum = reduce_row_f32(xr, ncols); // waits for the previous kernel
    ggml_cuda_pdl_lc();

    const float d = clamp ? fminf(fmaxf(sum, p0), p1) : __fmaf_rn(p0, sum, p1);

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        const float v = xr[col] / d;
        dst[row*ncols + col] = has_scale ? __fmaf_rn(out_scale, v, out_bias) : v;
    }
}

int ggml_cuda_match_row_norm_div(const ggml_cgraph * cgraph, int i, ggml_cuda_row_norm_div_match & match) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * sum = cgraph->nodes[i];
    const ggml_tensor * f   = cgraph->nodes[i + 1];
    const ggml_tensor * div = cgraph->nodes[i + 2];
    if (sum->op != GGML_OP_SUM_ROWS || (f->op != GGML_OP_SCALE && f->op != GGML_OP_CLAMP) || div->op != GGML_OP_DIV) {
        return 0;
    }
    const ggml_tensor * x = sum->src[0];
    if (f->src[0] != sum || div->src[0] != x || div->src[1] != f) {
        return 0;
    }
    if (x->type != GGML_TYPE_F32 || sum->type != GGML_TYPE_F32 || f->type != GGML_TYPE_F32 || div->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(div) || !ggml_are_same_shape(x, div) || x->ne[0] > INT_MAX ||
        ggml_nrows(x) > INT_MAX || !ggml_is_contiguous(sum) || !ggml_is_contiguous(f) || ggml_nelements(sum) != ggml_nrows(x)) {
        return 0;
    }

    int n = 3;
    ggml_op ops[5] = { GGML_OP_SUM_ROWS, f->op, GGML_OP_DIV, GGML_OP_RESHAPE, GGML_OP_SCALE };
    ggml_tensor * out = cgraph->nodes[i + 2];
    bool has_scale = false;
    if (i + 4 < cgraph->n_nodes) {
        const ggml_tensor * r = cgraph->nodes[i + 3];
        ggml_tensor       * s = cgraph->nodes[i + 4];
        if (r->op == GGML_OP_RESHAPE && r->src[0] == div && s->op == GGML_OP_SCALE && s->src[0] == r &&
            s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) && ggml_nelements(s) == ggml_nelements(div)) {
            n = 5;
            out = s;
            has_scale = true;
        }
    }

    const int outs[1] = { i + n - 1 };
    if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs, 1)) {
        if (n == 5) {
            // the plain division alone
            n = 3;
            out = cgraph->nodes[i + 2];
            has_scale = false;
            const int outs3[1] = { i + 2 };
            if (!ggml_can_fuse_subgraph(cgraph, i, n, ops, outs3, 1)) {
                return 0;
            }
        } else {
            return 0;
        }
    }

    // each thread reads x only at the columns it writes, after the row sum
    if (!ggml_cuda_elem_overlap_ok(out, x)) {
        return 0;
    }

    match.x     = x;
    match.dst   = out;
    match.clamp = f->op == GGML_OP_CLAMP;
    memcpy(&match.p0, (const float *) f->op_params + 0, sizeof(float));
    memcpy(&match.p1, (const float *) f->op_params + 1, sizeof(float));
    match.has_scale = has_scale;
    if (has_scale) {
        memcpy(&match.out_scale, (const float *) out->op_params + 0, sizeof(float));
        memcpy(&match.out_bias,  (const float *) out->op_params + 1, sizeof(float));
    }
    return n;
}

void ggml_cuda_op_row_norm_div(ggml_backend_cuda_context & ctx, const ggml_cuda_row_norm_div_match & match) {
    const ggml_tensor * x = match.x;
    const int     ncols = x->ne[0];
    const int64_t nrows = ggml_nrows(x);

    // the block size of sum_rows_f32_cuda for this shape
    const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int block_size = (nrows / nsm) < 2 ? 512 : (ncols < 1024 ? 32 : 128);

    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(dim3(nrows, 1, 1), dim3(block_size, 1, 1), 0, ctx.stream());

    auto launch = [&](auto kernel) {
        ggml_cuda_kernel_launch(kernel, launch_params, (const float *) x->data, (float *) match.dst->data, ncols,
            match.p0, match.p1, match.out_scale, match.out_bias);
    };
    if (match.clamp) {
        match.has_scale ? launch(row_norm_div_f32<true, true>) : launch(row_norm_div_f32<true, false>);
    } else {
        match.has_scale ? launch(row_norm_div_f32<false, true>) : launch(row_norm_div_f32<false, false>);
    }
}
