#pragma once

#include "common.cuh"

void ggml_cuda_op_inkling_control_gate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * cur,
        const ggml_tensor * msg,
        const ggml_tensor * forget_raw,
        const ggml_tensor * gain_raw,
        ggml_tensor * dst);

void ggml_cuda_op_add_scale(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * a,
        const ggml_tensor * b,
        ggml_tensor * dst,
        float scale,
        float bias);

void ggml_cuda_op_inkling_loop_down_concat(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_down,
        const ggml_tensor * a,
        const ggml_tensor * b,
        ggml_tensor * dst);

void ggml_cuda_op_inkling_loop_up_gate(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * w_gain,
        const ggml_tensor * w_forget,
        const ggml_tensor * gate,
        const ggml_tensor * cur,
        const ggml_tensor * msg,
        ggml_tensor * dst);

void ggml_cuda_op_inkling_shortconv_step(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * past,
        const ggml_tensor * cur_in,
        const ggml_tensor * weight,
        const ggml_tensor * cur_res,
        const ggml_tensor * outer_res,
        ggml_tensor * state_cell,
        ggml_tensor * dst,
        float scale,
        float bias);

void ggml_cuda_op_inkling_shortconv_step_dyn(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * past,
        const ggml_tensor * cur_in,
        const ggml_tensor * weight,
        const ggml_tensor * dynamic,
        const ggml_tensor * cur_res,
        ggml_tensor * state_cell,
        ggml_tensor * dst,
        int64_t n_heads,
        float scale,
        float bias);

void ggml_cuda_op_inkling_dense_kq_bias(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * rel_bias,
        const ggml_tensor * r_cur,
        const ggml_tensor * sinks,
        const ggml_tensor * rel_pos,
        ggml_tensor * dst);
