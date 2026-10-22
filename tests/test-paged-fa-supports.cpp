// [TAG_EXACT_CONCURRENCY] A device that claims a paged FLASH_ATTN_EXT must be able to launch it: every shape the
// paged kernel refuses at launch has to be refused by supports_op first, so the scheduler never hands it one.

#include "ggml.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

struct paged_shape {
    const char * name;
    int64_t      n_kv;       // K/V cells
    int64_t      n_tokens;   // queries
    int64_t      width;      // page table row width, 1 + n_kv/256 is what the kernel indexes
    int64_t      mask_rows;  // mask rows, one per query at least
    int64_t      mask_cols;  // mask columns, one per K cell at least
    bool         valid;
    int64_t      mask_planes = 1;  // the paged kernel reads one mask plane for every head
};

static bool supports(ggml_backend_dev_t dev, const paged_shape & s) {
    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*16 + ggml_graph_overhead(),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 256, s.n_tokens, 8, 1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, s.n_kv,     2, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 256, s.n_kv,     2, 1);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, s.mask_cols, s.mask_rows, s.mask_planes, 1);

    ggml_tensor * op = ggml_flash_attn_ext(ctx, q, k, v, m, 1.0f/sqrtf(256.0f), 0.0f, 0.0f);
    ggml_prec_set_acc(op, GGML_PREC_F32);
    op->src[5] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, s.width, s.n_tokens);

    const bool res = ggml_backend_dev_supports_op(dev, op);

    ggml_free(ctx);

    return res;
}

int main() {
    ggml_backend_load_all();

    const paged_shape shapes[] = {
        { "1024 cells, 4 queries",               1024,  4, 5,  4, 1024, true  },
        { "512 cells, 1 query",                   512,  1, 3,  1,  512, true  },
        { "table one page too narrow",           1024,  4, 4,  4, 1024, false },
        { "table one page too wide",             1024,  4, 6,  4, 1024, false },
        { "table of one column",                  512,  4, 1,  4,  512, false },
        { "cells not a whole number of pages",   1000,  4, 4,  4, 1000, false },
        { "mask with fewer rows than queries",   1024,  4, 5,  3, 1024, false },
        { "mask narrower than the cells",        1024,  4, 5,  4,  768, false },
        { "a mask plane per head",               1024,  4, 5,  4, 1024, false, 8 },
        { "a mask plane per K/V head",           1024,  4, 5,  4, 1024, false, 2 },
    };

    int rc = 0;
    int n_checked = 0;

    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        const char * reg_name = ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev));

        // only the CUDA backend (also built as ROCm and MUSA) reads the page table; the others are kept off this path by llama_exact_backend_reg()
        if (strcmp(reg_name, "CUDA") != 0 && strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0) {
            continue;
        }

        if (!supports(dev, shapes[0])) {
            printf("%s: no paged flash attention in this build, skipped\n", ggml_backend_dev_name(dev));
            continue;
        }

        n_checked++;

        for (const auto & s : shapes) {
            const bool got = supports(dev, s);
            printf("%s: %-36s supports_op %d, launchable %d\n", ggml_backend_dev_name(dev), s.name, got, s.valid);
            if (got != s.valid) {
                rc = 1;
            }
        }
    }

    if (n_checked == 0) {
        printf("no CUDA device with paged flash attention, nothing to check\n");
    } else {
        printf("%s\n", rc == 0 ? "ok: supports_op accepts exactly the paged shapes the kernel launches" :
                                 "FAIL: supports_op and the paged launch disagree");
    }

    return rc;
}
