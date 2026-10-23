// [TAG_PREEMPT] llama_memory_n_cells_used() reads the KV cells: a prefix copied to other sequences is counted once, and only cells
// whose every sequence is given are reported reclaimable. The server's preemption planner reads its occupancy from it.

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

static bool decode(llama_context * ctx, llama_seq_id seq, llama_pos p0, int n) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        batch.token[i]     = 1 + (p0 + i) % 50;
        batch.pos[i]       = p0 + i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = seq;
        batch.logits[i]    = i == n - 1;
    }
    batch.n_tokens = n;
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static int check(const char * what, int32_t got, int32_t want) {
    printf("%-52s %4d (want %d)\n", what, got, want);
    return got == want ? 0 : 1;
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[i + 1];
        }
    }
    if (!path) {
        fprintf(stderr, "usage: %s -m model.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 256;
    cparams.n_seq_max  = 3;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create a context\n");
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);

    // seq 0: 120 cells; its first 100 copied to seqs 1 and 2, which then add 30 and 40 of their own
    if (!decode(ctx, 0, 0, 120)) {
        return 1;
    }
    llama_memory_seq_cp(mem, 0, 1, 0, 100);
    llama_memory_seq_cp(mem, 0, 2, 0, 100);
    if (!decode(ctx, 1, 100, 30) || !decode(ctx, 2, 100, 40)) {
        return 1;
    }

    int bad = 0;
    int32_t reclaim = -1;

    const llama_seq_id s0[]  = { 0 };
    const llama_seq_id s01[] = { 0, 1 };
    const llama_seq_id all[] = { 0, 1, 2 };

    bad += check("used: 100 shared + 20 + 30 + 40", llama_memory_n_cells_used(mem, nullptr, 0, &reclaim), 190);
    bad += check("reclaimable with no sequence given", reclaim, 0);

    llama_memory_n_cells_used(mem, s0, 1, &reclaim);
    bad += check("reclaimable if seq 0 went: its own 20", reclaim, 20);

    llama_memory_n_cells_used(mem, s01, 2, &reclaim);
    bad += check("reclaimable if seqs 0 and 1 went: 20 + 30", reclaim, 50);

    llama_memory_n_cells_used(mem, all, 3, &reclaim);
    bad += check("reclaimable if every sequence went", reclaim, 190);

    // an id given twice is still one owner: the prefix shared with seq 2 stays, seq 0's own cells go
    const llama_seq_id s00[]  = { 0, 0 };
    const llama_seq_id s001[] = { 0, 0, 1 };
    llama_memory_n_cells_used(mem, s00, 2, &reclaim);
    bad += check("reclaimable if seq 0 (given twice) went: its own 20", reclaim, 20);

    llama_memory_n_cells_used(mem, s001, 3, &reclaim);
    bad += check("reclaimable if seqs 0 (twice) and 1 went: 20 + 30", reclaim, 50);

    // removing seq 0 frees only its own cells: the copied prefix is still held by 1 and 2
    llama_memory_seq_rm(mem, 0, -1, -1);
    bad += check("used after seq 0 is removed: 100 + 30 + 40", llama_memory_n_cells_used(mem, nullptr, 0, nullptr), 170);

    const llama_seq_id s1[] = { 1 };
    llama_memory_n_cells_used(mem, s1, 1, &reclaim);
    bad += check("reclaimable if seq 1 went: its own 30", reclaim, 30);

    bad += check("no memory: unknown", llama_memory_n_cells_used(nullptr, nullptr, 0, nullptr), -1);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("%s\n", bad == 0 ? "ok: shared cells are counted once and reclaimed only with every owner" : "FAIL");
    return bad == 0 ? 0 : 1;
}
