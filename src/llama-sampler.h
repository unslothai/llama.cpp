#pragma once

#include "llama.h"

#include <vector>

struct llama_vocab;
struct llama_grammar;

// sampler chain

struct llama_sampler_chain {
    llama_sampler_chain_params params;

    bool backend_draft_rows = false; // llama_sampler_chain_set_backend_draft_rows

    // has .backend_init() been called?
    bool is_init = false;

    uint32_t n_nodes = 0;

    struct info {
        bool is_backend;

        llama_sampler * ptr;
    };

    std::vector<info> samplers;

    // pre-allocated buffer for llama_sampler_sample to avoid repeated allocations
    std::vector<llama_token_data> cur;

    // timing

    mutable int64_t t_sample_us;

    mutable int32_t n_sample;
};

uint32_t llama_sampler_backend_n_nodes(const llama_sampler * sampler);
void llama_sampler_backend_begin(llama_sampler * sampler);

// the input tokens of the output rows of the sequence in the current ubatch, in ubatch order
// called before backend_set_input; several rows per sequence come from draft verification, where the sampler of
// row g must see the history the CPU sampler has after accepting the tokens of rows 1..g
void llama_sampler_backend_set_rows(llama_sampler * sampler, const llama_token * tokens, uint32_t n_rows);

// the number of chains the context builds for the sampler (min(n_outputs_max_per_seq, n_ubatch): a ubatch has at most
// n_ubatch rows of a sequence), called after backend_init, which validates against n_outputs_max_per_seq; sizes the
// inputs the samplers share between their chains
void llama_sampler_backend_set_n_chains(llama_sampler * sampler, uint32_t n_chains);

struct llama_sampler * llama_sampler_init_dry_testing(
        float   dry_multiplier,
        float   dry_base,
        int32_t dry_allowed_length,
        int32_t dry_penalty_last_n,
        const std::vector<std::vector<llama_token>> & seq_breakers);
