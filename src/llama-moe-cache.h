#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // true if t is a slot map (host memory that changes between graph computations)
    bool is_slot_map(const ggml_tensor * t) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache, -n if a run of n cached experts is too short to copy
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // for large batches: count a ubatch of more than 32 tokens (for the copy statistics)
    void count_large_ubatch();

    // for large batches: count the experts of w uploaded from host memory (one copy call)
    void count_upload(const ggml_tensor * w, size_t n_experts, size_t bytes);

    // for large batches: admit the most used experts of the layer of w, counts[e] is the number of tokens routed to expert e
    // must be called for each expert tensor of the layer after dst is filled, the slots are filled from dst
    void seed(ggml_backend_t backend, const ggml_tensor * w, const ggml_tensor * dst, const std::vector<int32_t> & counts);

    // after the graph is computed: drop the admissions of layers that were not seeded for all of their expert tensors
    void seed_end();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
