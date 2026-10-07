#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // the NextN/MTP layers are only cached if mtp is true
    llama_moe_cache(const llama_model & model, ggml_backend_t backend, ggml_backend_buffer_type_t buft, size_t size, bool mtp);
    ~llama_moe_cache();

    // smallest size that gives every group of host-resident expert layers on dev the slots for the experts of one token
    // in a context that is not an MTP context
    // returns 0 if no layer would be cached
    static size_t min_size(const llama_model & model, ggml_backend_dev_t dev);

    ggml_backend_t backend() const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
