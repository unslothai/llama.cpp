#include "llama-impl.h"
#include "llama-mmap.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <climits>
#include <atomic>
#include <mutex>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <vector>

struct llama_logger_state {
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_logger_state g_logger_state;

void llama_prefetch_rows(const ggml_tensor * tensor, const int32_t * rows, size_t n_rows) {
    if (!tensor || !tensor->data || !tensor->buffer || !ggml_backend_buffer_is_host(tensor->buffer) || n_rows == 0) {
        return;
    }

    GGML_ASSERT(ggml_is_matrix(tensor));

    const size_t row_bytes = ggml_row_size(tensor->type, tensor->ne[0]);
    const auto * base = (const char *) tensor->data;

    std::vector<llama_memory_range> mr;
    mr.reserve(n_rows);
    for (size_t i = 0; i < n_rows; ++i) {
        GGML_ASSERT(rows[i] >= 0 && rows[i] < tensor->ne[1]);
        mr.push_back({ base + (size_t) rows[i] * tensor->nb[1], row_bytes });
    }

    llama_prefetch(std::move(mr));
}

time_meas::time_meas(int64_t & t_acc, bool disable) : t_start_us(disable ? -1 : ggml_time_us()), t_acc(t_acc) {}

time_meas::~time_meas() {
    if (t_start_us >= 0) {
        t_acc += ggml_time_us() - t_start_us;
    }
}

void llama_log_get(ggml_log_callback * log_callback, void ** user_data) {
    ggml_log_get(log_callback, user_data);
}

void llama_log_set(ggml_log_callback log_callback, void * user_data) {
    ggml_log_set(log_callback, user_data);
    g_logger_state.log_callback = log_callback ? log_callback : llama_log_callback_default;
    g_logger_state.log_callback_user_data = user_data;
}

static void llama_log_internal_v(ggml_log_level level, const char * format, va_list args) {
    va_list args_copy;
    va_copy(args_copy, args);
    char buffer[128];
    int len = vsnprintf(buffer, 128, format, args);
    if (len < 128) {
        g_logger_state.log_callback(level, buffer, g_logger_state.log_callback_user_data);
    } else {
        char * buffer2 = new char[len + 1];
        vsnprintf(buffer2, len + 1, format, args_copy);
        buffer2[len] = 0;
        g_logger_state.log_callback(level, buffer2, g_logger_state.log_callback_user_data);
        delete[] buffer2;
    }
    va_end(args_copy);
}

void llama_log_internal(ggml_log_level level, const char * format, ...) {
    va_list args;
    va_start(args, format);
    llama_log_internal_v(level, format, args);
    va_end(args);
}

void llama_log_callback_default(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    fputs(text, stderr);
    fflush(stderr);
}

void llama_clear_tensor_data(ggml_tensor * t, size_t offset, size_t size) {
    static const std::vector<uint8_t> zeros(1024*1024, 0);

    // not all backend buffers implement ggml_backend_tensor_memset(), so write zeros instead
    // TODO: make this a generic fallback in `ggml_backend_tensor_memset` when `set_tensor` is available
    for (size_t ofs = 0; ofs < size; ofs += zeros.size()) {
        ggml_backend_tensor_set(t, zeros.data(), offset + ofs, std::min(size - ofs, zeros.size()));
    }
}

void replace_all(std::string & s, const std::string & search, const std::string & replace) {
    if (search.empty()) {
        return;
    }
    std::string builder;
    builder.reserve(s.length());
    size_t pos = 0;
    size_t last_pos = 0;
    while ((pos = s.find(search, last_pos)) != std::string::npos) {
        builder.append(s, last_pos, pos - last_pos);
        builder.append(replace);
        last_pos = pos + search.length();
    }
    builder.append(s, last_pos, std::string::npos);
    s = std::move(builder);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

std::string llama_format_tensor_shape(const std::vector<int64_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, ne.at(i));
    }
    return buf;
}

std::string llama_format_tensor_shape(const struct ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%6" PRId64, t->ne[0]);
    for (int i = 1; i < GGML_MAX_DIMS; i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), ", %6" PRId64, t->ne[i]);
    }
    return buf;
}

static std::string gguf_data_to_str(enum gguf_type type, const void * data, size_t i) {
    switch (type) {
        case GGUF_TYPE_UINT8:   return std::to_string(((const uint8_t  *)data)[i]);
        case GGUF_TYPE_INT8:    return std::to_string(((const int8_t   *)data)[i]);
        case GGUF_TYPE_UINT16:  return std::to_string(((const uint16_t *)data)[i]);
        case GGUF_TYPE_INT16:   return std::to_string(((const int16_t  *)data)[i]);
        case GGUF_TYPE_UINT32:  return std::to_string(((const uint32_t *)data)[i]);
        case GGUF_TYPE_INT32:   return std::to_string(((const int32_t  *)data)[i]);
        case GGUF_TYPE_UINT64:  return std::to_string(((const uint64_t *)data)[i]);
        case GGUF_TYPE_INT64:   return std::to_string(((const int64_t  *)data)[i]);
        case GGUF_TYPE_FLOAT32: return std::to_string(((const float    *)data)[i]);
        case GGUF_TYPE_FLOAT64: return std::to_string(((const double   *)data)[i]);
        case GGUF_TYPE_BOOL:    return ((const int8_t *)data)[i] != 0 ? "true" : "false";
        default:                return format("unknown type %d", type);
    }
}

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i, size_t max_len) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                const size_t arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::string res = "[";
                size_t j = 0;
                for (; j < arr_n && res.size() <= max_len; j++) {
                    if (j > 0) {
                        res += ", ";
                    }
                    if (arr_type == GGUF_TYPE_STRING) {
                        const char * s = gguf_get_arr_str(ctx_gguf, i, j);
                        const size_t need = res.size() + 2 * strlen(s) + 2;
                        if (res.capacity() < need) {
                            res.reserve(std::max(need, 2 * res.capacity()));
                        }
                        res += '"';
                        for (; *s; s++) {
                            if (*s == '\\' || *s == '"') {
                                res += '\\';
                            }
                            res += *s;
                        }
                        res += '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        res += "???";
                    } else {
                        res += gguf_data_to_str(arr_type, data, j);
                    }
                }
                if (j == arr_n) {
                    res += "]";
                }
                return res;
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

// [TAG_EXACT_CONCURRENCY]
bool llama_exact_backend_name(const char * reg_name) {
    return reg_name && (strcmp(reg_name, "CUDA") == 0 || strcmp(reg_name, "ROCm") == 0 || strcmp(reg_name, "MUSA") == 0);
}

// [TAG_EXACT_CONCURRENCY] Host buffers matter: an op runs where its weight lives, and a host weight's op moves to the GPU only for wide batches (ggml_backend_cuda_device_offload_op).
// The CPU matmul picks between its SGEMM and its vector dot by the batch width too.
bool llama_exact_buft_invariant(ggml_backend_buffer_type_t buft) {
    if (!buft || ggml_backend_buft_is_host(buft)) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;

    return reg && llama_exact_backend_name(ggml_backend_reg_name(reg));
}

bool llama_exact_concurrency() {
    static const bool enabled = []() {
        const char * val = getenv("LLAMA_EXACT_CONCURRENCY");
        return val && atoi(val) != 0;
    }();

    return enabled;
}

// [TAG_EXACT_CONCURRENCY] tokens one sequence contributes to a decode step, see llama.h
static std::atomic<uint32_t> g_exact_decode_tokens{1};

// One lock for token figure, sequence count and width, so an interleaved report cannot leave a width that covers neither.
static std::recursive_mutex g_exact_mutex;

// the most sequences any context was created with; the tokens figure is process wide, so raising it re-reports every context's width
static std::atomic<uint32_t> g_exact_max_n_seq{0};

static bool llama_exact_width_within_explicit_bound(uint32_t n_cols);

static bool llama_exact_width_of(uint32_t n_seq, uint32_t n_tokens, uint32_t & n_cols) {
    const uint64_t w = (uint64_t) n_seq * (uint64_t) n_tokens;

    if (w > (uint64_t) INT32_MAX) {
        LLAMA_LOG_ERROR("%s: a decode step of %u sequences with %u tokens each is too wide to report\n", __func__, n_seq, n_tokens);
        return false;
    }

    n_cols = (uint32_t) w;

    return true;
}

bool llama_exact_check_n_seq(uint32_t n_seq) {
    std::lock_guard<std::recursive_mutex> lock(g_exact_mutex);

    const uint32_t n_seq_max = std::max(n_seq, g_exact_max_n_seq.load(std::memory_order_relaxed));

    uint32_t n_cols = 0;

    return llama_exact_width_of(n_seq_max, llama_exact_decode_tokens(), n_cols) && llama_exact_width_within_explicit_bound(n_cols);
}

bool llama_exact_report_n_seq(uint32_t n_seq) {
    std::lock_guard<std::recursive_mutex> lock(g_exact_mutex);

    const uint32_t n_seq_max = std::max(n_seq, g_exact_max_n_seq.load(std::memory_order_relaxed));

    uint32_t n_cols = 0;

    if (!llama_exact_width_of(n_seq_max, llama_exact_decode_tokens(), n_cols) || !llama_set_exact_decode_width(n_cols)) {
        return false;
    }

    uint32_t cur = g_exact_max_n_seq.load(std::memory_order_relaxed);

    while (n_seq > cur && !g_exact_max_n_seq.compare_exchange_weak(cur, n_seq, std::memory_order_relaxed)) {
    }

    return true;
}

bool llama_set_exact_decode_tokens(uint32_t n_tokens) {
    n_tokens = n_tokens > 0 ? n_tokens : 1;

    std::lock_guard<std::recursive_mutex> lock(g_exact_mutex);

    // never lowered: a narrower context set up later would turn an existing speculative context's verify steps into prompts
    if (n_tokens <= g_exact_decode_tokens.load(std::memory_order_relaxed)) {
        return true;
    }

    // every context widens with the figure, so report the width first; one the explicit bound cannot cover leaves the old figure in place
    const uint32_t n_seq = g_exact_max_n_seq.load(std::memory_order_relaxed);

    uint32_t n_cols = 0;

    if (n_seq > 0 && (!llama_exact_width_of(n_seq, n_tokens, n_cols) || !llama_set_exact_decode_width(n_cols))) {
        return false;
    }

    g_exact_decode_tokens.store(n_tokens, std::memory_order_relaxed);

    return true;
}

uint32_t llama_exact_decode_tokens(void) {
    return g_exact_decode_tokens.load(std::memory_order_relaxed);
}

// [TAG_EXACT_CONCURRENCY] the widest decode ubatch reported so far, see llama.h; reached through the registry so an absent or late-loaded backend costs nothing
static std::atomic<uint32_t> g_exact_decode_width{0};

// an explicit column bound wins in the CUDA backend, so a width above it would leave decodes batched past the bound
static bool llama_exact_width_within_explicit_bound(uint32_t n_cols) {
    static const int explicit_cols = []() {
        const char * val = getenv("GGML_CUDA_BATCH_INVARIANT_MAX_COLS");
        return val ? atoi(val) : -1;
    }();

    if (explicit_cols > 0 && (uint32_t) explicit_cols < n_cols) {
        LLAMA_LOG_ERROR("%s: GGML_CUDA_BATCH_INVARIANT_MAX_COLS is %d but LLAMA_EXACT_CONCURRENCY needs at least %u columns for the decode step just requested; raise it, set it to 0 for no bound, or unset it\n",
                __func__, explicit_cols, n_cols);
        return false;
    }

    return true;
}

bool llama_set_exact_decode_width(uint32_t n_cols) {
    // the backends take the width as an int; a wider one would reach them negative and be ignored, while the maximum kept it
    if (n_cols > (uint32_t) INT32_MAX) {
        LLAMA_LOG_ERROR("%s: a decode width of %u columns is too wide to report\n", __func__, n_cols);
        return false;
    }

    if (!llama_exact_width_within_explicit_bound(n_cols)) {
        return false;
    }

    std::lock_guard<std::recursive_mutex> lock(g_exact_mutex);

    uint32_t cur = g_exact_decode_width.load(std::memory_order_relaxed);

    while (n_cols > cur && !g_exact_decode_width.compare_exchange_weak(cur, n_cols, std::memory_order_relaxed)) {
    }

    const uint32_t widest = g_exact_decode_width.load(std::memory_order_relaxed);

    for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
        ggml_backend_reg_t reg = ggml_backend_reg_get(i);

        auto * fn = (void (*)(int)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_set_exact_decode_width");
        if (fn) {
            fn((int) widest);
        }
    }

    return true;
}

uint32_t llama_exact_decode_width(void) {
    return g_exact_decode_width.load(std::memory_order_relaxed);
}
