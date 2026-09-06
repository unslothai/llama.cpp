#include "llama-impl.h"

#include "ggml-backend.h"
#include "gguf.h"
#include "llama.h"

#include <cinttypes>
#include <climits>
#include <atomic>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <sstream>

struct llama_logger_state {
    ggml_log_callback log_callback = llama_log_callback_default;
    void * log_callback_user_data = nullptr;
};

static llama_logger_state g_logger_state;

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

static std::string gguf_data_to_str(enum gguf_type type, const void * data, int i) {
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

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i) {
    const enum gguf_type type = gguf_get_kv_type(ctx_gguf, i);

    switch (type) {
        case GGUF_TYPE_STRING:
            return gguf_get_val_str(ctx_gguf, i);
        case GGUF_TYPE_ARRAY:
            {
                const enum gguf_type arr_type = gguf_get_arr_type(ctx_gguf, i);
                int arr_n = gguf_get_arr_n(ctx_gguf, i);
                const void * data = arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx_gguf, i);
                std::stringstream ss;
                ss << "[";
                for (int j = 0; j < arr_n; j++) {
                    if (arr_type == GGUF_TYPE_STRING) {
                        std::string val = gguf_get_arr_str(ctx_gguf, i, j);
                        // escape quotes
                        replace_all(val, "\\", "\\\\");
                        replace_all(val, "\"", "\\\"");
                        ss << '"' << val << '"';
                    } else if (arr_type == GGUF_TYPE_ARRAY) {
                        ss << "???";
                    } else {
                        ss << gguf_data_to_str(arr_type, data, j);
                    }
                    if (j < arr_n - 1) {
                        ss << ", ";
                    }
                }
                ss << "]";
                return ss.str();
            }
        default:
            return gguf_data_to_str(type, gguf_get_val_data(ctx_gguf, i), 0);
    }
}

// [TAG_EXACT_CONCURRENCY]
bool llama_exact_concurrency() {
    static const bool enabled = []() {
        const char * val = getenv("LLAMA_EXACT_CONCURRENCY");
        return val && atoi(val) != 0;
    }();

    return enabled;
}

// [TAG_EXACT_CONCURRENCY] tokens one sequence contributes to a decode step, see llama.h
static std::atomic<uint32_t> g_exact_decode_tokens{1};

// the most sequences any context so far was created with. The tokens figure is process
// wide, so raising it widens the decode step of every context that already exists; the
// width those contexts reported at creation is re-reported here with the new figure, or a
// context created under a narrower figure would batch above the bound it reported.
static std::atomic<uint32_t> g_exact_max_n_seq{0};

bool llama_exact_report_n_seq(uint32_t n_seq) {
    const uint32_t n_seq_max = std::max(n_seq, g_exact_max_n_seq.load(std::memory_order_relaxed));

    if (!llama_set_exact_decode_width(n_seq_max * llama_exact_decode_tokens())) {
        return false;
    }

    uint32_t cur = g_exact_max_n_seq.load(std::memory_order_relaxed);

    while (n_seq > cur && !g_exact_max_n_seq.compare_exchange_weak(cur, n_seq, std::memory_order_relaxed)) {
    }

    return true;
}

bool llama_set_exact_decode_tokens(uint32_t n_tokens) {
    n_tokens = n_tokens > 0 ? n_tokens : 1;

    // every context that exists widens with the figure, so the width they will need is
    // reported first; a figure the explicit bound cannot cover leaves the old one in place
    const uint32_t n_seq = g_exact_max_n_seq.load(std::memory_order_relaxed);

    if (n_seq > 0 && !llama_set_exact_decode_width(n_seq * n_tokens)) {
        return false;
    }

    g_exact_decode_tokens.store(n_tokens, std::memory_order_relaxed);

    return true;
}

uint32_t llama_exact_decode_tokens(void) {
    return g_exact_decode_tokens.load(std::memory_order_relaxed);
}

// [TAG_EXACT_CONCURRENCY] the widest decode ubatch reported so far, see llama.h. A backend that
// splits columns to make a decode exact reads it through ggml_backend_cuda_set_exact_decode_width,
// reached through the registry so that a backend that is absent or loaded late costs nothing.
static std::atomic<uint32_t> g_exact_decode_width{0};

// an explicit column bound given to the CUDA backend wins over the reported width there, so a
// width above it would leave decodes batched past the bound with the mode still reporting itself
// on; a width the bound does not cover is refused instead of stored
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
    if (!llama_exact_width_within_explicit_bound(n_cols)) {
        return false;
    }

    uint32_t cur = g_exact_decode_width.load(std::memory_order_relaxed);

    while (n_cols > cur && !g_exact_decode_width.compare_exchange_weak(cur, n_cols, std::memory_order_relaxed)) {
    }

    // The widest figure so far goes to every backend on every call, not only when it grew: a
    // width reported before a backend was loaded would otherwise never reach it, and every
    // context reports at creation, by which time the backends are there.
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
