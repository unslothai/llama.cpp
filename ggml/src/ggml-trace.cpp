#include "ggml-trace.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include <atomic>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#   include <process.h>
#   define GGML_TRACE_GETPID() _getpid()
#else
#   include <unistd.h>
#   define GGML_TRACE_GETPID() getpid()
#endif

int ggml_trace_flag = 0;

namespace {

struct trace_state {
    std::mutex  mtx;
    FILE *      f    = nullptr;
    std::string role;
    int64_t     t_open = 0;

    ~trace_state() {
        if (f) {
            fflush(f);
            fclose(f);
            f = nullptr;
        }
    }
};

trace_state & state() {
    static trace_state s;
    return s;
}

std::atomic<int> g_next_tid{0};

thread_local int          tls_tid     = -1;
thread_local int          tls_group   = -1;
thread_local const char * tls_subject = nullptr;
thread_local uint64_t     tls_uid     = 0;
thread_local std::string  tls_line;

// one line is built in the calling thread and handed to the file under the lock
void emit(const std::string & line) {
    trace_state & s = state();

    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.f == nullptr) {
        return;
    }
    fwrite(line.data(), 1, line.size(), s.f);
}

void append_escaped(std::string & out, const char * src) {
    if (src == nullptr) {
        return;
    }
    for (const char * p = src; *p; ++p) {
        const unsigned char c = (unsigned char) *p;
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
}

void append_i64(std::string & out, int64_t v) {
    char buf[24];
    snprintf(buf, sizeof(buf), "%lld", (long long) v);
    out += buf;
}

} // namespace

int64_t ggml_trace_time_us(void) {
    return ggml_time_us();
}

int ggml_trace_tid(void) {
    if (tls_tid < 0) {
        tls_tid = g_next_tid.fetch_add(1);
    }
    return tls_tid;
}

void ggml_trace_set_group(int group) {
    tls_group = group;
}

int ggml_trace_get_group(void) {
    return tls_group;
}

void ggml_trace_set_subject(const char * name, uint64_t uid) {
    tls_subject = name;
    tls_uid     = uid;
}

const char * ggml_trace_escape(char * dst, size_t dst_size, const char * src) {
    if (dst == nullptr || dst_size == 0) {
        return dst;
    }
    std::string tmp;
    append_escaped(tmp, src);
    const size_t n = tmp.size() < dst_size - 1 ? tmp.size() : dst_size - 1;
    memcpy(dst, tmp.data(), n);
    dst[n] = '\0';
    return dst;
}

int ggml_trace_open(const char * path, const char * role) {
    trace_state & s = state();

    std::lock_guard<std::mutex> lock(s.mtx);
    if (s.f != nullptr) {
        return 1;
    }
    if (path == nullptr || path[0] == '\0') {
        path = getenv("GGML_RPC_TRACE");
    }
    if (path == nullptr || path[0] == '\0') {
        return 0;
    }

    s.f = fopen(path, "wb");
    if (s.f == nullptr) {
        GGML_LOG_ERROR("%s: cannot open trace file %s\n", __func__, path);
        return 0;
    }
    s.role   = role != nullptr ? role : "unknown";
    s.t_open = ggml_time_us();

    char host[256] = "";
#ifndef _WIN32
    if (gethostname(host, sizeof(host) - 1) != 0) {
        host[0] = '\0';
    }
#endif

    std::string line = "{\"header\":1,\"role\":\"";
    append_escaped(line, s.role.c_str());
    line += "\",\"host\":\"";
    append_escaped(line, host);
    line += "\",\"pid\":";
    append_i64(line, GGML_TRACE_GETPID());
    line += ",\"t_open_us\":";
    append_i64(line, s.t_open);
    line += ",\"wall_us\":";
    // CLOCK_REALTIME at the same instant, only a sanity check for the merge tool
    {
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);
        append_i64(line, (int64_t) ts.tv_sec * 1000000 + ts.tv_nsec / 1000);
    }
    line += "}\n";

    fwrite(line.data(), 1, line.size(), s.f);
    fflush(s.f);

    ggml_trace_flag = 1;

    GGML_LOG_INFO("%s: tracing to %s (role %s)\n", __func__, path, s.role.c_str());
    return 1;
}

void ggml_trace_close(void) {
    trace_state & s = state();

    std::lock_guard<std::mutex> lock(s.mtx);
    ggml_trace_flag = 0;
    if (s.f != nullptr) {
        fflush(s.f);
        fclose(s.f);
        s.f = nullptr;
    }
}

void ggml_trace_clock_offset(const char * peer, int64_t t1, int64_t t2, int64_t t3, int64_t t4) {
    if (!ggml_trace_flag) {
        return;
    }
    // NTP style: the peer clock is ahead of ours by offset, the round trip is delay
    const int64_t offset = ((t2 - t1) + (t3 - t4)) / 2;
    const int64_t delay  = (t4 - t1) - (t3 - t2);

    std::string line = "{\"clock_offset\":1,\"peer\":\"";
    append_escaped(line, peer);
    line += "\",\"t1\":";      append_i64(line, t1);
    line += ",\"t2\":";        append_i64(line, t2);
    line += ",\"t3\":";        append_i64(line, t3);
    line += ",\"t4\":";        append_i64(line, t4);
    line += ",\"offset_us\":"; append_i64(line, offset);
    line += ",\"delay_us\":";  append_i64(line, delay);
    line += "}\n";

    emit(line);
}

void ggml_trace_event(const char * phase, const char * name, int64_t t0, int64_t t1, const char * fields) {
    if (!ggml_trace_flag) {
        return;
    }

    std::string & line = tls_line;
    line.clear();
    line += "{\"ph\":\"";
    append_escaped(line, phase);
    line += "\",\"n\":\"";
    append_escaped(line, name);
    line += "\",\"t0\":";
    append_i64(line, t0);
    line += ",\"t1\":";
    append_i64(line, t1);
    line += ",\"tid\":";
    append_i64(line, ggml_trace_tid());
    if (tls_group >= 0) {
        line += ",\"grp\":";
        append_i64(line, tls_group);
    }
    if (tls_subject != nullptr) {
        line += ",\"subj\":\"";
        append_escaped(line, tls_subject);
        line += "\"";
    }
    if (tls_uid != 0) {
        line += ",\"uid\":";
        append_i64(line, (int64_t) tls_uid);
    }
    if (fields != nullptr && fields[0] != '\0') {
        line += ",";
        line += fields;
    }
    line += "}\n";

    emit(line);
}

void ggml_trace_eventf(const char * phase, const char * name, int64_t t0, int64_t t1, const char * fmt, ...) {
    if (!ggml_trace_flag) {
        return;
    }

    char fields[1024];
    fields[0] = '\0';
    if (fmt != nullptr) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(fields, sizeof(fields), fmt, args);
        va_end(args);
    }

    ggml_trace_event(phase, name, t0, t1, fields);
}

// -----------------------------------------------------------------------------
// GPU spans
//
// The timing hooks are looked up once through the backend registry, so ggml-base does not have
// to link against any GPU runtime. A backend that does not offer them simply has no GPU rows in
// the trace.
// -----------------------------------------------------------------------------

typedef void (*ggml_trace_gpu_mark_t)(ggml_backend_t backend, uint64_t tag, int kind);
typedef int  (*ggml_trace_gpu_poll_t)(uint64_t * tags, int * kinds, int64_t * t_us, int max);

namespace {

struct trace_gpu_state {
    std::mutex                                mutex;
    bool                                      probed = false;
    ggml_trace_gpu_mark_t                     mark   = nullptr;
    ggml_trace_gpu_poll_t                     poll   = nullptr;
    std::unordered_map<uint64_t, int64_t>     starts;
    std::unordered_map<uint64_t, std::string> names;
    uint64_t                                  next_tag = 1;
};

trace_gpu_state & gpu() {
    static trace_gpu_state s;
    return s;
}

// caller holds gpu().mutex
void gpu_probe(ggml_backend_t backend) {
    trace_gpu_state & st = gpu();
    if (st.probed) {
        return;
    }
    st.probed = true;

    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr) {
        return;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    if (reg == nullptr) {
        return;
    }
    st.mark = (ggml_trace_gpu_mark_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_trace_mark");
    st.poll = (ggml_trace_gpu_poll_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_trace_poll");
    if (st.mark == nullptr) {
        GGML_LOG_INFO("ggml_trace: %s has no GPU timing hook, graph events will be host side only\n",
                      ggml_backend_dev_name(dev));
    }
}

} // namespace

uint64_t ggml_trace_gpu_begin(ggml_backend_t backend, const char * name) {
    trace_gpu_state & st = gpu();

    std::lock_guard<std::mutex> lock(st.mutex);
    gpu_probe(backend);
    if (st.mark == nullptr) {
        return 0;
    }
    const uint64_t tag = st.next_tag++;
    st.names[tag] = name != nullptr ? name : "gpu";
    st.mark(backend, tag, 0);
    return tag;
}

void ggml_trace_gpu_end(ggml_backend_t backend, uint64_t tag) {
    trace_gpu_state & st = gpu();

    std::lock_guard<std::mutex> lock(st.mutex);
    if (st.mark == nullptr || tag == 0) {
        return;
    }
    st.mark(backend, tag, 1);
}

void ggml_trace_gpu_flush(void) {
    trace_gpu_state & st = gpu();

    std::vector<std::string> names;
    std::vector<uint64_t>    tags;
    std::vector<int64_t>     t0s;
    std::vector<int64_t>     t1s;
    {
        std::lock_guard<std::mutex> lock(st.mutex);
        if (st.poll == nullptr) {
            return;
        }
        uint64_t tag_buf[64];
        int      kind_buf[64];
        int64_t  t_buf[64];
        int n = 0;
        do {
            n = st.poll(tag_buf, kind_buf, t_buf, 64);
            for (int i = 0; i < n; i++) {
                if (kind_buf[i] == 0) {
                    st.starts[tag_buf[i]] = t_buf[i];
                    continue;
                }
                auto it = st.starts.find(tag_buf[i]);
                if (it == st.starts.end()) {
                    continue;
                }
                auto nit = st.names.find(tag_buf[i]);
                names.push_back(nit != st.names.end() ? nit->second : std::string("gpu"));
                tags .push_back(tag_buf[i]);
                t0s  .push_back(it->second);
                t1s  .push_back(t_buf[i]);
                st.starts.erase(it);
                if (nit != st.names.end()) {
                    st.names.erase(nit);
                }
            }
        } while (n == 64);
    }

    for (size_t i = 0; i < names.size(); i++) {
        ggml_trace_eventf("gpu", names[i].c_str(), t0s[i], t1s[i],
                          "\"gpu_tag\":%llu", (unsigned long long) tags[i]);
    }
}
