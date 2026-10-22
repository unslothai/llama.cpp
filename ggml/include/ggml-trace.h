// RPC/llama event tracer, on when GGML_RPC_TRACE (or rpc-server --trace) names a file.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdarg.h>
#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

    GGML_API int ggml_trace_flag;

    // NULL path = GGML_RPC_TRACE; first usable path wins
    GGML_API int  ggml_trace_open(const char * path, const char * role);
    GGML_API void ggml_trace_close(void);
    GGML_API void ggml_trace_flush(void);

    GGML_API int64_t ggml_trace_time_us(void);

    GGML_API int ggml_trace_tid(void);

    GGML_API void ggml_trace_set_group(int group);
    GGML_API int  ggml_trace_get_group(void);

    GGML_API void ggml_trace_set_subject(const char * name, uint64_t uid);
    GGML_API const char * ggml_trace_get_subject(uint64_t * uid);

    GGML_API void ggml_trace_event(const char * phase, const char * name,
                                   int64_t t0, int64_t t1, const char * fields);

    GGML_API void ggml_trace_eventf(const char * phase, const char * name,
                                    int64_t t0, int64_t t1, const char * fmt, ...);

    // t1 client sends, t2 peer receives, t3 peer replies, t4 client receives (microseconds).
    GGML_API void ggml_trace_clock_offset(const char * peer, int64_t t1, int64_t t2, int64_t t3, int64_t t4);

    // begin returns 0 without hooks; never waits
    GGML_API uint64_t ggml_trace_gpu_begin(ggml_backend_t backend, const char * name);
    GGML_API void     ggml_trace_gpu_end  (ggml_backend_t backend, uint64_t tag);
    GGML_API void     ggml_trace_gpu_flush(void);

    GGML_API const char * ggml_trace_escape(char * dst, size_t dst_size, const char * src);

#ifdef  __cplusplus
}
#endif
