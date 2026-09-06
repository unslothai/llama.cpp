// Event tracer for the RPC backend and llama.cpp.
//
// The tracer is off unless the environment variable GGML_RPC_TRACE is set to a file path (the
// rpc-server also accepts --trace <path>). When it is off the only cost at a call site is one
// load and one branch on ggml_trace_flag; no clock is read and nothing is formatted.
//
// Every process writes JSON lines to its own file: one header object, then one object per event.
// scripts/rpc_trace/merge.py aligns the files of the two nodes with the clock offset measured by
// the client at connect time and emits a Chrome trace plus a per decode step summary.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <stdarg.h>
#include <stdint.h>

#ifdef  __cplusplus
extern "C" {
#endif

    // 1 while a trace file is open. Read it directly at the call sites, do not call a function:
    //
    //     const int64_t t0 = ggml_trace_flag ? ggml_time_us() : 0;
    //
    GGML_API int ggml_trace_flag;

    // Opens the trace file. `path` may be NULL, then GGML_RPC_TRACE is used. `role` names the
    // process in the merged trace ("client", "rpc-server"). Safe to call more than once, the
    // first call with a usable path wins. Returns 1 if tracing is on afterwards.
    GGML_API int  ggml_trace_open(const char * path, const char * role);
    GGML_API void ggml_trace_close(void);

    // Monotonic microseconds in the same clock as every timestamp written to the trace.
    GGML_API int64_t ggml_trace_time_us(void);

    // Stable small integer for the calling thread, used as the Chrome trace tid.
    GGML_API int ggml_trace_tid(void);

    // llama-server tags the threads that drive one pipeline group so that every event raised
    // underneath, down to the individual RPC command, carries the group id. -1 means no group.
    GGML_API void ggml_trace_set_group(int group);
    GGML_API int  ggml_trace_get_group(void);

    // Names the tensor or graph the next RPC commands of this thread belong to. The pointer must
    // stay valid until it is replaced; the RPC backend passes tensor->name.
    GGML_API void ggml_trace_set_subject(const char * name, uint64_t uid);

    // Writes one event. `phase` is a free form category ("rpc.client", "sched", ...), `name` the
    // event name. t0/t1 are microseconds from ggml_trace_time_us(); t1 == t0 makes it an instant.
    // `fields` is appended verbatim inside the JSON object and may be NULL, for example
    //     "\"bytes\":128,\"cmd\":\"GRAPH_COMPUTE\""
    GGML_API void ggml_trace_event(const char * phase, const char * name,
                                   int64_t t0, int64_t t1, const char * fields);

    // Same, with printf formatting for the extra fields.
    GGML_API void ggml_trace_eventf(const char * phase, const char * name,
                                    int64_t t0, int64_t t1, const char * fmt, ...);

    // Records the result of the four timestamp exchange with the peer into the header of this
    // trace file, so the merge tool can put both nodes on one time line.
    //   t1 client sends, t2 peer receives, t3 peer replies, t4 client receives (microseconds)
    GGML_API void ggml_trace_clock_offset(const char * peer, int64_t t1, int64_t t2, int64_t t3, int64_t t4);

    // GPU spans.
    //
    // A backend submit only queues the work, so host timestamps around it say nothing about when
    // the kernels ran. If the backend exposes the timing hooks through its registry (CUDA does),
    // these bracket the submit with two events on the compute stream and report them later on the
    // host monotonic scale. ggml_trace_gpu_begin returns a tag, 0 if the backend cannot do it.
    // Nothing here ever waits on the GPU: the spans are emitted by ggml_trace_gpu_flush once both
    // of their events have completed, which the caller does from a point where it is idle anyway.
    GGML_API uint64_t ggml_trace_gpu_begin(ggml_backend_t backend, const char * name);
    GGML_API void     ggml_trace_gpu_end  (ggml_backend_t backend, uint64_t tag);
    GGML_API void     ggml_trace_gpu_flush(void);

    // Escapes a string for the JSON output. Returns `dst`.
    GGML_API const char * ggml_trace_escape(char * dst, size_t dst_size, const char * src);

#ifdef  __cplusplus
}
#endif
