// [TAG_PREEMPT] Which tasks a park may pick. Every victim and rotation selector asks server_task::preempt_resumable():
// a TTS or realtime transcription slot feeds its own decode loop, so a resume could not replay it as a prompt.

#include "server-task.h"

#include <cstdio>

#undef NDEBUG
#include <cassert>

static server_task make_task(server_task_type type) {
    server_task task(type);
    task.id = 1;
    return task;
}

int main() {
    // a token prompt is replayed on resume
    assert(make_task(SERVER_TASK_TYPE_COMPLETION).preempt_resumable());
    assert(make_task(SERVER_TASK_TYPE_INFILL).preempt_resumable());
    assert(make_task(SERVER_TASK_TYPE_EMBEDDING).preempt_resumable());
    assert(make_task(SERVER_TASK_TYPE_RERANK).preempt_resumable());
    assert(make_task(SERVER_TASK_TYPE_DECISION).preempt_resumable());

    // their rows come from the vocoder loop or the audio stream, not from a prompt
    assert(!make_task(SERVER_TASK_TYPE_TTS).preempt_resumable());
    assert(!make_task(SERVER_TASK_TYPE_TRANSCRIBE_RT).preempt_resumable());

    // n_cmpl > 1 slots share one sequence
    server_task parent = make_task(SERVER_TASK_TYPE_COMPLETION);
    parent.add_child(parent.id, 2);
    assert(parent.is_parent());
    assert(!parent.preempt_resumable());
    assert(parent.child_tasks[0].is_child());
    assert(!parent.child_tasks[0].preempt_resumable());

    printf("ok: TTS, realtime transcription and n_cmpl > 1 tasks are never picked to park\n");
    return 0;
}
