// [TAG_EXACT_CONCURRENCY] a decision request whose questions share a prompt prefix is grouped so the
// children continue from the parent's cells through seq_cp. Exact mode gives a page to one sequence
// and refuses that copy, so the grouping must not happen there: every task keeps its whole prompt.

#include "server-decision.h"

#include <cstdio>
#include <cstdlib>

#undef NDEBUG
#include <cassert>

static server_task make_task(int id, const llama_tokens & tokens) {
    server_task task(SERVER_TASK_TYPE_DECISION);
    task.id     = id;
    task.tokens = server_tokens(tokens, false);
    return task;
}

int main() {
    // latched on first use, so set before anything reads it
#ifdef _WIN32
    _putenv_s("LLAMA_EXACT_CONCURRENCY", "1");
#else
    setenv("LLAMA_EXACT_CONCURRENCY", "1", 1);
#endif
    assert(common_exact_concurrency());

    std::vector<server_task> tasks;
    tasks.push_back(make_task(0, { 1, 2, 3, 4, 5 }));
    tasks.push_back(make_task(1, { 1, 2, 3, 4, 6 }));
    tasks.push_back(make_task(2, { 1, 2, 3, 7 }));

    const auto groups = server_decision_group_tasks(std::move(tasks), 4);

    assert(groups.size() == 3);
    for (size_t i = 0; i < groups.size(); ++i) {
        assert(groups[i].id == (int) i);
        assert(groups[i].n_tokens_shared == 0);
        assert(groups[i].child_tasks.empty());
        assert(groups[i].id_parent == -1);
    }

    printf("ok: no shared-prompt grouping under LLAMA_EXACT_CONCURRENCY\n");
    return 0;
}
