import os
import time
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool needs --kv-unified and more than one slot.

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    os.environ.pop("LLAMA_SERVER_PREEMPT_EVERY", None)
    os.environ.pop("LLAMA_SERVER_PREEMPT_PLANNER", None)
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)
    os.environ.pop("LLAMA_ARG_PREEMPT_ASYNC", None)


def _complete(n_predict: int, prompt: str = "Hi how are you"):
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "ignore_eos": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
    })
    return res


def test_forced_preemption_does_not_change_the_output():
    # With one request the batch shape is fixed, so any output difference is the preemption's fault.
    global server
    server.n_ctx = 512
    server.start()
    reference = _complete(64)
    assert reference.status_code == 200
    assert reference.body["timings"]["predicted_n"] == 64
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_EVERY = 8" in log.drain()

    preempted = _complete(64)
    assert preempted.status_code == 200
    assert preempted.body["timings"]["predicted_n"] == 64

    text = log.drain()
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6

    assert preempted.body["content"] == reference.body["content"]
    assert preempted.body["tokens"] == reference.body["tokens"]


def test_two_slots_that_overflow_the_pool_together_both_finish():
    # Each request fits the 256-cell pool alone (8 + 160) but not together, so one is parked and resumed.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict



_WORDS = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. "
) * 4


def _prompt_of_about(n_tokens: int, salt: str = "") -> tuple[str, int]:
    """A prompt whose token count is in [n_tokens - 12, n_tokens], measured on the server."""
    words = (salt + " " + _WORDS).split()
    while words:
        text = " ".join(words)
        res = server.make_request("POST", "/tokenize", data={"content": text})
        assert res.status_code == 200
        n = len(res.body["tokens"])
        if n <= n_tokens:
            assert n >= n_tokens - 12, f"could not land near {n_tokens} tokens, got {n}"
            return text, n
        words = words[: len(words) - max(1, (n - n_tokens) // 8)]
    raise AssertionError("empty prompt")


def test_two_prompts_that_overflow_the_pool_together_both_finish():
    # Both slots are still processing their prompts when the pool fills, and one is parked between chunks.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_a, n_a = _prompt_of_about(150, "Alpha")
    prompt_b, n_b = _prompt_of_about(150, "Bravo")
    n_predict = 16
    assert n_a + n_predict <= 256 and n_b + n_predict <= 256
    assert n_a + n_b + 2 * n_predict > 256

    results = parallel_function_calls([
        (_complete, (n_predict, prompt_a)),
        (_complete, (n_predict, prompt_b)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert len(res.body["tokens"]) == n_predict


def test_a_generating_slot_and_a_large_prompt_both_finish():
    # A long generation collides with a large prompt; the second request is sent right away so they overlap.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    # Shorter generations finished before the two collided.
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a <= 256 and n_b + n_predict_b <= 256
    assert 8 + n_predict_a + n_b + n_predict_b > 256

    def _late(n_predict, prompt):
        time.sleep(0.02)
        return _complete(n_predict, prompt)

    results = parallel_function_calls([
        (_complete, (n_predict_a, "Hi how are you")),
        (_late, (n_predict_b, prompt_b)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text

    assert results[0].status_code == 200
    assert results[0].body["timings"]["predicted_n"] == n_predict_a
    assert results[1].status_code == 200
    assert results[1].body["timings"]["predicted_n"] == n_predict_b


def test_preempt_ram_zero_disables_preemption():
    # --preempt-ram 0 restores the old behaviour: the KV-full path ends the requests.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "preempted:" not in text
    assert "Context size has been exceeded" in text
    assert any(res.status_code != 200 for res in results)


def test_metrics_and_slots_report_the_parked_state():
    global server
    server.n_ctx = 256
    server.server_metrics = True
    server.start()

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["n_preempt"] == 0

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])
    for res in results:
        assert res.status_code == 200

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    metrics = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ", 1)
            metrics[name[len("llamacpp:"):]] = float(value)
    assert metrics["n_preempt_total"] >= 1
    assert metrics["n_resume_total"] == metrics["n_preempt_total"]
    assert metrics["requests_preempted"] == 0
    assert metrics["preempt_ram_bytes"] == 0

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert sum(slot["n_preempt"] for slot in res.body) == 0, "n_preempt is per task and resets with the slot"


# [TAG_PREEMPT_ASYNC] parking and resuming on a stream of their own
#
# The copies are only asynchronous on a backend that can copy asynchronously and signal an
# event, which today means a GPU one. On a CPU-only build the server says so and falls back
# to the synchronous path, and the tests below that need the asynchronous one skip.

_ASYNC_BANNER = "parking and resuming asynchronously"


def _start_async(**kwargs) -> str:
    """Start the server with the asynchronous path asked for, and return its log so far."""
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()
    with open(server.log_path) as f:
        return f.read()


def _require_async(text: str):
    if _ASYNC_BANNER not in text:
        pytest.skip("this backend cannot copy asynchronously, the async park path is not exercised")


def test_async_preemption_does_not_change_the_output():
    # The same question the synchronous determinism test asks, of the asynchronous path:
    # with one request the batch has the same shape at every step, so a continuation that
    # was parked and resumed through a transfer and is not byte-identical to an
    # uninterrupted one is the transfer's fault and nothing else's.
    global server
    server.n_ctx = 512
    server.n_gpu_layer = 99
    text = _start_async()
    _require_async(text)

    res_plain = _complete(64)
    assert res_plain.status_code == 200

    server.stop()
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    log = LogReader(server.log_path)

    res_preempted = _complete(64)
    assert res_preempted.status_code == 200

    text = log.drain()
    _require_async(text)
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6
    # the asynchronous path is the one that ran, not the synchronous fallback: only it
    # splits a park and a resume into an issue and a completion
    assert "park completed after" in text
    assert "restore issued in" in text
    assert "restore completed after" in text

    assert res_preempted.body["content"] == res_plain.body["content"]
    assert res_preempted.body["tokens"] == res_plain.body["tokens"]


def test_async_preemption_under_load_keeps_every_slot_and_its_output():
    # Two requests that do not fit the pool together, parked and resumed asynchronously
    # while the other one keeps decoding. Every slot must finish, and finish with exactly
    # the tokens it produces when it has the pool to itself.
    global server
    server.n_ctx = 256
    server.n_gpu_layer = 99
    text = _start_async()
    _require_async(text)

    n_predict = 160
    prompts = [
        "Once upon a time there was a brave knight who",
        "The quick brown fox jumps over the lazy dog and",
    ]

    # each one alone, for the reference tokens
    alone = [_complete(n_predict, prompt) for prompt in prompts]
    for res in alone:
        assert res.status_code == 200

    server.stop()
    server.start()
    log = LogReader(server.log_path)

    together = parallel_function_calls([(_complete, (n_predict, prompt)) for prompt in prompts])

    text = log.drain()
    _require_async(text)
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    for res, ref in zip(together, alone):
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert res.body["tokens"] == ref.body["tokens"]


def _cancel_soon(n_predict: int, prompt: str, timeout: float):
    try:
        server.make_request("POST", "/completion", data={
            "n_predict": n_predict,
            "prompt": prompt,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 42,
        }, timeout=timeout)
    except Exception:
        pass  # the point is the drop, not the response


def test_cancel_while_a_copy_is_in_flight_frees_the_slot():
    # A cancelled request can reach release() with a park or a resume still running, which
    # is where the host buffer is freed and the cells are handed on. Both have to wait for
    # the copy first. LLAMA_SERVER_PREEMPT_EVERY keeps every slot cycling between the two
    # states, so cancelling at a spread of moments lands in both; what is asserted is that
    # the server survives it, the slots come back, and it still answers correctly.
    global server
    server.n_ctx = 512
    server.n_gpu_layer = 99
    # every 8 tokens, so a slot spends most of its life in one of the two copy states, but
    # not so often that the abandoned requests take minutes to drain
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    text = _start_async()
    _require_async(text)

    for i in range(4):
        _cancel_soon(96, "Once upon a time there was a brave knight who", 0.05 + 0.1 * i)

    # every slot back, and none of them still holding a parked sequence
    deadline = time.time() + 120
    while time.time() < deadline:
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        if all(not slot["is_processing"] for slot in res.body):
            break
        time.sleep(0.2)
    else:
        pytest.fail("a slot never came back after a cancel during a copy")

    for slot in res.body:
        assert slot["is_preempted"] is False

    if server.server_metrics:
        res = server.make_request("GET", "/metrics")
        for line in res.body.splitlines():
            if line.startswith("llamacpp:preempt_ram_bytes"):
                assert float(line.split(" ", 1)[1]) == 0, "a cancelled slot kept its parked memory"

    # and the server still works
    res = _complete(16)
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 16


def test_no_preempt_async_falls_back_to_the_synchronous_path():
    # The flag has to really switch it off, so that the two can be compared on one binary.
    global server
    server.n_ctx = 512
    server.n_gpu_layer = 99
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    log = LogReader(server.log_path)

    res = _complete(64)
    assert res.status_code == 200

    text = log.drain()
    assert _ASYNC_BANNER not in text
    assert "park issued in" not in text
    assert "restore issued in" not in text
    # the synchronous path still parks and resumes
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6


def test_a_prompt_arriving_into_a_nearly_full_pool_parks_rather_than_ends_everything():
    # [TAG_PREEMPT_ASYNC] The case the async path made worse than the synchronous one, and
    # that the existing tests miss because their victim holds almost no cells.
    #
    # Three slots are well into generating when a fourth request arrives whose prompt does
    # not fit in what is left. update_preemption() picks a victim and issues its park, but
    # an asynchronous park does not return the cells before update_slots() carries on. If
    # the loop leaves at that point, the batch is built into a pool that has not got any
    # smaller, llama_decode returns 1, and the retry ladder halves n_batch to 1 in
    # microseconds without ever polling the copy -- ending every request with "Context size
    # has been exceeded" while the room it wanted was one event query away.
    #
    # Pass is what the synchronous path gave: a park, and all four requests finish.
    global server
    server.n_ctx = 512
    server.n_gpu_layer = 99
    server.n_slots = 4
    server.start()
    log = LogReader(server.log_path)

    prompt_a, n_a = _prompt_of_about(100, "Alpha")
    prompt_b, n_b = _prompt_of_about(100, "Bravo")
    prompt_c, n_c = _prompt_of_about(100, "Charlie")
    prompt_d, n_d = _prompt_of_about(150, "Delta")

    # A, B and C oversubscribe the pool between them, so the pressure does not depend on
    # when D arrives, and every occupant is holding real cells rather than the handful the
    # other tests park. Each of the four still fits on its own.
    n_predict_abc = 130
    n_predict_d = 40
    assert max(n_a, n_b, n_c) + n_predict_abc < 512 and n_d + n_predict_d < 512
    assert n_a + n_b + n_c + 3 * n_predict_abc > 512

    def _late(n_predict, prompt):
        # D's prompt arrives into a pool the other three have already grown into; this
        # model decodes about 120 tokens a second, so they are all still running
        time.sleep(0.25)
        return _complete(n_predict, prompt)

    results = parallel_function_calls([
        (_complete, (n_predict_abc, prompt_a)),
        (_complete, (n_predict_abc, prompt_b)),
        (_complete, (n_predict_abc, prompt_c)),
        (_late,     (n_predict_d, prompt_d)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted" in text

    for i, res in enumerate(results):
        assert res.status_code == 200, (i, res.body)
    for i in range(3):
        assert results[i].body["timings"]["predicted_n"] == n_predict_abc
    assert results[3].body["timings"]["predicted_n"] == n_predict_d


def test_two_prompts_near_the_context_size_both_complete():
    # A prompt parked before taking cells and too close to n_ctx for the margin must still resume once the pool is empty.
    global server
    server.n_ctx = 256
    server.n_batch = 256
    server.start()
    log = LogReader(server.log_path)

    base = server.make_request("POST", "/tokenize", data={"content": "Once upon a time there was a little girl"}).body["tokens"]
    long_prompt = (base * 64)[:250]
    n_predict = 4
    together = parallel_function_calls([(_complete, (n_predict, long_prompt)) for _ in range(2)])

    text = log.drain()
    assert "cannot fit the pool" not in text

    for res in together:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_last_resort_parks_instead_of_ending_everyone():
    # With the planner off, the last resort parks the smaller slot instead of failing every slot.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_PLANNER = off" in log.drain()

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" not in text, "the planner was off, nothing may be parked ahead of the decode"
    assert "preempted as a last resort" in text
    assert "last resort: batch given up" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict


def test_the_last_resort_works_with_an_unlimited_budget():
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted as a last resort" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_last_resort_rewinds_a_prompt_in_flight():
    # The chunk that failed is processed again after the resume, neither skipped nor fed twice.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a <= 256 and n_b + n_predict_b <= 256
    assert 8 + n_predict_a + n_b + n_predict_b > 256

    def _late(n_predict, prompt):
        time.sleep(0.02)
        return _complete(n_predict, prompt)

    results = parallel_function_calls([
        (_complete, (n_predict_a, "Hi how are you")),
        (_late, (n_predict_b, prompt_b)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted as a last resort" in text

    assert results[0].status_code == 200
    assert results[0].body["timings"]["predicted_n"] == n_predict_a
    assert results[1].status_code == 200
    assert results[1].body["timings"]["predicted_n"] == n_predict_b
    # The count is the prompt plus the BOS the server adds.
    assert results[1].body["timings"]["prompt_n"] == n_b + 1


def test_a_resident_cycling_through_context_shifts_takes_turns_with_a_parked_head():
    # A resident cycling through context shifts is rotated out for the parked head, and both finish.
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    log = LogReader(server.log_path)

    n_predict = 12000
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "slot context shift" in text
    assert "rotated out after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_rotation_parks_a_resident_that_lets_the_head_in():
    # Two shifting residents and a parked third must all finish their tokens.
    global server
    server.n_slots = 3
    server.n_ctx = 384
    server.enable_ctx_shift = True
    server.start()
    n_predict = 9000
    prompts = [
        "Once upon a time there was a brave knight who",
        "The quick brown fox jumps over the lazy dog and",
        "In a small village by the sea there lived a fisherman who",
    ]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": p, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for p in prompts
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "rotated out after" in text
    assert "Context size has been exceeded" not in text


def test_a_parent_and_child_that_do_not_fit_alone_get_the_context_error_and_the_server_lives():
    # An n_cmpl=2 family that does not fit has no victim, so it gets the context error and the server keeps serving.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "n_predict": 160,
        "n_cmpl": 2,
        "prompt": "Once upon a time there was a brave knight who",
        "ignore_eos": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
    })
    assert res.status_code == 500
    assert "Context size has been exceeded" in res.body["error"]["message"]

    text = log.drain()
    assert "preempted as a last resort" not in text, "a family alone in the pool has no victim"
    assert "GGML_ASSERT" not in text

    after = _complete(8)
    assert after.status_code == 200
    assert after.body["timings"]["predicted_n"] == 8


def test_a_restored_slot_gives_its_idle_buffer_back_when_another_slot_needs_to_park():
    # Under a finite --preempt-ram an asynchronous slot keeps its pinned buffer after a
    # restore, for its next park, and that idle capacity counted against the budget. With
    # a budget that holds one sequence, the first restore spent it for good: every later
    # park of the other slot was refused. The idle buffer is given back when another slot
    # needs the room, and both slots go on being parked.
    global server
    # a pool of 8192 cells, but the model's own window is 2048, so each generation stays
    # under that; 1800 tokens of this model's state is about 1.1 MiB, so a budget of
    # 2 MiB holds one sequence and not two
    server.n_ctx = 8192
    server.n_gpu_layer = 99
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "256"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    text = _start_async()
    _require_async(text)
    log = LogReader(server.log_path)

    n_predict = 1800
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "idle parked RAM returned" in text, "the idle buffer of a restored slot was never given back"
    import re
    parked = re.findall(r"id\s+(\d+) \| task \d+ \| preempted on request", text)
    assert {"0", "1"} <= set(parked), f"only slots {sorted(set(parked))} were ever parked"

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False


def test_a_budget_that_holds_one_sequence_does_not_rotate_and_the_head_resumes_when_a_resident_finishes():
    # A budget that holds both heads but not a head plus the resident refuses rotation; every stream still finishes.
    global server
    server.n_slots = 3
    server.n_ctx = 2048
    server.enable_ctx_shift = True
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    server.start()
    # At 6000 tokens the resident finished before a rotation was ever asked for.
    n_predict = 12000
    prompts = [
        "Once upon a time there was a brave knight who",
        "The quick brown fox jumps over the lazy dog and",
        "In a small village by the sea there lived a fisherman who",
    ]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": p, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for p in prompts
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "no rotation: --preempt-ram 2 MiB" in text
    assert "resumed after" in text
    assert "Context size has been exceeded" not in text


def test_a_recurrent_model_is_served_without_preemption():
    # Preemption is off for a recurrent cache, and the forced-park knob parks nothing.
    global server
    path = os.environ.get("LLAMA_SERVER_TEST_RECURRENT_MODEL")
    if path:
        server.model_file = path
    else:
        server.model_file = None
        server.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
        server.model_hf_file = "mamba-130m-hf.Q2_K.gguf"
        server.offline = False
    server.n_ctx = 1024
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start(timeout_seconds=300)
    results = parallel_function_calls([
        (_complete, (64, "Once upon a time")),
        (_complete, (64, "The quick brown fox")),
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == 64
    text = open(server.log_path).read()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    assert "preempted" not in text
    assert "Context size has been exceeded" not in text
