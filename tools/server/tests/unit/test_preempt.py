import os
import re
import time
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool: when the next decode does not fit, one slot is parked
# (its sequence copied to host RAM, its cells released) instead of every slot being
# terminated. Both tests need more than one slot and --kv-unified, which is the only
# configuration where one slot can take another one's cells.

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
    os.environ.pop("LLAMA_SERVER_PREEMPT_GRANULARITY", None)
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
    # Park and restore the only running slot every 8 tokens. With one request the batch
    # has the same shape at every step whether or not the slot was parked in between, so
    # any difference in the output is the preemption's fault and nothing else's.
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
    # Each request alone fits in the pool: 8 prompt tokens plus 160 generated is well
    # under 256. Together they do not, 336 against 256. Without preemption the retry
    # ladder ends with "Context size has been exceeded" on every processing slot; with it
    # the smaller slot is parked until the leader finishes and its cells are purged, and
    # then it resumes from the token it was parked on.
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


def test_the_planner_counts_whole_pages_when_the_pool_allocates_in_pages():
    # A pool that hands out cells in blocks gives a whole block to one sequence, so a sequence
    # of n tokens occupies round_up(n, block) cells and holds the rest of its tail block against
    # everybody else. The planner has to count those cells: counting tokens, it sees room the
    # allocator cannot find, never parks anybody, and the retry ladder ends every request.
    #
    # llama_memory_alloc_granularity() reports the block size, and the only mode that returns
    # more than 1 today is exact concurrency, whose paged attention kernel needs a head size this
    # model does not have. LLAMA_SERVER_PREEMPT_GRANULARITY injects the figure instead: what is
    # under test is the server's arithmetic, which is the same at 64 as at 256.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_GRANULARITY = 64" in log.drain()

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    # every figure the planner logs is a whole number of blocks: "kv N/256" is what the pool is
    # holding and "(wanted N)" is that plus what the next decode reserves. Counting tokens, both
    # land wherever the sequences happen to be.
    held   = [int(n) for n in re.findall(r"kv (\d+)/256", text)]
    wanted = [int(n) for n in re.findall(r"\(wanted (\d+)\)", text)]
    assert held and wanted, f"the planner logged no figures:\n{text}"
    assert all(n % 64 == 0 for n in held + wanted), f"not whole blocks: {held} {wanted}"

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
        # about four tokens per word on this model's vocabulary
        words = words[: len(words) - max(1, (n - n_tokens) // 8)]
    raise AssertionError("empty prompt")


def test_two_prompts_that_overflow_the_pool_together_both_finish():
    # Neither slot ever generates before the pool is full: both are still processing their
    # prompts. A prompt-processing slot is between two chunks of its prompt, which is as
    # clean a boundary as the one between two sampled tokens, so it is parked the same way.
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
    # One slot is generating a long answer to a short prompt when a large prompt arrives
    # beside it. Together they need far more than the pool has. The prompt is admitted
    # chunk by chunk, whoever is smaller is parked when the pool fills, and both finish.
    # This model produces a thousand tokens a second, so the second request is sent right
    # behind the first rather than after a delay: its prompt takes several batches to
    # process, which is enough for the two to overlap however fast the first one runs.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    # b lives long enough for the two to collide: the first run of this used 16 tokens
    # and b was finished and purged before a had grown into it
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
    # --preempt-ram 0 is the switch back to the old behaviour: nothing is parked and the
    # KV-full path ends the requests the way it always did.
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
    # A client that wants to tell a parked chat from a slow one reads /slots, and an
    # operator reads /metrics. Both must show the preemption happening, and the counters
    # must survive the requests finishing.
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
    # Two prompts that each fit the context alone but not together. The second one is
    # parked before it takes any cells, and it is close enough to n_ctx that its sequence
    # plus its first batch would not leave the usual scheduling margin. It must still be
    # restored once the first one finishes: with nothing resident there is nobody to keep
    # the margin for. Before the fix it was parked for ever, with no restore ever tried.
    global server
    server.n_ctx = 256
    # the whole prompt in one batch, so the parked slot's first step is the whole prompt
    server.n_batch = 256
    server.start()
    log = LogReader(server.log_path)

    # sized in tokens, not words: the prompt is the token ids of a short sentence repeated
    base = server.make_request("POST", "/tokenize", data={"content": "Once upon a time there was a little girl"}).body["tokens"]
    long_prompt = (base * 64)[:240]
    n_predict = 4
    together = parallel_function_calls([(_complete, (n_predict, long_prompt)) for _ in range(2)])

    text = log.drain()
    assert "cannot fit the pool" not in text

    for res in together:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
