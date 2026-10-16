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
