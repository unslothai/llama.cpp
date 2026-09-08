import os
import re
import time
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool: one slot is parked, its sequence copied to host RAM and its cells released, instead of every slot being terminated. Needs --kv-unified.

server = ServerPreset.tinyllama2()

_ASYNC_BANNER = "parking and resuming asynchronously"

_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"
_PROMPT_C = "In a small village by the sea there lived a fisherman who"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.server_slots = True
    server.server_metrics = True
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    for name in ("LLAMA_SERVER_PREEMPT_EVERY", "LLAMA_SERVER_PREEMPT_GRANULARITY",
                 "LLAMA_SERVER_PREEMPT_PLANNER", "LLAMA_ARG_PREEMPT_RAM", "LLAMA_ARG_PREEMPT_ASYNC"):
        os.environ.pop(name, None)


def _start(**kwargs):
    """Start the server with these settings; its log starts empty again on every start."""
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()


def _start_async(**kwargs):
    """As _start, on the asynchronous park path; a backend that cannot copy off-thread skips the test."""
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    _start(n_gpu_layer=99, **kwargs)
    _require_async(_log())


def _log() -> str:
    return open(server.log_path).read()


def _require_async(text: str):
    if _ASYNC_BANNER not in text:
        pytest.skip("this backend cannot copy asynchronously, the async park path is not exercised")


def _complete(n_predict: int, prompt="Hi how are you", id_slot: int = -1, delay: float = 0.0):
    time.sleep(delay)
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict, "prompt": prompt, "id_slot": id_slot,
        "ignore_eos": True, "return_tokens": True, "temperature": 0.0, "seed": 42,
    })


def _complete_all(n_predict: int, prompts=(_PROMPT_A, _PROMPT_B)):
    return parallel_function_calls([(_complete, (n_predict, prompt)) for prompt in prompts])


def _prompt_of(n_tokens: int, text: str) -> list:
    """A prompt of exactly n_tokens tokens, as ids: no BOS is added to one of those."""
    base = server.make_request("POST", "/tokenize", data={"content": text}).body["tokens"]
    assert base
    return (base * (n_tokens // len(base) + 1))[:n_tokens]


def _assert_completed(results, n_predict: int):
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict


def _assert_recovered(text: str, parked: str = "preempted:"):
    """Nothing was ended for want of cells: a slot was parked and came back."""
    assert "Context size has been exceeded" not in text
    assert parked in text
    assert "resumed after" in text


def _metrics() -> dict:
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    return {
        name[len("llamacpp:"):]: float(value)
        for name, value in (line.split(" ", 1) for line in res.body.splitlines() if line.startswith("llamacpp:"))
    }


@pytest.mark.parametrize("mode", ["sync", "async", "no-async"])
def test_forced_parks_do_not_change_the_output(mode):
    if mode != "sync":
        server.n_gpu_layer = 99
        os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1" if mode == "async" else "0"
    _start(n_ctx=512)
    if mode == "async":
        _require_async(_log())
    reference = _complete(64)
    assert reference.status_code == 200
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start()
    preempted = _complete(64)
    assert preempted.status_code == 200
    assert preempted.body["timings"]["predicted_n"] == 64
    assert preempted.body["content"] == reference.body["content"]
    assert preempted.body["tokens"] == reference.body["tokens"]

    text = _log()
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6
    if mode == "async":
        assert "park completed after" in text
        assert "restore issued in" in text
        assert "restore completed after" in text
    if mode == "no-async":
        assert _ASYNC_BANNER not in text
        assert "park issued in" not in text


@pytest.mark.parametrize("knob", ["planner", "pages", "async", "last-resort", "last-resort-unlimited"])
def test_two_generations_that_do_not_fit_together_both_finish(knob):
    # each request fits the pool alone (168 of 256 cells) but not together; without preemption both end with "Context size has been exceeded"
    if knob == "pages":
        # a block allocator gives a whole block to one sequence, so the planner has to count cells: counting tokens it sees room the allocator cannot find
        os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    if knob.startswith("last-resort"):
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    if knob == "last-resort-unlimited":
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    (_start_async if knob == "async" else _start)(n_ctx=256)

    n_predict = 160
    results = _complete_all(n_predict)
    text = _log()
    _assert_recovered(text, "preempted as a last resort" if knob.startswith("last-resort") else "preempted:")
    _assert_completed(results, n_predict)
    for res in results:
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict

    if knob == "pages":
        held   = [int(n) for n in re.findall(r"kv (\d+)/256", text)]
        wanted = [int(n) for n in re.findall(r"\(wanted (\d+)\)", text)]
        assert held and wanted, f"the planner logged no figures:\n{text}"
        assert all(n % 64 == 0 for n in held + wanted), f"not whole blocks: {held} {wanted}"
    if knob.startswith("last-resort"):
        assert "preempted:" not in text, "the planner was off, nothing may be parked ahead of the decode"
        assert "last resort: batch given up" in text
    if knob == "planner":
        metrics = _metrics()
        assert metrics["n_preempt_total"] >= 1
        assert metrics["n_resume_total"] == metrics["n_preempt_total"]
        assert metrics["requests_preempted"] == 0
        assert metrics["preempt_ram_bytes"] == 0


@pytest.mark.parametrize("knob", ["ram-0", "family"])
def test_a_request_that_cannot_be_helped_gets_the_context_error_and_the_server_lives(knob):
    if knob == "ram-0":
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    else:
        # a family member is not a victim for the other, so a two-completion request gets the error it would get alone
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    _start(n_ctx=256)

    if knob == "ram-0":
        assert any(res.status_code != 200 for res in _complete_all(160))
    else:
        res = server.make_request("POST", "/completion", data={
            "n_predict": 160, "n_cmpl": 2, "prompt": _PROMPT_A,
            "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })
        assert res.status_code == 500
        assert "Context size has been exceeded" in res.body["error"]["message"]

    text = _log()
    assert "Context size has been exceeded" in text
    assert "preempted" not in text, "nothing could be parked here"
    assert "GGML_ASSERT" not in text
    after = _complete(8)
    assert after.status_code == 200
    assert after.body["timings"]["predicted_n"] == 8


@pytest.mark.parametrize("planner", ["on", "off"])
def test_a_late_prompt_and_a_generating_slot_both_finish(planner):
    if planner == "off":
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    _start(n_ctx=256)

    n_b = 150
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a + n_b + n_predict_b > 256
    results = parallel_function_calls([
        (_complete, (n_predict_a, "Hi how are you")),
        (_complete, (n_predict_b, _prompt_of(n_b, _PROMPT_C), -1, 0.02)),
    ])

    text = _log()
    assert "Context size has been exceeded" not in text
    assert ("preempted as a last resort" if planner == "off" else "preempted:") in text
    for res, n_predict in zip(results, (n_predict_a, n_predict_b)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    # the chunk in the batch given up is processed once after the rewind, never twice
    assert results[1].body["timings"]["prompt_n"] == n_b


def test_a_prompt_parked_before_its_first_token_is_issued_whole():
    # both prompts are too close to n_ctx to leave the usual margin, so the second is parked before it takes a cell and has to come back once the first has finished
    _start(n_ctx=256, n_batch=256)

    n_prompt = 240
    n_predict = 4
    long_prompt = _prompt_of(n_prompt, "Once upon a time there was a little girl")
    together = _complete_all(n_predict, [long_prompt, long_prompt])

    assert "cannot fit the pool" not in _log()
    _assert_completed(together, n_predict)
    for res in together:
        assert res.body["timings"]["prompt_n"] == n_prompt, "the prompt was not issued once and whole"


def test_a_resident_cycling_through_context_shifts_is_rotated_out_for_a_parked_head():
    # with context shift on a resident would hold its cells for as long as it generates, so once the head has waited its turn the resident is parked and the two take turns
    _start(n_slots=3, n_ctx=384, enable_ctx_shift=True)

    n_predict = 9000
    _assert_completed(_complete_all(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C)), n_predict)

    text = _log()
    _assert_recovered(text, "rotated out after")
    assert "slot context shift" in text


def test_cancel_while_a_copy_is_in_flight_frees_the_slot():
    # a cancelled request can reach release() with a park or a resume still running, where the host buffer is freed and the cells handed on, so both have to wait for the copy
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start_async(n_ctx=512)

    for i in range(4):
        try:
            server.make_request("POST", "/completion", data={
                "n_predict": 96, "prompt": _PROMPT_A, "ignore_eos": True, "temperature": 0.0, "seed": 42,
            }, timeout=0.05 + 0.1 * i)
        except Exception:
            pass  # the point is the drop, not the response

    for _ in range(600):
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        if all(not slot["is_processing"] for slot in res.body):
            break
        time.sleep(0.2)
    else:
        pytest.fail("a slot never came back after a cancel during a copy")
    for slot in res.body:
        assert slot["is_preempted"] is False
    assert _metrics()["preempt_ram_bytes"] == 0, "a cancelled slot kept its parked memory"

    res = _complete(16)
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 16


def test_a_started_slot_is_counted_by_the_cells_it_holds_not_by_the_prompt_it_keeps():
    # the last request waits for slot 0 and is started on it holding the first request's cells; counted by the prompt it keeps instead, the pool looks free and a parked slot is restored into cells that are still taken
    _start(n_ctx=256, n_slots=3)

    results = parallel_function_calls([
        (_complete, (60, _prompt_of(115, _PROMPT_C), 0)),
        (_complete, (100, _PROMPT_A, 1)),
        (_complete, (100, _PROMPT_B, 2)),
        (_complete, (8, _PROMPT_C, 0, 0.05)),
    ])

    text = _log()
    assert "trimmed to the" in text, "the started slot kept the cells of the request before it"
    assert "resume failed" not in text
    assert "Context size has been exceeded" not in text
    for res, n_predict in zip(results, (60, 100, 100, 8)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict


def test_a_recurrent_model_is_served_without_preemption():
    server.model_file = os.environ.get("LLAMA_SERVER_TEST_RECURRENT_MODEL")
    if server.model_file:
        server.model_hf_repo = server.model_hf_file = None
    else:
        server.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
        server.model_hf_file = "mamba-130m-hf.Q2_K.gguf"
        server.offline = False
    server.n_ctx = 1024
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start(timeout_seconds=300)

    results = _complete_all(64, ["Once upon a time", "The quick brown fox"])
    _assert_completed(results, 64)

    text = _log()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    assert "preempted" not in text
    assert "Context size has been exceeded" not in text
