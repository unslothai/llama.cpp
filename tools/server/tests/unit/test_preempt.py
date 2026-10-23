import base64
import json
import os
import re
import struct
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import tempfile
import pytest
import requests
from utils import *


server = ServerPreset.tinyllama2()

_ASYNC_BANNER = "parking and resuming asynchronously"

_scratch = []
_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
_policy = os.environ.get("LLAMA_ARG_PREEMPT_POLICY")

_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"
_PROMPT_C = "In a small village by the sea there lived a fisherman who"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "8192"
    server.server_slots = True
    server.server_metrics = True
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    if _cuda_devices is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_devices
    if _scratch:
        server.stop()
    while _scratch:
        os.remove(_scratch.pop())
    for name in ("LLAMA_SERVER_PREEMPT_EVERY", "LLAMA_SERVER_PREEMPT_GRANULARITY",
                 "LLAMA_SERVER_PREEMPT_PLANNER", "LLAMA_ARG_PREEMPT_RAM", "LLAMA_ARG_PREEMPT_ASYNC",
                 "LLAMA_SERVER_PREEMPT_FAIL_SAVE", "LLAMA_SERVER_PREEMPT_PAGEABLE", "LLAMA_ARG_SPEC_DRAFT_P_MIN", "LLAMA_ARG_SPEC_DRAFT_SAMPLING", "LLAMA_ARG_LOG_VERBOSITY",
                 "LLAMA_BATCH_DEBUG", "LLAMA_ARG_CTX_CHECKPOINTS",
                 "LLAMA_MEDIA_MARKER", "LLAMA_EXACT_CONCURRENCY", "LLAMA_ARG_PREEMPT_POLICY"):
        os.environ.pop(name, None)
    if _policy is not None:
        os.environ["LLAMA_ARG_PREEMPT_POLICY"] = _policy


def _start(**kwargs):
    """Start the server with these settings; its log starts empty again on every start."""
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()


def _start_async(**kwargs):
    """As _start, on the asynchronous park path; a backend that cannot copy off-thread skips the test."""
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    _start(n_gpu_layer=99, **kwargs)
    _require_async()


def _log() -> str:
    """The server log once its writer thread has stopped growing it: on a loaded host the log lags the response that came from it."""
    deadline = time.time() + 5.0
    last = -1
    while time.time() < deadline:
        size = os.path.getsize(server.log_path)
        if size == last:
            break
        last = size
        time.sleep(0.1)
    return open(server.log_path, errors="replace").read()


def _startup_log() -> str:
    """The server log up to its last startup line: the log is written by its own thread, so it can lag the server being up,
    and a capability banner missing from a lagging log is not a missing capability."""
    deadline = time.time() + 60.0
    while True:
        text = open(server.log_path, errors="replace").read()
        if "listening on " in text:
            return text
        if time.time() > deadline:
            pytest.fail("the server log never reached its last startup line")  # ty: ignore[invalid-argument-type]
        time.sleep(0.05)


def _require_async():
    if _ASYNC_BANNER not in _startup_log():
        pytest.skip("this backend cannot copy asynchronously, the async park path is not exercised")  # ty: ignore[too-many-positional-arguments]


def _complete(n_predict: int, prompt="Hi how are you", id_slot: int = -1, delay: float = 0.0, after_slot_busy=None,
              timeout: float = DEFAULT_REQUEST_TIMEOUT):
    time.sleep(delay)
    if after_slot_busy is not None:
        for _ in range(200):
            slots = server.make_request("GET", "/slots").body
            if any(s["id"] == after_slot_busy and s["is_processing"] for s in slots):
                break
            time.sleep(0.02)
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict, "prompt": prompt, "id_slot": id_slot,
        "ignore_eos": True, "return_tokens": True, "temperature": 0.0, "seed": 42,
    }, timeout=timeout)


def _complete_all(n_predict: int, prompts=(_PROMPT_A, _PROMPT_B), timeout: float = DEFAULT_REQUEST_TIMEOUT):
    return parallel_function_calls([(_complete, (n_predict, prompt, -1, 0.0, None, timeout)) for prompt in prompts])


def _server_progress():
    """What every slot has prefilled and decoded so far: it changes while the server is working, however slow the host."""
    return tuple((s["id"], s.get("id_task"), s.get("n_prompt_tokens_processed", 0), s.get("n_prompt_tokens", 0),
                  s.get("next_token", [{}])[0].get("n_decoded", 0), s.get("n_preempt", 0))
                 for s in server.make_request("GET", "/slots").body)


def _await(cond, what: str, alive=None, stall: float = 120.0, poll: float = 0.02) -> None:
    """Wait for cond() as long as the server keeps making progress, so a slow runner is only slower: fail if the request
    it depends on ends first (alive() false) or nothing has moved for `stall` seconds, never on a wall-clock deadline."""
    last, t_last, t_check = None, time.time(), 0.0
    while not cond():
        if alive is not None and not alive():
            if cond():
                return
            pytest.fail(f"{what}: the request it waits on ended first")  # ty: ignore[invalid-argument-type]
        now = time.time()
        if now - t_check >= 0.5:
            t_check = now
            cur = _server_progress()
            if cur != last:
                last, t_last = cur, now
            elif now - t_last > stall:
                pytest.fail(f"{what}: the server made no progress for {stall:.0f} s")  # ty: ignore[invalid-argument-type]
        time.sleep(poll)


def _wait_processing(slot_ids, timeout: float = 30.0):
    """Return once every one of these slots is processing; a request that ended first is a failure, not a hang."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        slots = server.make_request("GET", "/slots").body
        if all(any(s["id"] == i and s["is_processing"] for s in slots) for i in slot_ids):
            return
        time.sleep(0.005)
    pytest.fail(f"slots {slot_ids} never showed as processing")  # ty: ignore[invalid-argument-type]


def _complete_overlapping(n_predict, n_prompt, timeout: float = DEFAULT_REQUEST_TIMEOUT):
    """A leader on slot 0 and a follower on slot 1 that certainly overlap: the follower is sent once the leader is seen processing, so the lengths and not the client's speed decide what the pool has to hold."""
    leader = _prompt_of(n_prompt[0], _PROMPT_A)
    other  = _prompt_of(n_prompt[1], _PROMPT_B)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(_complete, n_predict[0], leader, 0, 0.0, None, timeout)
        _wait_processing([0])
        second = _complete(n_predict[1], other, 1, 0.0, None, timeout)
    return [first.result(), second]


def _complete_together(n_predict: int, n_prompt, timeout: float = DEFAULT_REQUEST_TIMEOUT):
    """Both prompts in one request, so they are queued at once and overlap whatever the client's or the host's speed."""
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict, "prompt": [_prompt_of(n_prompt[0], _PROMPT_A), _prompt_of(n_prompt[1], _PROMPT_B)],
        "ignore_eos": True, "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }, timeout=timeout)


def _wait_preempted(timeout: float = 30.0) -> bool:
    """True once some slot is parked: its cells are in host RAM and it wants them back. False once the server has gone
    `timeout` seconds without progress and without parking: a slow host only takes longer."""
    last, t_last = None, time.time()
    while True:
        slots = server.make_request("GET", "/slots").body
        if any(s["is_preempted"] for s in slots):
            return True
        cur = _server_progress()
        if cur != last:
            last, t_last = cur, time.time()
        elif time.time() - t_last > timeout:
            return False
        time.sleep(0.005)


def _prompt_of(n_tokens: int, text: str) -> list:
    """A prompt of exactly n_tokens tokens, as ids: no BOS is added to one of those."""
    base = server.make_request("POST", "/tokenize", data={"content": text}).body["tokens"]
    assert base
    return (base * (n_tokens // len(base) + 1))[:n_tokens]


def _assert_completed(results, n_predict: int):
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict


_PARK_MARKERS = ("preempted:", "preempted as a last resort", "preempted on request")


def _assert_nothing_parked(text: str):
    assert not any(m in text for m in _PARK_MARKERS), "nothing could be parked here"


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
        _require_async()
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
    if knob == "pages":
        os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    if knob.startswith("last-resort"):
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    if knob == "last-resort-unlimited":
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    n_ctx = 1024
    (_start_async if knob == "async" else _start)(n_ctx=n_ctx)

    n_predict = (460, 400)
    results = _complete_overlapping(n_predict, (500, 200))

    text = _log()
    _assert_recovered(text, "preempted as a last resort" if knob.startswith("last-resort") else "preempted:")
    for res, n_wanted in zip(results, n_predict):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_wanted
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_wanted

    if knob == "pages":
        held   = [int(n) for n in re.findall(rf"kv (\d+)/{n_ctx}", text)]
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
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    _start(n_ctx=256)

    if knob == "ram-0":
        assert _complete_together(110, (120, 60)).status_code != 200
    else:
        res = server.make_request("POST", "/completion", data={
            "n_predict": 160, "n_cmpl": 2, "prompt": _PROMPT_A,
            "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })
        assert res.status_code == 500
        assert "Context size has been exceeded" in res.body["error"]["message"]

    text = _log()
    assert "Context size has been exceeded" in text
    _assert_nothing_parked(text)
    assert "GGML_ASSERT" not in text
    after = _complete(8)
    assert after.status_code == 200
    assert after.body["timings"]["predicted_n"] == 8


def test_a_server_that_never_asked_for_parking_behaves_as_upstream():
    """--preempt-ram defaults to 0, so a unified-cache server started without it parks nothing."""
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)
    _start(n_ctx=256)

    text = _log()
    assert "preemption:" not in text, "a server that did not ask for parking announced it"
    assert _ASYNC_BANNER not in text, "the async park path was set up without being asked for"

    assert _complete_together(110, (120, 60)).status_code != 200

    text = _log()
    assert "Context size has been exceeded" in text
    _assert_nothing_parked(text)
    assert "last resort" not in text, "the retry ladder consulted the planner"
    assert "GGML_ASSERT" not in text

    metrics = _metrics()
    assert metrics["n_preempt_total"] == 0
    assert metrics["preempt_ram_bytes"] == 0

    after = _complete(8)
    assert after.status_code == 200
    assert after.body["timings"]["predicted_n"] == 8


@pytest.mark.parametrize("planner", ["on", "off"])
def test_a_late_prompt_and_a_generating_slot_both_finish(planner):
    if planner == "off":
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    # a long leader: the follower is sent once the leader is seen processing, and a short one could finish before the follower arrives
    _start(n_ctx=2048)

    n_b = 1500
    n_predict_a = 1900
    n_predict_b = 500
    assert 8 + n_predict_a + n_b + n_predict_b > 2048
    results = _complete_overlapping((n_predict_a, n_predict_b), (8, n_b))

    text = _log()
    assert "Context size has been exceeded" not in text
    assert ("preempted as a last resort" if planner == "off" else "preempted:") in text
    for res, n_predict in zip(results, (n_predict_a, n_predict_b)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    assert results[1].body["timings"]["prompt_n"] == n_b


def test_a_prompt_parked_before_its_first_token_is_issued_whole():
    _start(n_ctx=256, n_batch=256)

    n_prompt = 240
    n_predict = 4
    together = _complete_all(n_predict, [_prompt_of(n_prompt, _PROMPT_A), _prompt_of(n_prompt, _PROMPT_B)])

    assert "cannot fit the pool" not in _log()
    _assert_completed(together, n_predict)
    for res in together:
        assert res.body["timings"]["prompt_n"] == n_prompt, "the prompt was not issued once and whole"


def test_a_resident_cycling_through_context_shifts_is_rotated_out_for_a_parked_head():
    _start(n_slots=3, n_ctx=384, enable_ctx_shift=True)

    n_predict = 9000
    _assert_completed(_complete_all(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C), timeout=1800), n_predict)

    text = _log()
    _assert_recovered(text, "rotated out after")
    assert "slot context shift" in text


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_a_parked_head_that_needs_the_whole_pool_is_rotated_in(policy):
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _start(n_ctx=512, n_batch=512, cache_ram=0, enable_ctx_shift=True, n_predict=-1)
    head = _prompt_of(506, _PROMPT_B)
    alone = _raw_completion(1, head, 1, timeout=60)
    assert alone.status_code == 200, alone.body

    stop = threading.Event()

    def lead():
        with requests.post(server.make_url("/completion"), json={
            "n_predict": -1, "prompt": _prompt_of(320, _PROMPT_A), "id_slot": 0, "cache_prompt": False,
            "ignore_eos": True, "temperature": 0.0, "seed": 42, "stream": True,
        }, stream=True, timeout=120) as res:
            for _ in res.iter_lines():
                if stop.is_set():
                    break

    leader = threading.Thread(target=lead, daemon=True)
    leader.start()
    try:
        _await(lambda: _n_decoded(_slot_view(0)) > 40, "the leader decoding 40 tokens", alive=leader.is_alive, poll=0.005)
        res = _raw_completion(1, head, 1, timeout=600)
    finally:
        stop.set()
        leader.join(30)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 1
    text = _log()
    assert "Context size has been exceeded" not in text
    assert "rotated out after" in text


def test_a_park_whose_host_allocation_fails_is_parked_by_recompute():
    os.environ["LLAMA_SERVER_PREEMPT_FAIL_SAVE"] = "1"
    _start(n_ctx=1024, n_slots=2)

    leader = _prompt_of(500, _PROMPT_A)
    other = _prompt_of(200, _PROMPT_B)
    with ThreadPoolExecutor(1) as pool:
        first = pool.submit(_complete, 460, leader, 0)
        _wait_processing([0])
        second = _complete(400, other, 1)
    results = [first.result(), second]

    text = _log()
    assert "could not take the host memory" in text, "the injected allocation failure never fired"
    assert "tokens to re-prefill" in text, "the failed save did not fall back to recompute"
    assert "Context size has been exceeded" not in text
    for res, n_predict in zip(results, (460, 400)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_a_resident_that_cannot_be_swapped_out_is_rotated_by_recompute(policy):
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _start(n_ctx=2048, n_slots=2, n_batch=2048, enable_ctx_shift=True)

    def unending_request():
        return server.make_request("POST", "/completion", data={
            "n_predict": 100000, "prompt": _PROMPT_A, "n_keep": 1, "n_discard": 64,
            "ignore_eos": True, "temperature": 0.0, "seed": 42,
        }, timeout=600)

    unending = []

    def run_unending():
        try:
            unending.append(unending_request())
        except requests.exceptions.RequestException:
            pass

    t = threading.Thread(target=run_unending)
    t.start()

    try:
        _await(lambda: "slot context shift" in open(server.log_path, errors="replace").read(),
               "the resident reaching the end of the pool", alive=t.is_alive, poll=0.05)

        waiting = server.make_request("POST", "/completion", data={
            "n_predict": 8, "prompt": _prompt_of(1800, _PROMPT_C),
            "ignore_eos": True, "temperature": 0.0, "seed": 42,
        }, timeout=300)

        assert waiting.status_code == 200, waiting.body
        assert waiting.body["timings"]["predicted_n"] == 8, "the second request never made progress"
        assert not unending, "the first request ended before the second made progress"
    finally:
        server.stop()
        t.join(60)

    text = _log()
    assert "slot context shift" in text
    assert re.search(r"rotated out after .* cells dropped", text), "the rotation did not fall back to recompute"
    assert "Context size has been exceeded" not in text


def test_cancel_while_a_copy_is_in_flight_frees_the_slot():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start_async(n_ctx=512)

    for i in range(4):
        try:
            server.make_request("POST", "/completion", data={
                "n_predict": 96, "prompt": _PROMPT_A, "ignore_eos": True, "temperature": 0.0, "seed": 42,
            }, timeout=0.05 + 0.1 * i)
        except Exception:
            pass

    _await(lambda: all(not s["is_processing"] for s in server.make_request("GET", "/slots").body),
           "every slot coming back after a cancel during a copy", poll=0.2)
    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["is_transferring"] is False
    assert _metrics()["preempt_ram_bytes"] == 0, "a cancelled slot kept its parked memory"

    res = _complete(16)
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 16


def test_a_started_slot_is_counted_by_the_cells_it_holds_not_by_the_prompt_it_keeps():
    _start(n_ctx=1024, n_slots=3)

    ids = _prompt_of(500, _PROMPT_C)

    with ThreadPoolExecutor(4) as pool:
        first = pool.submit(_complete, 460, ids, 0)
        _wait_processing([0])
        follower = pool.submit(_complete, 8, ids[:8], 0)
        long_ones = [pool.submit(_complete, 400, _prompt_of(200, _PROMPT_A), 1),
                     pool.submit(_complete, 400, _prompt_of(200, _PROMPT_B), 2)]
        parked = _wait_preempted()
    results = [first.result(), follower.result(), long_ones[0].result(), long_ones[1].result()]

    text = _log()
    assert parked, "the pool never came under pressure, so no slot was waiting for the cells slot 0 keeps"
    assert "trimmed to the" in text, "the started slot kept the cells of the request before it"
    assert "resume failed" not in text
    assert "Context size has been exceeded" not in text
    for res, n_predict in zip(results, (460, 8, 400, 400)):
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

    # Not "Once upon a time": on CUDA this Q2_K model decodes bytes the content parser refuses, also on master.
    results = _complete_all(64, ["The quick brown fox", "Hello world"])
    _assert_completed(results, 64)

    text = _log()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    _assert_nothing_parked(text)
    assert "Context size has been exceeded" not in text


def _stream_completion(n_predict: int, prompt: str | list[int]) -> tuple[list[str], dict]:
    """One streaming completion: its SSE comment lines and the last response object."""
    url = f"http://{server.server_host}:{server.server_port}/completion"
    res = requests.post(url, json={
        "prompt": prompt, "n_predict": n_predict, "ignore_eos": True,
        "temperature": 0.0, "seed": 42, "stream": True,
    }, stream=True, timeout=600)
    assert res.status_code == 200, res.text
    comments, datas = [], []
    for raw in res.iter_lines():
        line = raw.decode("utf-8")
        if line.startswith(":"):
            comments.append(line)
        elif line.startswith("data: ") and line[6:] != "[DONE]":
            datas.append(json.loads(line[6:]))
    return comments, datas[-1]


@pytest.mark.parametrize("planner", ["on", "off"])
def test_a_budget_that_holds_no_sequence_parks_by_dropping_the_cells(planner):
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    if planner == "off":
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    # Two generations that overflow the pool together, each longer than the ~1500 tokens the 1 MiB budget holds when they do.
    # With 4000 each in 6144 cells they overflow unless the second starts ~1900 tokens after the first (2000 in 3840 allowed
    # only ~180, which a slow client exceeded), and the shorter one then holds over 2100 tokens, so the park drops its cells.
    # A longer training context lets one slot take 4000 tokens.
    # One batch for the re-prefill: this prompt in 32-token batches hangs the CUDA build with graphs on, also on master.
    _serve_with_training_context(8192)
    _start(n_ctx=6144, n_batch=4096)

    n_predict = 4000
    if planner == "on":
        results = parallel_function_calls([(_stream_completion, (n_predict, p)) for p in (_PROMPT_A, _PROMPT_B)])
        for comments, final in results:
            assert "error" not in final, final
            assert final["tokens_predicted"] == n_predict
        comments = [c for cs, _ in results for c in cs]
        assert ": preempted" in comments and ": resumed" in comments, comments
    else:
        # one request with both prompts: queued at once, whatever the client's speed
        res = _complete_together(n_predict, (8, 8))
        assert res.status_code == 200, res.body
        assert [r["tokens_predicted"] for r in res.body] == [n_predict, n_predict], res.body

    text = _log()
    assert "Context size has been exceeded" not in text
    assert "tokens to re-prefill" in text, "no park fell back to recompute"
    if planner == "off":
        assert "preempted as a last resort by dropping its cells" in text


_IMG_URL = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def test_a_media_chunk_is_reserved_whole_before_it_is_decoded():
    os.environ["LLAMA_MEDIA_MARKER"] = "<__media__>"
    server.model_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"
    server.model_hf_file = None
    server.model_alias = "tinygemma3"
    _start(n_ctx=400, n_batch=64, n_ubatch=64)

    image = base64.b64encode(requests.get(_IMG_URL, timeout=60).content).decode()
    prompt = {"prompt_string": "<__media__>\nWhat is in this image?", "multimodal_data": [image]}
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": prompt, "n_predict": 4, "temperature": 0.0, "seed": 42,
        })) for _ in range(2)
    ])

    text = _log()
    assert "failed to process mtmd chunk" not in text
    assert "preempted:" in text, "nothing was parked to make room for a chunk"
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["timings"]["prompt_n"] > 64, "the chunk fits one batch, so it never spans several decodes"


def test_an_mtp_draft_stays_inside_the_reservation_it_was_priced_for():
    path = os.environ.get("LLAMA_SERVER_TEST_MTP_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MTP_MODEL to a gguf carrying an MTP head")  # ty: ignore[too-many-positional-arguments]
    server.model_file = path
    server.model_hf_repo = server.model_hf_file = None
    server.spec_type = "draft-mtp"
    os.environ["LLAMA_ARG_SPEC_DRAFT_P_MIN"] = "0.0"
    os.environ["LLAMA_ARG_LOG_VERBOSITY"] = "5"
    _start(n_ctx=2048, n_slots=2, n_batch=2048, n_gpu_layer=99, spec_draft_n_max=128, spec_draft_n_min=1)

    n_prompt = 2045
    res = server.make_request("POST", "/completion", data={
        "prompt": _prompt_of(n_prompt, _PROMPT_C), "n_predict": 2, "ignore_eos": True,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }, timeout=600)

    assert res.status_code == 200, res.body
    assert res.body["timings"]["prompt_n"] == n_prompt
    assert res.body["tokens_predicted"] == 2

    text = _log()
    assert "truncating draft to" not in text, "the draft was scheduled past the tokens the planner reserved"
    assert "llama_decode[" not in text
    assert "Context size has been exceeded" not in text


def test_a_hybrid_model_parks_synchronously():
    path = os.environ.get("LLAMA_SERVER_TEST_HYBRID_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_HYBRID_MODEL to a hybrid attention/recurrent gguf")  # ty: ignore[too-many-positional-arguments]
    server.model_file = path
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=1024, n_gpu_layer=99)

    res = _complete(24, "Once upon a time")
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 24

    text = _log()
    assert "a recurrent state does not stay in one row" in text
    assert _ASYNC_BANNER not in text
    assert "park issued in" not in text
    _assert_recovered(text, "preempted on request")


def _generated_model(arch: str, name: str) -> str:
    """A model made by test-llama-archs beside the server binary; LLAMA_SERVER_TEST_GENERATED_MODELS names a directory that already holds it."""
    have = os.environ.get("LLAMA_SERVER_TEST_GENERATED_MODELS")
    if have:
        return os.path.join(have, f"{name}.gguf")
    gen = os.path.join(os.path.dirname(os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")),
                       "test-llama-archs.exe" if os.name == "nt" else "test-llama-archs")
    if not os.path.exists(gen):
        pytest.skip("build test-llama-archs, or set LLAMA_SERVER_TEST_GENERATED_MODELS to a directory of its models")  # ty: ignore[too-many-positional-arguments]
    out = tempfile.mkdtemp()
    subprocess.run([gen, "-a", f"^{arch}$", "-s", "2981215097", "-o", out], check=True, capture_output=True, timeout=300)
    fd, path = tempfile.mkstemp(suffix=".gguf")
    os.close(fd)
    _scratch.append(path)
    os.replace(os.path.join(out, f"{name}.gguf"), path)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    os.rmdir(out)
    return path


@pytest.mark.parametrize("arch,name,shared", [("deepseek4", "deepseek4-moe", False), ("llama", "llama-dense", True)])
def test_only_a_cache_whose_sequences_share_cells_is_parked(arch, name, shared):
    server.model_file = _generated_model(arch, name)
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=256, n_batch=128, n_ubatch=64, fa="on", n_gpu_layer=0)
    assert server.make_request("GET", "/props").body["preempt"]["enabled"] is shared

    results = parallel_function_calls([(_raw_completion, (80, [1] * 140)) for _ in range(2)])
    _assert_completed(results, 80)
    parks = [(res.body.get("preempt") or {}).get("parks", 0) for res in results]

    text = _log()
    assert "Context size has been exceeded" not in text
    if shared:
        assert sum(parks) >= 1, parks
        _assert_recovered(text)
    else:
        assert parks == [0, 0], parks
        _assert_nothing_parked(text)



def _mrope_model() -> str:
    path = os.environ.get("LLAMA_SERVER_TEST_MROPE_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MROPE_MODEL to an M-RoPE gguf")  # ty: ignore[too-many-positional-arguments]
    return path


def test_exact_concurrency_refuses_an_mrope_model_with_a_projector():
    path = _mrope_model()
    with tempfile.TemporaryDirectory() as tmp:
        mmproj = os.path.join(tmp, "mmproj.gguf")
        with open(mmproj, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))
        proc = subprocess.run([
            os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server"),
            "--model", path, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(server.server_port),
            # a batch that holds a whole ubatch, so the batch geometry check passes and the M-RoPE one is what refuses
            "-c", "512", "-b", "512", "-ub", "128", "--parallel", "2", "--kv-unified", "-fa", "on", "-ngl", "99", "--no-warmup", "--no-webui",
        ], env={**os.environ, "LLAMA_EXACT_CONCURRENCY": "1"}, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "does not support M-RoPE together with a projector" in out, out


def test_exact_concurrency_refuses_a_batch_that_cannot_hold_a_whole_ubatch():
    path = os.environ.get("LLAMA_SERVER_TEST_EXACT_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_EXACT_MODEL to a gguf exact concurrency accepts")  # ty: ignore[too-many-positional-arguments]
    proc = subprocess.run([
        os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server"),
        "--model", path, "--host", "127.0.0.1", "--port", str(server.server_port),
        "-c", "2048", "-b", "512", "-ub", "512", "--parallel", "2", "--kv-unified",
        "-fa", "on", "-ngl", "99", "--no-warmup", "--no-webui",
    ], env={**os.environ, "LLAMA_EXACT_CONCURRENCY": "1"}, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "needs a batch of at least 514 tokens for a 512-token ubatch" in out, out
    assert "Raise -b to 514" in out, out


def test_exact_concurrency_serves_an_mrope_model_without_a_projector():
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    _start(n_ctx=512, n_slots=2, n_batch=512, n_ubatch=128, fa="on", n_gpu_layer=99)

    res = _complete(16, "Once upon a time")
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 16

    text = _log()
    assert "does not support M-RoPE" not in text
    assert "the kv pool allocates 256 cells at a time" in text


def _ubatch_widths(text: str) -> list:
    """The tokens of every ubatch a split produced, in order; needs LLAMA_BATCH_DEBUG."""
    res = []
    pending = False
    for line in text.splitlines():
        if "added ubatch to split" in line:
            pending = True
        elif pending and "n_tokens" in line:
            res.append(int(line.split("=")[-1]))
            pending = False
    return res


def test_exact_concurrency_reuses_no_cached_prefix():
    """A cached prefix was computed in other ubatches than a cold prompt's; under exact concurrency it is recomputed, so the output does not depend on what ran before."""
    path = os.environ.get("LLAMA_SERVER_TEST_EXACT_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_EXACT_MODEL to a gguf exact concurrency accepts")  # ty: ignore[too-many-positional-arguments]
    server.model_file = path
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    _start(n_ctx=2048, n_slots=4, n_batch=128, n_ubatch=64, fa="on", n_gpu_layer=99, cache_ram=0)

    def run(prompt):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": 1, "id_slot": 0, "cache_prompt": True,
            "temperature": 0.0, "seed": 42, "n_probs": 8,
        })
        assert res.status_code == 200, res.body
        return res.body

    prompt = list(range(1000, 1064))
    cold = run(prompt)
    run(prompt[:16])
    warm = run(prompt)
    assert warm["timings"]["cache_n"] == 0, warm["timings"]
    assert warm["completion_probabilities"] == cold["completion_probabilities"]


def test_exact_concurrency_prefills_a_prompt_in_the_ubatches_it_would_get_alone():
    path = os.environ.get("LLAMA_SERVER_TEST_EXACT_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_EXACT_MODEL to a gguf exact concurrency accepts")  # ty: ignore[too-many-positional-arguments]
    server.model_file = path
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    os.environ["LLAMA_BATCH_DEBUG"] = "1"
    os.environ["LLAMA_ARG_LOG_VERBOSITY"] = "5"
    os.environ["LLAMA_ARG_CTX_CHECKPOINTS"] = "0"
    _start(n_ctx=16384, n_slots=4, n_batch=2048, n_ubatch=512, fa="on", n_gpu_layer=99, cache_ram=0)

    def prefill(first_token: int) -> list:
        mark = len(_log())
        res = server.make_request("POST", "/completion", data={
            "prompt": list(range(first_token, first_token + 3500)), "n_predict": 1,
            "cache_prompt": False, "temperature": 0.0, "seed": 42,
        }, timeout=600)
        assert res.status_code == 200, res.body
        assert res.body["timings"]["prompt_n"] == 3500
        text = _log()[mark:]
        return [w for w in _ubatch_widths(text) if w > 3]

    alone = prefill(1000)
    assert alone, "no ubatch was recorded, LLAMA_BATCH_DEBUG did not reach the log"

    def decoder(i):
        server.make_request("POST", "/completion", data={
            "prompt": list(range(20000 + 100 * i, 20000 + 100 * i + 8)), "n_predict": 100000,
            "cache_prompt": False, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        }, timeout=600)

    threads = [threading.Thread(target=decoder, args=(i,), daemon=True) for i in range(3)]
    for t in threads:
        t.start()

    try:
        _await(lambda: sum(1 for s in server.make_request("GET", "/slots").body if s["is_processing"] and s["n_prompt_tokens"] > 0) >= 3,
               "the decoders starting", poll=0.1)
        time.sleep(1.0)

        beside = prefill(50000)
    finally:
        server.stop()
        for t in threads:
            t.join(30)

    assert beside == alone, f"alone {alone}, beside three decoders {beside}"


def test_slots_reports_a_transferring_slot_apart_from_a_parked_one():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "1"
    _start_async(n_ctx=256)

    done = []
    t = threading.Thread(target=lambda: done.extend(_complete_all(900)))
    t.start()
    seen_parked = seen_transferring = False
    try:
        # polled for as long as the requests run, however long that is on this host
        while t.is_alive() and not (seen_parked and seen_transferring):
            res = server.make_request("GET", "/slots")
            assert res.status_code == 200
            for slot in res.body:
                assert not (slot["is_preempted"] and slot["is_transferring"]), slot
                if slot["is_transferring"]:
                    seen_transferring = True
                    assert slot["n_prompt_tokens"] > 0, "a slot with a copy in flight still holds its cells"
                seen_parked = seen_parked or slot["is_preempted"]
    finally:
        t.join(1800)

    assert len(done) == 2, done
    for res in done:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] > 0
    assert seen_parked, "no parked slot was ever reported"
    assert seen_transferring, "no slot with a copy in flight was ever reported"


def _shift_completion(n_predict: int):
    """A completion whose context shifts, on a token prompt so its length is exact."""
    return server.make_request("POST", "/completion", data={
        "prompt": [1] + list(range(10, 70)), "n_predict": n_predict, "n_keep": 16, "n_discard": 64,
        "ignore_eos": True, "return_tokens": True, "cache_prompt": False, "temperature": 0.0, "seed": 42,
    })


def test_a_park_right_after_a_context_shift_does_not_change_the_output():
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    _start(n_ctx=256, n_batch=32, n_ubatch=32, enable_ctx_shift=True, cache_ram=0)

    n_predict = 320
    reference = _shift_completion(n_predict)
    assert reference.status_code == 200, reference.body
    assert reference.body["timings"]["predicted_n"] == n_predict
    n_prompt = reference.body["timings"]["prompt_n"]
    server.stop()

    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "8192"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = str(256 - n_prompt)
    _start()

    parked = _shift_completion(n_predict)
    assert parked.status_code == 200, parked.body
    assert parked.body["timings"]["predicted_n"] == n_predict

    text = _log()
    assert "slot context shift" in text
    _assert_recovered(text, "preempted on request")
    first_diff = next((i for i, (a, b) in enumerate(zip(reference.body["tokens"], parked.body["tokens"])) if a != b), None)
    assert first_diff is None, f"the parked run diverged at token {first_diff}"


def test_a_sibling_prompt_with_an_invalid_token_is_refused_before_anything_streams():
    _start(n_ctx=256, n_slots=2, n_batch=256)

    res = server.make_request("POST", "/completion", data={
        "prompt": [[1] * 240, [1] * 240, [9999999]], "n_predict": 4, "temperature": 0.0, "seed": 42,
    })
    assert res.status_code == 400, res.body
    assert "invalid tokens" in str(res.body)

    text = _log()
    _assert_nothing_parked(text)


def test_a_recompute_park_bounds_its_draft_by_the_tokens_it_comes_back_with():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "1"
    server.spec_type = "ngram-mod"
    _start(n_ctx=2048, n_slots=2, n_batch=2048, n_ubatch=512, spec_ngram_mod_n_max=128, spec_ngram_mod_n_min=1)

    prompt = [1] + list(range(10, 110)) * 19 + list(range(10, 109))
    assert len(prompt) == 2000
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt, "n_predict": 24, "ignore_eos": True, "temperature": 0.0, "seed": 42, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    assert res.body["tokens_predicted"] == 24

    text = _log()
    assert "tokens to re-prefill" in text
    assert "cannot fit the pool" not in text
    assert "Context size has been exceeded" not in text


def test_a_ram_park_does_not_restart_a_probabilistic_draft():
    """A park that keeps the state in host RAM resumes the slot as it was, draft sampler included: the sampled output is that of a run never parked."""
    os.environ["LLAMA_ARG_SPEC_DRAFT_SAMPLING"] = "probabilistic"
    os.environ["LLAMA_ARG_SPEC_DRAFT_P_MIN"] = "0.0"

    def run(park_every):
        if park_every:
            os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = str(park_every)
        else:
            os.environ.pop("LLAMA_SERVER_PREEMPT_EVERY", None)
        server.spec_type = "draft-simple"
        server.model_draft_hf_repo = server.model_hf_repo
        _start(n_ctx=512, spec_draft_n_max=4, spec_draft_n_min=1)
        res = server.make_request("POST", "/completion", data={
            "prompt": "Once upon a time", "n_predict": 48, "ignore_eos": True,
            "temperature": 0.8, "seed": 123, "cache_prompt": False, "return_tokens": True,
        })
        server.stop()
        assert res.status_code == 200, res.body
        return res.body

    plain = run(0)
    parked = run(8)
    assert parked["preempt"]["parks"] > 0 and parked["preempt"]["recomputes"] == 0, parked["preempt"]
    assert parked["tokens"] == plain["tokens"]


def test_props_says_whether_exact_concurrency_is_running():
    _start(n_ctx=256)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["exact_concurrency"] is False


def test_props_reports_exact_concurrency_on():
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    _start(n_ctx=512, n_slots=2, n_batch=512, n_ubatch=128, fa="on", n_gpu_layer=99)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["exact_concurrency"] is True


def test_a_recompute_park_under_exact_concurrency_says_it_is_not_byte_identical():
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "4"
    _start(n_ctx=512, n_slots=2, n_batch=512, n_ubatch=128, fa="on", n_gpu_layer=99)

    res = _complete(16, "Once upon a time")
    assert res.status_code == 200, res.body
    text = _log()
    assert "tokens to re-prefill" in text
    assert "not guaranteed byte-identical" in text


def test_a_recompute_park_is_reported_to_the_client_and_to_metrics():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=2048, n_batch=2048)

    n_predict = 24
    prompt = _prompt_of(1950, _PROMPT_C)
    comments, final = _stream_completion(n_predict, prompt)

    assert "error" not in final, final
    assert final["tokens_predicted"] == n_predict
    assert final["preempt"]["parks"] >= 2, final["preempt"]
    assert final["preempt"]["recomputes"] == final["preempt"]["parks"], final["preempt"]

    resumed = [i for i, c in enumerate(comments) if c.startswith(": resumed")]
    assert len(resumed) == final["preempt"]["recomputes"], comments
    for i in resumed:
        assert comments[i + 1].startswith(": recomputed"), comments

    metrics = _metrics()
    assert metrics["preempt_recompute_total"] == final["preempt"]["recomputes"]
    assert metrics["preempt_recompute_total"] == metrics["n_preempt_total"]

    plain = server.make_request("POST", "/completion", data={
        "n_predict": 4, "prompt": _PROMPT_B, "temperature": 0.0, "seed": 42,
    })
    assert plain.status_code == 200, plain.body
    assert plain.body["preempt"] == {"parks": 0, "recomputes": 0}


def test_a_recompute_restore_does_not_count_its_replay_as_prompt():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=2048, n_batch=2048)

    n_prompt = 1950
    n_predict = 24
    assert _complete(n_predict, _prompt_of(n_prompt, _PROMPT_A)).status_code == 200
    res = _complete(n_predict, _prompt_of(n_prompt, _PROMPT_C))

    assert res.status_code == 200, res.body
    assert res.body["preempt"]["recomputes"] >= 1, res.body["preempt"]

    timings = res.body["timings"]
    assert timings["cache_n"] == 0, timings
    assert timings["prompt_n"] == n_prompt, timings
    assert timings["predicted_n"] == n_predict, timings


def test_slots_reports_the_recomputes_of_the_current_task():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=2048, n_batch=2048)

    done = []
    prompt = _prompt_of(1950, _PROMPT_C)
    t = threading.Thread(target=lambda: done.append(_complete(64, prompt)))
    t.start()

    seen = 0
    try:
        while seen == 0 and t.is_alive():
            res = server.make_request("GET", "/slots")
            assert res.status_code == 200
            for slot in res.body:
                assert slot["n_recompute"] <= slot["n_preempt"]
                seen = max(seen, slot["n_recompute"])
            time.sleep(0.02)
    finally:
        t.join(1800)

    assert seen > 0, "no slot ever reported a recompute park"
    assert done and done[0].status_code == 200, done
    assert done[0].body["preempt"]["recomputes"] >= seen


def test_a_swap_park_is_not_reported_as_a_recompute():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512)

    comments, final = _stream_completion(24, _PROMPT_A)

    assert final["preempt"]["parks"] >= 1, final["preempt"]
    assert final["preempt"]["recomputes"] == 0, final["preempt"]
    assert ": resumed" in comments and not any(c.startswith(": recomputed") for c in comments), comments
    assert _metrics()["preempt_recompute_total"] == 0


def _stream_responses(n_predict: int, prompt: str) -> dict:
    """One streaming /v1/responses request: the data of its response.completed event."""
    url = f"http://{server.server_host}:{server.server_port}/v1/responses"
    res = requests.post(url, json={
        "model": "test", "input": prompt, "max_output_tokens": n_predict,
        "temperature": 0.0, "stream": True,
    }, stream=True, timeout=600)
    assert res.status_code == 200, res.text
    completed = None
    for raw in res.iter_lines():
        line = raw.decode("utf-8")
        if line.startswith("data: "):
            data = json.loads(line[6:])
            if data.get("type") == "response.completed":
                completed = data
    assert completed is not None, "the stream never reached response.completed"
    return completed


def test_a_streamed_response_carries_the_preempt_record_where_a_plain_one_does():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512)

    completed = _stream_responses(24, _PROMPT_A)

    assert completed["response"]["preempt"]["parks"] >= 1, completed["response"]
    assert completed["response"]["preempt"]["recomputes"] == 0, completed["response"]

    plain = server.make_request("POST", "/v1/responses", data={
        "model": "test", "input": _PROMPT_B, "max_output_tokens": 4, "temperature": 0.0,
    })
    assert plain.status_code == 200, plain.body
    assert sorted(plain.body["preempt"]) == sorted(completed["response"]["preempt"]) == ["parks", "recomputes"]


def test_two_image_chats_that_outgrow_the_parking_budget_both_finish():
    os.environ["LLAMA_MEDIA_MARKER"] = "<__media__>"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    server.model_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"
    server.model_hf_file = None
    server.model_alias = "tinygemma3"
    _start(n_ctx=1024, n_slots=2, n_batch=64, n_ubatch=64)

    image = base64.b64encode(requests.get(_IMG_URL, timeout=60).content).decode()
    prompt = {"prompt_string": "<__media__>\nWhat is in this image?", "multimodal_data": [image]}
    n_predict = 700
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for _ in range(2)
    ])

    text = _log()
    assert "Context size has been exceeded" not in text
    assert "failed to process mtmd chunk" not in text
    assert "tokens to re-prefill" in text, "no park fell back to recompute"
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict


def test_a_park_during_the_prefill_does_not_count_the_restarted_prompt_twice():
    os.environ["LLAMA_SERVER_PREEMPT_FAIL_SAVE"] = "1"
    _start(n_ctx=2048, n_batch=256)

    resident = []
    t = threading.Thread(target=lambda: resident.append(_complete(1900, _PROMPT_A)), daemon=True)
    t.start()

    _await(lambda: any(s.get("n_prompt_tokens", 0) >= 1600 for s in server.make_request("GET", "/slots").body),
           "the resident growing into the pool", alive=t.is_alive, poll=0.005)

    n_prompt = 500
    n_predict = 8
    comments, final = _stream_completion(n_predict, _prompt_of(n_prompt, _PROMPT_B))
    t.join(120)

    assert resident and resident[0].status_code == 200, resident
    assert "error" not in final, final
    assert comments and comments[0] == ": preempted", comments

    parks = [(int(cells), int(again)) for cells, again in re.findall(
        r"preempted: (\d+) cells dropped .*? (\d+) tokens to re-prefill", _log())]
    assert any(0 < cells < again for cells, again in parks), parks

    timings = final["timings"]
    assert timings["prompt_n"] == n_prompt, timings
    assert timings["cache_n"] == 0, timings
    assert timings["predicted_n"] == n_predict, timings


def _slot_view(slot_id: int) -> dict:
    return next(s for s in server.make_request("GET", "/slots").body if s["id"] == slot_id)


def _n_decoded(slot: dict) -> int:
    return slot.get("next_token", [{}])[0].get("n_decoded", 0) if slot.get("is_processing") else 0


def _raw_completion(n_predict: int, prompt, id_slot: int = -1, timeout: float = DEFAULT_REQUEST_TIMEOUT):
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict, "prompt": prompt, "id_slot": id_slot, "cache_prompt": False,
        "ignore_eos": True, "temperature": 0.0, "seed": 42,
    }, timeout=timeout)


def _serve_with_training_context(n_ctx_train: int):
    """Serve a copy of the model that claims a longer training context, which caps the slot context: a generation that outlasts the 2 s a parked head waits needs more than 2048 tokens."""
    _start(n_ctx=256)
    path = server.make_request("GET", "/props").body["model_path"]
    server.stop()
    with open(path, "rb") as f:
        data = bytearray(f.read())
    key = b"llama.context_length"
    at = data.find(key) + len(key)
    assert at > len(key) and struct.unpack_from("<I", data, at)[0] == 4, "not a uint32 llama.context_length"
    struct.pack_into("<I", data, at + 4, n_ctx_train)
    fd, out = tempfile.mkstemp(suffix=".gguf")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    _scratch.append(out)
    server.model_file = out
    server.model_hf_repo = server.model_hf_file = None


def test_a_started_prompt_keeps_the_chunks_cache_reuse_shifts_in():
    """Trimming a started slot to its common prefix ahead of the batch builder must not discard what --cache-reuse would shift back in."""
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=512, n_batch=64, n_ubatch=32, cache_ram=0)

    def run(prompt, id_slot, n_cache_reuse=0):
        res = server.make_request("POST", "/completion", data={
            "n_predict": 1, "prompt": prompt, "id_slot": id_slot, "cache_prompt": True,
            "n_cache_reuse": n_cache_reuse, "temperature": 0.0, "seed": 42,
        })
        assert res.status_code == 200, res.body
        return res.body["timings"]

    run([1] * 50 + [2] * 50 + [3] * 200, 0)
    run([4] * 180, 1)
    timings = run([1] * 50 + [3] * 200, 0, n_cache_reuse=50)

    assert timings["cache_n"] > 200, f"the reusable chunk was trimmed away: {timings}"


def test_a_parked_head_is_not_overtaken_by_a_stream_of_short_requests():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "48"
    # On CUDA a prompt chunk across position 2048 of this model hangs llama_decode (also on base), so run on CPU; the code under test is host-side.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _serve_with_training_context(8192)
    _start(n_ctx=8192, n_slots=4, n_batch=1024)

    short_prompt = _prompt_of(1700, _PROMPT_B)
    head_prompt = _prompt_of(7200, _PROMPT_A)
    n_short, n_head = 32, 100
    t_stream_max = 240.0

    stop = threading.Event()
    shorts = []
    lock = threading.Lock()

    def stream(id_slot):
        while not stop.is_set():
            res = _raw_completion(n_short, short_prompt, id_slot)
            with lock:
                shorts.append((time.time(), res))

    workers = [threading.Thread(target=stream, args=(i % 3,), daemon=True) for i in range(6)]
    for w in workers:
        w.start()

    try:
        _await(lambda: len(shorts) >= 6, "the short-request stream starting", poll=0.01)
        # overtaken, the head waits until the stream stops: give it a span many short requests long, as slow as the host is
        t_first_short = max((r.body["timings"]["prompt_ms"] + r.body["timings"]["predicted_ms"]) / 1e3 for _, r in list(shorts))
        t_stream_max = max(t_stream_max, 30 * t_first_short)
        t_head = time.time()
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(_raw_completion, n_head, head_prompt, 3, t_stream_max + 600)
            while not fut.done() and time.time() - t_head < t_stream_max:
                time.sleep(0.05)
            stopped_early = not fut.done()
            stop.set()
            head = fut.result()
    finally:
        stop.set()
        for w in workers:
            w.join(120)

    assert head.status_code == 200, head.body
    assert head.body["timings"]["predicted_n"] == n_head
    for _, res in shorts:
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_short

    text = _log()
    assert "Context size has been exceeded" not in text
    assert not stopped_early, f"the head was still waiting when the stream was stopped after {t_stream_max} s"
    assert "prompt held:" in text, "no prompt was held back for the head"
    waits = [float(w) for w in re.findall(r"id +3 \| task \d+ \| resumed after ([\d.]+) s", text)]
    assert waits, "the head was never parked and resumed"
    # overtaken by the stream, the head would wait until the stream stopped (t_stream_max); waiting its turn, a few short requests' worth
    t_short = max(
        (res.body["timings"]["prompt_ms"] + res.body["timings"]["predicted_ms"]) / 1e3 for _, res in shorts
    )
    t_bound = max(10.0, 3 * t_short)
    assert max(waits) < t_bound, f"the head waited {waits} s behind requests that came after it (bound {t_bound:.1f} s, a short request took up to {t_short:.1f} s)"


@pytest.mark.slow
@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_a_long_resident_that_never_shifts_is_rotated_out_for_an_aged_head(policy):
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _serve_with_training_context(16384)
    _start(n_ctx=16384, n_slots=2, n_batch=2048)

    done = {}

    def run(name, n_predict, prompt, id_slot):
        res = _raw_completion(n_predict, prompt, id_slot, timeout=1800)
        done[name] = (time.time(), res)

    a = threading.Thread(target=run, args=("a", 16000, _prompt_of(64, _PROMPT_A), 0))
    t_a = time.time()
    a.start()
    _await(lambda: _n_decoded(_slot_view(0)) >= 2000, "the resident decoding 2000 tokens", alive=a.is_alive, poll=0.01)
    t_per_token = (time.time() - t_a) / 2000
    b = threading.Thread(target=run, args=("b", 8000, _prompt_of(1500, _PROMPT_B), 1))
    b.start()
    a.join(1800)
    b.join(1800)

    for name in ("a", "b"):
        assert done[name][1].status_code == 200, done[name][1].body
    assert done["a"][1].body["timings"]["predicted_n"] == 16000
    assert done["b"][1].body["timings"]["predicted_n"] == 8000

    text = _log()
    assert "Context size has been exceeded" not in text
    assert "slot context shift" not in text
    waits = [float(w) for w in re.findall(r"id +1 \| task \d+ \| resumed after ([\d.]+) s", text)]
    assert waits, "the second request was never parked"
    # waiting for the resident to end would take its remaining 14000 tokens; a rotation takes a small part of that, on any host
    t_bound = max(90.0, 0.5 * 14000 * t_per_token)
    assert max(waits) < t_bound, f"the parked request waited {waits} s behind a resident that ends by itself (bound {t_bound:.0f} s)"


def test_one_victim_that_covers_the_shortfall_is_parked_instead_of_several_small_ones():
    _serve_with_training_context(8192)
    _start(n_ctx=4864, n_slots=4, n_batch=2048)

    small = [_prompt_of(400, _PROMPT_A), _prompt_of(400, _PROMPT_B)]
    medium = _prompt_of(1400, _PROMPT_C)
    leader = _prompt_of(3267, _PROMPT_A)

    with ThreadPoolExecutor(4) as pool:
        residents = [pool.submit(_raw_completion, 300, p, i) for i, p in enumerate(small + [medium])]
        _await(lambda: all(_n_decoded(s) > 0 for s in server.make_request("GET", "/slots").body if s["id"] < 3),
               "the three residents decoding", alive=lambda: not all(r.done() for r in residents), poll=0.002)
        last = pool.submit(_raw_completion, 8, leader, 3)
        results = [r.result() for r in residents] + [last.result()]

    for res, n_predict in zip(results, (300, 300, 300, 8)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict

    text = _log()
    assert "Context size has been exceeded" not in text
    first_resume = text.find("resumed after")
    assert first_resume > 0, "nothing was parked"
    parks = [int(n) for n in re.findall(r"preempted: (\d+) cells", text[:first_resume])]
    assert len(parks) == 1, f"parked {parks} where one resident covered the shortfall"
    assert parks[0] >= 1400, parks


@pytest.mark.parametrize("ram", ["8192", "0", "unset"])
def test_props_reports_the_preemption_setup(ram):
    if ram == "unset":
        os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)
    else:
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = ram
    _start(n_ctx=256)

    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    preempt = res.body["preempt"]
    assert sorted(preempt) == ["async", "enabled", "granularity", "notices", "policy", "ram_mib", "recurrent"]
    assert preempt["enabled"] is (ram == "8192")
    assert preempt["ram_mib"] == (8192 if ram == "8192" else 0)
    assert preempt["async"] is False or ram == "8192"
    assert isinstance(preempt["async"], bool)
    assert preempt["notices"] == 2
    assert preempt["recurrent"] is False
    assert preempt["granularity"] == 1
    assert preempt["policy"] == (_policy or "fcfs")


def test_props_reports_preemption_off_without_a_unified_cache():
    _start(n_ctx=256, kv_unified=False)
    preempt = server.make_request("GET", "/props").body["preempt"]
    assert preempt["enabled"] is False
    assert preempt["ram_mib"] == 8192
    assert preempt["async"] is False


def test_props_reports_the_planning_granularity():
    os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    _start(n_ctx=1024)
    preempt = server.make_request("GET", "/props").body["preempt"]
    assert preempt["enabled"] is True
    assert preempt["granularity"] == 64


def test_a_growing_sequence_keeps_its_pinned_buffer_across_parks():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start_async(n_ctx=2048)

    res = _complete(400, _prompt_of(1000, _PROMPT_A))
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 400

    text = _log()
    parks = text.count("park completed after")
    sizes = re.findall(r"pinned park buffer (?:allocated|grown)", text)
    assert parks >= 40, parks
    assert 1 <= len(sizes) <= 4, f"{len(sizes)} pinned allocations for {parks} parks"


def test_a_slot_whose_park_buffer_comes_back_pageable_is_no_longer_reported_async():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    os.environ["LLAMA_SERVER_PREEMPT_PAGEABLE"] = "1"
    _start_async(n_ctx=512)
    assert server.make_request("GET", "/props").body["preempt"]["async"] is True

    res = _complete(32)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 32
    assert res.body["preempt"]["parks"] > 0, res.body["preempt"]

    assert "is pageable, so this slot parks synchronously from now on" in _log()
    assert server.make_request("GET", "/props").body["preempt"]["async"] is False


def test_a_sleeping_server_reports_the_async_fallback_too():
    """/props is answered from a cache while the server sleeps; it must carry the same live async value as the awake answer."""
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    os.environ["LLAMA_SERVER_PREEMPT_PAGEABLE"] = "1"
    _start_async(n_ctx=512, sleep_idle_seconds=2)

    res = _complete(32)
    assert res.status_code == 200, res.body
    assert res.body["preempt"]["parks"] > 0, res.body["preempt"]
    assert "is pageable, so this slot parks synchronously from now on" in _log()

    deadline = time.time() + 30
    props = server.make_request("GET", "/props").body
    while not props["is_sleeping"] and time.time() < deadline:
        time.sleep(0.1)
        props = server.make_request("GET", "/props").body
    assert props["is_sleeping"], "the server did not go to sleep"
    assert props["preempt"]["async"] is False, props["preempt"]


def test_a_multi_prompt_oai_completion_counts_the_parks_of_every_prompt():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start()

    body = {"prompt": [_PROMPT_A, _PROMPT_B], "max_tokens": 40, "temperature": 0.0}

    per_prompt = server.make_request("POST", "/completion", data=body)
    assert per_prompt.status_code == 200, per_prompt.body
    counts = [r["preempt"] for r in per_prompt.body]
    assert all(c["parks"] > 0 for c in counts), counts

    merged = server.make_request("POST", "/v1/completions", data=body)
    assert merged.status_code == 200, merged.body
    assert len(merged.body["choices"]) == 2
    assert merged.body["preempt"]["parks"] == sum(c["parks"] for c in counts), (merged.body["preempt"], counts)
    assert merged.body["preempt"]["recomputes"] == sum(c["recomputes"] for c in counts), (merged.body["preempt"], counts)


def _cached(n_predict: int, prompt: list, id_slot: int):
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict, "prompt": prompt, "id_slot": id_slot, "cache_prompt": True,
        "ignore_eos": True, "temperature": 0.0, "seed": 42,
    })


def test_a_started_slot_replacing_its_prompt_does_not_purge_an_idle_slot():
    _start(n_ctx=256, n_batch=128, n_ubatch=64, cache_ram=0)
    results = [_cached(1, [tok] * 100, slot) for slot, tok in ((0, 11), (1, 12), (0, 13), (1, 12))]
    for res in results:
        assert res.status_code == 200, res.body
    assert results[3].body["timings"]["prompt_n"] == 1, results[3].body["timings"]
    text = _log()
    assert "purging slot" not in text
    _assert_nothing_parked(text)


def test_an_idle_slot_inside_the_margin_keeps_its_cache():
    _start(n_ctx=256, n_batch=128, n_ubatch=64, cache_ram=0)
    first = [11] * 120
    assert _cached(1, first, 0).status_code == 200
    assert _cached(12, [12] * 120, 1).status_code == 200
    res = _cached(1, first, 0)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["prompt_n"] == 1, res.body["timings"]
    assert "purging slot" not in _log()


def test_a_started_slot_that_fits_once_trimmed_parks_nobody():
    _start(n_ctx=1024, n_slots=3, n_batch=512, n_ubatch=512, cache_ram=0)
    old = [100 + i % 50 for i in range(600)]
    idle = [300 + i % 30 for i in range(60)]
    assert _cached(1, old, 0).status_code == 200
    assert _cached(1, idle, 2).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        gen = pool.submit(_cached, 220, [200 + i for i in range(90)], 1)
        _wait_processing([1])
        started = _cached(4, old[:300] + [400 + i % 100 for i in range(336)], 0)
        generating = gen.result()
    for res in (started, generating):
        assert res.status_code == 200, res.body
        assert res.body["preempt"]["parks"] == 0, res.body["preempt"]
    rerun = _cached(1, idle, 2)
    assert rerun.body["timings"]["cache_n"] == 59, rerun.body["timings"]
    text = _log()
    assert "purging slot" not in text
    _assert_nothing_parked(text)


def _reparked_mid_reprefill(text: str) -> list:
    """(slot, cells parked, tokens it was re-prefilling) for every park that hit a slot before its re-prefill finished."""
    pending, res = {}, []
    for line in text.splitlines():
        m = re.search(r"id +(\d+) \| task \d+ \| resumed after [\d.]+ s: (\d+) tokens to re-prefill", line)
        if m:
            pending[m.group(1)] = int(m.group(2))
            continue
        m = re.search(r"id +(\d+) \| task \d+ \| (?:preempted|rotated out)[^:]*: (\d+) cells", line)
        if m and m.group(1) in pending:
            need = pending.pop(m.group(1))
            if int(m.group(2)) < need:
                res.append((m.group(1), int(m.group(2)), need))
    return res


def test_a_resumed_re_prefill_keeps_its_room_until_it_is_done():
    os.environ["LLAMA_SERVER_PREEMPT_FAIL_SAVE"] = "2"
    _start(n_ctx=1024, n_slots=4, n_batch=128, n_ubatch=128)
    prompts = [[10 + 20 * i + k % 20 for k in range(40)] for i in range(4)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: server.make_request("POST", "/completion", data={
            "n_predict": 900, "prompt": prompts[i], "id_slot": i, "cache_prompt": False,
            "ignore_eos": True, "temperature": 0.0, "seed": 42}), range(4)))
    _assert_completed(results, 900)
    text = _log()
    assert "tokens to re-prefill on resume" in text, "nothing was parked by dropping its cells"
    assert _reparked_mid_reprefill(text) == []


def _oai_n(prompt: list, n: int, max_tokens: int, timeout: float = DEFAULT_REQUEST_TIMEOUT):
    return server.make_request("POST", "/v1/completions", data={
        "prompt": prompt, "n": n, "max_tokens": max_tokens, "ignore_eos": True,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    }, timeout=timeout)


def test_a_single_request_beside_an_n_2_request_is_parked_so_both_finish():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=1024, n_slots=3, n_batch=128, n_ubatch=128)
    with ThreadPoolExecutor(1) as pool:
        single = pool.submit(_cached, 300, [100 + k % 50 for k in range(600)], 0)
        _wait_processing([0])
        family = _oai_n([200 + k % 50 for k in range(200)], 2, 250)
        single = single.result()
    assert single.status_code == 200, single.body
    assert single.body["timings"]["predicted_n"] == 300
    assert family.status_code == 200, family.body
    assert [c["finish_reason"] for c in family.body["choices"]] == ["length", "length"]
    _assert_recovered(_log())


def test_two_n_2_requests_that_do_not_fit_together_end_only_one():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=1024, n_slots=4, n_batch=128, n_ubatch=128)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(_oai_n, [100 + k % 50 for k in range(200)], 2, 250)
        _await(lambda: sum(s["is_processing"] for s in server.make_request("GET", "/slots").body) >= 2,
               "the first family on its two slots", alive=lambda: not first.done(), poll=0.005)
        second = pool.submit(_oai_n, [200 + k % 50 for k in range(200)], 2, 250)
        results = [first.result(), second.result()]
    assert sorted(r.status_code for r in results) == [200, 500], [r.body for r in results]
    ok = next(r for r in results if r.status_code == 200)
    assert [c["finish_reason"] for c in ok.body["choices"]] == ["length", "length"]
    assert "ending it alone" in _log()


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_an_n_2_prompt_that_parked_an_older_request_is_not_held_behind_it(policy):
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _start(n_ctx=1024, n_slots=3, n_batch=128, cache_ram=0)
    with ThreadPoolExecutor(1) as pool:
        single = pool.submit(_raw_completion, 300, [100 + k % 50 for k in range(600)], 0, 900)
        _await(lambda: _n_decoded(_slot_view(0)) > 4, "the single request decoding", alive=lambda: not single.done(), poll=0.002)
        family = _oai_n([200 + k % 50 for k in range(800)], 2, 1, timeout=900)
        single = single.result()
    assert single.status_code == 200, single.body
    assert single.body["timings"]["predicted_n"] == 300
    assert family.status_code == 200, family.body
    assert [c["finish_reason"] for c in family.body["choices"]] == ["length", "length"]
    _assert_recovered(_log())


def test_a_started_request_reusing_a_long_prefix_does_not_park_an_older_generating_one():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=1024, n_slots=2, n_batch=128, n_ubatch=128, cache_ram=0)
    earlier = [200 + k % 50 for k in range(600)]
    assert _cached(1, earlier, 1).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 600, [100 + k % 50 for k in range(40)], 0)
        _await(lambda: _n_decoded(_slot_view(0)) >= 280, "the older request decoding 280 tokens", alive=lambda: not older.done(), poll=0.002)
        newer = _cached(4, earlier + [300 + k % 50 for k in range(300)], 1)
        older = older.result()
    for res, n_predict in ((older, 600), (newer, 4)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    assert older.body["preempt"]["parks"] == 0, "the older generating request was parked for a newer one"
    assert newer.body["preempt"]["parks"] > 0


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_the_policy_decides_whether_the_newest_or_the_cheapest_slot_is_parked(policy):
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _start(n_ctx=1024, n_slots=3, n_batch=128, n_ubatch=128, cache_ram=0)
    assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    assert _cached(1, [300 + k % 50 for k in range(150)], 2).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 600, [100 + k % 50 for k in range(40)], 0)
        _await(lambda: _n_decoded(_slot_view(0)) >= 20, "the older request decoding 20 tokens", alive=lambda: not older.done(), poll=0.005)
        newer = _cached(500, [200 + k % 50 for k in range(300)], 1)
        older = older.result()
    for res, n_predict in ((older, 600), (newer, 500)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    if policy == "fcfs":
        assert older.body["preempt"]["parks"] == 0
        assert newer.body["preempt"]["parks"] > 0
    else:
        assert older.body["preempt"]["parks"] + newer.body["preempt"]["parks"] > 0
    text = _log()
    assert 0 <= text.find("purging slot 2") < text.find("preempted:"), "a running slot was parked while an idle one held cells"


def test_fcfs_resumes_parked_requests_in_arrival_order():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = "fcfs"
    _start(n_ctx=1024, n_slots=3, n_batch=128, n_ubatch=128)
    assert server.make_request("GET", "/props").body["preempt"]["policy"] == "fcfs"
    with ThreadPoolExecutor(3) as pool:
        futures = []
        for i, n_predict in enumerate((700, 450, 450)):
            futures.append(pool.submit(_cached, n_predict, [100 * (i + 1) + k % 50 for k in range(40)], i))
            _wait_processing([i])
        results = [f.result() for f in futures]
    for res, n_predict in zip(results, (700, 450, 450)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    assert results[0].body["preempt"]["parks"] == 0, "the oldest request was parked"
    text = _log()
    resumed = [int(m) for m in re.findall(r"id +(\d+) \| task \d+ \| resumed after", text)]
    assert resumed, "nothing was parked"
    parked = [int(m) for m in re.findall(r"id +(\d+) \| task \d+ \| preempted:", text)]
    assert parked[0] == 2, parked
    if 1 in parked:
        assert resumed.index(1) < resumed.index(2), resumed


def test_fcfs_runs_the_oldest_to_the_end_of_the_window_beside_a_held_prompt():
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = "fcfs"
    # a long window: the tiny model decodes thousands of tokens a second, and the park must last long enough to be seen;
    # not longer, since a loaded CI runner prefills and decodes it hundreds of times slower
    _serve_with_training_context(4096)
    _start(n_ctx=4096, n_slots=3, n_batch=256, n_ubatch=256, cache_ram=0)
    with ThreadPoolExecutor(2) as pool:
        # a livelock never ends, so the bound is the test's verdict; it is generous for a loaded host
        oldest = pool.submit(lambda: server.make_request("POST", "/completion", timeout=600, data={
            "n_predict": 100000, "prompt": [100 + k % 50 for k in range(20)], "id_slot": 0, "cache_prompt": True,
            "ignore_eos": True, "temperature": 0.0, "seed": 42}))
        _await(lambda: _n_decoded(_slot_view(0)) >= 100, "the oldest generating 100 tokens", alive=lambda: not oldest.done())
        # 120 cells and growing beside 4016: it cannot fit, so it is parked and is owed the cells first
        parked = pool.submit(_cached, 16, [200 + k % 50 for k in range(4000)], 1)
        assert _wait_preempted(120), "the second request was never parked, so the case was not reached"
        # held behind the parked one: a running slot, but no candidate, so the oldest is the only one left to pick
        held = _cached(16, [300 + k % 50 for k in range(30)], 2)
        oldest = oldest.result()
        parked = parked.result()
    for res in (oldest, parked, held):
        assert res.status_code == 200, res.body
    assert oldest.body["stop_type"] == "limit", oldest.body
    # a slow host may rotate the oldest out once for the head parked 30 s; the livelock parks it thousands of times
    assert oldest.body["preempt"]["parks"] <= 2, f"the oldest was parked {oldest.body['preempt']['parks']} times"
    assert parked.body["preempt"]["parks"] > 0
    assert "prompt held" in _log(), "no prompt was held behind the parked one, so the case was not reached"


def test_a_started_slot_trimmed_below_its_sliding_window_prefills_again():
    server.model_hf_repo = "ggml-org/tinygemma3-GGUF:Q8_0"
    server.model_hf_file = None
    server.model_alias = "tinygemma3"
    server.override_kvs = ["gemma3.attention.sliding_window=int:32"]
    _start(n_ctx=512, n_slots=2, n_batch=64, n_ubatch=64, cache_ram=0, n_threads=4)
    stale = [100 + k % 50 for k in range(200)]
    assert _cached(1, stale, 1).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        busy = pool.submit(_cached, 400, [300 + k % 50 for k in range(40)], 0)
        _await(lambda: _n_decoded(_slot_view(0)) >= 230, "the busy slot filling the pool", alive=lambda: not busy.done())
        # 40 tokens in common, far below the 32 newest cells the window kept of the stale 200
        res = _cached(16, stale[:40] + [400 + k % 50 for k in range(60)], 1)
        busy = busy.result()
    for r in (res, busy):
        assert r.status_code == 200, r.body
    text = _log()
    assert "trimmed to the" in text, "the started slot was not trimmed, so the case was not reached"
    assert "should not happen" not in text


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_the_old_policy_knob_selects_the_policy(policy):
    os.environ.pop("LLAMA_ARG_PREEMPT_POLICY", None)
    os.environ["LLAMA_SERVER_PREEMPT_POLICY"] = policy
    try:
        _start(n_ctx=256)
        assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    finally:
        os.environ.pop("LLAMA_SERVER_PREEMPT_POLICY", None)


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_the_command_line_policy_wins_over_the_old_knob(policy, monkeypatch):
    os.environ.pop("LLAMA_ARG_PREEMPT_POLICY", None)
    os.environ["LLAMA_SERVER_PREEMPT_POLICY"] = "cost" if policy == "fcfs" else "fcfs"
    popen = subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda args, *a, **k: popen([*args, "--preempt-policy", policy], *a, **k))
    try:
        _start(n_ctx=256)
        assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    finally:
        os.environ.pop("LLAMA_SERVER_PREEMPT_POLICY", None)


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_a_prompt_dropped_in_its_prefill_waits_until_it_fits_whole(policy):
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _serve_with_training_context(8192)
    _start(n_ctx=4096, n_slots=2, n_batch=256, n_ubatch=256, cache_ram=0)
    assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 1500, [100 + k % 50 for k in range(1500)], 0)
        _await(lambda: _n_decoded(_slot_view(0)) >= 100, "the older request decoding 100 tokens", alive=lambda: not older.done(), poll=0.005)
        newer = _cached(4, [200 + k % 50 for k in range(2500)], 1)
        older = older.result()
    for res, n_predict in ((older, 1500), (newer, 4)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    parked = older if older.body["preempt"]["parks"] > 0 else newer
    assert parked.body["preempt"]["parks"] == 1, (older.body["preempt"], newer.body["preempt"])


def test_oversubscribed_bounded_chats_are_not_time_sliced():
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=2048, n_slots=4, n_batch=256, n_ubatch=256, cache_ram=0)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: _cached(800, [50 + 100 * i + k % 50 for k in range(200)], i), range(4)))
    _assert_completed(results, 800)
    text = _log()
    assert "Context size has been exceeded" not in text
    assert "rotated out after" not in text
    parks = [r.body["preempt"]["parks"] for r in results]
    assert 1 <= sum(parks) <= 4, parks
    assert max(parks) == 1, f"a chat was resumed into room the residents still needed and parked again: {parks}"
