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

# Preemption on a unified KV pool: one slot is parked in host RAM instead of every slot being terminated. Needs --kv-unified.

server = ServerPreset.tinyllama2()

_ASYNC_BANNER = "parking and resuming asynchronously"

_scratch = []  # files a test wrote, removed after it
_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")  # restored after a test that hides the GPUs

_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"
_PROMPT_C = "In a small village by the sea there lived a fisherman who"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    # --preempt-ram defaults to 0, so parking must be enabled.
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
                 "LLAMA_SERVER_PREEMPT_FAIL_SAVE", "LLAMA_ARG_SPEC_DRAFT_P_MIN", "LLAMA_ARG_LOG_VERBOSITY",
                 "LLAMA_BATCH_DEBUG", "LLAMA_ARG_CTX_CHECKPOINTS",
                 "LLAMA_MEDIA_MARKER", "LLAMA_EXACT_CONCURRENCY", "LLAMA_ARG_PREEMPT_POLICY"):
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


def _require_async(text: str):
    if _ASYNC_BANNER not in text:
        pytest.skip("this backend cannot copy asynchronously, the async park path is not exercised")  # ty: ignore[too-many-positional-arguments]


def _complete(n_predict: int, prompt="Hi how are you", id_slot: int = -1, delay: float = 0.0, after_slot_busy=None,
              timeout: float = DEFAULT_REQUEST_TIMEOUT):
    time.sleep(delay)
    if after_slot_busy is not None:
        # Sent once that slot is processing, so the request queues behind it regardless of host speed.
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


def _wait_preempted(timeout: float = 30.0) -> bool:
    """True once some slot is parked: its cells are in host RAM and it wants them back."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        slots = server.make_request("GET", "/slots").body
        if any(s["is_preempted"] for s in slots):
            return True
        time.sleep(0.005)
    return False


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
    # Match park log lines, not the bare word: verbose logs print every slot's "is_preempted".
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
    # Each request fits the pool alone (960 and 600 of 1024 cells) but not together.
    if knob == "pages":
        # A block allocator gives each block to one sequence, so the planner must count cells, not tokens.
        os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    if knob.startswith("last-resort"):
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    if knob == "last-resort-unlimited":
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    n_ctx = 1024
    (_start_async if knob == "async" else _start)(n_ctx=n_ctx)

    # Lengths, not client speed, must force the overlap: equal requests fired together did not overlap on a Windows runner.
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
        # Family members are not victims for each other, so a two-completion request gets the same error as alone.
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    _start(n_ctx=256)

    if knob == "ram-0":
        # The overflow must come from lengths, since equal concurrent requests did not overlap on a Windows runner.
        assert any(res.status_code != 200 for res in _complete_overlapping((110, 100), (120, 60)))
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

    assert any(res.status_code != 200 for res in _complete_overlapping((110, 100), (120, 60)))

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
    # The chunk in the abandoned batch is processed once after the rewind, never twice.
    assert results[1].body["timings"]["prompt_n"] == n_b


def test_a_prompt_parked_before_its_first_token_is_issued_whole():
    # Both prompts are too close to n_ctx for the usual margin, so the second is parked before taking a cell and resumes after the first finishes.
    _start(n_ctx=256, n_batch=256)

    n_prompt = 240
    n_predict = 4
    # Different prompts: with the same one the second could reuse the first's cached cells and prompt_n would be 1.
    together = _complete_all(n_predict, [_prompt_of(n_prompt, _PROMPT_A), _prompt_of(n_prompt, _PROMPT_B)])

    assert "cannot fit the pool" not in _log()
    _assert_completed(together, n_predict)
    for res in together:
        assert res.body["timings"]["prompt_n"] == n_prompt, "the prompt was not issued once and whole"


def test_a_resident_cycling_through_context_shifts_is_rotated_out_for_a_parked_head():
    # With context shift a resident never frees its cells, so once the head has waited it is parked and the two take turns.
    _start(n_slots=3, n_ctx=384, enable_ctx_shift=True)

    # Rotation is clock-based, so a fast host needs a long generation, and a slow host then needs a longer wait.
    n_predict = 9000
    _assert_completed(_complete_all(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C), timeout=1800), n_predict)

    text = _log()
    _assert_recovered(text, "rotated out after")
    assert "slot context shift" in text


def test_a_park_whose_host_allocation_fails_is_parked_by_recompute():
    # A failed save used to stop the planner and let the pool overflow, though the victim could still be parked by dropping its cells.
    os.environ["LLAMA_SERVER_PREEMPT_FAIL_SAVE"] = "1"
    _start(n_ctx=1024, n_slots=2)

    # The leader ends at 960 of 1024 cells, so the second is parked regardless of client lag.
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


def test_a_resident_that_cannot_be_swapped_out_is_rotated_by_recompute():
    # 1 MiB holds no snapshot, so rotation used to refuse every resident and the head waited forever behind a context-shifting one.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    _start(n_ctx=2048, n_slots=2, n_batch=2048, enable_ctx_shift=True)

    # A small n_discard keeps the resident near the end of the pool, so its state never fits the budget.
    def unending_request():
        return server.make_request("POST", "/completion", data={
            "n_predict": 100000, "prompt": _PROMPT_A, "n_keep": 1, "n_discard": 64,
            "ignore_eos": True, "temperature": 0.0, "seed": 42,
        }, timeout=600)

    # The server is stopped with this request in flight, so the disconnect is expected.
    unending = []

    def run_unending():
        try:
            unending.append(unending_request())
        except requests.exceptions.RequestException:
            pass

    t = threading.Thread(target=run_unending)
    t.start()

    try:
        # The resident must be at the pool's limit and cycling before a second prompt cannot fit.
        for _ in range(3000):
            if "slot context shift" in _log():
                break
            time.sleep(0.05)
        else:
            pytest.fail("the resident never reached the end of the pool")  # ty: ignore[invalid-argument-type]

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
    # A cancelled request can reach release() with a copy running, so release must wait for it.
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start_async(n_ctx=512)

    for i in range(4):
        try:
            server.make_request("POST", "/completion", data={
                "n_predict": 96, "prompt": _PROMPT_A, "ignore_eos": True, "temperature": 0.0, "seed": 42,
            }, timeout=0.05 + 0.1 * i)
        except Exception:
            pass

    for _ in range(600):
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        if all(not slot["is_processing"] for slot in res.body):
            break
        time.sleep(0.2)
    else:
        pytest.fail("a slot never came back after a cancel during a copy")  # ty: ignore[invalid-argument-type]
    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["is_transferring"] is False
    assert _metrics()["preempt_ram_bytes"] == 0, "a cancelled slot kept its parked memory"

    res = _complete(16)
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 16


def test_a_started_slot_is_counted_by_the_cells_it_holds_not_by_the_prompt_it_keeps():
    # The last request starts on slot 0 holding the first request's cells; counting only its kept prompt made the pool look free and a restore landed on taken cells.
    _start(n_ctx=1024, n_slots=3)

    # 500 + 200 + 200 fit 1024 cells, so both small slots are parked with their whole prompt once slot 0 grows; slot 0 stays largest and ends at 960 cells, too few for either to return.
    ids = _prompt_of(500, _PROMPT_C)

    with ThreadPoolExecutor(4) as pool:
        first = pool.submit(_complete, 460, ids, 0)
        _wait_processing([0])
        # Queued behind slot 0 and a real prefix of what it holds: it starts on 960 cells keeping 8.
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
    # 1 MiB fits neither sequence, so the park drops the cells and the resume re-prefills; the last resort falls back the same way.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    if planner == "off":
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    # One batch for the re-prefill: this prompt in 32-token batches hangs the CUDA build with graphs on, also on master.
    _start(n_ctx=3840, n_batch=2048)

    n_predict = 2000
    if planner == "on":
        results = parallel_function_calls([(_stream_completion, (n_predict, p)) for p in (_PROMPT_A, _PROMPT_B)])
        for comments, final in results:
            assert "error" not in final, final
            assert final["tokens_predicted"] == n_predict
        comments = [c for cs, _ in results for c in cs]
        assert ": preempted" in comments and ": resumed" in comments, comments
    else:
        _assert_completed(_complete_all(n_predict), n_predict)

    text = _log()
    assert "Context size has been exceeded" not in text
    assert "tokens to re-prefill" in text, "no park fell back to recompute"
    if planner == "off":
        assert "preempted as a last resort by dropping its cells" in text


_IMG_URL = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


def test_a_media_chunk_is_reserved_whole_before_it_is_decoded():
    # A media chunk decodes whole outside the kv-full retry, so unless the planner reserves all its cells the second of two image requests fails mid-chunk.
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
    # Near the end of the pool the draft loop could attempt positions past the planner's reservation.
    path = os.environ.get("LLAMA_SERVER_TEST_MTP_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MTP_MODEL to a gguf carrying an MTP head")  # ty: ignore[too-many-positional-arguments]
    server.model_file = path
    server.model_hf_repo = server.model_hf_file = None
    server.spec_type = "draft-mtp"
    os.environ["LLAMA_ARG_SPEC_DRAFT_P_MIN"] = "0.0"  # only the bounds stop the draft loop
    os.environ["LLAMA_ARG_LOG_VERBOSITY"] = "5"  # logs when the wrapper truncates a draft
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
    # A hybrid's recurrent half compacts active sequences each batch, so a concurrent copy could read a moved row.
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


# [TAG_EXACT_CONCURRENCY] The paged pool places a cell by sequence and position, so layouts with shared positions cannot be served.

def _mrope_model() -> str:
    path = os.environ.get("LLAMA_SERVER_TEST_MROPE_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MROPE_MODEL to an M-RoPE gguf")  # ty: ignore[too-many-positional-arguments]
    return path


def test_exact_concurrency_refuses_an_mrope_model_with_a_projector():
    # Under M-RoPE all tokens of one image share a temporal position.
    path = _mrope_model()
    with tempfile.TemporaryDirectory() as tmp:
        mmproj = os.path.join(tmp, "mmproj.gguf")
        with open(mmproj, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))  # header only: the refusal comes first
        proc = subprocess.run([
            os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server"),
            "--model", path, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(server.server_port),
            "-c", "512", "--parallel", "2", "--kv-unified", "-fa", "on", "-ngl", "99", "--no-warmup", "--no-webui",
        ], env={**os.environ, "LLAMA_EXACT_CONCURRENCY": "1"}, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "does not support M-RoPE together with a projector" in out, out


def test_exact_concurrency_refuses_a_batch_that_cannot_hold_a_whole_ubatch():
    # -b 512 -ub 512 -np 2 would leave a 511-token prefill beside one decoder, so startup refuses the geometry.
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
    # The refusal is about images: a text prompt gives every token its own position.
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    # The mode needs a batch holding a whole ubatch beside a decode step of every slot.
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


def test_exact_concurrency_prefills_a_prompt_in_the_ubatches_it_would_get_alone():
    # Generated tokens enter the batch first, so a prompt got ubatches 512,512,512,509 beside three decoders and 512,512,512,512 alone.
    # -b 2048 -ub 512 -np 4: the batch holds a whole ubatch plus one decode per slot (516 tokens), as common_exact_batch_geometry() requires.
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
        # Decode steps are one token per slot, so the wide ubatches are the prompt's.
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
        for _ in range(600):
            slots = server.make_request("GET", "/slots").body
            if sum(1 for s in slots if s["is_processing"] and s["n_prompt_tokens"] > 0) >= 3:
                break
            time.sleep(0.1)
        else:
            pytest.fail("the decoders never started")  # ty: ignore[invalid-argument-type]
        time.sleep(1.0)

        beside = prefill(50000)
    finally:
        server.stop()
        for t in threads:
            t.join(30)

    assert beside == alone, f"alone {alone}, beside three decoders {beside}"


def test_slots_reports_a_transferring_slot_apart_from_a_parked_one():
    # Slots copying out or restoring still hold cells; only a fully parked slot holds nothing.
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "1"
    _start_async(n_ctx=256)

    done = []
    t = threading.Thread(target=lambda: done.extend(_complete_all(900)))
    t.start()
    seen_parked = seen_transferring = False
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not (seen_parked and seen_transferring):
            res = server.make_request("GET", "/slots")
            assert res.status_code == 200
            for slot in res.body:
                assert not (slot["is_preempted"] and slot["is_transferring"]), slot
                if slot["is_transferring"]:
                    seen_transferring = True
                    assert slot["n_prompt_tokens"] > 0, "a slot with a copy in flight still holds its cells"
                seen_parked = seen_parked or slot["is_preempted"]
    finally:
        t.join(180)

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
    # A shift defers the K transform to the next decode, so a park in between used to save new positions with old K.
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    _start(n_ctx=256, n_batch=32, n_ubatch=32, enable_ctx_shift=True, cache_ram=0)

    n_predict = 320
    reference = _shift_completion(n_predict)
    assert reference.status_code == 200, reference.body
    assert reference.body["timings"]["predicted_n"] == n_predict
    n_prompt = reference.body["timings"]["prompt_n"]
    server.stop()

    # The pool holds n_ctx cells, so the first shift is that many tokens in.
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
    # Validated before posting: parked behind a running sibling, it used to fail inside an already opened 200 stream.
    _start(n_ctx=256, n_slots=2, n_batch=256)

    res = server.make_request("POST", "/completion", data={
        "prompt": [[1] * 240, [1] * 240, [9999999]], "n_predict": 4, "temperature": 0.0, "seed": 42,
    })
    assert res.status_code == 400, res.body
    assert "invalid tokens" in str(res.body)

    text = _log()
    _assert_nothing_parked(text)


def test_a_recompute_park_bounds_its_draft_by_the_tokens_it_comes_back_with():
    # A recompute park empties the prompt, and bounding the draft by it failed a 2000-token request in a 2048-cell pool as impossible.
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


def test_props_says_whether_exact_concurrency_is_running():
    # A build ignoring the variable still starts, so clients read the answer here.
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
    # A re-prefilled state differs in the last bits on CUDA, so the mode reports it the first time.
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
    # A recompute resume is not bit-identical to the saved state; the request, /slots and /metrics report how often it happened.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    # This sequence exceeds the 1 MiB budget, so every park drops its cells.
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
    # Counting the re-prefill as prompt would move the prompt/generation boundary, and the generation time would cover only tokens after the last re-prefill.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=2048, n_batch=2048)

    n_prompt = 1950
    n_predict = 24
    res = _complete(n_predict, _prompt_of(n_prompt, _PROMPT_C))

    assert res.status_code == 200, res.body
    assert res.body["preempt"]["recomputes"] >= 1, res.body["preempt"]

    timings = res.body["timings"]
    assert timings["prompt_n"] == n_prompt, timings
    assert timings["predicted_n"] == n_predict, timings
    # Every re-prefill happens during generation, so the generation time is the longer.
    assert timings["predicted_ms"] > timings["prompt_ms"], timings


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
        deadline = time.time() + 180
        while time.time() < deadline and seen == 0 and t.is_alive():
            res = server.make_request("GET", "/slots")
            assert res.status_code == 200
            for slot in res.body:
                assert slot["n_recompute"] <= slot["n_preempt"]
                seen = max(seen, slot["n_recompute"])
            time.sleep(0.02)
    finally:
        t.join(180)

    assert seen > 0, "no slot ever reported a recompute park"
    assert done and done[0].status_code == 200, done
    assert done[0].body["preempt"]["recomputes"] >= seen


def test_a_swap_park_is_not_reported_as_a_recompute():
    # With room for its bytes the park keeps the saved sequence, and the client is told so.
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
    # Clients keep data["response"] from a streamed /v1/responses, so the record must be there, as in the non-streamed body.
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
    # Media chunks could not be recompute-parked, so with no budget the pool overflowed and ended both chats. Now the chunk is re-encoded from the task with its cells reserved whole.
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
    # A recompute park mid-prefill restarts the prefill, so the prompt count before the park must be dropped too.
    # The park's host allocation is forced to fail, so it drops the cells instead.
    os.environ["LLAMA_SERVER_PREEMPT_FAIL_SAVE"] = "1"
    _start(n_ctx=2048, n_batch=256)

    resident = []
    t = threading.Thread(target=lambda: resident.append(_complete(1900, _PROMPT_A)), daemon=True)
    t.start()

    # The pool must be nearly full before the second prompt starts, so its prefill runs out of cells.
    deadline = time.time() + 90
    while t.is_alive() and time.time() < deadline:
        slots = server.make_request("GET", "/slots").body
        if any(slot.get("n_prompt_tokens", 0) >= 1600 for slot in slots):
            break
        time.sleep(0.005)
    else:
        pytest.fail("the resident never grew into the pool")  # ty: ignore[invalid-argument-type]

    n_prompt = 500
    n_predict = 8
    comments, final = _stream_completion(n_predict, _prompt_of(n_prompt, _PROMPT_B))
    t.join(120)

    assert resident and resident[0].status_code == 200, resident
    assert "error" not in final, final
    assert comments and comments[0] == ": preempted", comments

    # A park that dropped fewer cells than the resume puts back was taken mid-prefill, the case under test.
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


def test_a_parked_head_is_not_overtaken_by_a_stream_of_short_requests():
    # A 7300-cell parked head used to wait forever while short requests kept a resident present. Once the head has waited its turn, never-resident prompts wait for it.
    # 1 MiB holds no short request's state, so only the reservation brings the head back before PREEMPT_ROTATE_RECOMPUTE_US.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    # Parks the head while it generates.
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "48"
    # On CUDA a prompt chunk across position 2048 of this model hangs llama_decode (also on base), so run on CPU; the code under test is host-side.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    _serve_with_training_context(8192)
    _start(n_ctx=8192, n_slots=4, n_batch=1024)

    short_prompt = _prompt_of(1700, _PROMPT_B)
    head_prompt = _prompt_of(7200, _PROMPT_A)
    n_short, n_head = 32, 100
    # A starved head waits forever, so any bound catches it; the head prefills 7200 tokens three times, 46 s on a CPU runner and 92 s on Windows.
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
        deadline = time.time() + 60
        while time.time() < deadline and len(shorts) < 6:
            time.sleep(0.01)
        t_head = time.time()
        with ThreadPoolExecutor(1) as pool:
            fut = pool.submit(_raw_completion, n_head, head_prompt, 3, t_stream_max + 180)
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
    # The head returns once the residents ahead of it finish, about one quantum after each park.
    waits = [float(w) for w in re.findall(r"id +3 \| task \d+ \| resumed after ([\d.]+) s", text)]
    assert waits, "the head was never parked and resumed"
    assert max(waits) < 10.0, f"the head waited {waits} s behind requests that came after it"


@pytest.mark.slow
def test_a_long_resident_that_never_shifts_is_rotated_out_for_an_aged_head():
    # Only context-shifting residents used to be rotated, so a long generation that fit held the pool and the parked slot waited for all of it.
    _serve_with_training_context(16384)
    _start(n_ctx=16384, n_slots=2, n_batch=2048)

    done = {}

    def run(name, n_predict, prompt, id_slot):
        res = _raw_completion(n_predict, prompt, id_slot, timeout=900)
        done[name] = (time.time(), res)

    # A leads B by 564 cells, so B is parked once the pool fills, and A then runs far longer than the 2 s a parked head waits.
    a = threading.Thread(target=run, args=("a", 16000, _prompt_of(64, _PROMPT_A), 0))
    a.start()
    deadline = time.time() + 300
    while time.time() < deadline and _n_decoded(_slot_view(0)) < 2000:
        time.sleep(0.01)
    b = threading.Thread(target=run, args=("b", 8000, _prompt_of(1500, _PROMPT_B), 1))
    b.start()
    a.join(900)
    b.join(900)

    for name in ("a", "b"):
        assert done[name][1].status_code == 200, done[name][1].body
    assert done["a"][1].body["timings"]["predicted_n"] == 16000
    assert done["b"][1].body["timings"]["predicted_n"] == 8000

    text = _log()
    assert "Context size has been exceeded" not in text
    assert "slot context shift" not in text
    assert "rotated out after 0 context shifts" in text, "the long resident was never rotated out"
    assert done["b"][0] < done["a"][0], "the parked slot only came back once the long resident had finished"


def test_one_victim_that_covers_the_shortfall_is_parked_instead_of_several_small_ones():
    # The second chunk is about 600 cells short: smallest-first parked both 400-token residents, where the 1400-token one alone covers it.
    _serve_with_training_context(8192)
    _start(n_ctx=4864, n_slots=4, n_batch=2048)

    small = [_prompt_of(400, _PROMPT_A), _prompt_of(400, _PROMPT_B)]
    medium = _prompt_of(1400, _PROMPT_C)
    leader = _prompt_of(3267, _PROMPT_A)

    with ThreadPoolExecutor(4) as pool:
        residents = [pool.submit(_raw_completion, 300, p, i) for i, p in enumerate(small + [medium])]
        # Every resident is generating before the leader arrives, so the leader's second chunk is the shortfall.
        deadline = time.time() + 60
        while time.time() < deadline:
            slots = server.make_request("GET", "/slots").body
            if all(_n_decoded(s) > 0 for s in slots if s["id"] < 3):
                break
            time.sleep(0.002)
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
    # Clients read this instead of guessing from the version.
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
    assert preempt["policy"] == "cost"


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
    # Pinned buffers sized exactly were re-pinned on every park of a grown sequence, costing about as much as the copy.
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


def test_a_multi_prompt_oai_completion_counts_the_parks_of_every_prompt():
    # An array prompt is one task per prompt, each parked separately; the OAI response merges them, so its counters are the sum.
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
    # 200 of 256 cells are ever held; charging the started slot's old 100 cells beside its new 100 purged slot 1, so its rerun re-prefilled.
    _start(n_ctx=256, n_batch=128, n_ubatch=64, cache_ram=0)
    results = [_cached(1, [tok] * 100, slot) for slot, tok in ((0, 11), (1, 12), (0, 13), (1, 12))]
    for res in results:
        assert res.status_code == 200, res.body
    assert results[3].body["timings"]["prompt_n"] == 1, results[3].body["timings"]
    text = _log()
    assert "purging slot" not in text
    _assert_nothing_parked(text)


def test_an_idle_slot_inside_the_margin_keeps_its_cache():
    # 120 idle and 132 generating cells fit 256 with room for the step; purging slot 0 only to keep the planning margin made its rerun re-prefill.
    _start(n_ctx=256, n_batch=128, n_ubatch=64, cache_ram=0)
    first = [11] * 120
    assert _cached(1, first, 0).status_code == 200
    assert _cached(12, [12] * 120, 1).status_code == 200
    res = _cached(1, first, 0)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["prompt_n"] == 1, res.body["timings"]
    assert "purging slot" not in _log()


def test_a_started_slot_that_fits_once_trimmed_parks_nobody():
    # Slot 0 keeps 300 of its 600 old cells: 300 + 336 + 60 + 310 fit 1024, but the untrimmed count parked the generating slot and purged slot 2.
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
    # The second park falls back to recompute; its resume was admitted for the whole sequence, but the restores behind it took the room and parked it again part way through.
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


def _oai_n(prompt: list, n: int, max_tokens: int):
    return server.make_request("POST", "/v1/completions", data={
        "prompt": prompt, "n": n, "max_tokens": max_tokens, "ignore_eos": True,
        "temperature": 0.0, "seed": 42, "cache_prompt": False,
    })


def test_a_single_request_beside_an_n_2_request_is_parked_so_both_finish():
    # 900 + 700 cells overflow 1024 while each fits alone; the n=1 slot was the exempt leader and the n=2 slots cannot be parked, so every slot failed.
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
    # Neither n=2 request can be parked; ending every slot also ended the one that fits alone.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=1024, n_slots=4, n_batch=128, n_ubatch=128)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(_oai_n, [100 + k % 50 for k in range(200)], 2, 250)
        deadline = time.time() + 30
        while time.time() < deadline and sum(s["is_processing"] for s in server.make_request("GET", "/slots").body) < 2:
            time.sleep(0.005)
        second = pool.submit(_oai_n, [200 + k % 50 for k in range(200)], 2, 250)
        results = [first.result(), second.result()]
    assert sorted(r.status_code for r in results) == [200, 500], [r.body for r in results]
    ok = next(r for r in results if r.status_code == 200)
    assert [c["finish_reason"] for c in ok.body["choices"]] == ["length", "length"]
    assert "ending it alone" in _log()


def test_a_started_request_reusing_a_long_prefix_does_not_park_an_older_generating_one():
    # Slot 1 keeps 600 cells of an earlier conversation; its follow-up started beside a generating request, was the largest, became the exempt leader, and the older request was parked for it.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=1024, n_slots=2, n_batch=128, n_ubatch=128, cache_ram=0)
    earlier = [200 + k % 50 for k in range(600)]
    assert _cached(1, earlier, 1).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 600, [100 + k % 50 for k in range(40)], 0)
        deadline = time.time() + 60
        while time.time() < deadline and _n_decoded(_slot_view(0)) < 280:
            time.sleep(0.002)
        newer = _cached(4, earlier + [300 + k % 50 for k in range(300)], 1)
        older = older.result()
    for res, n_predict in ((older, 600), (newer, 4)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    assert older.body["preempt"]["parks"] == 0, "the older generating request was parked for a newer one"
    assert newer.body["preempt"]["parks"] > 0


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_the_policy_decides_whether_the_newest_or_the_cheapest_slot_is_parked(policy):
    # The older request is smaller: cost parks it and exempts the larger one, fcfs parks the newer one. Both first purge the idle slot.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _start(n_ctx=1024, n_slots=3, n_batch=128, n_ubatch=128, cache_ram=0)
    assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    assert _cached(1, [300 + k % 50 for k in range(150)], 2).status_code == 200
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 600, [100 + k % 50 for k in range(40)], 0)
        deadline = time.time() + 60
        while time.time() < deadline and _n_decoded(_slot_view(0)) < 20:
            time.sleep(0.005)
        newer = _cached(500, [200 + k % 50 for k in range(300)], 1)
        older = older.result()
    for res, n_predict in ((older, 600), (newer, 500)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    parked, exempt = (newer, older) if policy == "fcfs" else (older, newer)
    assert exempt.body["preempt"]["parks"] == 0
    assert parked.body["preempt"]["parks"] > 0
    text = _log()
    assert 0 <= text.find("purging slot 2") < text.find("preempted:"), "a running slot was parked while an idle one held cells"


def test_fcfs_resumes_parked_requests_in_arrival_order():
    # The newest is parked first, so park time and arrival disagree: the older of the two parked requests comes back first.
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


def test_the_old_policy_knob_selects_fcfs():
    os.environ["LLAMA_SERVER_PREEMPT_POLICY"] = "fcfs"
    try:
        _start(n_ctx=256)
        assert server.make_request("GET", "/props").body["preempt"]["policy"] == "fcfs"
    finally:
        os.environ.pop("LLAMA_SERVER_PREEMPT_POLICY", None)


@pytest.mark.parametrize("policy", ["fcfs", "cost"])
def test_a_prompt_dropped_in_its_prefill_waits_until_it_fits_whole(policy):
    # The newer prompt is parked by dropping its cells part way through its prefill; resumed as soon as one chunk fitted, it was prefilled and dropped again until the older request ended.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_ARG_PREEMPT_POLICY"] = policy
    _serve_with_training_context(8192)
    _start(n_ctx=4096, n_slots=2, n_batch=256, n_ubatch=256, cache_ram=0)
    assert server.make_request("GET", "/props").body["preempt"]["policy"] == policy
    with ThreadPoolExecutor(1) as pool:
        older = pool.submit(_cached, 1500, [100 + k % 50 for k in range(1500)], 0)
        deadline = time.time() + 120
        while time.time() < deadline and _n_decoded(_slot_view(0)) < 100:
            time.sleep(0.005)
        newer = _cached(4, [200 + k % 50 for k in range(2500)], 1)
        older = older.result()
    for res, n_predict in ((older, 1500), (newer, 4)):
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
    parked = older if older.body["preempt"]["parks"] > 0 else newer
    assert parked.body["preempt"]["parks"] == 1, (older.body["preempt"], newer.body["preempt"])


def test_oversubscribed_bounded_chats_are_not_time_sliced():
    # Four chats of 200 + 800 tokens need twice the pool; residents that end by themselves used to be rotated out every 2 s, parking and restoring whole sequences over and over.
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "64"
    _start(n_ctx=2048, n_slots=4, n_batch=256, n_ubatch=256, cache_ram=0)
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: _cached(800, [50 + 100 * i + k % 50 for k in range(200)], i), range(4)))
    _assert_completed(results, 800)
    text = _log()
    assert "Context size has been exceeded" not in text
    assert "rotated out after" not in text
    parks = sum(r.body["preempt"]["parks"] for r in results)
    assert 1 <= parks <= 4, parks
