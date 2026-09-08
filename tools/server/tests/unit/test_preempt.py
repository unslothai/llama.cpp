import base64
import json
import os
import re
import struct
import subprocess
import threading
import time
import tempfile
import pytest
import requests
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
                 "LLAMA_SERVER_PREEMPT_PLANNER", "LLAMA_ARG_PREEMPT_RAM", "LLAMA_ARG_PREEMPT_ASYNC",
                 "LLAMA_MEDIA_MARKER", "LLAMA_EXACT_CONCURRENCY"):
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


def _complete(n_predict: int, prompt="Hi how are you", id_slot: int = -1, delay: float = 0.0, after_slot_busy=None):
    time.sleep(delay)
    if after_slot_busy is not None:
        # sent once that slot is processing, so the request queues behind it whatever the host's speed
        for _ in range(200):
            slots = server.make_request("GET", "/slots").body
            if any(s["id"] == after_slot_busy and s["is_processing"] for s in slots):
                break
            time.sleep(0.02)
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
        assert slot["is_transferring"] is False
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
        (_complete, (8, _PROMPT_C, 0, 0.0, 0)),
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

    # not "Once upon a time": what this Q2_K model decodes from it on CUDA carries bytes the content parser refuses, master included, which is not what this test measures
    results = _complete_all(64, ["The quick brown fox", "Hello world"])
    _assert_completed(results, 64)

    text = _log()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    assert "preempted" not in text
    assert "Context size has been exceeded" not in text


def _stream_completion(n_predict: int, prompt: str) -> tuple[list[str], dict]:
    """One streaming completion: its SSE comment lines and the last response object."""
    url = f"http://{server.server_host}:{server.server_port}/completion"
    res = requests.post(url, json={
        "prompt": prompt, "n_predict": n_predict, "ignore_eos": True,
        "temperature": 0.0, "seed": 42, "stream": True,
    }, stream=True, timeout=600)
    assert res.status_code == 200
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
    # 1 MiB holds neither sequence, so no victim fits the budget: the park drops the cells and the resume re-prefills the tokens, instead of the pool overflowing and ending both; the last resort falls back the same way
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    if planner == "off":
        os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    # one batch for the re-prefill: a prompt of this length in 32-token batches hangs the CUDA build with graphs on, master included, so that is not what this test measures
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
    # a chunk is decoded whole inside one iteration, through decodes the kv-full retry does not cover: unless the planner reserves every cell it takes, the second of two image requests that each fit alone fails part way through its chunk
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


def test_a_hybrid_model_parks_synchronously():
    # the recurrent half of a hybrid gathers the active sequences into contiguous rows on every batch, so a copy running beside the decode could read a row another sequence has been moved into
    path = os.environ.get("LLAMA_SERVER_TEST_HYBRID_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_HYBRID_MODEL to a hybrid attention/recurrent gguf")
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


# [TAG_EXACT_CONCURRENCY] the paged pool places a cell from the sequence and the position alone, so a layout that gives several tokens one position cannot be served

def _mrope_model() -> str:
    path = os.environ.get("LLAMA_SERVER_TEST_MROPE_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MROPE_MODEL to an M-RoPE gguf")
    return path


def test_exact_concurrency_refuses_an_mrope_model_with_a_projector():
    # every token of one image shares a temporal position under M-RoPE, so the pool would give the second one the first one's cell
    path = _mrope_model()
    with tempfile.TemporaryDirectory() as tmp:
        mmproj = os.path.join(tmp, "mmproj.gguf")
        with open(mmproj, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))  # header only: never read, the refusal comes first
        proc = subprocess.run([
            os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server"),
            "--model", path, "--mmproj", mmproj, "--host", "127.0.0.1", "--port", str(server.server_port),
            "-c", "512", "--parallel", "2", "--kv-unified", "-fa", "on", "-ngl", "99", "--no-warmup", "--no-webui",
        ], env={**os.environ, "LLAMA_EXACT_CONCURRENCY": "1"}, capture_output=True, text=True, timeout=900)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "does not support M-RoPE together with a projector" in out, out


def test_exact_concurrency_serves_an_mrope_model_without_a_projector():
    # the refusal is about images, not the rope layout: a text prompt gives every token its own position
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    _start(n_ctx=512, n_slots=2, fa="on", n_gpu_layer=99)

    res = _complete(16, "Once upon a time")
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 16

    text = _log()
    assert "does not support M-RoPE" not in text
    assert "the kv pool allocates 256 cells at a time" in text


def test_slots_reports_a_transferring_slot_apart_from_a_parked_one():
    # a copy out still owns its cells and a restore has already taken them back, so a reader counting residency has to keep counting both; only a fully parked slot holds nothing
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
    # the shift moves the positions and leaves the K transformation for the next decode, so a park in between used to save the new positions with the old K
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    _start(n_ctx=256, n_batch=32, n_ubatch=32, enable_ctx_shift=True, cache_ram=0)

    n_predict = 320
    reference = _shift_completion(n_predict)
    assert reference.status_code == 200, reference.body
    assert reference.body["timings"]["predicted_n"] == n_predict
    n_prompt = reference.body["timings"]["prompt_n"]
    server.stop()

    # park on the step the shift lands on: the pool holds n_ctx cells, so the first shift is that many tokens in
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
    # validated with the others ahead of posting: parked behind a running sibling, it used to fail inside a stream that had already opened 200
    _start(n_ctx=256, n_slots=2, n_batch=256)

    res = server.make_request("POST", "/completion", data={
        "prompt": [[1] * 240, [1] * 240, [9999999]], "n_predict": 4, "temperature": 0.0, "seed": 42,
    })
    assert res.status_code == 400, res.body
    assert "invalid tokens" in str(res.body)

    text = _log()
    assert "preempted" not in text


def test_a_recompute_park_bounds_its_draft_by_the_tokens_it_comes_back_with():
    # a recompute park moves the prompt out of the slot, and the draft was bounded by the empty prompt: 2000 tokens and a whole draft could not fit a 2048-cell pool "even alone", failing a request that fits
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
    # a client that asked for the mode reads the answer here: a build that ignores the variable starts all the same
    _start(n_ctx=256)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["exact_concurrency"] is False


def test_props_reports_exact_concurrency_on():
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    _start(n_ctx=512, n_slots=2, fa="on", n_gpu_layer=99)
    res = server.make_request("GET", "/props")
    assert res.status_code == 200
    assert res.body["exact_concurrency"] is True


def test_a_recompute_park_under_exact_concurrency_says_it_is_not_byte_identical():
    # a state that comes back from host memory is the state that left; one rebuilt by re-prefilling differs in the last bits on CUDA, so the mode says so the first time it happens
    server.model_file = _mrope_model()
    server.model_hf_repo = server.model_hf_file = None
    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "4"
    _start(n_ctx=512, n_slots=2, fa="on", n_gpu_layer=99)

    res = _complete(16, "Once upon a time")
    assert res.status_code == 200, res.body
    text = _log()
    assert "tokens to re-prefill" in text
    assert "not guaranteed byte-identical" in text


def test_two_image_chats_that_outgrow_the_parking_budget_both_finish():
    # a media chunk could not be parked by recompute, so with the host budget spent nothing could be parked at all and the pool overflowing ended both chats. The chunk comes back the way it went in: re-encoded off the task, its cells reserved whole
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
