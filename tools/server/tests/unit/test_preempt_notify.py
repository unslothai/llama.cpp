import os
import tempfile
import threading
import time
import pytest
import requests
from utils import *

# [TAG_PREEMPT] a streaming client is told when its slot is parked and restored, as SSE comments every existing client ignores; a keepalive every 2 s keeps proxies from giving up

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.temperature = 0.0
    server.seed = 42
    # A build without libcurl cannot fetch the model itself; point it at a local copy.
    local = os.environ.get("LLAMA_SERVER_TEST_MODEL")
    if local:
        server.model_hf_repo = None
        server.model_hf_file = None
        server.model_file = local
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    os.environ.pop("LLAMA_SERVER_PREEMPT_EVERY", None)


def _stream_raw(path: str, data: dict) -> tuple[list[str], list[str]]:
    """The SSE lines of one streaming request: (comment lines, data lines)."""
    url = f"http://{server.server_host}:{server.server_port}{path}"
    res = requests.post(url, json=data, stream=True)
    assert res.status_code == 200
    comments, datas = [], []
    for raw in res.iter_lines():
        line = raw.decode("utf-8")
        if line.startswith(":"):
            comments.append(line)
        elif line.startswith("data: "):
            datas.append(line[6:])
    return comments, datas


def _content(datas: list[str]) -> str:
    out = ""
    for d in datas:
        if d == "[DONE]":
            break
        j = json.loads(d)
        if "content" in j:
            out += j["content"]
        for ch in j.get("choices", []) or []:
            delta = ch.get("delta") or {}
            out += delta.get("content") or ""
    return out


def _completion_payload(n_predict: int) -> dict:
    return {
        "n_predict": n_predict,
        "prompt": "Hi how are you",
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
    }


def _chat_payload(n_predict: int) -> dict:
    return {
        "max_tokens": n_predict,
        "messages": [{"role": "user", "content": "Hi how are you"}],
        "temperature": 0.0,
        "seed": 42,
        "stream": True,
    }


_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"


def _start(**kwargs):
    """Start the server with these settings."""
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()


def _final(datas: list[str]) -> dict:
    """The last response object of a finished stream, past the [DONE] marker."""
    return json.loads([d for d in datas if d != "[DONE]"][-1])


def _stream_both(n_predict: int):
    """One streaming completion per prompt, both at once."""
    return parallel_function_calls([
        (_stream_raw, ("/completion", _completion_payload(n_predict) | {"prompt": prompt}))
        for prompt in (_PROMPT_A, _PROMPT_B)
    ])


def test_a_stream_announces_its_parks_and_the_body_is_unchanged():
    _start(n_ctx=512)
    ref_comments, ref_datas = _stream_raw("/completion", _completion_payload(64))
    assert not any(c.startswith(": preempted") or c.startswith(": resumed") for c in ref_comments)
    assert _content(ref_datas)
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    comments, datas = _stream_raw("/completion", _completion_payload(64))

    parked = [c for c in comments if c == ": preempted"]
    resumed = [c for c in comments if c == ": resumed"]
    assert len(parked) >= 6, comments
    assert len(resumed) == len(parked), comments
    seq = [c for c in comments if c in (": preempted", ": resumed")]
    assert seq == [": preempted", ": resumed"] * len(parked), seq
    def _pieces(ds):
        return [json.loads(d).get("content") for d in ds if d != "[DONE]"]

    assert _pieces(datas) == _pieces(ref_datas)
    assert _content(datas) == _content(ref_datas)
    final, ref_final = json.loads(datas[-1]), json.loads(ref_datas[-1])
    assert final["tokens_predicted"] == ref_final["tokens_predicted"] == 64


def test_the_oai_chat_stream_carries_the_same_comments():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512)
    comments, datas = _stream_raw("/v1/chat/completions", _chat_payload(48))
    assert ": preempted" in comments and ": resumed" in comments
    assert datas[-1] == "[DONE]"
    assert _content(datas)


def test_non_streaming_requests_see_nothing():
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512)
    res = server.make_request("POST", "/completion", data={
        "n_predict": 32,
        "prompt": "Hi how are you",
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
    })
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 32
    assert "preempted" not in res.body


def test_two_overflowing_streams_both_finish_and_the_parked_one_says_so():
    _start(n_ctx=256)

    n_predict = 160
    results = _stream_both(n_predict)
    announced = 0
    for comments, datas in results:
        final = _final(datas)
        assert final["timings"]["predicted_n"] == n_predict
        assert final["truncated"] is False
        if ": preempted" in comments:
            announced += 1
            assert ": resumed" in comments
    assert announced >= 1, [r[0] for r in results]


def test_a_stream_parked_before_its_first_token_starts_with_the_notice():
    # n_batch: the whole prompt in one batch, so the planner sees its size at once
    _start(n_ctx=512, n_batch=512)
    url = f"http://{server.server_host}:{server.server_port}/completion"
    first = _completion_payload(390) | {"prompt": " ".join([_PROMPT_A] * 6)}
    second = _completion_payload(32) | {"prompt": " ".join([_PROMPT_B] * 14)}

    timeline = []
    lock = threading.Lock()

    def _run(name, payload, started=None):
        res = requests.post(url, json=payload, stream=True)
        assert res.status_code == 200
        for raw in res.iter_lines():
            line = raw.decode("utf-8")
            if not line:
                continue
            with lock:
                timeline.append((time.monotonic(), name, line))
            if started is not None and line.startswith("data: "):
                started.set()

    started = threading.Event()
    t = threading.Thread(target=_run, args=("first", first, started))
    t.start()
    assert started.wait(30)
    _run("second", second)
    t.join(60)

    second_lines = [(ts, line) for ts, name, line in timeline if name == "second"]
    first_end = max(ts for ts, name, _ in timeline if name == "first")
    assert second_lines[0][1] == ": preempted", second_lines[:3]
    assert second_lines[0][0] < first_end
    events = [line for _, line in second_lines if line in (": preempted", ": resumed") or line.startswith("data: ")]
    assert events[0] == ": preempted" and events[1] == ": resumed" and events[2].startswith("data: "), events[:3]
    datas = [line[6:] for _, line in second_lines if line.startswith("data: ")]
    assert _content(datas)
    assert _final(datas)["tokens_predicted"] == 32


def test_a_resident_rotated_out_for_a_parked_head_is_told_so():
    _start(n_ctx=256, enable_ctx_shift=True)
    n_predict = 12000
    results = _stream_both(n_predict)
    n_parked = 0
    for comments, datas in results:
        final = _final(datas)
        assert final["tokens_predicted"] == n_predict
        seq = [c for c in comments if c in (": preempted", ": resumed")]
        assert seq == [": preempted", ": resumed"] * (len(seq) // 2), seq
        n_parked += len(seq) // 2
    assert n_parked >= 2, [r[0] for r in results]


def test_an_oversized_prompt_is_errored_instead_of_parked():
    # A slot just given a task has not passed the prompt checks yet, and a notice opens the stream, so parking it would turn a plain error response into 200 plus an in-stream one.
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512, n_batch=512)
    url = f"http://{server.server_host}:{server.server_port}/completion"
    resident = _completion_payload(390) | {"prompt": " ".join([_PROMPT_A] * 6)}
    oversized = _completion_payload(16) | {"prompt": " ".join([_PROMPT_B] * 80)}

    started = threading.Event()

    def _run_resident():
        res = requests.post(url, json=resident, stream=True)
        assert res.status_code == 200
        for raw in res.iter_lines():
            if raw.decode("utf-8").startswith("data: "):
                started.set()

    t = threading.Thread(target=_run_resident)
    t.start()
    try:
        assert started.wait(60)
        res = requests.post(url, json=oversized, stream=True)
        body = res.text
        assert res.status_code != 200, body
        assert not body.lstrip().startswith(":"), body
        assert "error" in json.loads(body), body
    finally:
        t.join(120)
