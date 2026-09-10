import json
import os
import re
import tempfile
import threading
import pytest
import requests
from utils import *

# [TAG_PREEMPT] a streaming client is told when its slot is parked and restored, as SSE comments every existing client ignores; a keepalive every 2 s keeps proxies from giving up

server = ServerPreset.tinyllama2()

_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"
_PROMPT_C = "In a small village by the sea there lived a fisherman who"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    # the server parks only when asked: --preempt-ram defaults to 0, and this suite is about parking
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "8192"
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    for name in ("LLAMA_SERVER_PREEMPT_EVERY", "LLAMA_ARG_PREEMPT_RAM"):
        os.environ.pop(name, None)


def _start(**kwargs):
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()


def _completion_payload(n_predict: int, prompt: str = "Hi how are you", **extra) -> dict:
    return {"n_predict": n_predict, "prompt": prompt, "ignore_eos": True,
            "temperature": 0.0, "seed": 42, "stream": True, **extra}


def _chat_payload(n_predict: int) -> dict:
    return {"max_tokens": n_predict, "messages": [{"role": "user", "content": "Hi how are you"}],
            "temperature": 0.0, "seed": 42, "stream": True}


def _post(path: str, data: dict):
    return requests.post(f"http://{server.server_host}:{server.server_port}{path}", json=data, stream=True)


def _stream_raw(path: str, data: dict) -> tuple[list[str], list[str]]:
    """The SSE lines of one streaming request: (comment lines, data lines)."""
    res = _post(path, data)
    assert res.status_code == 200
    comments, datas = [], []
    for raw in res.iter_lines():
        line = raw.decode("utf-8")
        if line.startswith(":"):
            comments.append(line)
        elif line.startswith("data: "):
            datas.append(line[6:])
    return comments, datas


def _stream_all(n_predict: int, prompts, **extra):
    return parallel_function_calls([
        (_stream_raw, ("/completion", _completion_payload(n_predict, prompt, **extra))) for prompt in prompts
    ])


def _behind_a_resident(payload: dict) -> tuple[int, str, list[str]]:
    """Run this request behind a resident holding the pool: its status, its body, and its SSE lines."""
    started = threading.Event()

    def _resident():
        res = _post("/completion", _completion_payload(390, " ".join([_PROMPT_A] * 6)))
        assert res.status_code == 200
        for raw in res.iter_lines():
            if raw.decode("utf-8").startswith("data: "):
                started.set()

    t = threading.Thread(target=_resident)
    t.start()
    try:
        assert started.wait(60)
        res = _post("/completion", payload)
        if res.status_code != 200:
            return res.status_code, res.text, []
        lines, alive = [], None
        for raw in res.iter_lines():
            line = raw.decode("utf-8")
            if line:
                alive = t.is_alive() if alive is None else alive
                lines.append(line)
        assert alive, "the resident had finished before this request was told anything"
        return 200, "", lines
    finally:
        t.join(120)


def _content(datas: list[str]) -> str:
    out = ""
    for d in datas:
        if d == "[DONE]":
            break
        j = json.loads(d)
        out += j.get("content") or ""
        for ch in j.get("choices", []) or []:
            out += (ch.get("delta") or {}).get("content") or ""
    return out


def _final(datas: list[str]) -> dict:
    """The last response object of a finished stream, past the [DONE] marker."""
    return json.loads([d for d in datas if d != "[DONE]"][-1])


def _notices(comments: list[str]) -> list[str]:
    return [c for c in comments if c in (": preempted", ": resumed")]


@pytest.mark.parametrize("path,payload", [
    ("/completion", _completion_payload(64)),
    ("/v1/chat/completions", _chat_payload(64)),
])
def test_every_park_in_a_stream_is_announced_paired_with_a_resume_and_changes_nothing(path, payload):
    _start(n_ctx=512)
    ref_comments, ref_datas = _stream_raw(path, payload)
    assert _notices(ref_comments) == []
    assert _content(ref_datas)
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    comments, datas = _stream_raw(path, payload)
    seq = _notices(comments)
    assert len(seq) >= 12, comments
    assert seq == [": preempted", ": resumed"] * (len(seq) // 2), seq
    assert _content(datas) == _content(ref_datas)


def _prefill_payload(path: str, prompt: str, n_predict: int) -> dict:
    """The same request on each streaming surface."""
    if path == "/completion":
        return {"prompt": prompt, "n_predict": n_predict, "ignore_eos": True,
                "temperature": 0.0, "seed": 42, "stream": True}
    if path == "/v1/responses":
        return {"model": "test", "input": prompt, "max_output_tokens": n_predict,
                "temperature": 0.0, "stream": True}
    return {"model": "test", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict, "temperature": 0.0, "stream": True}


@pytest.mark.parametrize("path", ["/completion", "/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_a_park_during_prompt_processing_opens_the_stream_with_the_notice(path):
    # a park before the first token is the case a client cannot tell from a stall, so the notice goes out with the response headers rather than waiting for a chunk that is not coming
    import time

    server.server_slots = True
    _start(n_ctx=2048, n_batch=256)

    def _resident():
        res = _post("/completion", _completion_payload(1900, _PROMPT_A))
        for _ in res.iter_lines():
            pass

    t = threading.Thread(target=_resident, daemon=True)
    t.start()

    # the pool has to be nearly full before the second prompt starts, so that its prefill is what runs out of cells
    # the resident grows by decoding: 1400 tokens took over 12 s on a loaded CI runner
    deadline = time.time() + 90
    while t.is_alive() and time.time() < deadline:
        slots = requests.get(f"http://{server.server_host}:{server.server_port}/slots").json()
        if any(slot.get("n_prompt_tokens", 0) >= 1400 for slot in slots):
            break
        time.sleep(0.02)
    else:
        pytest.fail("the resident never grew into the pool")

    res = _post(path, _prefill_payload(path, " ".join([_PROMPT_B] * 31), 8))
    assert res.status_code == 200

    t0 = time.time()
    seen = []
    for raw in res.iter_lines():
        line = raw.decode("utf-8")
        if line:
            seen.append((time.time() - t0, line))
    t.join(120)

    text = open(server.log_path).read()
    assert "preempted:" in text, "nothing was parked while the prompt was being processed"

    comments = [(at, line) for at, line in seen if line.startswith(":")]
    datas    = [(at, line) for at, line in seen if line.startswith("data:")]

    assert comments and comments[0][1] == ": preempted", [line for _, line in seen[:4]]
    assert datas, "the request never produced a chunk"

    # sent when the slot was parked, not batched with the chunk that came later
    assert comments[0][0] + 0.05 < datas[0][0], [(round(at, 3), line[:24]) for at, line in seen[:4]]
    assert any(line == ": resumed" for _, line in comments), [line for _, line in comments[:4]]


def test_a_stream_parked_before_its_first_token_starts_with_the_notice():
    # n_batch: the whole prompt in one batch, so the planner sees its size at once
    _start(n_ctx=512, n_batch=512)

    status, _, lines = _behind_a_resident(_completion_payload(32, " ".join([_PROMPT_B] * 14)))
    assert status == 200
    events = [l for l in lines if l in (": preempted", ": resumed") or l.startswith("data: ")]
    assert events[:2] == [": preempted", ": resumed"], events[:3]
    assert events[2].startswith("data: "), events[:3]
    datas = [l[6:] for l in lines if l.startswith("data: ")]
    assert _content(datas)
    assert _final(datas)["tokens_predicted"] == 32


def test_an_oversized_prompt_is_errored_instead_of_parked():
    # a slot just given a task has not passed the prompt checks yet, and a notice opens the stream, so parking it would turn a plain error response into 200 plus an in-stream one
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512, n_batch=512)

    status, body, _ = _behind_a_resident(_completion_payload(16, " ".join([_PROMPT_B] * 80)))
    assert status != 200, body
    assert not body.lstrip().startswith(":"), body
    assert "error" in json.loads(body), body


def test_a_rotation_tells_both_streams_and_a_head_parked_past_the_budget_is_kept_alive():
    # --preempt-ram 2 MiB holds one parked state but not a resident's and the head's at once, so that rotation is refused and the head waits parked for longer than the 2 s keepalive
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    _start(n_slots=3, n_ctx=2048, enable_ctx_shift=True)

    n_predict = 12000
    # the default parked keepalive is 2 s, which is also when a resident is rotated out for the head:
    # a park that ends with that rotation could beat its own keepalive. Ask for a 1 s ping instead, so
    # any park that outlasts one rotation window is still required to say so
    results = _stream_all(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C), sse_ping_interval=1)
    n_parked = n_keepalive = 0
    for comments, datas in results:
        assert _final(datas)["tokens_predicted"] == n_predict
        seq = _notices(comments)
        assert seq == [": preempted", ": resumed"] * (len(seq) // 2), seq
        n_parked += len(seq) // 2
        n_keepalive += comments.count(": preempt-keepalive")
    assert n_parked >= 2, [r[0] for r in results]
    assert n_keepalive >= 1, "a parked stream was left silent past its keepalive interval"

    text = open(server.log_path).read()
    assert "rotated out after" in text
    assert "no rotation: --preempt-ram 2 MiB" in text
    assert "resumed after" in text
    assert "Context size has been exceeded" not in text


def test_every_notice_of_a_multi_prompt_stream_names_the_prompt_it_is_about():
    # one request, two prompts: a client reading the shared stream can only tell the notices apart by their index, so index 0 has to be spelled out like any other
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    _start(n_ctx=512)

    comments, datas = _stream_raw("/completion", _completion_payload(32) | {"prompt": [_PROMPT_A, _PROMPT_B]})
    notices = [c for c in comments if c.startswith(": preempted") or c.startswith(": resumed")]
    assert notices, comments
    assert all(re.fullmatch(r": (preempted|resumed) [01]", c) for c in notices), notices
    for index in (0, 1):
        assert f": preempted {index}" in notices, notices
        assert f": resumed {index}" in notices, notices


def test_an_oversized_sibling_prompt_is_errored_before_a_valid_one_is_parked():
    # a request can carry several prompts; a valid one can be parked and its notice opens the stream, so the sibling that does not fit has to be found before any of them is queued
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "8192"
    _start(n_ctx=256, n_slots=3, n_batch=512)

    status, body, _ = _behind_a_resident(_completion_payload(8) | {"prompt": [[1] * 120, [1] * 300]})
    assert status == 400, (status, body)
    assert not body.lstrip().startswith(":"), body
    assert "error" in json.loads(body), body

