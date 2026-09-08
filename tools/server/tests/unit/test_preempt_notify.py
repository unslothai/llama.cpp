import os
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


def _completion_payload(n_predict: int, prompt: str = "Hi how are you") -> dict:
    return {"n_predict": n_predict, "prompt": prompt, "ignore_eos": True,
            "temperature": 0.0, "seed": 42, "stream": True}


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


def _stream_all(n_predict: int, prompts):
    return parallel_function_calls([
        (_stream_raw, ("/completion", _completion_payload(n_predict, prompt))) for prompt in prompts
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
    results = _stream_all(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C))
    n_parked = n_keepalive = 0
    for comments, datas in results:
        assert _final(datas)["tokens_predicted"] == n_predict
        seq = _notices(comments)
        assert seq == [": preempted", ": resumed"] * (len(seq) // 2), seq
        n_parked += len(seq) // 2
        n_keepalive += comments.count(": preempt-keepalive")
    assert n_parked >= 2, [r[0] for r in results]
    assert n_keepalive >= 1, "a parked stream was left silent past the 2 s keepalive"

    text = open(server.log_path).read()
    assert "rotated out after" in text
    assert "no rotation: --preempt-ram 2 MiB" in text
    assert "resumed after" in text
    assert "Context size has been exceeded" not in text
