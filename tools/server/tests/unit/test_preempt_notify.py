import os
import tempfile
import pytest
import requests
from utils import *

# [TAG_PREEMPT] A streaming client is told when its slot is parked and when it is
# restored, as SSE comments, and the body is byte for byte what it is without any park.
# Comments are legal SSE that every existing client ignores; a client that knows about
# preemption can show "paused" instead of a dead stream, and a keepalive every 2 s while
# parked keeps proxies and read timeouts from giving up on a wait that is by design long.

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


def test_a_stream_announces_its_parks_and_the_body_is_unchanged():
    global server
    server.n_ctx = 512
    server.start()
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
    # Every park is followed by its resume before the next park.
    seq = [c for c in comments if c in (": preempted", ": resumed")]
    assert seq == [": preempted", ": resumed"] * len(parked), seq
    # The generated text is byte for byte the unparked text, token by token. Only the
    # final chunk's wall-clock timings differ between the two runs.
    def _pieces(ds):
        return [json.loads(d).get("content") for d in ds if d != "[DONE]"]

    assert _pieces(datas) == _pieces(ref_datas)
    assert _content(datas) == _content(ref_datas)
    final, ref_final = json.loads(datas[-1]), json.loads(ref_datas[-1])
    assert final["tokens_predicted"] == ref_final["tokens_predicted"] == 64


def test_the_oai_chat_stream_carries_the_same_comments():
    global server
    server.n_ctx = 512
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    comments, datas = _stream_raw("/v1/chat/completions", _chat_payload(48))
    assert ": preempted" in comments and ": resumed" in comments
    assert datas[-1] == "[DONE]"
    assert _content(datas)


def test_non_streaming_requests_see_nothing():
    global server
    server.n_ctx = 512
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
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
    # The pair from test_preempt: each alone fits, together they do not, so one is
    # parked until the other finishes. The parked stream must carry the comments and
    # finish with its full output.
    global server
    server.n_ctx = 256
    server.start()

    n_predict = 160
    p1 = _completion_payload(n_predict) | {"prompt": "Once upon a time there was a brave knight who"}
    p2 = _completion_payload(n_predict) | {"prompt": "The quick brown fox jumps over the lazy dog and"}
    results = parallel_function_calls([
        (_stream_raw, ("/completion", p1)),
        (_stream_raw, ("/completion", p2)),
    ])
    announced = 0
    for comments, datas in results:
        final = json.loads([d for d in datas if d != "[DONE]"][-1])
        assert final["timings"]["predicted_n"] == n_predict
        assert final["truncated"] is False
        if ": preempted" in comments:
            announced += 1
            assert ": resumed" in comments
    assert announced >= 1, [r[0] for r in results]
