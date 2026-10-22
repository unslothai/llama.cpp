import os
import re
import tempfile
import pytest
from utils import *


server = ServerPreset.tinyllama2()

START_TIMEOUT = 300


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_threads = 2
    yield
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)


@pytest.mark.parametrize("preempt_ram", [None, "1024"])
def test_a_non_causal_decision_longer_than_the_ubatch_is_refused_and_the_server_lives(preempt_ram):
    global server
    server = ServerPreset.tinyopenjev()
    server.n_threads = 2
    server.override_kvs = ["qwen35.attention.causal=bool:false"]
    server.n_ctx = 1024
    server.n_slots = 2
    server.kv_unified = True
    server.n_batch = 256
    server.n_ubatch = 32
    if preempt_ram is not None:
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = preempt_ram
    server.start(timeout_seconds=START_TIMEOUT)
    res = server.make_request("POST", "/v1/systemone", data={
        "state": "Hello " * 30,
        "questions": {"q": {
            "type": "choice",
            "instructions": "Is this a greeting?",
            "criteria": {"yes": "a greeting", "no": "something else"},
        }},
    })
    assert res.status_code == 500
    assert "too large to process" in res.body["error"]["message"]
    assert server.make_request("GET", "/health").status_code == 200


def _decision(state: str):
    return server.make_request("POST", "/v1/systemone", data={
        "state": state,
        "questions": {"q": {
            "type": "choice",
            "instructions": "Is this a greeting?",
            "criteria": {"yes": "a greeting", "no": "something else"},
        }},
    })


def _decision_n_tokens(words: int) -> int:
    """Input tokens of a decision whose state is `words` words, read from the refusal of one longer than the ubatch (= the context here)."""
    res = _decision("Hello " * words)
    assert res.status_code != 200, "this state should not fit the context"
    m = re.search(r"input \((\d+) tokens\) is (?:too large to process|larger than the max context size)", res.body["error"]["message"])
    assert m, res.body
    return int(m.group(1))


def test_a_decision_as_long_as_the_context_resumes_after_a_park():
    """Admission takes a non-causal prompt of exactly n_ctx tokens, so a parked one must fit the pool again on resume."""
    global server
    server = ServerPreset.tinyopenjev()
    server.n_threads = 2
    server.override_kvs = ["qwen35.attention.causal=bool:false"]
    n_ctx = 256
    server.n_ctx = n_ctx
    server.n_slots = 2
    server.kv_unified = True
    server.n_batch = n_ctx
    server.n_ubatch = n_ctx
    server.cache_ram = 0
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "32"
    server.start(timeout_seconds=START_TIMEOUT)

    # the state that makes the input exactly n_ctx tokens: one word is one token here
    a, b = 2 * n_ctx, 2 * n_ctx + 20
    n_a, n_b = _decision_n_tokens(a), _decision_n_tokens(b)
    assert n_b - n_a == b - a, f"one word is not one token: {n_a} -> {n_b}"
    words = a - (n_a - n_ctx)
    assert _decision_n_tokens(words + 1) == n_ctx + 1

    results = parallel_function_calls([(_decision, ("Hello " * words,)) for _ in range(12)])
    for res in results:
        assert res.status_code == 200, res.body

    text = open(server.log_path, errors="replace").read()
    assert "preempted" in text, "no decision was parked, so the resume was not exercised"
    assert "cannot fit the pool" not in text


def _final_bodies():
    """The final record of a completion on every endpoint that reports how the request was served."""
    msgs = [{"role": "user", "content": "Hello"}]
    bodies = {}
    res = server.make_request("POST", "/completion", data={"prompt": "Hello", "n_predict": 4})
    bodies["completion"] = res.body
    res = server.make_request("POST", "/v1/completions", data={"prompt": "Hello", "max_tokens": 4})
    bodies["oai_completion"] = res.body
    res = server.make_request("POST", "/v1/chat/completions", data={"messages": msgs, "max_tokens": 4})
    bodies["chat"] = res.body
    res = server.make_request("POST", "/v1/responses", data={"input": "Hello", "max_output_tokens": 4})
    bodies["responses"] = res.body
    res = server.make_request("POST", "/v1/messages", data={"messages": msgs, "max_tokens": 4})
    bodies["messages"] = res.body
    events = list(server.make_stream_request("POST", "/v1/chat/completions", data={
        "messages": msgs, "max_tokens": 4, "stream": True, "stream_options": {"include_usage": True}}))
    bodies["chat_stream"] = events[-1]
    events = list(server.make_stream_request("POST", "/v1/responses", data={
        "input": "Hello", "max_output_tokens": 4, "stream": True}))
    bodies["responses_stream"] = [e for e in events if e.get("type") == "response.completed"][0]["response"]
    return bodies


@pytest.mark.parametrize("preempt_ram", [None, "1024"])
def test_the_preempt_record_is_sent_only_by_a_server_that_can_park(preempt_ram):
    global server
    server.n_slots = 2
    server.kv_unified = True
    server.jinja = True
    if preempt_ram is not None:
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = preempt_ram
    server.start(timeout_seconds=START_TIMEOUT)
    for name, body in _final_bodies().items():
        if preempt_ram is None:
            assert "preempt" not in body, name
        else:
            assert body["preempt"] == {"parks": 0, "recomputes": 0}, name


@pytest.mark.parametrize("preempt_ram", [None, "1024"])
def test_a_context_shift_is_left_to_the_decode_unless_parking_is_on(preempt_ram, tmp_path):
    global server
    server.n_ctx = 256
    server.n_slots = 2
    server.kv_unified = True
    server.enable_ctx_shift = True
    server.debug = True
    server.temperature = 0.0
    server.log_path = str(tmp_path / "server.log")
    if preempt_ram is not None:
        os.environ["LLAMA_ARG_PREEMPT_RAM"] = preempt_ram
    server.start(timeout_seconds=START_TIMEOUT)
    res = server.make_request("POST", "/completion", data={"prompt": "Once upon a time", "n_predict": 320})
    assert res.status_code == 200
    assert res.body["truncated"] is True
    server.stop()

    text = open(server.log_path).read()
    after = text[text.index("slot context shift"):]
    i_decode = after.index("n_batch (effective)")
    i_update = after.index("applying K-shift")
    if preempt_ram is None:
        assert i_decode < i_update
    else:
        assert i_update < i_decode
