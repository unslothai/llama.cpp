import os
import pytest
from utils import *

# Requests and models outside the parking path that the preemption change must leave as upstream serves them.

server = ServerPreset.tinyllama2()

START_TIMEOUT = 300  # the preload can be slow on a loaded host


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    yield
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)


@pytest.mark.parametrize("preempt_ram", [None, "1024"])
def test_a_non_causal_decision_longer_than_the_ubatch_is_refused_and_the_server_lives(preempt_ram):
    global server
    server = ServerPreset.tinyopenjev()
    server.override_kv = ["qwen35.attention.causal=bool:false"]
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
