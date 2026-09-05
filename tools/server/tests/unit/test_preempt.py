import os
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool: when the next decode does not fit, one slot is parked
# (its sequence copied to host RAM, its cells released) instead of every slot being
# terminated. Both tests need more than one slot and --kv-unified, which is the only
# configuration where one slot can take another one's cells.

server = ServerPreset.tinyllama2()


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.kv_unified = True
    server.server_slots = True
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    os.environ.pop("LLAMA_SERVER_PREEMPT_EVERY", None)


def _complete(n_predict: int, prompt: str = "Hi how are you"):
    res = server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "ignore_eos": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
    })
    return res


def test_forced_preemption_does_not_change_the_output():
    # Park and restore the only running slot every 8 tokens. With one request the batch
    # has the same shape at every step whether or not the slot was parked in between, so
    # any difference in the output is the preemption's fault and nothing else's.
    global server
    server.n_ctx = 512
    server.start()
    reference = _complete(64)
    assert reference.status_code == 200
    assert reference.body["timings"]["predicted_n"] == 64
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_EVERY = 8" in log.drain()

    preempted = _complete(64)
    assert preempted.status_code == 200
    assert preempted.body["timings"]["predicted_n"] == 64

    text = log.drain()
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6

    assert preempted.body["content"] == reference.body["content"]
    assert preempted.body["tokens"] == reference.body["tokens"]


def test_two_slots_that_overflow_the_pool_together_both_finish():
    # Each request alone fits in the pool: 8 prompt tokens plus 160 generated is well
    # under 256. Together they do not, 336 against 256. Without preemption the retry
    # ladder ends with "Context size has been exceeded" on every processing slot; with it
    # the smaller slot is parked until the leader finishes and its cells are purged, and
    # then it resumes from the token it was parked on.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict

