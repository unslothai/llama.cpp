import os
import time
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
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)


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



_WORDS = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt "
    "mollit anim id est laborum. "
) * 4


def _prompt_of_about(n_tokens: int, salt: str = "") -> tuple[str, int]:
    """A prompt whose token count is in [n_tokens - 12, n_tokens], measured on the server."""
    words = (salt + " " + _WORDS).split()
    while words:
        text = " ".join(words)
        res = server.make_request("POST", "/tokenize", data={"content": text})
        assert res.status_code == 200
        n = len(res.body["tokens"])
        if n <= n_tokens:
            assert n >= n_tokens - 12, f"could not land near {n_tokens} tokens, got {n}"
            return text, n
        # about four tokens per word on this model's vocabulary
        words = words[: len(words) - max(1, (n - n_tokens) // 8)]
    raise AssertionError("empty prompt")


def test_two_prompts_that_overflow_the_pool_together_both_finish():
    # Neither slot ever generates before the pool is full: both are still processing their
    # prompts. A prompt-processing slot is between two chunks of its prompt, which is as
    # clean a boundary as the one between two sampled tokens, so it is parked the same way.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_a, n_a = _prompt_of_about(150, "Alpha")
    prompt_b, n_b = _prompt_of_about(150, "Bravo")
    n_predict = 16
    assert n_a + n_predict <= 256 and n_b + n_predict <= 256
    assert n_a + n_b + 2 * n_predict > 256

    results = parallel_function_calls([
        (_complete, (n_predict, prompt_a)),
        (_complete, (n_predict, prompt_b)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert len(res.body["tokens"]) == n_predict


def test_a_generating_slot_and_a_large_prompt_both_finish():
    # One slot is generating a long answer to a short prompt when a large prompt arrives
    # beside it. Together they need far more than the pool has. The prompt is admitted
    # chunk by chunk, whoever is smaller is parked when the pool fills, and both finish.
    # This model produces a thousand tokens a second, so the second request is sent right
    # behind the first rather than after a delay: its prompt takes several batches to
    # process, which is enough for the two to overlap however fast the first one runs.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    # b lives long enough for the two to collide: the first run of this used 16 tokens
    # and b was finished and purged before a had grown into it
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a <= 256 and n_b + n_predict_b <= 256
    assert 8 + n_predict_a + n_b + n_predict_b > 256

    def _late(n_predict, prompt):
        time.sleep(0.02)
        return _complete(n_predict, prompt)

    results = parallel_function_calls([
        (_complete, (n_predict_a, "Hi how are you")),
        (_late, (n_predict_b, prompt_b)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text

    assert results[0].status_code == 200
    assert results[0].body["timings"]["predicted_n"] == n_predict_a
    assert results[1].status_code == 200
    assert results[1].body["timings"]["predicted_n"] == n_predict_b


def test_preempt_ram_zero_disables_preemption():
    # --preempt-ram 0 is the switch back to the old behaviour: nothing is parked and the
    # KV-full path ends the requests the way it always did.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "preempted:" not in text
    assert "Context size has been exceeded" in text
    assert any(res.status_code != 200 for res in results)
