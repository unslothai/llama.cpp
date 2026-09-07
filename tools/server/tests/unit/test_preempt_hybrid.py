import os
import tempfile
import threading
import pytest
from utils import *

# The hybrid half of preemption: WHICH mechanism parks a slot, and that the choice is made by
# the --preempt-ram budget. test_preempt.py covers the policy itself (who is parked, in what
# order they come back, that nobody is terminated); everything here is about the two ways of
# parking and the counters that tell them apart.
#
#   default budget      every park fits, so every park is a swap, and the output is exact
#   --preempt-ram 0     nothing fits, so every park is a recompute, and the run still finishes
#                       (on #184 this configuration disabled preemption and both requests died)
#   --preempt-ram 1     a budget small enough to hold some parks and not others: both

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
    server.server_metrics = True
    server.temperature = 0.0
    server.seed = 42
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    yield
    os.environ.pop("LLAMA_SERVER_PREEMPT_EVERY", None)
    os.environ.pop("LLAMA_ARG_PREEMPT_RAM", None)
    os.environ.pop("LLAMA_ARG_PREEMPT", None)


def _complete(n_predict: int, prompt: str = "Hi how are you"):
    return server.make_request("POST", "/completion", data={
        "n_predict": n_predict,
        "prompt": prompt,
        "ignore_eos": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
    })


class SlotWatcher:
    """Poll /slots while requests are in flight and keep every parked state it saw.

    The counters on /metrics say a park happened; only this says what a client polling /slots
    would have been told while it was happening."""

    def __init__(self):
        self.seen = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                res = server.make_request("GET", "/slots", timeout=5)
                if res.status_code == 200:
                    for slot in res.body:
                        if slot["is_preempted"]:
                            self.seen.append((slot["preempt_mode"], slot["n_preempt"]))
            except Exception:
                pass
            self._stop.wait(0.02)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)

    @property
    def modes(self):
        return {mode for mode, _ in self.seen}


def _metrics():
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    out = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ", 1)
            out[name[len("llamacpp:"):]] = float(value)
    return out


_WORDS = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
    "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure "
    "dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
) * 24


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
        words = words[: len(words) - max(1, (n - n_tokens) // 8)]
    raise AssertionError("empty prompt")


def test_default_budget_parks_by_swap_and_the_output_is_identical():
    # Under the default 8192 MiB every park fits in host RAM, so the hybrid must behave
    # exactly as #184: a copy out and a copy back, byte-identical output, and no recompute.
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

    preempted = _complete(64)
    assert preempted.status_code == 200
    assert preempted.body["content"] == reference.body["content"]
    assert preempted.body["tokens"] == reference.body["tokens"]

    text = log.drain()
    assert "parked by swap" in text
    assert "parked by recompute" not in text

    m = _metrics()
    assert m["n_preempt_total"] >= 6
    assert m["n_preempt_swap_total"] == m["n_preempt_total"]
    assert m["n_preempt_recompute_total"] == 0
    assert m["n_recompute_tokens_total"] == 0
    assert m["n_resume_total"] == m["n_preempt_total"]
    assert m["preempt_ram_bytes"] == 0


def test_preempt_ram_zero_parks_by_recompute_and_both_requests_finish():
    # The case #184 could not serve. With --preempt-ram 0 no sequence may be copied to host
    # RAM, so on #184 nothing was parked and the KV-full ladder ended both requests with
    # "Context size has been exceeded". Here the same budget means "always recompute", and
    # both requests finish with nothing held in host RAM at all.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    with SlotWatcher() as watcher:
        results = parallel_function_calls([
            (_complete, (n_predict, "Once upon a time there was a brave knight who")),
            (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
        ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "parked by recompute" in text
    assert "parked by swap" not in text

    # whatever /slots showed while a slot was parked, it can only have been a recompute; the
    # poll can miss a short park, so it is the mode that is asserted, not that it saw one
    assert watcher.modes <= {"recompute"}, watcher.seen

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict

    m = _metrics()
    assert m["n_preempt_total"] >= 1
    assert m["n_preempt_recompute_total"] == m["n_preempt_total"]
    assert m["n_preempt_swap_total"] == 0
    assert m["n_recompute_tokens_total"] >= 1
    assert m["n_resume_total"] == m["n_preempt_total"]
    assert m["preempt_ram_bytes"] == 0
    assert m["requests_preempted"] == 0


def test_a_small_budget_parks_both_ways():
    # The point of the hybrid: the budget is a boundary, not a switch. With four slots on a
    # small pool several sequences are parked at once, the first ones fit under 1 MiB and are
    # copied, the ones after them do not and are recomputed, and every request still finishes.
    #
    # Four slots and not two: with two slots only one can ever be parked at a time (the leader
    # is never a victim), one park of this model is well under 1 MiB, and the budget would
    # never be reached. This model holds about 740 bytes of KV per token, so 1 MiB runs out
    # at roughly 1400 parked tokens, which four 1500-token sequences cross and four
    # 400-token ones do not.
    global server
    server.n_ctx = 4096
    server.n_slots = 4
    # the preset's 32-token batch would take thousands of iterations to prefill four 600-token
    # prompts into a 4096-cell pool; nothing here depends on the batch being small
    server.n_batch = 512
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "1"
    server.start()
    log = LogReader(server.log_path)

    prompts = [_prompt_of_about(600, salt)[0] for salt in ("Alpha", "Bravo", "Charlie", "Delta")]
    # 4 x 1500 cells wanted from a pool of 4096: the early, smaller parks are copied and the
    # later, larger ones do not fit the budget and are recomputed
    n_predict = 900
    results = parallel_function_calls([(_complete, (n_predict, p)) for p in prompts])

    text = log.drain()
    assert "Context size has been exceeded" not in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert len(res.body["tokens"]) == n_predict

    m = _metrics()
    assert m["n_preempt_swap_total"] >= 1, "nothing was parked by swap under a 1 MiB budget"
    assert m["n_preempt_recompute_total"] >= 1, "the budget never pushed a park to recompute"
    assert m["n_preempt_swap_total"] + m["n_preempt_recompute_total"] == m["n_preempt_total"]
    assert m["n_resume_total"] == m["n_preempt_total"]
    assert m["preempt_ram_bytes"] == 0


def test_a_recomputed_request_does_not_double_count_its_prompt():
    # A recompute runs the prompt through the model again. That is the server's problem and not
    # the client's: usage.prompt_tokens is the prompt the client sent, and the timings describe
    # the request it made, however many times the server had to prefill it. The work is
    # reported separately, as n_recompute on /slots and n_recompute_tokens_total on /metrics.
    global server
    server.n_ctx = 512
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start()

    prompt = "Once upon a time there was a brave knight who"
    plain = server.make_request("POST", "/tokenize", data={"content": prompt})
    assert plain.status_code == 200
    n_prompt = len(plain.body["tokens"])

    n_predict = 64
    res = _complete(n_predict, prompt)
    assert res.status_code == 200

    m = _metrics()
    assert m["n_preempt_recompute_total"] >= 6, "the request was never recomputed"
    assert m["n_preempt_swap_total"] == 0
    # ... and it really did re-run more tokens than the request ever had
    assert m["n_recompute_tokens_total"] > n_prompt + n_predict

    # /tokenize does not add BOS, the prompt path does, so allow the one extra token
    assert res.body["tokens_evaluated"] - n_prompt in (0, 1)
    timings = res.body["timings"]
    assert timings["prompt_n"] == res.body["tokens_evaluated"], "the recompute was billed as prompt"
    assert timings["predicted_n"] == n_predict
    # a prompt timestamp dragged forward by the recomputes would report the whole wall clock as
    # prompt time and inflate the generation rate by the same amount
    assert timings["prompt_ms"] < timings["predicted_ms"]

    oai = server.make_request("POST", "/v1/completions", data={
        "prompt": prompt,
        "max_tokens": n_predict,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
    })
    assert oai.status_code == 200
    usage = oai.body["usage"]
    assert usage["prompt_tokens"] - n_prompt in (0, 1)
    assert usage["completion_tokens"] == n_predict
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
