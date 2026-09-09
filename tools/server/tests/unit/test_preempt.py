import os
import re
import time
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool: when the next decode does not fit, one slot is parked (its
# sequence copied to host RAM, its cells released) instead of every slot being terminated. Needs
# more than one slot and --kv-unified, the only configuration where slots share cells.

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
    os.environ.pop("LLAMA_SERVER_PREEMPT_GRANULARITY", None)
    os.environ.pop("LLAMA_SERVER_PREEMPT_PLANNER", None)
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
    # park and restore the only running slot every 8 tokens: the batch shape is the same at every
    # step, so any difference in the output is the preemption's fault
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
    # each request fits the pool alone (168 of 256 cells) but not together (336). Without
    # preemption both end with "Context size has been exceeded"; with it the smaller is parked
    # until the leader finishes, then resumes from the token it was parked on.
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


def test_the_planner_counts_whole_pages_when_the_pool_allocates_in_pages():
    # a pool that allocates in blocks gives a whole block to one sequence, so n tokens occupy
    # round_up(n, block) cells and the planner has to count cells: counting tokens it sees room
    # the allocator cannot find, never parks anybody, and the retry ladder ends every request.
    # LLAMA_SERVER_PREEMPT_GRANULARITY injects the block size, since the only mode that reports
    # one needs a head size this model does not have; the arithmetic is the same at 64 as at 256.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_GRANULARITY = 64" in log.drain()

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    # every figure the planner logs is a whole number of blocks: "kv N/256" is what the pool holds
    # and "(wanted N)" is that plus the next decode's reservation
    held   = [int(n) for n in re.findall(r"kv (\d+)/256", text)]
    wanted = [int(n) for n in re.findall(r"\(wanted (\d+)\)", text)]
    assert held and wanted, f"the planner logged no figures:\n{text}"
    assert all(n % 64 == 0 for n in held + wanted), f"not whole blocks: {held} {wanted}"

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
    # neither slot generates before the pool is full: a slot between two chunks of its prompt is
    # as clean a boundary as one between two sampled tokens, so it is parked the same way
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
    # a slot generating a long answer to a short prompt meets a large prompt arriving beside it,
    # needing far more than the pool has: the prompt is admitted chunk by chunk, whoever is
    # smaller is parked, and both finish. The second request follows immediately, since its
    # prompt takes several batches and that is enough overlap however fast the first one runs.
    global server
    server.n_ctx = 256
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    # b has to live long enough for the two to collide
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
    # --preempt-ram 0 switches back to the old behaviour: nothing is parked and the KV-full path
    # ends the requests the way it always did
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


def test_metrics_and_slots_report_the_parked_state():
    # /slots tells a parked chat from a slow one and /metrics reports it to an operator; both
    # must show the preemption, and the counters must survive the requests finishing
    global server
    server.n_ctx = 256
    server.server_metrics = True
    server.start()

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["n_preempt"] == 0

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])
    for res in results:
        assert res.status_code == 200

    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    metrics = {}
    for line in res.body.splitlines():
        if line.startswith("llamacpp:"):
            name, value = line.split(" ", 1)
            metrics[name[len("llamacpp:"):]] = float(value)
    assert metrics["n_preempt_total"] >= 1
    assert metrics["n_resume_total"] == metrics["n_preempt_total"]
    assert metrics["requests_preempted"] == 0
    assert metrics["preempt_ram_bytes"] == 0

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    assert sum(slot["n_preempt"] for slot in res.body) == 0, "n_preempt is per task and resets with the slot"


def test_two_prompts_near_the_context_size_both_complete():
    # two prompts that each fit the context alone but not together. The second is parked before
    # it takes any cells and is too close to n_ctx to leave the usual margin, but must still be
    # restored once the first finishes: with nothing resident there is nobody to keep it for.
    global server
    server.n_ctx = 256
    # the whole prompt in one batch, so the parked slot's first step is the whole prompt
    server.n_batch = 256
    server.start()
    log = LogReader(server.log_path)

    # sized in tokens, not words: the prompt is the token ids of a short sentence repeated
    base = server.make_request("POST", "/tokenize", data={"content": "Once upon a time there was a little girl"}).body["tokens"]
    long_prompt = (base * 64)[:240]
    n_predict = 4
    together = parallel_function_calls([(_complete, (n_predict, long_prompt)) for _ in range(2)])

    text = log.drain()
    assert "cannot fit the pool" not in text

    for res in together:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_last_resort_parks_instead_of_ending_everyone():
    # with the planner off, two generations that fit alone but not together fill the pool until a
    # single token finds no cell, where upstream ends every slot with the context error. Instead
    # the batch is given up, the smaller slot is parked, and both finish.
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)
    assert "LLAMA_SERVER_PREEMPT_PLANNER = off" in log.drain()

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" not in text, "the planner was off, nothing may be parked ahead of the decode"
    assert "preempted as a last resort" in text
    assert "last resort: batch given up" in text
    assert "resumed after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict
        assert res.body["truncated"] is False
        assert len(res.body["tokens"]) == n_predict


def test_the_last_resort_works_with_an_unlimited_budget():
    # --preempt-ram -1 is the documented unlimited setting; it must enable the last resort
    # the same as any positive budget does
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    server.start()
    log = LogReader(server.log_path)

    n_predict = 160
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted as a last resort" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_last_resort_rewinds_a_prompt_in_flight():
    # same, with a prompt being processed when the pool runs out: the failed chunk comes back off
    # the slot's tokens and is processed again after the resume, neither skipped nor fed twice
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
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
    assert "preempted as a last resort" in text

    assert results[0].status_code == 200
    assert results[0].body["timings"]["predicted_n"] == n_predict_a
    assert results[1].status_code == 200
    assert results[1].body["timings"]["predicted_n"] == n_predict_b
    # the chunk that was in the batch given up is processed once, after the rewind, and
    # the count is the prompt plus the BOS the server adds
    assert results[1].body["timings"]["prompt_n"] == n_b + 1


def test_a_resident_cycling_through_context_shifts_takes_turns_with_a_parked_head():
    # two generations that each outgrow the pool, with context shift on: the resident shifts and
    # would hold half the pool for as long as it generates, while the parked one never fits
    # beside it. After the head has waited its turn the resident is parked in its place and the
    # two take turns. n_predict is large enough that the resident is still going by then.
    global server
    server.n_ctx = 256
    server.enable_ctx_shift = True
    server.start()
    log = LogReader(server.log_path)

    n_predict = 12000
    results = parallel_function_calls([
        (_complete, (n_predict, "Once upon a time there was a brave knight who")),
        (_complete, (n_predict, "The quick brown fox jumps over the lazy dog and")),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "slot context shift" in text
    assert "rotated out after" in text

    for res in results:
        assert res.status_code == 200
        assert res.body["timings"]["predicted_n"] == n_predict


def test_the_rotation_parks_a_resident_that_lets_the_head_in():
    # three endless generations with context shift on: two residents cycle through shifts while
    # the third waits parked, and every rotation must let the head in so no stream ends short
    global server
    server.n_slots = 3
    server.n_ctx = 384
    server.enable_ctx_shift = True
    server.start()
    n_predict = 9000
    prompts = [
        "Once upon a time there was a brave knight who",
        "The quick brown fox jumps over the lazy dog and",
        "In a small village by the sea there lived a fisherman who",
    ]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": p, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for p in prompts
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "rotated out after" in text
    assert "Context size has been exceeded" not in text


def test_a_parent_and_child_that_do_not_fit_alone_get_the_context_error_and_the_server_lives():
    # a two-completion request is one conversation in two slots, and a family member is not a
    # victim for the other, so with nobody else to park it gets the context error it would get
    # alone and the server carries on serving
    global server
    server.n_ctx = 256
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    server.start()
    log = LogReader(server.log_path)

    res = server.make_request("POST", "/completion", data={
        "n_predict": 160,
        "n_cmpl": 2,
        "prompt": "Once upon a time there was a brave knight who",
        "ignore_eos": True,
        "return_tokens": True,
        "temperature": 0.0,
        "seed": 42,
    })
    assert res.status_code == 500
    assert "Context size has been exceeded" in res.body["error"]["message"]

    text = log.drain()
    assert "preempted as a last resort" not in text, "a family alone in the pool has no victim"
    assert "GGML_ASSERT" not in text

    after = _complete(8)
    assert after.status_code == 200
    assert after.body["timings"]["predicted_n"] == 8


def test_a_budget_that_holds_one_sequence_does_not_rotate_and_the_head_resumes_when_a_resident_finishes():
    # Three generations with no end in a pool one of them fills, with context shift on,
    # under a --preempt-ram that holds the two parked heads but not a head and the resident
    # at once. The resident is parked before the head is restored and freed, so a rotation
    # holds both states together: under this budget the first one asked for is refused and
    # said so, and the heads come back when the resident finishes instead. Every stream
    # still finishes its tokens and nothing gets the context error.
    global server
    server.n_slots = 3
    server.n_ctx = 2048
    server.enable_ctx_shift = True
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    server.start()
    # long enough that the resident is still cycling through shifts two seconds after the
    # heads were parked, which is when a rotation is first asked for: at 6000 this model
    # finished in under three seconds on a fast host and nothing was ever refused
    n_predict = 12000
    prompts = [
        "Once upon a time there was a brave knight who",
        "The quick brown fox jumps over the lazy dog and",
        "In a small village by the sea there lived a fisherman who",
    ]
    results = parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": p, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for p in prompts
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "no rotation: --preempt-ram 2 MiB" in text
    assert "resumed after" in text
    assert "Context size has been exceeded" not in text


def test_a_recurrent_model_is_served_without_preemption():
    # A recurrent cache holds one state per sequence whatever its length, so the token
    # count the planner measures says nothing about it: preemption is off for such a
    # model, said so at load, and the forced-park knob parks nothing.
    global server
    path = os.environ.get("LLAMA_SERVER_TEST_RECURRENT_MODEL")
    if path:
        server.model_file = path
    else:
        server.model_file = None
        server.model_hf_repo = "Felladrin/gguf-mamba-130m-hf"
        server.model_hf_file = "mamba-130m-hf.Q2_K.gguf"
        server.offline = False
    server.n_ctx = 1024
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start(timeout_seconds=300)
    results = parallel_function_calls([
        (_complete, (64, "Once upon a time")),
        (_complete, (64, "The quick brown fox")),
    ])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == 64
    text = open(server.log_path).read()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    assert "preempted" not in text
    assert "Context size has been exceeded" not in text
