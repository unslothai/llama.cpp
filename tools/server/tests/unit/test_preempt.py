import os
import re
import struct
import subprocess
import threading
import time
import tempfile
import pytest
from utils import *

# Preemption on a unified KV pool: one slot is parked, its sequence copied to host RAM and its cells released, instead of every slot being terminated. Needs --kv-unified.

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
    os.environ.pop("LLAMA_ARG_PREEMPT_ASYNC", None)


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


_PROMPT_A = "Once upon a time there was a brave knight who"
_PROMPT_B = "The quick brown fox jumps over the lazy dog and"
_PROMPT_C = "In a small village by the sea there lived a fisherman who"


def _start(**kwargs) -> LogReader:
    """Start the server with these settings, and read its log from the first line."""
    for key, value in kwargs.items():
        setattr(server, key, value)
    server.start()
    return LogReader(server.log_path)


def _late(n_predict: int, prompt: str, delay: float = 0.02):
    time.sleep(delay)
    return _complete(n_predict, prompt)


def _complete_all(n_predict: int, prompts=(_PROMPT_A, _PROMPT_B)):
    return parallel_function_calls([(_complete, (n_predict, prompt)) for prompt in prompts])


def _complete_all_raw(n_predict: int, prompts):
    """As _complete_all, without return_tokens: these ask for thousands of tokens."""
    return parallel_function_calls([
        (server.make_request, ("POST", "/completion", {
            "prompt": prompt, "n_predict": n_predict, "ignore_eos": True, "temperature": 0.0, "seed": 42,
        })) for prompt in prompts
    ])


def _assert_completed(results, n_predict: int, whole: bool = False):
    """Every request generated what it asked for; `whole` also pins the untruncated body."""
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == n_predict
        if whole:
            assert res.body["truncated"] is False
            assert len(res.body["tokens"]) == n_predict


def test_forced_preemption_does_not_change_the_output():
    _start(n_ctx=512)
    reference = _complete(64)
    assert reference.status_code == 200
    assert reference.body["timings"]["predicted_n"] == 64
    server.stop()

    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    log = _start()
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
    # each request fits the pool alone (168 of 256 cells) but not together; without preemption both end with "Context size has been exceeded"
    log = _start(n_ctx=256)

    n_predict = 160
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    _assert_completed(results, n_predict, whole=True)


def test_the_planner_counts_whole_pages_when_the_pool_allocates_in_pages():
    # a block allocator gives a whole block to one sequence, so the planner has to count cells: counting tokens it sees room the allocator cannot find. GRANULARITY injects the size.
    os.environ["LLAMA_SERVER_PREEMPT_GRANULARITY"] = "64"
    log = _start(n_ctx=256)
    assert "LLAMA_SERVER_PREEMPT_GRANULARITY = 64" in log.drain()

    n_predict = 160
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    held   = [int(n) for n in re.findall(r"kv (\d+)/256", text)]
    wanted = [int(n) for n in re.findall(r"\(wanted (\d+)\)", text)]
    assert held and wanted, f"the planner logged no figures:\n{text}"
    assert all(n % 64 == 0 for n in held + wanted), f"not whole blocks: {held} {wanted}"

    _assert_completed(results, n_predict, whole=True)


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
        words = words[: len(words) - max(1, (n - n_tokens) // 8)]
    raise AssertionError("empty prompt")


def test_two_prompts_that_overflow_the_pool_together_both_finish():
    log = _start(n_ctx=256)

    prompt_a, n_a = _prompt_of_about(150, "Alpha")
    prompt_b, n_b = _prompt_of_about(150, "Bravo")
    n_predict = 16
    assert n_a + n_predict <= 256 and n_b + n_predict <= 256
    assert n_a + n_b + 2 * n_predict > 256

    results = _complete_all(n_predict, [prompt_a, prompt_b])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    _assert_completed(results, n_predict)
    for res in results:
        assert len(res.body["tokens"]) == n_predict


def test_a_generating_slot_and_a_large_prompt_both_finish():
    log = _start(n_ctx=256)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a <= 256 and n_b + n_predict_b <= 256
    assert 8 + n_predict_a + n_b + n_predict_b > 256

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
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    log = _start(n_ctx=256)

    n_predict = 160
    results = _complete_all(n_predict)

    text = log.drain()
    assert "preempted:" not in text
    assert "Context size has been exceeded" in text
    assert any(res.status_code != 200 for res in results)


def test_metrics_and_slots_report_the_parked_state():
    _start(n_ctx=256, server_metrics=True)

    res = server.make_request("GET", "/slots")
    assert res.status_code == 200
    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["is_transferring"] is False
        assert slot["n_preempt"] == 0

    n_predict = 160
    results = _complete_all(n_predict)
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


# [TAG_PREEMPT_ASYNC] parking and resuming on a stream of their own, only on a backend that can copy asynchronously and signal an event; a CPU-only build falls back and these skip

_ASYNC_BANNER = "parking and resuming asynchronously"


def _start_async(**kwargs) -> str:
    """Start the server with the asynchronous path asked for, and return its log so far."""
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    return _start(**kwargs).drain()


def _require_async(text: str):
    if _ASYNC_BANNER not in text:
        pytest.skip("this backend cannot copy asynchronously, the async park path is not exercised")


def test_async_preemption_does_not_change_the_output():
    text = _start_async(n_ctx=512, n_gpu_layer=99)
    _require_async(text)

    res_plain = _complete(64)
    assert res_plain.status_code == 200

    server.stop()
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    log = _start()

    res_preempted = _complete(64)
    assert res_preempted.status_code == 200

    text = log.drain()
    _require_async(text)
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6
    assert "park completed after" in text
    assert "restore issued in" in text
    assert "restore completed after" in text

    assert res_preempted.body["content"] == res_plain.body["content"]
    assert res_preempted.body["tokens"] == res_plain.body["tokens"]


def test_async_preemption_under_load_keeps_every_slot_and_its_output():
    text = _start_async(n_ctx=256, n_gpu_layer=99)
    _require_async(text)

    n_predict = 160

    alone = [_complete(n_predict, prompt) for prompt in (_PROMPT_A, _PROMPT_B)]
    for res in alone:
        assert res.status_code == 200

    server.stop()
    log = _start()

    together = _complete_all(n_predict)

    text = log.drain()
    _require_async(text)
    assert "Context size has been exceeded" not in text
    assert "preempted:" in text
    assert "resumed after" in text

    _assert_completed(together, n_predict)
    for res, ref in zip(together, alone):
        assert res.body["truncated"] is False
        assert res.body["tokens"] == ref.body["tokens"]


def _cancel_soon(n_predict: int, prompt: str, timeout: float):
    try:
        server.make_request("POST", "/completion", data={
            "n_predict": n_predict,
            "prompt": prompt,
            "ignore_eos": True,
            "temperature": 0.0,
            "seed": 42,
        }, timeout=timeout)
    except Exception:
        pass  # the point is the drop, not the response


def test_cancel_while_a_copy_is_in_flight_frees_the_slot():
    # a cancelled request can reach release() with a park or a resume still running, where the host buffer is freed and the cells handed on, so both have to wait for the copy
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    text = _start_async(n_ctx=512, n_gpu_layer=99)
    _require_async(text)

    for i in range(4):
        _cancel_soon(96, _PROMPT_A, 0.05 + 0.1 * i)

    deadline = time.time() + 120
    while time.time() < deadline:
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        if all(not slot["is_processing"] for slot in res.body):
            break
        time.sleep(0.2)
    else:
        pytest.fail("a slot never came back after a cancel during a copy")

    for slot in res.body:
        assert slot["is_preempted"] is False
        assert slot["is_transferring"] is False

    if server.server_metrics:
        res = server.make_request("GET", "/metrics")
        for line in res.body.splitlines():
            if line.startswith("llamacpp:preempt_ram_bytes"):
                assert float(line.split(" ", 1)[1]) == 0, "a cancelled slot kept its parked memory"

    res = _complete(16)
    assert res.status_code == 200
    assert res.body["timings"]["predicted_n"] == 16


def test_no_preempt_async_falls_back_to_the_synchronous_path():
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    log = _start(n_ctx=512, n_gpu_layer=99)

    res = _complete(64)
    assert res.status_code == 200

    text = log.drain()
    assert _ASYNC_BANNER not in text
    assert "park issued in" not in text
    assert "restore issued in" not in text
    assert text.count("preempted on request") >= 6
    assert text.count("resumed after") >= 6


def test_a_prompt_arriving_into_a_nearly_full_pool_parks_rather_than_ends_everything():
    # [TAG_PREEMPT_ASYNC] the case the async path made worse than the synchronous one: an asynchronous park does not return the cells before update_slots() carries on
    log = _start(n_ctx=512, n_gpu_layer=99, n_slots=4)

    prompt_a, n_a = _prompt_of_about(100, "Alpha")
    prompt_b, n_b = _prompt_of_about(100, "Bravo")
    prompt_c, n_c = _prompt_of_about(100, "Charlie")
    prompt_d, n_d = _prompt_of_about(150, "Delta")

    n_predict_abc = 130
    n_predict_d = 40
    assert max(n_a, n_b, n_c) + n_predict_abc < 512 and n_d + n_predict_d < 512
    assert n_a + n_b + n_c + 3 * n_predict_abc > 512

    results = parallel_function_calls([
        (_complete, (n_predict_abc, prompt_a)),
        (_complete, (n_predict_abc, prompt_b)),
        (_complete, (n_predict_abc, prompt_c)),
        (_late,     (n_predict_d, prompt_d, 0.25)),
    ])

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted" in text

    for i, res in enumerate(results):
        assert res.status_code == 200, (i, res.body)
    for i in range(3):
        assert results[i].body["timings"]["predicted_n"] == n_predict_abc
    assert results[3].body["timings"]["predicted_n"] == n_predict_d


def test_two_prompts_near_the_context_size_both_complete():
    # the second prompt is parked before it takes any cells and is too close to n_ctx to leave the usual margin, but must still be restored once the first finishes
    log = _start(n_ctx=256, n_batch=256)

    base = server.make_request("POST", "/tokenize", data={"content": "Once upon a time there was a little girl"}).body["tokens"]
    long_prompt = (base * 64)[:240]
    n_predict = 4
    together = _complete_all(n_predict, [long_prompt, long_prompt])

    text = log.drain()
    assert "cannot fit the pool" not in text

    _assert_completed(together, n_predict)


def test_the_last_resort_parks_instead_of_ending_everyone():
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    log = _start(n_ctx=256)
    assert "LLAMA_SERVER_PREEMPT_PLANNER = off" in log.drain()

    n_predict = 160
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted:" not in text, "the planner was off, nothing may be parked ahead of the decode"
    assert "preempted as a last resort" in text
    assert "last resort: batch given up" in text
    assert "resumed after" in text

    _assert_completed(results, n_predict, whole=True)


def test_the_last_resort_works_with_an_unlimited_budget():
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "-1"
    log = _start(n_ctx=256)

    n_predict = 160
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "preempted as a last resort" in text

    _assert_completed(results, n_predict)


def test_the_last_resort_rewinds_a_prompt_in_flight():
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    log = _start(n_ctx=256)

    prompt_b, n_b = _prompt_of_about(150, "Charlie")
    n_predict_a = 230
    n_predict_b = 90
    assert 8 + n_predict_a <= 256 and n_b + n_predict_b <= 256
    assert 8 + n_predict_a + n_b + n_predict_b > 256

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
    # the chunk in the batch given up is processed once after the rewind; the count is the prompt plus the BOS the server adds
    assert results[1].body["timings"]["prompt_n"] == n_b + 1


def test_a_resident_cycling_through_context_shifts_takes_turns_with_a_parked_head():
    # with context shift on the resident would hold half the pool for as long as it generates, so once the head has waited its turn the resident is parked and the two take turns
    log = _start(n_ctx=256, enable_ctx_shift=True)

    n_predict = 12000
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "slot context shift" in text
    assert "rotated out after" in text

    _assert_completed(results, n_predict)


def test_the_rotation_parks_a_resident_that_lets_the_head_in():
    _start(n_slots=3, n_ctx=384, enable_ctx_shift=True)
    n_predict = 9000
    results = _complete_all_raw(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C))
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "rotated out after" in text
    assert "Context size has been exceeded" not in text


def test_a_parent_and_child_that_do_not_fit_alone_get_the_context_error_and_the_server_lives():
    # a family member is not a victim for the other, so a two-completion request gets the context error it would get alone and the server carries on
    os.environ["LLAMA_SERVER_PREEMPT_PLANNER"] = "off"
    log = _start(n_ctx=256)

    res = server.make_request("POST", "/completion", data={
        "n_predict": 160,
        "n_cmpl": 2,
        "prompt": _PROMPT_A,
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


def test_a_restored_slot_gives_its_idle_buffer_back_when_another_slot_needs_to_park():
    # an asynchronous slot keeps its pinned buffer after a restore, and that idle capacity counts against --preempt-ram: unless it is given back, the first restore spends the budget
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "256"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    text = _start_async(n_ctx=8192, n_gpu_layer=99)
    _require_async(text)
    log = LogReader(server.log_path)

    n_predict = 1800
    results = _complete_all(n_predict)

    text = log.drain()
    assert "Context size has been exceeded" not in text
    assert "idle parked RAM returned" in text, "the idle buffer of a restored slot was never given back"
    parked = re.findall(r"id\s+(\d+) \| task \d+ \| preempted on request", text)
    assert {"0", "1"} <= set(parked), f"only slots {sorted(set(parked))} were ever parked"

    _assert_completed(results, n_predict)
    for res in results:
        assert res.body["truncated"] is False


def test_a_budget_that_holds_one_sequence_does_not_rotate_and_the_head_resumes_when_a_resident_finishes():
    # a rotation holds both states at once, since the resident is parked before the head is restored and freed, so a budget for two heads but not a head plus the resident must refuse
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "2"
    _start(n_slots=3, n_ctx=2048, enable_ctx_shift=True)
    n_predict = 12000
    results = _complete_all_raw(n_predict, (_PROMPT_A, _PROMPT_B, _PROMPT_C))
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == n_predict
    text = open(server.log_path).read()
    assert "no rotation: --preempt-ram 2 MiB" in text
    assert "resumed after" in text
    assert "Context size has been exceeded" not in text


def test_a_recurrent_model_is_served_without_preemption():
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
    results = _complete_all(64, ["Once upon a time", "The quick brown fox"])
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == 64
    text = open(server.log_path).read()
    assert "preemption: off, the recurrent cache holds one state per sequence" in text
    assert "preempted" not in text
    assert "Context size has been exceeded" not in text


def test_a_hybrid_model_parks_synchronously():
    # the recurrent half of a hybrid gathers the active sequences into contiguous rows on every batch, so a copy running beside the decode could read a row another sequence has been moved into
    path = os.environ.get("LLAMA_SERVER_TEST_HYBRID_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_HYBRID_MODEL to a hybrid attention/recurrent gguf")
    server.model_file = path
    server.model_hf_repo = None
    server.model_hf_file = None
    server.n_ctx = 1024
    server.n_gpu_layer = 99
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "1"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "8"
    server.start(timeout_seconds=600)

    res = _complete(24, "Once upon a time")
    assert res.status_code == 200, res.body
    assert res.body["timings"]["predicted_n"] == 24

    text = open(server.log_path).read()
    assert "a recurrent state does not stay in one row" in text
    assert _ASYNC_BANNER not in text
    assert "park issued in" not in text
    assert "preempted on request" in text
    assert "resumed after" in text


# [TAG_EXACT_CONCURRENCY] the paged pool places a cell from the sequence and the position alone, so a layout that gives several tokens one position cannot be served

def _server_bin() -> str:
    return os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")


def _empty_gguf(path: str):
    """A header-only gguf: enough for a projector argument that is never read."""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 0))


def _exact_env() -> dict:
    return {**os.environ, "LLAMA_EXACT_CONCURRENCY": "1"}


def test_exact_concurrency_refuses_an_mrope_model_with_a_projector():
    # every token of one image shares a temporal position under M-RoPE, so the pool would give the second one the first one's cell and refuse the batch at the first image
    path = os.environ.get("LLAMA_SERVER_TEST_MROPE_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MROPE_MODEL to an M-RoPE gguf")

    with tempfile.TemporaryDirectory() as tmp:
        mmproj = os.path.join(tmp, "mmproj.gguf")
        _empty_gguf(mmproj)
        proc = subprocess.run([
            _server_bin(), "--model", path, "--mmproj", mmproj,
            "--host", "127.0.0.1", "--port", str(server.server_port),
            "-c", "512", "--parallel", "2", "--kv-unified", "-fa", "on",
            "-ngl", "99", "--no-warmup", "--no-webui",
        ], env=_exact_env(), capture_output=True, text=True, timeout=900)

    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "does not support M-RoPE together with a projector" in out, out


def test_exact_concurrency_serves_an_mrope_model_without_a_projector():
    # the refusal is about images, not the rope layout: a text prompt gives every token its own position, which the pool places
    path = os.environ.get("LLAMA_SERVER_TEST_MROPE_MODEL")
    if not path:
        pytest.skip("set LLAMA_SERVER_TEST_MROPE_MODEL to an M-RoPE gguf")

    os.environ["LLAMA_EXACT_CONCURRENCY"] = "1"
    try:
        server.model_file = path
        server.model_hf_repo = None
        server.model_hf_file = None
        log = _start(n_ctx=512, n_slots=2, fa="on", n_gpu_layer=99)

        res = _complete(16, "Once upon a time")
        assert res.status_code == 200, res.body
        assert res.body["timings"]["predicted_n"] == 16

        text = log.drain()
        assert "does not support M-RoPE" not in text
        assert "the kv pool allocates 256 cells at a time" in text
    finally:
        os.environ.pop("LLAMA_EXACT_CONCURRENCY", None)


def test_slots_reports_a_transferring_slot_apart_from_a_parked_one():
    # a copy out still owns its cells and a restore has already taken them back, so a reader counting residency has to keep counting both; only a fully parked slot holds nothing
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = "1"
    text = _start_async(n_ctx=256, n_gpu_layer=99)
    _require_async(text)

    # two generations that do not fit the pool together: one is parked for real while the forced parks keep copies in flight. Without a context shift they stop at the pool, so only the parked and transferring states are pinned here.
    n_predict = 900
    done = []

    def _run():
        done.extend(_complete_all_raw(n_predict, (_PROMPT_A, _PROMPT_B)))

    t = threading.Thread(target=_run)
    t.start()

    seen_parked = False
    seen_transferring = False
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not (seen_parked and seen_transferring):
            res = server.make_request("GET", "/slots")
            assert res.status_code == 200
            for slot in res.body:
                parked = slot["is_preempted"]
                transferring = slot["is_transferring"]
                assert not (parked and transferring), slot
                if transferring:
                    seen_transferring = True
                    assert slot["n_prompt_tokens"] > 0, "a slot with a copy in flight still holds its cells"
                seen_parked = seen_parked or parked
    finally:
        t.join(180)

    assert len(done) == 2, done
    for res in done:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] > 0
    assert seen_parked, "no parked slot was ever reported"
    assert seen_transferring, "no slot with a copy in flight was ever reported"


def _shift_completion(n_predict: int):
    """A completion whose context shifts, on a token prompt so its length is exact."""
    return server.make_request("POST", "/completion", data={
        "prompt": [1] + list(range(10, 70)),
        "n_predict": n_predict,
        "n_keep": 16,
        "n_discard": 64,
        "ignore_eos": True,
        "return_tokens": True,
        "cache_prompt": False,
        "temperature": 0.0,
        "seed": 42,
    })


def test_a_park_right_after_a_context_shift_does_not_change_the_output():
    # the shift moves the positions and leaves the K transformation for the next decode, so a park in between used to save the new positions with the old K
    os.environ["LLAMA_ARG_PREEMPT_ASYNC"] = "0"
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "0"
    _start(n_ctx=256, n_batch=32, n_ubatch=32, enable_ctx_shift=True, cache_ram=0)

    n_predict = 320
    reference = _shift_completion(n_predict)
    assert reference.status_code == 200, reference.body
    assert reference.body["timings"]["predicted_n"] == n_predict
    n_prompt = reference.body["timings"]["prompt_n"]
    server.stop()

    # park on the step the shift lands on: the pool holds n_ctx cells, so the first shift is that many tokens in
    os.environ["LLAMA_ARG_PREEMPT_RAM"] = "8192"
    os.environ["LLAMA_SERVER_PREEMPT_EVERY"] = str(256 - n_prompt)
    log = _start()

    parked = _shift_completion(n_predict)
    assert parked.status_code == 200, parked.body
    assert parked.body["timings"]["predicted_n"] == n_predict

    text = log.drain()
    assert "slot context shift" in text
    assert "preempted on request" in text
    assert "resumed after" in text

    first_diff = next((i for i, (a, b) in enumerate(zip(reference.body["tokens"], parked.body["tokens"])) if a != b), None)
    assert first_diff is None, f"the parked run diverged at token {first_diff}"
