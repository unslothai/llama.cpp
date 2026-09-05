"""Unified-pool parking integration tests, using only stories260K."""
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from utils import ServerPreset, TMP_DIR, download_file


@pytest.fixture(scope="session", autouse=True)
def load_server_presets():
    # This module needs only the tiny model, not every server preset.
    return download_file("https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories260K.gguf",
                         os.path.join(TMP_DIR, "stories260K.gguf"))


def start_server(model, *, ctx=512):
    server = ServerPreset.tinyllama2()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model
    server.n_ctx = ctx
    server.n_slots = 2
    server.n_batch = 32
    server.n_ubatch = 32
    server.n_threads = 2
    server.kv_unified = True
    server.server_continuous_batching = True
    server.server_metrics = True
    server.server_slots = True
    server.cache_ram = 0
    server.start()
    return server


def metrics(server):
    text = server.make_request("GET", "/metrics").body
    return {line.split()[0].removeprefix("llamacpp:"): float(line.split()[1])
            for line in text.splitlines() if line and not line.startswith("#")}


def request(prompt="Once upon a time", n=160):
    return dict(prompt=prompt, n_predict=n, temperature=0, seed=1234,
                ignore_eos=True, cache_prompt=False, return_tokens=True)


def test_forced_knob_identity(load_server_presets, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PREEMPT_EVERY", "0")
    server = start_server(load_server_presets)
    baseline = server.make_request("POST", "/completion", request())
    assert baseline.status_code == 200
    server.stop()
    monkeypatch.setenv("LLAMA_SERVER_PREEMPT_EVERY", "32")
    server = start_server(load_server_presets)
    parked = server.make_request("POST", "/completion", request())
    assert parked.status_code == 200
    assert parked.body["content"].encode() == baseline.body["content"].encode()
    assert parked.body["tokens"] == baseline.body["tokens"]
    m = metrics(server)
    assert m["preempt_forced_total"] == 4
    assert m["preempt_restores_total"] == 4
    assert m["preempt_parked"] == m["preempt_ram_bytes"] == 0


@pytest.mark.parametrize("high,low", [(94, 80), (97, 90), (99, 98)])
def test_pressure_finishes_and_honours_band(load_server_presets, monkeypatch, high, low):
    monkeypatch.setenv("LLAMA_ARG_PREEMPT_HIGH", str(high))
    monkeypatch.setenv("LLAMA_ARG_PREEMPT_LOW", str(low))
    server = start_server(load_server_presets)
    # Each request fits by itself; the live total cannot fit in 512 cells.
    bodies = [request("Once " * 100 + ending, n=300)
              for ending in ("a fox went home", "a cat found a tree")]
    samples = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(server.make_request, "POST", "/completion", body) for body in bodies]
        while not all(f.done() for f in futures):
            samples.extend(server.make_request("GET", "/slots").body)
            time.sleep(0.005)
        results = [f.result() for f in futures]
    for res in results:
        assert res.status_code == 200, res.body
        assert res.body["tokens_predicted"] == 300
        assert not res.body["truncated"]
    m = metrics(server)
    assert m["preemptions_total"] >= 1
    assert m["preemptions_total"] == m["preempt_restores_total"]
    assert m["preempt_restore_max_cells"] <= m["preempt_low_cells"]
    assert m["preempt_high_cells"] == 512 * high // 100
    assert m["preempt_low_cells"] == 512 * low // 100
    assert any(s["is_parked"] for s in samples)
    assert all(s["is_processing"] for s in samples if s["is_parked"])
    assert all(s["preempt_restore_projected_cells"] <= s["preempt_low_cells"] for s in samples)
    assert m["preempt_parked"] == m["preempt_ram_bytes"] == 0
    assert m["preempt_restore_failures_total"] == 0


def test_metrics_exposed(load_server_presets):
    server = start_server(load_server_presets)
    m = metrics(server)
    for name in ("preemptions_total", "preempt_restores_total", "preempt_forced_total",
                 "preempt_ram_denied_total", "preempt_restore_failures_total",
                 "preempt_unused_cells_total", "preempt_copy_seconds_total",
                 "preempt_parked", "preempt_ram_bytes", "preempt_resident_cells",
                 "preempt_projected_cells", "preempt_restore_max_cells"):
        assert name in m
        assert m[name] == 0
    slots = server.make_request("GET", "/slots").body
    assert len(slots) == 2
    assert all(not s["is_parked"] for s in slots)


def test_zero_ram_declines_forced_parks(load_server_presets, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_PREEMPT_RAM", "0")
    monkeypatch.setenv("LLAMA_SERVER_PREEMPT_EVERY", "32")
    server = start_server(load_server_presets)
    res = server.make_request("POST", "/completion", request())
    assert res.status_code == 200
    m = metrics(server)
    assert m["preemptions_total"] == 0
    assert m["preempt_ram_denied_total"] > 0


def test_long_prefills_can_wait(load_server_presets):
    server = start_server(load_server_presets)
    bodies = [request("Once " * 420 + ending, n=32) for ending in ("a cat", "a dog")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda body: server.make_request("POST", "/completion", body), bodies))
    assert all(r.status_code == 200 for r in results), [r.body for r in results]
    assert all(r.body["tokens_predicted"] == 32 for r in results)
    m = metrics(server)
    assert m["preemptions_total"] >= 1
    assert m["preemptions_total"] == m["preempt_restores_total"]


def test_shared_prompt_is_not_parked(load_server_presets, monkeypatch):
    monkeypatch.setenv("LLAMA_SERVER_PREEMPT_EVERY", "16")
    server = start_server(load_server_presets)
    body = request(n=64)
    body["n_cmpl"] = 2
    res = server.make_request("POST", "/completion", body)
    assert res.status_code == 200
    assert len(res.body) == 2
    assert all(item["tokens_predicted"] == 64 for item in res.body)
    assert metrics(server)["preemptions_total"] == 0


def test_cancel_parked_stream_releases_ram(load_server_presets):
    import requests
    server = start_server(load_server_presets)
    url = f"http://{server.server_host}:{server.server_port}"
    responses = []
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            def start(i):
                body = request("Once " * 100 + ("a dog" if i else "a cat"), n=350)
                body.update(stream=True, id_slot=i)
                return requests.post(url + "/completion", json=body, stream=True, timeout=20)
            responses = list(pool.map(start, range(2)))
            deadline = time.monotonic() + 10
            parked = None
            while time.monotonic() < deadline:
                slots = server.make_request("GET", "/slots").body
                parked = next((s for s in slots if s["is_parked"]), None)
                if parked:
                    break
                time.sleep(.002)
            assert parked is not None
            assert parked["preempt_bytes"] > 0
            responses[parked["id"]].close()
            while time.monotonic() < deadline:
                m = metrics(server)
                if m["preempt_ram_bytes"] == 0:
                    break
                time.sleep(.01)
            assert m["preempt_ram_bytes"] == 0
            assert m["preempt_parked"] == 0
    finally:
        for response in responses:
            response.close()
