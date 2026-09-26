"""Request preemption tests. Use a shared two-slot pool and actual SSE streams."""
import json
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import requests

from utils import ServerPreset


def start(monkeypatch, ctx=1024, every=0):
    monkeypatch.setenv("LLAMA_SERVER_PREEMPT_EVERY", str(every))
    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_ctx = ctx
    server.n_batch = 64
    server.n_ubatch = 64
    server.n_threads = 1
    server.n_gpu_layer = 0  # CPU reference isolates state preservation from CUDA batch/layout rounding.
    server.n_predict = -1
    server.kv_unified = True
    server.server_continuous_batching = True
    server.server_slots = True
    server.server_metrics = True
    server.start()
    return server


def url(server, path):
    return f"http://{server.server_host}:{server.server_port}{path}"


def body(slot=0, n=600, priority=0, prompt="Once upon a time"):
    return dict(prompt=prompt, id_slot=slot, n_predict=n, temperature=0,
                seed=123, ignore_eos=True, cache_prompt=False,
                return_tokens=True, priority=priority)


def metrics(server):
    response = requests.get(url(server, "/metrics"), timeout=10)
    response.raise_for_status()
    return {line.split()[0]: float(line.split()[1])
            for line in response.text.splitlines()
            if line.startswith("llamacpp:") and "{" not in line}


def slots(server):
    response = requests.get(url(server, "/slots"), timeout=10)
    response.raise_for_status()
    return response.json()


def stream(server, payload, chunks):
    with requests.post(url(server, "/completion"), json={**payload, "stream": True},
                       stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if line.startswith(b"data: "):
                item = json.loads(line[6:])
                assert "error" not in item, item
                chunks.append(item)
    assert chunks[-1]["stop"]
    return "".join(item.get("content", "") for item in chunks)


def until(predicate, timeout=20):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("condition not reached before timeout")


def generated(chunks):
    return sum(len(item.get("tokens", [])) for item in chunks if not item.get("stop"))


@pytest.mark.parametrize("prompt", ["Once upon a time", "The little dog went"])
def test_forced_identity(monkeypatch, prompt):
    outputs = []
    for every in (0, 30):
        server = start(monkeypatch, every=every)
        response = requests.post(url(server, "/completion"), json=body(n=240, prompt=prompt), timeout=60)
        response.raise_for_status()
        outputs.append(response.json())
        assert metrics(server)["llamacpp:preemptions_total"] == (7 if every else 0)
        assert metrics(server)["llamacpp:preempt_ram_bytes"] == 0
        server.stop()
    assert outputs[0]["tokens"] == outputs[1]["tokens"]
    assert outputs[0]["content"].encode() == outputs[1]["content"].encode()


def test_pool_pressure_and_priority(monkeypatch):
    server = start(monkeypatch)
    chunks = [[], []]
    observed = []
    with ThreadPoolExecutor(2) as pool:
        # Give the low-priority request a lead so it is the longest. It must
        # still yield when it is the sole member of the lowest-priority tier.
        low = pool.submit(stream, server, body(1, priority=0), chunks[1])
        until(lambda: generated(chunks[1]) >= 30)
        high = pool.submit(stream, server, body(0, priority=10), chunks[0])
        while not (high.done() and low.done()):
            observed.extend(slots(server))
            time.sleep(0.005)
        high.result()
        low.result()
    # Each request is < 1024 tokens; together they exceed the shared pool.
    assert all(generated(items) == 600 for items in chunks)
    parked = [s for s in observed if s["is_preempted"]]
    assert parked
    assert all(s["priority"] == 0 and s["id"] == 1 for s in parked)
    assert metrics(server)["llamacpp:preemptions_total"] > 0
    assert metrics(server)["llamacpp:requests_preempted"] == 0


def test_explicit_round_trip(monkeypatch):
    server = start(monkeypatch, ctx=4096)
    payload = body(n=1800)
    reference = requests.post(url(server, "/completion"), json=payload, timeout=60).json()
    chunks = []
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(stream, server, payload, chunks)
        until(lambda: generated(chunks) >= 100)
        response = requests.post(url(server, "/slots/0?action=park"), timeout=10)
        response.raise_for_status()
        parked = response.json()
        assert parked["is_preempted"] and parked["preempt_ram_bytes"] > 0
        assert parked["preempt_manual"]
        # The client can drain already-sent chunks, but the server generates no more.
        before = slots(server)[0]["next_token"][0]["n_decoded"]
        other = requests.post(url(server, "/completion"), json=body(1, n=100), timeout=60)
        other.raise_for_status()
        assert other.json()["tokens_predicted"] == 100
        after = slots(server)[0]
        assert after["is_preempted"]
        assert after["next_token"][0]["n_decoded"] == before
        assert metrics(server)["llamacpp:requests_preempted"] == 1
        response = requests.post(url(server, "/slots/0?action=unpark"), timeout=10)
        response.raise_for_status()
        assert not response.json()["preempt_manual"]
        content = future.result(timeout=60)
    assert content.encode() == reference["content"].encode()
    tokens = [token for item in chunks if not item.get("stop") for token in item.get("tokens", [])]
    assert tokens == reference["tokens"]
    assert metrics(server)["llamacpp:preempt_ram_bytes"] == 0


def test_metrics_exposed(monkeypatch):
    server = start(monkeypatch)
    text = requests.get(url(server, "/metrics"), timeout=10).text
    for name, kind in [("preemptions_total", "counter"), ("preempt_restores_total", "counter"),
                       ("preempt_blocked_total", "counter"), ("preempt_copy_seconds_total", "counter"),
                       ("preempt_ram_bytes", "gauge"), ("requests_preempted", "gauge")]:
        assert f"# TYPE llamacpp:{name} {kind}" in text
    for slot in slots(server):
        assert slot["priority"] == 0 and not slot["is_preempted"]
        assert slot["n_preempt"] == 0 and slot["preempt_ram_bytes"] == 0


@pytest.mark.parametrize("priority", [1.5, "10", True, 2**31, -(2**31)-1])
def test_priority_validation(monkeypatch, priority):
    server = start(monkeypatch)
    response = requests.post(url(server, "/completion"), json=body(n=1, priority=priority), timeout=10)
    assert response.status_code == 400


def test_cancel_parked_releases_ram(monkeypatch):
    server = start(monkeypatch, ctx=4096)
    response = requests.post(url(server, "/completion"), json={**body(n=1800), "stream": True, "sse_ping_interval": 1},
                             headers={"X-Conversation-Id": "preempt-cancel"},
                             stream=True, timeout=60)
    try:
        until(lambda: slots(server)[0].get("next_token", [{}])[0].get("n_decoded", 0) > 10)
        parked = requests.post(url(server, "/slots/0?action=park"), timeout=10)
        parked.raise_for_status()
        assert parked.json()["preempt_ram_bytes"] > 0
    finally:
        requests.delete(url(server, "/v1/stream?conv_id=preempt-cancel"), timeout=10).raise_for_status()
        response.close()
    until(lambda: not slots(server)[0]["is_processing"])
    assert metrics(server)["llamacpp:preempt_ram_bytes"] == 0
    assert slots(server)[0]["n_prompt_tokens"] == 0


def test_snapshot_ram_bound(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_PREEMPT_RAM", "1")
    server = start(monkeypatch, ctx=2048, every=1700)
    # A 1700-token stories260K snapshot needs > 1 MiB. Refusing the
    # diagnostic round trip must not damage the still-resident request.
    response = requests.post(url(server, "/completion"), json=body(n=1800), timeout=60)
    response.raise_for_status()
    assert response.json()["tokens_predicted"] == 1800
    assert metrics(server)["llamacpp:preemptions_total"] == 0
    assert metrics(server)["llamacpp:preempt_ram_bytes"] == 0
