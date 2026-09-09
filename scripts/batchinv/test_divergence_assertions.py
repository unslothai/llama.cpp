#!/usr/bin/env python3
"""What divergence.py has to reject: a fake server, no model, no GPU.

Run it as `python3 scripts/batchinv/test_divergence_assertions.py`.
"""
import contextlib, io, json, os, sys, tempfile, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("UNSLOTH_WORKSPACE", "/nonexistent")  # no model is loaded here
import divergence


class FakeMetrics:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return b"fake_metrics 1"


divergence.urllib.request.urlopen = lambda *a, **kw: FakeMetrics()


def call_main(argv):
    sys.argv = argv
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            divergence.main()
    except SystemExit as e:
        return e.code, buf.getvalue()
    return 0, buf.getvalue()


class FakeServer:
    def __init__(self, *a, **kw):
        self.env_resolved = {"LLAMA_EXACT_CONCURRENCY": "1"}
        self.args = ["bin", "-m", "fake-model"]

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def run(n_tokens_by_name, argv, label):
    divergence.Server = FakeServer
    divergence.completion = lambda port, prompt, n, ignore_eos=True: {
        "tokens": [1] * n_tokens_by_name[_name_of(prompt)],
        "content": "x",
        "timings": {"predicted_per_second": 1.0},
    }
    divergence.run_concurrent = lambda port, names, n, ignore_eos=True: (
        {nm: {"tokens": [1] * n_tokens_by_name[nm], "content": "x",
              "timings": {"predicted_per_second": 1.0}} for nm in names}, 1.0)
    code, _ = call_main(argv)
    out = json.load(open(argv[argv.index("--out") + 1]))
    print(f"[{label}] exit={code} status={out['status']!r} "
          f"rounds_completed={out['rounds_completed']}")
    return code, out


def _name_of(prompt):
    from prompts import PROMPTS
    for k, v in PROMPTS.items():
        if v == prompt:
            return k
    raise KeyError("unknown prompt")


if __name__ == "__main__":
    out_dir = tempfile.mkdtemp(prefix="divergence-assertions-")
    base = ["divergence.py", "--label", "selfcheck", "--binary", "bin", "--n-predict", "512",
            "--repeats", "1", "--out", os.path.join(out_dir, "selfcheck.json")]
    # the shape the audit reported: one token each, compared with one-token references
    code, out = run({"P0": 1, "P1": 1, "P2": 1, "P3": 1}, base, "all one token")
    assert code == 1 and "reference is not 512 tokens" in out["status"], out["status"]

    # references are the right length, one neighbour comes back short from the concurrent round
    divergence.Server = FakeServer
    divergence.completion = lambda port, prompt, n, ignore_eos=True: {
        "tokens": [1] * n, "content": "x", "timings": {"predicted_per_second": 1.0}}
    divergence.run_concurrent = lambda port, names, n, ignore_eos=True: (
        {nm: {"tokens": [1] * (7 if nm == "P2" else n), "content": "x",
              "timings": {"predicted_per_second": 1.0}} for nm in names}, 1.0)
    code, _ = call_main(base)
    out = json.load(open(base[base.index("--out") + 1]))
    print(f"[short neighbour] exit={code} status={out['status']!r} "
          f"round_identical={out['rounds'][0]['identical']}")
    assert code == 1 and "P2" in out["status"], out["status"]

    # a neighbour that diverges while P0 matches
    divergence.completion = lambda port, prompt, n, ignore_eos=True: {
        "tokens": [1] * n, "content": "x", "timings": {"predicted_per_second": 1.0}}
    divergence.run_concurrent = lambda port, names, n, ignore_eos=True: (
        {nm: {"tokens": ([1] * (n - 1) + [9]) if nm == "P3" else [1] * n, "content": "x",
              "timings": {"predicted_per_second": 1.0}} for nm in names}, 1.0)
    code, _ = call_main(base)
    out = json.load(open(base[base.index("--out") + 1]))
    r = out["rounds"][0]
    print(f"[P3 diverges] exit={code} identical={r['identical']} "
          f"per_seq={r['per_seq_identical']} first_diff={r['per_seq_first_diff']}")
    assert code == 2 and r["identical"] is False and r["per_seq_identical"]["P3"] is False

    # the same divergence under the legacy shape: P0 alone still reads as identical, which is why
    # it is behind a flag now
    code, _ = call_main(base + ["--p0-only"])
    out = json.load(open(base[base.index("--out") + 1]))
    r = out["rounds"][0]
    print(f"[legacy p0-only] exit={code} identical={r['identical']} keys_new={sorted(k for k in r if k.startswith('per_seq'))} "
          f"solo_first_diff={out['solo_first_diff']}")
    assert code == 0 and r["identical"] is True and not [k for k in r if k.startswith("per_seq")]
    print("all divergence.py assertion checks passed")
