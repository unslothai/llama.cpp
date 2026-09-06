#!/usr/bin/env python3
"""Baseline / patched divergence harness: solo P0 vs P0 sharing batches with P1..P3."""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS

# Environment recorded with every run. LLAMA_EXACT_CONCURRENCY inherited from the shell is what
# decides whether a run labelled as the mode-off reference actually was one, so it is not optional.
RECORDED_ENV = ("LLAMA_EXACT_CONCURRENCY", "GGML_CUDA_BATCH_INVARIANT",
                "GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "LLAMA_SERVER_PREEMPT_EVERY",
                "CUDA_VISIBLE_DEVICES")

MODEL_REL = "models/Qwen3.5-4B-MTP-GGUF/Qwen3.5-4B-UD-Q4_K_XL.gguf"


def model_path():
    """Resolved when the server args are built, so --help works without the variable set."""
    ws = os.environ.get("UNSLOTH_WORKSPACE")
    if not ws:
        raise RuntimeError("UNSLOTH_WORKSPACE is not set; it must point at the workspace holding "
                           + MODEL_REL)
    return os.path.join(ws, MODEL_REL)


def post(port, path, payload, timeout=1800):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(port, path, timeout=10):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode())


def completion(port, prompt, n_predict):
    return post(port, "/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "seed": 0,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "cache_prompt": False, "return_tokens": True, "samplers": ["top_k", "temperature"],
    })


class Server:
    def __init__(self, port, binary, extra, env_extra, log_path, spec, kv_unified=True):
        self.port, self.log_path = port, log_path
        self.args = [binary, "-m", model_path(), "--port", str(port), "--host", "127.0.0.1",
                     "--parallel", "4", "-c", "8192",
                     "--flash-attn", "on", "--metrics", "-ngl", "99", "--no-warmup",
                     "--seed", "0", "--spec-type", spec]
        if kv_unified:
            self.args += ["--kv-unified"]
        if spec == "draft-mtp":
            self.args += ["--spec-draft-n-max", "2"]
        self.args += extra
        self.env = dict(os.environ)
        self.env["CUDA_VISIBLE_DEVICES"] = "3"
        self.env.update(env_extra)
        # what the server will actually see, not what this run meant to set
        self.env_resolved = {k: self.env[k] for k in RECORDED_ENV if k in self.env}
        self.p = None
        self.fh = None

    def __enter__(self):
        self.fh = open(self.log_path, "ab")
        self.fh.write(("\n=== " + " ".join(self.args) + "\n=== env " +
                       json.dumps(self.env_resolved) + "\n").encode())
        self.fh.flush()
        self.p = subprocess.Popen(self.args, stdout=self.fh, stderr=subprocess.STDOUT,
                                  env=self.env, start_new_session=True)
        print(f"[server] pid={self.p.pid} port={self.port} log={self.log_path}", flush=True)
        try:
            deadline = time.time() + 600
            while time.time() < deadline:
                if self.p.poll() is not None:
                    raise RuntimeError(f"server died rc={self.p.returncode}, see {self.log_path}")
                try:
                    if get(self.port, "/health").get("status") == "ok":
                        print("[server] ready", flush=True)
                        return self
                except Exception:
                    time.sleep(1.0)
            raise RuntimeError("server did not become healthy")
        except BaseException:
            # __exit__ is not called when __enter__ raises, so a server that started but never
            # reported healthy would keep the GPU, the port and the log handle
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *a):
        # note: POSIX only. On Windows this needs CREATE_NEW_PROCESS_GROUP at Popen and
        # terminate()/kill() here; the runs this harness backs are Linux only.
        if self.p is not None:
            print(f"[server] stopping pid={self.p.pid}", flush=True)
            try:
                os.killpg(os.getpgid(self.p.pid), signal.SIGTERM)
                self.p.wait(timeout=60)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.p.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    self.p.wait(timeout=60)
                except Exception:
                    pass
            self.p = None
        if self.fh is not None:
            self.fh.close()
            self.fh = None


def run_concurrent(port, names, n_predict):
    barrier = threading.Barrier(len(names))
    lock = threading.Lock()
    out = {}
    errors = []

    def work(name):
        try:
            barrier.wait()
            res = completion(port, PROMPTS[name], n_predict)
        except BaseException as e:
            with lock:
                errors.append((name, e))
            # release the others rather than let them block on a barrier that will never fill
            barrier.abort()
            return
        with lock:
            out[name] = res

    ts = [threading.Thread(target=work, args=(n,)) for n in names]
    t0 = time.time()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.time() - t0

    # a thread exception used to only print a traceback, so a run where P1..P3 failed and P0
    # succeeded was still reported as a clean four-way concurrency result
    if errors:
        raise RuntimeError("concurrent requests failed: " +
                           "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in errors))
    missing = set(names) - set(out)
    if missing:
        raise RuntimeError(f"concurrent requests produced no result for {sorted(missing)}")

    return out, wall


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference")
    ap.add_argument("--label", required=True)
    ap.add_argument("--port", type=int, default=9601)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--spec", default="none")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-kv-unified", action="store_true")
    a = ap.parse_args()

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    server = Server(a.port, a.binary, a.extra, env_extra, a.out + ".server.log", a.spec,
                    kv_unified=not a.no_kv_unified)
    res = {"label": a.label, "spec": a.spec, "n_predict": a.n_predict,
           "env_requested": env_extra, "env": server.env_resolved,
           "model": server.args[2], "args": server.args,
           "extra": a.extra, "binary": a.binary,
           "kv_unified": not a.no_kv_unified}

    with server as s:
        solo = completion(a.port, PROMPTS["P0"], a.n_predict)
        ref = json.load(open(a.reference))["tokens"] if a.reference else solo["tokens"]
        res["solo_first_diff"] = first_diff(ref, solo["tokens"])
        res["reference"] = a.reference
        res["solo"] = {"n_tokens": len(ref), "tok_per_s": solo["timings"]["predicted_per_second"],
                       "text_sha": None}
        # solo repeat, to prove solo itself is stable
        solo2 = completion(a.port, PROMPTS["P0"], a.n_predict)
        res["solo_repeat_first_diff"] = first_diff(ref, solo2["tokens"])
        res["rounds"] = []
        for r in range(a.repeats):
            outs, wall = run_concurrent(a.port, ["P0", "P1", "P2", "P3"], a.n_predict)
            p0 = outs["P0"]["tokens"]
            fd = first_diff(ref, p0)
            agg = sum(outs[n]["timings"]["predicted_per_second"] for n in outs)
            row = {"round": r, "first_diff": fd, "n_tokens": len(p0),
                   "identical": fd is None, "wall_s": wall,
                   "p0_tok_per_s": outs["P0"]["timings"]["predicted_per_second"],
                   "aggregate_tok_per_s": agg,
                   "per_req_n": {n: len(outs[n]["tokens"]) for n in outs}, "p0_tokens": p0}
            res["rounds"].append(row)
            print(f"[round {r}] first_diff={fd} identical={fd is None} wall={wall:.1f}s agg={agg:.1f} tok/s", flush=True)
        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/metrics") as response:
            res["metrics"] = response.read().decode()
        with open(a.out + ".p0_solo.json", "w") as f:
            json.dump({"tokens": ref, "content": solo["content"]}, f)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "rounds"}, indent=2), flush=True)
    print(json.dumps(res["rounds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
