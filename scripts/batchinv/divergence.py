#!/usr/bin/env python3
"""Baseline / patched divergence harness: every prompt solo, then all four sharing batches.

Each sequence is compared with its own solo reference, and every response has to carry the
requested number of tokens: a run that stops at EOS proves nothing about a batch it never
shared. `--p0-only` keeps the older single-reference comparison and JSON shape.
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import PROMPTS

# recorded with every run; LLAMA_EXACT_CONCURRENCY inherited from the shell decides whether a run
# labelled as the mode-off reference actually was one, so it is not optional
RECORDED_ENV = ("LLAMA_EXACT_CONCURRENCY", "GGML_CUDA_BATCH_INVARIANT",
                "GGML_CUDA_BATCH_INVARIANT_MAX_COLS", "LLAMA_SERVER_PREEMPT_EVERY",
                "LLAMA_KV_CACHE_DEBUG", "LLAMA_BATCH_DEBUG", "CUDA_VISIBLE_DEVICES")

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


def completion(port, prompt, n_predict, ignore_eos=True):
    # ignore_eos by default: these runs are fixed length, and a short answer would be compared
    # against an equally short reference and reported identical without sharing a single batch
    return post(port, "/completion", {
        "prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "top_k": 1,
        "top_p": 1.0, "min_p": 0.0, "typical_p": 1.0, "seed": 0, "ignore_eos": ignore_eos,
        "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
        "cache_prompt": False, "return_tokens": True, "samplers": ["top_k", "temperature"],
    })


def short_responses(outs, n_predict):
    """Names whose response does not carry exactly the requested number of tokens."""
    return {n: len(o["tokens"]) for n, o in outs.items() if len(o["tokens"]) != n_predict}


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
            # __exit__ is not called when __enter__ raises, and a server that started but never
            # reported healthy would keep the GPU, the port and the log handle
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *a):
        # note: POSIX only; Windows would need CREATE_NEW_PROCESS_GROUP at Popen
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


def run_concurrent(port, names, n_predict, ignore_eos=True):
    barrier = threading.Barrier(len(names))
    lock = threading.Lock()
    out = {}
    errors = []

    def work(name):
        try:
            barrier.wait()
            res = completion(port, PROMPTS[name], n_predict, ignore_eos)
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

    # without this a run where P1..P3 failed and P0 succeeded reads as a clean four-way result
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
    ap.add_argument("--no-ignore-eos", action="store_true",
                    help="let a response stop at EOS; lengths are then not required to match")
    ap.add_argument("--p0-only", action="store_true",
                    help="older comparison and JSON shape: P0 against one reference")
    a = ap.parse_args()

    names = ["P0", "P1", "P2", "P3"]
    ignore_eos = not a.no_ignore_eos

    env_extra = dict(kv.split("=", 1) for kv in a.env)
    server = Server(a.port, a.binary, a.extra, env_extra, a.out + ".server.log", a.spec,
                    kv_unified=not a.no_kv_unified)
    res = {"label": a.label, "spec": a.spec, "n_predict": a.n_predict,
           "env_requested": env_extra, "env": server.env_resolved,
           "model": server.args[2], "args": server.args,
           "extra": a.extra, "binary": a.binary,
           "kv_unified": not a.no_kv_unified,
           "ignore_eos": ignore_eos, "p0_only": a.p0_only,
           "status": "incomplete", "rounds_completed": 0}

    def finish(code):
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(json.dumps({k: v for k, v in res.items() if k != "rounds"}, indent=2), flush=True)
        print(json.dumps(res.get("rounds", []), indent=2), flush=True)
        sys.exit(code)

    with server as s:
        # every prompt gets its own solo reference, captured before any of them share a batch
        solo = {}
        for n in (["P0"] if a.p0_only else names):
            solo[n] = completion(a.port, PROMPTS[n], a.n_predict, ignore_eos)

        ref = {n: o["tokens"] for n, o in solo.items()}
        if a.reference:
            loaded = json.load(open(a.reference))
            # a legacy file holds P0's tokens alone
            ref = {"P0": loaded["tokens"]} if "tokens" in loaded else \
                  {n: loaded[n]["tokens"] for n in loaded}

        res["reference"] = a.reference
        res["solo"] = {n: {"n_tokens": len(o["tokens"]),
                           "tok_per_s": o["timings"]["predicted_per_second"]}
                       for n, o in solo.items()}
        if a.p0_only:
            res["solo"] = res["solo"]["P0"]
        fd_solo = {n: first_diff(ref[n], solo[n]["tokens"]) for n in solo if n in ref}
        res["solo_first_diff"] = fd_solo["P0"] if a.p0_only else fd_solo

        # a reference that is not the requested length cannot certify anything below it
        if ignore_eos:
            bad = dict(short_responses(solo, a.n_predict))
            bad.update({n: len(t) for n, t in ref.items() if len(t) != a.n_predict})
            if bad:
                res["status"] = f"reference is not {a.n_predict} tokens: {bad}"
                print("[fail] " + res["status"], flush=True)
                finish(1)

        # solo repeat, to prove solo itself is stable
        solo2 = completion(a.port, PROMPTS["P0"], a.n_predict, ignore_eos)
        res["solo_repeat_first_diff"] = first_diff(ref["P0"], solo2["tokens"])

        res["rounds"] = []
        for r in range(a.repeats):
            outs, wall = run_concurrent(a.port, names, a.n_predict, ignore_eos)
            agg = sum(outs[n]["timings"]["predicted_per_second"] for n in outs)
            compared = [n for n in names if n in ref]
            fds = {n: first_diff(ref[n], outs[n]["tokens"]) for n in compared}
            row = {"round": r,
                   "first_diff": fds["P0"], "n_tokens": len(outs["P0"]["tokens"]),
                   "identical": all(fds[n] is None for n in compared),
                   "per_seq_first_diff": fds,
                   "per_seq_identical": {n: fds[n] is None for n in compared},
                   "wall_s": wall,
                   "p0_tok_per_s": outs["P0"]["timings"]["predicted_per_second"],
                   "aggregate_tok_per_s": agg,
                   "per_req_n": {n: len(outs[n]["tokens"]) for n in outs},
                   "tokens": {n: outs[n]["tokens"] for n in outs},
                   "p0_tokens": outs["P0"]["tokens"]}
            if a.p0_only:
                # the older shape: P0 against one reference and nothing else
                for k in ("per_seq_first_diff", "per_seq_identical", "tokens"):
                    row.pop(k)

            res["rounds"].append(row)
            print(f"[round {r}] identical={row['identical']} first_diff={fds} "
                  f"wall={wall:.1f}s agg={agg:.1f} tok/s", flush=True)

            short = short_responses(outs, a.n_predict) if ignore_eos else {}
            if short:
                row["status"] = f"round produced {short} of {a.n_predict} tokens"
                res["status"] = row["status"]
                print("[fail] " + row["status"], flush=True)
                finish(1)

            row["status"] = "ok"
            res["rounds_completed"] = r + 1

        with urllib.request.urlopen(f"http://127.0.0.1:{a.port}/metrics") as response:
            res["metrics"] = response.read().decode()
        with open(a.out + (".p0_solo.json" if a.p0_only else ".solo.json"), "w") as f:
            if a.p0_only:
                json.dump({"tokens": ref["P0"], "content": solo["P0"]["content"]}, f)
            else:
                json.dump({n: {"tokens": ref[n], "content": solo[n]["content"]} for n in solo}, f)

    res["status"] = "ok"
    res["identical"] = all(r["identical"] for r in res["rounds"])
    finish(0 if res["identical"] else 2)


if __name__ == "__main__":
    main()
