#!/usr/bin/env python3
"""Cost of the knob: solo tok/s and four-chat aggregate tok/s, knob off and on, back to back."""
import argparse, json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from divergence import Server, completion, run_concurrent
from prompts import PROMPTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", required=True)
    ap.add_argument("--spec", default="none")
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--port", type=int, default=9602)
    ap.add_argument("--modes", default="0,1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    modes = a.modes.split(",")
    rows = []
    for pair in range(a.pairs):
        for mode in modes:
            env = {"LLAMA_EXACT_CONCURRENCY": mode, "GGML_CUDA_BATCH_INVARIANT": "0" if mode == "0" else "2"}
            with Server(a.port, a.binary, [], env, a.out + ".server.log", a.spec) as s:
                completion(a.port, PROMPTS["P0"], 32)  # warm
                solo = completion(a.port, PROMPTS["P0"], a.n_predict)
                outs, wall = run_concurrent(a.port, ["P0", "P1", "P2", "P3"], a.n_predict)
                row = {
                    "pair": pair, "mode": mode, "spec": a.spec,
                    "solo_tok_per_s": solo["timings"]["predicted_per_second"],
                    "solo_prompt_tok_per_s": solo["timings"]["prompt_per_second"],
                    "four_aggregate_tok_per_s": sum(o["timings"]["predicted_per_second"] for o in outs.values()),
                    "four_wall_s": wall,
                    "four_total_tokens": sum(len(o["tokens"]) for o in outs.values()),
                }
                row["four_wall_tok_per_s"] = row["four_total_tokens"] / wall
                rows.append(row)
                print(json.dumps(row), flush=True)
    with open(a.out, "w") as f:
        json.dump(rows, f, indent=2)

    print("\n=== summary ===", flush=True)
    for mode in modes:
        rs = [r for r in rows if r["mode"] == mode]
        for k in ("solo_tok_per_s", "four_aggregate_tok_per_s", "four_wall_tok_per_s", "solo_prompt_tok_per_s"):
            vals = sorted(r[k] for r in rs)
            print(f"mode={mode} {k}: median {vals[len(vals)//2]:.1f}  values {[round(v,1) for v in vals]}", flush=True)


if __name__ == "__main__":
    main()
