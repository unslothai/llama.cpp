#!/usr/bin/env python3
"""Per-phase host cost table from one llama-server trace file.

Reads the client jsonl written by GGML_RPC_TRACE and reports, for the "server" phase
events, the total time, the number of occurrences and the mean, split into decode
iterations and prefill iterations. An iteration is a prefill one when the batch it
built was larger than the number of slots that were processing, which is exactly the
case where a prompt was still being consumed.
"""
import json, sys, collections

def load(path):
    ev = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("ph") == "server":
                ev.append(d)
    return ev

def main():
    path = sys.argv[1]
    ev = load(path)
    if not ev:
        print("no server events in %s" % path)
        return
    # iterations, to split prefill from decode
    iters = sorted([e for e in ev if e["n"] == "iteration"], key=lambda e: e["t0"])
    subs  = [e for e in ev if e["n"] != "iteration"]

    # a submit whose n1 (token count) is bigger than the slot count of its iteration
    # means prompt tokens were in the batch
    submits = {}
    for e in ev:
        if e["n"] == "submit":
            submits.setdefault((e["t0"]), e)

    def kind_of(it):
        n_slots = it.get("n1", 0)
        n_tok = 0
        for e in subs:
            if e["n"] == "submit" and it["t0"] <= e["t0"] <= it["t1"]:
                n_tok = max(n_tok, e.get("n1", 0))
        return "prefill" if n_tok > n_slots else "decode"

    kinds = [kind_of(it) for it in iters]
    bounds = [(it["t0"], it["t1"], k) for it, k in zip(iters, kinds)]
    bounds.sort()

    import bisect
    starts = [b[0] for b in bounds]

    def bucket(t0):
        i = bisect.bisect_right(starts, t0) - 1
        if i < 0:
            return "other"
        if t0 <= bounds[i][1]:
            return bounds[i][2]
        return "other"

    agg = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for e in ev:
        b = bucket(e["t0"])
        a = agg[b][e["n"]]
        a[0] += e["t1"] - e["t0"]
        a[1] += 1

    n_dec = kinds.count("decode")
    n_pre = kinds.count("prefill")
    print("\n=== %s ===" % path)
    print("iterations: decode %d, prefill %d" % (n_dec, n_pre))
    for b, n_it in (("decode", n_dec), ("prefill", n_pre), ("other", 1)):
        if b not in agg:
            continue
        print("\n-- %s iterations (n=%d) --" % (b, n_it))
        print("%-22s %12s %9s %10s %12s" % ("span", "total ms", "count", "mean us", "ms/iter"))
        rows = sorted(agg[b].items(), key=lambda kv: -kv[1][0])
        for name, (tot, cnt) in rows:
            print("%-22s %12.1f %9d %10.1f %12.3f" %
                  (name, tot / 1000.0, cnt, tot / cnt, tot / 1000.0 / max(1, n_it)))

    # the longest single span of each name, useful for the prefill outliers
    print("\n-- longest single span per name --")
    longest = {}
    for e in ev:
        d = e["t1"] - e["t0"]
        if d > longest.get(e["n"], (0, None))[0]:
            longest[e["n"]] = (d, e)
    for name, (d, e) in sorted(longest.items(), key=lambda kv: -kv[1][0])[:20]:
        print("%-22s %10.1f ms  n0=%s n1=%s" % (name, d / 1000.0, e.get("n0"), e.get("n1")))

main()
