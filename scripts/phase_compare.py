#!/usr/bin/env python3
"""Side by side per-phase table for two llama-server traces."""
import json, sys, collections, bisect

def load(path):
    ev = []
    for line in open(path):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("ph") == "server":
            ev.append(d)
    return ev

def agg(path):
    ev = load(path)
    iters = sorted([e for e in ev if e["n"] == "iteration"], key=lambda e: e["t0"])
    def kind_of(it):
        n_slots = it.get("n1", 0)
        n_tok = 0
        for e in ev:
            if e["n"] == "submit" and it["t0"] <= e["t0"] <= it["t1"]:
                n_tok = max(n_tok, e.get("n1", 0))
        return "prefill" if n_tok > n_slots else "decode"
    kinds = [kind_of(it) for it in iters]
    bounds = sorted([(it["t0"], it["t1"], k) for it, k in zip(iters, kinds)])
    starts = [b[0] for b in bounds]
    def bucket(t0):
        i = bisect.bisect_right(starts, t0) - 1
        if i < 0: return "other"
        return bounds[i][2] if t0 <= bounds[i][1] else "other"
    out = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    for e in ev:
        a = out[bucket(e["t0"])][e["n"]]
        a[0] += e["t1"] - e["t0"]
        a[1] += 1
    return out, kinds.count("decode"), kinds.count("prefill")

a, ad, ap = agg(sys.argv[1])
b, bd, bp = agg(sys.argv[2])
print("base decode iters %d prefill %d | new decode %d prefill %d" % (ad, ap, bd, bp))
for b_name, na, nb in (("decode", ad, bd), ("prefill", ap, bp)):
    print("\n-- %s iterations, ms per iteration --" % b_name)
    print("%-22s %10s %10s %9s | %10s %10s" % ("span", "base", "new", "delta", "base us/c", "new us/c"))
    names = set(a[b_name]) | set(b[b_name])
    rows = []
    for n in names:
        ta, ca = a[b_name].get(n, [0, 0])
        tb, cb = b[b_name].get(n, [0, 0])
        rows.append((n, ta/1000.0/max(1,na), tb/1000.0/max(1,nb),
                     ta/max(1,ca), tb/max(1,cb), ta))
    for n, pa, pb, ua, ub, srt in sorted(rows, key=lambda r: -r[5]):
        print("%-22s %10.3f %10.3f %+9.3f | %10.1f %10.1f" % (n, pa, pb, pb-pa, ua, ub))
