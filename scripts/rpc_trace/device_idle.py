#!/usr/bin/env python3
"""Where the idle time of ONE device goes, on a two node layer split.

merge.py answers "where does a decode step go", per pipeline group. This answers a different
question, and it is the one that decides what a fix would have to look like:

  the bottleneck GPU is busy B percent of the window. The other 100-B percent is idle. Is it
  idle while the OTHER device is computing, or while NEITHER device is computing?

Those have different fixes. Idle while the other device computes is a scheduling problem: the
pipeline groups are not offset, or a group is waiting at a synchronisation point, so the work
that should have covered this device was somewhere else. Idle while neither device computes is a
host problem: something on the CPU is between the two devices and nothing can run anywhere.

The split is reported separately for the prefill phase and the decode phase of the cell, because
a serving cell spends its first seconds prefilling every slot and a device that is idle there is
idle for a completely different reason than one that is idle in steady decode.

Usage:  device_idle.py client.jsonl peer.jsonl [--out report.txt]
"""

import argparse
import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge import load, union, union_len, clip, gaps, Index          # noqa: E402


def phase_windows(client):
    """(prefill, decode) lists of (t0,t1) iteration windows, per group.

    An iteration is prefill when it submitted more tokens than it had slots processing: in decode
    every slot contributes exactly one token, so n_tokens <= n_slots. The classification is per
    iteration and not per time range, because the groups do not enter decode together.
    """
    subs = defaultdict(list)                      # grp -> [(t0,t1,n_tokens)]
    for e in client.events:
        if e.get("ph") == "server" and e.get("n") == "submit":
            subs[e.get("grp", 0)].append((e["t0"], e["t1"], e.get("n1", 0)))
    for g in subs:
        subs[g].sort()

    pre, dec = defaultdict(list), defaultdict(list)
    for e in client.events:
        if e.get("ph") != "server" or e.get("n") != "iteration":
            continue
        g = e.get("grp", 0)
        n_slots = e.get("n1", 0)
        toks = [n for a, b, n in subs.get(g, []) if a >= e["t0"] and b <= e["t1"]]
        (pre if (toks and max(toks) > max(n_slots, 1)) else dec)[g].append((e["t0"], e["t1"]))
    return pre, dec


def report(files, client, out):
    servers = [f for f in files if f.role == "rpc-server"]

    def shift(f, e):
        return (e["t0"] - f.offset_us, e["t1"] - f.offset_us)

    busy = {
        "local": union([shift(client, e) for e in client.events if e.get("ph") == "gpu"]),
        "peer":  union([shift(f, e) for f in servers for e in f.events if e.get("ph") == "gpu"]),
    }
    if not busy["local"] or not busy["peer"]:
        out.write("one of the two devices has no GPU spans; nothing to decompose\n")
        return

    pre, dec = phase_windows(client)
    groups = sorted(set(list(pre.keys()) + list(dec.keys())))

    all_iters = [iv for g in groups for iv in pre[g] + dec[g]]
    w0 = min(a for a, _ in all_iters)
    w1 = max(b for _, b in all_iters)

    # Host phases, for attributing the stretches in which neither device computes.
    # server/iteration is the PARENT span of every other server phase, so it is not a candidate:
    # it would win every attribution and say nothing. What it does not cover inside an iteration
    # is reported as "inside an iteration, untraced", which is a real answer and a different one.
    host = {}
    for e in client.events:
        if e.get("ph") in ("server", "sched", "llama") and e.get("n") != "iteration":
            host.setdefault("%s/%s" % (e.get("ph"), e.get("n")), []).append((e["t0"], e["t1"], e))
    host = {k: Index(v) for k, v in host.items()}
    iter_ix = Index([(e["t0"], e["t1"], e) for e in client.events
                     if e.get("ph") == "server" and e.get("n") == "iteration"])

    # phase boundary: prefill of the cell is everything up to the last prefill iteration
    pre_all = union([iv for g in groups for iv in pre[g]])
    dec_all = union([iv for g in groups for iv in dec[g]])
    t_pre_end = max((b for _, b in pre_all), default=w0)

    phases = [("whole window", w0, w1),
              ("prefill phase", w0, t_pre_end),
              ("decode phase", t_pre_end, w1)]

    out.write("window %.3f s, %d groups, prefill ends %.3f s in "
              "(%.1f%% of the window)\n" % (
                  (w1 - w0) / 1e6, len(groups), (t_pre_end - w0) / 1e6,
                  100.0 * (t_pre_end - w0) / max(w1 - w0, 1)))
    out.write("prefill iterations %d, decode iterations %d\n\n" % (
        sum(len(pre[g]) for g in groups), sum(len(dec[g]) for g in groups)))

    for pname, p0, p1 in phases:
        span = p1 - p0
        if span <= 0:
            continue
        out.write("=== %s: %.3f s\n" % (pname, span / 1e6))
        for dev in ("local", "peer"):
            other = "peer" if dev == "local" else "local"
            b = clip(busy[dev], p0, p1)
            ob = Index([(a, c, None) for a, c in busy[other]])
            idle = gaps(b, p0, p1)
            t_busy = union_len(b)
            t_idle = union_len(idle)

            covered = 0.0
            n_sched, n_host = 0, 0
            len_sched, len_host = [], []
            attr = defaultdict(float)
            by_grp = defaultdict(float)
            for g0, g1 in idle:
                c = ob.covered(g0, g1)
                covered += c
                if c > 0.5 * (g1 - g0):
                    n_sched += 1
                    len_sched.append(g1 - g0)
                else:
                    n_host += 1
                    len_host.append(g1 - g0)
                # the stretches of this hole in which the other device is ALSO idle
                dead = gaps(clip(busy[other], g0, g1), g0, g1)
                for d0, d1 in dead:
                    best, bestc = None, 0.0
                    for name, ix in host.items():
                        cv = ix.covered(d0, d1)
                        if cv > bestc:
                            bestc, best = cv, name
                    if best is not None:
                        attr[best] += bestc
                    rest = (d1 - d0) - bestc
                    if rest > 0:
                        inside = iter_ix.covered(d0, d1)
                        attr["inside an iteration, untraced"] += min(rest, inside)
                        attr["between iterations"] += max(rest - inside, 0.0)
                    for g in groups:
                        if union_len(clip(dec[g] + pre[g], d0, d1)) > 0.5 * (d1 - d0):
                            by_grp["group %d live" % g] += d1 - d0

            out.write("  %-5s busy %6.2f%%  idle %6.2f%%  "
                      "(idle while %s computes %6.2f%%, idle with neither computing %6.2f%%)\n"
                      % (dev, 100.0 * t_busy / span, 100.0 * t_idle / span, other,
                         100.0 * covered / span, 100.0 * (t_idle - covered) / span))
            out.write("        %d idle stretches: %d mostly-covered (median %.2f ms), "
                      "%d mostly-dead (median %.2f ms)\n"
                      % (len(idle), n_sched, _med(len_sched) / 1000.0,
                         n_host, _med(len_host) / 1000.0))
            top = sorted(attr.items(), key=lambda kv: -kv[1])[:6]
            if top and top[0][1] > 0:
                out.write("        neither computing, by host phase: %s\n"
                          % ", ".join("%s %.2f%%" % (k, 100.0 * v / span)
                                      for k, v in top if v > 0))
            n_steps = sum(len(clip(dec[g] + pre[g], p0, p1)) for g in groups)
            if n_steps:
                out.write("        per group-step in this phase: idle %.2f ms "
                          "(%.2f ms while %s computes, %.2f ms with neither)\n"
                          % (t_idle / n_steps / 1000.0, covered / n_steps / 1000.0,
                             other, (t_idle - covered) / n_steps / 1000.0))
        out.write("\n")

    # how the two groups sit relative to each other: the overlap of their GPU demand
    out.write("=== group offset in the decode phase\n")
    for g in groups:
        d = clip(dec[g], t_pre_end, w1)
        out.write("  group %d: %d decode iterations, median %.1f ms, "
                  "covering %.1f%% of the decode phase\n"
                  % (g, len(d), _med([b - a for a, b in d]) / 1000.0,
                     100.0 * union_len(d) / max(w1 - t_pre_end, 1)))
    if len(groups) == 2:
        a = clip(dec[groups[0]], t_pre_end, w1)
        b = clip(dec[groups[1]], t_pre_end, w1)
        both = union_len(a) + union_len(b) - union_len(a + b)
        out.write("  the two groups are inside an iteration at the same time for %.1f%% "
                  "of the decode phase\n" % (100.0 * both / max(w1 - t_pre_end, 1)))


def _med(xs):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[len(xs) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--out")
    args = ap.parse_args()
    files, client = load(args.traces)
    out = open(args.out, "w") if args.out else sys.stdout
    try:
        if client is None:
            out.write("no client trace\n")
        else:
            report(files, client, out)
    finally:
        if args.out:
            out.close()


if __name__ == "__main__":
    main()
