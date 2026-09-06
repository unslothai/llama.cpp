#!/usr/bin/env python3
"""Merge the event traces of the nodes of an RPC layer split.

Each process writes JSON lines (see ggml/include/ggml-trace.h): one header, then one object per
event. The client also writes the result of a four timestamp exchange with every peer it connects
to, which gives the offset between the two monotonic clocks. This tool puts every file on the
client's time line and produces

  * a Chrome trace (chrome://tracing, or https://ui.perfetto.dev) with one row per node, thread
    and pipeline group, plus a row per GPU carrying the CUDA event timings, and
  * a text summary per decode step: local compute, transfer, peer compute, logits return,
    sampling, and the idle fraction of each GPU.

Usage:
    merge.py client.jsonl peer.jsonl --chrome trace.json --summary summary.txt
"""

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------- reading


class TraceFile:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.offsets = []          # clock offset records written by the client
        self.events = []
        self.offset_us = 0         # this file's clock minus the client's clock

        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # a trace of a process that was killed can end in a partial line
                    continue
                if "header" in rec:
                    self.header = rec
                elif "clock_offset" in rec:
                    self.offsets.append(rec)
                elif "ph" in rec and "t0" in rec and "t1" in rec:
                    self.events.append(rec)

    @property
    def role(self):
        return self.header.get("role", "unknown")

    @property
    def host(self):
        return self.header.get("host", "")

    def label(self):
        return "%s %s" % (self.host or "node", self.role)


def load(paths):
    files = [TraceFile(p) for p in paths]

    clients = [f for f in files if f.role in ("llama-server", "rpc-client")]
    servers = [f for f in files if f.role == "rpc-server"]

    if not clients:
        # a trace of the peer alone is still useful, it just has no common time line
        return files, None

    client = clients[0]

    # map each peer endpoint to its measured offset; the exchange is repeated per connection, the
    # median is used so a single delayed reply does not move the alignment
    by_host = defaultdict(list)
    for rec in client.offsets:
        host = rec.get("peer", "").split(":")[0]
        by_host[host].append(rec.get("offset_us", 0))

    for f in servers:
        cand = None
        for host, values in by_host.items():
            if host == f.host or f.host.startswith(host) or host.startswith(f.host):
                cand = values
                break
        if cand is None and len(by_host) == 1:
            cand = list(by_host.values())[0]
        if cand is None:
            sys.stderr.write(
                "warning: no clock offset for %s, its events are left on their own clock\n" % f.path)
            continue
        cand = sorted(cand)
        f.offset_us = cand[len(cand) // 2]

    return files, client


# ---------------------------------------------------------------------------- intervals


def union_len(intervals):
    """total length covered by a list of (t0, t1)"""
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur0, cur1 = intervals[0]
    for t0, t1 in intervals[1:]:
        if t0 > cur1:
            total += cur1 - cur0
            cur0, cur1 = t0, t1
        else:
            cur1 = max(cur1, t1)
    total += cur1 - cur0
    return total


def union(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for t0, t1 in intervals[1:]:
        if t0 > out[-1][1]:
            out.append([t0, t1])
        else:
            out[-1][1] = max(out[-1][1], t1)
    return [(a, b) for a, b in out]


def clip(intervals, w0, w1):
    out = []
    for t0, t1 in intervals:
        a, b = max(t0, w0), min(t1, w1)
        if b > a:
            out.append((a, b))
    return out


def gaps(intervals, w0, w1):
    """the holes of a union inside [w0, w1]"""
    out = []
    cur = w0
    for t0, t1 in union(clip(intervals, w0, w1)):
        if t0 > cur:
            out.append((cur, t0))
        cur = max(cur, t1)
    if cur < w1:
        out.append((cur, w1))
    return out


# ---------------------------------------------------------------------------- chrome trace


def chrome_trace(files, client):
    out = []
    t_base = None
    for f in files:
        for e in f.events:
            t = e["t0"] - f.offset_us
            t_base = t if t_base is None else min(t_base, t)
    if t_base is None:
        t_base = 0

    for pid, f in enumerate(files, start=1):
        out.append({"ph": "M", "pid": pid, "tid": 0, "name": "process_name",
                    "args": {"name": f.label()}})
        out.append({"ph": "M", "pid": pid, "tid": 0, "name": "process_sort_index",
                    "args": {"sort_index": pid}})

        gpu_rows = {}
        named = set()

        for e in f.events:
            t0 = e["t0"] - f.offset_us - t_base
            t1 = e["t1"] - f.offset_us - t_base
            cat = e.get("ph", "")
            tid = e.get("tid", 0)

            if cat == "gpu":
                # one row per device, well away from the host thread ids
                key = e.get("n", "gpu")
                if key not in gpu_rows:
                    gpu_rows[key] = 10000 + len(gpu_rows)
                    out.append({"ph": "M", "pid": pid, "tid": gpu_rows[key], "name": "thread_name",
                                "args": {"name": "GPU %s" % key}})
                tid = gpu_rows[key]
            elif tid not in named:
                named.add(tid)
                label = "thread %d" % tid
                if e.get("grp") is not None:
                    label = "group %d thread %d" % (e["grp"], tid)
                out.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                            "args": {"name": label}})

            args = {k: v for k, v in e.items() if k not in ("ph", "n", "t0", "t1", "tid")}
            out.append({"ph": "X", "pid": pid, "tid": tid, "cat": cat, "name": e.get("n", "?"),
                        "ts": t0, "dur": max(t1 - t0, 0), "args": args})

            # the phases inside one RPC command, as slices nested in the command
            for name, a, b in sub_phases(e):
                a -= f.offset_us + t_base
                b -= f.offset_us + t_base
                if b > a:
                    out.append({"ph": "X", "pid": pid, "tid": tid, "cat": cat + ".phase",
                                "name": name, "ts": a, "dur": b - a})

    # nested slices must not start before their parent
    out.sort(key=lambda e: (e.get("ts", -1), -e.get("dur", 0)))
    return {"traceEvents": out, "displayTimeUnit": "ms"}


def sub_phases(e):
    cat = e.get("ph", "")
    if cat == "rpc.client":
        t_send0 = e.get("t_send0", 0)
        t_send1 = e.get("t_send1", 0)
        res = []
        if t_send0:
            res.append(("queue", e["t0"], t_send0))
            res.append(("send", t_send0, t_send1))
        if e.get("reply"):
            res.append(("wait reply", t_send1, e.get("t_wait", t_send1)))
            res.append(("read reply", e.get("t_wait", t_send1), e.get("t_recv1", t_send1)))
        return [r for r in res if r[2] > r[1]]
    if cat == "rpc.server":
        res = [("receive", e.get("t_recv0", 0), e.get("t_recv1", 0)),
               ("execute", e.get("t_exec0", 0), e.get("t_exec1", 0)),
               ("reply", e.get("t_send0", 0), e.get("t_send1", 0))]
        return [r for r in res if r[1] and r[2] > r[1]]
    return []


# ---------------------------------------------------------------------------- summary

LOGITS_MIN_BYTES = 64 * 1024


class Index:
    """intervals sorted by start, with a bisect lookup of the ones overlapping a window"""

    def __init__(self, items):
        # items: list of (t0, t1, payload)
        self.items = sorted(items, key=lambda x: x[0])
        self.starts = [x[0] for x in self.items]
        self.max_dur = max((x[1] - x[0] for x in self.items), default=0)

    def overlapping(self, w0, w1):
        lo = bisect.bisect_left(self.starts, w0 - self.max_dur)
        out = []
        for t0, t1, payload in self.items[lo:]:
            if t0 >= w1:
                break
            if t1 > w0:
                out.append((t0, t1, payload))
        return out

    def covered(self, w0, w1):
        return union_len(clip([(a, b) for a, b, _ in self.overlapping(w0, w1)], w0, w1))


def summarize(files, client, out):
    servers = [f for f in files if f.role == "rpc-server"]

    def shift(f, e):
        return (e["t0"] - f.offset_us, e["t1"] - f.offset_us)

    # GPU busy intervals of each node, on the client's clock
    gpu_local = Index([shift(client, e) + (e,) for e in client.events if e.get("ph") == "gpu"])
    gpu_peer = Index([shift(f, e) + (e,) for f in servers for e in f.events if e.get("ph") == "gpu"])

    iters = [e for e in client.events if e.get("ph") == "server" and e.get("n") == "iteration"]
    if not iters:
        out.write("no llama-server iterations in the trace\n")
        return

    w0 = min(e["t0"] for e in iters)
    w1 = max(e["t1"] for e in iters)
    span = max(w1 - w0, 1)

    out.write("trace window %.3f s, %d decode steps\n" % (span / 1e6, len(iters)))
    for f in files:
        out.write("  %-28s %-14s offset %+.3f ms, %d events\n"
                  % (os.path.basename(f.path), f.label(), f.offset_us / 1000.0, len(f.events)))
    out.write("\n")

    groups = sorted({e.get("grp", 0) for e in iters})

    # one index per (category, name, group), so a step is a bisect and not a scan of the file
    idx = {}
    for e in client.events:
        key = (e.get("ph"), e.get("n"), e.get("grp", 0))
        idx.setdefault(key, []).append((e["t0"], e["t1"], e))
    for key in list(idx):
        idx[key] = Index(idx[key])

    empty = Index([])

    def get(cat, name, grp):
        return idx.get((cat, name, grp), empty)

    # the RPC commands of one group, all command types together
    rpc_by_grp = {}
    for grp in groups:
        rpc_by_grp[grp] = Index([(e["t0"], e["t1"], e) for e in client.events
                                 if e.get("ph") == "rpc.client" and e.get("grp", 0) == grp])

    rows = []
    for grp in groups:
        steps = [e for e in iters if e.get("grp", 0) == grp]
        acc = defaultdict(float)
        n = 0
        wire_out = 0
        wire_in = 0
        worst = (0, "", 0)
        host = [get(cat, name, grp) for cat, name in
                (("server", "batch_build"), ("server", "submit"), ("server", "synchronize"),
                 ("server", "post_decode"), ("server", "sampling"), ("server", "result_send"),
                 ("llama", "graph_compute"), ("sched", "split"), ("sched", "copy_stage"))]

        for it in steps:
            t0, t1 = it["t0"], it["t1"]
            if t1 <= t0:
                continue
            n += 1

            cmds = [e for _, _, e in rpc_by_grp[grp].overlapping(t0, t1)]

            send = [(e.get("t_send0", e["t0"]), e.get("t_send1", e["t1"])) for e in cmds]
            recv = [(e.get("t_wait", 0), e.get("t_recv1", 0)) for e in cmds if e.get("reply")]
            recv = [r for r in recv if r[0] and r[1] > r[0]]
            logits = [(e["t0"], e.get("t_recv1", e["t1"])) for e in cmds
                      if e.get("n") == "GET_TENSOR" and e.get("bytes_in", 0) >= LOGITS_MIN_BYTES]

            wire_out += sum(e.get("bytes_out", 0) for e in cmds)
            wire_in += sum(e.get("bytes_in", 0) for e in cmds)

            local_iv = clip([(a, b) for a, b, _ in gpu_local.overlapping(t0, t1)], t0, t1)
            peer_iv = clip([(a, b) for a, b, _ in gpu_peer.overlapping(t0, t1)], t0, t1)

            acc["step"] += t1 - t0
            acc["build"] += get("server", "batch_build", grp).covered(t0, t1)
            acc["submit"] += get("server", "submit", grp).covered(t0, t1)
            acc["sync"] += get("server", "synchronize", grp).covered(t0, t1)
            acc["post"] += get("server", "post_decode", grp).covered(t0, t1)
            acc["sampling"] += get("server", "sampling", grp).covered(t0, t1)
            acc["send"] += get("server", "result_send", grp).covered(t0, t1)
            acc["local_gpu"] += union_len(local_iv)
            acc["peer_gpu"] += union_len(peer_iv)
            acc["transfer"] += union_len(clip(send + recv, t0, t1))
            acc["logits"] += union_len(clip(logits, t0, t1))
            acc["stage"] += get("sched", "copy_stage", grp).covered(t0, t1)

            # the longest stretch of the step in which neither GPU was busy
            hole = gaps(local_iv + peer_iv, t0, t1)
            acc["idle_both"] += union_len(hole)
            for g0, g1 in hole:
                if g1 - g0 > worst[0]:
                    worst = (g1 - g0, name_gap(host, g0, g1), g0)

        if n == 0:
            continue
        rows.append((grp, n, acc, wire_out, wire_in, worst))

    hdr = ("group  steps  step_ms  build  submit   sync   post  sampl   send | "
           "localGPU  peerGPU  transfer  logits  stage | idle_both")
    out.write(hdr + "\n")
    out.write("-" * len(hdr) + "\n")
    for grp, n, acc, wo, wi, worst in rows:
        def ms(k):
            return acc[k] / n / 1000.0
        out.write("%5d  %5d  %7.1f %6.1f %7.1f %6.1f %6.1f %6.1f %6.1f | "
                  "%8.1f %8.1f %9.1f %7.1f %6.1f | %9.1f\n"
                  % (grp, n, ms("step"), ms("build"), ms("submit"), ms("sync"), ms("post"),
                     ms("sampling"), ms("send"), ms("local_gpu"), ms("peer_gpu"),
                     ms("transfer"), ms("logits"), ms("stage"), ms("idle_both")))
    out.write("\n")

    for grp, n, acc, wo, wi, worst in rows:
        out.write("group %d: %.1f kB out and %.1f kB in per step over RPC; "
                  "biggest idle gap %.1f ms in %s\n"
                  % (grp, wo / n / 1024.0, wi / n / 1024.0, worst[0] / 1000.0, worst[1]))

    busy_local = gpu_local.covered(w0, w1)
    busy_peer = gpu_peer.covered(w0, w1)
    out.write("\nover the whole window: local GPU busy %.1f%% (idle %.1f%%), "
              "peer GPU busy %.1f%% (idle %.1f%%)\n"
              % (100.0 * busy_local / span, 100.0 * (1 - busy_local / span),
                 100.0 * busy_peer / span, 100.0 * (1 - busy_peer / span)))
    if not gpu_local.items:
        out.write("note: no GPU spans on the client, so the local GPU row is empty "
                  "(CPU backend, or a build without the CUDA timing hook)\n")
    if not gpu_peer.items:
        out.write("note: no GPU spans from the peer, so the peer GPU row is empty\n")


def name_gap(indexes, g0, g1):
    """what the host was doing during a stretch in which no GPU was busy"""
    best = None
    best_cov = 0
    for ix in indexes:
        for a, b, e in ix.overlapping(g0, g1):
            cov = min(b, g1) - max(a, g0)
            if cov > best_cov:
                best_cov = cov
                best = e
    if best is None:
        return "nothing traced"
    return "%s/%s (%.0f%% of the gap)" % (best.get("ph"), best.get("n"),
                                          100.0 * best_cov / max(g1 - g0, 1))


# ---------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="trace files, client first")
    ap.add_argument("--chrome", help="write a Chrome trace here")
    ap.add_argument("--summary", help="write the text summary here (default: stdout)")
    args = ap.parse_args()

    files, client = load(args.traces)

    if args.chrome:
        with open(args.chrome, "w") as f:
            json.dump(chrome_trace(files, client), f)
        sys.stderr.write("wrote %s (%d events)\n"
                         % (args.chrome, sum(len(t.events) for t in files)))

    out = open(args.summary, "w") if args.summary else sys.stdout
    try:
        if client is None:
            out.write("no client trace given, nothing to align against\n")
        else:
            summarize(files, client, out)
    finally:
        if args.summary:
            out.close()


if __name__ == "__main__":
    main()
