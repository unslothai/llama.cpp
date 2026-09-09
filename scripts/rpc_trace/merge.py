#!/usr/bin/env python3
"""Merge the event traces of an RPC layer split onto the client's time line, into a Chrome trace
and a per decode step summary: merge.py client.jsonl peer.jsonl --chrome t.json --summary s.txt
"""

import argparse
import bisect
import json
import os
import sys
from collections import defaultdict

class TraceFile:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.offsets = []
        self.events = []
        self.offset_us = 0         # this file's clock minus the client's clock
        self._syncs = None

        with open(path, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # a killed process can leave a partial last line
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

    def sync_spans(self):
        """(t0, t1) of every TRACE_SYNC this peer served, on the peer's own clock"""
        if self._syncs is None:
            self._syncs = [(e.get("t_recv0") or e["t0"], e.get("t_send1") or e["t1"])
                           for e in self.events
                           if e.get("ph") == "rpc.server" and e.get("n") == "TRACE_SYNC"]
        return self._syncs


def host_like(a, b):
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


def endpoint_host(endpoint):
    """host part of "host:port", "[::1]:port" or a bare host"""
    if endpoint.startswith("["):
        return endpoint[1:endpoint.index("]")] if "]" in endpoint else endpoint[1:]
    head, sep, tail = endpoint.rpartition(":")
    return head if sep and tail.isdigit() else endpoint


def served_the_sync(f, rec):
    """did peer `f` answer this clock exchange?

    The client stores the peer's own t2 (reply built) and t3 (reply sent) inside the record, and
    the peer traced the very same TRACE_SYNC command on that same clock, so the record belongs to
    the peer whose TRACE_SYNC span brackets [t2, t3]. No host name is involved.
    """
    t2, t3 = rec.get("t2"), rec.get("t3")
    if t2 is None or t3 is None:
        return False
    return any(t0 <= t2 and t3 <= t1 for t0, t1 in f.sync_spans())


def load(paths):
    files = [TraceFile(p) for p in paths]

    clients = [f for f in files if f.role in ("llama-server", "rpc-client")]
    servers = [f for f in files if f.role == "rpc-server"]

    if not clients:
        # the peer alone is still useful, it just has no common time line
        return files, None

    client = clients[0]

    # keyed by the whole endpoint: two peers on one host differ only in the port, and a host name
    # is not an identity at all once the peers are addressed by IP or by a DNS alias
    by_ep = defaultdict(list)
    for rec in client.offsets:
        by_ep[rec.get("peer", "")].append(rec)

    taken = {}                 # id(file) -> offset samples
    left = dict(by_ep)

    # 1. the peer's own record of the clock exchange, which is unambiguous when it is there
    for ep, recs in list(left.items()):
        owners = [f for f in servers if any(served_the_sync(f, r) for r in recs)]
        if len(owners) == 1:
            taken.setdefault(id(owners[0]), []).extend(r.get("offset_us", 0) for r in recs)
            del left[ep]

    # 2. host name, only when the header carries one and it picks out a single endpoint. An empty
    #    host (Windows, where the tracer writes no host at all) must never prefix-match.
    for f in servers:
        if id(f) in taken or not f.host:
            continue
        hits = [ep for ep in left
                if endpoint_host(ep) and host_like(endpoint_host(ep), f.host)]
        rivals = [g for g in servers
                  if g is not f and id(g) not in taken and g.host == f.host]
        if len(hits) == 1 and not rivals:
            taken[id(f)] = [r.get("offset_us", 0) for r in left.pop(hits[0])]

    # 3. one peer and one endpoint left over: they can only belong together
    rest = [f for f in servers if id(f) not in taken]
    if len(rest) == 1 and len(left) == 1:
        taken[id(rest[0])] = [r.get("offset_us", 0) for r in left.popitem()[1]]

    for f in servers:
        cand = taken.get(id(f))
        if not cand:
            sys.stderr.write(
                "warning: no clock offset for %s, its events are left on their own clock "
                "(no TRACE_SYNC span in it matches a clock exchange, and its header host %r does "
                "not identify it)\n" % (f.path, f.host))
            continue
        # median over the connections, so one delayed reply does not move the alignment
        cand = sorted(cand)
        f.offset_us = cand[len(cand) // 2]

    return files, client


def union_len(intervals):
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
    out = []
    cur = w0
    for t0, t1 in union(clip(intervals, w0, w1)):
        if t0 > cur:
            out.append((cur, t0))
        cur = max(cur, t1)
    if cur < w1:
        out.append((cur, w1))
    return out


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
                # one row per device, away from the host thread ids
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


# only used for traces from a build that did not record the tensor name yet
LOGITS_MIN_BYTES = 64 * 1024

# the graph outputs llama.cpp reads back with GET_TENSOR: logits, embeddings and the norm the
# pooled embedding is taken from. Anything else a layer split copies back (a staged hidden state,
# a KV entry) also runs past 64 KiB, so the size alone says nothing about what the bytes were.
OUTPUT_PREFIX = "result_"


def is_output_tensor(subj):
    if not subj:
        return False
    # the scheduler names a staging copy "<backend>#<tensor>#<n>"
    return any(part.startswith(OUTPUT_PREFIX) for part in subj.split("#"))


def get_tensor_subjects(client):
    """(any GET_TENSOR at all, any of them naming its tensor)"""
    gets = [e for e in client.events
            if e.get("ph") == "rpc.client" and e.get("n") == "GET_TENSOR"]
    return bool(gets), any(e.get("subj") for e in gets)


def output_returns(cmds, by_subject):
    """the GET_TENSOR replies that carried a graph output back to the client"""
    gets = [e for e in cmds if e.get("n") == "GET_TENSOR"]
    if by_subject:
        gets = [e for e in gets if is_output_tensor(e.get("subj"))]
    else:
        gets = [e for e in gets if e.get("bytes_in", 0) >= LOGITS_MIN_BYTES]
    return [(e["t0"], e.get("t_recv1", e["t1"])) for e in gets]


class Index:
    def __init__(self, items):
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

    any_get, by_subject = get_tensor_subjects(client)
    if any_get and not by_subject:
        out.write("note: no GET_TENSOR records a tensor name, so the logits column falls back to "
                  "the %d kB size rule and may also count staged hidden states\n"
                  % (LOGITS_MIN_BYTES // 1024))
    out.write("\n")

    groups = sorted({e.get("grp", 0) for e in iters})

    idx = {}
    for e in client.events:
        key = (e.get("ph"), e.get("n"), e.get("grp", 0))
        idx.setdefault(key, []).append((e["t0"], e["t1"], e))
    for key in list(idx):
        idx[key] = Index(idx[key])

    empty = Index([])

    def get(cat, name, grp):
        return idx.get((cat, name, grp), empty)

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
        idle_by = defaultdict(float)
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
            logits = output_returns(cmds, by_subject)

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

            hole = gaps(local_iv + peer_iv, t0, t1)
            acc["idle_both"] += union_len(hole)
            for g0, g1 in hole:
                cov = attribute_gap(host, g0, g1)
                if g1 - g0 > worst[0]:
                    name, share = max(cov.items(), key=lambda kv: kv[1],
                                      default=("nothing traced", 0))
                    worst = (g1 - g0,
                             "%s (%.0f%% of the gap)" % (name, 100.0 * share / max(g1 - g0, 1)),
                             g0)
                for name, v in cov.items():
                    idle_by[name] += v
                idle_by["unattributed"] += (g1 - g0) - sum(cov.values())

        if n == 0:
            continue
        rows.append((grp, n, acc, wire_out, wire_in, worst, idle_by))

    hdr = ("group  steps  step_ms  build  submit   sync   post  sampl   send | "
           "localGPU  peerGPU  transfer  logits  stage | idle_both")
    out.write(hdr + "\n")
    out.write("-" * len(hdr) + "\n")
    for grp, n, acc, wo, wi, worst, idle_by in rows:
        def ms(k):
            return acc[k] / n / 1000.0
        out.write("%5d  %5d  %7.1f %6.1f %7.1f %6.1f %6.1f %6.1f %6.1f | "
                  "%8.1f %8.1f %9.1f %7.1f %6.1f | %9.1f\n"
                  % (grp, n, ms("step"), ms("build"), ms("submit"), ms("sync"), ms("post"),
                     ms("sampling"), ms("send"), ms("local_gpu"), ms("peer_gpu"),
                     ms("transfer"), ms("logits"), ms("stage"), ms("idle_both")))
    out.write("\n")

    for grp, n, acc, wo, wi, worst, idle_by in rows:
        out.write("group %d: %.1f kB out and %.1f kB in per step over RPC; "
                  "biggest idle gap %.1f ms in %s\n"
                  % (grp, wo / n / 1024.0, wi / n / 1024.0, worst[0] / 1000.0, worst[1]))
        # now that a gap is split over every phase it touches there are more names to show
        top = sorted(idle_by.items(), key=lambda kv: -kv[1])[:6]
        out.write("         idle with neither GPU busy, per step: %s\n"
                  % ", ".join("%s %.1f ms" % (k, v / n / 1000.0) for k, v in top if v > 0))

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


def attribute_gap(indexes, g0, g1):
    """what the host was doing during a stretch in which no GPU was busy

    Returns {"ph/name": microseconds} for every traced phase that covers part of the gap, so a
    gap that runs batch_build -> submit -> synchronize back to back is fully accounted for
    instead of being credited to whichever single phase happened to be the longest.

    Phases nest (a sched/split inside a server/submit) and can straddle each other (an
    llama/graph_compute reaching from a submit into the following synchronize), so a point in
    the gap is charged to exactly one phase: the innermost scope covering it, that is the
    shortest one, ties broken by the later start and then by the name so the split is stable.
    Every microsecond is therefore counted at most once and the total can never exceed g1 - g0.
    """
    spans = []
    for ix in indexes:
        for a, b, e in ix.overlapping(g0, g1):
            a, b = max(a, g0), min(b, g1)
            if b > a:
                spans.append((a, b, "%s/%s" % (e.get("ph"), e.get("n"))))
    cov = defaultdict(float)
    if not spans:
        return cov

    edges = sorted({p for a, b, _ in spans for p in (a, b)})
    for lo, hi in zip(edges, edges[1:]):
        here = [s for s in spans if s[0] <= lo and s[1] >= hi]
        if here:
            inner = min(here, key=lambda s: (s[1] - s[0], -s[0], s[2]))
            cov[inner[2]] += hi - lo
    return cov


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
