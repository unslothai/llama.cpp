#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for slice_objects.sh, slice_costs.sh and wait_for_jobs.sh. Run: python3 scripts/unsloth/test_slicing.py

ninja and gh are stand-in scripts on PATH; everything else (bash, awk, sort, xargs) is the real tool.
"""
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def stub(dirpath, name, body):
    p = Path(dirpath) / name
    p.write_text("#!/usr/bin/env bash\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)


def run(args, env, stdin=None):
    return subprocess.run(["bash", *map(str, args)], env=env, capture_output=True, text=True, input=stdin)


# `ninja -t commands all` as gcc, nvcc and MSVC print it, including Windows backslash paths.
COMMANDS = "\n".join(
    [f"/usr/bin/c++ -O3 -o common/CMakeFiles/common.dir/f{i}.cpp.o -c /s/f{i}.cpp" for i in range(20)]
    + [f'"C:\\nvcc.exe" -x cu -c C:\\s\\k{i}.cu -o ggml\\src\\ggml-cuda\\CMakeFiles\\ggml-cuda.dir\\k{i}.cu.obj' for i in range(9)]
    + [f"cl.exe /nologo /Foggml\\src\\CMakeFiles\\ggml-cpu.dir\\c{i}.c.obj /FS -c C:\\s\\c{i}.c" for i in range(6)]
    + ["/usr/bin/c++ -shared -o bin/libllama.so a.o b.o"]
)
EXPECTED = sorted({f"common/CMakeFiles/common.dir/f{i}.cpp.o" for i in range(20)}
                  | {f"ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/k{i}.cu.obj" for i in range(9)}
                  | {f"ggml/src/CMakeFiles/ggml-cpu.dir/c{i}.c.obj" for i in range(6)})

with tempfile.TemporaryDirectory(prefix="slice_") as t:
    t = Path(t)
    (t / "bin").mkdir()
    (t / "commands.txt").write_text(COMMANDS + "\n")
    stub(t / "bin", "ninja", f'cat "{t}/commands.txt"\n')
    env = dict(os.environ, PATH=f"{t / 'bin'}:{os.environ['PATH']}")
    so, sc = HERE / "slice_objects.sh", HERE / "slice_costs.sh"

    # Round-robin: disjoint, complete, deterministic, forward slashes.
    for n in (1, 2, 3, 4):
        parts = [run([so, t, k, n], env).stdout.split() for k in range(n)]
        flat = [o for p in parts for o in p]
        check(f"round-robin N={n}: complete and disjoint", sorted(flat) == EXPECTED, f"{len(flat)} vs {len(EXPECTED)}")
    check("no backslashes survive", all("\\" not in o for o in run([so, t, 0, 1], env).stdout.split()))
    check("the link command is not an object", "bin/libllama.so" not in run([so, t, 0, 1], env).stdout)
    check("deterministic", run([so, t, 1, 3], env).stdout == run([so, t, 1, 3], env).stdout)
    r = run([so, t, 3, 3], env)
    check("slice outside 0..N-1 refused", r.returncode == 2, r.stderr)
    r = run([so, t, "x", 3], env)
    check("non-integer slice refused", r.returncode == 2, r.stderr)
    (t / "empty.txt").write_text("")
    stub(t / "bin", "ninja", f'cat "{t}/empty.txt"\n')
    r = run([so, t, 0, 2], env)
    check("no objects refused (never an empty slice list)", r.returncode == 1, r.stderr)
    stub(t / "bin", "ninja", f'cat "{t}/commands.txt"\n')

    # xargs -d '\n' (as the workflows read slice lists) keeps every path intact.
    lst = run([so, t, 0, 1], env).stdout
    echoed = subprocess.run(["xargs", "-d", "\n", "printf", "%s\n"], input=lst, capture_output=True, text=True).stdout
    check("xargs -d newline round-trips the slice list", echoed.split() == lst.split())

    # Cost-balanced dealing: one heavy object per slice, the light ones spread, estimates equal.
    costs = t / "costs.tsv"
    heavy = set(run([so, t, 0, 3], env).stdout.split()[:3])  # round-robin puts all three in slice 0
    weights = {o: (60000 if o in heavy else 1000) for o in EXPECTED}
    costs.write_text("".join(f"{w}\t{o}\n" for o, w in sorted(weights.items())))
    parts = [run([so, t, k, 3, costs], env).stdout.split() for k in range(3)]
    check("costed N=3: complete and disjoint", sorted(o for p in parts for o in p) == EXPECTED)
    loads = [sum(weights[o] for o in p) for p in parts]
    check("costed N=3: loads balanced", max(loads) - min(loads) <= 1000, str(loads))
    check("costed N=3: one heavy object per slice",
          all(sum(weights[o] == 60000 for o in p) == 1 for p in parts), str(parts))
    rr = [run([so, t, k, 3], env).stdout.split() for k in range(3)]
    check("round-robin is less balanced here (the case costs fix)",
          max(sum(weights[o] for o in p) for p in rr) > max(loads))
    partial = t / "partial.tsv"
    partial.write_text(f"90000\t{EXPECTED[0]}\n")
    parts = [run([so, t, k, 2, partial], env).stdout.split() for k in range(2)]
    check("objects without a cost still dealt (complete)", sorted(o for p in parts for o in p) == EXPECTED)
    r = run([so, t, 0, 2, t / "missing.tsv"], env)
    check("missing costs file falls back to round-robin", r.returncode == 0 and "round-robin" in r.stderr
          and r.stdout == run([so, t, 0, 2], env).stdout)

    # slice_costs.sh export: last record per object wins, backslashes normalised, non-objects dropped.
    b = t / "build"
    b.mkdir()
    (b / ".ninja_log").write_text("# ninja log v7\n"
                                  "0\t5000\t1\tggml\\src\\a.cpp.obj\th\n"
                                  "0\t100\t1\tbin\\llama.dll\th\n"
                                  "10\t7010\t2\tggml\\src\\a.cpp.obj\th\n"
                                  "0\t250\t1\tcommon/b.cpp.o\th\n")
    r = run([sc, "export", b, t / "out.tsv"], env)
    check("export: last record wins, slashes normalised, objects only",
          (t / "out.tsv").read_text() == "250\tcommon/b.cpp.o\n7000\tggml/src/a.cpp.obj\n", (t / "out.tsv").read_text())
    r = run([sc, "export", t / "nobuild", t / "none.tsv"], env)
    check("export without a log writes an empty file, exit 0", r.returncode == 0 and (t / "none.tsv").read_text() == "")

    # slice_costs.sh merge: a ccache hit never replaces a real compile time; objects not built drop out.
    c = t / "c.tsv"
    c.write_text("60000\theavy.o\n40000\tgone.o\n5000\tlight.o\n")
    (t / "leg.tsv").write_text("120\theavy.o\n150\tlight.o\n90\tnew.o\n")
    (t / "warm.tsv").write_text("9000\tlight.o\n")
    run([sc, "merge", c, t / "leg.tsv", t / "warm.tsv"], env)
    got = dict(reversed(line.split("\t")) for line in c.read_text().splitlines())
    check("merge: hit timing keeps the old cost", got.get("heavy.o") == "60000", str(got))
    check("merge: a real compile replaces the cost", got.get("light.o") == "9000", str(got))
    check("merge: a new object takes its only timing", got.get("new.o") == "90", str(got))
    check("merge: objects absent from the timings keep their cost", got.get("gone.o") == "40000", str(got))
    empty = t / "empty_timings.tsv"
    empty.write_text("")
    before = c.read_text()
    run([sc, "merge", c, empty], env)
    check("merge: a failed build (no timings) leaves the costs intact", c.read_text() == before)
    fresh = t / "fresh.tsv"
    run([sc, "merge", fresh, t / "leg.tsv"], env)
    check("merge into a missing costs file", fresh.exists() and len(fresh.read_text().splitlines()) == 3)

    # wait_for_jobs.sh against a stand-in `gh api` (one TSV record per job, as --jq prints).
    wj = HERE / "wait_for_jobs.sh"
    jobs = t / "jobs.tsv"
    stub(t / "bin", "gh", f'[ -f "{t}/fail" ] && exit 1; cat "{jobs}"\n')
    wenv = dict(env, GITHUB_REPOSITORY="o/r", GITHUB_RUN_ID="1", GH_TOKEN="x")
    jobs.write_text("CUDA / warm linux-x64/p 1/3\tcompleted\tsuccess\n"
                    "CUDA / warm linux-x64/p 2/3\tcompleted\tfailure\n"
                    "CUDA / x64/p\tin_progress\tnone\n")
    r = run([wj, "--match", " / warm linux-x64/p ", "--expect", 2, "--poll", 1, "--out", t / "o.tsv"], wenv)
    check("wait: all matching completed -> 0", r.returncode == 0, r.stdout + r.stderr)
    check("wait: --out lists the matching jobs", (t / "o.tsv").read_text().count("\n") == 2)
    r = run([wj, "--exclude", "CUDA / x64/p", "--poll", 1], wenv)
    check("wait: --exclude drops the caller", r.returncode == 0, r.stdout)
    r = run([wj, "--match", " / warm linux-x64/p ", "--expect", 3, "--deadline-min", 0, "--poll", 1], wenv)
    check("wait: fewer records than --expect -> deadline exit 3", r.returncode == 3, r.stdout)
    r = run([wj, "--deadline-min", 0, "--poll", 1], wenv)
    check("wait: a pending job at the deadline -> 3", r.returncode == 3, r.stdout)
    r = run([wj, "--require", "CUDA / warm linux-x64/p 1/3", "--match", " / warm ", "--poll", 1], wenv)
    check("wait: --require met -> 0", r.returncode == 0, r.stdout)
    r = run([wj, "--require", "Assemble", "--match", " / warm ", "--deadline-min", 0, "--poll", 1], wenv)
    check("wait: all completed but a required job absent -> deadline exit 3", r.returncode == 3, r.stdout)
    (t / "fail").write_text("")
    r = run([wj, "--poll", 0], wenv)
    check("wait: 10 API failures in a row -> 4", r.returncode == 4, r.stdout)
    r = run([wj, "--expect", "x"], wenv)
    check("wait: bad integer argument -> 2", r.returncode == 2, r.stderr)

print(f"\n{len(FAILS)} failure(s)" if FAILS else "\nall passed")
sys.exit(1 if FAILS else 0)
