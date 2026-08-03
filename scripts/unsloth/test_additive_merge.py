#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for additive_merge.py. Run: python3 scripts/unsloth/test_additive_merge.py

Every case builds a real git conflict rather than a hand-written one, so the
markers are exactly what git produces.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "additive_merge.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(repo, *args, **kw):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=repo, capture_output=True, text=True, **kw)


def make_conflict(base_txt, ours_txt, theirs_txt):
    """Build a real git conflict, return (repo, conflicted file path)."""
    d = Path(tempfile.mkdtemp(prefix="am_"))
    git(d, "init", "-q", "-b", "main")
    f = d / "f.c"
    f.write_text(base_txt)
    git(d, "add", "-A"); git(d, "commit", "-qm", "base")
    git(d, "checkout", "-qb", "side")
    f.write_text(theirs_txt)
    git(d, "add", "-A"); git(d, "commit", "-qm", "theirs")
    git(d, "checkout", "-q", "main")
    f.write_text(ours_txt)
    git(d, "add", "-A"); git(d, "commit", "-qm", "ours")
    git(d, "-c", "merge.conflictStyle=diff3", "merge", "side")
    return d, f


def run(repo, *extra):
    rep = repo / "r.json"
    p = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--report", str(rep), *extra],
                       capture_output=True, text=True)
    return p.returncode, json.loads(rep.read_text()) if rep.exists() else {}


# --- 1. pure add/add: the real recurring shape ------------------------------
base = "switch (arch) {\n    case A:\n        break;\n}\n"
ours = "switch (arch) {\n    case A:\n    case LLM_ARCH_INKLING:\n        break;\n}\n"
theirs = "switch (arch) {\n    case A:\n    case LLM_ARCH_DEEPSEEK4:\n        break;\n}\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
txt = f.read_text()
check("add/add resolves", rc == 0 and rep["ok"], rep)
check("add/add unions both labels",
      "LLM_ARCH_DEEPSEEK4" in txt and "LLM_ARCH_INKLING" in txt and "<<<<" not in txt, txt)
check("add/add puts upstream first (matches hand repin)",
      txt.index("DEEPSEEK4") < txt.index("INKLING"), txt)
check("add/add stages the file",
      git(repo, "diff", "--name-only", "--diff-filter=U").stdout.strip() == "")

# --- 2. edit/edit on a shared line: must refuse -----------------------------
base = "if (a == X || a == Y) {\n"
ours = "if (a == X || a == Y || a == KIMI) {\n"
theirs = "if (a == X || a == Y || a == MINIMAX) {\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
check("edit/edit refuses", rc == 1 and not rep["ok"])
check("edit/edit says why", "base is not empty" in (rep["refused"][0]["reason"] if rep["refused"] else ""),
      rep)
check("edit/edit leaves markers in place", "<<<<" in f.read_text())

# --- 3. same line added twice: must refuse, not duplicate -------------------
base = "a\nz\n"
ours = "a\ncase FOO:\n    break;\nz\n"
theirs = "a\ncase FOO:\n    break;\nz\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
check("identical add/add is not a conflict at all", rc == 1 and "no conflicted files" in json.dumps(rep))

base = "a\nz\n"
ours = "a\ncase FOO:\n    break;\nz\n"
theirs = "a\ncase BAR:\n    break;\nz\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
check("overlapping add/add refuses (shared 'break;')",
      rc == 1 and "made twice" in json.dumps(rep), rep)

# --- 4. one file good, one file bad: refuse the whole merge ----------------
d = Path(tempfile.mkdtemp(prefix="am_"))
git(d, "init", "-q", "-b", "main")
(d / "good.c").write_text("x\ny\n")
(d / "bad.c").write_text("if (a || b) {\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "base")
git(d, "checkout", "-qb", "side")
(d / "good.c").write_text("x\ncase UP:\ny\n")
(d / "bad.c").write_text("if (a || b || up) {\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "theirs")
git(d, "checkout", "-q", "main")
(d / "good.c").write_text("x\ncase MINE:\ny\n")
(d / "bad.c").write_text("if (a || b || mine) {\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "ours")
git(d, "-c", "merge.conflictStyle=diff3", "merge", "side")
rc, rep = run(d)
check("mixed: overall refuses", rc == 1 and not rep["ok"])
check("mixed: both files still unmerged in the index",
      {ln.split("\t")[-1] for ln in git(d, "ls-files", "-u").stdout.splitlines()} == {"good.c", "bad.c"},
      git(d, "ls-files", "-u").stdout)
check("mixed: the resolvable file is NOT half-written",
      "<<<<" in (d / "good.c").read_text(), (d / "good.c").read_text())

# --- 5. dry-run writes nothing --------------------------------------------
base = "switch (arch) {\n    case A:\n        break;\n}\n"
ours = "switch (arch) {\n    case A:\n    case MINE:\n        break;\n}\n"
theirs = "switch (arch) {\n    case A:\n    case UP:\n        break;\n}\n"
repo, f = make_conflict(base, ours, theirs)
before = f.read_text()
rc, rep = run(repo, "--dry-run")
check("dry-run reports ok", rc == 0 and rep["ok"])
check("dry-run does not touch the file", f.read_text() == before)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
