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
check("add/add puts upstream first (order is arbitrary, this pins it)",
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

# --- 6. both sides append a block, colliding only on `};` -------------------
# The real shape from tools/mtmd/clip-model.h and ggml/src/ggml-backend-meta.cpp:
# two unrelated structs added at the same anchor. Only the closing brace is shared.
base = "struct keep {\n    int a;\n};\n"
ours = "struct keep {\n    int a;\n};\n\nstruct mine {\n    int m;\n};\n"
theirs = "struct keep {\n    int a;\n};\n\nstruct theirs {\n    int t;\n};\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
txt = f.read_text()
check("block append sharing only `};` resolves", rc == 0 and rep["ok"], rep)
check("block append keeps both structs",
      "struct mine" in txt and "struct theirs" in txt and "<<<<" not in txt, txt)
check("block append does not duplicate the shared brace",
      txt.count("struct mine") == 1 and txt.count("struct theirs") == 1, txt)

# --- 7. shared line is boilerplate the file already repeats -----------------
# common/chat.cpp: every init function opens with the same two lines, so both
# sides adding a new one collide on boilerplate, not on one construct twice.
boiler = "    common_chat_params data;\n    auto parser = mk();\n"
base = ("static params init_a() {\n" + boiler + "}\n"
        "static params init_b() {\n" + boiler + "}\n")
ours = base + "static params init_mine() {\n" + boiler + "}\n"
theirs = base + "static params init_theirs() {\n" + boiler + "}\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
txt = f.read_text()
check("shared boilerplate resolves", rc == 0 and rep["ok"], rep)
check("shared boilerplate keeps both functions",
      "init_mine" in txt and "init_theirs" in txt and "<<<<" not in txt, txt)

# --- 8. one construct added twice: must STILL refuse ------------------------
# tools/mtmd/mtmd-image.cpp: upstream landed its own copy of a helper we had
# already written. The signature line appears nowhere else, so it is distinctive.
base = "static void other() {\n}\n"
ours = base + "static bool resize_pillow(\n        const img & i,\n        bool use_lanczos) {\n    return a;\n}\n"
theirs = base + "static bool resize_pillow(\n        const img & i,\n        bool use_lanczos) {\n    return b;\n}\n"
repo, f = make_conflict(base, ours, theirs)
rc, rep = run(repo)
check("one construct added twice still refuses", rc == 1 and not rep["ok"], rep)
check("refusal names the distinctive line",
      "use_lanczos" in json.dumps(rep) or "resize_pillow" in json.dumps(rep), rep)
check("refused file keeps its markers", "<<<<" in f.read_text())

# --- 9. RPC version triple: three-way merge per field -----------------------
HDR = "#define RPC_PROTO_MAJOR_VERSION    5\n#define RPC_PROTO_MINOR_VERSION    {}\n#define RPC_PROTO_PATCH_VERSION    {}\n"


def make_rpc_conflict(ours_v, theirs_v, base_v=(0, 0), ops=3):
    """Same as make_conflict but the file is ggml-rpc.h next to a real ggml.h."""
    d = Path(tempfile.mkdtemp(prefix="am_rpc_"))
    git(d, "init", "-q", "-b", "main")
    inc = d / "ggml" / "include"
    inc.mkdir(parents=True)
    members = "".join(f"        GGML_OP_{i},\n" for i in range(ops))
    (inc / "ggml.h").write_text("    enum ggml_op {\n" + members + "        GGML_OP_COUNT,\n    };\n")
    f = inc / "ggml-rpc.h"
    tail = f'\nstatic_assert(GGML_OP_COUNT == {ops}, "x");\n'
    f.write_text(HDR.format(*base_v) + tail)
    git(d, "add", "-A"); git(d, "commit", "-qm", "base")
    git(d, "checkout", "-qb", "side")
    f.write_text(HDR.format(*theirs_v) + tail)
    git(d, "add", "-A"); git(d, "commit", "-qm", "theirs")
    git(d, "checkout", "-q", "main")
    f.write_text(HDR.format(*ours_v) + tail)
    git(d, "add", "-A"); git(d, "commit", "-qm", "ours")
    git(d, "-c", "merge.conflictStyle=diff3", "merge", "side")
    return d, f


# upstream bumped minor, we bumped patch: each field moved on one side only
repo, f = make_rpc_conflict(ours_v=(0, 1), theirs_v=(1, 0))
rc, rep = run(repo)
txt = f.read_text()
check("rpc version triple resolves", rc == 0 and rep["ok"], rep)
check("rpc takes minor from upstream and patch from us",
      "MINOR_VERSION    1" in txt and "PATCH_VERSION    1" in txt and "<<<<" not in txt, txt)
check("rpc keeps the major untouched", "MAJOR_VERSION    5" in txt, txt)

# both sides moved the same field: nothing to merge, must refuse
repo, f = make_rpc_conflict(ours_v=(0, 4), theirs_v=(0, 2))
rc, rep = run(repo)
check("rpc refuses when one field moved on both sides", rc == 1 and not rep["ok"], rep)
check("rpc refusal says which field", "PATCH_VERSION" in json.dumps(rep), rep)

# a stale op count is the silent wire mismatch this header exists to prevent
repo, f = make_rpc_conflict(ours_v=(0, 1), theirs_v=(1, 0), ops=3)
(repo / "ggml" / "include" / "ggml.h").write_text(
    "    enum ggml_op {\n" + "".join(f"        GGML_OP_{i},\n" for i in range(4))
    + "        GGML_OP_COUNT,\n    };\n")
rc, rep = run(repo)
check("rpc refuses when static_assert disagrees with the merged enum",
      rc == 1 and not rep["ok"], rep)
check("rpc says the count is stale", "GGML_OP_COUNT" in json.dumps(rep), rep)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
