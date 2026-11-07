#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for repin.py. Run: python3 scripts/unsloth/test_repin.py

repin_one is run for real against local repositories: the GitHub API answer is stubbed and the
clone/fetch URLs are pointed at a local upstream, everything else (merge, additive_merge.py,
reconcile_op_count.py, commits) is the real code path.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import repin  # noqa: E402

FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(d, *a):
    return subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                          check=True, capture_output=True, text=True).stdout.strip()


def write(d, ops, n, patch):
    (d / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (d / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    enum = "\n".join(f"    GGML_OP_{o}," for o in ["NONE"] + ops)
    (d / "ggml" / "include" / "ggml.h").write_text(f"enum ggml_op {{\n{enum}\n    GGML_OP_COUNT,\n}};\n")
    names = "\n".join(f'    "{o}",' for o in ["NONE"] + ops)
    syms = "\n".join(f'    "{o.lower()}(x)",' for o in ["NONE"] + ops)
    (d / "ggml" / "src" / "ggml.c").write_text(
        f"static const char * GGML_OP_NAME[GGML_OP_COUNT] = {{\n{names}\n}};\n\n"
        f'static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT != {n}");\n\n'
        f"static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {{\n{syms}\n}};\n\n"
        f'static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT != {n}");\n')
    (d / "ggml" / "include" / "ggml-rpc.h").write_text(
        "#define RPC_PROTO_MAJOR_VERSION    3\n#define RPC_PROTO_MINOR_VERSION    0\n"
        f"#define RPC_PROTO_PATCH_VERSION    {patch}\n\n\n\n"
        f'static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");\n')


OPS = [f"OP{i}" for i in range(20)]


def upstream(pin_patch):
    """tag b1 = 21 ops at RPC 3.0.0; branch pin adds PIN_NEW; tag b2 adds UP_NEW mid-enum. Both take
    the asserts 21 -> 22, so merging b2 into the pin leaves 23 ops under stale 22s."""
    d = Path(tempfile.mkdtemp(prefix="repin_up_"))
    git(d, "init", "-q", "-b", "master")
    write(d, OPS, 21, 0)
    git(d, "add", "-A"); git(d, "commit", "-qm", "b1"); git(d, "tag", "b1")
    git(d, "checkout", "-qb", "pin")
    write(d, OPS + ["PIN_NEW"], 22, pin_patch)
    git(d, "commit", "-qam", "pin adds an op")
    pin = git(d, "rev-parse", "HEAD")
    git(d, "checkout", "-q", "master")
    write(d, OPS[:10] + ["UP_NEW"] + OPS[10:], 22, 1)
    git(d, "commit", "-qam", "upstream adds an op"); git(d, "tag", "b2")
    return d, pin


def repin_against(up, pin):
    real_run = repin.run

    def run(args, cwd=None, check=True, quiet=True):
        args = [str(up) if isinstance(a, str) and a.startswith("https://github.com/") else a for a in args]
        return real_run(args, cwd=cwd, check=check, quiet=quiet)

    repin.run = run
    repin.gh_json = lambda path: {"state": "open", "head": {
        "repo": {"full_name": "unslothai/llama.cpp"}, "ref": "pin", "sha": pin}}
    try:
        p = {"url": f"https://github.com/unslothai/llama.cpp/pull/1/commits/{pin}", "required": True,
             "src": "unslothai/llama.cpp", "num": 1, "sha": pin}
        return repin.repin_one(p, "b2", Path(tempfile.mkdtemp(prefix="repin_work_")))
    finally:
        repin.run = real_run


# The repinned commit must be what the nightly would build: asserts at the counted 23 and PATCH
# raised to cover both ops, committed (the pin SHA is what the nightly fetches, not a work tree).
up, pin = upstream(pin_patch=1)
out = repin_against(up, pin)
check("a pin and upstream that both added an op are repinned", out["action"] == "repin", out)
if out["action"] == "repin":
    repo = Path(out["repo_path"])
    src = git(repo, "show", f"{out['new_sha']}:ggml/src/ggml.c")
    rpc = git(repo, "show", f"{out['new_sha']}:ggml/include/ggml-rpc.h")
    check("the repinned commit has the op-count asserts reconciled to 23",
          src.count('static_assert(GGML_OP_COUNT == 23, "GGML_OP_COUNT != 23");') == 2 and "== 22" not in rpc
          and "GGML_OP_COUNT == 23" in rpc, src + rpc)
    check("and PATCH raised to 0 + 2 (the pin's bump and upstream's merged into one)",
          "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc, rpc)
    check("the work tree is clean (the rewrite is committed)",
          git(repo, "status", "--porcelain", "--untracked-files=no") == "")
    check("the markdown says the asserts were reconciled", "reconciled" in repin.markdown("b2", [out]))

# A pin that added an op without bumping RPC_PROTO_*_VERSION is not repinned around.
up, pin = upstream(pin_patch=0)
out = repin_against(up, pin)
check("a pin that added an op without an RPC bump is refused, not repinned",
      out["action"] == "conflict" and "reconcile_op_count" in out["note"], out)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
