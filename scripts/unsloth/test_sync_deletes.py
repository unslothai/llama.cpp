#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for sync_deletes.py. Run: python3 scripts/unsloth/test_sync_deletes.py

Every case builds a real git merge, so the index stages are the ones git
actually produces rather than a hand-written approximation.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "sync_deletes.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=repo, capture_output=True, text=True)


def scenario(path, ours_deletes=True, upstream_modifies=True, upstream_adds=None):
    """Base has `path`; upstream edits it; we delete it. Returns (repo, merge_base)."""
    d = Path(tempfile.mkdtemp(prefix="sd_"))
    git(d, "init", "-q", "-b", "main")
    f = d / path
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("name: base\n")
    (d / "src").mkdir(exist_ok=True)
    (d / "src" / "model.cpp").write_text("int x;\n")
    git(d, "add", "-A"); git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD").stdout.strip()

    git(d, "checkout", "-qb", "upstream")
    if upstream_modifies:
        f.write_text("name: upstream edit\n")
    if upstream_adds:
        na = d / upstream_adds
        na.parent.mkdir(parents=True, exist_ok=True)
        na.write_text("name: new upstream workflow\n")
    git(d, "add", "-A"); git(d, "commit", "-qm", "upstream")

    git(d, "checkout", "-q", "main")
    if ours_deletes:
        git(d, "rm", "-q", str(f.relative_to(d)))
        git(d, "commit", "-qm", "fork deletes it")
    git(d, "-c", "merge.conflictStyle=diff3", "merge", "--no-ff", "--no-edit", "upstream")
    return d, base


def run(repo, base=None):
    args = [sys.executable, str(SCRIPT), "--repo", str(repo)]
    if base:
        args += ["--merge-base", base]
    r = subprocess.run(args, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


# 1. the 68-of-68 historical case
d, base = scenario(".github/workflows/build-apple.yml")
rc, out = run(d)
check("resolves an upstream-workflow modify/delete", rc == 0, out)
check("the file stays deleted", not (d / ".github/workflows/build-apple.yml").exists())
check("no unmerged paths remain", git(d, "ls-files", "-u").stdout.strip() == "")

# 2. a workflow we own must never be touched automatically
d, base = scenario(".github/workflows/unsloth-prebuilt.yml")
rc, out = run(d)
check("refuses a fork-owned workflow", rc == 1 and "we own this workflow" in out, out)

# 3. a source file in the same shape must never be touched
d, base = scenario("src/model.cpp")
rc, out = run(d)
check("refuses a source file", rc == 1 and "not an upstream workflow path" in out, out)

# 4. a workflow upstream added, which is not a conflict at all
d, base = scenario(".github/workflows/build-apple.yml",
                   upstream_adds=".github/workflows/build-wasm.yml")
rc, out = run(d, base)
check("drops a newly added upstream workflow", rc == 0, out)
check("the added workflow is gone", not (d / ".github/workflows/build-wasm.yml").exists())

# 5. an upstream composite ACTION must survive; only workflows are dropped
d, base = scenario(".github/workflows/build-apple.yml",
                   upstream_adds=".github/actions/ccache-buckets/action.yml")
rc, out = run(d, base)
check("keeps upstream composite actions", (d / ".github/actions/ccache-buckets/action.yml").exists(), out)

# 6. source files are never removed by the added-workflow sweep
check("source file untouched throughout", (d / "src" / "model.cpp").exists())

# 7. an unusable --merge-base. git diff exits nonzero with empty stdout, which
# reads exactly like "upstream added nothing" if only stdout is looked at, so
# the run reported success and a sync would have carried every newly added
# upstream workflow in. The listing failing has to fail the script.
d, base = scenario(".github/workflows/build-apple.yml",
                   upstream_adds=".github/workflows/build-wasm.yml")
rc, out = run(d, "0000000000000000000000000000000000000000")
check("fails when the added-workflow listing cannot run", rc == 1, out)
check("says which rev it could not use",
      "0000000000" in out and "could not list" in out, out)
check("and leaves the added workflow in place to be dealt with",
      (d / ".github/workflows/build-wasm.yml").exists(), out)

# 8. upstream RENAMES a workflow rather than adding one. Rename detection calls
# the new path R, not A, so --diff-filter=A saw nothing and the script exited 0
# with the renamed upstream workflow left live in the fork.
def rename_scenario():
    d = Path(tempfile.mkdtemp(prefix="sd_"))
    git(d, "init", "-q", "-b", "main")
    wf = d / ".github" / "workflows"
    wf.mkdir(parents=True)
    # Long enough that git scores the move as a rename rather than add+delete.
    (wf / "old.yml").write_text("".join(f"# line {i}\n" for i in range(40)))
    (d / "src").mkdir()
    (d / "src" / "model.cpp").write_text("int x;\n")
    git(d, "add", "-A"); git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD").stdout.strip()

    git(d, "checkout", "-qb", "upstream")
    git(d, "mv", ".github/workflows/old.yml", ".github/workflows/new.yml")
    git(d, "commit", "-qm", "upstream renames it")

    git(d, "checkout", "-q", "main")
    git(d, "merge", "--no-ff", "--no-edit", "upstream")
    return d, base


d, base = rename_scenario()
rc, out = run(d, base)
check("drops an upstream workflow that arrived by rename", rc == 0, out)
check("the renamed workflow is gone",
      not (d / ".github/workflows/new.yml").exists(), out)
check("source is still untouched", (d / "src" / "model.cpp").exists(), out)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all sync_deletes tests passed")
sys.exit(1 if FAILS else 0)
