#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Resolve the modify/delete conflicts a fork sync produces, and only those.

Measured over every merge commit in this fork: 149 file-level conflicts, of which 68 (45 percent) are the same one.
Upstream edits a workflow this fork deleted on purpose, git cannot know which side wins, and a human keeps the deletion.
Every single time: 68 of 68 historical instances resolved by keeping the deletion, with no exceptions.

That is not a heuristic, it is the fork's stated invariant.
This fork owns no upstream CI.
scripts/unsloth/upstream-sync.json requires the diff from the sync point to master to touch only .github/ and scripts/unsloth/, and verify_upstream_sync.py already treats these deletions as legitimate.
The deletions are policy, so re-applying them is bookkeeping.

Scope is deliberately tight, because the cost of being wrong is a workflow silently reappearing and firing on the fork:

  - only paths under .github/workflows/
  - only modify/delete conflicts, never content conflicts
  - never a file named unsloth-*.yml, which is ours; if one of those is ever in a modify/delete conflict, something is wrong and a human should look
  - the delete must be on our side; upstream deleting a file we modified is the opposite situation and is left alone

The same policy covers a workflow upstream ADDED since the last sync.
That is not a conflict at all, so git merges it in silently and the fork acquires a workflow that starts firing on it.
Replaying sync e8735f35d3 caught exactly this: resolving only the conflicts left .github/workflows/build-wasm.yml in the tree, where the recorded human resolution had deleted it.
With --added handled too, the replay reproduces that tree exactly.

Anything else is left conflicted.
Exits 0 if it resolved everything it was asked about, 1 if any conflict remains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PREFIX = ".github/workflows/"
OURS = "unsloth-"


def git(*args: str, cwd: str = ".") -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def unmerged(cwd: str) -> dict[str, set[int]]:
    """path -> set of stages present (1 base, 2 ours, 3 theirs)."""
    out: dict[str, set[int]] = {}
    r = git("ls-files", "-u", cwd=cwd)
    # A listing that could not run gives empty stdout, which reads as "no
    # conflicts" and makes this script report that it resolved everything it
    # was asked to. Not a repo, or an unreadable index, has to be an error.
    if r.returncode != 0:
        raise RuntimeError(f"git ls-files -u in {cwd}: {r.stderr.strip()}")
    for line in r.stdout.split("\n"):
        if not line.strip():
            continue
        meta, path = line.split("\t", 1)
        stage = int(meta.split()[2])
        out.setdefault(path.strip(), set()).add(stage)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".")
    ap.add_argument("--merge-base", help="also drop upstream workflows added since this rev")
    ap.add_argument("--report", metavar="PATH")
    a = ap.parse_args()

    try:
        stages = unmerged(a.repo)
    except RuntimeError as e:
        print(f"sync_deletes: {e}", file=sys.stderr)
        if a.report:
            Path(a.report).write_text(json.dumps(
                {"ok": False, "resolved": [], "removed_added": [],
                 "left": [str(e)]}, indent=2))
        return 1
    resolved, left, added = [], [], []
    for path, st in sorted(stages.items()):
        name = path.rsplit("/", 1)[-1]
        # stage 2 missing means our side deleted it; stage 3 present means upstream still has it.
        # That is the sync case, and only that.
        ours_deleted = 2 not in st and 3 in st
        if (path.startswith(PREFIX) and not name.startswith(OURS) and ours_deleted):
            r = git("rm", "-q", "--force", "--", path, cwd=a.repo)
            if r.returncode == 0:
                resolved.append(path)
            else:
                left.append(f"{path}: git rm failed: {r.stderr.strip()}")
        else:
            why = ("we own this workflow" if name.startswith(OURS)
                   else "not an upstream workflow path" if not path.startswith(PREFIX)
                   else "not a delete on our side")
            left.append(f"{path}: {why}")

    # Workflows upstream added since the merge base.
    # No conflict, so nothing above sees them, and the fork silently gains CI that fires on its own repo.
    if a.merge_base:
        # Against the WORKING TREE, not HEAD: mid-merge, HEAD is still our pre-merge commit, so merge_base..HEAD describes our side rather than the merge result and finds nothing.
        # --no-renames, or an upstream workflow that was RENAMED reads as R rather than A and this filter drops it.
        # The fork would then carry the renamed workflow and it would start firing here, which is the exact thing this block exists to stop.
        r = git("diff", "--name-status", "--diff-filter=A", "--no-renames",
                a.merge_base, "--", PREFIX, cwd=a.repo)
        if r.returncode != 0:
            # An unusable --merge-base produces empty stdout, which is indistinguishable from "upstream added nothing" if the exit code is ignored.
            # This script reporting success is what tells a sync it can proceed, so a listing that never ran has to be a failure, not a quiet zero: otherwise the sync carries every newly added upstream workflow in and they start firing on the fork.
            left.append(f"{a.merge_base}: could not list workflows added since "
                        f"it: {r.stderr.strip()}")
        else:
            for line in r.stdout.split("\n"):
                if not line.strip():
                    continue
                path = line.split("\t", 1)[1].strip()
                name = path.rsplit("/", 1)[-1]
                if name.startswith(OURS):
                    continue
                rm = git("rm", "-q", "--force", "--", path, cwd=a.repo)
                if rm.returncode == 0:
                    added.append(path)
                else:
                    # Same reasoning as the resolve loop above: a removal that did not happen is reported, never dropped.
                    left.append(f"{path}: git rm failed: {rm.stderr.strip()}")

    for p in resolved:
        print(f"resolved  {p}: kept the fork's deletion")
    for p in added:
        print(f"removed   {p}: upstream added it; this fork carries no upstream workflows")
    for p in left:
        print(f"left      {p}", file=sys.stderr)
    if a.report:
        Path(a.report).write_text(json.dumps(
            {"ok": not left, "resolved": resolved,
             "removed_added": added, "left": left}, indent=2))
    return 1 if left else 0


if __name__ == "__main__":
    sys.exit(main())
