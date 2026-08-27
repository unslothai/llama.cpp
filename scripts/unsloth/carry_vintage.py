#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Report which carry files are an unmodified older copy of the upstream PR.

A carry branch replays an upstream PR onto an aged base tag. When that PR moves,
the question before every refresh is the same one: have we actually changed this
file, or are we just holding a stale copy of theirs? Answering it by hand means
diffing every file the PR touches and reading each hunk, which is what made the
08-27 GLM-5-Next refresh expensive, and two of those hand answers were wrong.

The mechanical answer: if our version of a file is byte-identical to the version
at SOME commit of the upstream PR, then we never edited it, and their newer copy
supersedes ours with nothing lost. That is a fact about blob hashes, not a
judgement. Files we really did change match no upstream commit and are reported
as diverged, which is correct: on 08-27 gguf-py/gguf/tensor_mapping.py did not
match, because it genuinely carried qwen4exp additions as well.

This only reports. It does not resolve, stage or write anything, because
"upstream superseded ours" is not the same as "we want upstream's", and holding
a deliberately older vintage is a legitimate decision this script cannot see.

Use it to decide whether a refresh should merge or simply rebuild:

    python3 scripts/unsloth/carry_vintage.py \\
        --carry <carry sha> --pr-ref refs/pull/27754/head --base refs/tags/b10639

When every file is SUPERSEDED, rebuilding the carry from the PR head avoids the
merge, and its conflicts, entirely. A file the PR head still has and the carry
does not is reported as OMITTED and blocks that advice, because a rebuild would
restore it; a file the PR DELETED is merely ABSENT and blocks nothing.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(args) + ": " + r.stderr.strip())
    return r.stdout.strip()


def blob(rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", f"{rev}:{path}"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carry", required=True, help="carry branch commit")
    ap.add_argument("--pr-ref", required=True, help="upstream PR head ref or sha")
    ap.add_argument("--base", required=True, help="base tag the PR forked from")
    ap.add_argument("--max-commits", type=int, default=60,
                    help="how far back through the PR to look for a match")
    ap.add_argument("--report", metavar="PATH", help="write a JSON summary here")
    a = ap.parse_args()

    head = git("rev-parse", a.pr_ref)
    fork = git("merge-base", head, a.base)
    # --no-renames, because rename detection hides exactly the path that
    # matters here. `git diff --name-only` prints only the NEW name of a
    # rename, so a PR moving `old` to `new` never puts `old` in this list. A
    # carry that deliberately keeps `old` is then never looked at: `new` comes
    # back SUPERSEDED, nothing diverges, and the summary says a rebuild is
    # equivalent when a rebuild deletes the file the carry is holding. Without
    # detection the rename is a delete plus an add, so `old` is classified --
    # DIVERGED, since our copy is the fork's and matches no upstream vintage.
    files = [f for f in git("diff", "--name-only", "--no-renames",
                            fork, head).split("\n") if f]
    # fork..head, not head: `--max-count` caps the output, it does not bound
    # the walk, so a bare `head` runs straight past the fork point into the
    # base branch. A file our carry deliberately holds at the BASE version
    # then matches a pre-fork commit and is called SUPERSEDED, and the summary
    # says rebuilding from the PR head is equivalent -- it is not, it re-adds
    # what the carry dropped. Only commits of the PR itself are vintages.
    history = [c for c in git("rev-list", f"--max-count={a.max_commits}",
                              f"{fork}..{head}").split("\n") if c]

    superseded, diverged, absent, omitted = [], [], [], []
    for path in files:
        ours = blob(a.carry, path)
        if ours is None:
            # Two very different reasons a path is missing from the carry. If
            # the PR DELETED it, the carry agrees and a rebuild reproduces
            # that. If it still exists at the PR head, the carry dropped it on
            # purpose, and a rebuild would re-add it -- the same mistake as
            # calling a file held at the base version superseded.
            (absent if blob(head, path) is None else omitted).append(path)
            continue
        if ours == blob(head, path):
            superseded.append({"path": path, "vintage": head, "current": True})
            continue
        # Bounding the walk to fork..head is not enough on its own. A PR with
        # several commits usually does not touch every file in its first one,
        # so the commits BEFORE the one that first changed this path still
        # carry the fork's blob -- inside the range. A carry holding the file
        # at the base version matches one of those and is called SUPERSEDED
        # again, and the summary again says a rebuild is equivalent when it
        # would overwrite exactly what the carry is holding. Our copy being
        # the fork's copy is not evidence the PR ever produced it, so it is
        # never a vintage; such a file falls through to DIVERGED, which is
        # where a file needing a human decision belongs.
        at_fork = blob(fork, path)
        hit = None if ours == at_fork else \
            next((c for c in history if blob(c, path) == ours), None)
        if hit:
            superseded.append({"path": path, "vintage": hit, "current": False})
        else:
            diverged.append(path)

    print(f"carry {a.carry[:10]} vs {a.pr_ref} ({head[:10]}), {len(files)} file(s) touched")
    print()
    for e in superseded:
        note = "already at PR head" if e["current"] else f"our copy is upstream {e['vintage'][:10]}"
        print(f"  SUPERSEDED  {e['path']}\n              {note}")
    for p in diverged:
        print(f"  DIVERGED    {p}\n              matches no upstream vintage; we changed it, or we are "
              "holding the base version on purpose. Keep it")
    for p in omitted:
        print(f"  OMITTED     {p}\n              exists at the PR head, not in the carry; "
              "a rebuild would re-add it")
    for p in absent:
        print(f"  ABSENT      {p}\n              deleted by the PR, not in the carry")
    print()
    if diverged:
        print(f"{len(diverged)} file(s) genuinely diverge. A refresh has to merge, "
              "and those files are the only ones needing judgement.")
    if omitted:
        print(f"{len(omitted)} file(s) the PR head still has are missing from the "
              "carry. Rebuilding would restore them, so it is NOT equivalent to "
              "merging; keep or re-drop each one deliberately.")
    if not diverged and not omitted:
        print("Nothing diverges. Rebuilding the carry from the PR head is "
              "equivalent to merging it, without the conflicts.")

    if a.report:
        with open(a.report, "w") as fh:
            json.dump({"head": head, "superseded": superseded,
                       "diverged": diverged, "omitted": omitted,
                       "absent": absent}, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
