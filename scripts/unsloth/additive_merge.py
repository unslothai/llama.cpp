#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Resolve merge conflicts that are provably pure add/add, and only those.

The conflict that keeps breaking the nightly is always the same shape: upstream
registers a new architecture in a fallthrough group and one of our pinned PRs
registers another one at the same spot. Neither side changed a line the other
side touched -- both only added, at a place where the merge base had nothing.
The union of the two additions is the resolution, and it is mechanical.

Anything else is left conflicted and reported. In particular a conflict where
the merge base is non-empty means at least one side *edited* shared text, and
picking a side or unioning them is a guess. This script never guesses.

Reads a conflicted work tree, writes resolutions in place, exits 0 if every
conflict in every file was resolved and 1 otherwise. `--report` emits JSON
describing what it did for the caller to quote in a PR body.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

OURS = "<<<<<<< "
BASE = "||||||| "
SEP = "======="
THEIRS = ">>>>>>> "


class Unresolvable(Exception):
    """A conflict this script is not allowed to decide."""


def parse_conflicts(lines: list[str]) -> list[tuple[int, int, list[str], list[str], list[str]]]:
    """Split diff3-style content into (start, end, ours, base, theirs) regions.

    Raises Unresolvable if the markers do not nest as diff3 promises, which
    means the file is not in the state we think it is.
    """
    regions = []
    i = 0
    n = len(lines)
    while i < n:
        if not lines[i].startswith(OURS):
            i += 1
            continue
        start = i
        ours: list[str] = []
        base: list[str] = []
        theirs: list[str] = []
        cur = ours
        seen_base = False
        i += 1
        while True:
            if i >= n:
                raise Unresolvable(f"unterminated conflict starting at line {start + 1}")
            ln = lines[i]
            if ln.startswith(OURS):
                raise Unresolvable(f"nested conflict marker at line {i + 1}")
            if ln.startswith(BASE):
                cur = base
                seen_base = True
            elif ln.rstrip("\n") == SEP:
                cur = theirs
            elif ln.startswith(THEIRS):
                i += 1
                break
            else:
                cur.append(ln)
            i += 1
        if not seen_base:
            # Without the base section we cannot tell add/add from edit/edit.
            raise Unresolvable(
                f"conflict at line {start + 1} has no base section; "
                "re-checkout with --conflict=diff3"
            )
        regions.append((start, i, ours, base, theirs))
    return regions


def nonblank(lines: list[str]) -> list[str]:
    return [ln.strip() for ln in lines if ln.strip()]


def resolve_region(ours: list[str], base: list[str], theirs: list[str]) -> list[str]:
    """Return the union, or raise if this region is not a pure add/add."""
    if nonblank(base):
        raise Unresolvable(
            "merge base is not empty, so at least one side edited existing text"
        )
    if not nonblank(ours) or not nonblank(theirs):
        # One side added and the other added nothing: git would not have
        # conflicted, so seeing this means the region is not what we expect.
        raise Unresolvable("one side of the conflict is empty")
    if ours == theirs:
        # Both sides added byte-identical text; one copy is the resolution.
        return list(ours)
    shared = set(nonblank(ours)) & set(nonblank(theirs))
    if shared:
        # Overlapping content is the signature of one construct added twice,
        # not two independent additions. Unioning it would duplicate code.
        raise Unresolvable(
            "both sides add the same line(s), so this is one change made twice: "
            + ", ".join(sorted(shared)[:3])
        )
    # Upstream first, then ours: the same order a human repin produces.
    return list(theirs) + list(ours)


def decide_file(path: Path) -> tuple[str, list[dict]]:
    """Return the resolved content and a per-hunk record, without writing."""
    lines = path.read_text(encoding="utf-8", errors="surrogateescape").splitlines(keepends=True)
    regions = parse_conflicts(lines)
    if not regions:
        raise Unresolvable("no conflict markers found")

    out: list[str] = []
    prev = 0
    hunks = []
    for start, end, ours, base, theirs in regions:
        resolution = resolve_region(ours, base, theirs)
        out.extend(lines[prev:start])
        out.extend(resolution)
        prev = end
        hunks.append(
            {
                "ours": "".join(ours),
                "theirs": "".join(theirs),
                "resolution": "".join(resolution),
            }
        )
    out.extend(lines[prev:])
    return "".join(out), hunks


def conflicted_files(repo: Path) -> list[str]:
    r = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=repo, capture_output=True, text=True, check=True,
    )
    return [f for f in r.stdout.splitlines() if f]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=".", help="conflicted work tree")
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--dry-run", action="store_true", help="decide, but do not write")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    files = conflicted_files(repo)
    report: dict = {"resolved": [], "refused": [], "ok": False}

    if not files:
        report["refused"].append({"file": "-", "reason": "no conflicted files"})

    # Decide every file before writing any of them. A refusal on the second
    # file must not leave the first one already rewritten on disk: the caller
    # would then be looking at a tree that is neither the conflict nor the
    # resolution.
    pending: list[tuple[Path, str]] = []
    for f in files:
        try:
            content, hunks = decide_file(repo / f)
            pending.append((repo / f, content))
            report["resolved"].append({"file": f, "hunks": hunks})
        except Unresolvable as e:
            report["refused"].append({"file": f, "reason": str(e)})
        except OSError as e:
            report["refused"].append({"file": f, "reason": f"cannot read: {e}"})

    report["ok"] = bool(files) and not report["refused"]

    if report["ok"] and not args.dry_run:
        for p, content in pending:
            p.write_text(content, encoding="utf-8", errors="surrogateescape")
        subprocess.run(["git", "add", "--"] + files, cwd=repo, check=True)

    for r in report["resolved"]:
        print(f"resolved {r['file']}")
    for r in report["refused"]:
        print(f"refused  {r['file']}: {r['reason']}", file=sys.stderr)

    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
