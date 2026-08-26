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
import re
import subprocess
import sys
from pathlib import Path

OURS = "<<<<<<< "
BASE = "||||||| "
SEP = "======="
THEIRS = ">>>>>>> "

# A line that closes a block and nothing else. Two sides that both append a
# block collide on it without meaning the same thing.
DELIMITER_RE = re.compile(r"^[}\)\];,]+$")

VERSION_RE = re.compile(r"^#define\s+(RPC_PROTO_(?:MAJOR|MINOR|PATCH)_VERSION)\s+(\d+)\s*$")
OP_ENUM_OPEN = "    enum ggml_op {"
OP_ENUM_CLOSE = "    };"
OP_MEMBER_RE = re.compile(r"^\s*(GGML_OP_[A-Z0-9_]+)\s*(=\s*[0-9]+\s*)?,\s*(//.*)?$")
RPC_ASSERT_RE = re.compile(r"static_assert\(GGML_OP_COUNT == (\d+)")


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


def is_delimiter(line: str) -> bool:
    """A closing brace / paren / semicolon and nothing else."""
    s = line.strip()
    return s == "#endif" or bool(DELIMITER_RE.match(s))


def code_only(line: str) -> str:
    """Drop // comments and the contents of quotes, so braces inside them do not count."""
    out = []
    quote = ""
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = ""
        elif c in "\"'":
            quote = c
        elif c == "/" and i + 1 < len(line) and line[i + 1] == "/":
            break
        else:
            out.append(c)
        i += 1
    return "".join(out)


def brace_balanced(lines: list[str]) -> bool:
    """Every brace this text opens, it also closes. Blocks only, never fragments."""
    depth = 0
    for ln in lines:
        for c in code_only(ln):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0:
                    return False
    return depth == 0


def resolve_region(
    ours: list[str],
    base: list[str],
    theirs: list[str],
    elsewhere: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the union, or raise if this region is not a pure add/add.

    `elsewhere` holds the file's lines outside every conflict, used to tell an
    idiom the file already repeats from the construct actually being added.
    """
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
    # A shared line only proves "one change made twice" when it is distinctive.
    # A closing brace is not, and neither is a line the file already repeats
    # outside this conflict: both sides writing `common_chat_params data;` means
    # they both wrote a function of that file's usual shape, not the same one.
    distinctive = sorted(s for s in shared if not is_delimiter(s) and s not in elsewhere)
    if distinctive:
        raise Unresolvable(
            "both sides add the same line(s), so this is one change made twice: "
            + ", ".join(distinctive[:3])
        )
    if shared and not (brace_balanced(ours) and brace_balanced(theirs)):
        # Sides overlap and at least one is not a whole block, so the union
        # would splice two fragments together.
        raise Unresolvable(
            "sides share a line and one of them is not brace-balanced, "
            "so this is not two complete blocks"
        )
    # Upstream first, then ours. Order is arbitrary for disjoint additions;
    # real repins have gone both ways.
    return list(theirs) + list(ours)


def parse_versions(lines: list[str]) -> dict[str, tuple[int, str]] | None:
    """{macro: (value, source line)} if the region is only version defines."""
    out: dict[str, tuple[int, str]] = {}
    for ln in lines:
        if not ln.strip():
            continue
        m = VERSION_RE.match(ln.strip())
        if not m:
            return None
        out[m.group(1)] = (int(m.group(2)), ln)
    return out or None


def three_way(ours: int, base: int, theirs: int) -> int | None:
    """The side that moved, or None when both moved differently."""
    if ours == theirs:
        return ours
    if ours == base:
        return theirs
    if theirs == base:
        return ours
    return None


def ggml_op_count(repo: Path) -> int:
    """Count enum ggml_op members in the merged tree. Raises if the shape is unfamiliar."""
    header = repo / "ggml" / "include" / "ggml.h"
    try:
        lines = header.read_text(encoding="utf-8", errors="surrogateescape").splitlines()
    except OSError as e:
        raise Unresolvable(f"cannot read {header.name} to recount the ops: {e}")
    try:
        start = lines.index(OP_ENUM_OPEN)
    except ValueError:
        raise Unresolvable("could not find `enum ggml_op {` to recount the ops")
    count = 0
    for ln in lines[start + 1:]:
        if ln == OP_ENUM_CLOSE:
            # GGML_OP_COUNT is the last member and names the count of the rest.
            return count - 1
        if not ln.strip() or ln.strip().startswith("//"):
            continue
        if not OP_MEMBER_RE.match(ln):
            # A #if, or two members on one line: stop rather than miscount.
            raise Unresolvable(f"unexpected line in enum ggml_op, not recounting: {ln.strip()!r}")
        count += 1
    raise Unresolvable("enum ggml_op is not terminated")


def resolve_rpc_version(
    ours: list[str], base: list[str], theirs: list[str], repo: Path, whole_file: str
) -> list[str]:
    """Merge the RPC protocol version triple field by field.

    Each field moved on at most one side, so ordinary three-way merge settles it
    without needing to know the project's bump convention. Refuses otherwise.
    """
    vo, vb, vt = parse_versions(ours), parse_versions(base), parse_versions(theirs)
    if not (vo and vb and vt):
        raise Unresolvable("not a pure RPC version block")
    if not (set(vo) == set(vb) == set(vt)):
        raise Unresolvable("the two sides define different version macros")

    out = []
    for macro in vb:
        merged = three_way(vo[macro][0], vb[macro][0], vt[macro][0])
        if merged is None:
            raise Unresolvable(f"{macro} moved on both sides ({vo[macro][0]} vs {vt[macro][0]})")
        if macro.endswith("MAJOR_VERSION") and merged != vb[macro][0]:
            # A major bump resets the other two, and how that interacts with the
            # other side's bump is a judgement call.
            raise Unresolvable("the RPC major version changed; resolve this one by hand")
        # Keep the line from whichever side supplied the value, so the column
        # alignment in this header survives.
        src = vo[macro][1] if merged == vo[macro][0] else vt[macro][1]
        out.append(src)

    # The neighbouring static_assert pins the op count. It usually merges on its
    # own, so check it rather than rewrite it: a stale count here is exactly the
    # silent wire-protocol mismatch this file exists to prevent.
    found = RPC_ASSERT_RE.search(whole_file)
    if found:
        actual = ggml_op_count(repo)
        if int(found.group(1)) != actual:
            raise Unresolvable(
                f"static_assert says GGML_OP_COUNT == {found.group(1)} but the merged "
                f"enum has {actual}; fix the assert by hand"
            )
    return out


def decide_file(path: Path, repo: Path = Path(".")) -> tuple[str, list[dict]]:
    """Return the resolved content and a per-hunk record, without writing."""
    text = path.read_text(encoding="utf-8", errors="surrogateescape")
    lines = text.splitlines(keepends=True)
    regions = parse_conflicts(lines)
    if not regions:
        raise Unresolvable("no conflict markers found")

    # Everything the file says outside its conflicts. A line repeated here is an
    # idiom of the file, not evidence that one construct was added twice.
    elsewhere: set[str] = set()
    cursor = 0
    for start, end, *_ in regions:
        elsewhere.update(nonblank(lines[cursor:start]))
        cursor = end
    elsewhere.update(nonblank(lines[cursor:]))
    frozen = frozenset(elsewhere)

    is_rpc_header = path.name == "ggml-rpc.h"

    out: list[str] = []
    prev = 0
    hunks = []
    for start, end, ours, base, theirs in regions:
        try:
            resolution = resolve_region(ours, base, theirs, frozen)
        except Unresolvable:
            if not is_rpc_header:
                raise
            # The version triple is an edit/edit by construction, so the add/add
            # path can never take it.
            resolution = resolve_rpc_version(ours, base, theirs, repo, text)
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
            content, hunks = decide_file(repo / f, repo)
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
