#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Reconcile the hand-written ggml op counts after the pins are merged.

ggml keeps the number of ops in `static_assert(GGML_OP_COUNT == N, "GGML_OP_COUNT != N")`
lines (two in ggml/src/ggml.c, one in ggml/include/ggml-rpc.h). Every PR that adds an op
bumps them from N to N+1. Two pins that each add an op therefore make the SAME edit, N to
N+1, and git merges identical edits silently, so the merged tree has N+2 enum entries, asserts
that still say N+1, and does not compile. Observed 10-13: ggml-org#25731 (Inkling,
GGML_OP_FLASH_ATTN_EXT_BANDED) and unslothai#268 (GGML_OP_MOE_CACHE_LOOKUP) both took 101 to
102; the enum had 103 entries.

The assert is a tripwire for one real mistake: an op added to `enum ggml_op` without its
name and symbol (the tables are sized by GGML_OP_COUNT, so a missing entry compiles and
silently shifts every later name). So this checks that, and only then moves the number:

  1. count the `enum ggml_op` entries before GGML_OP_COUNT in ggml/include/ggml.h;
  2. count the entries of each table that must have one per op in ggml/src/ggml.c
     (GGML_OP_NAME and GGML_OP_SYMBOL);
  3. a table that does not match the enum is a real merge bug: report file:line, exit 1,
     rewrite nothing;
  4. every table matches: rewrite each `static_assert(<COUNT> == M` under ggml/ with M != the
     count (the number, and the same number in that assert's message) and exit 0.

The same for GGML_UNARY_OP_COUNT (GGML_UNARY_OP_NAME) and GGML_GLU_OP_COUNT (GGML_GLU_OP_NAME),
which follow the same pattern. A family with no asserts in the tree is skipped; one with
asserts whose enum or tables cannot be found exits 2, since a changed layout upstream must
be looked at rather than guessed.

ggml-rpc.h's assert is different: it exists to make whoever changes the op count bump
RPC_PROTO_*_VERSION, so a client and server with different op tables refuse each other. It is
only rewritten with --base-ref (the tag the pins were merged onto), against which:

  * the op count changed and RPC_PROTO_{MAJOR,MINOR,PATCH}_VERSION did not: a pin added an op
    without bumping the protocol. Exit 1, rewrite nothing.
  * the version moved, but MAJOR and MINOR are unchanged and PATCH moved by less than the op
    count rose (two pins that each bumped PATCH by one merge to a single bump): PATCH is set to
    base PATCH + the op count delta, and the output says so.
  * otherwise the assert is rewritten like the others.

With --base-ref the version is checked whenever the RPC assert is in the tree, also when the
assert already says the merged count: a pin that added an op and updated the assert but not
RPC_PROTO_*_VERSION is the same unbumped protocol. Without --base-ref a stale RPC assert is a
refusal (exit 1), never a silent rewrite; a current one passes, and the output says the version
was not checked.

Nothing is committed: the tree is left modified, and the caller commits it if a tree id has to
cover the rewrite. Run after the merge loop and before merge_checks.py, pin_contract.py and the
build. Exits 0 when the tree is consistent (rewritten or already right), 1 on a table/enum
mismatch or an RPC version that was not bumped, 2 when the layout cannot be parsed.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER = "ggml/include/ggml.h"
SOURCE = "ggml/src/ggml.c"
RPC_HEADER = "ggml/include/ggml-rpc.h"
RPC_PARTS = ("MAJOR", "MINOR", "PATCH")
# (count constant, enum name, tables in ggml.c that need one entry per enum value)
FAMILIES = (
    ("GGML_OP_COUNT", "ggml_op", ("GGML_OP_NAME", "GGML_OP_SYMBOL")),
    ("GGML_UNARY_OP_COUNT", "ggml_unary_op", ("GGML_UNARY_OP_NAME",)),
    ("GGML_GLU_OP_COUNT", "ggml_glu_op", ("GGML_GLU_OP_NAME",)),
)
SCAN_SUFFIXES = {".h", ".c", ".cpp", ".hpp", ".cu", ".cuh", ".m", ".mm"}


class LayoutError(Exception):
    """The enum or a table could not be found or read."""


def blank_comments_and_strings(text: str) -> str:
    """Same length as text, comments blanked to spaces, string and char literals to 'S'.

    Newlines are kept so offsets still map to line numbers. A string becomes one run of S
    characters, which is all the item counter needs; a comma or brace inside a literal is gone.
    """
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = "S"
            i = j
        else:
            i += 1
    return "".join(out)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def braced_body(clean: str, open_pos: int) -> tuple[int, int]:
    """(start, end) of the text between the brace at open_pos and its match."""
    depth = 0
    for i in range(open_pos, len(clean)):
        if clean[i] == "{":
            depth += 1
        elif clean[i] == "}":
            depth -= 1
            if depth == 0:
                return open_pos + 1, i
    raise LayoutError("unbalanced braces")


def top_level_items(body: str) -> list[str]:
    """Comma-separated items at brace/paren depth 0, empty ones (a trailing comma) dropped."""
    items, depth, cur = [], 0, []
    for c in body:
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        if c == "," and depth == 0:
            items.append("".join(cur))
            cur = []
        else:
            cur.append(c)
    items.append("".join(cur))
    return [x.strip() for x in items if x.strip()]


def enum_count(text: str, clean: str, enum: str, count: str) -> tuple[int, int]:
    """(value of `count` in `enum <enum> {`, line of the enum)."""
    m = re.search(r"\benum\s+" + re.escape(enum) + r"\s*\{", clean)
    if not m:
        raise LayoutError(f"enum {enum} not found in {HEADER}")
    start, end = braced_body(clean, m.end() - 1)
    # preprocessor lines inside an enum would make the count depend on the build; refuse them
    if re.search(r"^\s*#", clean[start:end], re.M):
        raise LayoutError(f"enum {enum} ({HEADER}:{line_of(text, m.start())}) has preprocessor lines")
    value = 0
    for item in top_level_items(clean[start:end]):
        name, _, init = (x.strip() for x in item.partition("="))
        if init:
            if not re.fullmatch(r"\d+", init):
                raise LayoutError(f"enum {enum}: {name} = {init} is not a plain integer")
            value = int(init)
        if name == count:
            return value, line_of(text, m.start())
        value += 1
    raise LayoutError(f"{count} is not in enum {enum} ({HEADER}:{line_of(text, m.start())})")


def table_count(text: str, clean: str, table: str) -> tuple[int, int]:
    """(number of initializers of `static const char * <table>[...] = {`, line of the table)."""
    m = re.search(r"\bstatic\s+const\s+char\s*\*\s*" + re.escape(table) + r"\s*\[[^\]]*\]\s*=\s*\{", clean)
    if not m:
        raise LayoutError(f"table {table} not found in {SOURCE}")
    start, end = braced_body(clean, m.end() - 1)
    return len(top_level_items(clean[start:end])), line_of(text, m.start())


def find_asserts(root: Path, count: str) -> list[tuple[Path, int, int, int, int]]:
    """Every real `static_assert(<count> == N` under ggml/: (file, line, N, start, end of N).

    Searched in the comment/string-blanked text, so a commented-out or quoted assert is ignored.
    """
    pat = re.compile(r"\bstatic_assert\s*\(\s*" + re.escape(count) + r"\s*==\s*(\d+)\s*,")
    hits = []
    for p in sorted((root / "ggml").rglob("*")):
        if p.suffix not in SCAN_SUFFIXES or not p.is_file():
            continue
        text = p.read_text(errors="replace")
        if count not in text:
            continue
        for m in pat.finditer(blank_comments_and_strings(text)):
            hits.append((p, line_of(text, m.start()), int(m.group(1)), m.start(1), m.end(1)))
    return hits


def rewrite_assert(text: str, start: int, end: int, old: int, new: int) -> str:
    """The number at text[start:end], and the same number inside the assert's message string."""
    clean = blank_comments_and_strings(text)
    text = text[:start] + str(new) + text[end:]
    clean = clean[:start] + str(new) + clean[end:]
    comma = clean.find(",", start)
    s = text.find('"', comma)
    if comma >= 0 and s >= 0 and not text[comma + 1:s].strip():
        e = s
        while e < len(clean) and clean[e] == "S":
            e += 1
        text = text[:s] + re.sub(r"(?<!\d)" + str(old) + r"(?!\d)", str(new), text[s:e]) + text[e:]
    return text


def rpc_version(text: str) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    """RPC_PROTO_*_VERSION values and the (start, end) of each number."""
    clean = blank_comments_and_strings(text)
    vals, spans = {}, {}
    for part in RPC_PARTS:
        m = re.search(r"^[ \t]*#[ \t]*define[ \t]+RPC_PROTO_" + part + r"_VERSION[ \t]+(\d+)\b", clean, re.M)
        if not m:
            raise LayoutError(f"RPC_PROTO_{part}_VERSION not found in {RPC_HEADER}")
        vals[part], spans[part] = int(m.group(1)), m.span(1)
    return vals, spans


def git_show(root: Path, ref: str, path: str) -> str:
    r = subprocess.run(["git", "-C", str(root), "show", f"{ref}:{path}"], capture_output=True, text=True)
    if r.returncode:
        raise LayoutError(f"cannot read {path} at {ref}: {r.stderr.strip()}")
    return r.stdout


def check_rpc(root: Path, base_ref: str | None, n: int) -> tuple[str, int | None]:
    """('ok' | 'refuse: why', PATCH to write or None) for the RPC op-count assert versus base_ref."""
    if not base_ref:
        return ("refuse: the RPC assert guards RPC_PROTO_*_VERSION and can only be rewritten "
                "against the base tag; pass --base-ref", None)
    try:
        bh = git_show(root, base_ref, HEADER)
        base_n, _ = enum_count(bh, blank_comments_and_strings(bh), "ggml_op", "GGML_OP_COUNT")
        base_v, _ = rpc_version(git_show(root, base_ref, RPC_HEADER))
        cur_v, _ = rpc_version((root / RPC_HEADER).read_text(errors="replace"))
    except (LayoutError, OSError) as e:
        return (f"refuse: {e}", None)
    fmt = lambda v: ".".join(str(v[p]) for p in RPC_PARTS)  # noqa: E731
    delta = n - base_n
    print(f"rpc  GGML_OP_COUNT {base_n} at {base_ref} -> {n}; RPC_PROTO {fmt(base_v)} -> {fmt(cur_v)}")
    if delta and cur_v == base_v:
        return (f"refuse: GGML_OP_COUNT changed by {delta:+d} versus {base_ref} but RPC_PROTO_*_VERSION "
                f"is still {fmt(base_v)}; a pin added or removed an op without bumping the RPC protocol", None)
    same_mm = all(cur_v[p] == base_v[p] for p in ("MAJOR", "MINOR"))
    if delta > 0 and same_mm and cur_v["PATCH"] - base_v["PATCH"] < delta:
        return ("ok", base_v["PATCH"] + delta)
    return ("ok", None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="the merged llama.cpp tree")
    ap.add_argument("--base-ref", help="git ref of the tag the pins were merged onto; required to "
                    "rewrite the RPC op-count assert")
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    try:
        htext = (root / HEADER).read_text(errors="replace")
        stext = (root / SOURCE).read_text(errors="replace")
    except OSError as e:
        print(f"reconcile_op_count: {e}", file=sys.stderr)
        return 2
    hclean, sclean = blank_comments_and_strings(htext), blank_comments_and_strings(stext)
    rpc_path = root / RPC_HEADER

    # file -> [(start, end, old, new, is_assert)] spans to replace
    plan: dict[Path, list[tuple[int, int, int, int, bool]]] = {}
    bad = layout = 0
    for count, enum, tables in FAMILIES:
        asserts = find_asserts(root, count)
        if not asserts:
            print(f"skip {count}: no static_assert in the tree")
            continue
        try:
            n, eline = enum_count(htext, hclean, enum, count)
            sizes = {t: table_count(stext, sclean, t) for t in tables}
        except LayoutError as e:
            print(f"ERROR {count}: {e}", file=sys.stderr)
            layout += 1
            continue
        wrong = {t: v for t, v in sizes.items() if v[0] != n}
        if wrong:
            for t, (k, tline) in wrong.items():
                print(f"{SOURCE}:{tline}: {t} has {k} entries, enum {enum} has {n} before {count} "
                      f"({HEADER}:{eline}); a merge dropped or duplicated an entry", file=sys.stderr)
            bad += 1
            continue
        stale = [a for a in asserts if a[2] != n]
        # the RPC assert guards RPC_PROTO_*_VERSION: with --base-ref the version is checked even
        # when the assert is current (a pin may have updated it and not bumped the protocol)
        rpc_stale = any(p == rpc_path for p, *_ in stale)
        if any(p == rpc_path for p, *_ in asserts) and not args.base_ref and not rpc_stale:
            print(f"note {RPC_HEADER}: RPC_PROTO_*_VERSION not checked (no --base-ref); the assert already says {n}")
        elif any(p == rpc_path for p, *_ in asserts):
            verdict, patch = check_rpc(root, args.base_ref, n)
            if verdict != "ok":
                print(f"{RPC_HEADER}: {verdict[len('refuse: '):]}", file=sys.stderr)
                bad += 1
                continue
            if patch is not None:
                vals, spans = rpc_version(rpc_path.read_text(errors="replace"))
                old = vals["PATCH"]
                plan.setdefault(rpc_path, []).append((*spans["PATCH"], old, patch, False))
                print(f"fix  {RPC_HEADER}: RPC_PROTO_PATCH_VERSION {old} -> {patch} (the op count rose by "
                      f"more than PATCH moved: identical bumps from several pins merged into one)")
        if not stale:
            print(f"ok   {count} = {n}: enum and {', '.join(tables)} agree, {len(asserts)} assert(s) already say {n}")
            continue
        for p, i, v, st, en in stale:
            plan.setdefault(p, []).append((st, en, v, n, True))
            print(f"fix  {p.relative_to(root)}:{i}: {count} == {v} -> {n}")
        print(f"ok   {count} = {n}: enum and {', '.join(tables)} agree; {len(stale)} of {len(asserts)} "
              f"assert(s) {'would be ' if args.dry_run else ''}rewritten")

    if bad or layout:
        print(f"\nreconcile_op_count: {bad} mismatch(es) or unbumped RPC version(s), {layout} unreadable "
              "layout(s); nothing rewritten", file=sys.stderr)
        return 1 if bad else 2
    if not args.dry_run:
        for p, edits in plan.items():
            text = p.read_text()
            # back to front, so earlier offsets stay valid
            for st, en, old, new, is_assert in sorted(edits, reverse=True):
                if is_assert:
                    text = rewrite_assert(text, st, en, old, new)
                else:
                    text = text[:st] + str(new) + text[en:]
            p.write_text(text)
    nfix = sum(map(len, plan.values()))
    print(f"reconcile_op_count: {'clean' if not plan else ('would rewrite ' if args.dry_run else 'rewrote ') + str(nfix) + ' line(s)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
