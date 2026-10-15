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
  4. equal counts do not prove order (the tables are indexed by enum value), so GGML_OP_NAME
     must be the enum names in enum order with GGML_OP_ stripped (upstream holds to this for
     every entry); GGML_OP_SYMBOL is free text, so with --base-ref every op the base had must
     keep the base's symbol, and the symbols of new ops are only counted. A break is exit 1;
  5. all match: rewrite each `static_assert(<COUNT> == M` under ggml/ with M != the
     count (the number, and the same number in that assert's message) and exit 0.

The same for GGML_UNARY_OP_COUNT (GGML_UNARY_OP_NAME) and GGML_GLU_OP_COUNT (GGML_GLU_OP_NAME),
which follow the same pattern. The enum and tables of every family are checked whether or not an
assert is left; with --base-ref a family whose base had an assert and whose merged tree has none
(deleted or commented out) exits 1. A family with no asserts and no enum or table is skipped; one
with asserts whose enum or tables cannot be found exits 2, since a changed layout upstream must
be looked at rather than guessed.

ggml-rpc.h's assert is different: it exists to make whoever changes the op count bump
RPC_PROTO_*_VERSION, so a client and server with different op tables refuse each other. It is
only rewritten with --base-ref (the tag the pins were merged onto), against which:

  * the op count changed and RPC_PROTO_{MAJOR,MINOR,PATCH}_VERSION did not: a pin added an op
    without bumping the protocol. Exit 1, rewrite nothing.
  * the version moved, but MAJOR and MINOR are unchanged and PATCH moved by less than the op
    count changed, up or down (two pins that each bumped PATCH by one merge to a single bump):
    PATCH is set to base PATCH + |op count delta|, and the output says so.
  * otherwise the assert is rewritten like the others.

The merged tree alone hides two cases, so each pin merged since --base-ref (the side parents of
HEAD's first-parent merges, or the pin a base was merged into) is also checked against its own fork
point, merge-base(base, pin): a pin that changed GGML_OP_COUNT there and left RPC_PROTO_*_VERSION
equal is refused even when another pin's bump covers the merged version, and a version that equals
the base's is accepted when the op-changing pins all bumped it (upstream made the same bump from
the same fork point), then PATCH is raised as above.

With --base-ref the version is checked whenever ggml-rpc.h exists, also when the assert already
says the merged count (a pin that added an op and updated the assert but not RPC_PROTO_*_VERSION
is the same unbumped protocol) and when it is gone: an RPC assert the base had and the merged tree
lacks (deleted or commented out) exits 1. Without --base-ref a stale RPC assert is a
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
        elif c == "'" and re.search(r"(?<![\w.])\.?\d[\w.']*$", text[max(0, i - 64):i]):
            i += 1      # a C++14 digit separator (100'000, 0x1'0000), not a char literal
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


def item_spans(body: str) -> list[tuple[int, int]]:
    """(start, end) in body of each comma-separated item at brace/paren depth 0, empty ones dropped."""
    spans, depth, st = [], 0, 0
    for i, c in enumerate(body + ","):
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth -= 1
        elif c == "," and depth == 0:
            if body[st:i].strip():
                spans.append((st, i))
            st = i + 1
    return spans


def top_level_items(body: str) -> list[str]:
    """Comma-separated items at brace/paren depth 0, empty ones (a trailing comma) dropped."""
    return [body[a:b].strip() for a, b in item_spans(body)]


def enum_count(text: str, clean: str, enum: str, count: str) -> tuple[int, int]:
    """(value of `count` in `enum <enum> {`, line of the enum)."""
    n, line, _ = enum_entries(text, clean, enum, count)
    return n, line


def enum_entries(text: str, clean: str, enum: str, count: str) -> tuple[int, int, list[str]]:
    """(value of `count`, line of the enum, names of the entries before `count`)."""
    m = re.search(r"\benum\s+" + re.escape(enum) + r"\s*\{", clean)
    if not m:
        raise LayoutError(f"enum {enum} not found in {HEADER}")
    start, end = braced_body(clean, m.end() - 1)
    # preprocessor lines inside an enum would make the count depend on the build; refuse them
    if re.search(r"^\s*#", clean[start:end], re.M):
        raise LayoutError(f"enum {enum} ({HEADER}:{line_of(text, m.start())}) has preprocessor lines")
    value, names = 0, []
    for item in top_level_items(clean[start:end]):
        name, _, init = (x.strip() for x in item.partition("="))
        if init:
            if not re.fullmatch(r"\d+", init):
                raise LayoutError(f"enum {enum}: {name} = {init} is not a plain integer")
            if int(init) != value:
                # the tables are indexed by enum value: a gap, alias or reordering breaks that
                raise LayoutError(f"enum {enum}: {name} = {init}, expected the sequential value {value}")
        if name == count:
            return value, line_of(text, m.start()), names
        names.append(name)
        value += 1
    raise LayoutError(f"{count} is not in enum {enum} ({HEADER}:{line_of(text, m.start())})")


def table_count(text: str, clean: str, table: str) -> tuple[int, int]:
    """(number of initializers of `static const char * <table>[...] = {`, line of the table)."""
    items, line = table_entries(text, clean, table)
    return len(items), line


def table_entries(text: str, clean: str, table: str) -> tuple[list[tuple[str, int]], int]:
    """([(initializer with comments dropped, its line)], line of the table)."""
    m = re.search(r"\bstatic\s+const\s+char\s*\*\s*" + re.escape(table) + r"\s*\[[^\]]*\]\s*=\s*\{", clean)
    if not m:
        raise LayoutError(f"table {table} not found in {SOURCE}")
    start, end = braced_body(clean, m.end() - 1)
    # an initializer under #if is counted here and missing from a build where the condition is false
    if re.search(r"^\s*#", clean[start:end], re.M):
        raise LayoutError(f"table {table} ({SOURCE}:{line_of(text, m.start())}) has preprocessor lines")
    items = []
    for a, b in item_spans(clean[start:end]):
        a, b = start + a, start + b
        # keep string literals (S in clean) and code; drop comment characters (blank in clean only)
        raw = "".join(t for t, c in zip(text[a:b], clean[a:b]) if c == t or c == "S")
        items.append((" ".join(raw.split()), line_of(text, a + len(clean[a:b]) - len(clean[a:b].lstrip()))))
    return items, line_of(text, m.start())


def check_order(names: list[str], prefix: str, table: str, entries: list[tuple[str, int]]) -> list[str]:
    """Problems with a NAME table that is not the enum names, in enum order, prefix stripped.

    The tables are indexed by enum value, so equal counts are not enough: two entries in the wrong
    order compile and name the wrong op. Upstream follows this rule for every entry of GGML_OP_NAME,
    GGML_UNARY_OP_NAME and GGML_GLU_OP_NAME (checked on b11580), so any break of it is refused.
    """
    bad = []
    for i, (name, (entry, eline)) in enumerate(zip(names, entries)):
        want = '"' + (name[len(prefix):] if name.startswith(prefix) else name) + '"'
        if entry != want:
            bad.append(f"{SOURCE}:{eline}: {table}[{i}] is {entry}, enum entry {i} is {name} (expected {want})")
    return bad


def git_text(root: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="surrogateescape")


def git_lines(root: Path, *args: str) -> list[str]:
    """stdout lines; `git grep` exits 1 for no match, anything else nonzero (a bad ref) is an error."""
    r = git_text(root, *args)
    if r.returncode == 1 and args[0] == "grep":
        return []
    if r.returncode:
        raise LayoutError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.splitlines()


def base_has_assert(root: Path, ref: str, count: str) -> bool:
    """Whether the tree at ref has a real (not commented out) `static_assert(<count> ==` under ggml/."""
    pat = assert_re(count)
    for path in git_lines(root, "grep", "-l", "-F", count, ref, "--", "ggml"):
        path = path.split(":", 1)[1] if path.startswith(ref + ":") else path
        if Path(path).suffix in SCAN_SUFFIXES and pat.search(blank_comments_and_strings(git_show(root, ref, path))):
            return True
    return False


def check_symbols(root: Path, ref: str, names: list[str], entries: list[tuple[str, int]]) -> list[str]:
    """GGML_OP_SYMBOL is free text, so its order cannot be derived from the enum. What can be checked:
    every op the base already had keeps the base's symbol at its (possibly moved) index, so a symbol
    shifted by a dropped or misplaced line is caught. The symbols of new ops are only counted."""
    bh, bs = git_show(root, ref, HEADER), git_show(root, ref, SOURCE)
    _, _, bnames = enum_entries(bh, blank_comments_and_strings(bh), "ggml_op", "GGML_OP_COUNT")
    bentries, _ = table_entries(bs, blank_comments_and_strings(bs), "GGML_OP_SYMBOL")
    if len(bnames) != len(bentries):
        raise LayoutError(f"GGML_OP_SYMBOL at {ref} has {len(bentries)} entries for {len(bnames)} ops")
    base = dict(zip(bnames, (e for e, _ in bentries)))
    return [f"{SOURCE}:{eline}: GGML_OP_SYMBOL[{i}] is {entry}, but {name} has {base[name]} at {ref}; "
            "a symbol line is out of place"
            for i, (name, (entry, eline)) in enumerate(zip(names, entries))
            if name in base and entry != base[name]]


def assert_re(count: str) -> re.Pattern:
    return re.compile(r"\bstatic_assert\s*\(\s*" + re.escape(count) + r"\s*==\s*(\d+)\s*,")


def find_asserts(root: Path, count: str) -> list[tuple[Path, int, int, int, int]]:
    """Every real `static_assert(<count> == N` under ggml/: (file, line, N, start, end of N).

    Searched in the comment/string-blanked text, so a commented-out or quoted assert is ignored.
    """
    pat = assert_re(count)
    hits = []
    for p in sorted((root / "ggml").rglob("*")):
        if p.suffix not in SCAN_SUFFIXES or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="surrogateescape")
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
    s = comma + 1
    while 0 < s < len(clean) and clean[s].isspace():     # comments between are blank in clean
        s += 1
    if comma >= 0 and s < len(text) and text[s] == '"':
        e = s
        while e < len(clean) and clean[e] == "S":
            e += 1
        text = text[:s] + re.sub(r"(?<!\d)" + str(old) + r"(?!\d)", str(new), text[s:e]) + text[e:]
    return text


def rpc_version_matches(text: str) -> dict[str, re.Match]:
    """The one active #define of each RPC_PROTO_*_VERSION text has (parts it lacks are left out).

    A define under #if/#ifdef/#ifndef, or a second one, makes the version depend on the build, as a
    preprocessor line inside the enum or a table does: LayoutError.
    """
    clean = blank_comments_and_strings(text)
    found = {}
    for part in RPC_PARTS:
        prefix = r"^[ \t]*#[ \t]*define[ \t]+RPC_PROTO_" + part + r"_VERSION\b"
        # every definition, whatever its value, so a second one in another form is not missed
        ms = list(re.finditer(prefix + r"[^\n]*", clean, re.M))
        if not ms:
            continue
        if len(ms) > 1:
            raise LayoutError(f"{len(ms)} RPC_PROTO_{part}_VERSION definitions in {RPC_HEADER}")
        # the value must be the whole plain integer: "1 - 1" is 0 to the compiler, not 1
        ms = [re.compile(prefix + r"[ \t]+(\d+)[ \t\r]*", re.M).fullmatch(clean, ms[0].start(), ms[0].end())]
        if ms[0] is None:
            raise LayoutError(f"RPC_PROTO_{part}_VERSION is not a plain integer in {RPC_HEADER}")
        depth = sum(-1 if d == "endif" else 1
                    for d in re.findall(r"^[ \t]*#[ \t]*(if|ifdef|ifndef|endif)\b", clean[:ms[0].start()], re.M))
        if depth:
            raise LayoutError(f"RPC_PROTO_{part}_VERSION is defined under a conditional in {RPC_HEADER}")
        found[part] = ms[0]
    return found


def rpc_version(text: str) -> tuple[dict[str, int], dict[str, tuple[int, int]]]:
    """RPC_PROTO_*_VERSION values and the (start, end) of each number."""
    found = rpc_version_matches(text)
    vals, spans = {}, {}
    for part in RPC_PARTS:
        if part not in found:
            raise LayoutError(f"RPC_PROTO_{part}_VERSION not found in {RPC_HEADER}")
        vals[part], spans[part] = int(found[part].group(1)), found[part].span(1)
    return vals, spans


def git_show(root: Path, ref: str, path: str) -> str:
    r = git_text(root, "show", f"{ref}:{path}")
    if r.returncode:
        raise LayoutError(f"cannot read {path} at {ref}: {r.stderr.strip()}")
    return r.stdout


def base_has_file(root: Path, ref: str, path: str) -> bool:
    return bool(git_lines(root, "ls-tree", "--name-only", ref, "--", path))


def is_ancestor(root: Path, a: str, b: str) -> bool:
    r = git_text(root, "merge-base", "--is-ancestor", a, b)
    if r.returncode not in (0, 1):
        raise LayoutError(f"git merge-base --is-ancestor {a} {b} failed: {r.stderr.strip()}")
    return r.returncode == 0


def merged_pins(root: Path, base_ref: str) -> list[str]:
    """The commits merged onto base_ref to make HEAD, read off its first-parent history.

    The nightly checks out the base and merges each pin (`git merge --no-ff`): every first-parent
    merge's other parents are pins. repin.py merges the base INTO a pin: a merge whose other parents
    are all in the base has the pin as its first parent, and everything below it is that pin's own
    history. Non-merge commits (a reconcile commit) are skipped. [] when HEAD has no such merges.
    """
    pins = []
    for line in git_lines(root, "rev-list", "--first-parent", "--parents", f"{base_ref}..HEAD"):
        parents = line.split()[1:]
        if len(parents) < 2:
            continue
        side = [p for p in parents[1:] if not is_ancestor(root, p, base_ref)]
        if side:
            pins += side
        else:
            pins.append(parents[0])
            break
    return pins


def unbumped_pins(root: Path, base_ref: str) -> tuple[list[str], int]:
    """(problems, number of pins that changed GGML_OP_COUNT and bumped RPC_PROTO_*_VERSION).

    Each merged pin against its own fork point merge-base(base_ref, pin): a pin that changed the op
    count must have moved the version. The merged tree alone cannot say this. One pin's bump hides
    another's missing one (both PATCH 0 -> 1 or 0 -> 0 merge to 1), and upstream bumping PATCH to the
    same value as a pin, from the same fork point, leaves the merged version equal to the base's.
    """
    bad, bumped = [], 0
    for pin in merged_pins(root, base_ref):
        fork = git_lines(root, "merge-base", base_ref, pin)[0]
        ph, fh = git_show(root, pin, HEADER), git_show(root, fork, HEADER)
        pn, _ = enum_count(ph, blank_comments_and_strings(ph), "ggml_op", "GGML_OP_COUNT")
        fn, _ = enum_count(fh, blank_comments_and_strings(fh), "ggml_op", "GGML_OP_COUNT")
        if pn == fn:
            continue
        pv, _ = rpc_version(git_show(root, pin, RPC_HEADER))
        fv, _ = rpc_version(git_show(root, fork, RPC_HEADER))
        # a lower version is no bump either
        if tuple(pv[p] for p in RPC_PARTS) <= tuple(fv[p] for p in RPC_PARTS):
            bad.append(f"merged pin {pin[:12]} changed GGML_OP_COUNT by {pn - fn:+d} versus its fork point "
                       f"{fork[:12]} without bumping RPC_PROTO_*_VERSION")
        else:
            bumped += 1
    return bad, bumped


def check_rpc(root: Path, base_ref: str | None, n: int, has_assert: bool = True) -> tuple[str, int | None]:
    """('ok' | 'refuse: why', PATCH to write or None) for RPC_PROTO_*_VERSION versus base_ref.

    has_assert: the merged tree still has a real GGML_OP_COUNT assert in ggml-rpc.h. One the base
    had and the merge lost (deleted or commented out) is a refusal: it is what forces the bump.
    """
    if not base_ref:
        return ("refuse: the RPC assert guards RPC_PROTO_*_VERSION and can only be rewritten "
                "against the base tag; pass --base-ref", None)
    try:
        bh = git_show(root, base_ref, HEADER)
        base_n, _ = enum_count(bh, blank_comments_and_strings(bh), "ggml_op", "GGML_OP_COUNT")
        base_rpc = git_show(root, base_ref, RPC_HEADER)
        base_v, _ = rpc_version(base_rpc)
        cur_v, _ = rpc_version((root / RPC_HEADER).read_text(encoding="utf-8", errors="surrogateescape"))
        unbumped, bumped = unbumped_pins(root, base_ref)
    except (LayoutError, OSError) as e:
        return (f"refuse: {e}", None)
    if not has_assert and assert_re("GGML_OP_COUNT").search(blank_comments_and_strings(base_rpc)):
        return (f"refuse: the GGML_OP_COUNT static_assert that {RPC_HEADER} has at {base_ref} is gone from the "
                "merged tree (deleted or commented out); it is what makes an op change bump RPC_PROTO_*_VERSION", None)
    fmt = lambda v: ".".join(str(v[p]) for p in RPC_PARTS)  # noqa: E731
    # a resolution that kept a pin's older series over the base's rolls the protocol back
    if (cur_v["MAJOR"], cur_v["MINOR"]) < (base_v["MAJOR"], base_v["MINOR"]):
        return (f"refuse: RPC_PROTO_*_VERSION regressed from {fmt(base_v)} at {base_ref} to {fmt(cur_v)} "
                "in the merged tree", None)
    delta = n - base_n
    print(f"rpc  GGML_OP_COUNT {base_n} at {base_ref} -> {n}; RPC_PROTO {fmt(base_v)} -> {fmt(cur_v)}")
    if unbumped:
        return ("refuse: " + "; ".join(unbumped) + "; a pin added or removed an op without bumping the RPC "
                "protocol", None)
    # unchanged with a bumping pin merged: upstream made the same bump from the same fork point
    if delta and cur_v == base_v and not bumped:
        return (f"refuse: GGML_OP_COUNT changed by {delta:+d} versus {base_ref} but RPC_PROTO_*_VERSION "
                f"is still {fmt(base_v)}; a pin added or removed an op without bumping the RPC protocol", None)
    same_mm = all(cur_v[p] == base_v[p] for p in ("MAJOR", "MINOR"))
    # with an op change the PATCH repair below raises it past the base's anyway
    if same_mm and not delta and cur_v["PATCH"] < base_v["PATCH"]:
        return (f"refuse: RPC_PROTO_*_VERSION regressed from {fmt(base_v)} at {base_ref} to {fmt(cur_v)} "
                "in the merged tree", None)
    # removed ops are handled like added ones: every pin bumps PATCH by one per op it changes, and
    # identical bumps merge to one in either direction
    if delta and same_mm and cur_v["PATCH"] - base_v["PATCH"] < abs(delta):
        return ("ok", base_v["PATCH"] + abs(delta))
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
        htext = (root / HEADER).read_text(encoding="utf-8", errors="surrogateescape")
        stext = (root / SOURCE).read_text(encoding="utf-8", errors="surrogateescape")
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
        # with --base-ref the RPC version is checked even when no GGML_OP_COUNT assert is left
        # also when the merge deleted ggml-rpc.h: the base had it, so check_rpc refuses its absence
        rpc_check = count == "GGML_OP_COUNT" and bool(args.base_ref) and (
            rpc_path.is_file() or base_has_file(root, args.base_ref, RPC_HEADER))
        # before the enum and tables are parsed: a merge that deleted the last assert must not pass
        # because it also left a layout this cannot read (that would take the skip below)
        if not asserts and args.base_ref:
            try:
                had = base_has_assert(root, args.base_ref, count)
            except LayoutError as e:
                print(f"ERROR {count}: {e}", file=sys.stderr)
                bad += 1
                continue
            if had:
                print(f"{count}: {args.base_ref} has a static_assert({count} == ...) under ggml/ and the merged "
                      "tree has none (deleted or commented out); restore it", file=sys.stderr)
                bad += 1
                continue
        # the enum and tables are checked whether or not any assert is left
        try:
            n, eline, names = enum_entries(htext, hclean, enum, count)
            entries = {t: table_entries(stext, sclean, t) for t in tables}
        except LayoutError as e:
            if not asserts and not rpc_check:
                print(f"skip {count}: no static_assert in the tree and {e}")
                continue
            print(f"ERROR {count}: {e}", file=sys.stderr)
            layout += 1
            continue
        wrong = {t: (len(v[0]), v[1]) for t, v in entries.items() if len(v[0]) != n}
        if wrong:
            for t, (k, tline) in wrong.items():
                print(f"{SOURCE}:{tline}: {t} has {k} entries, enum {enum} has {n} before {count} "
                      f"({HEADER}:{eline}); a merge dropped or duplicated an entry", file=sys.stderr)
            bad += 1
            continue
        order = check_order(names, count[:-len("COUNT")], tables[0], entries[tables[0]][0])
        if count == "GGML_OP_COUNT" and args.base_ref:
            try:
                order += check_symbols(root, args.base_ref, names, entries["GGML_OP_SYMBOL"][0])
            except LayoutError as e:
                order.append(f"cannot check GGML_OP_SYMBOL order: {e}")
        if order:
            for x in order[:10]:
                print(x, file=sys.stderr)
            print(f"{count}: {len(order)} table entr{'y is' if len(order) == 1 else 'ies are'} out of enum order",
                  file=sys.stderr)
            bad += 1
            continue
        if not asserts and not rpc_check:
            print(f"ok   {count} = {n}: enum and {', '.join(tables)} agree; no static_assert in the tree")
            continue
        stale = [a for a in asserts if a[2] != n]
        # the RPC assert guards RPC_PROTO_*_VERSION: with --base-ref the version is checked whenever
        # ggml-rpc.h exists, also when the assert is current (a pin may have updated it and not bumped
        # the protocol) or gone (a pin may have deleted or commented it out)
        rpc_assert = any(p == rpc_path for p, *_ in asserts)
        rpc_stale = any(p == rpc_path for p, *_ in stale)
        if rpc_assert and not args.base_ref and not rpc_stale:
            print(f"note {RPC_HEADER}: RPC_PROTO_*_VERSION not checked (no --base-ref); the assert already says {n}")
        elif rpc_assert or rpc_check:
            verdict, patch = check_rpc(root, args.base_ref, n, rpc_assert)
            if verdict != "ok":
                print(f"{RPC_HEADER}: {verdict[len('refuse: '):]}", file=sys.stderr)
                bad += 1
                continue
            if patch is not None:
                vals, spans = rpc_version(rpc_path.read_text(encoding="utf-8", errors="surrogateescape"))
                old = vals["PATCH"]
                plan.setdefault(rpc_path, []).append((*spans["PATCH"], old, patch, False))
                print(f"fix  {RPC_HEADER}: RPC_PROTO_PATCH_VERSION {old} -> {patch} (the op count changed by "
                      f"more than PATCH moved: identical bumps from several pins merged into one)")
        if not stale:
            print(f"ok   {count} = {n}: enum and {', '.join(tables)} agree, {len(asserts)} assert(s) already say {n}")
            continue
        for p, i, v, st, en in stale:
            plan.setdefault(p, []).append((st, en, v, n, True))
            print(f"fix  {p.relative_to(root).as_posix()}:{i}: {count} == {v} -> {n}")
        print(f"ok   {count} = {n}: enum and {', '.join(tables)} agree; {len(stale)} of {len(asserts)} "
              f"assert(s) {'would be ' if args.dry_run else ''}rewritten")

    if bad or layout:
        print(f"\nreconcile_op_count: {bad} mismatch(es) or unbumped RPC version(s), {layout} unreadable "
              "layout(s); nothing rewritten", file=sys.stderr)
        return 1 if bad else 2
    if not args.dry_run:
        for p, edits in plan.items():
            text = p.read_text(encoding="utf-8", errors="surrogateescape")
            # back to front, so earlier offsets stay valid
            for st, en, old, new, is_assert in sorted(edits, reverse=True):
                if is_assert:
                    text = rewrite_assert(text, st, en, old, new)
                else:
                    text = text[:st] + str(new) + text[en:]
            p.write_text(text, encoding="utf-8", errors="surrogateescape")
    nfix = sum(map(len, plan.values()))
    print(f"reconcile_op_count: {'clean' if not plan else ('would rewrite ' if args.dry_run else 'rewrote ') + str(nfix) + ' line(s)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
