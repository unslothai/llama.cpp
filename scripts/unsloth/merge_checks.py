#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Post-merge checks for the two mistakes a clean build does not catch.

Both of these were made for real on 08-27, resolving GLM-5-Next against the qwen4exp carry, and both survived compilation:

  1. A resolver unioned two byte-identical additions and produced the same `MODEL_ARCH.GLM5NEXT` key twice in the tensor map.
     Python keeps the last definition of a duplicate key, silently, so the file imports, the build passes, and the converter reads the wrong mapping.

  2. The same union kept both an arch in a shared fallthrough condition AND a dedicated `else if (arch == LLM_ARCH_GLM5NEXT)` arm below it.
     The shared condition matches first, so the dedicated arm is dead.
     It compiles, and the model runs with the indexer cache that arm was supposed to build.

Neither is a merge resolver.
They decide nothing and rewrite nothing.
They turn a silent wrong answer into a loud one, which is the property that was missing.

Both are deliberately narrow, because a check that fires wrongly blocks a release just as effectively as a bad merge:

  - An arch is only consumed by an EARLIER arm that is a pure disjunction of `arch == LLM_ARCH_*` terms.
    A conditional arm may not run, so what follows it stays reachable.
    The later arm is then dead if it is a disjunction whose alternatives are all consumed, or a plain conjunction requiring a consumed arch, since its other conditions can only narrow it further.
    Anything mixing `||` and `&&` is left alone, and only a term that is exactly `arch == X` counts, so `arch != X` is never read as requiring that arch.

  - The unreachable-arm check reads one physical line, so a condition split across lines is skipped rather than analysed.
    Checked against the whole of src/: a line-joining variant finds exactly the same zero findings, because every multiline arch condition there is a standalone `if` with no `else if` chain below it.
    Widening the regex would add false-positive surface on the release path and buy nothing today, so it stays narrow and this is recorded as a known limitation rather than fixed.

  - Chains are grouped by BRACE DEPTH, not by indentation.
    This is an accuracy fix, not a widening: measured over all 181 src/**/*.cpp, depth and indentation report the same zero findings, so nothing new fires and the good tree stays clean, but depth keeps 339 arms in chains against 260 and is right in both directions where they differ.
    Indentation drops the enclosing chain at any nested `if`, which silences the check on the exact 08-27 shape, and it glues two unrelated `if`s at one indent into a single chain whenever the second opener is a skipped multiline condition, which reports a reachable arm as dead and blocks a release on good code.

  - The duplicate-key check compares keys by their source text, so it only looks at keys whose value cannot change between evaluations: literals, names, attributes and tuples of those. `{fresh(): 1, fresh(): 2}` reads as one key twice and is really two entries, and a finding here stops the nightly.

  - There is deliberately NO duplicate-C++-definition check.
    The obvious version keys on function name and flags legitimate overloads: it reported `llama_model_base::create_tensor`, which is two different signatures.
    A real duplicate is an ODR violation the compiler already rejects, so the only gain would be failing sooner, which does not justify a false positive on the release path.

Exits 0 when clean, 1 when anything is found.
`--report` emits JSON.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

ARCH = re.compile(r"arch == (LLM_ARCH_\w+)")
COND = re.compile(r"^\s*(?:\}\s*)?else if \((.*)\)\s*\{\s*$|^\s*if \((.*)\)\s*\{\s*$")
PURE_TERM = re.compile(r"arch == LLM_ARCH_\w+")
# Encoding prefixes a raw string may carry: LR"(...)", u8R"(...)" and so on.
_RAW_PREFIX = ("", "L", "u", "U", "u8")


# Node types whose value does not depend on when the expression is evaluated.
# An allowlist, not a denylist, so an expression shape nobody thought about is
# treated as unstable and simply not checked, rather than blocking a release.
_STABLE = (ast.Constant, ast.Name, ast.Attribute, ast.Tuple, ast.Load,
           ast.UnaryOp, ast.USub, ast.UAdd, ast.Invert)


def _stable_key(node: ast.AST) -> bool:
    """True when this key expression names the same object every evaluation.

    `{fresh(): 1, fresh(): 2}` unparses to the same text twice and is still two
    entries, so a call anywhere in the key means the two are not comparable by
    text. The keys this check exists for, `MODEL_ARCH.GLM5NEXT` and plain
    literals, are all stable.
    """
    return all(isinstance(n, _STABLE) for n in ast.walk(node))


def duplicate_dict_keys(path: Path) -> list[str]:
    """Keys defined twice in one dict literal. Always at best dead code."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as e:
        return [f"{path}:{e.lineno}: does not parse: {e.msg}"]
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        keys = [ast.unparse(k) for k in node.keys
                if k is not None and _stable_key(k)]
        for key, n in collections.Counter(keys).items():
            if n > 1:
                out.append(f"{path}:{node.lineno}: key {key} defined {n} times "
                           "in one dict; Python keeps only the last")
    return out


def _pure_arch_disjunction(cond: str) -> bool:
    """True when the condition is only `arch == X` terms joined by `||`."""
    if "&&" in cond:
        return False
    terms = [t.strip() for t in cond.split("||")]
    return bool(terms) and all(PURE_TERM.fullmatch(t) for t in terms)


def _raw_delim(text: str, i: int) -> str | None:
    """The delimiter of a raw string starting at `i`, or None if one does not.

    `i` indexes the `R`.
    The delimiter is what sits between `R"` and `(`, and the literal ends only at `)delim"`, which is the whole reason a raw string cannot be found with a plain regex.
    """
    if not text.startswith('R"', i):
        return None
    j = text.find("(", i + 2)
    if j == -1:
        return None
    delim = text[i + 2:j]
    if len(delim) > 16 or any(c in ' ()\\\t\n' for c in delim):
        return None
    # An `R` glued to an identifier is part of that identifier, not a prefix.
    k = i
    while k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_"):
        k -= 1
    return delim if text[k:i] in _RAW_PREFIX else None


def _decommented(text: str) -> list[str]:
    """The file with comments and literals blanked, line structure preserved.

    Braces inside a string literal or a comment are not braces.
    src/ is full of both (llama-chat.cpp alone embeds dozens of `{` in template strings), so counting them raw would desynchronise the depth for the rest of the file.

    Scanned once, left to right, rather than by substituting one construct at a time.
    Order cannot fix a substitution pass: blanking raw strings first lets an `R"(` written inside a comment swallow everything to the next `)"`, and blanking comments first lets a `//` inside a raw string end the line.
    Only position decides which construct is real, and a scan is what knows it.
    """
    blank = lambda s: re.sub(r"[^\n]", " ", s)            # noqa: E731
    out: list[str] = []
    i, n = 0, len(text)
    while i < n:
        delim = _raw_delim(text, i)
        if delim is not None:
            close = f'){delim}"'
            k = text.find(close, i + 2 + len(delim) + 1)
            end = n if k == -1 else k + len(close)
        elif text.startswith("//", i):
            k = text.find("\n", i)
            end = n if k == -1 else k
        elif text.startswith("/*", i):
            k = text.find("*/", i + 2)
            end = n if k == -1 else k + 2
        elif text[i] in "\"'":
            q, j = text[i], i + 1
            while j < n and text[j] != q and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            end = min(j + 1, n)
        else:
            out.append(text[i])
            i += 1
            continue
        out.append(blank(text[i:end]))
        i = end
    return "".join(out).split("\n")


def _depths(line: str, start: int) -> tuple[int, int]:
    """(brace depth after this line, lowest depth reached inside it)."""
    d = lo = start
    for ch in line:
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            lo = min(lo, d)
    return d, lo


def if_else_chains(text: str) -> list[list[tuple[int, str]]]:
    """Group `if` / `else if` conditions into chains by BRACE DEPTH.

    Indentation is not the structure.
    Keying chains on it, and resetting on any change, means a nested `if` inside an arm replaces the enclosing chain, so the outer arms after it are analysed as a fresh chain and an arch the outer chain already matched looks unmatched.
    That is a gate that stops gating on a shape that is ordinary C++: the arch dispatch at llama-model.cpp:2434 is exactly one nested `if` away from it.

    The same key also mis-JOINS.
    Two unrelated `if`s at the same indentation become one chain whenever the second one's opener is a condition the regex skips, and then a perfectly reachable arm is reported unreachable, which blocks a release on good code.
    Depth gets both right: a chain lives at the depth its `if` opened at, and ends when a brace takes the file back past it.
    """
    raw = text.split("\n")
    clean = _decommented(text)
    open_chains: dict[int, list[tuple[int, str]]] = {}
    done: list[list[tuple[int, str]]] = []

    def flush(key: int) -> None:
        c = open_chains.pop(key, None)
        if c and len(c) > 1:
            done.append(c)

    # A chain's closing brace and its `else if` are often on separate lines, which llama.cpp does in src/llama-quant.cpp:461 among others.
    # Closing the chain the moment the brace line dedents would end it one line before the arm that continues it, and the duplicate arch arm after it would then be a fresh chain with nothing taken yet, so the gate passes it.
    # Ending a chain is therefore deferred one line: the next line either continues it, or it really is over.
    # Blank lines do not decide either way.
    pending: set[int] = set()

    depth = 0
    for i, (line, cline) in enumerate(zip(raw, clean)):
        after, lo = _depths(cline, depth)
        stripped = cline.lstrip()
        # Matched on the decommented line: COND anchors on the `{` ending the line, so `if (arch == X) { // shared` matched nothing on the raw line and the arm vanished from the chain.
        # _decommented blanks in place, so the spans still index the raw line and the condition text below is taken from there, intact.
        # A line that blanked away entirely is commented-out code and opens nothing.
        m = COND.search(cline) if stripped else None
        # `} else if (...) {` and a bare `else if (...) {` after its own `}` line both continue the chain that lives at the depth this line dips to; a plain `if` opens one at the depth it starts from.
        cont = bool(m) and (stripped.startswith("}") or stripped.startswith("else"))
        key = lo if cont else depth
        if stripped:
            if cont:
                pending.discard(key)       # this line continues it after all
            for k in sorted(pending, reverse=True):
                flush(k)
            pending.clear()
        # Any chain whose closing brace this line just passed is over, unless the next line turns out to continue it.
        for k in [k for k in sorted(open_chains, reverse=True) if k >= lo]:
            if not (cont and k == key):
                pending.add(k)
        if m:
            g = 1 if m.group(1) is not None else 2
            cond = line[m.start(g):m.end(g)]
            if cont and key in open_chains:
                open_chains[key].append((i + 1, cond))
            else:
                flush(key)
                open_chains[key] = [(i + 1, cond)]
            # A line that both dedents and opens a chain at the same key would otherwise leave that key pending and flush the chain it just opened on the next line.
            pending.discard(key)
        depth = after
    for k in sorted(open_chains, reverse=True):
        flush(k)
    return done


def _blocked_by(cond: str, taken: set[str]) -> set[str]:
    """Arches that make this arm dead, given what earlier arms already took.

    A pure disjunction is dead only when EVERY alternative is taken.
    A plain conjunction is dead as soon as ONE of its `arch == X` conjuncts is, since the other conditions can only narrow it further: an earlier unconditional `arch == X` arm makes a later `arch == X && enabled` unreachable, and that arm is not a pure disjunction so it used to be skipped entirely.

    Anything mixing `||` and `&&` is left alone rather than guessed at, and only a term that is exactly `arch == X` counts, so `!(arch == X)` and `arch != X` cannot be read as requiring that arch.
    """
    archs = set(ARCH.findall(cond))
    if not archs:
        return set()
    if _pure_arch_disjunction(cond):
        return archs if archs <= taken else set()
    if "||" in cond:
        return set()
    hit = set()
    for term in cond.split("&&"):
        m = ARCH.fullmatch(term.strip().strip("()").strip())
        if m and m.group(1) in taken:
            hit.add(m.group(1))
    return hit


def unreachable_arch_arms(path: Path) -> list[str]:
    """`else if` arms whose arch a preceding pure-disjunction arm already took."""
    out = []
    for chain in if_else_chains(path.read_text()):
        taken: set[str] = set()
        for lineno, cond in chain:
            dead = _blocked_by(cond, taken)
            if dead:
                out.append(f"{path}:{lineno}: unreachable, "
                           f"{', '.join(sorted(dead))} already matched earlier "
                           "in this if/else chain")
            # Only an unconditional arm consumes an arch.
            # A conditional one may not run, so what follows it can still be reached.
            if _pure_arch_disjunction(cond):
                taken |= set(ARCH.findall(cond))
    return out


# ---------------------------------------------------------------- enum counts
#
# Observed 11-23: upstream b12635 added a ggml op and moved `static_assert(GGML_OP_COUNT == 101)` in
# ggml/include/ggml-rpc.h; our Inkling pins add GGML_OP_FLASH_ATTN_EXT_BANDED and had resolved that line to
# upstream's 101, so the merged tree had 102 ops. It merged clean and every gate built, because ggml-rpc.h is
# only compiled with GGML_RPC=ON, and the nightly died hours later in the CPU and Vulkan legs. A pin that adds
# an enum entry has to move every count assert in the tree with it, and a text merge cannot know that.
# Checked statically here so it costs well under a second and does not depend on which backends a build enables.

_COUNT_ASSERT = re.compile(
    r"static_assert\s*\(\s*(?:([A-Za-z_]\w*_COUNT)\s*==\s*(\d+)|(\d+)\s*==\s*([A-Za-z_]\w*_COUNT))\b")
_ENUM = re.compile(r"\benum\s+(?:class\s+)?(\w+)?\s*(?::\s*[\w:\s]+)?\{")
_SCAN_DIRS = ("ggml", "src", "common", "tools")
_SCAN_EXT = {".c", ".cpp", ".h", ".hpp", ".cu", ".cuh", ".m", ".mm"}


def _scan_files(root: Path):
    for d in _SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.suffix in _SCAN_EXT and p.is_file():
                yield p


def enum_members(code: str) -> dict[str, tuple[str, int | None]]:
    """`{member: (enum name, value)}` for every enum in decommented code.

    A value is known only while the members count up from literals: `A = 0, B, C = 5, D` gives 0, 1, 5, 6.
    After any non-literal initializer, or a preprocessor line inside the body, values are unknown (None),
    so a count that cannot be computed exactly is never compared rather than guessed.
    """
    out: dict[str, tuple[str, int | None]] = {}
    for m in _ENUM.finditer(code):
        close = code.find("}", m.end())
        if close == -1:
            continue
        body = code[m.end():close]
        name = m.group(1) or "<anonymous>"
        value: int | None = 0
        known = "#" not in body
        for item in body.split(","):
            item = item.strip()
            if not item:
                continue
            ident, _, init = item.partition("=")
            ident = ident.strip()
            if not re.fullmatch(r"[A-Za-z_]\w*", ident):
                known = False
                continue
            if init.strip():
                lit = init.strip()
                if known and re.fullmatch(r"-?\d+", lit):
                    value = int(lit)
                else:
                    known = False
            out[ident] = (name, value if known else None)
            if known:
                value += 1
    return out


def count_asserts(root: Path) -> tuple[list[tuple[Path, int, str, int]], dict[str, tuple[Path, str, int | None]]]:
    """Every `static_assert(X_COUNT == N)` in the tree, and where each X_COUNT is an enum member."""
    asserts: list[tuple[Path, int, str, int]] = []
    members: dict[str, tuple[Path, str, int | None]] = {}
    for p in _scan_files(root):
        raw = p.read_text(errors="replace")
        has_assert = "static_assert" in raw and "_COUNT" in raw
        has_enum = p.suffix in (".h", ".hpp", ".cuh") and "enum" in raw and "_COUNT" in raw
        if not (has_assert or has_enum):
            continue
        lines = _decommented(raw)
        code = "\n".join(lines)
        if has_assert:
            for i, line in enumerate(lines, 1):
                for m in _COUNT_ASSERT.finditer(line):
                    sym = m.group(1) or m.group(4)
                    n = int(m.group(2) or m.group(3))
                    asserts.append((p, i, sym, n))
        if has_enum:
            for ident, (ename, val) in enum_members(code).items():
                if ident.endswith("_COUNT"):
                    # a COUNT defined twice (two headers, or #if branches) is ambiguous: never compared
                    members[ident] = (p, ename, None) if ident in members else (p, ename, val)
    return asserts, members


def _enum_change_note(root: Path, header: Path, enum: str, base: str | None) -> str:
    """Which entries of `enum` the merged tree adds over `base`, and the merges that touched its header."""
    if not base:
        return ""
    rel = header.relative_to(root).as_posix()
    try:
        old = subprocess.run(["git", "-C", str(root), "show", f"{base}:{rel}"],
                             capture_output=True, text=True, timeout=20)
        log = subprocess.run(["git", "-C", str(root), "log", "--first-parent", "--format=%s",
                              f"{base}..HEAD", "--", rel], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    note = []
    if old.returncode == 0:
        before = {k for k, (e, _) in enum_members("\n".join(_decommented(old.stdout))).items() if e == enum}
        after = {k for k, (e, _) in enum_members(
            "\n".join(_decommented(header.read_text(errors="replace")))).items() if e == enum}
        added = sorted(after - before)
        if added:
            note.append(f"entries added over {base}: {', '.join(added)}")
    subjects = [s for s in log.stdout.splitlines() if s.strip()] if log.returncode == 0 else []
    if subjects:
        note.append(f"merges touching {rel}: {'; '.join(subjects[:8])}")
    return ("; " + "; ".join(note)) if note else ""


def enum_count_asserts(root: Path, base: str | None = None) -> tuple[list[str], int]:
    """`static_assert(X_COUNT == N)` whose N is not the enum's actual count. Returns (findings, asserts checked)."""
    asserts, members = count_asserts(root)
    out, checked = [], 0
    for path, line, sym, n in asserts:
        hdr, enum, actual = members.get(sym, (None, None, None))
        if actual is None:
            continue
        checked += 1
        if actual != n:
            rel = path.relative_to(root) if path.is_relative_to(root) else path
            out.append(f"{rel}:{line}: static_assert({sym} == {n}) but {enum} in "
                       f"{hdr.relative_to(root)} counts {actual}"
                       f"{_enum_change_note(root, hdr, enum, base)}")
    return out, checked


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="tree to check")
    ap.add_argument("--report", metavar="PATH", help="write a JSON summary here")
    ap.add_argument("--base", metavar="REF",
                    help="upstream base of the merged tree; names the enum entries and merges behind a count finding")
    a = ap.parse_args()
    root = Path(a.root)

    findings: list[str] = []
    scanned = {"python": 0, "cpp": 0, "count_asserts": 0}
    for p in sorted(root.glob("gguf-py/**/*.py")):
        scanned["python"] += 1
        findings += duplicate_dict_keys(p)
    for p in sorted(root.glob("src/**/*.cpp")):
        scanned["cpp"] += 1
        findings += unreachable_arch_arms(p)
    counts, scanned["count_asserts"] = enum_count_asserts(root, a.base)
    findings += counts

    print(f"merge_checks: scanned {scanned['python']} python and {scanned['cpp']} c++ files, "
          f"{scanned['count_asserts']} enum count asserts")
    for f in findings:
        print(f"  {f}")
    if a.report:
        Path(a.report).write_text(json.dumps(
            {"ok": not findings, "scanned": scanned, "findings": findings}, indent=2))
    if findings:
        print(f"merge_checks: {len(findings)} problem(s)", file=sys.stderr)
        return 1
    print("merge_checks: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
