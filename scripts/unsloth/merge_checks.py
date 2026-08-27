#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Post-merge checks for the two mistakes a clean build does not catch.

Both of these were made for real on 08-27, resolving GLM-5-Next against the
qwen4exp carry, and both survived compilation:

  1. A resolver unioned two byte-identical additions and produced the same
     `MODEL_ARCH.GLM5NEXT` key twice in the tensor map. Python keeps the last
     definition of a duplicate key, silently, so the file imports, the build
     passes, and the converter reads the wrong mapping.

  2. The same union kept both an arch in a shared fallthrough condition AND a
     dedicated `else if (arch == LLM_ARCH_GLM5NEXT)` arm below it. The shared
     condition matches first, so the dedicated arm is dead. It compiles, and the
     model runs with the indexer cache that arm was supposed to build.

Neither is a merge resolver. They decide nothing and rewrite nothing. They turn
a silent wrong answer into a loud one, which is the property that was missing.

Both are deliberately narrow, because a check that fires wrongly blocks a
release just as effectively as a bad merge:

  - The unreachable-arm check only fires when the EARLIER condition is a pure
    disjunction of `arch == LLM_ARCH_*` terms. If it contains `&&` or anything
    else, a later arm testing the same arch can genuinely be reachable, so the
    chain is skipped. With that restriction the verdict follows from the code:
    matching that arch always takes the earlier branch.

  - The unreachable-arm check reads one physical line, so a condition split
    across lines is skipped rather than analysed. Checked against the whole of
    src/: a line-joining variant finds exactly the same zero findings, because
    every multiline arch condition there is a standalone `if` with no `else if`
    chain below it. Widening the regex would add false-positive surface on the
    release path and buy nothing today, so it stays narrow and this is recorded
    as a known limitation rather than fixed.

  - Chains are grouped by BRACE DEPTH, not by indentation. This is an accuracy
    fix, not a widening: measured over all 181 src/**/*.cpp, depth and
    indentation report the same zero findings, so nothing new fires and the
    good tree stays clean, but depth keeps 339 arms in chains against 260 and
    is right in both directions where they differ. Indentation drops the
    enclosing chain at any nested `if`, which silences the check on the exact
    08-27 shape, and it glues two unrelated `if`s at one indent into a single
    chain whenever the second opener is a skipped multiline condition, which
    reports a reachable arm as dead and blocks a release on good code.

  - There is deliberately NO duplicate-C++-definition check. The obvious version
    keys on function name and flags legitimate overloads: it reported
    `llama_model_base::create_tensor`, which is two different signatures. A real
    duplicate is an ODR violation the compiler already rejects, so the only gain
    would be failing sooner, which does not justify a false positive on the
    release path.

Exits 0 when clean, 1 when anything is found. `--report` emits JSON.
"""

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import sys
from pathlib import Path

ARCH = re.compile(r"arch == (LLM_ARCH_\w+)")
COND = re.compile(r"^\s*(?:\}\s*)?else if \((.*)\)\s*\{\s*$|^\s*if \((.*)\)\s*\{\s*$")
PURE_TERM = re.compile(r"arch == LLM_ARCH_\w+")
# Blanked before braces are counted, so a `{` in a comment or a chat template
# string does not shift the depth for everything after it.
_BLOCK = re.compile(r"/\*.*?\*/", re.S)
_LINE = re.compile(r"//.*")
_STR = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'')


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
        keys = [ast.unparse(k) for k in node.keys if k is not None]
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


def _decommented(text: str) -> list[str]:
    """The file with comments and literals blanked, line structure preserved.

    Braces inside a string literal or a comment are not braces. src/ is full of
    both (llama-chat.cpp alone embeds dozens of `{` in template strings), so
    counting them raw would desynchronise the depth for the rest of the file.
    """
    text = _BLOCK.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    out = []
    for line in text.split("\n"):
        line = _STR.sub(lambda m: " " * len(m.group(0)), line)
        out.append(_LINE.sub(lambda m: " " * len(m.group(0)), line))
    return out


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

    Indentation is not the structure. Keying chains on it, and resetting on any
    change, means a nested `if` inside an arm replaces the enclosing chain, so
    the outer arms after it are analysed as a fresh chain and an arch the outer
    chain already matched looks unmatched. That is a gate that stops gating on
    a shape that is ordinary C++: the arch dispatch at llama-model.cpp:2434 is
    exactly one nested `if` away from it.

    The same key also mis-JOINS. Two unrelated `if`s at the same indentation
    become one chain whenever the second one's opener is a condition the regex
    skips, and then a perfectly reachable arm is reported unreachable, which
    blocks a release on good code. Depth gets both right: a chain lives at the
    depth its `if` opened at, and ends when a brace takes the file back past it.
    """
    raw = text.split("\n")
    clean = _decommented(text)
    open_chains: dict[int, list[tuple[int, str]]] = {}
    done: list[list[tuple[int, str]]] = []

    def flush(key: int) -> None:
        c = open_chains.pop(key, None)
        if c and len(c) > 1:
            done.append(c)

    # A chain's closing brace and its `else if` are often on separate lines,
    # which llama.cpp does in src/llama-quant.cpp:461 among others. Closing the
    # chain the moment the brace line dedents would end it one line before the
    # arm that continues it, and the duplicate arch arm after it would then be
    # a fresh chain with nothing taken yet, so the gate passes it. Ending a
    # chain is therefore deferred one line: the next line either continues it,
    # or it really is over. Blank lines do not decide either way.
    pending: set[int] = set()

    depth = 0
    for i, (line, cline) in enumerate(zip(raw, clean)):
        after, lo = _depths(cline, depth)
        stripped = cline.lstrip()
        # Matched on the decommented line: COND anchors on the `{` ending the
        # line, so `if (arch == X) { // shared` matched nothing on the raw line
        # and the arm vanished from the chain. _decommented blanks in place, so
        # the spans still index the raw line and the condition text below is
        # taken from there, intact. A line that blanked away entirely is
        # commented-out code and opens nothing.
        m = COND.search(cline) if stripped else None
        # `} else if (...) {` and a bare `else if (...) {` after its own `}`
        # line both continue the chain that lives at the depth this line dips
        # to; a plain `if` opens one at the depth it starts from.
        cont = bool(m) and (stripped.startswith("}") or stripped.startswith("else"))
        key = lo if cont else depth
        if stripped:
            if cont:
                pending.discard(key)       # this line continues it after all
            for k in sorted(pending, reverse=True):
                flush(k)
            pending.clear()
        # Any chain whose closing brace this line just passed is over, unless
        # the next line turns out to continue it.
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
            # A line that both dedents and opens a chain at the same key would
            # otherwise leave that key pending and flush the chain it just
            # opened on the next line.
            pending.discard(key)
        depth = after
    for k in sorted(open_chains, reverse=True):
        flush(k)
    return done


def unreachable_arch_arms(path: Path) -> list[str]:
    """`else if` arms whose arch a preceding pure-disjunction arm already took."""
    out = []
    for chain in if_else_chains(path.read_text()):
        taken: set[str] = set()
        for lineno, cond in chain:
            archs = set(ARCH.findall(cond))
            if archs and _pure_arch_disjunction(cond) and archs <= taken:
                out.append(f"{path}:{lineno}: unreachable, "
                           f"{', '.join(sorted(archs))} already matched earlier "
                           "in this if/else chain")
            if _pure_arch_disjunction(cond):
                taken |= archs
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="tree to check")
    ap.add_argument("--report", metavar="PATH", help="write a JSON summary here")
    a = ap.parse_args()
    root = Path(a.root)

    findings: list[str] = []
    scanned = {"python": 0, "cpp": 0}
    for p in sorted(root.glob("gguf-py/**/*.py")):
        scanned["python"] += 1
        findings += duplicate_dict_keys(p)
    for p in sorted(root.glob("src/**/*.cpp")):
        scanned["cpp"] += 1
        findings += unreachable_arch_arms(p)

    print(f"merge_checks: scanned {scanned['python']} python and {scanned['cpp']} c++ files")
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
