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


def unreachable_arch_arms(path: Path) -> list[str]:
    """`else if` arms whose arch a preceding pure-disjunction arm already took."""
    lines = path.read_text().split("\n")
    chains: list[list[tuple[int, str]]] = []
    cur: list[tuple[int, str]] = []
    cur_indent = None
    for i, line in enumerate(lines):
        m = COND.search(line)
        if not m:
            continue
        indent = len(line) - len(line.lstrip())
        cond = m.group(1) or m.group(2) or ""
        if line.lstrip().startswith("}") and cur_indent == indent:
            cur.append((i + 1, cond))
        else:
            if len(cur) > 1:
                chains.append(cur)
            cur, cur_indent = [(i + 1, cond)], indent
    if len(cur) > 1:
        chains.append(cur)

    out = []
    for chain in chains:
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
