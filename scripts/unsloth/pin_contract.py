#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Assert the merged tree still contains what each pin carries.

The nightly proves the pins MERGED. That is not the same as proving they are in
the release, and the difference has cost us three outages:

  * ggml-org#28133 was squash-merged upstream. The pinned commit stopped being
    an ancestor of the base tag, so the merge was not a no-op -- it re-applied
    code the base already had. It happened to conflict, which is the only
    reason anybody noticed. A pin in that state that merges quietly ships
    nothing and nothing says so.
  * an additive resolution can keep the wrong side, or a later pin can land on
    top of an earlier one, and the arch registration the pin exists for is
    simply not in the tree any more. It still compiles.
  * a pin can rot into contributing nothing at all while its entry stays in
    pr-set.json for weeks.

So: derive from each pin's OWN diff what it puts in the tree, then check the
merged tree still has it. Nothing to maintain -- the expectation comes out of
the commit, so a repin regenerates it.

Four assertions per pin, cheapest first:

  symbols   every LLM_ARCH_/GGML_OP_/PROJECTOR_TYPE_/... name the pin
            introduces, in each file it introduces it to. Per FILE, not per
            tree: LLM_ARCH_INKLING surviving in llama-arch.h while its arm was
            dropped from llama-model.cpp is exactly the failure being looked
            for, and a tree-wide grep passes it.
  files     every file the pin adds still exists.
  lines     every non-comment code line the pin adds is still in that file.
            Catches a resolution that ate a hunk without touching a symbol.
  redundancy
            a pin whose added lines the BASE TAG already has is work upstream
            took. Reported, never fatal -- upstream landing a feature overnight
            must not stop that night's release.

What this CANNOT do, stated plainly so nobody reads more into a pass than is
there: the contract is re-derived from the pin, so it can only ever prove the
MERGE did not lose something. A regression inside the pin itself regenerates a
smaller contract that passes. Proving a feature works is feature_matrix.py's
job, and it needs a build.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reconcile_op_count import (  # noqa: E402
    FAMILIES, HEADER, SCAN_SUFFIXES, LayoutError, assert_re, blank_comments_and_strings, enum_count,
)

PIN_RE = re.compile(
    r"^https://github\.com/([^/]+)/llama\.cpp/pull/(\d+)/commits/([0-9a-f]{40})/?$"
)

# Identifier families that name a FEATURE. Deliberately not "every new symbol":
# a helper function renamed by a later upstream commit is not a lost feature,
# but a missing LLM_ARCH_ entry always is. These are the tables that decide
# whether an architecture, an op, a projector or a quant type exists at all.
SYMBOL_FAMILIES = (
    "LLM_ARCH_", "LLM_TENSOR_", "LLM_KV_", "LLM_TYPE_",
    "PROJECTOR_TYPE_", "GGML_OP_", "GGML_TYPE_", "LLAMA_FTYPE_",
)
SYMBOL_RE = re.compile(r"\b(?:" + "|".join(SYMBOL_FAMILIES) + r")[A-Z0-9_]+\b")

# The subset that names a whole feature rather than one of its tensors. Used
# only to keep --emit readable; the check itself uses all of SYMBOL_FAMILIES.
HEADLINE = ("LLM_ARCH_", "GGML_OP_", "GGML_TYPE_", "PROJECTOR_TYPE_", "LLAMA_FTYPE_")

# A line worth tracking for survival. Comments and short punctuation drift with
# every reformat and would make the check noise; a substantial code line does
# not move on its own.
TRIVIAL_RE = re.compile(r"^\s*(?://|/\*|\*|\*/|#\s|$)")
MIN_LINE = 12

# Comments are stripped before anything is read off a line. A pin that merely
# NAMES an arch in a comment has not registered it, and holding the comment's
# wording as a contract fails the moment upstream rewords it. Observed on
# unslothai#70, whose comment mentions GGML_OP_SSM_SCAN to explain why it does
# NOT use it.
COMMENT_RE = re.compile(r"//.*$|/\*.*?\*/|(?<!\S)#(?!\s*(?:include|define|if|el|endif|pragma)).*$")

# The op-count asserts belong to reconcile_op_count.py, which runs first and rewrites them to the
# counted enum when two pins each took the same N to N+1 (each pin's own line then survives in
# no merged tree), after proving the op tables match the enum. It is the check those lines stood for.
# It also raises RPC_PROTO_PATCH_VERSION when the pins' identical PATCH bumps merged into one.
# A pin's own op-count assert, as a whole line reconcile_op_count.py rewrites (any family, ggml/ only), is
# exempt; anything else on those lines is still owed. Its PATCH line stays in
# the contract and is satisfied by any PATCH at least the pin's under the same MAJOR.MINOR; for a pin
# that changes the GGML_OP_COUNT enum (the only delta reconcile_op_count.py raises PATCH from, and
# which it checks per pin itself) a newer MAJOR.MINOR also counts.
RPC_PATCH_RE = re.compile(r"^#\s*define\s+RPC_PROTO_PATCH_VERSION\b")
# A PATCH line the pin keeps owning still survives when reconcile_op_count.py raised PATCH past it
# (another pin's op bump merged into the same line): same MAJOR and MINOR, PATCH at least the pin's.
# Read off comment-blanked text, as reconcile_op_count.rpc_version reads it.
RPC_VERSION_RE = {p: re.compile(r"^[ \t]*#[ \t]*define[ \t]+RPC_PROTO_" + p + r"_VERSION[ \t]+(\d+)", re.M)
                  for p in ("MAJOR", "MINOR", "PATCH")}

# Binary and generated paths whose "lines" are meaningless.
SKIP_SUFFIXES = (".npy", ".png", ".jpg", ".gguf", ".bin", ".safetensors", ".ico", ".pdf")


class Failure(Exception):
    """A pin whose contract the merged tree does not satisfy."""


def git(args: list[str], cwd: Path | None = None, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:3])}...: {r.stderr.strip()[:300]}")
    return r.stdout


def blob(rev: str, path: str, cwd: Path) -> str | None:
    """The tree entry as "mode oid", or None if the rev has no such path.

    Lifted from carry_vintage.py, mode included for the same reason: a change
    that only chmods a file it otherwise took verbatim has identical content,
    and an oid-only comparison would call that "contributed nothing".
    """
    r = subprocess.run(["git", "ls-tree", "--full-tree", "-z", rev, "--", path],
                       cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        return None
    mode, _type, oid = r.stdout.split("\0")[0].split("\t", 1)[0].split()
    return f"{mode} {oid}"


def load_effective(prs_json: str) -> list[dict]:
    """The pin list the resolve step actually merged.

    Not the same as pr-set.json: resolve drops an optional pin once its PR is
    no longer open, and re-reading the file would then check a pin that is not
    in the tree and report it missing. The step already has the effective list
    as an output, so take it rather than recomputing the filter here and
    getting it subtly different.
    """
    return [{"url": p.get("url", ""), "src": p["repo"].split("/")[0],
             "num": int(p["number"]), "sha": p["sha"], "required": True}
            for p in json.loads(prs_json)]


def load_pins(pr_set: Path) -> list[dict]:
    data = json.loads(pr_set.read_text())
    pins = []
    for entry in data["prs"]:
        url = entry if isinstance(entry, str) else entry["url"]
        m = PIN_RE.match(url)
        if not m:
            raise SystemExit(f"malformed pin: {url}")
        pins.append({"url": url, "src": m.group(1), "num": int(m.group(2)),
                     "sha": m.group(3),
                     "required": True if isinstance(entry, str)
                     else entry.get("required", True)})
    return pins


def op_assert_lines(text: str) -> dict[str, str]:
    """Stripped lines of text holding op-count static_asserts -> what of the line is still owed.

    Exactly the asserts reconcile_op_count.py finds and rewrites (any family, comment-blanked), to the
    terminating `;`, so a continuation line holding the message is exempt along with the first. A line
    of nothing but asserts owes "" (exempt); code sharing the line owes itself, without the assert.
    """
    joined = blank_comments_and_strings(text)
    clean, hit = list(joined), [False] * len(joined)
    for count, _, _ in FAMILIES:
        for m in assert_re(count).finditer(joined):
            end = joined.find(";", m.end())
            for i in range(m.start(), end + 1 if end >= 0 else m.end()):
                hit[i] = True
                if clean[i] != "\n":
                    clean[i] = " "
    out, pos = {}, 0
    for line in text.split("\n"):
        span = range(pos, pos + len(line))
        if line.strip() and any(hit[i] for i in span):
            rest = "" if not "".join(clean[i] for i in span).strip() else \
                "".join(text[i] for i in span if not hit[i]).strip()
            out[line.strip()] = rest
        pos += len(line) + 1
    return out


def op_count(rev: str, cwd: Path) -> int | None:
    text = git(["show", f"{rev}:{HEADER}"], cwd, check=False)
    try:
        return enum_count(text, blank_comments_and_strings(text), "ggml_op", "GGML_OP_COUNT")[0]
    except LayoutError:
        return None


def derive(pin: dict, base: str, cwd: Path) -> dict:
    """What this pin puts in the tree, read off its own diff against the base.

    The fork point is merge-base(pin, base), not the pin's parent: a pin that
    has already had the base merged into it (which repin.py and every carry
    branch produce) would otherwise look like it contributed all of upstream.
    """
    fork = git(["merge-base", pin["sha"], base], cwd).strip()
    diff = git(["diff", "--no-renames", fork, pin["sha"]], cwd)

    symbols: dict[str, set[str]] = defaultdict(set)
    lines: dict[str, list[str]] = defaultdict(list)
    added: list[tuple[str, str]] = []
    cur = None
    for ln in diff.split("\n"):
        if ln.startswith("+++ b/"):
            cur = ln[6:]
        elif ln.startswith("+++ "):
            cur = None                       # /dev/null: a deletion
        elif cur and ln.startswith("+") and not ln.startswith("+++"):
            added.append((cur, ln[1:]))
    # reconcile_op_count.py rewrites every op-count assert (any family), so no line of a pin's own
    # assert is owed. The PATCH line is (see rpc_raised); main_op_pin is read off the enum itself, as
    # reconcile_op_count.py reads it, not off an assert line the pin may have only reworded.
    asserts = {path: op_assert_lines(git(["show", f"{pin['sha']}:{path}"], cwd, check=False))
               for path in {p for p, _ in added}
               if path.startswith("ggml/") and Path(path).suffix in SCAN_SUFFIXES}
    pin_n = op_count(pin["sha"], cwd)
    main_op_pin = pin_n is not None and pin_n != op_count(fork, cwd)
    for cur, text in added:
        stripped = text.strip()
        owed = asserts.get(cur, {}).get(stripped, stripped)
        if len(owed) >= MIN_LINE and not TRIVIAL_RE.match(owed):
            lines[cur].append(owed)
        code = COMMENT_RE.sub("", text).strip()
        if code:
            symbols[cur].update(SYMBOL_RE.findall(code))

    # Only symbols the base does not ALREADY have in that file are evidence of
    # this pin. Upstream naming an arch in a file the pin also touches is not
    # something the pin is owed.
    new_symbols: dict[str, list[str]] = {}
    for path, names in symbols.items():
        fresh = sorted(n for n in names
                       if n not in git(["show", f"{base}:{path}"], cwd, check=False))
        if fresh:
            new_symbols[path] = fresh

    status = git(["diff", "--name-status", "--no-renames", fork, pin["sha"]], cwd)
    added, owned = [], []
    for ln in status.split("\n"):
        if not ln.strip():
            continue
        code, path = ln.split("\t", 1)
        owned.append(path)
        if code.startswith("A"):
            added.append(path)

    return {
        "fork": fork,
        "symbols": new_symbols,
        "added_files": added,
        "owned_paths": owned,
        "main_op_pin": main_op_pin,
        "lines": {p: v for p, v in lines.items()
                  if not p.endswith(SKIP_SUFFIXES)},
    }


def redundancy(contract: dict, base: str, cwd: Path) -> tuple[int, int]:
    """How much of what this pin adds the base tag already has.

    This is the pr-set.json retirement rule, mechanised: "delete the entry once
    a base tag carries the work". Upstream almost always SQUASHES, so the
    pinned commit never becomes an ancestor and no ancestry test will ever say
    the work landed; comparing the text is the only thing that can.

    Measured on the real set at b10775, the separation is not close: the pin
    that upstream had already absorbed (ggml-org#28133) scored 99%, and the
    highest live pin scored 33%.
    """
    total = hit = 0
    for path, wanted in contract["lines"].items():
        text = git(["show", f"{base}:{path}"], cwd, check=False)
        total += len(wanted)
        hit += sum(1 for w in wanted if w in text)
    return hit, total


def rpc_raised(pin_text: str, text: str, allow_newer_series: bool = False) -> bool:
    """text has the pin's RPC_PROTO MAJOR and MINOR and a PATCH at least the pin's.

    allow_newer_series: a higher MAJOR.MINOR also counts (a pin that changes the op enum, whose bump
    reconcile_op_count.py checks per pin and re-derives against the base's version).
    """
    v = [{p: (m.group(1) if (m := r.search(t)) else None) for p, r in RPC_VERSION_RE.items()}
         for t in map(blank_comments_and_strings, (pin_text, text))]
    series = [(v[i]["MAJOR"], v[i]["MINOR"]) for i in (0, 1)]
    if (allow_newer_series and None not in series[0] + series[1]
            and tuple(map(int, series[1])) > tuple(map(int, series[0]))):
        return True
    return (v[0]["PATCH"] is not None and v[1]["PATCH"] is not None
            and all(v[0][p] == v[1][p] for p in ("MAJOR", "MINOR"))
            and int(v[1]["PATCH"]) >= int(v[0]["PATCH"]))


def check(pin: dict, contract: dict, root: Path, base: str, cwd: Path,
          threshold: float) -> list[str]:
    problems = []

    for path, names in sorted(contract["symbols"].items()):
        target = root / path
        text = target.read_text(errors="replace") if target.is_file() else ""
        for name in names:
            if name not in text:
                problems.append(
                    f"{name} is missing from {path}; the pin adds it there and "
                    "the merged tree does not have it")

    for path in contract["added_files"]:
        if not (root / path).exists():
            problems.append(f"{path} is added by the pin and missing from the merged tree")

    for path, wanted in sorted(contract["lines"].items()):
        target = root / path
        if not target.is_file():
            continue                          # already reported, or a deletion
        text = target.read_text(errors="replace")
        lost = [w for w in wanted if w not in text]
        if any(RPC_PATCH_RE.match(w) for w in lost) and rpc_raised(
                git(["show", f"{pin['sha']}:{path}"], cwd, check=False), text,
                allow_newer_series=contract.get("main_op_pin", False)):
            lost = [w for w in lost if not RPC_PATCH_RE.match(w)]
        if not lost:
            continue
        kept = len(wanted) - len(lost)
        ratio = kept / len(wanted)
        if ratio < threshold:
            problems.append(
                f"{path} kept {kept}/{len(wanted)} of the lines this pin adds "
                f"({ratio:.0%}); first missing: {lost[0][:90]}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=".", help="the merged tree to check")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pr-set", help="scripts/unsloth/pr-set.json")
    src.add_argument("--prs-json", help="the resolve step's `prs` output: the pins it "
                                        "actually merged, optional ones already dropped")
    ap.add_argument("--base", required=True, help="upstream base tag the mix was built on")
    ap.add_argument("--git-dir", help="repo the pin commits are reachable from "
                                      "(default: --root)")
    ap.add_argument("--threshold", type=float, default=1.0,
                    help="fraction of a pin's added lines that must survive per file")
    ap.add_argument("--redundant-at", type=float, default=0.95,
                    help="report a pin whose added lines the base tag already has "
                         "at this fraction or more (never fatal)")
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--emit", action="store_true",
                    help="print the derived contracts and check nothing")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cwd = Path(args.git_dir).resolve() if args.git_dir else root
    pins = (load_pins(Path(args.pr_set)) if args.pr_set
            else load_effective(args.prs_json))

    report: dict = {"base": args.base, "ok": False, "pins": [], "notices": []}
    failed = 0
    notices: list[str] = []

    for pin in pins:
        name = f"{pin['src']}#{pin['num']}"
        try:
            contract = derive(pin, args.base, cwd)
        except RuntimeError as e:
            report["pins"].append({"pin": name, "sha": pin["sha"], "problems": [str(e)]})
            print(f"ERROR {name}: {e}", file=sys.stderr)
            failed += 1
            continue

        entry = {
            "pin": name,
            "sha": pin["sha"],
            "fork": contract["fork"],
            "symbols": contract["symbols"],
            "added_files": contract["added_files"],
            "line_count": sum(len(v) for v in contract["lines"].values()),
            "problems": [],
        }

        if args.emit:
            report["pins"].append(entry)
            # Only the families that NAME a feature are printed. Every symbol
            # is still checked; a new file legitimately contributes a hundred
            # LLM_TENSOR_ names and listing them buries the one that matters.
            sym = sorted({s for v in contract["symbols"].values() for s in v
                          if s.startswith(HEADLINE)})
            print(f"{name:>18}  {entry['line_count']:>5} lines, "
                  f"{len(contract['added_files'])} new files, symbols: "
                  f"{', '.join(sym) if sym else '-'}")
            continue

        problems = check(pin, contract, root, args.base, cwd, args.threshold)
        hit, total = redundancy(contract, args.base, cwd)
        entry["problems"] = problems
        entry["redundant_lines"] = [hit, total]

        if total and hit / total >= args.redundant_at:
            note = (f"the base tag already has {hit}/{total} ({hit / total:.0%}) of the "
                    "lines this pin adds; upstream has taken this work and the entry "
                    "should be deleted from pr-set.json")
            entry["notices"] = [note]
            notices.append(f"{name}: {note}")

        report["pins"].append(entry)
        if problems:
            failed += 1
            print(f"FAIL {name}", file=sys.stderr)
            for p in problems:
                print(f"     {p}", file=sys.stderr)
        else:
            print(f"ok   {name}: {len(contract['symbols'])} file(s) with new symbols, "
                  f"{entry['line_count']} line(s) accounted for")

    report["ok"] = failed == 0 or args.emit
    report["notices"] = notices
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if args.emit:
        return 0

    # Notices after the verdict lines, never mixed into them: "upstream took
    # this, drop the entry" is housekeeping and must not read as a failure.
    for n in notices:
        print(f"note {n}")
    if failed:
        print(f"\n{failed} pin(s) are not intact in the merged tree", file=sys.stderr)
        return 1
    print(f"\nall {len(pins)} pins are intact in the merged tree"
          + (f", {len(notices)} can be retired" if notices else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
