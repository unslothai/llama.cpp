#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Prove an upstream-sync merge is additive only: it may bring in upstream content, but it must
never modify, revert or delete anything this fork authored.

The invariant, stated precisely. Let

    B = merge-base(fork_master, upstream_master)
    F = fork_master        (the fork before the sync)
    U = upstream_master
    M = the merge commit under test

For every path P that the fork touched in B..F, the merge must satisfy M:P == F:P, byte for
byte. That is the whole rule, and it is checked exactly rather than approximated by reading a
diff. Three things can break it and each is reported separately:

    MODIFIED  M:P exists but differs from F:P. Upstream edited a file we own, or a conflict
              was resolved in upstream's favour.
    DELETED   P is in F and gone in M. Our work was dropped.
    LOSTCOMMIT  a commit reachable from F is not reachable from M. History was rewritten.

Deletions of upstream files are legitimate and are checked in the other direction: every path
missing from M must ALSO have been missing from F. A file the fork already deleted staying
deleted preserves our state; a file the fork had that vanishes is a violation.

Two extra layers beyond the byte comparison, because a byte-identical file can still be
semantically wrong if a neighbouring definition moved:

  * AST check on the Python surface (gguf-py and convert scripts). Every top-level class,
    function and assignment name the fork defines must still be defined in the merged tree,
    and every enum member the fork added must still carry the same value. A renumbered
    GGML_TYPE_* would be caught here even if the file "looks" additive.
  * enum-value check on the C headers, by regex over the id assignments that matter, since a
    silently renumbered type id is the failure mode that would corrupt published GGUFs.

Usage:
    verify_upstream_sync.py --repo <path> --merge <rev> [--fork origin/master]
                            [--upstream upstream/master] [--json out.json]

Exit 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict


def git(repo: str, *args: str, ok_fail: bool = False) -> str:
    p = subprocess.run(("git", "-C", repo) + args, capture_output=True, text=True)
    if p.returncode and not ok_fail:
        raise SystemExit(f"git {' '.join(args)} failed:\n{p.stderr.strip()}")
    return p.stdout

def lines(s: str) -> list[str]:
    return [x for x in s.splitlines() if x.strip()]

def blob(repo: str, rev: str, path: str) -> bytes | None:
    """File content at a revision, or None if the path does not exist there."""
    p = subprocess.run(("git", "-C", repo, "show", f"{rev}:{path}"),
                       capture_output=True)
    return p.stdout if p.returncode == 0 else None


# ---------------------------------------------------------------- AST surface

def py_surface(src: bytes) -> dict[str, str]:
    """Top-level names a Python file defines, plus every enum-ish member and its literal value.

    Keys are dotted so a member cannot collide across classes. Values are a repr of the
    assigned constant where there is one, else the node type, so a renumber shows up as a
    changed value rather than a missing name.
    """
    try:
        tree = ast.parse(src.decode("utf-8", "replace"))
    except SyntaxError:
        return {}

    out: dict[str, str] = {}

    def const(node: ast.AST) -> str:
        """A value fingerprint that survives reformatting but not a real change.

        ast.unparse normalises whitespace, quote style and line breaks, so a reflow is
        invisible while an edited element is not. Falling back to the node type name would
        make every tuple, list and dict compare equal, which is how a dropped entry such as
        OWNED = ("a", "b") -> ("a",) slips past.
        """
        if isinstance(node, ast.Constant):
            return repr(node.value)
        try:
            return ast.unparse(node)
        except Exception:
            return type(node).__name__

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for n in body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out[f"{prefix}{n.name}"] = "def"
            elif isinstance(n, ast.ClassDef):
                out[f"{prefix}{n.name}"] = "class"
                walk(n.body, f"{prefix}{n.name}.")
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name):
                        out[f"{prefix}{t.id}"] = const(n.value)
            elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
                out[f"{prefix}{n.target.id}"] = const(n.value) if n.value else "ann"
    walk(tree.body, "")
    return out


# --------------------------------------------------------- C enum value check

C_ENUM = re.compile(
    rb"^\s*(GGML_TYPE_[A-Z0-9_]+|GGML_FTYPE_[A-Z0-9_]+|LLAMA_FTYPE_[A-Z0-9_]+)\s*=\s*(-?\d+)",
    re.M)

def c_enum_values(src: bytes) -> dict[str, int]:
    return {m.group(1).decode(): int(m.group(2)) for m in C_ENUM.finditer(src)}


# --------------------------------------------------------------------- checks

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--merge", required=True, help="the merge commit under test")
    ap.add_argument("--fork", default="origin/master", help="fork state BEFORE the sync")
    ap.add_argument("--upstream", default="upstream/master")
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--show", type=int, default=15, help="max paths to print per category")
    a = ap.parse_args()

    R = a.repo
    if git(R, "rev-parse", "--is-shallow-repository").strip() == "true":
        print("REFUSING: shallow clone, history checks would be meaningless. "
              "Re-clone without --depth.", file=sys.stderr)
        return 2

    M = git(R, "rev-parse", a.merge).strip()
    F = git(R, "rev-parse", a.fork).strip()
    U = git(R, "rev-parse", a.upstream).strip()
    B = git(R, "merge-base", a.fork, a.upstream).strip()
    print(f"merge      {M[:12]}\nfork       {F[:12]}  ({a.fork})\n"
          f"upstream   {U[:12]}  ({a.upstream})\nmerge-base {B[:12]}\n")

    rep: dict = {"merge": M, "fork": F, "upstream": U, "base": B, "checks": {}}
    fail = False

    # --- 1. no fork commit may be dropped -----------------------------------
    lost = lines(git(R, "rev-list", a.fork, f"^{M}"))
    fork_commits = len(lines(git(R, "rev-list", f"{B}..{a.fork}")))
    rep["checks"]["history"] = {"fork_commits_since_base": fork_commits,
                                "lost": lost}
    if lost:
        fail = True
        print(f"FAIL  history: {len(lost)} fork commits are NOT ancestors of the merge")
        for c in lost[:a.show]:
            print(f"        {c[:12]}  {git(R, 'log', '-1', '--format=%s', c).strip()[:70]}")
    else:
        print(f"PASS  history: all {fork_commits} fork commits since the base are ancestors "
              f"of the merge, none dropped")

    # --- 2. every path the fork touched must survive byte-identical ---------
    # --diff-filter with -M off: a rename upstream must not silently "move" our file.
    touched = sorted(set(lines(git(R, "diff", "--name-only", "--no-renames", f"{B}..{a.fork}"))))
    modified, deleted, ok = [], [], 0
    for p in touched:
        fb = blob(R, a.fork, p)
        mb = blob(R, M, p)
        if fb is None:
            # the fork itself deleted it; it must still be absent
            if mb is not None:
                modified.append((p, "fork deleted it, merge resurrected it"))
            else:
                ok += 1
            continue
        if mb is None:
            deleted.append(p)
        elif mb != fb:
            modified.append((p, f"{len(fb)} B -> {len(mb)} B"))
        else:
            ok += 1
    rep["checks"]["content"] = {"fork_touched_paths": len(touched), "identical": ok,
                                "modified": modified, "deleted": deleted}
    if modified or deleted:
        fail = True
        print(f"FAIL  content: of {len(touched)} fork-touched paths, "
              f"{len(modified)} modified and {len(deleted)} deleted")
        for p, why in modified[:a.show]:
            print(f"        MODIFIED  {p}  ({why})")
        for p in deleted[:a.show]:
            print(f"        DELETED   {p}")
    else:
        print(f"PASS  content: all {len(touched)} paths the fork touched are byte-identical "
              f"in the merge")

    # --- 3. nothing the fork had may vanish, even if it never touched it -----
    fork_tree = set(lines(git(R, "ls-tree", "-r", "--name-only", a.fork)))
    merge_tree = set(lines(git(R, "ls-tree", "-r", "--name-only", M)))
    vanished = sorted(fork_tree - merge_tree)
    # a vanished path is only acceptable if upstream deleted it AND the fork never touched it
    unexplained = [p for p in vanished if p in set(touched)]
    rep["checks"]["tree"] = {"fork_files": len(fork_tree), "merge_files": len(merge_tree),
                             "vanished": vanished, "unexplained": unexplained}
    if unexplained:
        fail = True
        print(f"FAIL  tree: {len(unexplained)} fork-authored files vanished from the merge")
        for p in unexplained[:a.show]:
            print(f"        {p}")
    else:
        print(f"PASS  tree: {len(fork_tree)} fork files -> {len(merge_tree)} merged files, "
              f"{len(vanished)} vanished and none of them fork-authored")

    # --- 4. every deletion must be upstream's own, never ours ---------------
    # Upstream retires its own files and a sync has to carry that through, so a deletion is
    # only a violation if the file is one WE own. Two independent tests, both must hold:
    # the path is absent from upstream's tree (upstream really did delete it), and the fork
    # never touched it.
    gone = sorted(set(lines(git(R, "diff", "--name-only", "--diff-filter=D",
                               f"{a.fork}..{M}"))))
    touched_set = set(touched)
    ours, not_upstream = [], []
    for p in gone:
        if p in touched_set:
            ours.append(p)
        elif blob(R, a.upstream, p) is not None:
            not_upstream.append(p)      # still exists upstream, so nobody asked us to drop it
    rep["checks"]["deletions"] = {"deleted_vs_fork": gone, "fork_authored": ours,
                                  "still_present_upstream": not_upstream}
    if ours or not_upstream:
        fail = True
        print(f"FAIL  deletions: {len(gone)} deletions, {len(ours)} fork-authored, "
              f"{len(not_upstream)} not deleted upstream either")
        for p in (ours + not_upstream)[:a.show]:
            print(f"        {p}")
    else:
        print(f"PASS  deletions: all {len(gone)} deletions are upstream retiring its own "
              f"files, none fork-authored")

    # --- 5. AST surface on Python the fork owns -----------------------------
    pyfiles = [p for p in touched if p.endswith(".py")]
    lost_names: list[tuple[str, str, str, str]] = []
    for p in pyfiles:
        fb, mb = blob(R, a.fork, p), blob(R, M, p)
        if fb is None or mb is None:
            continue
        fs, ms = py_surface(fb), py_surface(mb)
        for name, val in fs.items():
            if name not in ms:
                lost_names.append((p, name, val, "MISSING"))
            elif ms[name] != val:
                lost_names.append((p, name, val, ms[name]))
    rep["checks"]["ast"] = {"python_files": len(pyfiles), "regressions": lost_names}
    if lost_names:
        fail = True
        print(f"FAIL  ast: {len(lost_names)} Python definitions lost or changed value")
        for p, n, was, now in lost_names[:a.show]:
            print(f"        {p}:{n}  was {was}  now {now}")
    else:
        print(f"PASS  ast: every top-level definition and constant in the {len(pyfiles)} "
              f"fork-touched Python files survives with the same value")

    # --- 6. C enum ids, the failure that would corrupt published GGUFs ------
    # Deliberately NOT limited to fork-touched headers. A published GGUF stores these ids, so
    # an id that shifts or collides is a data-corruption bug no matter who moved it, and on a
    # fork whose master carries no source changes the fork-touched set would be empty.
    # The reference to compare against depends on who owns the header. For one the fork
    # customises, our values must survive. For one the fork does not, the merged copy must
    # match UPSTREAM exactly, and upstream growing an enum (a new type, so a new _COUNT) is
    # the sync working, not a violation. Comparing an unowned header against the fork's stale
    # copy would flag every legitimate upstream addition.
    ID_HEADERS = ["ggml/include/ggml.h", "include/llama.h"]
    hdrs = sorted(set([p for p in touched if p.endswith((".h", ".hpp"))]) |
                  {p for p in ID_HEADERS if blob(R, M, p) is not None})
    enum_bad: list[tuple[str, str, int, object]] = []
    dupes: list[tuple[str, int, list[str]]] = []
    owned = set(touched)
    for p in hdrs:
        ref = a.fork if p in owned else a.upstream
        rb, mb = blob(R, ref, p), blob(R, M, p)
        if rb is None or mb is None:
            continue
        fe, me = c_enum_values(rb), c_enum_values(mb)
        for name, v in fe.items():
            now = me.get(name, "MISSING")
            if now == v:
                continue
            # A _COUNT sentinel is not an id, it is one past the last one, so a branch that
            # legitimately adds types must move it. Allow it to grow and require it to still
            # bound every real id; anything else, including a shrink, is a violation.
            if name.endswith("_COUNT") and isinstance(now, int) and now > v:
                fam = name[:-len("_COUNT")]
                real = [x for n2, x in me.items() if n2.startswith(fam) and n2 != name]
                if real and now > max(real):
                    continue
            enum_bad.append((f"{p} [vs {ref}]", name, v, now))
        # two names sharing one id in the same enum family is a collision
        byfam: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for name, v in me.items():
            fam = name.split("_")[0] + ("_FTYPE" if "_FTYPE_" in name else "_TYPE")
            byfam[fam][v].append(name)
        for fam, vals in byfam.items():
            for v, names in vals.items():
                if len(names) > 1 and not any(n.endswith("_COUNT") for n in names):
                    dupes.append((f"{p}:{fam}", v, sorted(names)))
    rep["checks"]["c_enums"] = {"headers": len(hdrs), "changed": enum_bad, "collisions": dupes}
    if enum_bad or dupes:
        fail = True
        print(f"FAIL  c_enums: {len(enum_bad)} ids changed, {len(dupes)} id collisions")
        for p, n, was, now in enum_bad[:a.show]:
            print(f"        {p}:{n}  was {was}  now {now}")
        for where, v, names in dupes[:a.show]:
            print(f"        COLLISION {where} = {v}: {', '.join(names)}")
    else:
        print(f"PASS  c_enums: every GGML_TYPE/GGML_FTYPE/LLAMA_FTYPE id the fork defines "
              f"keeps its value across {len(hdrs)} headers, no collisions")

    # --- context: what the merge actually brought in ------------------------
    added = len(lines(git(R, "diff", "--name-only", "--diff-filter=A", f"{a.fork}..{M}")))
    changed = len(lines(git(R, "diff", "--name-only", f"{a.fork}..{M}")))
    rep["summary"] = {"files_added_by_merge": added, "files_changed_by_merge": changed,
                      "pass": not fail}
    print(f"\nmerge brings in {changed} changed files, {added} of them new")

    if a.json:
        with open(a.json, "w") as f:
            json.dump(rep, f, indent=1)
        print(f"report: {a.json}")

    print("\n" + ("VIOLATION: the sync is not additive-only" if fail
                  else "PASS: the sync is additive only, nothing fork-authored was altered"))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
