#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for pin_contract.py. Run: python3 scripts/unsloth/test_pin_contract.py

Every case builds a real repository with a real base tag, a real pin branch and
a real merge, then damages the merged tree the way a bad resolution damages it.
A hand-written fixture would only prove the checker reads its own output format.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "pin_contract.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                          cwd=repo, capture_output=True, text=True)


ARCH_H_BASE = """\
enum llm_arch {
    LLM_ARCH_LLAMA,
    LLM_ARCH_UNKNOWN,
};
"""
MODEL_CPP_BASE = """\
void build_model(llm_arch arch) {
    switch (arch) {
        case LLM_ARCH_LLAMA:
            build_llama();
            break;
    }
}
"""


def make_repo():
    """A base tag `b1` plus a pin branch adding one architecture, merged."""
    d = Path(tempfile.mkdtemp(prefix="pc_"))
    git(d, "init", "-q", "-b", "main")
    (d / "src").mkdir()
    (d / "src" / "llama-arch.h").write_text(ARCH_H_BASE)
    (d / "src" / "llama-model.cpp").write_text(MODEL_CPP_BASE)
    git(d, "add", "-A"); git(d, "commit", "-qm", "base")
    git(d, "tag", "b1")

    git(d, "checkout", "-qb", "pin")
    (d / "src" / "llama-arch.h").write_text(
        ARCH_H_BASE.replace("    LLM_ARCH_UNKNOWN,",
                            "    LLM_ARCH_INKLING,\n    LLM_ARCH_UNKNOWN,"))
    (d / "src" / "llama-model.cpp").write_text(
        MODEL_CPP_BASE.replace("    }\n}",
                               "        case LLM_ARCH_INKLING:\n"
                               "            build_inkling_with_banded_bias();\n"
                               "            break;\n    }\n}"))
    (d / "src" / "inkling.cpp").write_text(
        "void build_inkling_with_banded_bias() { do_the_banded_thing(); }\n")
    git(d, "add", "-A"); git(d, "commit", "-qm", "add inkling")
    sha = git(d, "rev-parse", "HEAD").stdout.strip()

    git(d, "checkout", "-q", "main")
    git(d, "merge", "-q", "--no-ff", "--no-edit", "-m", "merge pin", "pin")

    # Outside the work tree on purpose: a test that commits after this would
    # otherwise sweep the pin file into the pin's own diff.
    pr_set = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
    pr_set.write_text(json.dumps({"prs": [
        f"https://github.com/unslothai/llama.cpp/pull/1/commits/{sha}"]}))
    return d, pr_set, sha


def run(repo, pr_set, *extra):
    rep = repo / "r.json"
    p = subprocess.run([sys.executable, str(SCRIPT), "--root", str(repo),
                        "--pr-set", str(pr_set), "--base", "b1",
                        "--report", str(rep), *extra],
                       capture_output=True, text=True)
    return p.returncode, (json.loads(rep.read_text()) if rep.exists() else {}), p.stderr


# --- 1. an intact merge passes -------------------------------------------
repo, pr_set, sha = make_repo()
rc, rep, err = run(repo, pr_set)
check("intact merge passes", rc == 0 and rep["ok"], err)
check("intact merge finds the new arch",
      "LLM_ARCH_INKLING" in json.dumps(rep["pins"][0]["symbols"]), rep)
check("intact merge reports no notices", rep["notices"] == [], rep)

# --- 2. the arm is dropped from ONE file: a tree-wide grep would pass ------
# The real shape: LLM_ARCH_INKLING survives in the enum and the dispatch arm
# that makes it do anything is gone.
repo, pr_set, sha = make_repo()
p = repo / "src" / "llama-model.cpp"
p.write_text(MODEL_CPP_BASE)
rc, rep, err = run(repo, pr_set)
check("a dropped dispatch arm fails", rc == 1 and not rep["ok"], err)
check("the failure names the file, not just the symbol",
      any("llama-model.cpp" in x for x in rep["pins"][0]["problems"]), rep)
check("the enum copy of the symbol does not rescue it",
      "LLM_ARCH_INKLING" in (repo / "src" / "llama-arch.h").read_text())

# --- 3. a whole added file goes missing ----------------------------------
repo, pr_set, sha = make_repo()
(repo / "src" / "inkling.cpp").unlink()
rc, rep, err = run(repo, pr_set)
check("a missing added file fails", rc == 1, err)
check("the failure names the file",
      any("inkling.cpp" in x for x in rep["pins"][0]["problems"]), rep)

# --- 4. a hunk is eaten without touching a symbol -------------------------
repo, pr_set, sha = make_repo()
(repo / "src" / "inkling.cpp").write_text(
    "void build_inkling_with_banded_bias() { }\n")   # body gone, name kept
rc, rep, err = run(repo, pr_set)
check("an eaten body fails on line survival", rc == 1, err)
check("line survival names what went missing",
      any("do_the_banded_thing" in x for x in rep["pins"][0]["problems"]), rep)

# --- 5. redundancy: the base already has everything the pin adds ----------
# Built the way it happens for real: upstream lands the same work, so the base
# tag has it and the pin is not an ancestor of anything.
d = Path(tempfile.mkdtemp(prefix="pc_"))
git(d, "init", "-q", "-b", "main")
(d / "src").mkdir()
(d / "src" / "f.cpp").write_text("int a() { return 1; }\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "root")
git(d, "checkout", "-qb", "pin")
(d / "src" / "f.cpp").write_text(
    "int a() { return 1; }\nint the_new_helper() { return 42; }\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "pin work")
sha5 = git(d, "rev-parse", "HEAD").stdout.strip()
git(d, "checkout", "-q", "main")
(d / "src" / "f.cpp").write_text(                    # upstream squashed the same work
    "int a() { return 1; }\nint the_new_helper() { return 42; }\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "upstream squash of the same change")
git(d, "tag", "b1")
ps5 = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
ps5.write_text(json.dumps({"prs": [
    f"https://github.com/unslothai/llama.cpp/pull/1/commits/{sha5}"]}))
rc, rep, err = run(d, ps5)
check("a pin the base already carries is reported", rep["notices"], rep)
check("redundancy says to delete the entry",
      "deleted from pr-set.json" in " ".join(rep["notices"]), rep)
check("redundancy is NOT fatal", rc == 0, err)

# --- 6. --emit checks nothing ---------------------------------------------
repo, pr_set, sha = make_repo()
(repo / "src" / "inkling.cpp").unlink()
rc, rep, err = run(repo, pr_set, "--emit")
check("--emit does not check", rc == 0 and rep["ok"], err)
check("--emit still derives the contract",
      rep["pins"][0]["added_files"] == ["src/inkling.cpp"], rep)

# --- 7. a comment is not a contract ---------------------------------------
# unslothai#70 has a comment naming GGML_OP_SSM_SCAN to say it does NOT use it.
# Holding comment wording would fail the moment upstream rewords it.
repo, pr_set, sha = make_repo()
git(repo, "checkout", "-q", "pin")
(repo / "src" / "note.cpp").write_text(
    "// unlike LLM_ARCH_MISTRAL this one does its own thing\nint g() { return 0; }\n")
git(repo, "add", "-A"); git(repo, "commit", "-qm", "comment")
sha7 = git(repo, "rev-parse", "HEAD").stdout.strip()
git(repo, "checkout", "-q", "main")
git(repo, "merge", "-q", "--no-ff", "--no-edit", "-m", "m2", "pin")
(repo / "src" / "note.cpp").write_text(              # comment reworded, code kept
    "// this one does its own thing\nint g() { return 0; }\n")
pr_set.write_text(json.dumps({"prs": [
    f"https://github.com/unslothai/llama.cpp/pull/1/commits/{sha7}"]}))
rc, rep, err = run(repo, pr_set)
check("a reworded comment does not fail the pin", rc == 0, err)
check("no symbol was harvested from the comment",
      "LLM_ARCH_MISTRAL" not in json.dumps(rep["pins"][0]["symbols"]), rep)

# --- 8. an op-count assert rewritten by reconcile_op_count.py ---------------
# Two pins each add an op and each take static_assert(GGML_OP_COUNT == 1) to 2;
# the merged tree has 3 ops, so reconcile_op_count.py rewrites the asserts to 3
# and neither pin's own "== 2" line survives. That must not fail the pin, and
# every other line the pin adds is still held.
d = Path(tempfile.mkdtemp(prefix="pc_"))
git(d, "init", "-q", "-b", "main")
(d / "ggml").mkdir()
OPS_BASE = ('#define RPC_PROTO_PATCH_VERSION    0\n'
            'enum ggml_op {\n    GGML_OP_NONE,\n    GGML_OP_COUNT,\n};\nstatic_assert(GGML_OP_COUNT == 1, "GGML_OP_COUNT != 1");\n')
(d / "ggml" / "ggml.c").write_text(OPS_BASE)
git(d, "add", "-A"); git(d, "commit", "-qm", "base"); git(d, "tag", "b1")
shas = []
for op in ("GGML_OP_BANDED", "GGML_OP_MOE_CACHE"):
    git(d, "checkout", "-qb", op, "b1")
    (d / "ggml" / "ggml.c").write_text(
        OPS_BASE.replace("    GGML_OP_COUNT,", f"    {op},\n    GGML_OP_COUNT,")
        .replace("== 1, \"GGML_OP_COUNT != 1\"", "== 2, \"GGML_OP_COUNT != 2\"")
        .replace("VERSION    0", "VERSION    1"))
    (d / "ggml" / f"{op.lower()}.c").write_text(f"int {op.lower()}_kernel_entry(void) {{ return 7; }}\n")
    git(d, "add", "-A"); git(d, "commit", "-qm", op)
    shas.append(git(d, "rev-parse", "HEAD").stdout.strip())
git(d, "checkout", "-q", "main")
for op in ("GGML_OP_BANDED", "GGML_OP_MOE_CACHE"):
    r = git(d, "merge", "-q", "--no-ff", "--no-edit", "-m", f"merge {op}", op)
    if r.returncode:                                 # both inserted above COUNT: keep both
        (d / "ggml" / "ggml.c").write_text(
            OPS_BASE.replace("    GGML_OP_COUNT,", "    GGML_OP_BANDED,\n    GGML_OP_MOE_CACHE,\n    GGML_OP_COUNT,")
            .replace("== 1, \"GGML_OP_COUNT != 1\"", "== 2, \"GGML_OP_COUNT != 2\"")
        .replace("VERSION    0", "VERSION    1"))
        git(d, "add", "-A"); git(d, "commit", "-qm", f"merge {op}")
c = d / "ggml" / "ggml.c"
c.write_text(c.read_text().replace("== 2, \"GGML_OP_COUNT != 2\"", "== 3, \"GGML_OP_COUNT != 3\"")
             .replace("VERSION    1", "VERSION    2"))   # and the PATCH bump that covers both ops
ps8 = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
ps8.write_text(json.dumps({"prs": [
    f"https://github.com/unslothai/llama.cpp/pull/{i}/commits/{s}" for i, s in enumerate(shas, 1)]}))
rc, rep, err = run(d, ps8)
check("a reconciled op-count assert does not fail the pins", rc == 0, err)
check("the pins still own their new op symbols",
      all(any("GGML_OP_" in x for x in p["symbols"].get("ggml/ggml.c", [])) for p in rep["pins"]), rep)
(d / "ggml" / "ggml_op_banded.c").write_text("int other(void) { return 0; }\n")
rc, rep, err = run(d, ps8)
check("any other lost line still fails", rc == 1 and "ggml_op_banded_kernel_entry" in err, err)

# --- 9. a PATCH bump that is not an op-count change keeps its contract ------
# A pin that bumps RPC_PROTO_PATCH_VERSION for a non-op RPC change is not
# reconcile_op_count.py's to rewrite, so losing that line must fail the pin.
d = Path(tempfile.mkdtemp(prefix="pc_"))
git(d, "init", "-q", "-b", "main")
(d / "ggml").mkdir()
(d / "ggml" / "ggml.c").write_text(OPS_BASE)
git(d, "add", "-A"); git(d, "commit", "-qm", "base"); git(d, "tag", "b1")
git(d, "checkout", "-qb", "rpc", "b1")
(d / "ggml" / "ggml.c").write_text(OPS_BASE.replace("VERSION    0", "VERSION    1")
                                   + "int rpc_new_command_handler(void) { return 3; }\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "rpc change")
sha9 = git(d, "rev-parse", "HEAD").stdout.strip()
git(d, "checkout", "-q", "main")
git(d, "merge", "-q", "--no-ff", "--no-edit", "-m", "merge rpc", "rpc")
ps9 = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
ps9.write_text(json.dumps({"prs": [f"https://github.com/unslothai/llama.cpp/pull/9/commits/{sha9}"]}))
rc, rep, err = run(d, ps9)
check("a non-op PATCH bump that survived passes", rc == 0, err)
c = d / "ggml" / "ggml.c"
c.write_text(c.read_text().replace("VERSION    1", "VERSION    0"))
rc, rep, err = run(d, ps9)
check("a non-op PATCH bump that was lost fails the pin",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)

# --- 10. a unary-only op change that bumps PATCH keeps the PATCH line --------
# reconcile_op_count.py only raises PATCH from the GGML_OP_COUNT delta, so a pin whose only count
# change is GGML_UNARY_OP_COUNT must still own its PATCH bump; its own unary assert stays exempt.
UN_BASE = OPS_BASE + 'static_assert(GGML_UNARY_OP_COUNT == 1, "GGML_UNARY_OP_COUNT != 1");\n'
d = Path(tempfile.mkdtemp(prefix="pc_"))
git(d, "init", "-q", "-b", "main")
(d / "ggml").mkdir()
(d / "ggml" / "ggml.c").write_text(UN_BASE)
git(d, "add", "-A")
git(d, "commit", "-qm", "base")
git(d, "tag", "b1")
git(d, "checkout", "-qb", "unary", "b1")
(d / "ggml" / "ggml.c").write_text(UN_BASE.replace("VERSION    0", "VERSION    1")
                                   .replace("UNARY_OP_COUNT == 1, \"GGML_UNARY_OP_COUNT != 1",
                                            "UNARY_OP_COUNT == 2, \"GGML_UNARY_OP_COUNT != 2"))
git(d, "add", "-A")
git(d, "commit", "-qm", "unary op")
sha10 = git(d, "rev-parse", "HEAD").stdout.strip()
git(d, "checkout", "-q", "main")
git(d, "merge", "-q", "--no-ff", "--no-edit", "-m", "merge unary", "unary")
ps10 = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
ps10.write_text(json.dumps({"prs": [f"https://github.com/unslothai/llama.cpp/pull/10/commits/{sha10}"]}))
c = d / "ggml" / "ggml.c"
c.write_text(c.read_text().replace("UNARY_OP_COUNT == 2, \"GGML_UNARY_OP_COUNT != 2",
                                   "UNARY_OP_COUNT == 3, \"GGML_UNARY_OP_COUNT != 3"))
rc, rep, err = run(d, ps10)
check("a rewritten unary assert does not fail the pin", rc == 0, err)
c.write_text(c.read_text().replace("VERSION    1", "VERSION    0"))
rc, rep, err = run(d, ps10)
check("a unary-only pin whose PATCH bump was lost fails",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)

# --- 11. an RPC-only pin whose PATCH bump reconcile_op_count.py raised -------
# Two op pins and an RPC-only pin each bump PATCH 0 -> 1; the three identical edits merge to one, and
# reconcile_op_count.py raises PATCH to 0 + 2 for the two ops. The RPC-only pin's own "PATCH 1" line
# is gone, but its bump is still covered (PATCH only went up): that must not fail the pin. A PATCH
# that went below the pin's, or a MINOR that moved, still does.
RPC_H = ("#define RPC_PROTO_MAJOR_VERSION    3\n#define RPC_PROTO_MINOR_VERSION    6\n"
         "#define RPC_PROTO_PATCH_VERSION    {p}\n\n\n\n"
         'static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");\n')
ENUM_C = "enum ggml_op {{\n    GGML_OP_NONE,\n{ops}    GGML_OP_COUNT,\n}};\n"
d = Path(tempfile.mkdtemp(prefix="pc_"))
git(d, "init", "-q", "-b", "main")
(d / "ggml").mkdir()
(d / "ggml" / "ggml-rpc.h").write_text(RPC_H.format(p=0, n=1))
(d / "ggml" / "ggml.c").write_text(ENUM_C.format(ops=""))
git(d, "add", "-A"); git(d, "commit", "-qm", "base"); git(d, "tag", "b1")
shas = []
git(d, "checkout", "-qb", "rpconly", "b1")
(d / "ggml" / "ggml-rpc.h").write_text(RPC_H.format(p=1, n=1))
(d / "ggml" / "rpc.c").write_text("int rpc_new_command_handler(void) { return 3; }\n")
git(d, "add", "-A"); git(d, "commit", "-qm", "rpc only"); shas.append(git(d, "rev-parse", "HEAD").stdout.strip())
for op in ("GGML_OP_BANDED", "GGML_OP_MOE_CACHE"):
    git(d, "checkout", "-qb", op, "b1")
    (d / "ggml" / "ggml-rpc.h").write_text(RPC_H.format(p=1, n=2))
    (d / "ggml" / "ggml.c").write_text(ENUM_C.format(ops=f"    {op},\n"))
    git(d, "add", "-A"); git(d, "commit", "-qm", op); shas.append(git(d, "rev-parse", "HEAD").stdout.strip())
git(d, "checkout", "-q", "main")
for b in ("rpconly", "GGML_OP_BANDED", "GGML_OP_MOE_CACHE"):
    if git(d, "merge", "-q", "--no-ff", "--no-edit", b).returncode:     # both ops above COUNT: keep both
        (d / "ggml" / "ggml.c").write_text(ENUM_C.format(ops="    GGML_OP_BANDED,\n    GGML_OP_MOE_CACHE,\n"))
        git(d, "add", "-A"); git(d, "commit", "-qm", f"merge {b}")
rpc = d / "ggml" / "ggml-rpc.h"
check("(fixture) the three PATCH bumps merged into one", "PATCH_VERSION    1\n" in rpc.read_text(), rpc.read_text())
rpc.write_text(RPC_H.format(p=2, n=3))                 # what reconcile_op_count.py writes
ps11 = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
ps11.write_text(json.dumps({"prs": [
    f"https://github.com/unslothai/llama.cpp/pull/{i}/commits/{s}" for i, s in enumerate(shas, 11)]}))
rc, rep, err = run(d, ps11)
check("an RPC-only pin whose PATCH reconcile raised past it passes", rc == 0, err)
rpc.write_text(RPC_H.format(p=0, n=3))
rc, rep, err = run(d, ps11)
check("the RPC-only pin with PATCH back below its bump still fails",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)
rpc.write_text(RPC_H.format(p=2, n=3).replace("MINOR_VERSION    6", "MINOR_VERSION    7"))
rc, rep, err = run(d, ps11)
check("a higher PATCH under a different MINOR does not count as the pin's bump",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)


def pin_repo(base_files, pin_files, num):
    """Base tag b1 with base_files, one pin commit writing pin_files, merged; returns (repo, pr-set)."""
    d = Path(tempfile.mkdtemp(prefix="pc_"))
    git(d, "init", "-q", "-b", "main")
    for files in (base_files, pin_files):
        for rel, body in files.items():
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(body)
        git(d, "add", "-A")
        git(d, "commit", "-qm", "c")
        if files is base_files:
            git(d, "tag", "b1")
            git(d, "checkout", "-qb", "pin")
    sha = git(d, "rev-parse", "HEAD").stdout.strip()
    git(d, "checkout", "-q", "main")
    git(d, "merge", "-q", "--no-ff", "--no-edit", "-m", "merge pin", "pin")
    ps = Path(tempfile.mkdtemp(prefix="pcset_")) / "pr-set.json"
    ps.write_text(json.dumps({"prs": [f"https://github.com/unslothai/llama.cpp/pull/{num}/commits/{sha}"]}))
    return d, ps


RPC_V = ("#define RPC_PROTO_MAJOR_VERSION    3\n#define RPC_PROTO_MINOR_VERSION    {mi}\n"
         "#define RPC_PROTO_PATCH_VERSION    {p}\n")
OP_H = "enum ggml_op {{\n    GGML_OP_NONE,\n{ops}    GGML_OP_COUNT,\n}};\n"
HANDLER = "int rpc_new_command_handler(void) { return 3; }\n"

# --- 12. a commented RPC_PROTO define is not the tree's version --------------
# An RPC-only pin bumps PATCH 0 -> 1. The merged tree has PATCH 0 active under a comment naming PATCH 2:
# the compiler sees 0, as reconcile_op_count.rpc_version does, so the bump is lost.
d, ps = pin_repo({"ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=0)},
                 {"ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=1), "ggml/src/rpc.c": HANDLER}, 12)
h = d / "ggml/include/ggml-rpc.h"
h.write_text("/* a later revision:\n#define RPC_PROTO_PATCH_VERSION    2\n*/\n" + RPC_V.format(mi=6, p=0))
rc, rep, err = run(d, ps)
check("a PATCH named only in a comment does not cover a lost bump",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)

# --- 13. rewording an op-count assert does not make a pin an op pin ----------
# The pin bumps PATCH for an RPC command and only rewords the GGML_OP_COUNT assert message; the enum is
# unchanged, so reconcile_op_count.py never raises PATCH for it and the pin still owns its bump.
ASSERT_C = 'static_assert(GGML_OP_COUNT == 1, "{msg}");\n'
d, ps = pin_repo({"ggml/include/ggml.h": OP_H.format(ops=""), "ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=0),
                  "ggml/src/ggml.c": ASSERT_C.format(msg="GGML_OP_COUNT != 1")},
                 {"ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=1), "ggml/src/rpc.c": HANDLER,
                  "ggml/src/ggml.c": ASSERT_C.format(msg="op count changed: update the tables")}, 13)
h = d / "ggml/include/ggml-rpc.h"
h.write_text(RPC_V.format(mi=6, p=0))
rc, rep, err = run(d, ps)
check("an assert-message edit does not drop an RPC-only pin's PATCH bump",
      rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)
h.write_text(RPC_V.format(mi=7, p=0))
rc, rep, err = run(d, ps)
check("nor lets a newer MINOR stand in for it", rc == 1 and "RPC_PROTO_PATCH_VERSION" in err, err)

# --- 14. a multiline op-count assert reconcile_op_count.py rewrote -----------
# reconcile_op_count.py rewrites the number on both lines; neither line is the pin's to keep.
ML = 'static_assert(GGML_OP_COUNT == {n},\n    "GGML_OP_COUNT != {n}");\nint keep_this_line_of_code(void);\n'
d, ps = pin_repo({"ggml/include/ggml.h": OP_H.format(ops=""), "ggml/src/ggml.c": ML.format(n=1)},
                 {"ggml/include/ggml.h": OP_H.format(ops="    GGML_OP_BANDED,\n"),
                  "ggml/src/ggml.c": ML.format(n=2) + "int banded_kernel_entry_point(void);\n"}, 14)
c = d / "ggml/src/ggml.c"
c.write_text(c.read_text().replace("== 2", "== 3").replace("!= 2", "!= 3"))
rc, rep, err = run(d, ps)
check("a reconciled multiline op-count assert does not fail the pin", rc == 0, err)
c.write_text(ML.format(n=3))
rc, rep, err = run(d, ps)
check("the multiline exemption covers only the assert", rc == 1 and "banded_kernel_entry_point" in err, err)

# --- 15. an op pin's PATCH requirement is a floor, not exempt ----------------
# The pin adds one op and takes PATCH 0 -> 2; a resolution that leaves PATCH 1 lost part of its bump.
d, ps = pin_repo({"ggml/include/ggml.h": OP_H.format(ops=""), "ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=0),
                  "ggml/src/ggml.c": ASSERT_C.format(msg="GGML_OP_COUNT != 1")},
                 {"ggml/include/ggml.h": OP_H.format(ops="    GGML_OP_BANDED,\n"),
                  "ggml/include/ggml-rpc.h": RPC_V.format(mi=6, p=2),
                  "ggml/src/ggml.c": ASSERT_C.format(msg="GGML_OP_COUNT != 2").replace("== 1", "== 2")}, 15)
h = d / "ggml/include/ggml-rpc.h"
for v, want, name in ((RPC_V.format(mi=6, p=1), 1, "an op pin's PATCH left below its bump fails"),
                      (RPC_V.format(mi=6, p=3), 0, "an op pin's PATCH raised past its bump passes"),
                      (RPC_V.format(mi=7, p=0), 0, "an op pin under a newer MINOR passes"),
                      (RPC_V.format(mi=5, p=9), 1, "an op pin under an older MINOR fails")):
    h.write_text(v)
    rc, rep, err = run(d, ps)
    check(name, rc == want and (want == 0 or "RPC_PROTO_PATCH_VERSION" in err), err)

# --- 16. only whole-line asserts reconcile_op_count.py rewrites are exempt ----
# Code sharing a line with an op-count assert, and an assert outside ggml/ (which reconcile never
# rewrites), are still the pin's: a merge that drops them loses the pin. The base already names every
# symbol on those lines, so only line survival can catch the loss.
SHARED = 'static_assert(GGML_OP_COUNT == 1, "count"); int required_kernel_entry(void);\n'
OUTSIDE = 'static_assert(GGML_OP_COUNT == 1, "tools guard on the op count");\n'
BASE_C = "int required_kernel_entry(void); int count = GGML_OP_COUNT;\n"
BASE_T = "int tool_ops = GGML_OP_COUNT;\n"
d, ps = pin_repo({"ggml/include/ggml.h": OP_H.format(ops=""), "ggml/src/ggml.c": BASE_C, "tools/t.cpp": BASE_T},
                 {"ggml/src/ggml.c": BASE_C + SHARED, "tools/t.cpp": BASE_T + OUTSIDE}, 16)
rc, rep, err = run(d, ps)
check("(fixture) an intact merge with both lines passes", rc == 0, err)
(d / "ggml/src/ggml.c").write_text(BASE_C)
rc, rep, err = run(d, ps)
check("code after an op-count assert on the same line is still owed", rc == 1 and "ggml/src/ggml.c" in err, err)
(d / "ggml/src/ggml.c").write_text(BASE_C + SHARED)
(d / "tools/t.cpp").write_text(BASE_T)
rc, rep, err = run(d, ps)
check("an op-count assert outside ggml/ is still owed", rc == 1 and "tools/t.cpp" in err, err)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
