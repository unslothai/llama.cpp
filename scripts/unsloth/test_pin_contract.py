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

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
