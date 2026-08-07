#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Negative controls for verify_upstream_sync.py.

A checker that only ever prints PASS is worthless, so this builds a throwaway repository
shaped like the real one (a base, an upstream that moves on, a fork that customises its own
files), then injects each failure mode one at a time and asserts the checker reports it.

The four modes are the four ways a sync can actually eat our work:

    modify   upstream reformats or reverts a file we own
    delete   our file is dropped in the merge
    renumber a published GGML/LLAMA enum id shifts, which corrupts every GGUF that stores it
    subtle   a value inside our Python changes without the file being obviously touched

Run: python3 scripts/unsloth/test_verify_upstream_sync.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKER = os.path.join(HERE, "verify_upstream_sync.py")


def sh(cwd: str, *args: str) -> str:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"{' '.join(args)} failed in {cwd}:\n{p.stderr}")
    return p.stdout


def git(repo: str, *args: str) -> str:
    return sh(repo, "git", "-c", "user.email=t@t", "-c", "user.name=t", *args)


def write(repo: str, path: str, text: str) -> None:
    full = os.path.join(repo, path)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w") as f:
        f.write(text)


HEADER_BASE = """
    enum ggml_type {
        GGML_TYPE_F32 = 0,
        GGML_TYPE_F16 = 1,
        GGML_TYPE_COUNT = 2,
    };
"""
HEADER_UPSTREAM = """
    enum ggml_type {
        GGML_TYPE_F32 = 0,
        GGML_TYPE_F16 = 1,
        GGML_TYPE_Q4_0 = 2,
        GGML_TYPE_COUNT = 3,
    };
"""
OURS_PY = 'OWNED = ("unslothai/", "danielhanchen/")\nLIMIT = 7\n\n\ndef pin():\n    return OWNED\n'


def build_repo(tmp: str) -> str:
    """base -> upstream advances; fork adds its own files and deletes an upstream one."""
    up = os.path.join(tmp, "upstream")
    os.makedirs(up)
    git(up, "init", "-q", "-b", "master")
    write(up, "ggml/include/ggml.h", HEADER_BASE)
    write(up, "src/model.cpp", "int main(){return 0;}\n")
    write(up, ".github/workflows/ci.yml", "name: CI\n")
    git(up, "add", "-A"); git(up, "commit", "-qm", "base")

    fork = os.path.join(tmp, "fork")
    sh(tmp, "git", "clone", "-q", up, fork)
    git(fork, "remote", "add", "upstream", up)

    # upstream moves on: a new type, so a new COUNT
    write(up, "ggml/include/ggml.h", HEADER_UPSTREAM)
    write(up, "src/model.cpp", "int main(){return 1;}\n")
    git(up, "add", "-A"); git(up, "commit", "-qm", "upstream: add Q4_0")

    # the fork customises only its own tree, and drops upstream CI
    write(fork, "scripts/unsloth/repin.py", OURS_PY)
    write(fork, ".github/workflows/unsloth-prebuilt.yml", "name: prebuilt\n")
    os.remove(os.path.join(fork, ".github/workflows/ci.yml"))
    git(fork, "add", "-A"); git(fork, "commit", "-qm", "unsloth: our own CI and scripts")
    git(fork, "branch", "-f", "forkmaster", "HEAD")
    git(fork, "fetch", "-q", "upstream", "master")
    return fork


def merge(fork: str, name: str) -> str:
    git(fork, "checkout", "-q", "-B", name, "forkmaster")
    p = subprocess.run(["git", "merge", "--no-commit", "--no-ff", "upstream/master"],
                       cwd=fork, capture_output=True, text=True)
    # the fork's deletion of upstream CI conflicts as modify/delete; keep it deleted
    conf = sh(fork, "git", "diff", "--name-only", "--diff-filter=U").split()
    if conf:
        git(fork, "rm", "-q", *conf)
    git(fork, "commit", "-qm", f"merge {name}")
    return sh(fork, "git", "rev-parse", "HEAD").strip()


def run_checker(fork: str, rev: str) -> tuple[int, str]:
    p = subprocess.run([sys.executable, CHECKER, "--repo", fork, "--merge", rev,
                    "--fork", "forkmaster", "--upstream", "upstream/master"],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    fails = 0
    with tempfile.TemporaryDirectory() as tmp:
        fork = build_repo(tmp)

        rc, out = run_checker(fork, merge(fork, "clean"))
        if rc == 0 and "PASS: the sync is additive only" in out:
            print("PASS  clean merge is accepted")
        else:
            fails += 1
            print(f"FAIL  clean merge was rejected (rc={rc})\n{out}")

        cases = [
            ("modify our file", "content",
             lambda: write(fork, "scripts/unsloth/repin.py", OURS_PY + "# upstream reflow\n")),
            ("delete our file", "content",
             lambda: os.remove(os.path.join(fork, "scripts/unsloth/repin.py"))),
            ("renumber a published id", "c_enums",
             lambda: write(fork, "ggml/include/ggml.h",
                           HEADER_UPSTREAM.replace("GGML_TYPE_F16 = 1", "GGML_TYPE_F16 = 5"))),
            ("subtle value change in our Python", "ast",
             lambda: write(fork, "scripts/unsloth/repin.py",
                           OURS_PY.replace('"danielhanchen/"', '"someone-else/"'))),
        ]
        for i, (label, expect, mutate) in enumerate(cases):
            rev = merge(fork, f"bad{i}")
            mutate()
            git(fork, "add", "-A")
            git(fork, "commit", "-qm", f"negative control: {label}")
            rev = sh(fork, "git", "rev-parse", "HEAD").strip()
            rc, out = run_checker(fork, rev)
            caught = rc == 1 and any(l.startswith("FAIL  " + expect) for l in out.splitlines())
            if caught:
                print(f"PASS  caught: {label}  (via {expect})")
            else:
                fails += 1
                print(f"FAIL  MISSED: {label}  (expected FAIL {expect}, rc={rc})\n{out}")

    print(f"\n{'all negative controls caught' if not fails else f'{fails} FAILURES'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
