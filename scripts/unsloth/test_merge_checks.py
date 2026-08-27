#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for merge_checks.py. Run: python3 scripts/unsloth/test_merge_checks.py

The positive cases are reduced from the two real 08-27 mistakes. The negative
cases are the shapes that must NOT fire, because a check that blocks a good
merge costs a release just as surely as a bad merge does.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "merge_checks.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def run(py=None, cpp=None):
    d = Path(tempfile.mkdtemp(prefix="mc_"))
    (d / "gguf-py" / "gguf").mkdir(parents=True)
    (d / "src" / "models").mkdir(parents=True)
    (d / "gguf-py" / "gguf" / "t.py").write_text(py or "x = {}\n")
    (d / "src" / "t.cpp").write_text(cpp or "int main() { return 0; }\n")
    r = subprocess.run([sys.executable, str(SCRIPT), "--root", str(d)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


DUP_KEY = """
MAP = {
    ARCH.QWEN4EXP: {"a": 1},
    ARCH.GLM5NEXT: {"b": 2},
    ARCH.GLM5NEXT: {"b": 2},
}
"""
OK_KEYS = """
MAP = {
    ARCH.QWEN4EXP: {"a": 1},
    ARCH.GLM5NEXT: {"b": 2},
}
"""
DEAD_ARM = """
void f() {
    if (arch == LLM_ARCH_FALCON_H1) {
        a();
    } else if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_GLM5NEXT) {
        b();
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""
LIVE_ARM = """
void f() {
    if (arch == LLM_ARCH_GLM5NEXT && hparams.indexer_head_size > 0) {
        a();
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        b();
    }
}
"""
DISTINCT_ARMS = """
void f() {
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_QWEN35) {
        a();
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        b();
    }
}
"""

rc, out = run(py=DUP_KEY)
check("catches a duplicate dict key", rc == 1 and "GLM5NEXT" in out, out)
rc, out = run(py=OK_KEYS)
check("clean on distinct dict keys", rc == 0, out)

rc, out = run(cpp=DEAD_ARM)
check("catches an unreachable arch arm", rc == 1 and "unreachable" in out, out)
rc, out = run(cpp=LIVE_ARM)
check("does NOT fire when the earlier arm has &&", rc == 0, out)
rc, out = run(cpp=DISTINCT_ARMS)
check("does NOT fire on distinct arches", rc == 0, out)

rc, out = run()
check("clean tree exits 0", rc == 0, out)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all merge_checks tests passed")
sys.exit(1 if FAILS else 0)
