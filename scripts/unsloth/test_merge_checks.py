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

# A nested `if` inside an arm must not end the enclosing chain. Tracking chains
# by indentation resets on the nested arm and analyses the outer `else if` as a
# fresh chain, so the dedicated GLM5NEXT arm below a shared fallthrough -- the
# exact 08-27 mistake -- stops being reported.
NESTED_DEAD_ARM = """
void f() {
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_GLM5NEXT) {
        if (hparams.indexer_head_size > 0) {
            a();
        } else if (hparams.n_expert > 0) {
            b();
        }
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""
# Two unrelated chains at the same indentation. The second one's opener is a
# multiline condition, which the regex deliberately skips, so an indentation
# key appends the reachable GLM5NEXT arm to the FIRST chain and calls it dead.
# Nothing here is unreachable, and firing would block a release on good code.
SEPARATE_CHAINS = """
void f() {
    if (arch == LLM_ARCH_GLM5NEXT) {
        a();
    }
    unrelated();
    if (hparams.moe_every_n_layers > 0 &&
        il % hparams.moe_every_n_layers == 1) {
        b();
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""
# A brace inside a string literal or a comment is not a brace. Miscounting one
# shifts the depth for the rest of the file, which would silence every chain
# after it.
BRACES_IN_LITERALS = """
void f() {
    const char * tmpl = "{% if x %}{{ y }}{% endif %}";
    // a stray } in a comment {
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_GLM5NEXT) {
        a();
    } else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""

rc, out = run(cpp=NESTED_DEAD_ARM)
check("catches a dead arm across a nested if", rc == 1 and "unreachable" in out, out)
rc, out = run(cpp=SEPARATE_CHAINS)
check("does NOT glue two chains at the same indentation", rc == 0, out)
rc, out = run(cpp=BRACES_IN_LITERALS)
check("still analyses a chain after braces in a string or comment",
      rc == 1 and "unreachable" in out, out)

# llama.cpp puts `else if` on its own line as often as not, src/llama-quant.cpp
# among them. Ending the chain on the brace line loses the arm that follows, so
# the duplicate arch started a fresh chain with nothing taken and passed.
NEXT_LINE_ELSE = """
void f() {
    if (arch == LLM_ARCH_QWEN3NEXT || arch == LLM_ARCH_GLM5NEXT) {
        a();
    }
    else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""
# Same, with a blank line in between: still one chain.
NEXT_LINE_ELSE_BLANK = """
void f() {
    if (arch == LLM_ARCH_GLM5NEXT) {
        a();
    }

    else if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    }
}
"""
# The other direction, which deferring the close could break: a chain that
# really has ended, followed by an unrelated chain at the same depth. Joining
# them reports a reachable arm as dead and blocks a release on good code.
CLOSED_THEN_NEW = """
void f() {
    if (arch == LLM_ARCH_GLM5NEXT) {
        a();
    } else {
        b();
    }
    g();
    if (arch == LLM_ARCH_GLM5NEXT) {
        c();
    } else if (arch == LLM_ARCH_QWEN3NEXT) {
        d();
    }
}
"""

rc, out = run(cpp=NEXT_LINE_ELSE)
check("catches a dead arm when else if starts on the next line",
      rc == 1 and "unreachable" in out, out)
rc, out = run(cpp=NEXT_LINE_ELSE_BLANK)
check("a blank line between } and else does not end the chain",
      rc == 1 and "unreachable" in out, out)
rc, out = run(cpp=CLOSED_THEN_NEW)
check("a genuinely closed chain does not absorb the next one", rc == 0, out)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all merge_checks tests passed")
sys.exit(1 if FAILS else 0)
