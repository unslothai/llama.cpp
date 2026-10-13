#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for reconcile_op_count.py. Run: python3 scripts/unsloth/test_reconcile_op_count.py

The positive case is the real 10-13 shape: a base with 101 ops, two pins that each add one
and each bump the asserts to 102, merged into a tree with 103 enum entries.
The negative cases are the shapes that must NOT be rewritten, because a rewrite that hides a
dropped table entry ships wrong op names just as surely as the assert would have stopped it.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "reconcile_op_count.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


BASE_OPS = [f"OP{i}" for i in range(101)]


def header(ops, unary=("ABS", "SGN"), enum_extra=""):
    body = "\n".join(f"        GGML_OP_{o}," for o in ops)
    ubody = "\n".join(f"        GGML_UNARY_OP_{u}," for u in unary)
    return f"""\
#pragma once
    // an apostrophe in a comment: don't
    enum ggml_op {{
        GGML_OP_NONE = 0,
{body}
{enum_extra}
        GGML_OP_COUNT,
    }};

    enum ggml_unary_op {{
{ubody}

        GGML_UNARY_OP_COUNT,
    }};

    enum ggml_glu_op {{
        GGML_GLU_OP_REGLU,
        GGML_GLU_OP_GEGLU,

        GGML_GLU_OP_COUNT, // number of glu ops
    }};
"""


def source(names, symbols, n_assert, unary=("ABS", "SGN"), u_assert=2):
    nm = "\n".join(f'    "{x}",' for x in names)
    sy = "\n".join(f'    "{x}",' for x in symbols)
    un = "\n".join(f'    "{x}",' for x in unary)
    return f"""\
#include "ggml.h"

static const char * GGML_OP_NAME[GGML_OP_COUNT] = {{
{nm}
}};

static_assert(GGML_OP_COUNT == {n_assert}, "GGML_OP_COUNT != {n_assert}");

static const char * GGML_OP_SYMBOL[GGML_OP_COUNT] = {{
{sy}
}};

static_assert(GGML_OP_COUNT == {n_assert}, "GGML_OP_COUNT != {n_assert}");

static_assert(GGML_OP_POOL_COUNT == 2, "GGML_OP_POOL_COUNT != 2");

static const char * GGML_UNARY_OP_NAME[GGML_UNARY_OP_COUNT] = {{
{un}
}};

static_assert(GGML_UNARY_OP_COUNT == {u_assert}, "GGML_UNARY_OP_COUNT != {u_assert}");

static const char * GGML_GLU_OP_NAME[GGML_GLU_OP_COUNT] = {{
    "REGLU",
    "GEGLU",
}};

static_assert(GGML_GLU_OP_COUNT == 2, "GGML_GLU_OP_COUNT != 2");
"""


RPC = """\
#define RPC_PROTO_MAJOR_VERSION    {major}
#define RPC_PROTO_MINOR_VERSION    0
#define RPC_PROTO_PATCH_VERSION    {patch}
// static_assert(GGML_OP_COUNT == {n}, "commented out, never rewritten");
static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
"""


def git(d, *a):
    subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a],
                   check=True, capture_output=True)


def write_tree(d, ops, names, symbols, n_assert, rpc_n, major, patch, **kw):
    (d / "ggml" / "include").mkdir(parents=True, exist_ok=True)
    (d / "ggml" / "src").mkdir(parents=True, exist_ok=True)
    (d / "ggml" / "include" / "ggml.h").write_text(header(ops, **{k: v for k, v in kw.items() if k in ("unary", "enum_extra")}))
    (d / "ggml" / "src" / "ggml.c").write_text(source(names, symbols, n_assert, **{k: v for k, v in kw.items() if k in ("unary", "u_assert")}))
    (d / "ggml" / "include" / "ggml-rpc.h").write_text(RPC.format(n=n_assert if rpc_n is None else rpc_n, major=major, patch=patch))


def make(ops, names, symbols, n_assert, rpc_n=None, major=8, patch=None, **kw):
    """A git repo whose tag `base` is the plain 101-op tree at RPC 8.0.0, worked over to the given tree.

    patch defaults to 0 when the tree has the base's op count, else 1: what each op-adding pin does.
    """
    d = Path(tempfile.mkdtemp(prefix="roc_"))
    git(d, "init", "-q")
    write_tree(d, BASE, NAMES, SYMS, 101, None, 8, 0)
    git(d, "add", "-A")
    git(d, "commit", "-qm", "base")
    git(d, "tag", "base")
    if patch is None:
        patch = 0 if len(ops) == len(BASE) else 1
    write_tree(d, ops, names, symbols, n_assert, rpc_n, major, patch, **kw)
    return d


def run(d, *extra, base="base"):
    cmd = [sys.executable, str(SCRIPT), "--root", str(d), *extra] + (["--base-ref", base] if base else [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def snapshot(d):
    return {p: p.read_text() for p in sorted(d.rglob("*")) if p.is_file() and ".git" not in p.parts}


# 101 ops (NONE + 100 more) in the base, so the base assert says 101.
BASE = BASE_OPS[:100]
NAMES = ["NONE"] + BASE
SYMS = ["none"] + [o.lower() + "(x)" for o in BASE]

d = make(BASE, NAMES, SYMS, 101)
before = snapshot(d)
rc, out = run(d)
check("an untouched upstream tree is clean", rc == 0 and "clean" in out, out)
check("and nothing is rewritten", snapshot(d) == before)

# The 10-13 shape: Inkling adds FLASH_ATTN_EXT_BANDED, #268 adds MOE_CACHE_LOOKUP, each took 101 -> 102.
ops = BASE + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"]
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
         SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102)
rc, out = run(d)
src = (d / "ggml" / "src" / "ggml.c").read_text()
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("101 + 1 + 1: two pins that each bumped to 102 are reconciled", rc == 0, out)
check("both ggml.c asserts say 103",
      src.count('static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");') == 2, src)
check("no 102 is left in ggml.c", "102" not in src, src)
check("the rpc assert says 103 and keeps its message",
      'static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");' in rpc, rpc)
check("two pins that each bumped PATCH 0 -> 1 merged to one bump: PATCH is raised to 0 + 2",
      "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc and "RPC_PROTO_PATCH_VERSION 1 -> 2" in out, out + rpc)
check("the commented-out rpc assert is untouched",
      '// static_assert(GGML_OP_COUNT == 102, "commented out, never rewritten");' in rpc, rpc)
check("the report names file and line of each rewrite",
      "ggml/src/ggml.c:" in out and "ggml/include/ggml-rpc.h:5" in out and "102 -> 103" in out, out)
check("unrelated asserts are left alone",
      'static_assert(GGML_OP_POOL_COUNT == 2, "GGML_OP_POOL_COUNT != 2");' in src
      and 'static_assert(GGML_UNARY_OP_COUNT == 2, "GGML_UNARY_OP_COUNT != 2");' in src, src)
after = snapshot(d)
rc, out = run(d)
check("a second run is a no-op", rc == 0 and "clean" in out and snapshot(d) == after, out)

# --dry-run reports the same thing and writes nothing.
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
         SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102)
before = snapshot(d)
rc, out = run(d, "--dry-run")
check("--dry-run reports the rewrite", rc == 0 and "102 -> 103" in out, out)
check("--dry-run changes nothing", snapshot(d) == before)

# A merge that kept the enum entry and lost the SYMBOL entry: the bug the assert exists for.
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
         SYMS + ["moe_cache_lookup(x)"], 102)
before = snapshot(d)
rc, out = run(d)
check("a table missing an entry fails", rc == 1, out)
check("the failure names the table with file and line",
      "GGML_OP_SYMBOL has 102 entries" in out and "ggml/src/ggml.c:" in out and "ggml/include/ggml.h:" in out, out)
check("the table that does match is not reported", "GGML_OP_NAME has" not in out, out)
check("nothing is rewritten when a table is wrong", snapshot(d) == before)

# A duplicated table entry (a union kept both sides) is the same class of bug.
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP", "MOE_CACHE_LOOKUP"],
         SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102)
rc, out = run(d)
check("a duplicated table entry fails", rc == 1 and "GGML_OP_NAME has 104 entries" in out, out)

# One op family wrong blocks the rewrite of the others too: the tree is not consistent.
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
         SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102,
         unary=("ABS", "SGN", "NEW"), u_assert=2)
src = (d / "ggml" / "src" / "ggml.c").read_text()
src = src.replace('    "NEW",\n', "")
(d / "ggml" / "src" / "ggml.c").write_text(src)
before = snapshot(d)
rc, out = run(d)
check("a unary table missing an entry fails", rc == 1 and "GGML_UNARY_OP_NAME has 2 entries" in out, out)
check("and the op asserts are not rewritten either", snapshot(d) == before)

# Unary ops follow the same pattern.
d = make(BASE, NAMES, SYMS, 101, unary=("ABS", "SGN", "NEW_A", "NEW_B"), u_assert=3)
rc, out = run(d)
src = (d / "ggml" / "src" / "ggml.c").read_text()
check("a stale unary count is reconciled",
      rc == 0 and 'static_assert(GGML_UNARY_OP_COUNT == 4, "GGML_UNARY_OP_COUNT != 4");' in src, out)
check("and the op count, already right, is untouched",
      src.count('static_assert(GGML_OP_COUNT == 101, "GGML_OP_COUNT != 101");') == 2, src)

# Commas, braces and quotes inside entries and comments must not be counted as entries.
d = make(BASE, NAMES, SYMS, 101)
src = (d / "ggml" / "src" / "ggml.c").read_text()
src = src.replace('    "NONE",', '    "NONE", // first, {never} "counted"\n    /* a, b, c */')
src = src.replace('    "none",', '    "f(x,y)",')
(d / "ggml" / "src" / "ggml.c").write_text(src)
h = (d / "ggml" / "include" / "ggml.h").read_text()
h = h.replace("        GGML_OP_NONE = 0,", "        GGML_OP_NONE = 0, // none, really\n        /* GGML_OP_GONE, */")
(d / "ggml" / "include" / "ggml.h").write_text(h)
rc, out = run(d)
check("commas in strings and comments are not entries", rc == 0 and "clean" in out, out)

# A number inside the message that is not the stale count is left alone.
d = make(ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
         SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102)
p = d / "ggml" / "src" / "extra.c"
p.write_text('static_assert(GGML_OP_COUNT == 102, "102 ops, 1020 is not 102x");\n')
rc, out = run(d)
check("only the stale number is rewritten in a message",
      rc == 0 and p.read_text() == 'static_assert(GGML_OP_COUNT == 103, "103 ops, 1020 is not 103x");\n',
      p.read_text())

# The RPC assert guards the protocol version. Two ops added and RPC_PROTO_* left at 8.0.0: a pin
# forgot the bump, and rewriting would hide it.
TWO = (ops, NAMES + ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"],
       SYMS + ["flash_attn_ext_banded(x)", "moe_cache_lookup(x)"], 102)
d = make(*TWO, patch=0)
before = snapshot(d)
rc, out = run(d)
check("op count rose and the RPC version did not move: exit 1",
      rc == 1 and "without bumping the RPC protocol" in out, out)
check("and nothing is rewritten, not even ggml.c", snapshot(d) == before)

# Without the base ref there is nothing to check the version against: refuse, do not rewrite.
d = make(*TWO)
before = snapshot(d)
rc, out = run(d, base=None)
check("a stale RPC assert without --base-ref exits 1", rc == 1 and "--base-ref" in out, out)
check("and nothing is rewritten", snapshot(d) == before)

d = make(*TWO)
rc, out = run(d, base="no-such-tag")
check("an unreadable --base-ref exits 1", rc == 1 and "cannot read" in out, out)

# PATCH already moved by at least the op delta: assert rewritten, PATCH left alone.
d = make(*TWO, patch=3)
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("PATCH that already covers the op delta is kept",
      rc == 0 and "#define RPC_PROTO_PATCH_VERSION    3\n" in rpc and "GGML_OP_COUNT == 103" in rpc, out + rpc)

# A MAJOR bump is a protocol break on its own: PATCH is not touched.
d = make(*TWO, major=9, patch=0)
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("a MAJOR bump counts as moved",
      rc == 0 and "#define RPC_PROTO_MAJOR_VERSION    9\n" in rpc
      and "#define RPC_PROTO_PATCH_VERSION    0\n" in rpc and "GGML_OP_COUNT == 103" in rpc, out + rpc)

# --dry-run reports the PATCH bump too and writes nothing.
d = make(*TWO)
before = snapshot(d)
rc, out = run(d, "--dry-run")
check("--dry-run reports the PATCH bump", rc == 0 and "RPC_PROTO_PATCH_VERSION 1 -> 2" in out, out)
check("--dry-run leaves the version alone", snapshot(d) == before)

# A ggml.c assert only needs the tables; it does not need --base-ref when the RPC one is right.
d = make(*TWO, rpc_n=103, patch=2)
rc, out = run(d, base=None)
check("ggml.c asserts are rewritten without --base-ref when the RPC assert is already right",
      rc == 0 and "GGML_OP_COUNT == 102 -> 103" in out and "rpc " not in out, out)

# Commented-out and quoted asserts are text, not declarations: they stay as they are.
d = make(*TWO)
p = d / "ggml" / "src" / "ggml.c"
noise = ('// static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");\n'
         '/* static_assert(GGML_OP_COUNT == 102, "x");\n   static_assert(GGML_OP_COUNT == 7, "y"); */\n'
         'static const char * doc = "static_assert(GGML_OP_COUNT == 102, \\"q\\");";\n')
p.write_text(noise + p.read_text())
rc, out = run(d)
src = p.read_text()
check("commented and quoted asserts are left untouched", rc == 0 and src.startswith(noise), src[:400])
check("while the two real ones are rewritten",
      src.count('static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");') == 2, out)
check("and only real asserts are reported", out.count("fix  ggml/src/ggml.c") == 2, out)

# Two asserts on one line: both rewritten, offsets stay right.
d = make(*TWO)
p = d / "ggml" / "src" / "extra.c"
p.write_text('static_assert(GGML_OP_COUNT == 102, "a 102"); static_assert(GGML_OP_COUNT == 102, "b 102");\n')
rc, out = run(d)
check("two asserts on one line are both rewritten",
      rc == 0 and p.read_text() == 'static_assert(GGML_OP_COUNT == 103, "a 103"); static_assert(GGML_OP_COUNT == 103, "b 103");\n',
      p.read_text())

# A layout this does not understand must stop the run, not be guessed at.
d = make(BASE, NAMES, SYMS, 101)
h = d / "ggml" / "include" / "ggml.h"
h.write_text(h.read_text().replace("enum ggml_op {", "enum ggml_op_renamed {"))
rc, out = run(d)
check("asserts present but the enum gone exits 2", rc == 2 and "enum ggml_op not found" in out, out)

d = make(BASE, NAMES, SYMS, 101, enum_extra="#ifdef GGML_EXTRA\n        GGML_OP_EXTRA,\n#endif")
rc, out = run(d)
check("a preprocessor line inside the enum exits 2", rc == 2 and "preprocessor" in out, out)

# A family with no asserts at all has nothing to reconcile.
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace('static_assert(GGML_GLU_OP_COUNT == 2, "GGML_GLU_OP_COUNT != 2");', ""))
rc, out = run(d)
check("a family without asserts is skipped", rc == 0 and "skip GGML_GLU_OP_COUNT" in out, out)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all reconcile_op_count tests passed")
sys.exit(1 if FAILS else 0)
