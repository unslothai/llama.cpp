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
rc, out = run(d, base=None)    # "none" -> "f(x,y)" changes a base symbol, which --base-ref would refuse
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
check("and the output says the RPC version was not checked", "RPC_PROTO_*_VERSION not checked" in out, out)

# One pin added an op and moved every assert, RPC included, to 102 but left RPC_PROTO_* at 8.0.0:
# the RPC assert is current, the protocol was still not bumped.
ONE = (BASE + ["MOE_CACHE_LOOKUP"], NAMES + ["MOE_CACHE_LOOKUP"], SYMS + ["moe_cache_lookup(x)"], 102)
d = make(*ONE, patch=0)
before = snapshot(d)
rc, out = run(d)
check("a current RPC assert with an unbumped version and --base-ref exits 1",
      rc == 1 and "without bumping the RPC protocol" in out, out)
check("and nothing is rewritten", snapshot(d) == before)
rc, out = run(d, "--dry-run")
check("--dry-run refuses the unbumped version too", rc == 1 and "without bumping the RPC protocol" in out, out)

d = make(*ONE)
before = snapshot(d)
rc, out = run(d)
check("a current RPC assert with a bumped version is clean",
      rc == 0 and "clean" in out and "RPC_PROTO 8.0.0 -> 8.0.1" in out and snapshot(d) == before, out)

d = make(*ONE, patch=0)
rc, out = run(d, base=None)
check("without --base-ref a current RPC assert passes and says the version was not checked",
      rc == 0 and "RPC_PROTO_*_VERSION not checked" in out, out)

# A pin that deletes or comments out the RPC assert must not switch the version check off.
RPC_ASSERT = 'static_assert(GGML_OP_COUNT == {n}, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");'


def drop_rpc_assert(d, n, how):
    p = d / "ggml" / "include" / "ggml-rpc.h"
    line = RPC_ASSERT.format(n=n)
    p.write_text(p.read_text().replace(line, "" if how.startswith("delete") else "// " + line))


for how in ("deleted", "commented out"):
    d = make(*ONE, patch=0)
    drop_rpc_assert(d, 102, how)
    before = snapshot(d)
    rc, out = run(d)
    check(f"RPC assert {how} and the version not bumped: exit 1",
          rc == 1 and "is gone from the merged tree" in out, out)
    check(f"and nothing is rewritten ({how})", snapshot(d) == before)

d = make(*ONE)
drop_rpc_assert(d, 102, "delete")
rc, out = run(d)
check("RPC assert deleted with the version bumped still exits 1 (assert removed)",
      rc == 1 and "is gone from the merged tree" in out, out)

# Every GGML_OP_COUNT assert gone, ggml.c too: the RPC check still runs.
d = make(*ONE)
drop_rpc_assert(d, 102, "delete")
p = d / "ggml" / "src" / "ggml.c"
p.write_text(p.read_text().replace('static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");', ""))
rc, out = run(d)
check("no GGML_OP_COUNT assert left anywhere: the family is still checked and refuses",
      rc == 1 and "base has a static_assert(GGML_OP_COUNT" in out and "skip GGML_OP_COUNT" not in out, out)

# A base without the RPC assert: the version is still checked against the op count.
d = make(BASE, NAMES, SYMS, 101)
drop_rpc_assert(d, 101, "delete")
git(d, "commit", "-qam", "no rpc assert")
git(d, "tag", "base_noassert")
write_tree(d, *ONE, None, 8, 0)
drop_rpc_assert(d, 102, "delete")
rc, out = run(d, base="base_noassert")
check("base without the RPC assert, op added, version not bumped: exit 1",
      rc == 1 and "without bumping the RPC protocol" in out, out)
write_tree(d, *ONE, None, 8, 1)
drop_rpc_assert(d, 102, "delete")
rc, out = run(d, base="base_noassert")
check("base without the RPC assert, op added, PATCH bumped: clean", rc == 0 and "RPC_PROTO 8.0.0 -> 8.0.1" in out, out)

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

# A family whose last assert the merge deleted: the base had one, so that is refused with --base-ref.
GLU_ASSERT = 'static_assert(GGML_GLU_OP_COUNT == 2, "GGML_GLU_OP_COUNT != 2");'
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace(GLU_ASSERT, ""))
before = snapshot(d)
rc, out = run(d)
check("the last GLU assert deleted, base had one: exit 1",
      rc == 1 and "base has a static_assert(GGML_GLU_OP_COUNT" in out, out)
check("and nothing is rewritten", snapshot(d) == before)
src.write_text(src.read_text().replace('"REGLU",', '"REGLU", // ' + GLU_ASSERT))
rc, out = run(d)
check("a commented-out assert counts as deleted", rc == 1 and "base has a static_assert(GGML_GLU_OP_COUNT" in out, out)

# Without --base-ref there is no base to compare with, but the enum and table are still checked.
rc, out = run(d, base=None)
check("no assert and no --base-ref: the family is still validated",
      rc == 0 and "ok   GGML_GLU_OP_COUNT = 2: enum and GGML_GLU_OP_NAME agree; no static_assert" in out, out)
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace(GLU_ASSERT, "").replace('    "GEGLU",\n', ""))
rc, out = run(d, base=None)
check("no assert left and a dropped GLU table entry: exit 1",
      rc == 1 and "GGML_GLU_OP_NAME has 1 entries" in out, out)

# A family that never had asserts, and whose enum is not there either, is skipped.
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace(GLU_ASSERT, ""))
h = d / "ggml" / "include" / "ggml.h"
h.write_text(h.read_text().replace("enum ggml_glu_op {", "enum ggml_glu_op_other {"))
git(d, "commit", "-qam", "no glu")
git(d, "tag", "base_noglu")
rc, out = run(d, base="base_noglu")
check("a family with no assert and no enum, in base and merge: skipped",
      rc == 0 and "skip GGML_GLU_OP_COUNT" in out, out)

# Equal counts do not prove order: the tables are indexed by enum value.
NEW2 = ["FLASH_ATTN_EXT_BANDED", "MOE_CACHE_LOOKUP"]
d = make(*TWO[:1], NAMES + NEW2[::-1], TWO[2], 102)
before = snapshot(d)
rc, out = run(d)
check("two new GGML_OP_NAME entries swapped: exit 1",
      rc == 1 and 'GGML_OP_NAME[101] is "MOE_CACHE_LOOKUP", enum entry 101 is GGML_OP_FLASH_ATTN_EXT_BANDED' in out
      and "ggml/src/ggml.c:" in out, out)
check("and nothing is rewritten", snapshot(d) == before)
rc, out = run(d, base=None)
check("the name order is checked without --base-ref too", rc == 1 and "out of enum order" in out, out)

d = make(BASE, NAMES, SYMS, 101, unary=("ABS", "SGN", "NEW"), u_assert=2)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace('    "ABS",\n    "SGN",', '    "SGN",\n    "ABS",'))
rc, out = run(d)
check("swapped unary names: exit 1", rc == 1 and "GGML_UNARY_OP_NAME[0] is \"SGN\"" in out, out)
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace('    "REGLU",\n    "GEGLU",', '    "GEGLU",\n    "REGLU",'))
rc, out = run(d)
check("swapped GLU names: exit 1", rc == 1 and "GGML_GLU_OP_NAME[0] is \"GEGLU\"" in out, out)

# GGML_OP_SYMBOL is free text: a new op's symbol inserted before an existing op's (count right,
# names right) shifts that op's symbol, which the base mapping catches.
d = make(*TWO[:2], SYMS[:50] + ["flash_attn_ext_banded(x)"] + SYMS[50:] + ["moe_cache_lookup(x)"], 102)
rc, out = run(d)
check("a symbol line out of place: exit 1",
      rc == 1 and "GGML_OP_SYMBOL[50] is \"flash_attn_ext_banded(x)\"" in out, out)
d = make(*TWO[:2], SYMS + ["moe_cache_lookup(x)", "flash_attn_ext_banded(x)"], 102)
rc, out = run(d)
check("two new symbols swapped are not detectable (only counted): passes", rc == 0, out)

# Two pins that each remove an op and each bump PATCH 0 -> 1: PATCH goes to 0 + 2, as for adds.
d = make(BASE[:-2], NAMES[:-2], SYMS[:-2], 100)
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("op count fell by 2 with PATCH moved by 1: PATCH raised to 2",
      rc == 0 and "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc and "RPC_PROTO_PATCH_VERSION 1 -> 2" in out
      and "GGML_OP_COUNT == 99" in rpc, out + rpc)
d = make(BASE[:-2], NAMES[:-2], SYMS[:-2], 100, patch=0)
rc, out = run(d)
check("op count fell and the version did not move: exit 1",
      rc == 1 and "without bumping the RPC protocol" in out, out)

# --- the pins as real merges: each one is checked against its own fork point -----------------
MID = BASE[:50] + ["UP_NEW"] + BASE[50:]          # an op inserted mid-enum: merges apart from an append


def tree(ops, n, patch):
    return (ops, ["NONE"] + ops, ["none"] + [o.lower() + "(x)" for o in ops], n, None, 8, patch)


def commit_tree(d, msg, t):
    write_tree(d, *t)
    git(d, "add", "-A")
    git(d, "commit", "-qm", msg)


def history(upstream, pins, merge_into_pin=False, resolved=None):
    """tag fork = the 101-op tree; tag base = fork + upstream (None: no change); pins branch off fork.

    resolved: the tree a conflicted merge is resolved to (an unbumped PATCH line next to a bumped one
    conflicts; the resolution keeps the bump, which is what hides the unbumped pin)."""
    d = Path(tempfile.mkdtemp(prefix="roc_"))
    git(d, "init", "-q", "-b", "main")
    commit_tree(d, "fork", tree(BASE, 101, 0))
    git(d, "tag", "fork")
    if upstream:
        commit_tree(d, "upstream", upstream)
    git(d, "tag", "base")
    for i, t in enumerate(pins):
        git(d, "checkout", "-qb", f"pin{i}", "fork")
        commit_tree(d, f"pin{i}", t)
    if merge_into_pin:                               # repin.py: the base merged into the pin
        git(d, "checkout", "-q", "pin0")
        git(d, "merge", "-q", "--no-ff", "--no-edit", "base")
    else:                                            # the nightly: the pins merged onto the base
        git(d, "checkout", "-q", "--detach", "base")
        for i in range(len(pins)):
            r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q",
                                "--no-ff", "--no-edit", f"pin{i}"], capture_output=True)
            if r.returncode:
                commit_tree(d, f"merge pin{i}", resolved)
    return d


# argon 1: upstream and a pin each add an op and bump PATCH 0 -> 1 from the same fork point. The
# merged version equals the base's, yet both bumped: accepted, PATCH raised to cover the pin's op.
for into_pin in (False, True):
    d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 1)], merge_into_pin=into_pin)
    rc, out = run(d)
    rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
    src = (d / "ggml" / "src" / "ggml.c").read_text()
    how = "base merged into the pin" if into_pin else "pin merged onto the base"
    check(f"upstream and a pin made the same PATCH bump ({how}): reconciled, not refused",
          rc == 0 and "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc and "GGML_OP_COUNT == 103" in rpc
          and src.count("GGML_OP_COUNT == 103") == 2, out)

d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 0)], resolved=tree(MID + ["PIN_NEW"], 102, 1))
before = snapshot(d)
rc, out = run(d)
check("upstream bumped and the pin did not: still refused",
      rc == 1 and "without bumping" in out and snapshot(d) == before, out)

# argon 3: pin A bumps PATCH, pin B adds an op without: A's bump must not cover B.
A, B = BASE[:30] + ["PIN_A"] + BASE[30:], BASE + ["PIN_B"]
AB = BASE[:30] + ["PIN_A"] + BASE[30:] + ["PIN_B"]
d = history(None, [tree(A, 102, 1), tree(B, 102, 0)], resolved=tree(AB, 102, 1))
before = snapshot(d)
rc, out = run(d)
check("one pin's RPC bump does not mask another op-changing pin's missing one",
      rc == 1 and "without bumping RPC_PROTO_*_VERSION" in out and snapshot(d) == before, out)
d = history(None, [tree(A, 102, 1), tree(B, 102, 1)])
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("both pins bumped (the 10-13 shape as real merges): PATCH 0 + 2",
      rc == 0 and "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc and "GGML_OP_COUNT == 103" in rpc, out)

# argon 4: a C++14 digit separator is not a char literal; char literals still are.
d = make(*TWO)
p = d / "ggml" / "src" / "extra.cpp"
p.write_text("static const int BUF = 100'000;\nstatic const wchar_t Q = L'\"';\n"
             'static_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");\n')
rc, out = run(d)
check("an assert after a digit separator is still found and rewritten",
      rc == 0 and 'static_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");' in p.read_text(), out + p.read_text())

# argon 5: a comment between the condition and the message.
d = make(*TWO)
p = d / "ggml" / "src" / "extra.c"
p.write_text('static_assert(GGML_OP_COUNT == 102, /* keep "102" */ "GGML_OP_COUNT != 102");\n')
rc, out = run(d)
check("the message is rewritten past a comment",
      rc == 0 and p.read_text() == 'static_assert(GGML_OP_COUNT == 103, /* keep "102" */ "GGML_OP_COUNT != 103");\n',
      p.read_text())

# argon 6: a fatal git error is not "no match".
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("roc", SCRIPT)
roc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roc)
d = make(BASE, NAMES, SYMS, 101)
try:
    r = roc.base_has_assert(d, "no-such-ref", "GGML_OP_COUNT")
    check("base_has_assert on a bad ref raises", False, f"returned {r}")
except roc.LayoutError:
    check("base_has_assert on a bad ref raises", True)
check("base_has_assert still finds the base's assert", roc.base_has_assert(d, "base", "GGML_OP_COUNT"))

# argon 7: a byte that is not UTF-8 survives the rewrite, and does not crash it.
d = make(*TWO)
p = d / "ggml" / "src" / "extra.c"
p.write_bytes(b'// caf\xe9\nstatic_assert(GGML_OP_COUNT == 102, "GGML_OP_COUNT != 102");\n')
rc, out = run(d)
check("a non-UTF-8 byte is kept and the assert rewritten",
      rc == 0 and p.read_bytes() == b'// caf\xe9\nstatic_assert(GGML_OP_COUNT == 103, "GGML_OP_COUNT != 103");\n',
      out + repr(p.read_bytes()))

# astra 5 / argon 8: explicit enum values that are not sequential break the value-indexed tables.
d = make(BASE, NAMES, SYMS, 101)
h = d / "ggml" / "include" / "ggml.h"
h.write_text(h.read_text().replace("GGML_OP_OP0,", "GGML_OP_OP0 = 2,").replace("GGML_OP_OP1,", "GGML_OP_OP1 = 1,")
             .replace("GGML_OP_OP2,", "GGML_OP_OP2 = 3,"))
before = snapshot(d)
rc, out = run(d)
check("non-sequential explicit enum values exit 2", rc == 2 and "expected the sequential value" in out
      and snapshot(d) == before, out)

# astra 1: the last GLU assert deleted and the GLU table unreadable must not take the skip.
d = make(*TWO)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace(GLU_ASSERT, "")
               .replace("static const char * GGML_GLU_OP_NAME", "static const char * const GGML_GLU_OP_NAME"))
before = snapshot(d)
rc, out = run(d)
check("a deleted assert next to an unreadable table is refused, not skipped",
      rc == 1 and "base has a static_assert(GGML_GLU_OP_COUNT" in out and snapshot(d) == before, out)

# astra 3: an initializer under #if is missing from builds where the condition is false.
d = make(*TWO)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace('    "flash_attn_ext_banded(x)",\n    "moe_cache_lookup(x)",',
                                       '#ifdef GGML_USE_CUDA\n    "flash_attn_ext_banded(x)",\n    "moe_cache_lookup(x)"\n#endif'))
before = snapshot(d)
rc, out = run(d)
check("a preprocessor line inside a table exits 2", rc == 2 and "GGML_OP_SYMBOL" in out and "preprocessor" in out
      and snapshot(d) == before, out)

# astra 4: deleting ggml-rpc.h, which the base has, must not switch the RPC check off.
d = make(*TWO)
(d / "ggml" / "include" / "ggml-rpc.h").unlink()
before = snapshot(d)
rc, out = run(d)
check("ggml-rpc.h deleted by the merge: exit 1, nothing rewritten", rc == 1 and snapshot(d) == before, out)

# astra r7 2: the merge kept a pin's older protocol series (8.0.1) over upstream's 9.0.0.
d = history((BASE, ["NONE"] + BASE, ["none"] + [o.lower() + "(x)" for o in BASE], 101, None, 9, 0),
            [tree(BASE + ["PIN_NEW"], 102, 1)],
            resolved=tree(BASE + ["PIN_NEW"], 102, 1))
h = d / "ggml" / "include" / "ggml-rpc.h"
h.write_text(h.read_text().replace("RPC_PROTO_MAJOR_VERSION    9", "RPC_PROTO_MAJOR_VERSION    8"))
before = snapshot(d)
rc, out = run(d)
check("a merged RPC series older than the base's is refused", rc == 1 and "regressed" in out and snapshot(d) == before, out)

# astra r7 3: a pin that changed the op count and LOWERED its version did not bump it.
d = history(None, [(BASE + ["PIN_NEW"], ["NONE"] + BASE + ["PIN_NEW"],
                    ["none"] + [o.lower() + "(x)" for o in BASE + ["PIN_NEW"]], 102, None, 7, 9)])
h = d / "ggml" / "include" / "ggml-rpc.h"
h.write_text(h.read_text().replace("RPC_PROTO_MAJOR_VERSION    7", "RPC_PROTO_MAJOR_VERSION    8")
             .replace("RPC_PROTO_PATCH_VERSION    9", "RPC_PROTO_PATCH_VERSION    1"))
before = snapshot(d)
rc, out = run(d)
check("a pin whose op change lowered RPC_PROTO_* is refused as unbumped",
      rc == 1 and "without bumping RPC_PROTO_*_VERSION" in out and snapshot(d) == before, out)

# astra r7 4: an RPC define under #if is not the version every build sees.
for repl, word, name in (
        ("#if 0\n#define RPC_PROTO_PATCH_VERSION    1\n#else\n#define RPC_PROTO_PATCH_VERSION    0\n#endif\n",
         "definitions", "an #if/#else pair of RPC_PROTO defines"),
        ("#ifndef GGML_RPC_OLD\n#define RPC_PROTO_PATCH_VERSION    1\n#endif\n",
         "conditional", "a single RPC_PROTO define under #ifndef")):
    d = make(*TWO)
    h = d / "ggml" / "include" / "ggml-rpc.h"
    h.write_text(h.read_text().replace("#define RPC_PROTO_PATCH_VERSION    1\n", repl))
    before = snapshot(d)
    rc, out = run(d)
    check(f"{name} is refused, nothing rewritten", rc == 1 and word in out and snapshot(d) == before, out)

# astra r8 1: no op change, and the merge kept an older PATCH than the base's (upstream bumped it).
d = history((BASE, ["NONE"] + BASE, ["none"] + [o.lower() + "(x)" for o in BASE], 101, None, 8, 2), [])
h = d / "ggml" / "include" / "ggml-rpc.h"
h.write_text(h.read_text().replace("RPC_PROTO_PATCH_VERSION    2", "RPC_PROTO_PATCH_VERSION    1"))
before = snapshot(d)
rc, out = run(d)
check("a PATCH below the base's with no op change is refused", rc == 1 and "regressed" in out and snapshot(d) == before, out)

# astra r8 2: an RPC define that is not a plain integer, or a second one the number regex misses.
for repl, name in (("#define RPC_PROTO_PATCH_VERSION    1 - 1\n", "an expression as the PATCH value"),
                   ("#define RPC_PROTO_PATCH_VERSION    1\n#undef RPC_PROTO_PATCH_VERSION\n"
                    "#define RPC_PROTO_PATCH_VERSION (0)\n", "a parenthesized redefinition")):
    d = make(*TWO)
    h = d / "ggml" / "include" / "ggml-rpc.h"
    h.write_text(h.read_text().replace("#define RPC_PROTO_PATCH_VERSION    1\n", repl))
    before = snapshot(d)
    rc, out = run(d)
    check(f"{name} is refused, nothing rewritten", rc == 1 and "RPC_PROTO_PATCH_VERSION" in out
          and snapshot(d) == before, out)

# astra r9 1: a raw string holding a quote and assert-like text is a literal, not code.
d = make(*TWO)
p = d / "ggml" / "src" / "extra.c"
RAW = 'const char * doc = R"doc(a " quote\nstatic_assert(GGML_OP_COUNT == 101, "literal 101");\n)doc";\n'
p.write_text(RAW + 'static_assert(GGML_OP_COUNT == 101, "real 101");\n')
rc, out = run(d)
check("a raw string's assert-like text is left alone and the real assert after it is rewritten",
      rc == 0 and p.read_text() == RAW + 'static_assert(GGML_OP_COUNT == 103, "real 103");\n', p.read_text() + out)

# astra r9 2: a carry pin refreshed by repin.py (the base merged into it), then merged onto the base by the
# nightly. Upstream and the pin each added an op and made the same PATCH 0 -> 1 edit.
d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 1)], merge_into_pin=True)
git(d, "checkout", "-q", "--detach", "base")
git(d, "merge", "-q", "--no-ff", "--no-edit", "pin0")
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("a refreshed carry pin is judged on its own bump, not refused: PATCH 0 + 2",
      rc == 0 and "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc and "GGML_OP_COUNT == 103" in rpc, out)
d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 0)],     # the nightly merge is redone below
            resolved=tree(MID + ["PIN_NEW"], 102, 1))
git(d, "checkout", "-q", "pin0")
r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff",
                    "--no-edit", "base"], capture_output=True)
if r.returncode:                                     # repin.py's resolution: both ops, upstream's PATCH
    commit_tree(d, "refresh pin0", tree(MID + ["PIN_NEW"], 103, 1))
git(d, "checkout", "-q", "--detach", "base")
r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff",
                    "--no-edit", "pin0"], capture_output=True)
check("(fixture) the refreshed pin merges onto the base cleanly", r.returncode == 0, r.stdout + r.stderr)
before = snapshot(d)
rc, out = run(d)
check("a refreshed carry pin that never bumped is still refused",
      rc == 1 and "without bumping" in out and snapshot(d) == before, out)

# astra r10 1: an unrelated ggml.h edit after repin.py's base merge does not hide the pin's own bump.
d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 1)], merge_into_pin=True)
h = d / "ggml" / "include" / "ggml.h"
h.write_text(h.read_text() + "// a later unrelated declaration\nint ggml_unrelated_helper(void);\n")
git(d, "commit", "-qam", "unrelated header edit on the pin")
git(d, "checkout", "-q", "--detach", "base")
git(d, "merge", "-q", "--no-ff", "--no-edit", "pin0")
rc, out = run(d)
rpc = (d / "ggml" / "include" / "ggml-rpc.h").read_text()
check("a refreshed carry pin with a later unrelated header edit: PATCH 0 + 2",
      rc == 0 and "#define RPC_PROTO_PATCH_VERSION    2\n" in rpc, out)

# astra r10 2: a family whose enum is declared, with no assert left, and a table this cannot read is refused,
# not skipped (the skip is for a family the tree does not have at all).
d = make(BASE, NAMES, SYMS, 101)
src = d / "ggml" / "src" / "ggml.c"
src.write_text(src.read_text().replace(GLU_ASSERT, "").replace('    "GEGLU",\n', '#ifdef X\n    "GEGLU",\n#endif\n'))
before = snapshot(d)
rc, out = run(d, base=None)
check("a declared family with no assert and an unreadable table exits 2, nothing rewritten",
      rc == 2 and "GGML_GLU_OP_NAME" in out and snapshot(d) == before, out)

# astra r11: the pin merged an upstream commit newer than the tag the nightly uses (same ops, same RPC version).
def newer_upstream_refresh(pin_patch):
    d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, pin_patch)],
                resolved=tree(MID + ["PIN_NEW"], 102, 1))
    git(d, "checkout", "-q", "-b", "upstream_next", "base")
    (d / "ggml" / "README.md").write_text("an unrelated upstream change past the tag\n")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "upstream after the tag")
    git(d, "checkout", "-q", "pin0")
    r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff",
                        "--no-edit", "upstream_next"], capture_output=True)
    if r.returncode:
        commit_tree(d, "refresh pin0 onto newer upstream", tree(MID + ["PIN_NEW"], 103, 1))
    git(d, "checkout", "-q", "--detach", "base")
    git(d, "merge", "-q", "--no-ff", "--no-edit", "pin0")
    return d


# Its own bump cannot be told apart from upstream's past the tag (astra r12: a changed-op upstream past the tag
# hides a missing bump), so such a pin is refused outright with the fix: repin it onto the tag.
for patch, how in ((1, "that bumped"), (0, "that never bumped")):
    d = newer_upstream_refresh(patch)
    before = snapshot(d)
    rc, out = run(d)
    check(f"a pin {how}, refreshed onto upstream past the tag: refused, repin onto the tag",
          rc == 1 and "past base" in out and "repin" in out and snapshot(d) == before, out)
    git(d, "checkout", "-q", "pin0")                 # astra r12 1: the refreshed pin checked on its own
    before = snapshot(d)
    rc, out = run(d)
    check(f"the same pin {how} checked directly: refused the same way",
          rc == 1 and "past base" in out and snapshot(d) == before, out)

# astra r12 2: upstream past the tag changed the ops and bumped; the pin's own op had no bump.
d = history(tree(MID, 102, 1), [tree(BASE + ["PIN_NEW"], 102, 0)], resolved=tree(MID + ["PIN_NEW"], 102, 1))
git(d, "checkout", "-q", "-b", "upstream_next", "base")
commit_tree(d, "upstream adds another op past the tag", tree(MID + ["UP_NEXT"], 103, 2))
git(d, "checkout", "-q", "pin0")
r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff",
                    "--no-edit", "upstream_next"], capture_output=True)
if r.returncode:
    commit_tree(d, "refresh pin0", tree(MID + ["UP_NEXT", "PIN_NEW"], 103, 2))
git(d, "checkout", "-q", "--detach", "base")
r = subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", "merge", "-q", "--no-ff",
                    "--no-edit", "pin0"], capture_output=True)
if r.returncode:
    commit_tree(d, "merge pin0", tree(MID + ["UP_NEXT", "PIN_NEW"], 103, 2))
before = snapshot(d)
rc, out = run(d)
check("upstream's bump past the tag does not cover the pin's missing one: refused",
      rc == 1 and snapshot(d) == before, out)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all reconcile_op_count tests passed")
sys.exit(1 if FAILS else 0)
