#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for feature_matrix.py. Run: python3 scripts/unsloth/test_feature_matrix.py

The thing worth testing here is not that a passing probe passes. It is that a
probe which exits 0 having proved NOTHING is reported as a failure, because both
real harnesses do exactly that:

    test-llama-archs -a <excluded arch>   prints SKIP, exits 0
    test-backend-ops test -o <typo>       matches nothing, exits 0

So the fakes below are the real output shapes, verbatim, and the assertions are
about what the script refuses to call a pass.
"""
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "feature_matrix.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def fake(dirp: Path, name: str, stdout: str, rc: int = 0):
    p = dirp / name
    p.write_text("#!/bin/sh\ncat <<'XEOF'\n" + stdout + "\nXEOF\nexit " + str(rc) + "\n")
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# Real output shapes, copied from actual runs.
ARCHS_OK = """main: using seed 1234
|     Model arch.|                         Device|Config|   NMSE vs. CPU|Roundtrip|
|----------------|-------------------------------|------|---------------|---------|
|         inkling|                    NVIDIA B200|   MoE|  OK (1.82e-11)|     SKIP|
|         inkling|Intel(R) Xeon(R) Platinum 8559C|   MoE|  OK (0.00e+00)|     SKIP|"""
ARCHS_ALL_SKIP = """main: using seed 1234
|     Model arch.|                         Device|Config|   NMSE vs. CPU|Roundtrip|
|----------------|-------------------------------|------|---------------|---------|
|         inkling|                    NVIDIA B200| Dense|SKIP           |     SKIP|"""
ARCHS_ABSENT = """main: using seed 1234
|     Model arch.|                         Device|Config|   NMSE vs. CPU|Roundtrip|
|----------------|-------------------------------|------|---------------|---------|"""
OPS_OK = """  FLASH_ATTN_EXT_BANDED(hsk=64): OK
  13/13 tests passed
  Backend CUDA0: OK"""
OPS_NOTHING = """Backend 1/2: CUDA0
Backend 2/2: CPU
  Skipping CPU backend
2/2 backends passed
OK"""
MTMD_OK = """test_projector_registry (185 assertion(s))                          [PASS]

tests      : 1
assertions : 185
failures   : 0"""
MTMD_NOTHING = """tests      : 0
assertions : 0
failures   : 0"""


def build(archs=ARCHS_OK, ops=OPS_OK, mtmd=MTMD_OK, checks=None):
    d = Path(tempfile.mkdtemp(prefix="fm_"))
    (d / "bin").mkdir()
    fake(d / "bin", "test-llama-archs", archs)
    fake(d / "bin", "test-backend-ops", ops)
    fake(d / "bin", "test-mtmd-impl", mtmd)
    manifest = d / "feature-checks.json"
    manifest.write_text(json.dumps({
        "schema": 1,
        "features": {"inkling": {"owner": "unslothai#172", "checks": checks or [
            {"kind": "arch", "arch": "inkling"},
            {"kind": "backend-op", "op": "FLASH_ATTN_EXT_BANDED"},
        ]}},
        "unchecked": {"unslothai#95": "no feature surface"},
    }))
    return d, manifest


def run(d, manifest, *extra):
    rep = d / "r.json"
    p = subprocess.run([sys.executable, str(SCRIPT), "--build-dir", str(d),
                        "--feature-checks", str(manifest), "--report", str(rep), *extra],
                       capture_output=True, text=True)
    return p.returncode, (json.loads(rep.read_text()) if rep.exists() else {}), p.stdout + p.stderr


# --- 1. everything genuinely ran ------------------------------------------
d, m = build()
rc, rep, out = run(d, m, "--gpu")
check("a real pass passes", rc == 0 and rep["ok"], out)
check("the evidence is recorded, not just the verdict",
      "1.82e-11" not in out and "2/2 device rows" in out, out)

# --- 2. the arch harness skipped the arch and exited 0 ---------------------
d, m = build(archs=ARCHS_ALL_SKIP)
rc, rep, out = run(d, m, "--gpu")
check("an all-SKIP arch run is a failure", rc == 1, out)
check("and says nothing was decoded", "nothing was decoded" in out, out)

# --- 3. the arch is not in the harness at all -----------------------------
d, m = build(archs=ARCHS_ABSENT)
rc, rep, out = run(d, m, "--gpu")
check("an arch with no row at all is a failure", rc == 1, out)
check("and says it is not in the harness", "not in the harness" in out, out)

# --- 4. the op filter matched nothing -------------------------------------
d, m = build(ops=OPS_NOTHING)
rc, rep, out = run(d, m, "--gpu")
check("an op filter that matched nothing is a failure", rc == 1, out)
check("and says the filter matched nothing", "matched nothing" in out, out)

# --- 5. the op ran and failed ---------------------------------------------
d, m = build(ops="  11/13 tests passed\n  Backend CUDA0: FAIL")
rc, rep, out = run(d, m, "--gpu")
check("a failing op is a failure", rc == 1 and "11/13" in out, out)

# --- 6. no GPU: op probes are deferred, not passed and not failed ---------
d, m = build()
rc, rep, out = run(d, m)
check("without a GPU the op probe is deferred", rc == 0 and rep["deferred"] == 1, out)
check("deferral is stated in the summary", "need a GPU" in out, out)
check("deferral is not counted as evidence",
      len(rep["features"][0]["results"]) == 1, rep)

# --- 7. a feature with nothing but GPU checks reads as unproven, not ok ---
d, m = build(checks=[{"kind": "backend-op", "op": "FLASH_ATTN_EXT_BANDED"}])
rc, rep, out = run(d, m)
check("a wholly deferred feature does not print ok",
      rc == 0 and "nothing provable without a GPU" in out and "\nok   inkling" not in out, out)

# --- 8. the mtmd probe ran no assertions ----------------------------------
d, m = build(mtmd=MTMD_NOTHING, checks=[{"kind": "mtmd", "projector": "kimik3"}])
rc, rep, out = run(d, m, "--gpu")
check("an mtmd run with zero assertions is a failure", rc == 1, out)

# --- 9. unchecked pins are reported, not hidden ---------------------------
d, m = build()
rc, rep, out = run(d, m, "--gpu")
check("knowingly unchecked pins are printed", "unslothai#95 has no runtime check" in out, out)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
