#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for relocate_kpack.py. Run: python3 scripts/unsloth/test_relocate_kpack.py

Fake libraries carry the reference exactly as TheRock embeds it (msgpack map in .rocm_kpack_ref);
the rewrite must keep every byte but the 10-byte prefix, so the msgpack lengths stay valid.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "relocate_kpack.py"
FAILS = []

REF_BLAS = b'\x82\xb2kpack_search_paths\x91\xd9"../.kpack/blas_lib_@GFXARCH@.kpack\xabkernel_name\xd92math-libs/BLAS/rocBLAS/stage/lib/librocblas.so.5.8'
REF_SOLVER = b'\x82\xb2kpack_search_paths\x91\xd9$../.kpack/solver_lib_@GFXARCH@.kpack\xabkernel_name\xd9\x07solver!'


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def run(bundle, kpack, archs):
    return subprocess.run([sys.executable, str(SCRIPT), "--bundle", str(bundle), "--kpack-src", str(kpack), "--archs", archs],
                          capture_output=True, text=True)


def fixture(libs):
    d = Path(tempfile.mkdtemp(prefix="kp_"))
    bundle, kpack = d / "bin", d / ".kpack"
    bundle.mkdir()
    kpack.mkdir()
    for name, body in libs.items():
        (bundle / name).write_bytes(body)
    for arch in ("gfx1100", "gfx1103", "gfx1151"):
        (kpack / f"blas_lib_{arch}.kpack").write_bytes(b"KPACK" + arch.encode())
    return bundle, kpack


def main():
    head, tail = b"\x7fELF" + b"\0" * 60, b"\0" * 32
    # Two copies, as patchelf leaves them; the solver reference must stay untouched.
    rocblas = head + REF_BLAS + b"\0" * 16 + REF_BLAS + tail
    bundle, kpack = fixture({"librocblas.so.5.8": rocblas, "librocsolver.so.0": head + REF_SOLVER + tail,
                             "libggml-hip.so": head + tail, "rocblas.dll": rocblas})
    r = run(bundle, kpack, "gfx1100;gfx1103")
    check("exit 0", r.returncode == 0, r.stderr)
    out = (bundle / "librocblas.so.5.8").read_bytes()
    check("same length", len(out) == len(rocblas))
    check("both copies rewritten", out.count(b'"rocblas/k/blas_lib_@GFXARCH@.kpack') == 2 and b"../.kpack/blas" not in out)
    check("only the prefix changed", sum(a != b for a, b in zip(out, rocblas)) <= 2 * 10)
    check("dll rewritten too", b"rocblas/k/blas_lib" in (bundle / "rocblas.dll").read_bytes())
    check("solver reference untouched", (bundle / "librocsolver.so.0").read_bytes() == head + REF_SOLVER + tail)
    check("requested archs shipped", sorted(p.name for p in (bundle / "rocblas" / "k").iterdir())
          == ["blas_lib_gfx1100.kpack", "blas_lib_gfx1103.kpack"])

    bundle, kpack = fixture({"librocblas.so": rocblas})
    r = run(bundle, kpack, "gfx1201")
    check("missing arch archive fails", r.returncode != 0 and "gfx1201" in (r.stderr + r.stdout))

    bundle, kpack = fixture({"librocblas.so": head + tail})
    r = run(bundle, kpack, "gfx1151")
    check("unsplit dist is a no-op", r.returncode == 0 and not (bundle / "rocblas").exists(), r.stderr)

    if FAILS:
        print(f"\n{len(FAILS)} failed: {FAILS}")
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
