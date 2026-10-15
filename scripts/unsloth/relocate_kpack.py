#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Ship rocBLAS's split device code inside a flat ROCm bundle.

TheRock strips the GPU kernels out of librocblas / libhipblaslt (`.hip_fatbin` is NOBITS) into
`<rocm>/.kpack/blas_lib_<gfx>.kpack`, and records where to find them in each library's
`.rocm_kpack_ref` as `../.kpack/blas_lib_@GFXARCH@.kpack`, relative to the library's directory.
Our bundles are flat, so that path points outside the bundle: every rocBLAS kernel that lives in
the library itself (not in rocblas/library) fails with hipErrorInvalidKernelFile, and llama.cpp
aborts in hipblasSgemm on the first F32 GEMM (unslothai/unsloth#13663).

This copies the archives to `rocblas/k/` and rewrites the reference prefix `../.kpack/` to the
same-length `rocblas/k/`, so no msgpack length or section size changes. `rocblas/` already ships
next to the libraries and Studio copies that tree whole, so installs that predate this fix pick
the archives up too. Run it after the libraries are copied and before signing.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

OLD_PREFIX = b"../.kpack/"
NEW_PREFIX = b"rocblas/k/"
ARCHIVE="blas_lib"
REF = re.compile(re.escape(OLD_PREFIX) + rb"([a-z_]+)_@GFXARCH@\.kpack")
assert len(OLD_PREFIX) == len(NEW_PREFIX)


def _libraries(bundle: Path):
    for path in sorted(bundle.iterdir()):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix.lower() == ".dll" or ".so" in path.name:
            yield path


def relocate(bundle: Path, kpack_src: Path, archs: list[str]) -> list[str]:
    old = OLD_PREFIX + ARCHIVE.encode() + b"_@GFXARCH@.kpack"
    new = NEW_PREFIX + ARCHIVE.encode() + b"_@GFXARCH@.kpack"
    referencing = []
    for lib in _libraries(bundle):
        data = lib.read_bytes()
        if any(m.group(1).decode() == ARCHIVE for m in REF.finditer(data)):
            referencing.append((lib, data))
    if not referencing:
        # A ROCm dist that keeps the kernels inside the libraries needs nothing shipped.
        return []

    dest = bundle / NEW_PREFIX.decode()
    dest.mkdir(parents=True, exist_ok=True)
    for arch in archs:
        src = kpack_src / f"{ARCHIVE}_{arch}.kpack"
        if not src.is_file():
            raise SystemExit(f"missing {src}: the ROCm dist has no {ARCHIVE} kernels for {arch}")
        shutil.copy2(src, dest / src.name)
    # Every copy: patchelf can leave a second one in a moved segment, and the loader reads by address.
    for lib, data in referencing:
        lib.write_bytes(data.replace(old, new))
    return [lib.name for lib, _ in referencing]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--bundle", required=True, type=Path, help="flat bundle dir holding the ROCm libraries")
    p.add_argument("--kpack-src", required=True, type=Path, help="the ROCm dist's .kpack dir")
    p.add_argument("--archs", required=True, help="gfx targets, comma or semicolon separated")
    a = p.parse_args()
    archs = [x.strip() for x in re.split(r"[,;]", a.archs) if x.strip()]
    patched = relocate(a.bundle, a.kpack_src, archs)
    if patched:
        print(f"kpack: {', '.join(archs)} -> {NEW_PREFIX.decode()}; references rewritten in {', '.join(patched)}")
    else:
        print(f"kpack: no library in {a.bundle} references ../.kpack/{ARCHIVE}; nothing to ship")
    return 0


if __name__ == "__main__":
    sys.exit(main())
