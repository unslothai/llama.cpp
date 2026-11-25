#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Report the CUDA runtime a finished CUDA bundle needs, and what it ships.

Reads the archive (app-*.tar.gz / app-*.zip), not the build tree: only the archive
shows what ships. For each CUDA runtime library the ggml-cuda backend loads
(libcudart, libcublas, libcublasLt, ...; cudart64_N / cublas64_N / cublasLt64_N on
Windows) it reports whether the bundle ships it, and on Linux where `ldd` resolves
it with LD_LIBRARY_PATH cleared, i.e. what a host without a CUDA toolkit is missing.
The report goes to stdout and, under Actions, to the job summary.

Report only: a bundle that needs the runtime from the host passes. The one failure
is a bundle whose BUILD_INFO.txt says `cuda runtime: bundled` but that does not ship
(Linux: resolve from inside the bundle) every library, a packaging bug. The driver
(libcuda.so.1 / nvcuda.dll) always comes from the host and is not judged.

Usage: check_cuda_runtime.py ARCHIVE [ARCHIVE ...]
Exit 1 only on that mismatch; anything else, including an unreadable archive, is a
warning and exit 0, so the report can never cost a leg.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

CUDA_RUNTIME_SONAME_RE = re.compile(
    r"^lib(cudart|cublas|cublasLt|nvrtc|nvJitLink|cufft|curand|cusparse|cusolver)\.so\.\d+$"
)
_C_ENV = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "LC_ALL": "C", "LANG": "C"}


def build_info(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return out


def claims_bundled(info: dict[str, str]) -> bool:
    return info.get("cuda runtime", "").startswith("bundled")


def report_linux(archive: Path) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
    """(BUILD_INFO, [(library, status)], mismatches)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not m.name.startswith("/") and ".." not in Path(m.name).parts]
            if hasattr(tarfile, "tar_filter"):  # 3.12+ and the 3.8-3.11 security backports
                tar.extractall(root, members=members, filter="tar")
            else:
                tar.extractall(root, members=members)
        info = build_info((root / "BUILD_INFO.txt").read_text()) if (root / "BUILD_INFO.txt").is_file() else {}
        real_root = root.resolve()
        rows: list[tuple[str, str]] = []
        bad: list[str] = []
        for backend in sorted(root.glob("libggml-cuda.so*")):
            if not backend.is_file() or backend.is_symlink():
                continue
            ldd = subprocess.run(["ldd", str(backend)], capture_output=True, text=True, env=_C_ENV)
            resolved = re.findall(r"^\s*(\S+)\s+=>\s+(\S+)", ldd.stdout, re.MULTILINE)
            for name, where in resolved:
                if not CUDA_RUNTIME_SONAME_RE.match(name):
                    continue
                inside = where != "not" and real_root in Path(where).resolve().parents
                status = ("shipped" if inside else "missing (needs the host)" if where == "not"
                          else f"from the host: {where}")
                rows.append((name, status))
                if claims_bundled(info) and not inside:
                    bad.append(f"{archive.name}: BUILD_INFO says the CUDA runtime is bundled, but {name} is {status}")
        return info, rows, bad


def report_windows(archive: Path) -> tuple[dict[str, str], list[tuple[str, str]], list[str]]:
    with zipfile.ZipFile(archive) as z:
        names = {Path(n).name.lower() for n in z.namelist()}
        info = build_info(z.read("BUILD_INFO.txt").decode()) if "BUILD_INFO.txt" in z.namelist() else {}
    m = re.fullmatch(r"cuda(\d+)", info.get("runtime line", ""))
    if "ggml-cuda.dll" not in names or not m:
        return info, [], []
    rows, bad = [], []
    for n in (f"cudart64_{m.group(1)}.dll", f"cublas64_{m.group(1)}.dll", f"cublasLt64_{m.group(1)}.dll"):
        shipped = n.lower() in names
        rows.append((n, "shipped" if shipped else "missing (needs the host)"))
        if claims_bundled(info) and not shipped:
            bad.append(f"{archive.name}: BUILD_INFO says the CUDA runtime is bundled, but {n} is not in it")
    return info, rows, bad


def main() -> int:
    archives = [Path(a) for a in sys.argv[1:]]
    if not archives:
        print(__doc__)
        return 0
    summary: list[str] = []
    mismatches: list[str] = []
    for archive in archives:
        try:
            if archive.name.endswith(".zip"):
                info, rows, bad = report_windows(archive)
            elif shutil.which("ldd"):
                info, rows, bad = report_linux(archive)
            else:
                print(f"::warning::ldd not found; no CUDA runtime report for {archive.name}")
                continue
        except Exception as exc:  # report only: never cost the leg
            print(f"::warning::no CUDA runtime report for {archive.name}: {exc}")
            continue
        claim = info.get("cuda runtime", "(no cuda runtime line in BUILD_INFO.txt)")
        print(f"{archive.name}: cuda runtime: {claim}")
        for name, status in rows:
            print(f"  {name}: {status}")
        missing = [n for n, s in rows if s.startswith("missing")]
        if missing and not claims_bundled(info):
            print(f"::notice::{archive.name} does not ship {', '.join(missing)}; the host must provide "
                  "them or llama.cpp loads no CUDA device and runs on the CPU")
        summary += [f"### CUDA runtime: {archive.name}", "", f"`cuda runtime: {claim}`", "",
                    "| library | status |", "|---|---|", *[f"| `{n}` | {s} |" for n, s in rows], ""]
        mismatches += bad
    if os.environ.get("GITHUB_STEP_SUMMARY") and summary:
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write("\n".join(summary) + "\n")
    for m in mismatches:
        print(f"::error::{m}")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
