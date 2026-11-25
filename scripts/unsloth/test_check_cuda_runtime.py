#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for package_bundle.py's CUDA runtime record and check_cuda_runtime.py.

Builds stub ELF libraries with the same sonames and NEEDED graph as a real CUDA 13
bundle (libggml-cuda -> libcudart, libcublas, libcublasLt), so it runs on any Linux
host with gcc and no CUDA. Run: python3 scripts/unsloth/test_check_cuda_runtime.py
"""
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "package_bundle.py"
CHECK = HERE / "check_cuda_runtime.py"
FAILS = []
NEED = "libcudart.so.13, libcublas.so.13, libcublasLt.so.13"


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def cc(out: Path, soname, needed, libdir: Path, exe=False):
    src = out.parent / (out.name + ".c")
    src.write_text("int main(void){return 0;}\n" if exe else f"int f_{abs(hash(out.name))}(void){{return 0;}}\n")
    cmd = ["gcc", "-o", str(out), str(src), "-L", str(libdir), "-Wl,--no-as-needed",
           "-Wl,-rpath,$ORIGIN", "-Wl,--enable-new-dtags"]
    if not exe:
        cmd[1:1] = ["-shared", "-fPIC"]
        cmd.append(f"-Wl,-soname,{soname}")
    cmd += [f"-l:{n}" for n in needed]
    subprocess.run(cmd, check=True)
    src.unlink()


def make_tree(root: Path):
    """bin/ (build output) linked against stub CUDA 13 runtime libs that stay outside it."""
    cuda, binp = root / "cuda" / "lib", root / "bin"
    cuda.mkdir(parents=True)
    binp.mkdir()
    cc(cuda / "libcublasLt.so.13", "libcublasLt.so.13", [], cuda)
    cc(cuda / "libcudart.so.13", "libcudart.so.13", [], cuda)
    cc(cuda / "libcublas.so.13", "libcublas.so.13", ["libcublasLt.so.13"], cuda)
    cc(binp / "libggml-base.so.1", "libggml-base.so.1", [], binp)
    both = root / "linkdir"
    both.mkdir()
    for p in list(cuda.iterdir()) + [binp / "libggml-base.so.1"]:
        os.symlink(p, both / p.name)
    cc(binp / "libggml-cuda.so", "libggml-cuda.so",
       ["libggml-base.so.1", "libcudart.so.13", "libcublas.so.13", "libcublasLt.so.13"], both)
    for exe in ("llama-server", "llama-cli", "llama-quantize"):
        cc(binp / exe, None, ["libggml-base.so.1"], binp, exe=True)
    return binp, cuda


def package(root: Path, binp: Path, klass: str, extra_env=None):
    out = root / f"out-{klass}-{len(list(root.glob('out-*')))}"
    env = {**os.environ, "PLATFORM": "linux", "ARCH": "x64", "BIN_DIR": str(binp), "SRC_DIR": str(root),
           "OUT_DIR": str(out), "TAG": "b1-mix-test", "SOURCE_COMMIT": "0" * 40, "PROFILE": f"cuda13-{klass}",
           "LINE": "cuda13", "KLASS": klass, "RANK": "1", "TOOLKIT_LINE": "13.3", "ARCHS": "86-real"}
    env.pop("BUNDLE_CUDA_RUNTIME", None)
    env.update(extra_env or {})
    r = subprocess.run([sys.executable, str(PACKAGE)], capture_output=True, text=True, env=env)
    archives = sorted(out.glob("*.tar.gz")) if out.is_dir() else []
    return r, (archives[0] if archives else None)


def read(archive: Path, name: str) -> str:
    with tarfile.open(archive) as t:
        return t.extractfile(f"./{name}").read().decode()


def run_check(archive: Path, summary: Path | None = None, ld_path: str = ""):
    env = {**os.environ, "LD_LIBRARY_PATH": ld_path}
    env.pop("GITHUB_STEP_SUMMARY", None)
    if summary:
        env["GITHUB_STEP_SUMMARY"] = str(summary)
    return subprocess.run([sys.executable, str(CHECK), str(archive)], capture_output=True, text=True, env=env)


def retar(src: Path, dst: Path, build_info_extra: str = "", add: list[Path] = ()):
    with tempfile.TemporaryDirectory() as d:
        with tarfile.open(src) as t:
            t.extractall(d, **({"filter": "tar"} if hasattr(tarfile, "tar_filter") else {}))
        bi = Path(d) / "BUILD_INFO.txt"
        if build_info_extra:
            bi.write_text("\n".join(l for l in bi.read_text().splitlines() if not l.startswith("cuda runtime:"))
                          + "\n" + build_info_extra + "\n")
        for p in add:
            shutil.copy2(p, Path(d) / p.name)
        with tarfile.open(dst, "w:gz") as t:
            t.add(d, arcname=".")
    return dst


if not shutil.which("gcc") or not shutil.which("ldd"):
    print("SKIP  gcc/ldd not found")
    sys.exit(0)

with tempfile.TemporaryDirectory() as d:
    root = Path(d)
    binp, cuda = make_tree(root)

    for klass in ("portable", "newer"):
        r, arc = package(root, binp, klass)
        check(f"{klass} packages", r.returncode == 0 and arc is not None, r.stdout + r.stderr)
        if not arc:
            continue
        with tarfile.open(arc) as t:
            names = {Path(m.name).name for m in t.getmembers()}
        check(f"{klass} ships no CUDA runtime (unchanged)", not any(n.startswith(("libcudart", "libcublas")) for n in names),
              sorted(names))
        bi = read(arc, "BUILD_INFO.txt")
        check(f"{klass} BUILD_INFO names the runtime it needs",
              f"cuda runtime: not bundled; requires {NEED} (a CUDA 13.x toolkit" in bi and "on the library path" in bi, bi)
        info = json.loads(read(arc, "UNSLOTH_PREBUILT_INFO.json"))
        check(f"{klass} JSON records it", info.get("cuda_runtime_bundled") is False
              and info.get("cuda_runtime_libraries") == NEED.split(", ") and "requires" in info.get("cuda_runtime", ""), info)
        summary = root / f"summary-{klass}.md"
        c = run_check(arc, summary)
        # Missing on a host without CUDA 13, "from the host" on one that has it: a pass either way.
        rows = [l for l in c.stdout.splitlines() if l.strip().startswith(tuple(NEED.split(", ")))]
        check(f"{klass}: report-only check passes and reports each library",
              c.returncode == 0 and len(rows) == 3 and all(("missing" in l) or ("from the host" in l) for l in rows),
              c.stdout + c.stderr)
        check(f"{klass}: notice only when something is missing",
              ("::notice::" in c.stdout) == any("missing" in l for l in rows), c.stdout)
        check(f"{klass}: job summary written", summary.is_file() and "| `libcudart.so.13` |" in summary.read_text(),
              summary.read_text() if summary.is_file() else "no summary")
    portable = package(root, binp, "portable")[1]

    # The check resolves with LD_LIBRARY_PATH cleared (what a host without a toolkit sees), so a runtime
    # reachable only through the caller's LD_LIBRARY_PATH is not credited to the host, and it is still a pass.
    c = run_check(portable, ld_path=str(cuda))
    rows = [l for l in c.stdout.splitlines() if l.strip().startswith(tuple(NEED.split(", ")))]
    check("LD_LIBRARY_PATH is ignored and the check still passes",
          c.returncode == 0 and len(rows) == 3 and str(cuda) not in c.stdout
          and all(("missing" in l) or ("from the host" in l) for l in rows), c.stdout)

    # Linux ignores BUNDLE_CUDA_RUNTIME, as before: the record stays truthful.
    r, arc = package(root, binp, "portable", {"BUNDLE_CUDA_RUNTIME": "1"})
    check("Linux BUNDLE_CUDA_RUNTIME=1 still ships nothing and says so",
          r.returncode == 0 and arc is not None and "not bundled" in read(arc, "BUILD_INFO.txt"), r.stdout + r.stderr)

    # A bundle that claims a bundled runtime it does not carry is the one failure.
    liar = retar(portable, root / "app-liar.tar.gz", f"cuda runtime: bundled ({NEED})")
    c = run_check(liar)
    check("claimed-bundled archive without the runtime fails", c.returncode == 1 and "::error::" in c.stdout, c.stdout)
    honest = retar(portable, root / "app-honest.tar.gz", f"cuda runtime: bundled ({NEED})",
                   [cuda / "libcudart.so.13", cuda / "libcublas.so.13", cuda / "libcublasLt.so.13"])
    c = run_check(honest)
    check("claimed-bundled archive that ships and resolves it passes", c.returncode == 0 and "shipped" in c.stdout, c.stdout)

    # An unreadable archive never costs the leg.
    junk = root / "app-junk.tar.gz"
    junk.write_bytes(b"not a tarball")
    c = run_check(junk)
    check("unreadable archive is a warning, exit 0", c.returncode == 0 and "::warning::" in c.stdout, c.stdout)

    def zip_with(path: Path, runtime_line: str, dlls):
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("BUILD_INFO.txt", f"runtime line: cuda13\n{runtime_line}\n")
            z.writestr("ggml-cuda.dll", b"x")
            for n in dlls:
                z.writestr(n, b"x")
        return path
    full = ["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"]
    c = run_check(zip_with(root / "app-w-arm64.zip", "cuda runtime: bundled (x)", full))
    check("zip: claimed bundled with the three DLLs passes", c.returncode == 0, c.stdout)
    c = run_check(zip_with(root / "app-w-arm64-bad.zip", "cuda runtime: bundled (x)", full[:2]))
    check("zip: claimed bundled missing cublasLt fails", c.returncode == 1 and "cublasLt64_13.dll" in c.stdout, c.stdout)
    c = run_check(zip_with(root / "app-w-x64.zip", "cuda runtime: not bundled; requires x", []))
    check("zip: not bundled, no DLLs: reported, passes", c.returncode == 0 and "missing (needs the host)" in c.stdout, c.stdout)
    c = run_check(zip_with(root / "app-w-old.zip", "", []))
    check("zip: an archive from before the record passes", c.returncode == 0, c.stdout)

print(f"\n{len(FAILS)} failure(s)")
sys.exit(1 if FAILS else 0)
