#!/usr/bin/env python3
"""Cross-platform packager for Unsloth llama.cpp prebuilt bundles.

Curates the shipped executables + their local dynamic-library closure + the
dynamically-loaded ggml backend modules, writes the in-bundle metadata
(BUILD_INFO.txt / UNSLOTH_PREBUILT_INFO.json), and archives the result.

The curation + archive engine is OS-generic -- adding a new OS means
implementing one PlatformStrategy (its dependency-walk tool, lib-name
convention, backend glob, and archive format), not writing a new packaging
script. This mirrors how llama-cpp-binaries' setup.py centralizes the
per-platform packaging so each build workflow stays thin.

The CUDA runtime (libcudart/libcublas, cudart DLLs) is intentionally NOT
bundled: the installer pairs it with the user's PyTorch runtime, selected by
runtime_line.

Linux is the CI-validated path. macOS/Windows strategies follow the correct
platform conventions (otool/@loader_path/tar.gz; dir-local DLLs/zip) but have
not yet been exercised on their runners.

Configuration is read from the environment (see read_config). Runs both inside
the build workflow and standalone for local testing.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

# Force C locale so tool output (readelf/otool) is not localized.
_C_ENV = {**os.environ, "LC_ALL": "C", "LANG": "C"}


def _run(cmd: list[str]) -> str:
    # Fail loudly: a missing/erroring readelf|otool would otherwise yield an
    # empty closure and silently ship a bundle with missing libraries.
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=_C_ENV)
    except FileNotFoundError:
        sys.exit(f"ERROR: required tool '{cmd[0]}' not found")
    if r.returncode != 0:
        sys.exit(f"ERROR: {' '.join(cmd)} failed (rc={r.returncode}): {r.stderr.strip()}")
    return r.stdout


class PlatformStrategy:
    name = "generic"
    exe_suffix = ""
    archive_ext = ".tar.gz"
    rpath = ""
    binaries = ["llama-server", "llama-quantize"]

    def shipped_binaries(self) -> list[str]:
        return [b + self.exe_suffix for b in self.binaries]

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        """Names of dynamic libs `path` needs that are *local* (live in bin_dir)."""
        raise NotImplementedError

    def backend_patterns(self) -> list[str]:
        """Globs for the dlopen'd ggml backend modules (not found via the walk)."""
        raise NotImplementedError

    def supports_symlinks(self) -> bool:
        return True

    def archive(self, stage: Path, out_path: Path) -> None:
        raise NotImplementedError


class LinuxStrategy(PlatformStrategy):
    name = "linux"
    rpath = "$ORIGIN"

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        # Locale-independent: key only on the (NEEDED) tag and the [name].
        needed = re.findall(r"\(NEEDED\)[^\[]*\[([^\]]+)\]", _run(["readelf", "-d", str(path)]))
        return [n for n in needed if (bin_dir / n).exists() or (bin_dir / n).is_symlink()]

    def backend_patterns(self) -> list[str]:
        return ["libggml-cpu-*.so*", "libggml-cuda.so*", "libggml-rpc.so*"]

    def archive(self, stage: Path, out_path: Path) -> None:
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(stage, arcname=".")


class MacOSStrategy(PlatformStrategy):
    name = "macos"
    rpath = "@loader_path"

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        out = _run(["otool", "-L", str(path)])
        deps: list[str] = []
        for line in out.splitlines()[1:]:  # first line echoes the file path
            m = re.match(r"\s+(\S+)\s+\(", line)
            if not m:
                continue
            ref = m.group(1)
            base = os.path.basename(ref)
            # @rpath/@loader_path/relative refs that exist locally are "ours"
            if (ref.startswith("@") or not ref.startswith("/")) and (bin_dir / base).exists():
                deps.append(base)
        return deps

    def backend_patterns(self) -> list[str]:
        return ["libggml-*.dylib"]

    def archive(self, stage: Path, out_path: Path) -> None:
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(stage, arcname=".")


class WindowsStrategy(PlatformStrategy):
    name = "windows"
    exe_suffix = ".exe"
    archive_ext = ".zip"
    rpath = ""  # Windows resolves DLLs from the executable's directory

    # No portable readelf/otool equivalent; the project's own DLLs live beside
    # the binaries in build/bin/Release, so bundle those by name convention.
    LOCAL_DLL_PREFIXES = ("ggml", "llama", "mtmd")

    def local_needed(self, path: Path, bin_dir: Path) -> list[str]:
        return [
            p.name for p in bin_dir.glob("*.dll")
            if p.name.lower().startswith(self.LOCAL_DLL_PREFIXES)
        ]

    def backend_patterns(self) -> list[str]:
        return ["ggml-cpu-*.dll", "ggml-cuda.dll", "ggml-rpc.dll"]

    def supports_symlinks(self) -> bool:
        return False

    def archive(self, stage: Path, out_path: Path) -> None:
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage).as_posix())


STRATEGIES = {s.name: s for s in (LinuxStrategy(), MacOSStrategy(), WindowsStrategy())}


def _copy_one(strategy: PlatformStrategy, bin_dir: Path, stage: Path, name: str) -> None:
    src, dst = bin_dir / name, stage / name
    if dst.exists() or dst.is_symlink():
        return
    if strategy.supports_symlinks() and src.is_symlink():
        target = os.readlink(src)
        os.symlink(target, dst)
        _copy_one(strategy, bin_dir, stage, os.path.basename(target))
    elif src.exists():
        shutil.copy2(src, dst, follow_symlinks=True)


def curate(strategy: PlatformStrategy, bin_dir: Path, stage: Path) -> None:
    roots: list[Path] = []
    for b in strategy.shipped_binaries():
        if not (bin_dir / b).exists():
            sys.exit(f"ERROR: missing {bin_dir / b}")
        shutil.copy2(bin_dir / b, stage / b)
        roots.append(stage / b)

    # Backend modules are dlopen'd, so they never appear in the NEEDED graph;
    # copy them explicitly and treat them as extra roots so their own local
    # dependencies get pulled into the closure too.
    for pat in strategy.backend_patterns():
        for match in sorted(bin_dir.glob(pat)):
            _copy_one(strategy, bin_dir, stage, match.name)
            roots.append(stage / match.name)

    # Walk the local NEEDED closure from every root, scanning each lib once.
    queue = list(roots)
    while queue:
        for need in strategy.local_needed(queue.pop(), bin_dir):
            if not (stage / need).exists() and not (stage / need).is_symlink():
                _copy_one(strategy, bin_dir, stage, need)
                queue.append(stage / need)


# Metadata writers below emit the linux-x64-cuda bundle schema.
def detect_nvcc_sms() -> tuple[str, list[str], str]:
    if not shutil.which("nvcc"):
        return "unavailable", [], "nvcc not found"
    r = subprocess.run(["nvcc", "--list-gpu-arch"], capture_output=True, text=True, env=_C_ENV)
    if r.returncode != 0:
        return "unavailable", [], f"nvcc failed (rc={r.returncode})"
    sms = sorted(set(re.findall(r"compute_(\d+)", r.stdout)), key=int)
    return "available", sms, f"detected {len(sms)} SM targets"


def write_metadata(stage: Path, strategy: PlatformStrategy, cfg: dict, archs: list[str]) -> None:
    short = cfg["commit"][:7]
    min_sm, max_sm = min(map(int, archs)), max(map(int, archs))
    nvcc_status, nvcc_sms, nvcc_msg = detect_nvcc_sms()
    note = f"CUDA {cfg['line'].removeprefix('cuda')} {cfg['klass']} bundle."
    if cfg["line"] == "cuda13":
        note += " CUDA 13 does not include SM70."

    licenses = [f"Third-party licenses bundled with this llama.cpp prebuilt ({cfg['tag']}).",
                f"Source: https://github.com/{cfg['source_repo']} @ {cfg['commit']}", ""]
    src = Path(cfg["src_dir"])
    if (src / "LICENSE").is_file():
        licenses += ["=== llama.cpp LICENSE ===", (src / "LICENSE").read_text(), ""]
    lic_dir = src / "licenses"
    if lic_dir.is_dir():
        for lic in sorted(lic_dir.glob("*")):
            if lic.is_file():
                licenses += [f"=== {lic.name} ===", lic.read_text(), ""]
    (stage / "THIRD_PARTY_LICENSES.txt").write_text("\n".join(licenses))

    info = {
        "upstream_tag": cfg["tag"],
        "source_repo": cfg["source_repo"],
        "source_repo_url": f"https://github.com/{cfg['source_repo']}",
        "source_ref_kind": "tag",
        "requested_source_ref": cfg["tag"],
        "resolved_source_ref": cfg["tag"],
        "source_commit": cfg["commit"],
        "source_commit_short": short,
        "platform": f"{strategy.name}-{cfg['arch']}-cuda",
        "bundle_profile": cfg["profile"],
        "runtime_line": cfg["line"],
        "coverage_class": cfg["klass"],
        "bundle_kind": cfg["klass"],
        "bundle_rank": int(cfg["rank"]),
        "toolkit_line": cfg["toolkit_line"],
        "docker_image": cfg["docker_image"],
        "cuda_archs": archs,
        "supported_sms": archs,
        "nvcc_validation_status": nvcc_status,
        "nvcc_detected_sms": nvcc_sms,
        "nvcc_validation_message": nvcc_msg,
        "min_sm": min_sm,
        "max_sm": max_sm,
        "notes": note,
        "build_shared_libs": True,
        "ggml_backend_dl": True,
        "ggml_cpu_all_variants": True,
        "rpath": strategy.rpath,
    }
    (stage / "UNSLOTH_PREBUILT_INFO.json").write_text(json.dumps(info, indent=2))

    build_info = [
        f"llama.cpp version: {cfg['tag']}",
        f"requested source ref: {cfg['tag']}",
        f"resolved source ref: {cfg['tag']}",
        f"variant: {cfg['profile']}",
        f"runtime line: {cfg['line']}",
        f"coverage class: {cfg['klass']}",
        f"bundle kind: {cfg['klass']}",
        f"bundle rank: {cfg['rank']}",
        f"docker image: {cfg['docker_image']}",
        "backend: CUDA",
        f"toolkit version: {cfg['toolkit_line']}",
        f"cuda archs: {';'.join(archs)}",
        f"supported sms: {','.join(archs)}",
        f"nvcc validation: {nvcc_status}",
        f"min sm: {min_sm}",
        f"max sm: {max_sm}",
        f"os: {strategy.name}",
        f"arch: {cfg['arch']}",
        "build_shared_libs: ON",
        "ggml_backend_dl: ON",
        "ggml_cpu_all_variants: ON",
        "ggml_cuda_nccl: OFF",
        f"rpath: {strategy.rpath}",
        "llama_openssl: ON",
        "openssl_linkage: dynamic",
        "cxx_runtime: dynamic",
        f"source commit: {cfg['commit']}",
        f"source commit short: {short}",
        f"built at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"notes: {note}",
    ]
    (stage / "BUILD_INFO.txt").write_text("\n".join(build_info) + "\n")


def read_config() -> dict:
    def need(k: str) -> str:
        v = os.environ.get(k)
        if not v:
            sys.exit(f"ERROR: missing required env {k}")
        return v

    return {
        "bin_dir": need("BIN_DIR"),
        "src_dir": need("SRC_DIR"),
        "out_dir": need("OUT_DIR"),
        "tag": need("TAG"),
        "commit": need("SOURCE_COMMIT"),
        "profile": need("PROFILE"),
        "line": need("LINE"),
        "klass": need("KLASS"),
        "rank": need("RANK"),
        "toolkit_line": need("TOOLKIT_LINE"),
        "archs": need("ARCHS"),
        "platform": os.environ.get("PLATFORM", "linux"),
        "arch": os.environ.get("ARCH", "x64"),
        "docker_image": os.environ.get("DOCKER_IMAGE", ""),
        "source_repo": os.environ.get("SOURCE_REPO", "ggml-org/llama.cpp"),
        "asset_name": os.environ.get("ASSET_NAME", ""),
    }


def main() -> int:
    cfg = read_config()
    strategy = STRATEGIES.get(cfg["platform"])
    if strategy is None:
        sys.exit(f"ERROR: unknown PLATFORM '{cfg['platform']}' (have {sorted(STRATEGIES)})")

    archs = [a for a in re.split(r"[ ;,]+", cfg["archs"]) if a]
    bin_dir = Path(cfg["bin_dir"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    stage = Path(tempfile.mkdtemp())
    try:
        curate(strategy, bin_dir, stage)
        write_metadata(stage, strategy, cfg, archs)

        asset = cfg["asset_name"] or (
            f"app-{cfg['tag']}-{strategy.name}-{cfg['arch']}-{cfg['profile']}{strategy.archive_ext}"
        )
        out_path = out_dir / asset
        strategy.archive(stage, out_path)

        print(f"wrote {out_path}")
        for p in sorted(stage.iterdir()):
            print(f"  {p.name}")
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
