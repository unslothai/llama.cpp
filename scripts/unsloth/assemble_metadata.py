#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Assemble the release-level sidecars for an Unsloth llama.cpp prebuilt release.

Produces, matching the schema consumed by unslothai/unsloth's installer:
  - llama-prebuilt-manifest.json : describes every locally-built bundle in this
    release (CUDA x64/arm64 profiles + ROCm Linux/Windows per gfx target +
    macOS arm64/x64 + CPU Linux/Windows x64+arm64 + Vulkan Linux/Windows x64),
    with the dispatch metadata the installer needs to pick the right one.
  - llama-prebuilt-sha256.json   : a cross-OS integrity index covering both the
    locally-built bundles AND the upstream ggml-org assets the installer still
    pulls (arm64 CPU + the Windows CUDA cudart/runtime) + the source tarballs.

Run after the build matrix has dropped the app-*.{tar.gz,zip} bundles into --dist.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM_REPO = "ggml-org/llama.cpp"

BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-(?P<platform>linux|windows)-(?P<arch>x64|arm64)-(?P<profile>cuda1[23]-(?:older|newer|portable))\.(?P<ext>tar\.gz|zip)$"
)

ROCM_BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-(?P<platform>linux|windows)-x64-rocm-(?P<gfx>gfx[0-9a-zA-Z]+)\.(?P<ext>tar\.gz|zip)$"
)

# CPU-only and Vulkan bundles, built locally by unsloth-prebuilt-cpu.yml /
# unsloth-prebuilt-vulkan.yml. Like ROCm/macOS they are raw build/bin archives
# with no embedded UNSLOTH_PREBUILT_INFO.json, so everything in the manifest
# entry is derived from the filename. CPU covers x64 + arm64 (the arm64 slices
# supersede the upstream ggml-org CPU passthroughs); Vulkan is x64-only.
CPU_BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-(?P<platform>linux|windows)-(?P<arch>x64|arm64)-cpu\.(?P<ext>tar\.gz|zip)$"
)

VULKAN_BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-(?P<platform>linux|windows)-x64-vulkan\.(?P<ext>tar\.gz|zip)$"
)

# macOS slices are built by unsloth-prebuilt-macos.yml and land in dist/ under
# upstream's own naming (the installer expects that name). They carry no
# embedded UNSLOTH_PREBUILT_INFO.json, so -- like ROCm -- everything is derived
# from the filename.
MACOS_BUNDLE_RE = re.compile(
    r"^llama-(?P<tag>[^/]+)-bin-macos-(?P<arch>arm64|x64)\.tar\.gz$"
)

# Per-(platform, arch) dispatch keys for the published manifest + sha256 index.
# Linux x64 keeps the historical "linux-cuda" so older unsloth installers stay
# compatible; the others get distinct kinds so installers cleanly ignore a
# bundle they can't run instead of trying to launch the wrong binary.
KIND_BY_CUDA = {
    ("linux",   "x64"):   {"manifest": "linux-cuda",       "sha": "linux-cuda-app"},
    ("linux",   "arm64"): {"manifest": "linux-arm64-cuda", "sha": "linux-arm64-cuda-app"},
    ("windows", "x64"):   {"manifest": "windows-cuda",     "sha": "windows-cuda-app"},
}

KIND_BY_ROCM_PLATFORM = {
    "linux":   {"manifest": "linux-rocm",   "sha": "linux-rocm-app"},
    "windows": {"manifest": "windows-rocm", "sha": "windows-rocm-app"},
}

# CPU + Vulkan slices. These supersede the upstream ggml-org CPU/Vulkan
# passthroughs (we now build them ourselves). The manifest kinds match what the
# installer selects per (platform, arch): x64 keeps the historical
# linux-cpu/windows-cpu, arm64 uses linux-arm64/windows-arm64 (the same kinds
# the installer's upstream-fallback path used). The "-app" sha kinds mark them
# as locally-built bundles.
KIND_BY_CPU = {
    ("linux",   "x64"):   {"manifest": "linux-cpu",     "sha": "linux-cpu-app"},
    ("linux",   "arm64"): {"manifest": "linux-arm64",   "sha": "linux-arm64-app"},
    ("windows", "x64"):   {"manifest": "windows-cpu",   "sha": "windows-cpu-app"},
    ("windows", "arm64"): {"manifest": "windows-arm64", "sha": "windows-arm64-app"},
}

KIND_BY_VULKAN_PLATFORM = {
    "linux":   {"manifest": "linux-vulkan",   "sha": "linux-vulkan-app"},
    "windows": {"manifest": "windows-vulkan", "sha": "windows-vulkan-app"},
}

# macOS slices: install_kind / sha-index kind / manifest bundle_profile per arch.
# We build these ourselves now (upstream's arm64 release stamps minos=26 and
# won't dyld-load on macOS < 26), so they are recorded as locally-built bundles
# rather than upstream passthroughs.
MACOS_SLICE = {
    "arm64": {"manifest": "macos-arm64", "sha": "macos-arm64-app", "profile": "macos-metal-arm64"},
    "x64":   {"manifest": "macos-x64",   "sha": "macos-x64-app",   "profile": "macos-cpu-x64"},
}

# Mapping from the umbrella gfx target name (as it appears in the asset
# filename) to the concrete gfx architectures it compiles for. Mirrors the
# `mapped_target` switch in unsloth-prebuilt-rocm.yml; kept duplicated so the
# manifest can stay self-describing without parsing the workflow.
ROCM_TARGET_MAP = {
    "gfx1151": ["gfx1151"],
    "gfx1150": ["gfx1150"],
    "gfx120X": ["gfx1200", "gfx1201"],
    "gfx110X": ["gfx1100", "gfx1101", "gfx1102", "gfx1103"],
    "gfx103X": ["gfx1030", "gfx1031", "gfx1032", "gfx1034"],
    "gfx90a": ["gfx90a"],
    "gfx908": ["gfx908"],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bundle_info(bundle: Path) -> dict:
    """Read the UNSLOTH_PREBUILT_INFO.json embedded in a built bundle.

    Linux/macOS bundles are .tar.gz; Windows bundles are .zip -- dispatch on the
    extension so the Windows CUDA bundles can be read too.
    """
    target = "UNSLOTH_PREBUILT_INFO.json"
    if bundle.name.endswith(".zip"):
        with zipfile.ZipFile(bundle) as zf:
            for n in zf.namelist():
                if n.endswith(target):
                    return json.loads(zf.read(n))
    else:
        with tarfile.open(bundle, "r:gz") as tar:
            for m in tar.getmembers():
                if m.isfile() and m.name.endswith(target):
                    return json.loads(tar.extractfile(m).read())
    sys.exit(f"ERROR: {bundle.name} has no {target}")


def _request(url: str, token: str | None) -> urllib.request.Request:
    req = urllib.request.Request(url, headers={"User-Agent": "unsloth-prebuilt-assembler"})
    if token and "api.github.com" in url:
        req.add_header("Authorization", f"Bearer {token}")
        req.add_header("Accept", "application/vnd.github+json")
    return req


def _with_retry(fn, *, attempts: int = 4, base: float = 2.0):
    for i in range(attempts):
        try:
            return fn()
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            code = getattr(e, "code", None)
            # give up on the last try or on a non-transient 4xx (429 is transient)
            if i == attempts - 1 or (code is not None and 400 <= code < 500 and code != 429):
                raise
            time.sleep(base * (2 ** i))


def http_json(url: str, token: str | None) -> object:
    def go():
        with urllib.request.urlopen(_request(url, token), timeout=120) as resp:
            return json.loads(resp.read())
    return _with_retry(go)


def sha256_url(url: str, token: str | None) -> str:
    def go():
        h = hashlib.sha256()
        with urllib.request.urlopen(_request(url, token), timeout=300) as resp:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    return _with_retry(go)


def upstream_assets(tag: str, token: str | None) -> dict[str, dict]:
    """name -> {url, digest} for the upstream release at `tag`."""
    data = http_json(f"https://api.github.com/repos/{UPSTREAM_REPO}/releases/tags/{tag}", token)
    out: dict[str, dict] = {}
    for asset in data.get("assets", []):  # type: ignore[union-attr]
        out[asset["name"]] = {
            "url": asset["browser_download_url"],
            "digest": asset.get("digest"),  # "sha256:<hex>" since 2024, else None
        }
    return out


def asset_digest_or_hash(asset: dict, token: str | None) -> str:
    """Prefer GitHub's published asset digest; stream-hash as fallback."""
    raw = (asset.get("digest") or "").strip().lower()
    if raw.startswith("sha256:"):
        h = raw.split(":", 1)[1]
        if len(h) == 64 and all(c in "0123456789abcdef" for c in h):
            return h
    return sha256_url(asset["url"], token)


def build_artifacts(
    cuda_bundles: list[tuple[str, str, str, dict]],
    rocm_bundles: list[tuple[str, str, str]],
    macos_bundles: list[tuple[str, str]],
    cpu_bundles: list[tuple[str, str, str]],
    vulkan_bundles: list[tuple[str, str]],
) -> list[dict]:
    """cuda_bundles: list of (asset_name, platform, arch, embedded UNSLOTH_PREBUILT_INFO).
    rocm_bundles:    list of (asset_name, platform, gfx_target).
    macos_bundles:   list of (asset_name, arch).
    cpu_bundles:     list of (asset_name, platform, arch).
    vulkan_bundles:  list of (asset_name, platform).

    CUDA fields come from each bundle's own embedded metadata, so the manifest
    can never disagree with what was actually compiled. ROCm, macOS, CPU and
    Vulkan bundles are raw archives (no embedded info), so their manifest
    entries are derived from the filename + the ROCM_TARGET_MAP / MACOS_SLICE
    tables.
    """
    artifacts = []
    for asset_name, platform, arch, info in cuda_bundles:
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": KIND_BY_CUDA[(platform, arch)]["manifest"],
            "bundle_profile": info["bundle_profile"],
            "runtime_line": info["runtime_line"],
            "coverage_class": info["coverage_class"],
            "supported_sms": info["supported_sms"],
            "min_sm": info["min_sm"],
            "max_sm": info["max_sm"],
            "rank": info["bundle_rank"],
            "toolkit_version": info["toolkit_line"],
        })
    for asset_name, platform, gfx in rocm_bundles:
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": KIND_BY_ROCM_PLATFORM[platform]["manifest"],
            "gfx_target": gfx,
            "mapped_targets": ROCM_TARGET_MAP.get(gfx, [gfx]),
        })
    for asset_name, arch in macos_bundles:
        # No runtime_line/coverage_class for macOS (no CUDA/ROCm runtime to
        # match); emitted as explicit null so the key set stays stable, and a
        # fixed rank since there is a single slice per arch.
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": MACOS_SLICE[arch]["manifest"],
            "bundle_profile": MACOS_SLICE[arch]["profile"],
            "runtime_line": None,
            "coverage_class": None,
            "rank": 50,
        })
    # CPU + Vulkan: no CUDA/ROCm runtime to match, so runtime_line/coverage_class
    # are explicit null (stable key set). A single slice per (backend, platform,
    # arch), so a fixed rank; CPU ranks last (1000) as the universal fallback,
    # matching the installer's own direct-scan rank for a CPU bundle.
    for asset_name, platform, arch in cpu_bundles:
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": KIND_BY_CPU[(platform, arch)]["manifest"],
            "bundle_profile": f"{platform}-cpu-{arch}",
            "runtime_line": None,
            "coverage_class": None,
            "rank": 1000,
        })
    for asset_name, platform in vulkan_bundles:
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": KIND_BY_VULKAN_PLATFORM[platform]["manifest"],
            "bundle_profile": f"{platform}-vulkan-x64",
            "runtime_line": None,
            "coverage_class": None,
            "rank": 60,
        })
    return artifacts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ref", default=None,
                    help="git ref the source was built from; defaults to refs/tags/<tag>")
    ap.add_argument("--source-repo", default=UPSTREAM_REPO,
                    help="repo holding the source ref: upstream, or the publish repo for merged mix tags")
    ap.add_argument("--base-tag", default=None,
                    help="upstream release tag the build is based on; defaults to --tag (differs for mix builds)")
    ap.add_argument("--pr-set", default="[]",
                    help='JSON array of merged upstream PRs: [{"number":..,"sha":..,"url":..},..]')
    ap.add_argument("--commit", required=True)
    ap.add_argument("--dist", required=True, type=Path, help="dir holding the built app-*.tar.gz bundles")
    ap.add_argument("--out", required=True, type=Path, help="dir to write the two JSON sidecars into")
    ap.add_argument("--publish-repo", required=True, help="repo the bundles+manifest are published to")
    ap.add_argument("--token", default=None, help="GitHub token (else $GH_TOKEN/$GITHUB_TOKEN)")
    args = ap.parse_args()

    token = args.token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    tag, commit, short = args.tag, args.commit, args.commit[:7]
    ref = args.ref or f"refs/tags/{tag}"
    source_repo = args.source_repo
    base_tag = args.base_tag or tag
    pr_set = json.loads(args.pr_set)
    # Upstream release assets exist only for a vanilla build of an upstream tag
    # (a mix build's merged tree exists in no repo, only in its release assets).
    is_upstream_release = source_repo == UPSTREAM_REPO and ref == f"refs/tags/{tag}" and not pr_set
    ref_kind = "tag" if is_upstream_release else "mix" if pr_set else "ref"
    source_ref = tag if ref == f"refs/tags/{tag}" else ref
    args.out.mkdir(parents=True, exist_ok=True)

    def base_entry(kind: str, repo: str, digest: str) -> dict:
        return {
            "kind": kind,
            "repo": repo,
            "sha256": digest,
            "source_commit": commit,
            "source_commit_short": short,
            "upstream_tag": base_tag,
        }

    sha_artifacts: dict[str, dict] = {}

    # 1a) locally-built CUDA bundles (Linux x64/arm64 .tar.gz + Windows x64
    # .zip): hash in parallel. All carry embedded UNSLOTH_PREBUILT_INFO.json.
    found: list[tuple[str, str, str, dict]] = []
    cuda_paths = sorted(args.dist.glob("app-*-linux-*.tar.gz")) + sorted(args.dist.glob("app-*-windows-*.zip"))
    for p in cuda_paths:
        m = BUNDLE_RE.match(p.name)
        if not m:
            continue
        found.append((p.name, m.group("platform"), m.group("arch"), read_bundle_info(p)))
    if not found:
        print(f"ERROR: no app-* CUDA bundles in {args.dist}", file=sys.stderr)
        return 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        local_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), found))
    for (name, platform, arch, _info), digest in zip(found, local_digests):
        sha_artifacts[name] = base_entry(KIND_BY_CUDA[(platform, arch)]["sha"], args.publish_repo, digest)

    # 1b) locally-built ROCm bundles (linux .tar.gz + windows .zip): hash in
    # parallel. No embedded metadata; we derive everything from the filename.
    rocm_found: list[tuple[str, str, str]] = []
    for p in sorted(list(args.dist.glob("app-*-rocm-*.tar.gz")) + list(args.dist.glob("app-*-rocm-*.zip"))):
        m = ROCM_BUNDLE_RE.match(p.name)
        if not m:
            continue
        rocm_found.append((p.name, m.group("platform"), m.group("gfx")))
    if rocm_found:
        with ThreadPoolExecutor(max_workers=4) as pool:
            rocm_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), rocm_found))
        for (name, platform, _gfx), digest in zip(rocm_found, rocm_digests):
            sha_artifacts[name] = base_entry(KIND_BY_ROCM_PLATFORM[platform]["sha"], args.publish_repo, digest)
    else:
        # Warning, not error: ROCm can legitimately be empty when a dispatch run
        # narrows operating_systems to skip both Windows and Ubuntu. The daily
        # schedule always builds the full set, so this fires only on manual runs.
        print("WARNING: no app-*-rocm-*.{tar.gz,zip} bundles found", file=sys.stderr)

    # 1c) locally-built macOS slices (arm64 Metal + x64 CPU): hash in parallel.
    # No embedded metadata; we derive everything from the filename. We build
    # these ourselves now, so they are NOT recorded as upstream passthroughs in
    # section 2.
    macos_found: list[tuple[str, str]] = []
    for p in sorted(args.dist.glob("llama-*-bin-macos-*.tar.gz")):
        m = MACOS_BUNDLE_RE.match(p.name)
        if not m:
            continue
        macos_found.append((p.name, m.group("arch")))
    if macos_found:
        with ThreadPoolExecutor(max_workers=4) as pool:
            macos_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), macos_found))
        for (name, arch), digest in zip(macos_found, macos_digests):
            sha_artifacts[name] = base_entry(MACOS_SLICE[arch]["sha"], args.publish_repo, digest)
    else:
        # Like ROCm: warn rather than error, so a partial dispatch run still
        # assembles. The daily schedule always builds both slices.
        print("WARNING: no llama-*-bin-macos-*.tar.gz bundles found", file=sys.stderr)

    # 1d) locally-built CPU + Vulkan bundles (Linux .tar.gz + Windows .zip).
    # No embedded metadata; everything is derived from the filename. These
    # replace the upstream ggml-org CPU/Vulkan passthroughs that section 2 used
    # to record -- the release now ships our own builds for these slices.
    def scan_bundles(regex) -> list[tuple[str, "re.Match[str]"]]:
        out: list[tuple[str, "re.Match[str]"]] = []
        for p in sorted(list(args.dist.glob("app-*.tar.gz")) + list(args.dist.glob("app-*.zip"))):
            m = regex.match(p.name)
            if m:
                out.append((p.name, m))
        return out

    cpu_found = [(name, m.group("platform"), m.group("arch")) for name, m in scan_bundles(CPU_BUNDLE_RE)]
    if cpu_found:
        with ThreadPoolExecutor(max_workers=4) as pool:
            cpu_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), cpu_found))
        for (name, platform, arch), digest in zip(cpu_found, cpu_digests):
            sha_artifacts[name] = base_entry(KIND_BY_CPU[(platform, arch)]["sha"], args.publish_repo, digest)
    else:
        print("WARNING: no app-*-cpu.{tar.gz,zip} bundles found", file=sys.stderr)

    vulkan_found = [(name, m.group("platform")) for name, m in scan_bundles(VULKAN_BUNDLE_RE)]
    if vulkan_found:
        with ThreadPoolExecutor(max_workers=4) as pool:
            vulkan_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), vulkan_found))
        for (name, platform), digest in zip(vulkan_found, vulkan_digests):
            sha_artifacts[name] = base_entry(KIND_BY_VULKAN_PLATFORM[platform]["sha"], args.publish_repo, digest)
    else:
        print("WARNING: no app-*-vulkan.{tar.gz,zip} bundles found", file=sys.stderr)

    # 2) upstream per-OS bundles: read GitHub's published asset.digest from the
    #    API response; fall back to a streaming hash if a digest is missing.
    #    macOS + x64 CPU/Vulkan are absent here on purpose -- we build those
    #    ourselves (1c/1d).
    #    A mix build has no upstream release for its tag, so the whole section
    #    is skipped; its uncovered hosts fall back to a source build of the
    #    merged tree instead of a vanilla upstream binary missing the PRs.
    if not is_upstream_release:
        print(f"WARNING: {source_repo}@{ref} is not an upstream release tag; "
              "skipping upstream asset index entries", file=sys.stderr)
    else:
        assets = upstream_assets(tag, token)
        wanted: list[tuple[str, str]] = []  # (name, kind)
        for name in sorted(assets):
            if re.fullmatch(r"cudart-llama-bin-win-cuda-\d+\.\d+-x64\.zip", name):
                wanted.append((name, "windows-cuda-upstream"))
            # The win-cuda BINARY zips must be recorded under their own names too:
            # the installer resolves an attempt's hash by exact asset name first
            # and only then falls back to the cudart alias, so without these
            # entries every Windows CUDA binary gets paired with the cudart digest
            # and fails download verification.
            elif re.fullmatch(
                rf"llama-{re.escape(tag)}-bin-win-cuda-\d+\.\d+-x64\.zip", name
            ):
                wanted.append((name, "windows-cuda-upstream"))
        # x64 CPU + Vulkan are no longer passthroughs -- we build them ourselves
        # (sections 1d above). arm64 CPU is now built too (1d emits the
        # locally-built linux-arm64/windows-arm64 bundles), but the installer
        # still selects the upstream arm64 asset until it is switched over to
        # those bundles; keep these passthrough checksums until that installer
        # flip lands, then drop them.
        for name, kind in (
            (f"llama-{tag}-bin-ubuntu-arm64.tar.gz",      "linux-arm64-upstream"),
            (f"llama-{tag}-bin-win-cpu-arm64.zip",        "windows-arm64-upstream"),
        ):
            if name not in assets:
                print(f"WARNING: upstream asset {name} not found at {tag}; skipping", file=sys.stderr)
                continue
            wanted.append((name, kind))
        for name, kind in wanted:
            sha_artifacts[name] = base_entry(kind, UPSTREAM_REPO, asset_digest_or_hash(assets[name], token))

    # 3) source tarballs: prefer a local copy in dist -- the workflow downloads
    #    them from codeload so the published asset and its recorded checksum are
    #    the exact same bytes. Fall back to stream-hashing codeload if absent
    #    (e.g. a standalone/local run that didn't pre-fetch them). codeload
    #    doesn't expose pre-computed digests, so we always hash the content.
    source_jobs = [
        (f"llama.cpp-source-{tag}.tar.gz", "upstream-source",
         f"https://codeload.github.com/{source_repo}/tar.gz/{ref}"),
        (f"llama.cpp-source-commit-{commit}.tar.gz", "exact-source",
         f"https://codeload.github.com/{source_repo}/tar.gz/{commit}"),
    ]

    def source_digest(name: str, url: str) -> str:
        local = args.dist / name
        return sha256_file(local) if local.is_file() else sha256_url(url, token)

    with ThreadPoolExecutor(max_workers=2) as pool:
        source_digests = list(pool.map(lambda j: source_digest(j[0], j[2]), source_jobs))
    for (name, kind, _url), digest in zip(source_jobs, source_digests):
        sha_artifacts[name] = base_entry(kind, source_repo, digest)

    # 4) manifest, then hash it into the index. Both sidecars share the same
    #    source-description header; merged_prs records the exact PR head SHAs
    #    a mix build compiled (empty for vanilla builds).
    common = {
        "schema_version": 1,
        "component": "llama.cpp",
        "source_repo": source_repo,
        "source_repo_url": f"https://github.com/{source_repo}",
        "source_ref_kind": ref_kind,
        "requested_source_ref": source_ref,
        "resolved_source_ref": source_ref,
        "source_commit": commit,
        "source_commit_short": short,
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": base_tag,
        "merged_prs": pr_set,
    }
    manifest = {
        **common,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": build_artifacts(found, rocm_found, macos_found, cpu_found, vulkan_found),
    }
    manifest_path = args.out / "llama-prebuilt-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    sha_artifacts["llama-prebuilt-manifest.json"] = base_entry(
        "published-manifest", args.publish_repo, sha256_file(manifest_path)
    )

    sha256_doc = {
        **common,
        "release_tag": tag,
        "artifacts": sha_artifacts,
    }
    (args.out / "llama-prebuilt-sha256.json").write_text(json.dumps(sha256_doc, indent=2))

    print(f"wrote manifest ({len(manifest['artifacts'])} artifacts) and sha256 index "
          f"({len(sha_artifacts)} entries) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
