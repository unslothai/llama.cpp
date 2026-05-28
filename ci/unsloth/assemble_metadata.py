#!/usr/bin/env python3
"""Assemble the release-level sidecars for an Unsloth llama.cpp prebuilt release.

Produces, matching the schema published at unslothai/llama.cpp:
  - llama-prebuilt-manifest.json : describes the Linux CUDA bundles this repo
    builds (profile -> runtime_line / coverage_class / supported SMs / rank).
  - llama-prebuilt-sha256.json   : a cross-OS integrity index covering both the
    locally-built CUDA bundles AND the upstream ggml-org assets the installer
    pulls for Windows/macOS/Linux-CPU + the source tarballs, each hashed.

Run after the build matrix has dropped the app-*.tar.gz bundles into --dist.
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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

UPSTREAM_REPO = "ggml-org/llama.cpp"
UPSTREAM_URL = f"https://github.com/{UPSTREAM_REPO}"

BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-linux-(?P<arch>x64|arm64)-(?P<profile>cuda1[23]-(?:older|newer|portable))\.tar\.gz$"
)

ROCM_BUNDLE_RE = re.compile(
    r"^app-(?P<tag>[^/]+)-(?P<platform>linux|windows)-x64-rocm-(?P<gfx>gfx[0-9a-zA-Z]+)\.(?P<ext>tar\.gz|zip)$"
)

# Per-arch dispatch keys for the published manifest + sha256 index. x64 keeps
# the historical "linux-cuda" so older unsloth installers stay compatible;
# arm64 gets distinct kinds so those installers cleanly ignore it instead of
# trying to run an arm64 binary on x86_64.
KIND_BY_ARCH = {
    "x64":   {"manifest": "linux-cuda",       "sha": "linux-cuda-app"},
    "arm64": {"manifest": "linux-arm64-cuda", "sha": "linux-arm64-cuda-app"},
}

KIND_BY_ROCM_PLATFORM = {
    "linux":   {"manifest": "linux-rocm",   "sha": "linux-rocm-app"},
    "windows": {"manifest": "windows-rocm", "sha": "windows-rocm-app"},
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
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bundle_info(tarball: Path) -> dict:
    """Read the UNSLOTH_PREBUILT_INFO.json embedded in a built bundle."""
    with tarfile.open(tarball, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.endswith("UNSLOTH_PREBUILT_INFO.json"):
                return json.loads(tar.extractfile(m).read())
    sys.exit(f"ERROR: {tarball.name} has no UNSLOTH_PREBUILT_INFO.json")


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


def build_manifest(
    tag: str,
    commit: str,
    cuda_bundles: list[tuple[str, str, dict]],
    rocm_bundles: list[tuple[str, str, str]],
) -> dict:
    """cuda_bundles: list of (asset_name, arch, embedded UNSLOTH_PREBUILT_INFO).
    rocm_bundles:    list of (asset_name, platform, gfx_target).

    CUDA fields come from each bundle's own embedded metadata, so the manifest
    can never disagree with what was actually compiled. ROCm bundles are raw
    archives (no embedded info), so their manifest entries are derived from
    the filename + the ROCM_TARGET_MAP table.
    """
    artifacts = []
    for asset_name, arch, info in cuda_bundles:
        artifacts.append({
            "asset_name": asset_name,
            "install_kind": KIND_BY_ARCH[arch]["manifest"],
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
    return {
        "schema_version": 1,
        "component": "llama.cpp",
        "source_repo": UPSTREAM_REPO,
        "source_repo_url": UPSTREAM_URL,
        "source_ref_kind": "tag",
        "requested_source_ref": tag,
        "resolved_source_ref": tag,
        "source_commit": commit,
        "source_commit_short": commit[:7],
        "upstream_repo": UPSTREAM_REPO,
        "upstream_tag": tag,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifacts": artifacts,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--dist", required=True, type=Path, help="dir holding the built app-*.tar.gz bundles")
    ap.add_argument("--out", required=True, type=Path, help="dir to write the two JSON sidecars into")
    ap.add_argument("--publish-repo", required=True, help="repo the bundles+manifest are published to")
    ap.add_argument("--token", default=None, help="GitHub token (else $GH_TOKEN/$GITHUB_TOKEN)")
    args = ap.parse_args()

    token = args.token or os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    tag, commit, short = args.tag, args.commit, args.commit[:7]
    args.out.mkdir(parents=True, exist_ok=True)

    def base_entry(kind: str, repo: str, digest: str) -> dict:
        return {
            "kind": kind,
            "repo": repo,
            "sha256": digest,
            "source_commit": commit,
            "source_commit_short": short,
            "upstream_tag": tag,
        }

    sha_artifacts: dict[str, dict] = {}

    # 1a) locally-built CUDA bundles (both x64 and arm64): hash in parallel.
    found: list[tuple[str, str, dict]] = []
    for p in sorted(args.dist.glob("app-*-linux-*.tar.gz")):
        m = BUNDLE_RE.match(p.name)
        if not m:
            continue
        found.append((p.name, m.group("arch"), read_bundle_info(p)))
    if not found:
        print(f"ERROR: no app-*.tar.gz CUDA bundles in {args.dist}", file=sys.stderr)
        return 1
    with ThreadPoolExecutor(max_workers=4) as pool:
        local_digests = list(pool.map(lambda b: sha256_file(args.dist / b[0]), found))
    for (name, arch, _info), digest in zip(found, local_digests):
        sha_artifacts[name] = base_entry(KIND_BY_ARCH[arch]["sha"], args.publish_repo, digest)

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

    # 2) upstream per-OS bundles: GitHub now publishes a sha256 digest on each
    #    release asset (since early 2024), so we read it from the API response
    #    instead of re-downloading every bundle. ~2 GB of CI bandwidth saved
    #    per run; falls back to streaming hash if a digest is missing.
    assets = upstream_assets(tag, token)
    wanted: list[tuple[str, str]] = []  # (name, kind)
    for name in sorted(assets):
        if re.fullmatch(r"cudart-llama-bin-win-cuda-\d+\.\d+-x64\.zip", name):
            wanted.append((name, "windows-cuda-upstream"))
    for name, kind in (
        (f"llama-{tag}-bin-macos-arm64.tar.gz",       "macos-arm64-upstream"),
        (f"llama-{tag}-bin-macos-x64.tar.gz",         "macos-x64-upstream"),
        (f"llama-{tag}-bin-ubuntu-x64.tar.gz",        "linux-cpu-upstream"),
        (f"llama-{tag}-bin-win-cpu-x64.zip",          "windows-cpu-upstream"),
        (f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", "linux-vulkan-upstream"),
        (f"llama-{tag}-bin-win-vulkan-x64.zip",       "windows-vulkan-upstream"),
        (f"llama-{tag}-bin-ubuntu-arm64.tar.gz",      "linux-arm64-upstream"),
        (f"llama-{tag}-bin-win-cpu-arm64.zip",        "windows-arm64-upstream"),
    ):
        if name not in assets:
            print(f"WARNING: upstream asset {name} not found at {tag}; skipping", file=sys.stderr)
            continue
        wanted.append((name, kind))
    for name, kind in wanted:
        sha_artifacts[name] = base_entry(kind, UPSTREAM_REPO, asset_digest_or_hash(assets[name], token))

    # 3) source tarballs: codeload doesn't expose pre-computed digests, so
    #    stream-hash both URLs in parallel.
    source_jobs = [
        (f"llama.cpp-source-{tag}.tar.gz", "upstream-source",
         f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/refs/tags/{tag}"),
        (f"llama.cpp-source-commit-{commit}.tar.gz", "exact-source",
         f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{commit}"),
    ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        source_digests = list(pool.map(lambda j: sha256_url(j[2], token), source_jobs))
    for (name, kind, _url), digest in zip(source_jobs, source_digests):
        sha_artifacts[name] = base_entry(kind, UPSTREAM_REPO, digest)

    # 4) manifest, then hash it into the index
    manifest = build_manifest(tag, commit, found, rocm_found)
    manifest_path = args.out / "llama-prebuilt-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    sha_artifacts["llama-prebuilt-manifest.json"] = base_entry(
        "published-manifest", args.publish_repo, sha256_file(manifest_path)
    )

    sha256_doc = {
        "artifacts": sha_artifacts,
        "component": "llama.cpp",
        "release_tag": tag,
        "requested_source_ref": tag,
        "resolved_source_ref": tag,
        "schema_version": 1,
        "source_commit": commit,
        "source_commit_short": short,
        "source_ref_kind": "tag",
        "source_repo": UPSTREAM_REPO,
        "source_repo_url": UPSTREAM_URL,
        "upstream_tag": tag,
    }
    (args.out / "llama-prebuilt-sha256.json").write_text(json.dumps(sha256_doc, indent=2))

    print(f"wrote manifest ({len(manifest['artifacts'])} artifacts) and sha256 index "
          f"({len(sha_artifacts)} entries) to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
