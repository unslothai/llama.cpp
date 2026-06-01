#!/usr/bin/env python3
"""Generate the Unsloth Studio llama.cpp release manifest + checksum assets.

Writes llama-prebuilt-manifest.json and llama-prebuilt-sha256.json in the exact
schema studio/install_llama_prebuilt.py consumes (parse_published_release_bundle /
parse_approved_release_checksums). Used by unsloth-macos-prebuilt.yml after the
macOS slices are built and validated, before the release is published.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
from pathlib import Path

# install_kind -> (manifest bundle_profile, sha256 kind tag)
SLICE_PROFILES = {
    "macos-arm64": ("macos-metal-arm64", "macos-arm64-app"),
    "macos-x64": ("macos-cpu-x64", "macos-x64-app"),
}
MANIFEST_ASSET = "llama-prebuilt-manifest.json"
SHA256_ASSET = "llama-prebuilt-sha256.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_slice(value: str) -> tuple[str, Path]:
    # "macos-arm64:/path/to/llama-bNNNN-bin-macos-arm64.tar.gz"
    kind, _, path = value.partition(":")
    if kind not in SLICE_PROFILES or not path:
        raise argparse.ArgumentTypeError(f"invalid --artifact {value!r}")
    return kind, Path(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--upstream-tag", required=True, help="e.g. b9439")
    ap.add_argument("--source-commit", required=True, help="full 40-hex commit sha")
    ap.add_argument("--release-tag", required=True, help="fork release tag")
    ap.add_argument("--source-repo", default="ggml-org/llama.cpp")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument(
        "--artifact",
        required=True,
        action="append",
        type=parse_slice,
        metavar="KIND:PATH",
        help="repeatable, e.g. macos-arm64:llama-b9439-bin-macos-arm64.tar.gz",
    )
    ap.add_argument("--source-archive", type=Path, help="llama.cpp-source-<tag>.tar.gz")
    ap.add_argument(
        "--exact-source-archive",
        type=Path,
        help="llama.cpp-source-commit-<sha>.tar.gz",
    )
    args = ap.parse_args()

    commit = args.source_commit.strip().lower()
    short = commit[:7]
    repo_url = f"https://github.com/{args.source_repo}"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    source_fields = {
        "source_repo": args.source_repo,
        "source_repo_url": repo_url,
        "source_ref_kind": "tag",
        "requested_source_ref": args.upstream_tag,
        "resolved_source_ref": args.upstream_tag,
        "source_commit": commit,
        "source_commit_short": short,
    }

    # Manifest: one artifacts[] entry per built slice.
    manifest_artifacts = []
    checksum_artifacts: dict[str, dict] = {}
    for kind, path in args.artifact:
        if not path.is_file():
            ap.error(f"artifact file not found: {path}")
        profile, hash_kind = SLICE_PROFILES[kind]
        name = path.name
        manifest_artifacts.append(
            {
                "asset_name": name,
                "install_kind": kind,
                "bundle_profile": profile,
                "runtime_line": None,
                "coverage_class": None,
                "rank": 50,
            }
        )
        checksum_artifacts[name] = {
            "sha256": sha256_file(path),
            "repo": "unslothai/llama.cpp",
            "kind": hash_kind,
            "upstream_tag": args.upstream_tag,
            "source_commit": commit,
            "source_commit_short": short,
        }

    manifest = {
        "schema_version": 1,
        "component": "llama.cpp",
        "upstream_repo": args.source_repo,
        "upstream_tag": args.upstream_tag,
        "generated_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
        **source_fields,
        "artifacts": manifest_artifacts,
    }
    manifest_path = out / MANIFEST_ASSET
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    # Source archives let the source-build fallback resolve an approved tree.
    # The exact-commit entry alone satisfies validated_checksums_for_bundle.
    if args.source_archive and args.source_archive.is_file():
        checksum_artifacts[f"llama.cpp-source-{args.upstream_tag}.tar.gz"] = {
            "sha256": sha256_file(args.source_archive),
            "repo": args.source_repo,
            "kind": "upstream-source",
        }
    if args.exact_source_archive and args.exact_source_archive.is_file():
        checksum_artifacts[f"llama.cpp-source-commit-{commit}.tar.gz"] = {
            "sha256": sha256_file(args.exact_source_archive),
            "repo": args.source_repo,
            "kind": "exact-source",
        }

    # The manifest's own sha is cross-checked by validated_checksums_for_bundle,
    # so hash it after it is written and add it to the checksum asset.
    checksum_artifacts[MANIFEST_ASSET] = {
        "sha256": sha256_file(manifest_path),
        "repo": "unslothai/llama.cpp",
        "kind": "published-manifest",
    }

    checksums = {
        "schema_version": 1,
        "component": "llama.cpp",
        "release_tag": args.release_tag,
        "upstream_tag": args.upstream_tag,
        **source_fields,
        "artifacts": checksum_artifacts,
    }
    (out / SHA256_ASSET).write_text(
        json.dumps(checksums, indent=2) + "\n", encoding="utf-8"
    )

    print(f"wrote {manifest_path}")
    print(f"wrote {out / SHA256_ASSET}")
    print(f"artifacts: {', '.join(sorted(checksum_artifacts))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
