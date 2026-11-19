#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Rewrite a source archive in place without the files an install never reads. Every
# install downloads the published archive, so it must stay small; build jobs keep the full tree.
#
# Usage: slim_source_archive.sh <archive.tar.gz>
set -euo pipefail

archive="${1:?usage: slim_source_archive.sh <archive.tar.gz>}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

tar -xzf "$archive" -C "$work"
root="$(find "$work" -mindepth 1 -maxdepth 1 -type d)"
name="$(basename "$root")"

# GGUF conversion loads these from models/ at runtime, so read the names rather than pin them.
builtin_vocabs="$(grep -rhoE '_set_vocab_builtin\("[^"]+"' "$root/conversion" "$root/convert_hf_to_gguf.py" \
  | sed -E 's/.*"([^"]+)"/\1/' | sort -u || true)"
[ -n "$builtin_vocabs" ] || { echo "no _set_vocab_builtin callers found in $name" >&2; exit 1; }

keep=()
for vocab in $builtin_vocabs; do
  file="ggml-vocab-$vocab.gguf"
  [ -f "$root/models/$file" ] || { echo "conversion needs models/$file, missing from $name" >&2; exit 1; }
  keep+=(! -name "$file")
done

# tests/ and pocs/ stay: CMakeLists.txt adds them, so a plain configure fails without them.
rm -rf "$root/docs" "$root/media" "$root/benches"
find "$root/models" -maxdepth 1 -type f -name 'ggml-vocab-*' "${keep[@]}" -delete

# A re-run of assemble must reproduce the bytes of any asset it already uploaded.
touch -r "$root/CMakeLists.txt" "$root" "$root/models"
tar --sort=name --owner=0 --group=0 --numeric-owner -C "$work" -cf - "$name" | gzip -9n > "$archive.slim"
mv "$archive.slim" "$archive"
