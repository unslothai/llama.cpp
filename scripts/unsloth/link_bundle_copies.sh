#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# The ROCm copy step flattens soname links into full copies, so a bundle carries each large
# library three times. Turn identical copies back into links to the longest name.
#
# Usage: link_bundle_copies.sh <bundle-dir>
set -euo pipefail

cd "${1:?usage: link_bundle_copies.sh <bundle-dir>}"

declare -A kept
while IFS= read -r file; do
  key="${file%%.so*} $(sha256sum "$file" | cut -d' ' -f1)"
  if [ -n "${kept[$key]:-}" ]; then
    ln -sf "${kept[$key]}" "$file"
  else
    kept[$key]="$file"
  fi
done < <(find . -maxdepth 1 -type f -name 'lib*.so*' -printf '%f\n' \
  | awk '{ print length($0), $0 }' | sort -k1,1nr -k2,2 | cut -d' ' -f2-)
