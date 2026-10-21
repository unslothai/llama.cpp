#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Print slice K of N of the objects `all` compiles (host objects too: they miss after a core-header
# edit). With a costs file (slice_costs.sh) objects go longest first to the least-loaded slice, else
# round-robin. Every job of a leg must pass the same costs file or slices overlap/miss (only a
# slowdown: the leg builds whatever is left). Forward slashes: xargs eats MSVC/nvcc backslashes.
# Usage: slice_objects.sh <build dir> <K> <N> [costs.tsv]   (0 <= K < N)
set -euo pipefail
dir="${1:?build dir}"; k="${2:?slice}"; n="${3:?slices}"; costs="${4:-}"
case "$k$n" in *[!0-9]*) echo "slice and slices must be integers" >&2; exit 2 ;; esac
[ "$k" -lt "$n" ] || { echo "slice $k is outside 0..$(( n - 1 ))" >&2; exit 2; }
objs="$(ninja -C "$dir" -t commands all \
  | grep -oE '(-o |/Fo)[^ "]+\.(o|obj)( |$)' | sed -E 's#^(-o |/Fo)##; s# $##' | tr '\\' '/' | sort -u)"
[ -n "$objs" ] || { echo "no object targets found under $dir; refusing an empty slice list" >&2; exit 1; }
if [ -z "$costs" ] || [ ! -s "$costs" ]; then
  [ -z "$costs" ] || echo "no slice costs at $costs; dealing round-robin" >&2
  printf '%s\n' "$objs" | awk -v k="$k" -v n="$n" 'NR % n == k'
  exit 0
fi
tab="$(printf '\t')"
echo "slice costs: $costs (cksum $(cksum < "$costs" | cut -d' ' -f1))" >&2
printf '%s\n' "$objs" \
  | awk -F'\t' -v cf="$costs" '
      BEGIN { while ((getline l < cf) > 0) { split(l, f, "\t"); if (f[1] ~ /^[0-9]+$/ && f[2] != "") { c[f[2]] = f[1] + 0; s += f[1]; m++ } } }
      { o[NR] = $0 }
      END { avg = m ? int(s / m) : 1; for (i = 1; i <= NR; i++) { w = (o[i] in c) ? c[o[i]] : avg; known += (o[i] in c); printf "%d\t%s\n", w, o[i] }
            printf "costs known for %d of %d objects\n", known, NR > "/dev/stderr" }' \
  | LC_ALL=C sort -t "$tab" -k1,1nr -k2,2 \
  | awk -F'\t' -v k="$k" -v n="$n" '
      { b = 0; for (i = 1; i < n; i++) if (load[i] < load[b]) b = i
        load[b] += $1; total += $1; if (b == k) { print $2; cnt++ } }
      END { printf "slice %d of %d: %d objects, ~%.0f s of %.0f s compile\n", k, n, cnt, load[k] / 1000, total / 1000 > "/dev/stderr" }'
