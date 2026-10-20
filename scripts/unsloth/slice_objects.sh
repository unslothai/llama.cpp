#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Print slice K of N of the objects a configured ninja build compiles for `all`.
#
# Every object, not just ggml-cuda/ggml-hip: after a core-header edit the CPU variants, llama,
# common and the tools miss too (6-7 min of a cold CUDA leg), and slicing them leaves the
# merging job little but hits and the link. Objects come from the compile commands of `all`
# (-o for gcc/clang/nvcc, /Fo for MSVC), so nothing outside the real build is compiled.
# With a costs file (slice_costs.sh: per-object compile ms from earlier builds) the objects are
# dealt longest first, each to the least-loaded slice, so every slice gets about total/N of work;
# objects without a cost count as the mean. Without one, sorted and dealt round-robin
# (line % N == K): the heavy template instances sit together and spread across slices. Every job
# of a leg must pass the same costs file (they restore the same ccache generation) to get
# disjoint slices; a mismatch only means the leg compiles what a shard missed.
# Paths are printed with forward slashes: MSVC and nvcc on Windows write backslashes, which
# xargs and shells eat as escapes, and ninja canonicalises either form to the same target.
#
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
