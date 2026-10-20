#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Per-object compile times for slice_objects.sh, kept in the ccache dir so they travel with it.
#
# A leg and its warm jobs restore the same ccache generation, so they read the same costs file
# and deal identical slices. Each warm job exports the timings of the slice it compiled into its
# delta (slice-costs.d/); the leg folds those and its own build into slice-costs.tsv before the
# cache is saved, so the next run is balanced by what each object really cost.
#
# A ccache hit is not a compile time: an entry is replaced only by a timing over HIT_MS, and an
# object that only ever hit keeps its last real cost. Objects absent from this build drop out.
#
# Usage: slice_costs.sh export <build dir> <out.tsv>
#        slice_costs.sh merge  <costs.tsv> <timings.tsv>...
# Files are "<ms>\t<object>", objects relative to the build dir with forward slashes.
set -euo pipefail
HIT_MS="${HIT_MS:-3000}"
tab="$(printf '\t')"

case "${1:-}" in
  export)
    dir="${2:?build dir}"; out="${3:?output file}"
    log="$dir/.ninja_log"
    [ -f "$log" ] || { echo "no $log; nothing to export" >&2; : > "$out"; exit 0; }
    # .ninja_log v5+: start_ms end_ms mtime output hash. The last record for an output wins.
    awk -F'\t' '!/^#/ && NF >= 4 { p = $4; gsub(/\\/, "/", p); if (p ~ /\.(o|obj)$/) t[p] = $2 - $1 }
                END { for (p in t) printf "%d\t%s\n", t[p], p }' "$log" \
      | LC_ALL=C sort -t "$tab" -k2,2 > "$out"
    echo "exported $(grep -c . "$out" || true) object timings to $out" >&2
    ;;
  merge)
    costs="${2:?costs file}"; shift 2
    [ "$#" -gt 0 ] || { echo "no timings to merge" >&2; exit 0; }
    old="$costs"; [ -f "$old" ] || old=/dev/null
    awk -F'\t' -v hit="$HIT_MS" -v oldf="$old" '
      FILENAME == oldf { if (NF >= 2) prev[$2] = $1 + 0; next }
      NF >= 2 {
        seen[$2] = 1; v = $1 + 0
        if (v > hit && (!($2 in real) || v > real[$2])) real[$2] = v
        if (!($2 in any) || v > any[$2]) any[$2] = v
      }
      END { for (o in seen) printf "%d\t%s\n", (o in real) ? real[o] : ((o in prev) ? prev[o] : any[o]), o }
    ' "$old" "$@" | LC_ALL=C sort -t "$tab" -k2,2 > "$costs.tmp"
    mv "$costs.tmp" "$costs"
    echo "slice costs: $(grep -c . "$costs" || true) objects in $costs" >&2
    ;;
  *)
    echo "usage: slice_costs.sh export <build dir> <out.tsv> | merge <costs.tsv> <timings.tsv>..." >&2
    exit 2
    ;;
esac
