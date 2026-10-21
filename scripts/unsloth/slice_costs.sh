#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Per-object compile ms for slice_objects.sh, kept in the ccache dir so a leg and its warm jobs
# read the same file. Warm jobs export their slice into slice-costs.d/; the leg merges all into
# slice-costs.tsv. Only a timing over HIT_MS (a real compile, not a ccache hit) replaces a cost, and
# objects missing from the timings (a failed build) keep theirs.
# Usage: slice_costs.sh export <build dir> <out.tsv> | merge <costs.tsv> <timings.tsv>...
set -euo pipefail
HIT_MS="${HIT_MS:-3000}"
tab="$(printf '\t')"

case "${1:-}" in
  export)
    dir="${2:?build dir}"; out="${3:?output file}"
    log="$dir/.ninja_log"
    [ -f "$log" ] || { echo "no $log; nothing to export" >&2; : > "$out"; exit 0; }
    # .ninja_log: start_ms end_ms mtime output hash; the last record wins.
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
      END { for (o in prev) if (!(o in seen)) printf "%d\t%s\n", prev[o], o
            for (o in seen) printf "%d\t%s\n", (o in real) ? real[o] : ((o in prev) ? prev[o] : any[o]), o }
    ' "$old" "$@" | LC_ALL=C sort -t "$tab" -k2,2 > "$costs.tmp"
    mv "$costs.tmp" "$costs"
    echo "slice costs: $(grep -c . "$costs" || true) objects in $costs" >&2
    ;;
  *)
    echo "usage: slice_costs.sh export <build dir> <out.tsv> | merge <costs.tsv> <timings.tsv>..." >&2
    exit 2
    ;;
esac
