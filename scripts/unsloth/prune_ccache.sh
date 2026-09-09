#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Prune superseded ccache generations from this repository's Actions cache.
#
# Entries are immutable, so every run writes a NEW cache per (backend, os, profile) and
# restore-keys returns only the newest match: older generations are landfill. At the repo limit
# GitHub evicts by its own LRU and can take a LIVE cache; a partial cache is worse than none
# (a cache capped to 30% of need hit 14.2% and cost a 204-minute build instead of 54).
#
# Runs after `resolve` and again after publish. One generation per prefix is safe at run start
# because the prefix restore only returns the newest entry; peak usage during a run is
# old + new. Keys that are not ccache-* are reported, never touched.
#
# Usage: prune_ccache.sh [--keep N] [--limit-gb N] [--dry-run]
#   GH_TOKEN with actions:write on GITHUB_REPOSITORY; writes GITHUB_STEP_SUMMARY if set.
set -euo pipefail

KEEP=1; LIMIT_GB=75; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --keep) KEEP="$2"; shift 2 ;;
    --limit-gb) LIMIT_GB="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$KEEP" in [1-9]|[1-9][0-9]) ;; *) echo "invalid --keep $KEEP" >&2; exit 2 ;; esac

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
tmp="$(mktemp -d)"
all="$tmp/caches.tsv"
gh api --paginate "repos/$repo/actions/caches?per_page=100" \
  -q '.actions_caches[] | "\(.id)\t\(.created_at)\t\(.size_in_bytes)\t\(.ref)\t\(.version)\t\(.key)"' > "$all" || true

gib() { awk -v b="${1:-0}" 'BEGIN { printf "%.1f", b / 1073741824 }'; }
total=$(awk -F'\t' '{s+=$3} END {printf "%d", s+0}' "$all")
cc_total=$(awk -F'\t' '$6 ~ /^ccache-/ {s+=$3} END {printf "%d", s+0}' "$all")
sc_total=$(awk -F'\t' '$6 !~ /^ccache-/ {s+=$3} END {printf "%d", s+0}' "$all")
sc_count=$(awk -F'\t' '$6 !~ /^ccache-/' "$all" | grep -c . || true)
echo "$(grep -c . "$all" || true) caches, $(gib "$total") GiB total: ccache $(gib "$cc_total") GiB, other (sccache etc.) $(gib "$sc_total") GiB in $sc_count entries"

# Group by restore prefix: the key minus -<tag>- and -<run id>-<attempt>. The ROCm toolchain
# version stays in because the restore prefix has it: a rocm_version=latest dispatch before the
# weekly nightly would otherwise leave only a cache the weekly legs cannot restore (retired
# toolchains age out via GitHub's 7-day expiry). Per (ref, version) too: a branch cannot restore a sibling's cache.
grouped="$tmp/grouped.tsv"; : > "$grouped"
while IFS=$'\t' read -r id created size ref ver key; do
  [ -z "${id:-}" ] && continue
  case "$key" in ccache-*) ;; *) continue ;; esac
  pre="$(printf '%s' "$key" | sed -E 's/-[0-9]+-[0-9]+$//; s/-b[0-9]+(-mix-[0-9a-f]+)?-?$//')"
  [ "$pre" = "$key" ] && continue
  printf '%s\t%s\t%s\t%s\n' "$ref|$ver|$pre" "$created" "$id" "$size" >> "$grouped"
done < "$all"
sort -t"$(printf '\t')" -k1,1 -k2,2r -o "$grouped" "$grouped"

prev=""; n=0; freed=0; deleted=0
while IFS=$'\t' read -r pre created id size; do
  [ -z "${pre:-}" ] && continue
  if [ "$pre" != "$prev" ]; then prev="$pre"; n=1; else n=$(( n + 1 )); fi
  [ "$n" -le "$KEEP" ] && continue
  if [ "$DRY" = 1 ]; then
    echo "would delete $id ($(gib "$size") GiB) ${pre##*|}"
    freed=$(( freed + size )); deleted=$(( deleted + 1 )); continue
  fi
  # </dev/null or gh eats the loop's stdin and the sweep stops after one.
  if gh api -X DELETE "repos/$repo/actions/caches/$id" --silent < /dev/null 2>/dev/null; then
    freed=$(( freed + size )); deleted=$(( deleted + 1 ))
  fi
done < "$grouped"

after=$(( total - freed ))
verb="pruned"; [ "$DRY" = 1 ] && verb="would prune"
echo "$verb $deleted superseded ccache generations, $(gib "$freed") GiB; $(gib "$after") GiB of $LIMIT_GB GB remains"
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### ccache budget"
    echo ""
    echo "| metric | value |"
    echo "| --- | --- |"
    echo "| kept per prefix | $KEEP |"
    echo "| ccache generations $verb | $deleted |"
    echo "| freed | $(gib "$freed") GiB |"
    echo "| ccache after | $(gib $(( cc_total - freed ))) GiB |"
    echo "| sccache and other entries | $(gib "$sc_total") GiB in $sc_count |"
    echo "| cache total after | $(gib "$after") GiB of $LIMIT_GB GB |"
  } >> "$GITHUB_STEP_SUMMARY"
fi
# 80% of the limit: above this the next generation lands on GitHub's LRU.
if [ "$after" -gt $(( LIMIT_GB * 1000000000 * 8 / 10 )) ]; then
  echo "::warning::cache total is $(gib "$after") GiB of the $LIMIT_GB GB repo limit; at the limit GitHub evicts by LRU and a partially evicted ccache collapses the hit rate. Raise the limit or shrink the CUDA max-size caps."
fi
