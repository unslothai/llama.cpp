#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Wait inside a running job for other jobs of this run. Prebuilt jobs all start after `resolve`
# (one CI janitor pass covers the run), so consumers wait here instead of on `needs:`.
# Usage: wait_for_jobs.sh [--match S] [--exclude S]... [--require S]... [--expect N] [--deadline-min M]
#        [--poll S] [--out FILE]   (FILE gets name<TAB>status<TAB>conclusion; needs GH_TOKEN with actions:read)
# --require: a job whose name contains S must exist (startup race). Exit 0 when all matching jobs completed, 3 at the deadline, 4 after 10 API failures in a row.
set -uo pipefail

MATCH=""; EXCLUDES=(); REQUIRES=(); EXPECT=0; DEADLINE_MIN=330; POLL=60; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --match) MATCH="$2"; shift 2 ;;
    --exclude) EXCLUDES+=("$2"); shift 2 ;;
    --require) REQUIRES+=("$2"); shift 2 ;;
    --expect) EXPECT="$2"; shift 2 ;;
    --deadline-min) DEADLINE_MIN="$2"; shift 2 ;;
    --poll) POLL="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
case "$EXPECT$DEADLINE_MIN$POLL" in *[!0-9]*) echo "--expect/--deadline-min/--poll take integers" >&2; exit 2 ;; esac

repo="${GITHUB_REPOSITORY:?}"; run="${GITHUB_RUN_ID:?}"
deadline=$(( $(date +%s) + DEADLINE_MIN * 60 ))
fails=0
excl="$(printf '%s\n' "${EXCLUDES[@]+"${EXCLUDES[@]}"}")"

while :; do
  # filter=latest: a re-run attempt sees the jobs it would actually wait on.
  if ! jobs="$(gh api --paginate "repos/$repo/actions/runs/$run/jobs?per_page=100&filter=latest" \
        --jq '.jobs[] | [.name, .status, (.conclusion // "none")] | @tsv' 2>/dev/null)"; then
    fails=$(( fails + 1 ))
    if [ "$fails" -ge 10 ]; then
      echo "::warning::jobs API failed 10 times running; giving up the wait"
      exit 4
    fi
    echo "jobs API call failed ($fails/10); retrying in ${POLL}s"
    sleep "$POLL"; continue
  fi
  fails=0
  sel="$(printf '%s\n' "$jobs" | awk -F'\t' -v m="$MATCH" -v ex="$excl" '
    BEGIN { n = split(ex, X, "\n") }
    !NF { next }
    m != "" && index($1, m) == 0 { next }
    { for (i = 1; i <= n; i++) if (X[i] != "" && index($1, X[i]) > 0) next; print }')"
  total="$(printf '%s\n' "$sel" | grep -c . || true)"
  pending="$(printf '%s\n' "$sel" | awk -F'\t' 'NF && $2 != "completed"' | grep -c . || true)"
  missing=0
  for r in "${REQUIRES[@]+"${REQUIRES[@]}"}"; do
    printf '%s\n' "$sel" | awk -F'\t' -v r="$r" 'index($1, r) { f = 1 } END { exit !f }' || missing=$(( missing + 1 ))
  done
  if [ "$total" -ge "$EXPECT" ] && [ "$pending" -eq 0 ] && [ "$missing" -eq 0 ]; then
    echo "all $total matching job(s) completed"
    printf '%s\n' "$sel" | awk -F'\t' 'NF { printf "  %-70s %s\n", $1, $3 }'
    [ -n "$OUT" ] && printf '%s\n' "$sel" > "$OUT"
    exit 0
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "::warning::waited ${DEADLINE_MIN} min; ${pending} of ${total} matching job(s) still running (expected >= ${EXPECT}, ${missing} required missing)"
    [ -n "$OUT" ] && printf '%s\n' "$sel" > "$OUT"
    exit 3
  fi
  echo "$(date -u +%H:%M:%S) waiting: ${pending} of ${total} matching job(s) running (expected >= ${EXPECT}, ${missing} required missing)"
  sleep "$POLL"
done
