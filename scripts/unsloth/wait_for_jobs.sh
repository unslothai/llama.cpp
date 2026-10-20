#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Spin-wait inside a running job until other jobs of this run have finished.
#
# Every prebuilt job is created right after `resolve` instead of behind `needs:`, so the whole
# run is queued at once and one CI janitor pass covers it (unslothai/* share a busy runner pool;
# a job GitHub creates hours in sees that queue fresh). The jobs that consume other jobs' output
# (the CUDA build merging its warm shards, reclaim, alert) start with everything else and wait
# here instead.
#
# Usage: wait_for_jobs.sh [--match SUBSTR] [--exclude SUBSTR]... [--expect N]
#                         [--deadline-min M] [--poll S] [--out FILE]
#   --match    only jobs whose name contains SUBSTR (default: every job)
#   --exclude  drop jobs whose name contains SUBSTR (repeatable; give your own job's name)
#   --expect   keep waiting until at least N matching records exist
#   --out      write "name<TAB>status<TAB>conclusion" of the matching jobs when done
# Exit 0 once every matching job is completed, 3 on the deadline, 4 if the jobs API failed
# 10 times running. Callers choose whether a timeout is fatal.
# Needs GH_TOKEN with actions:read, GITHUB_REPOSITORY and GITHUB_RUN_ID.
set -uo pipefail

MATCH=""; EXCLUDES=(); EXPECT=0; DEADLINE_MIN=330; POLL=60; OUT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --match) MATCH="$2"; shift 2 ;;
    --exclude) EXCLUDES+=("$2"); shift 2 ;;
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
  if [ "$total" -ge "$EXPECT" ] && [ "$pending" -eq 0 ]; then
    echo "all $total matching job(s) completed"
    printf '%s\n' "$sel" | awk -F'\t' 'NF { printf "  %-70s %s\n", $1, $3 }'
    [ -n "$OUT" ] && printf '%s\n' "$sel" > "$OUT"
    exit 0
  fi
  if [ "$(date +%s)" -gt "$deadline" ]; then
    echo "::warning::waited ${DEADLINE_MIN} min; ${pending} of ${total} matching job(s) still running (expected >= ${EXPECT})"
    [ -n "$OUT" ] && printf '%s\n' "$sel" > "$OUT"
    exit 3
  fi
  echo "$(date -u +%H:%M:%S) waiting: ${pending} of ${total} matching job(s) running (expected >= ${EXPECT})"
  sleep "$POLL"
done
