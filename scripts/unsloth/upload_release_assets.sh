#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Upload a directory of files to a draft release, then check the release really
# holds them before the caller flips draft=false.
# Run: upload_release_assets.sh --tag TAG --repo OWNER/REPO --dist DIR
#
# `gh release create ... dist/*` uploads with a fixed 5-worker pool and no
# per-connection timeout, so a few wedged PUTs block everything behind them. In
# run 31335302864 six large bundles stalled at ~0.03 MB/s and held the pool for
# 3h45m, while the other 25 assets took 37s in total. The job finished 25
# minutes short of its 350m cap.
#
# So: bound each attempt, bound the whole phase, and verify against the API
# before we publish.
set -euo pipefail

# Defaults come from measured healthy throughput on the runners, 33-47 MB/s.
JOBS="${UPLOAD_JOBS:-4}"
ATTEMPTS="${UPLOAD_ATTEMPTS:-4}"
# Slowest rate we still call "progressing". ~17x below healthy, so a bad day
# retries nothing but a wedged PUT dies fast.
MIN_RATE_MB_S="${UPLOAD_MIN_RATE_MB_S:-2}"
# Per-attempt allowance for setup and server-side commit, which do not scale
# with file size.
GRACE_SECONDS="${UPLOAD_GRACE_SECONDS:-120}"
# Healthy is ~4 minutes for 7.2 GB, so only a real pathology trips this.
DEADLINE_MINUTES="${UPLOAD_DEADLINE_MINUTES:-90}"
HEARTBEAT_SECONDS="${UPLOAD_HEARTBEAT_SECONDS:-60}"

log() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { printf '%s ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }

file_size() { stat -c %s "$1"; }

# GitHub rewrites characters outside [A-Za-z0-9._-] to '.', so verify against
# the name the API will report. printf, not basename: basename's trailing
# newline is also outside the set and would become a phantom '.' on every name.
asset_name() { printf '%s' "${1##*/}" | tr -c 'A-Za-z0-9._-' '.'; }

# Worker mode. The parent fans out with xargs by re-invoking this script, which
# is safer than `export -f`: an unexported function fails per file at runtime.
if [ "${1:-}" = "--upload-one" ]; then
  f="$2"
  : "${TAG:?}" "${REPO:?}" "${UPLOAD_DEADLINE_EPOCH:?}"
  name="$(asset_name "$f")"
  bytes="$(file_size "$f")"
  mb=$(( bytes / 1000000 ))
  budget=$(( GRACE_SECONDS + mb / MIN_RATE_MB_S ))

  for attempt in $(seq 1 "$ATTEMPTS"); do
    now="$(date +%s)"
    if [ "$now" -ge "$UPLOAD_DEADLINE_EPOCH" ]; then
      die "deadline reached before uploading $name"
    fi
    # Keep one file's budget inside the phase deadline. Floor at 1: `timeout 0`
    # means no timeout, which brings back the hang this script prevents.
    remaining=$(( UPLOAD_DEADLINE_EPOCH - now ))
    this_budget="$budget"
    if [ "$this_budget" -gt "$remaining" ]; then this_budget="$remaining"; fi
    if [ "$this_budget" -lt 1 ]; then this_budget=1; fi

    start="$now"
    # --clobber keeps a retry after a killed upload idempotent, else GitHub
    # 422s on the duplicate name. Take the status here, not from $? after an
    # `if`: a false `if` with no else exits 0, so every stall would read clean.
    rc=0
    timeout -k 30 "$this_budget" gh release upload "$TAG" --repo "$REPO" --clobber "$f" || rc=$?
    elapsed=$(( $(date +%s) - start ))
    if [ "$rc" -eq 0 ]; then
      if [ "$elapsed" -lt 1 ]; then elapsed=1; fi
      log "uploaded $name (${mb} MB in ${elapsed}s, $(( mb / elapsed )) MB/s, attempt ${attempt})"
      exit 0
    fi
    if [ "$rc" -ge 124 ]; then
      log "STALLED $name: no completion in ${elapsed}s (budget ${this_budget}s, ${mb} MB); attempt ${attempt}/${ATTEMPTS}"
    else
      log "FAILED $name: gh exit ${rc} after ${elapsed}s; attempt ${attempt}/${ATTEMPTS}"
    fi
    if [ "$attempt" -eq "$ATTEMPTS" ]; then
      die "gave up on $name after ${ATTEMPTS} attempts"
    fi
    sleep $(( attempt * 15 ))
  done
  exit 1
fi

# Parent mode.
TAG="" REPO="" DIST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --tag)  TAG="$2";  shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --dist) DIST="$2"; shift 2 ;;
    *) die "unknown argument: $1" ;;
  esac
done
[ -n "$TAG" ]  || die "--tag is required"
[ -n "$REPO" ] || die "--repo is required"
[ -n "$DIST" ] || die "--dist is required"
[ -d "$DIST" ] || die "dist directory not found: $DIST"

# NUL-delimited: a name with a space would otherwise split into two bad paths.
mapfile -d '' -t FILES < <(find "$DIST" -maxdepth 1 -type f -print0 | sort -z)
[ "${#FILES[@]}" -gt 0 ] || die "no files to upload in $DIST"

total_bytes=0
for f in "${FILES[@]}"; do total_bytes=$(( total_bytes + $(file_size "$f") )); done
log "uploading ${#FILES[@]} assets ($(( total_bytes / 1000000 )) MB) to draft $TAG with ${JOBS} workers"

UPLOAD_DEADLINE_EPOCH=$(( $(date +%s) + DEADLINE_MINUTES * 60 ))
export TAG REPO UPLOAD_DEADLINE_EPOCH JOBS ATTEMPTS MIN_RATE_MB_S GRACE_SECONDS

self="$(readlink -f "$0")"

# The incident was 4 hours of silence, so report what the API has accepted.
heartbeat() {
  while sleep "$HEARTBEAT_SECONDS"; do
    n="$(gh release view "$TAG" --repo "$REPO" --json assets --jq '[.assets[]|select(.state=="uploaded")]|length' 2>/dev/null || echo '?')"
    log "heartbeat: ${n}/${#FILES[@]} assets uploaded, $(( (UPLOAD_DEADLINE_EPOCH - $(date +%s)) / 60 ))m left in budget"
  done
}
heartbeat & hb_pid=$!
trap 'kill "$hb_pid" 2>/dev/null || true' EXIT

upload_pass() {
  # Run through `bash`, so a lost exec bit cannot break the publish.
  printf '%s\0' "$@" | xargs -0 -P "$JOBS" -n 1 bash "$self" --upload-one
}

pass_rc=0
upload_pass "${FILES[@]}" || pass_rc=$?
[ "$pass_rc" -eq 0 ] || log "upload pass reported failures (xargs exit ${pass_rc}); verification decides"

# gh exiting 0 does not prove the asset is complete, so set BAD to every local
# file the release does not hold at the same size and state "uploaded". Read the
# API here, not inside `< <(...)`, where a failed read exits only the subshell
# and leaves BAD empty, i.e. publishes a release we never checked.
verify() {
  local remote f
  remote="$(gh release view "$TAG" --repo "$REPO" --json assets \
    --jq '.assets[] | select(.state=="uploaded") | "\(.name)\t\(.size)"')" \
    || die "could not read release assets for verification"
  BAD=()
  for f in "${FILES[@]}"; do
    if ! grep -qxF "$(asset_name "$f")	$(file_size "$f")" <<<"$remote"; then
      BAD+=("$f")
    fi
  done
}

verify
if [ "${#BAD[@]}" -gt 0 ]; then
  log "verification found ${#BAD[@]} missing or mismatched assets; re-uploading"
  for f in "${BAD[@]}"; do log "  - $(asset_name "$f")"; done
  upload_pass "${BAD[@]}" || true
  verify
fi

if [ "${#BAD[@]}" -gt 0 ]; then
  for f in "${BAD[@]}"; do printf 'ERROR: asset never landed: %s\n' "$(asset_name "$f")" >&2; done
  die "refusing to publish: ${#BAD[@]}/${#FILES[@]} assets missing after re-upload"
fi

log "verified all ${#FILES[@]} assets present, sized and uploaded"
