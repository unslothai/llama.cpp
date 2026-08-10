#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Tests for upload_release_assets.sh. Run: bash scripts/unsloth/test_upload_release_assets.sh
#
# Every case runs the real uploader against a stub `gh` that reproduces the
# failure modes seen against uploads.github.com: a PUT that wedges and never
# returns, a transient 5xx, an asset committed at the wrong size, an asset left
# in a non-uploaded state, and a `gh` that exits 0 without the asset landing.
# Budgets are shrunk so a stall is killed in ~1s instead of ~500s.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/upload_release_assets.sh"
STUBDIR="$(mktemp -d)"
trap 'rm -rf "$STUBDIR"' EXIT
mkdir -p "$STUBDIR/bin"

cat > "$STUBDIR/bin/gh" <<'STUB'
#!/usr/bin/env bash
# Stub gh: serves `release upload` and `release view` off a file registry.
set -uo pipefail
REG="${STUB_REG:?}"; mkdir -p "$REG/assets" "$REG/attempts"

in_list() { case " ${2:-} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

if [ "$1" = "release" ] && [ "$2" = "upload" ]; then
  file="${*: -1}"; size="$(stat -c %s "$file")"
  # GitHub sanitises the asset name server-side; mirror that here so the test
  # exercises the same name mapping the uploader has to verify against.
  name="$(printf '%s' "$(basename "$file")" | tr -c 'A-Za-z0-9._-' '.')"
  c="$REG/attempts/$name"; n=$(( $(cat "$c" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$c"

  if in_list "$name" "${STUB_STALL_ALWAYS:-}"; then sleep 300; exit 0; fi
  if in_list "$name" "${STUB_STALL_ONCE:-}" && [ "$n" -le 1 ]; then sleep 300; exit 0; fi
  if in_list "$name" "${STUB_FAIL_ALWAYS:-}"; then echo "stub: HTTP 502" >&2; exit 1; fi
  if in_list "$name" "${STUB_FAIL_ONCE:-}" && [ "$n" -le 1 ]; then echo "stub: HTTP 502" >&2; exit 1; fi
  # Below: exits 0, but leaves the release in a state the caller must catch.
  if in_list "$name" "${STUB_WRONGSIZE:-}" && [ "$n" -le 1 ]; then
    printf '%s\t%s\tuploaded\n' "$name" "$(( size - 1 ))" > "$REG/assets/$name"; exit 0
  fi
  if in_list "$name" "${STUB_NOTUPLOADED:-}" && [ "$n" -le 1 ]; then
    printf '%s\t%s\tstarter\n' "$name" "$size" > "$REG/assets/$name"; exit 0
  fi
  if in_list "$name" "${STUB_SILENT_DROP:-}"; then exit 0; fi
  printf '%s\t%s\tuploaded\n' "$name" "$size" > "$REG/assets/$name"
  exit 0
fi

if [ "$1" = "release" ] && [ "$2" = "view" ]; then
  if [ "${STUB_VIEW_FAILS:-}" = 1 ]; then echo "stub: HTTP 503" >&2; exit 1; fi
  jqexpr=""
  for ((i=1;i<=$#;i++)); do
    if [ "${!i}" = "--jq" ]; then j=$((i+1)); jqexpr="${!j}"; fi
  done
  { echo '{"assets":['
    first=1
    for a in "$REG"/assets/*; do
      [ -e "$a" ] || continue
      IFS=$'\t' read -r n s st < "$a"
      if [ "$first" = 1 ]; then first=0; else echo ','; fi
      printf '{"name":"%s","size":%s,"state":"%s"}' "$n" "$s" "$st"
    done
    echo ']}'; } | jq -r "$jqexpr"
  exit 0
fi
echo "stub: unhandled: $*" >&2; exit 64
STUB
chmod +x "$STUBDIR/bin/gh"
export PATH="$STUBDIR/bin:$PATH"

export UPLOAD_GRACE_SECONDS=1 UPLOAD_MIN_RATE_MB_S=1000 UPLOAD_HEARTBEAT_SECONDS=3
export UPLOAD_JOBS=4 UPLOAD_ATTEMPTS=3

FAILS=()

run_case() { # name expected_rc [env ...]
  local name="$1" want="$2"; shift 2
  local d; d="$(mktemp -d)"; mkdir -p "$d/dist"
  local i
  for i in 01 02 03 04 05 06; do head -c 1000000 /dev/zero > "$d/dist/bundle-$i.tar.gz"; done
  # A name GitHub will rewrite, so the verify path's name mapping is covered.
  head -c 1000 /dev/zero > "$d/dist/has space.json"
  local out rc
  out="$(STUB_REG="$d/reg" env "$@" bash "$SCRIPT" --tag T --repo o/r --dist "$d/dist" 2>&1)"; rc=$?
  if [ "$rc" = "$want" ]; then
    printf 'PASS  %s\n' "$name"
  else
    printf 'FAIL  %s  :: rc=%s want=%s\n' "$name" "$rc" "$want"
    printf '%s\n' "$out" | sed 's/^/      | /'
    FAILS+=("$name")
  fi
  rm -rf "$d"
}

run_case "happy path"                    0 IGNORED=1
run_case "stall, recovers on retry"      0 STUB_STALL_ONCE="bundle-02.tar.gz bundle-05.tar.gz"
run_case "transient 502, recovers"       0 STUB_FAIL_ONCE="bundle-03.tar.gz"
run_case "permanent stall aborts"        1 STUB_STALL_ALWAYS="bundle-04.tar.gz"
run_case "permanent 502 aborts"          1 STUB_FAIL_ALWAYS="bundle-01.tar.gz"
run_case "wrong size caught, re-uploaded" 0 STUB_WRONGSIZE="bundle-06.tar.gz"
run_case "non-uploaded state re-uploaded" 0 STUB_NOTUPLOADED="bundle-02.tar.gz"
run_case "gh exits 0, asset never lands"  1 STUB_SILENT_DROP="bundle-03.tar.gz"
run_case "phase deadline aborts"          1 UPLOAD_DEADLINE_MINUTES=0 STUB_STALL_ALWAYS="bundle-01.tar.gz"
# The verify read must fail the script, not just its subshell: a process
# substitution would swallow this and report every asset as verified.
run_case "verification API outage aborts" 1 STUB_VIEW_FAILS=1

echo
echo "${#FAILS[@]} failure(s)${FAILS[*]:+: ${FAILS[*]}}"
[ "${#FAILS[@]}" -eq 0 ]
