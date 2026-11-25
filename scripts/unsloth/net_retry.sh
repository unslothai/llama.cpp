# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
#
# Sourced, not run: `. scripts/unsloth/net_retry.sh`. Bounded retries for the GitHub API and git
# remotes, so one 5xx, 429 or dropped connection does not fail a workflow step. Every step that
# reaches the network must carry a retry; check_workflow_network_retries.py enforces it, and it
# recognises any helper whose name contains "retry".
#
#   net_retry CMD...   CMD up to 4 times (3 retries, 15/30/60 s apart). For idempotent commands
#                      whose stdout is not consumed, or is consumed only on success
#                      (git fetch, git ls-remote, git push of an exact sha).
#   gh_retry ARGS...   `gh ARGS...` the same way, with two differences:
#                      * stdout is buffered and printed only for the attempt that succeeded, so a
#                        paginated read that died on page 3 never leaves pages 1-2 behind for the
#                        caller to parse twice;
#                      * a definitive answer is not retried: 400/401/404/410/422 and "not found"
#                        come back at once, so `gh release view` of a tag that does not exist still
#                        costs one call, and a typo'd PR number still fails in seconds. 403 and 429
#                        are retried, since GitHub reports its secondary rate limit as 403.
#
# Only the network command is retried. Never wrap a build, a test or a script that does more than
# one network call: a retry there re-runs work and can hide a real failure behind a later success.
# NET_RETRY_SLEEP overrides the base delay (tests use 0).

net_retry() {
  local i rc=0 base="${NET_RETRY_SLEEP:-15}"
  for i in 1 2 3 4; do
    "$@" && return 0
    rc=$?
    [ "$i" = 4 ] && break
    echo "net_retry: '$1' failed (exit $rc); attempt $i/4, retrying in $(( base * (1 << (i - 1)) ))s" >&2
    sleep $(( base * (1 << (i - 1)) ))
  done
  return "$rc"
}

gh_retry() {
  local i rc=0 out err base="${NET_RETRY_SLEEP:-15}"
  err="$(mktemp)"
  for i in 1 2 3 4; do
    # rc from the assignment itself: `$?` after an `if` with no else is 0, not gh's exit code.
    out="$(gh "$@" 2>"$err")" && rc=0 || rc=$?
    if [ "$rc" = 0 ]; then
      cat "$err" >&2
      rm -f "$err"
      [ -z "$out" ] || printf '%s\n' "$out"
      return 0
    fi
    cat "$err" >&2
    if grep -qE 'HTTP (400|401|404|410|422)\b|[Nn]ot [Ff]ound|Could not resolve to' "$err"; then
      break
    fi
    [ "$i" = 4 ] && break
    echo "gh_retry: 'gh $1 ${2:-}' failed (exit $rc); attempt $i/4, retrying in $(( base * (1 << (i - 1)) ))s" >&2
    sleep $(( base * (1 << (i - 1)) ))
  done
  rm -f "$err"
  return "$rc"
}
