#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Decide whether Finalize keeps the artifacts of a run that published nothing.

Run 45410618008: one CPU leg lost a Chocolatey download, assemble refused to
publish, and Finalize deleted every artifact of the run, the merged source
tarball included. The automatic `rerun-failed-jobs` then reran that one leg,
which died at "Artifact not found for name: app-source-...", so a single
transient failure cost the whole release and a fresh 60-job dispatch.

The rule is now: a run that published nothing keeps its artifacts while a rerun
of its failed jobs could still finish it, and deletes them only when nothing is
left to retry:

  * resolve did not succeed: nothing was built, so a rerun starts from scratch;
  * no job failed or was cancelled (a publish:false test, a subset dispatch):
    nothing will ever consume the bundles;
  * this is attempt MAX_ATTEMPT or later: the retries are spent.

Kept artifacts are still bounded in time by the repository's artifact retention.

When the artifacts are kept, Finalize also fails on purpose (`rerun=true`).
`gh run rerun --failed` reruns failed jobs and their dependents, and Finalize
depends only on resolve, so a green Finalize would never run again: the
rerun's published bundles would not be reclaimed and the health alert opened by
attempt 1 would never clear.

Reads the waiter's jobs.tsv (name<TAB>status<TAB>conclusion, see
wait_for_jobs.sh) and prints `rerun=`, `keep=` and `reason=` lines.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Job names track unsloth-prebuilt.yml (and the retry workflow's filters).
RESOLVE = "Resolve tag"
# Best-effort jobs that never gate a publish, same exclusions as the assemble waiter.
PRUNE = "Prune superseded ccache generations"
WARM = " / warm "
# Conclusions a rerun of the failed jobs picks up again.
RERUNNABLE = {"failure", "cancelled", "timed_out", "startup_failure"}


def parse_jobs(text: str) -> list[tuple[str, str, str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        parts += [""] * (3 - len(parts))
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def decide(jobs: list[tuple[str, str, str]], attempt: int, max_attempt: int,
           keep_requested: bool) -> dict:
    """Return {'rerun': bool, 'keep': bool, 'reason': str} for an unpublished run."""
    resolve = [c for n, _, c in jobs if n == RESOLVE]
    retry = sorted(n for n, s, c in jobs
                   if n not in (RESOLVE, PRUNE) and WARM not in n
                   and (c in RERUNNABLE or s != "completed"))

    if not resolve or resolve[0] != "success":
        rerun, reason = False, "resolve did not succeed, so nothing a rerun could reuse was built"
    elif not retry:
        rerun, reason = False, "no job failed or was cancelled, so nothing is left to rerun"
    elif attempt >= max_attempt:
        rerun, reason = False, (f"attempt {attempt} is the last one ({max_attempt}) that keeps "
                                "artifacts for a rerun")
    else:
        rerun = True
        shown = ", ".join(retry[:5]) + (f" and {len(retry) - 5} more" if len(retry) > 5 else "")
        reason = (f"attempt {attempt} of {max_attempt}: {shown} can be rerun with "
                  "`gh run rerun <run id> --failed`, which needs these artifacts")
    if keep_requested and not rerun:
        reason = f"keep_artifacts is set ({reason})"
    return {"rerun": rerun, "keep": rerun or keep_requested, "reason": reason}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", required=True, type=Path, help="wait_for_jobs.sh --out file")
    ap.add_argument("--attempt", required=True, type=int, help="github.run_attempt")
    ap.add_argument("--max-attempt", required=True, type=int,
                    help="first attempt that no longer keeps artifacts for a rerun")
    ap.add_argument("--keep-requested", default="false", help="inputs.keep_artifacts")
    args = ap.parse_args()

    jobs = parse_jobs(args.jobs.read_text() if args.jobs.is_file() else "")
    d = decide(jobs, args.attempt, args.max_attempt, args.keep_requested == "true")
    print(f"rerun={'true' if d['rerun'] else 'false'}")
    print(f"keep={'true' if d['keep'] else 'false'}")
    print(f"reason={d['reason']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
