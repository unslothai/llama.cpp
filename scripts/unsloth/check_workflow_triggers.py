#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail when a workflow subscribes to `pull_request_target`, or to `workflow_run` unjustified.

`pull_request_target` runs in the BASE repository's context: with its secrets, and with a
token that can write. The pull request supplies the code. Combine the two and a fork
controls what executes next to credentials it must never see. The trigger has a legitimate
narrow use, labelling and commenting without ever checking out the head, but it is
one `ref:` away from arbitrary execution, and that line is easy to cross by accident.

This repository had the dangerous version until it was deleted:
`.github/workflows/bench.yml.disabled` triggered on `pull_request_target`, checked out
`github.event.pull_request.head.sha`, ran the pull request's own build, and then did
`cat results.github.env >> $GITHUB_ENV` with `secrets.GITHUB_TOKEN` and
`secrets.IMGUR_CLIENT_ID` in the same job. A pull-request-controlled file written into
`$GITHUB_ENV` sets variable NAMES as well as values, so it reaches `PATH` and friends;
that is arbitrary execution, not data. It was inert only because of its file extension,
since GitHub loads `.yml` and `.yaml` and nothing else.

Which is exactly why this check exists. Deleting the file did not remove the hazard,
because this is a fork and an upstream sync can reintroduce upstream's `bench.yml` under
its original, ACTIVE name at any time. `unsloth-upstream-sync-guard.yml` checks that the
recorded sync point is still an ancestor of master; it says nothing about what the synced
workflows do. So the reintroduction would arrive green.

The host is `unsloth-pr-set-lint.yml`, chosen because its `push` and `pull_request`
triggers both include `.github/workflows/*.yml`, i.e. every workflow edit runs it,
including the merge commit of a sync. `unslothai/unsloth` enforces the same rule with
`scripts/lint_workflow_triggers.py`, and the escape-hatch comment is spelled identically
so the two repositories read the same way.

`workflow_run` is restricted rather than banned. It does not run fork code by itself, and
both live uses here are safe: `unsloth-prebuilt-retry.yml` has no checkout step at all,
and `unsloth-repin-bot.yml` checks out the default branch with `persist-credentials:
false`. But it is privileged, it is easy to hand an artifact from an untrusted run, so it
has to be argued for in the file rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with 'pip install pyyaml'", file=sys.stderr)
    sys.exit(2)

BANNED = ("pull_request_target",)
RESTRICTED = ("workflow_run",)
ALLOW_COMMENT = "# lint:workflow_triggers-allow-workflow_run"


def triggers(doc) -> set[str]:
    """The keys of `on:`, which PyYAML parses as the boolean True."""
    if not isinstance(doc, dict):
        return set()
    on = doc.get(True) if True in doc else doc.get("on")
    if isinstance(on, str):
        return {on}
    if isinstance(on, list):
        return set(on)
    if isinstance(on, dict):
        return set(on.keys())
    return set()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=".", help="repository root to scan")
    a = ap.parse_args()

    wf_dir = Path(a.root) / ".github" / "workflows"
    if not wf_dir.is_dir():
        print(f"::error::{wf_dir} does not exist", file=sys.stderr)
        return 2

    # Only .yml and .yaml, because that is all GitHub loads. A neighbouring
    # bench.yml.disabled is genuinely inert and is not this check's business; the risk it
    # carries is that someone renames it, and at that moment this check sees it.
    files = sorted(list(wf_dir.glob("*.yml")) + list(wf_dir.glob("*.yaml")))
    if not files:
        print(f"::error::no workflows found under {wf_dir}", file=sys.stderr)
        return 2

    findings: list[str] = []
    for path in files:
        try:
            doc = yaml.safe_load(path.read_text())
        except Exception as exc:
            print(f"::error file={path}::failed to parse: {exc}", file=sys.stderr)
            return 2
        found = triggers(doc)

        for t in BANNED:
            if t in found:
                findings.append(
                    f"{path.name}: BANNED trigger '{t}'. It runs with this repository's "
                    "secrets and a write-capable token while the pull request supplies "
                    "the code, which is how bench.yml came to check out a fork's head "
                    "next to its credentials. Use 'pull_request' and move any privileged "
                    "step into a separate workflow that runs after merge."
                )

        for t in RESTRICTED:
            if t in found and ALLOW_COMMENT not in path.read_text():
                findings.append(
                    f"{path.name}: RESTRICTED trigger '{t}' needs an explicit "
                    f"'{ALLOW_COMMENT}' comment in the file, with the reason. It is "
                    "privileged, and consuming an artifact from an untrusted run is the "
                    "usual way it goes wrong."
                )

    if findings:
        print("Workflow trigger check failed:", file=sys.stderr)
        for f in findings:
            print(f"::error::{f}", file=sys.stderr)
        return 1

    print(f"{len(files)} workflow(s): no pull_request_target, no unjustified workflow_run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
