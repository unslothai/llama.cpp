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
import re
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
# The introducer a continuation line must use. Any adjacent comment used to count, which
# let this repository's own SPDX/Copyright header serve as the safety argument.
JUSTIFIED = "Justified:"


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


def _block_scalar_lines(text: str) -> set[int]:
    """Indices of every line inside a YAML block scalar (`run: |`, `script: >` ...).

    `#` starts a comment in YAML and in shell alike, so a `#` line inside a `run: |`
    body looks exactly like a YAML comment to a line-by-line reader. The waiver would
    then be honoured for a marker the workflow merely ECHOES, which is content a fork's
    own script can contain.

    The ranges come from the parser, via `yaml.compose` and each node's `start_mark` and
    `end_mark`, rather than from a regex over the source. Two attempts at recognising the
    opener lexically both had holes, and they were holes of the same shape: first only
    `|-2` and not the equally valid `|2-`, then only `run:` and not `run :`. Every miss
    puts an entire script body back in scope as ordinary comment lines, so each one is a
    full bypass, and the supply of valid spellings is larger than the supply of patience
    for enumerating them. PyYAML already knows exactly which lines are scalar content;
    asking it is both shorter and complete.
    """
    inside: set[int] = set()
    try:
        root = yaml.compose(text)
    except yaml.YAMLError:
        return inside
    if root is None:
        return inside
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, yaml.ScalarNode):
            if node.style in ("|", ">"):
                # start_mark.line is the line holding the indicator; the body begins
                # after it and runs to end_mark.line.
                for i in range(node.start_mark.line + 1, node.end_mark.line + 1):
                    inside.add(i)
        elif isinstance(node, yaml.SequenceNode):
            stack.extend(node.value)
        elif isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                stack.append(key)
                stack.append(value)
    return inside


def waiver_status(text: str) -> tuple[bool, str]:
    """Is the workflow_run waiver present AS A COMMENT, and does it carry a reason?

    A bare substring test accepted the marker anywhere, including inside a quoted `run:`
    string, and accepted it with no reason at all. Both let a workflow claim the waiver
    without making the argument the marker is supposed to record, which is the whole
    value of requiring it. The reason may sit on the marker line, or on a following
    comment line introduced by `Justified:`, which is how both live waivers here are
    written.

    That introducer is required rather than decorative. Accepting any adjacent comment
    meant the file's own `# SPDX-License-Identifier` and `# Copyright` header counted as
    the argument, so putting the bare marker directly above the header waived the trigger
    while saying nothing at all. Every workflow in this repository opens with that header,
    which made it the easiest reason in the tree to borrow by accident.

    Lines inside a block scalar do not count, however much they look like comments:
    `run: |` followed by `# lint:...-allow-workflow_run reason` is shell text, not a
    statement by the workflow's author about the workflow.
    """
    lines = text.split("\n")
    in_block = _block_scalar_lines(text)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # A comment line, not a marker buried in a string or a run: body.
        if i in in_block or not stripped.startswith("#") or ALLOW_COMMENT.lstrip("# ") not in stripped:
            continue
        tail = stripped.split(ALLOW_COMMENT.lstrip("# "), 1)[1].strip(" #-")
        # The label is optional inline, but writing it must not itself count as the
        # reason. `strip(" #:-")` turned a bare `Justified:` into the non-empty string
        # `Justified` and accepted it, so the inline spelling passed where the separate
        # `# Justified:` line was correctly refused. The label is removed explicitly and
        # what remains has to be text.
        if tail.casefold().startswith(JUSTIFIED.casefold()):
            tail = tail[len(JUSTIFIED):]
        if tail.strip(" #:-"):
            return True, ""
        for k, follow in enumerate(lines[i + 1:], start = i + 1):
            f = follow.strip()
            if k in in_block or not f.startswith("#"):
                break
            body = f.lstrip("# ").strip()
            if body.casefold().startswith(JUSTIFIED.casefold()):
                if body[len(JUSTIFIED):].strip(" #:-"):
                    return True, ""
                break
        return False, (
            f"carries '{ALLOW_COMMENT}' with no reason written next to it or on a "
            f"following '# {JUSTIFIED} ...' comment line."
        )
    return False, f"has no '{ALLOW_COMMENT}' comment."


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
            if t not in found:
                continue
            ok, why = waiver_status(path.read_text())
            if not ok:
                findings.append(
                    f"{path.name}: RESTRICTED trigger '{t}' {why} It is privileged, and "
                    "consuming an artifact from an untrusted run is the usual way it goes "
                    f"wrong, so it needs '{ALLOW_COMMENT}' on a comment line with the "
                    f"reason either on that line or on a following '# {JUSTIFIED} ...' "
                    "comment line."
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
