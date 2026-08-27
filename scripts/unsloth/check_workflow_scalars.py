#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail when a workflow string is close to GitHub's 21000 character cap.

GitHub's template compiler refuses any single string in a workflow file longer
than 21000 characters. The whole file then fails to compile, and the failure is
close to invisible:

  - the run has zero jobs, no annotations and an empty check suite, so there is
    nothing to click on;
  - `gh run view` says only "This run likely failed because of a workflow file
    issue";
  - the run is reported against whatever event triggered it, even an event the
    workflow does not subscribe to, because compilation never got as far as
    reading `on:`;
  - and nothing local catches it. yaml parses the file, actionlint passes it,
    and so does GitHub's own published parser (@actions/workflow-parser). The
    limit is enforced only server-side.

On 08-27 a 14-line explanatory comment added inside the `resolve` job's script
took it from 20503 to 21620 characters and silently disabled the entire release
workflow for four pushes. The only way to see the real error was to fire a
workflow_dispatch at the ref, which returns it as a 422:

    (Line: 125, Col: 14): Exceeded max expression length 21000

Comments inside a `run:` block scalar are part of the string and count against
the limit. Comments in the YAML around it do not, so prose belongs above a step
rather than inside it. Past that, the fix is to split the script into more
steps: a step boundary costs nothing and resets the budget.

Measuring the parsed scalar is not exactly what GitHub measures, so the warn
threshold is deliberately well below the cap rather than a character-perfect
model of it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# What GitHub enforces, and the point at which a script is big enough that the
# next edit to it can cross the line without anyone thinking about size.
CAP = 21000
WARN = 20000


def scalars(node, path: str = ""):
    """Every string in the document, with a path that names where it lives."""
    if isinstance(node, str):
        yield path, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from scalars(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from scalars(v, f"{path}[{i}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--cap", type=int, default=CAP, help="hard limit, fails")
    ap.add_argument("--warn", type=int, default=WARN, help="soft limit, warns")
    a = ap.parse_args()

    files = sorted(Path(a.root, ".github/workflows").glob("*.y*ml"))
    if not files:
        print(f"no workflows under {a.root}/.github/workflows", file=sys.stderr)
        return 1

    over, near = [], []
    for f in files:
        try:
            doc = yaml.safe_load(f.read_text())
        except yaml.YAMLError as e:
            print(f"::error file={f}::not valid YAML: {e}", file=sys.stderr)
            return 1
        for path, s in scalars(doc):
            if len(s) > a.cap:
                over.append((len(s), f, path))
            elif len(s) > a.warn:
                near.append((len(s), f, path))

    for n, f, path in sorted(near, reverse=True):
        print(f"::warning file={f}::{path} is {n} characters, within "
              f"{a.cap - n} of GitHub's {a.cap} character limit. Move any "
              "prose out of the block scalar into YAML comments above the "
              "step, or split the script into another step.")
    for n, f, path in sorted(over, reverse=True):
        print(f"::error file={f}::{path} is {n} characters, over GitHub's "
              f"{a.cap} character limit. GitHub will refuse to compile this "
              "file and every run of it will fail with no jobs and no "
              "annotation. Split the script into another step; a step "
              "boundary resets the budget.", file=sys.stderr)

    biggest = max((len(s) for f in files for _, s in scalars(yaml.safe_load(f.read_text()))),
                  default=0)
    print(f"{len(files)} workflow(s), largest string {biggest} of {a.cap}"
          f"{f', {len(near)} within {a.cap - a.warn} of the limit' if near else ''}")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main())
