#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail when a workflow reads `steps.<id>.outputs.<key>` that no step writes.

GitHub does not error on this. It substitutes the empty string, so the failure
surfaces far away and looks like something else entirely: on 09-18 splitting the
resolve step moved the whole `{ echo "tag=..." ... } >> "$GITHUB_OUTPUT"` block
into the new step, every `steps.r.outputs.*` silently became empty, and the run
died four steps later in pin_contract.py with

    json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

under an error message about the merged tree missing code a pin carries. Nothing
pointed at the real cause. This check does.

Only `run:` steps are inspected, and only literal `echo "key=..."` writes, which
is how every producer in this repo emits. A step that writes outputs some other
way can be listed in ALLOW below rather than weakening the check for everyone.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

# (workflow stem, step id) pairs whose outputs are not literal `echo "key="` writes.
ALLOW: set[tuple[str, str]] = set()

REF = re.compile(r"steps\.([A-Za-z_][\w-]*)\.outputs\.([A-Za-z_][\w-]*)")
EMIT = re.compile(r'echo "([A-Za-z_][\w-]*)=')
# `foo=$(...)` heredoc form, e.g. printf "%s<<EOF\n..." >> "$GITHUB_OUTPUT"
EMIT_HEREDOC = re.compile(r'"?([A-Za-z_][\w-]*)<<')


def emitted(step: dict) -> set[str]:
    run = step.get("run") or ""
    return set(EMIT.findall(run)) | set(EMIT_HEREDOC.findall(run))


def check(path: Path) -> list[str]:
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as exc:
        return [f"{path}: could not parse: {exc}"]
    if not isinstance(doc, dict) or "jobs" not in doc:
        return []

    problems: list[str] = []
    for job_name, job in (doc.get("jobs") or {}).items():
        if not isinstance(job, dict):
            continue
        steps = [s for s in (job.get("steps") or []) if isinstance(s, dict)]
        produced: dict[str, set[str]] = {}
        # A `uses:` step's outputs are declared by the action, not by this file, so it is
        # not knowable from here and is not checked. Only `run:` steps are.
        for s in steps:
            if s.get("id") and "uses" not in s:
                produced.setdefault(s["id"], set()).update(emitted(s))
        action_ids = {s["id"] for s in steps if s.get("id") and "uses" in s}

        # References are resolved per job: `steps.` only ever names a step of the
        # same job, so a matching id in another job must not satisfy this one.
        scope = yaml.safe_dump({"j": job})
        for step_id, key in set(REF.findall(scope)):
            if (path.stem, step_id) in ALLOW or step_id in action_ids:
                continue
            if step_id not in produced:
                problems.append(
                    f"::error file={path}::job `{job_name}` reads "
                    f"steps.{step_id}.outputs.{key}, but no step in that job has id `{step_id}`")
            elif key not in produced[step_id]:
                problems.append(
                    f"::error file={path}::job `{job_name}` reads "
                    f"steps.{step_id}.outputs.{key}, but step `{step_id}` never writes `{key}` "
                    f"(it writes: {', '.join(sorted(produced[step_id])) or 'nothing'}). "
                    f"GitHub substitutes an empty string for this, so it fails somewhere else.")
    return problems


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    files = sorted((root / ".github/workflows").glob("*.yml"))
    problems: list[str] = []
    for f in files:
        problems.extend(check(f))
    for p in problems:
        print(p)
    print(f"{len(files)} workflow(s), {len(problems)} dangling output reference(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
