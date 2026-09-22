#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for check_workflow_triggers.py, in the self-running style the sibling tests use."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "check_workflow_triggers.py"
REPO = HERE.parents[1]
FAILS: list[str] = []


def check(name: str, ok: bool, out: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILS.append(name)
        if out:
            print("    ---")
            for line in out.strip().split("\n"):
                print(f"    {line}")


def run(root: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(CHECK), "--root", str(root)],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def build(files: dict[str, str]) -> Path:
    root = Path(tempfile.mkdtemp())
    wf = root / ".github" / "workflows"
    wf.mkdir(parents=True)
    for name, text in files.items():
        (wf / name).write_text(text)
    return root


SAFE = "name: ok\non:\n  pull_request:\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"
PRT = "name: bad\non:\n  pull_request_target:\n    types: [opened]\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"
WFRUN = "name: wr\non:\n  workflow_run:\n    workflows: ['x']\n    types: [completed]\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n      - run: true\n"

print("check_workflow_triggers")

# --- 1. the live tree must pass ------------------------------------------
rc, out = run(REPO)
check("the repository's own workflows pass", rc == 0, out)

# --- 2. pull_request_target is refused ----------------------------------
rc, out = run(build({"bad.yml": PRT}))
check("pull_request_target fails", rc == 1, out)
check("and the finding names the file", "bad.yml" in out, out)

# --- 3. the .yaml spelling is covered too -------------------------------
rc, out = run(build({"bad.yaml": PRT}))
check("pull_request_target in a .yaml file fails", rc == 1, out)

# --- 4. a renamed bench.yml is exactly the regression this guards -------
rc, out = run(build({"bench.yml": PRT}))
check("a reintroduced bench.yml fails", rc == 1, out)

# --- 5. ... while the .disabled form is inert and ignored ---------------
root = build({"ok.yml": SAFE})
(root / ".github" / "workflows" / "bench.yml.disabled").write_text(PRT)
rc, out = run(root)
check("bench.yml.disabled is not loaded by GitHub, so it is not flagged", rc == 0, out)

# --- 6. workflow_run needs a written justification -----------------------
rc, out = run(build({"wr.yml": WFRUN}))
check("unjustified workflow_run fails", rc == 1, out)

justified = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run  reruns a failed release, no checkout\nname: wr\n",
)
rc, out = run(build({"wr.yml": justified}))
check("justified workflow_run passes", rc == 0, out)

# --- 6b. the waiver must be a COMMENT and must carry a reason ------------
bare = WFRUN.replace("name: wr\n", "# lint:workflow_triggers-allow-workflow_run\nname: wr\n")
rc, out = run(build({"wr.yml": bare}))
check("the marker alone, with no reason, fails", rc == 1, out)

in_string = WFRUN.replace(
    "      - run: true\n",
    '      - run: echo "# lint:workflow_triggers-allow-workflow_run"\n',
)
rc, out = run(build({"wr.yml": in_string}))
check("the marker inside a run: string does not count as a waiver", rc == 1, out)

under = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run\n# no checkout, reruns a failed release\nname: wr\n",
)
rc, out = run(build({"wr.yml": under}))
check("a reason on the comment line beneath the marker passes", rc == 0, out)

# --- 7. an empty or missing tree is an error, not a pass -----------------
rc, out = run(build({}))
check("an empty workflows directory is an error", rc == 2, out)
rc, out = run(Path(tempfile.mkdtemp()))
check("a missing workflows directory is an error", rc == 2, out)

print()
print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
