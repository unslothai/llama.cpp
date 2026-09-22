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

# A continuation line must announce itself with `Justified:`. Accepting any adjacent
# comment meant this repository's own SPDX/Copyright header served as the safety
# argument, and every workflow here opens with that header, so it was the easiest reason
# in the tree to borrow by accident.
under = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run\n"
    "# Justified: no checkout, reruns a failed release\nname: wr\n",
)
rc, out = run(build({"wr.yml": under}))
check("a 'Justified:' line beneath the marker passes", rc == 0, out)

header = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run\n"
    "# SPDX-License-Identifier: AGPL-3.0-only\n"
    "# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.\nname: wr\n",
)
rc, out = run(build({"wr.yml": header}))
check("the license header is not a justification", rc == 1, out)

bare_justified = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run\n# Justified:\nname: wr\n",
)
rc, out = run(build({"wr.yml": bare_justified}))
check("'Justified:' with nothing after it is not a justification", rc == 1, out)

# --- 7. an empty or missing tree is an error, not a pass -----------------
rc, out = run(build({}))
check("an empty workflows directory is an error", rc == 2, out)
rc, out = run(Path(tempfile.mkdtemp()))
check("a missing workflows directory is an error", rc == 2, out)

print()
# A marker inside a `run: |` BODY is shell text the workflow prints, not a statement by
# its author. `#` opens a comment in YAML and in shell alike, so the two are
# indistinguishable line by line, and honouring the block-scalar form would let the
# waiver be claimed by content the workflow merely echoes. The sibling check above covers
# the single-line quoted form; this is the multi-line one, which is how `run:` is
# actually written.
WFRUN_MARKER_IN_BLOCK = (
    "name: wr\n"
    "on:\n  workflow_run:\n    workflows: ['x']\n    types: [completed]\n"
    "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: |\n"
    "          # lint:workflow_triggers-allow-workflow_run consumes no artifact\n"
    "          echo hi\n"
)
# YAML fixes no order between the chomping and indentation indicators, so `|2-` is as
# valid as `|-2`. Matching one order only left the whole body of a `run: |2-` step back
# in scope as ordinary comment lines.
rc, out = run(build({"wr.yml": WFRUN_MARKER_IN_BLOCK.replace("run: |", "run: |2-")}))
check("the marker inside a run: |2- body does not count either", rc == 1, out)

rc, out = run(build({"wr.yml": WFRUN_MARKER_IN_BLOCK}))
check("the marker inside a run: block scalar does not count as a waiver", rc == 1, out)
check(
    "and the finding says the marker is absent rather than unreasoned",
    "has no '# lint:workflow_triggers-allow-workflow_run' comment" in out,
    out,
)

# The same file with a real YAML comment carrying a reason passes, so the rule above
# rejects the location and not the justification.
rc, out = run(build({
    "wr.yml": "# lint:workflow_triggers-allow-workflow_run consumes no artifact\n"
              + WFRUN_MARKER_IN_BLOCK,
}))
check("a real YAML comment with a reason still passes", rc == 0, out)

# `run : |` is valid YAML and PyYAML reads the value as a block scalar just the same. The
# lexical opener matched only `run:`, so the body went back into scope as ordinary comment
# lines. The ranges now come from the parser, which is what makes enumerating spellings
# unnecessary rather than merely tedious: every miss was a full bypass, not a rough edge.
rc, out = run(build({"wr.yml": WFRUN_MARKER_IN_BLOCK.replace("run: |", "run : |")}))
check("the marker inside a 'run : |' body does not count either", rc == 1, out)

# The inline label must not stand in for the reason it introduces. Stripping punctuation
# turned a bare `Justified:` into the non-empty string `Justified` and accepted it, so the
# inline spelling passed where the separate `# Justified:` line was correctly refused.
inline_label = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run Justified:\nname: wr\n",
)
rc, out = run(build({"wr.yml": inline_label}))
check("an inline 'Justified:' with nothing after it is not a justification", rc == 1, out)

inline_real = WFRUN.replace(
    "name: wr\n",
    "# lint:workflow_triggers-allow-workflow_run Justified: no checkout at all\nname: wr\n",
)
rc, out = run(build({"wr.yml": inline_real}))
check("an inline 'Justified: <reason>' passes", rc == 0, out)

# A real waiver comment sitting immediately after a `run: |` block must survive. PyYAML
# reports such a block as ending at (the following line, column 0), so including the
# endpoint unconditionally swallowed that line; when it held the waiver the file failed
# with "has no comment", which is a false failure on a correct file and the worse of the
# two errors this rule can make.
after_block = (
    "name: wr\n"
    "on:\n  workflow_run:\n    workflows: ['x']\n    types: [completed]\n"
    "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: |\n"
    "          echo hi\n"
    "# lint:workflow_triggers-allow-workflow_run Justified: no checkout at all\n"
)
rc, out = run(build({"wr.yml": after_block}))
check("a waiver comment right after a run: | block still counts", rc == 0, out)

# A multi-line QUOTED scalar is scalar content too, and a continuation line of one can
# begin with `#`. Recording only `|` and `>` nodes read that as a real comment.
quoted_multiline = (
    "name: wr\n"
    "on:\n  workflow_run:\n    workflows: ['x']\n    types: [completed]\n"
    "jobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    '      - run: "echo one\n'
    "          # lint:workflow_triggers-allow-workflow_run Justified: fake\n"
    '          echo two"\n'
)
rc, out = run(build({"wr.yml": quoted_multiline}))
check("the marker inside a multiline quoted scalar does not count", rc == 1, out)

# The directive has to end at a token boundary. A substring test accepted
# `...-allow-workflow_runs` and then read the trailing `s` as the justification, so one
# mistyped character both claimed the waiver and supplied its own reason -- and the
# canonical bare marker was correctly refused on the very same input.
for near in (
    "# lint:workflow_triggers-allow-workflow_runs",
    "# lint:workflow_triggers-allow-workflow_run-anyway",
    "# lint:workflow_triggers-allow-workflow_run_now",
):
    rc, out = run(build({"wr.yml": WFRUN.replace("name: wr\n", near + "\nname: wr\n")}))
    check(f"a near-match directive is not the waiver: {near.split(':', 1)[1]}", rc == 1, out)

# And the canonical directive followed by real punctuation still reads its reason.
for good in (
    "# lint:workflow_triggers-allow-workflow_run no checkout at all",
    "# lint:workflow_triggers-allow-workflow_run: no checkout at all",
    "# lint:workflow_triggers-allow-workflow_run Justified: no checkout at all",
):
    rc, out = run(build({"wr.yml": WFRUN.replace("name: wr\n", good + "\nname: wr\n")}))
    check(f"the canonical directive still reads its reason: {good[-24:]}", rc == 0, out)

print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)