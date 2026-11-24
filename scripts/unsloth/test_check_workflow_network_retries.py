#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for check_workflow_network_retries.py, in the self-running style the sibling tests use."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECK = HERE / "check_workflow_network_retries.py"
REPO = HERE.parents[1]
FAILS: list[str] = []

sys.path.insert(0, str(HERE))
from check_workflow_network_retries import check_doc, check_script


def check(name: str, ok: bool, out: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILS.append(name)
        if out:
            print("    ---")
            for line in out.strip().split("\n"):
                print(f"    {line}")


def flagged(script: str) -> bool:
    return bool(check_script(script))


print("check_workflow_network_retries: run lines")
# Bare calls are flagged.
for bare in ("choco install ninja --no-progress",
             "sudo apt-get update",
             "sudo apt install -y cmake ninja-build unzip curl",
             "python -m pip install pefile",
             "uv pip install foo",
             "brew install libomp",
             "winget install Foo.Bar",
             'curl -sL "$url" | tar -xz',
             "curl.exe -o out.exe -L https://example.com/x.exe",
             "wget -q https://apt.llvm.org/llvm.sh",
             '$h = (Invoke-WebRequest "$u/" -UseBasicParsing).Content',
             'git -C "$src" fetch -q --depth 1 origin "$sha"',
             "git clone https://github.com/a/b",
             'CODE="$(curl -sS -X POST https://api.example.com)"',
             "gh release download v1 -R a/b"):
    check(f"flags: {bare}", flagged(bare))

# Inline retry constructs pass.
for ok in ("curl -fsSL --retry 5 --retry-all-errors -o x https://e.com/x",
           "sudo apt-get -o Acquire::Retries=5 install -y patchelf",
           "python -m pip install --retries 5 --timeout 60 pefile",
           "wget -q --tries=5 https://apt.llvm.org/llvm.sh",
           "Invoke-WebRequest -Uri $u -OutFile $f -MaximumRetryCount 3",
           "retry git fetch -q origin main",
           "Invoke-DownloadWithRetry -Uri $u -OutFile $f"):
    check(f"passes: {ok}", not flagged(ok))

# Quoted text and the bare word in a package list are not calls.
check("throw message is not a call", not flagged('throw "choco install ninja failed"'))
check("echo text is not a call", not flagged("echo 'run apt-get update first'"))
check("curl as a package name is not a call", not flagged("sudo apt-get -o Acquire::Retries=5 install -y unzip curl"))
check("comment is not a call", not flagged("# choco install llvm used to 504"))

print("check_workflow_network_retries: loops")
BASH_LOOP = """for i in 1 2 3; do
  git fetch -q origin "$sha" && break
  [ "$i" -lt 3 ] || exit 1
  sleep $((20 * i))
done"""
check("bash retry loop passes", not flagged(BASH_LOOP))
PS_LOOP = """foreach ($i in 1..3) {
  choco install llvm --no-progress
  if ($LASTEXITCODE -eq 0) { break }
  Start-Sleep -Seconds (30 * $i)
}"""
check("pwsh retry loop passes", not flagged(PS_LOOP))
TRY_LOOP = """foreach ($i in 1..3) {
  try { $h = (Invoke-WebRequest "$u/" -ErrorAction Stop).Content; break }
  catch {
    if ($i -eq 3) { throw }
    Start-Sleep -Seconds (20 * $i)
  }
}"""
check("pwsh try/catch loop passes", not flagged(TRY_LOOP))
check("one-line loop passes",
      not flagged("for i in 1 2 3; do curl -sf -o x https://e.com && break; sleep 5; done"))
NO_SLEEP = """while read -r p; do
  curl -fsSL "https://e.com/$p" | git apply -
done"""
check("a loop that never sleeps is not a retry", flagged(NO_SLEEP))
AFTER_LOOP = """for i in 1 2 3; do
  echo $i; sleep 1
done
choco install ninja"""
check("a call after a retry loop is still flagged", flagged(AFTER_LOOP))

print("check_workflow_network_retries: allow marker")
check("marker with a reason on the line passes",
      not flagged("curl -X POST https://e.com  # lint:network-retry-allow best-effort notify"))
check("marker in the comment block above passes",
      not flagged("# lint:network-retry-allow a POST must not be repeated\n# more words\ncurl -sS -X POST https://e.com"))
check("marker without a reason is refused",
      flagged("# lint:network-retry-allow\ncurl -sS -X POST https://e.com"))
check("marker separated by code is refused",
      flagged("# lint:network-retry-allow reason here\necho hi\ncurl -sS https://e.com"))

print("check_workflow_network_retries: downloading actions")
CCACHE = "hendrikmuhs/ccache-action@d62db5f07c26379fc4b4e0916f098a92573c3b03"
bare = {"jobs": {"a": {"steps": [{"name": "ccache", "uses": CCACHE}]}}}
check("bare ccache-action is flagged", bool(check_doc(bare)))
coe = {"jobs": {"a": {"steps": [{"name": "ccache", "uses": CCACHE, "continue-on-error": True}]}}}
check("continue-on-error ccache-action passes", not check_doc(coe))
retried = {"runs": {"steps": [
    {"name": "ccache", "id": "ccache", "uses": CCACHE, "continue-on-error": True},
    {"name": "ccache (retry)", "uses": CCACHE, "if": "steps.ccache.outcome == 'failure'"}]}}
check("composite first attempt + gated retry passes", not check_doc(retried))

print("check_workflow_network_retries: the live tree")
proc = subprocess.run([sys.executable, str(CHECK), "--root", str(REPO)], capture_output=True, text=True, check=False)
check("the repository's prebuilt workflows pass", proc.returncode == 0, proc.stdout + proc.stderr)

root = Path(tempfile.mkdtemp())
(root / ".github" / "workflows").mkdir(parents=True)
(root / ".github" / "actions" / "setup").mkdir(parents=True)
(root / ".github" / "workflows" / "unsloth-prebuilt-x.yml").write_text(
    "on: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - uses: ./tooling/.github/actions/setup\n")
(root / ".github" / "actions" / "setup" / "action.yml").write_text(
    "runs:\n  using: composite\n  steps:\n    - shell: bash\n      run: sudo apt-get update\n")
proc = subprocess.run([sys.executable, str(CHECK), "--root", str(root)], capture_output=True, text=True, check=False)
check("a local composite action the workflow calls is checked too",
      proc.returncode == 1 and "actions/setup/action.yml" in proc.stdout, proc.stdout + proc.stderr)

if FAILS:
    print(f"{len(FAILS)} failure(s)")
    sys.exit(1)
print("all passed")
