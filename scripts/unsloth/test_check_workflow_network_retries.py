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
             "gh release download v1 -R a/b",
             "BASE=\"$(gh api 'repos/ggml-org/llama.cpp/releases?per_page=100' | jq -r .)\"",
             'if ! PR_SHAS="$(gh api "repos/$SRC/pulls/$N/commits" --paginate)"; then',
             'gh release view "$TAG" --repo "$REPO" --json isDraft',
             'gh release edit "$TAG" --draft=false',
             'GH_TOKEN="$T" gh pr create --repo a/b --head x',
             'done < <(gh issue list --repo "$REPO" --label x)',
             'gh run view "$ID" --log-failed',
             'git -C "$path" push "$url" "HEAD:refs/heads/x"',
             "git push -q --force origin HEAD:refs/heads/x",
             "hf download org/model --include '*.gguf'",
             "huggingface-cli download org/model",
             "docker pull ghcr.io/a/b:latest",
             "npm ci",
             "npx playwright install chromium",
             "conda install -y cmake",
             "rustup target add aarch64-apple-darwin",
             "vcpkg install curl"):
    check(f"flags: {bare}", flagged(bare))

# Inline retry constructs pass.
for ok in ("curl -fsSL --retry 5 --retry-all-errors -o x https://e.com/x",
           "sudo apt-get -o Acquire::Retries=5 install -y patchelf",
           "python -m pip install --retries 5 --timeout 60 pefile",
           "wget -q --tries=5 https://apt.llvm.org/llvm.sh",
           "Invoke-WebRequest -Uri $u -OutFile $f -MaximumRetryCount 3",
           "retry git fetch -q origin main",
           "Invoke-DownloadWithRetry -Uri $u -OutFile $f",
           "BASE=\"$(gh_retry api 'repos/ggml-org/llama.cpp/releases?per_page=100' | jq -r .)\"",
           'GH_TOKEN="$T" gh_retry pr create --repo a/b --head x',
           'net_retry git -C "$path" push "$url" "HEAD:refs/heads/x"',
           'if [ -n "$(net_retry git ls-remote "$M" "refs/pins/$S")" ]; then'):
    check(f"passes: {ok}", not flagged(ok))

# Local-only commands that share a word with a network one are not calls.
for local in ("git log -1 --format=%H",
              "gh auth status",
              "echo 'gh api failed'",
              "docker build -t x .",
              "npm run build"):
    check(f"not a network call: {local}", not flagged(local))

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

# Every unsloth-* workflow is in scope, not only the prebuilt ones, and so is a scripts/unsloth/*.sh a
# step runs. An upstream workflow is not: it is replaced on every sync and never runs here.
root = Path(tempfile.mkdtemp())
(root / ".github" / "workflows").mkdir(parents=True)
(root / "scripts" / "unsloth").mkdir(parents=True)
(root / ".github" / "workflows" / "unsloth-guard.yml").write_text(
    "on: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: bash tooling/scripts/unsloth/prune.sh\n")
(root / ".github" / "workflows" / "build.yml").write_text(
    "on: push\njobs:\n  a:\n    runs-on: ubuntu-latest\n    steps:\n"
    "      - run: sudo apt-get update\n")
(root / "scripts" / "unsloth" / "prune.sh").write_text("#!/bin/bash\ngh api repos/a/b/actions/caches\n")
proc = subprocess.run([sys.executable, str(CHECK), "--root", str(root)], capture_output=True, text=True, check=False)
check("a script an unsloth-* workflow runs is checked",
      proc.returncode == 1 and "scripts/unsloth/prune.sh,line=2" in proc.stdout, proc.stdout + proc.stderr)
check("an upstream workflow is out of scope", "build.yml" not in proc.stdout, proc.stdout)

print("net_retry.sh: the helper the workflows source")
HELPER = HERE / "net_retry.sh"
stub = Path(tempfile.mkdtemp())
# A fake gh: fails with the stderr in $GH_ERR until the attempt counter reaches $GH_OK_AT, printing a
# partial page on each failure, so a caller that saw failed output would see it twice.
(stub / "gh").write_text(
    "#!/bin/bash\nn=$(( $(cat \"$GH_COUNT\" 2>/dev/null || echo 0) + 1 )); echo $n > \"$GH_COUNT\"\n"
    "if [ \"$n\" -lt \"$GH_OK_AT\" ]; then echo partial; echo \"$GH_ERR\" >&2; exit 1; fi\necho page-all\n")
(stub / "gh").chmod(0o755)


def helper(cmd: str, ok_at: int, err: str = "HTTP 502: Bad Gateway") -> tuple[int, str, int]:
    count = stub / "count"
    count.unlink(missing_ok=True)
    env = {"PATH": f"{stub}:/usr/bin:/bin", "NET_RETRY_SLEEP": "0", "GH_COUNT": str(count),
           "GH_OK_AT": str(ok_at), "GH_ERR": err}
    p = subprocess.run(["bash", "-c", f'. "{HELPER}"; {cmd}'], capture_output=True, text=True, env=env, check=False)
    return p.returncode, p.stdout, int(count.read_text()) if count.exists() else 0


rc, out, n = helper("gh_retry api x", ok_at=3)
check("gh_retry recovers from two 502s", rc == 0 and n == 3, f"rc={rc} n={n} out={out!r}")
check("gh_retry prints only the successful attempt's stdout", out == "page-all\n", repr(out))
rc, out, n = helper("gh_retry api x", ok_at=99)
check("gh_retry gives up after 4 attempts with gh's exit code", rc == 1 and n == 4 and out == "",
      f"rc={rc} n={n} out={out!r}")
rc, out, n = helper("gh_retry release view t", ok_at=99, err="release not found")
check("gh_retry does not retry 'not found'", rc == 1 and n == 1, f"rc={rc} n={n}")
rc, out, n = helper("gh_retry api x", ok_at=99, err="gh: Not Found (HTTP 404)")
check("gh_retry does not retry a 404", rc == 1 and n == 1, f"rc={rc} n={n}")
rc, out, n = helper("gh_retry api x", ok_at=2, err="HTTP 403: You have exceeded a secondary rate limit")
check("gh_retry retries a 403 secondary rate limit", rc == 0 and n == 2, f"rc={rc} n={n}")
rc, out, n = helper("net_retry gh api x", ok_at=3)
check("net_retry recovers and returns 0", rc == 0 and n == 3, f"rc={rc} n={n}")
rc, out, n = helper("set -e; net_retry gh api x; echo unreachable", ok_at=99)
check("net_retry failure still fails a set -e script", rc != 0 and "unreachable" not in out and n == 4,
      f"rc={rc} n={n} out={out!r}")
rc, out, n = helper('set -euo pipefail; X="$(gh_retry api x)"; echo unreachable', ok_at=99)
check("gh_retry failure inside $(...) still fails a set -euo pipefail script",
      rc != 0 and "unreachable" not in out, f"rc={rc} out={out!r}")
rc, out, n = helper('set -euo pipefail; [ "$(gh_retry api x | tr a-z A-Z)" = PAGE-ALL ] && echo piped', ok_at=2)
check("gh_retry output pipes on success", rc == 0 and out == "piped\n", f"rc={rc} out={out!r}")

if FAILS:
    print(f"{len(FAILS)} failure(s)")
    sys.exit(1)
print("all passed")
