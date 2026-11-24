#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Fail when a prebuilt workflow step downloads something without a retry.

The prebuilt nightly is ~60 legs and the publish job refuses if any one of them fails, so a
single 5xx from a package feed costs the whole release. Run 45410618008 lost
`CPU / windows/x64` to one Chocolatey 504 on `choco install llvm`. Every step that reaches a
third-party host (package managers, toolkit CDNs, git remotes) therefore has to carry a
bounded retry.

Scope: `.github/workflows/unsloth-prebuilt*.yml` plus every local composite action those
workflows call (`uses: ./.github/actions/X` or `./<checkout>/.github/actions/X`).

A `run:` line that calls one of NETWORK passes when
  * the line itself carries a retry construct (`--retry`, `--retries`, `--tries`,
    `Acquire::Retries`, `-MaximumRetryCount`, or a call through a helper whose name
    contains "retry", such as `retry git fetch` or `Invoke-DownloadWithRetry`), or
  * it sits inside a loop (`for`, `foreach`, `while`, `until`) whose body sleeps, which is
    what a hand-written retry loop looks like in both bash and PowerShell, or
  * the line, or the comment block directly above it, carries
    `lint:network-retry-allow <reason>`.

A `uses:` step of an action in DOWNLOADING_ACTIONS passes when it is `continue-on-error`
(best effort, with a fallback after it) or is itself the retry of a failed attempt
(`if: steps.<id>.outcome == 'failure'`).

This is a lint, not a proof: it reads text, so a retry it cannot see (for example inside a
script the step calls) needs the allow marker with the reason.
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

NETWORK = [
    ("choco install", re.compile(r"\bchoco(?:\.exe)?\s+(?:install|upgrade)\b")),
    ("winget install", re.compile(r"\bwinget(?:\.exe)?\s+(?:install|upgrade)\b")),
    ("apt", re.compile(r"\bapt(?:-get)?\s+(?:-\S+\s+)*(?:update|install|upgrade|dist-upgrade)\b")),
    ("brew", re.compile(r"\bbrew\s+(?:install|update|upgrade|fetch|tap|bundle)\b")),
    ("pip install", re.compile(r"\bpip3?(?:\.exe)?\s+install\b")),
    ("uv pip install", re.compile(r"\buv\s+(?:pip\s+install|sync|add)\b")),
    # The next token must look like an argument, so `apt install ... unzip curl` is not a call.
    ("curl", re.compile(r"\bcurl(?:\.exe)?\s+(?=[-\"'$]|https?:)")),
    ("wget", re.compile(r"\bwget\s+(?=[-\"'$]|https?:)")),
    ("Invoke-WebRequest", re.compile(r"\b(?:Invoke-WebRequest|Invoke-RestMethod|iwr|Start-BitsTransfer)\b|\.DownloadFile\(")),
    ("git clone/fetch", re.compile(r"\bgit(?:\s+-[Cc]\s+\S+)*\s+(?:clone|fetch|pull|ls-remote)\b")),
    ("gh release download", re.compile(r"\bgh\s+release\s+download\b")),
]
RETRY_ON_LINE = re.compile(
    r"--retry\b|--retries\b|--tries\b|Acquire::Retries|-MaximumRetryCount\b|\b\w*[Rr]etry\w*\s+\S")
LOOP_HEADER = re.compile(r"^\s*(?:for|foreach|while|until)\b|^\s*foreach\s*\(|\|\s*ForEach-Object\b")
SLEEP = re.compile(r"\bsleep\b|\bStart-Sleep\b", re.IGNORECASE)
ALLOW = "lint:network-retry-allow"
DOWNLOADING_ACTIONS = ("hendrikmuhs/ccache-action", "jimver/cuda-toolkit")
# Message text is not a call: `throw "choco install ninja failed"` must not count. A double-quoted
# string holding a command substitution (`CODE="$(curl ...)"`) is kept, since that IS a call.
QUOTED = re.compile(r"\"(?:[^\"\\$]|\\.|\$(?!\())*\"|'[^'\n]*'")
LOCAL_ACTION = re.compile(r"^\./(?:[^/]+/)?\.github/actions/([^/@]+)$")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _is_comment(line: str) -> bool:
    s = line.strip()
    return s.startswith(("#", "::")) or s.lower().startswith("rem ")


def _has_reason(text: str) -> bool:
    return len(re.findall(r"[^\W\d_]", text)) >= 3


def _allowed(lines: list[str], i: int) -> bool:
    """The marker with a reason on the line, or in the comment block right above it."""
    cands = [lines[i]]
    k = i - 1
    while k >= 0 and _is_comment(lines[k]):
        cands.append(lines[k])
        k -= 1
    for c in cands:
        pos = c.find(ALLOW)
        if pos >= 0 and _has_reason(c[pos + len(ALLOW):]):
            return True
    return False


def _in_retry_loop(lines: list[str], i: int) -> bool:
    """Is line i inside (or on) a loop whose body sleeps?"""
    if LOOP_HEADER.search(lines[i]) and SLEEP.search(lines[i]):
        return True                       # one-line `for ...; do ...; sleep ...; done`
    floor = _indent(lines[i])
    for k in range(i - 1, -1, -1):
        line = lines[k]
        if not line.strip():
            continue
        ind = _indent(line)
        if ind >= floor:
            continue
        floor = ind                       # an enclosing block header
        if LOOP_HEADER.search(line):
            body = []
            for j in range(k + 1, len(lines)):
                if lines[j].strip() and _indent(lines[j]) <= ind:
                    body.append(lines[j])    # `done` / `}` / `} catch {` line closing it
                    break
                body.append(lines[j])
            if any(SLEEP.search(b) for b in body if not _is_comment(b)):
                return True
        if floor == 0:
            break
    return False


def check_script(script: str) -> list[tuple[int, str, str]]:
    """(line number, what, text) for each network call without a visible retry."""
    lines = script.split("\n")
    bad = []
    for i, line in enumerate(lines):
        if not line.strip() or _is_comment(line):
            continue
        code = QUOTED.sub('""', line)
        for what, pat in NETWORK:
            if not pat.search(code):
                continue
            if RETRY_ON_LINE.search(line) or _in_retry_loop(lines, i) or _allowed(lines, i):
                break
            bad.append((i + 1, what, line.strip()))
            break
    return bad


def _steps(doc) -> list[tuple[str, dict]]:
    out = []
    if not isinstance(doc, dict):
        return out
    for job_id, job in (doc.get("jobs") or {}).items():
        for idx, step in enumerate((job or {}).get("steps") or []):
            if isinstance(step, dict):
                out.append((f"{job_id}/{step.get('name') or step.get('id') or idx}", step))
    runs = doc.get("runs")
    if isinstance(runs, dict):
        for idx, step in enumerate(runs.get("steps") or []):
            if isinstance(step, dict):
                out.append((f"runs/{step.get('name') or step.get('id') or idx}", step))
    return out


def check_doc(doc) -> list[str]:
    errors = []
    for where, step in _steps(doc):
        run = step.get("run")
        if isinstance(run, str):
            for n, what, text in check_script(run):
                errors.append(f"{where}: run line {n}: {what} without a retry: {text}")
        uses = step.get("uses")
        if isinstance(uses, str) and uses.split("@")[0].lower() in DOWNLOADING_ACTIONS:
            coe = step.get("continue-on-error")
            cond = str(step.get("if") or "")
            if coe is not True and not re.search(r"steps\.[\w-]+\.outcome\s*==\s*'failure'", cond):
                errors.append(f"{where}: {uses.split('@')[0]} downloads; make it continue-on-error "
                              "with a fallback, or retry it from a step gated on the first's outcome")
    return errors


def targets(root: Path) -> list[Path]:
    wfs = sorted((root / ".github" / "workflows").glob("unsloth-prebuilt*.yml"))
    actions: set[Path] = set()
    todo = list(wfs)
    seen: set[Path] = set()
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        for m in re.finditer(r"uses:\s*['\"]?(\./[^\s'\"#]+)", path.read_text()):
            local = LOCAL_ACTION.match(m.group(1).rstrip("/"))
            if local:
                act = root / ".github" / "actions" / local.group(1) / "action.yml"
                if act.is_file() and act not in actions:
                    actions.add(act)
                    todo.append(act)
    return wfs + sorted(actions)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".", help="repository root")
    args = ap.parse_args(argv)
    root = Path(args.root)
    files = targets(root)
    if not files:
        print(f"::error::no unsloth-prebuilt*.yml under {root}/.github/workflows")
        return 1
    fail = 0
    for path in files:
        rel = path.relative_to(root)
        try:
            doc = yaml.safe_load(path.read_text())
        except yaml.YAMLError as e:
            print(f"::error file={rel}::does not parse: {e}")
            fail = 1
            continue
        for err in check_doc(doc):
            print(f"::error file={rel}::{err}")
            fail = 1
    print(f"{len(files)} file(s) checked; {'FAIL' if fail else 'ok'}: every network step "
          f"{'must carry' if fail else 'carries'} a retry or a `{ALLOW} <reason>` comment")
    return fail


if __name__ == "__main__":
    sys.exit(main())
