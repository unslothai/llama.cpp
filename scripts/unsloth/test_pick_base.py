#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for pick_base.py. Run: python3 scripts/unsloth/test_pick_base.py

Every case runs the real script against a real synthetic upstream history, so the merges, additive_merge.py,
merge_checks.py and pin_contract.py all run for real:

    b1 -- b2 -- b3 -- b4 -- b5 -- b6        upstream tags (b4 and b5 rewrite the line the pin edits; b6 adds a dict key the pin also adds)
     \\
      pin                                    unslothai#1: edits that line and adds the key
"""
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "pick_base.py"
WORKFLOW = HERE.parent.parent / ".github/workflows/unsloth-prebuilt.yml"
FAILS = []
T0 = 1_700_000_000  # b<n> is published at T0 + n hours


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(repo, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
                       cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, (args, r.stderr)
    return r.stdout.strip()


def commit(repo, files, msg, tag=None):
    for path, text in files.items():
        p = repo / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", msg)
    if tag:
        git(repo, "tag", tag)


MODEL = "void build_model() {\n    int heads = 8;\n}\n"
KEYS = 'NAMES = {\n    "llama": 1,\n}\n'


def make_repo():
    d = Path(tempfile.mkdtemp(prefix="pb_"))
    git(d, "init", "-q", "-b", "master")
    commit(d, {"src/llama-model.cpp": MODEL, "gguf-py/gguf/names.py": KEYS}, "b1", "b1")
    git(d, "checkout", "-qb", "pin")
    commit(d, {"src/llama-model.cpp": MODEL.replace("int heads = 8;", "int heads = 8 + inkling_extra_heads();"),
               "gguf-py/gguf/names.py": KEYS.replace('{\n', '{\n    "inkling": 2,\n')}, "pin")
    pin = git(d, "rev-parse", "HEAD")
    git(d, "checkout", "-q", "master")
    commit(d, {"README.md": "two\n"}, "b2", "b2")
    commit(d, {"README.md": "three\n"}, "b3", "b3")
    commit(d, {"src/llama-model.cpp": MODEL.replace("int heads = 8;", "int heads = hparams.n_head();")}, "b4", "b4")
    commit(d, {"src/llama-model.cpp": MODEL.replace("int heads = 8;", "int heads = hparams.n_head(il);")}, "b5", "b5")
    # b6 undoes the b4/b5 rewrite (so the pin merges as text again) and adds the pin's dict key at the bottom:
    # the merge is clean and the result defines "inkling" twice, which only merge_checks.py sees.
    commit(d, {"src/llama-model.cpp": MODEL, "gguf-py/gguf/names.py": KEYS.replace('}\n', '    "inkling": 3,\n}\n')}, "b6", "b6")
    # The checkout resolve runs from: something that is not any upstream tag, restored afterwards.
    git(d, "checkout", "-q", "-b", "fork", "b1")
    commit(d, {"scripts/unsloth/pr-set.json": "{}\n"}, "fork")
    prs = [{"repo": "unslothai/llama.cpp", "number": 1, "sha": pin,
            "url": f"https://github.com/unslothai/llama.cpp/pull/1/commits/{pin}", "title": "pin"}]
    return d, prs


def releases(*nums, draft=()):
    return [{"tag_name": f"b{n}", "draft": n in draft, "prerelease": True,
             "published_at": datetime.fromtimestamp(T0 + 3600 * n, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
            for n in nums]


def run(repo, prs, *args, published=None):
    tmp = Path(tempfile.mkdtemp(prefix="pbrun_"))
    rel = tmp / "rel.json"
    rel.write_text(json.dumps(releases(1, 2, 3, 4, 5, 6)))
    extra = []
    if published is not None:
        (tmp / "own.json").write_text(json.dumps(published))
        extra = ["--published", str(tmp / "own.json")]
    note = tmp / "note.txt"
    report = tmp / "report.json"
    r = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--prs-json", json.dumps(prs),
                        "--releases", str(rel), "--note", str(note), "--report", str(report), *extra, *args],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(tmp),
                                                            "GITHUB_STEP_SUMMARY": str(tmp / "summary.md"),
                                                            "HOME": str(tmp)})
    out = {"rc": r.returncode, "chosen": r.stdout.strip(), "stderr": r.stderr,
           "note": note.read_text() if note.exists() else None,
           "report": json.loads(report.read_text()) if report.exists() else {},
           "summary": (tmp / "summary.md").read_text() if (tmp / "summary.md").exists() else ""}
    return out


def own(tag, n=10):
    return [{"tag_name": tag, "draft": False, "published_at": "2026-01-01T00:00:00Z"},
            {"tag_name": "b1-mix-0000000", "draft": False, "published_at": "2025-01-01T00:00:00Z"},
            {"tag_name": f"b{n}-mix-1111111", "draft": True, "published_at": None}]


def restored(repo, head):
    return (git(repo, "rev-parse", "HEAD") == head and not git(repo, "status", "--porcelain")
            and (repo / "scripts/unsloth/pr-set.json").exists())


def main():
    repo, prs = make_repo()
    head = git(repo, "rev-parse", "HEAD")

    # The newest aged tag merges: no fallback, no note, no summary line.
    r = run(repo, prs, "--first", "b3", published=own("b1-mix-abcdef0"))
    check("newest tag merges -> it is built", r["rc"] == 0 and r["chosen"] == "b3", r["stderr"])
    check("no fallback -> empty note and no summary", r["note"] == "" and r["summary"] == "", r)
    check("checkout restored after a clean pick", restored(repo, head))

    # Tag N (b4) refuses the pin, N-1 (b3) merges: b3 is built and the note names b4 and the refusing pin.
    r = run(repo, prs, "--first", "b4", published=own("b1-mix-abcdef0"))
    check("N refuses, N-1 merges -> N-1 is built", r["rc"] == 0 and r["chosen"] == "b3", r["stderr"])
    check("note names the skipped tag and the refusing pin",
          r["note"] == "built b3 instead of the newest aged tag b4: b4 refused by unslothai/llama.cpp#1 (merge)", r["note"])
    check("fallback is a warning annotation", "::warning::base-tag fallback: built b3" in r["stderr"], r["stderr"])
    check("fallback is in the job summary", "### Base-tag fallback" in r["summary"] and "b4 refused" in r["summary"])
    check("checkout restored after a fallback", restored(repo, head))

    # Several refuse: newest first, every skipped tag named.
    r = run(repo, prs, "--first", "b5", published=own("b2-mix-abcdef0"))
    check("b5 and b4 refuse -> b3", r["chosen"] == "b3" and r["report"].get("candidates") == ["b5", "b4", "b3", "b2"],
          r["report"])
    check("note names both skipped tags", "b5 refused by unslothai/llama.cpp#1 (merge); b4 refused" in (r["note"] or ""))

    # merge_checks.py refusing counts as a refusal: b6 merges as text but defines "inkling" twice.
    r = run(repo, prs, "--first", "b6", published=own("b1-mix-abcdef0"))
    check("merge_checks failure falls back", r["chosen"] == "b3" and "b6 refused by the merged tree (merge_checks)" in (r["note"] or ""),
          r["note"])

    # Never below the last published release's base: published b4 -> only b5, b4 are candidates, both refuse.
    r = run(repo, prs, "--first", "b5", published=own("b4-mix-abcdef0"))
    check("no fallback below the published base", r["report"].get("candidates") == ["b5", "b4"], r["report"])
    check("all candidates refuse -> newest printed unchanged, no note (the merge loop then fails as before)",
          r["rc"] == 0 and r["chosen"] == "b5" and r["note"] == "", r)
    check("all candidates refusing is a warning", "none of b5, b4 merges the pin set" in r["stderr"], r["stderr"])

    # The floor is the most recently PUBLISHED release, drafts ignored; a plain b#### release counts too.
    r = run(repo, prs, "--first", "b5", published=own("b3"))
    check("plain-tag published release sets the floor", r["report"].get("candidates") == ["b5", "b4", "b3"]
          and r["chosen"] == "b3", r["report"])

    # Depth bounds the walk: depth 1 tries b5 and b4 only.
    r = run(repo, prs, "--first", "b5", "--depth", "1", published=own("b1-mix-abcdef0"))
    check("depth bounds the walk", r["report"].get("candidates") == ["b5", "b4"] and r["chosen"] == "b5", r["report"])

    # Strict: depth 0 is the explicit-tag default; only the given tag, which refuses, so it is printed as is.
    r = run(repo, prs, "--first", "b5", "--depth", "0", published=own("b1-mix-abcdef0"))
    check("depth 0 is strict", r["report"].get("candidates") == ["b5"] and r["chosen"] == "b5" and r["note"] == "", r)

    # Published releases unreadable: the floor is unknown, so nothing older is tried.
    r = run(repo, prs, "--first", "b5")
    check("unknown floor -> strict", r["report"].get("candidates") == ["b5"] and "fallback disabled" in r["stderr"], r)

    # The aging window applies to the older tags too, and drafts never count.
    cutoff = str(T0 + 3600 * 3 - 1)  # b3 and newer are too young
    r = run(repo, prs, "--first", "b5", "--cutoff", cutoff, published=own("b1-mix-abcdef0"))
    check("older candidates respect the aging cutoff", r["report"].get("candidates") == ["b5", "b2", "b1"]
          and r["chosen"] == "b2", r["report"])

    # Explicit candidates (local dry runs): NAME=REF builds a tag that is not published yet from a commit.
    b4 = git(repo, "rev-parse", "b4")
    tmp = Path(tempfile.mkdtemp(prefix="pbc_"))
    p = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--prs-json", json.dumps(prs),
                        "--candidates", f"b9={b4}", "b3", "--note", str(tmp / "n")],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(tmp), "HOME": str(tmp)})
    check("--candidates NAME=REF dry run", p.stdout.strip() == "b3" and "b9 refused" in (tmp / "n").read_text(), p.stderr)

    # A checkout with edits to tracked files is refused before any trial, and the edits survive.
    tracked = git(repo, "ls-files").split()[0]
    before = (repo / tracked).read_text()
    (repo / tracked).write_text(before + "local edit\n")
    p = subprocess.run([sys.executable, str(SCRIPT), "--repo", str(repo), "--prs-json", json.dumps(prs),
                        "--candidates", "b5", "b3"],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "RUNNER_TEMP": str(tmp), "HOME": str(tmp)})
    check("dirty tracked files are refused and kept", p.returncode != 0 and "uncommitted changes" in p.stderr
          and (repo / tracked).read_text() == before + "local edit\n", p.stderr)
    (repo / tracked).write_text(before)
    check("checkout restored after a dry run", restored(repo, head))

    # The workflow side of strictness: an explicit tag defaults to 0 (strict), "latest" to the env depth, and the
    # base_fallback input overrides both. These are the exact lines from the resolve step, run in bash.
    wf = WORKFLOW.read_text()
    lines = [ln.strip() for ln in wf.splitlines() if re.match(r'\s*(\[ -n "\$FB_N" \]|\[\[ "\$FB_N" =~)', ln)]
    check("workflow carries the FB_N default and validation lines", len(lines) == 2, lines)
    for req, given, want in (("b12690", "", "0"), ("latest", "", "6"), ("b12690", "3", "3"), ("latest", "0", "0")):
        sh = (f'set -euo pipefail; UNSLOTH_LLAMA_BASE_FALLBACK_DEPTH=6; REQ={req}; FB_N="{given}"\n'
              + "\n".join(lines) + '\necho "$FB_N"')
        out = subprocess.run(["bash", "-c", sh], capture_output=True, text=True)
        check(f"workflow FB_N for tag={req} base_fallback='{given}' -> {want}", out.stdout.strip() == want, out.stderr)
    out = subprocess.run(["bash", "-c", 'set -euo pipefail; UNSLOTH_LLAMA_BASE_FALLBACK_DEPTH=6; REQ=latest; FB_N="x"\n'
                          + "\n".join(lines)], capture_output=True, text=True)
    check("workflow refuses a non-numeric base_fallback", out.returncode != 0 and "refusing base_fallback" in out.stderr)
    check("workflow env default depth is 6", 'UNSLOTH_LLAMA_BASE_FALLBACK_DEPTH: "6"' in wf)

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("all pick_base tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
