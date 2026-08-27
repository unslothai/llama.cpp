#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for carry_vintage.py. Run: python3 scripts/unsloth/test_carry_vintage.py

Builds a throwaway repo shaped like a real carry: an upstream base tag, an
upstream PR branch with two commits on top of it, and a carry branch that
replayed the PR onto the base while deliberately dropping one file's change.

The case that matters is that dropped file. Its content equals the BASE
version, which is reachable from the PR head, so a vintage search that walks
the PR head's whole ancestry finds a "match" in a commit that is not part of
the PR at all, calls the file SUPERSEDED, and concludes that rebuilding from
the PR head is equivalent to merging -- which would re-add exactly what the
carry dropped.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "carry_vintage.py"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def git(d, *args):
    r = subprocess.run(["git", *args], cwd=d, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(" ".join(args) + ": " + r.stderr)
    return r.stdout.strip()


def write(d, name, text):
    (Path(d) / name).write_text(text)


def fixture():
    d = tempfile.mkdtemp(prefix="cv_")
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    write(d, "dropped.txt", "base version\n")
    write(d, "taken.txt", "base version\n")
    write(d, "edited.txt", "base version\n")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD")

    # Upstream PR: two commits, touching all three files.
    git(d, "checkout", "-q", "-b", "pr")
    write(d, "dropped.txt", "upstream v1\n")
    write(d, "taken.txt", "upstream v1\n")
    write(d, "edited.txt", "upstream v1\n")
    git(d, "commit", "-qam", "pr c1")
    mid = git(d, "rev-parse", "HEAD")
    write(d, "taken.txt", "upstream v2\n")
    write(d, "edited.txt", "upstream v2\n")
    git(d, "commit", "-qam", "pr c2")
    head = git(d, "rev-parse", "HEAD")

    # The carry: PR replayed onto base, with dropped.txt held at the base
    # version on purpose, edited.txt at an older PR vintage, and taken.txt
    # already at the PR head.
    git(d, "checkout", "-q", "-b", "carry", base)
    write(d, "dropped.txt", "base version\n")
    write(d, "taken.txt", "upstream v2\n")
    write(d, "edited.txt", "upstream v1\n")
    git(d, "commit", "-qam", "carry")
    carry = git(d, "rev-parse", "HEAD")
    return d, base, mid, head, carry


d, base, mid, head, carry = fixture()
report = str(Path(d) / "report.json")
r = subprocess.run([sys.executable, str(SCRIPT), "--carry", carry, "--pr-ref", head,
                    "--base", base, "--report", report],
                   cwd=d, capture_output=True, text=True)
check("runs clean", r.returncode == 0, r.stderr)
out = json.loads(Path(report).read_text())
sup = {e["path"]: e["vintage"] for e in out["superseded"]}

check("a file held at the BASE version is not called superseded",
      "dropped.txt" in out["diverged"], json.dumps(out, indent=1))
check("no vintage is a commit outside the PR",
      all(v in (mid, head) for v in sup.values()), json.dumps(sup, indent=1))
check("a file already at the PR head is superseded",
      sup.get("taken.txt") == head, json.dumps(sup, indent=1))
check("a file at an older PR commit is superseded at that vintage",
      sup.get("edited.txt") == mid, json.dumps(sup, indent=1))
check("a real divergence still forces a merge",
      "has to merge" in r.stdout, r.stdout[-300:])

# With the deliberately dropped file removed from the picture, everything the
# PR touched really is superseded and the rebuild advice is correct.
git(d, "checkout", "-q", "carry")
write(d, "dropped.txt", "upstream v1\n")
git(d, "commit", "-qam", "take it after all")
carry2 = git(d, "rev-parse", "HEAD")
r2 = subprocess.run([sys.executable, str(SCRIPT), "--carry", carry2, "--pr-ref", head,
                     "--base", base], cwd=d, capture_output=True, text=True)
check("still recommends a rebuild when nothing diverges",
      r2.returncode == 0 and "Nothing diverges" in r2.stdout, r2.stdout[-300:])

# A carry that deliberately OMITS a file the PR head still has. Nothing
# diverges, every file the carry does have is superseded, and the naive answer
# is "rebuild from the PR head" -- which restores the omitted file and loses
# the omission. A file the PR DELETED is a different case: the carry not having
# it is agreement, and a rebuild reproduces it exactly.
def omission_fixture():
    d = tempfile.mkdtemp(prefix="cv_")
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    write(d, "keep.txt", "base version\n")
    write(d, "doomed.txt", "base version\n")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD")

    # The PR edits keep.txt, adds win.cmake, and deletes doomed.txt.
    git(d, "checkout", "-q", "-b", "pr")
    write(d, "keep.txt", "upstream v1\n")
    write(d, "win.cmake", "windows-only build tweak\n")
    git(d, "rm", "-q", "doomed.txt")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "pr c1")
    head = git(d, "rev-parse", "HEAD")

    # The carry replays it but never took win.cmake.
    git(d, "checkout", "-q", "-b", "carry", base)
    write(d, "keep.txt", "upstream v1\n")
    git(d, "rm", "-q", "doomed.txt")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "carry without win.cmake")
    return d, base, head, git(d, "rev-parse", "HEAD")


d2, base2, head2, carry3 = omission_fixture()
# outside the repo, so the "writes nothing" check below sees a clean tree
report2 = str(Path(tempfile.mkdtemp(prefix="cvr_")) / "report.json")
r3 = subprocess.run([sys.executable, str(SCRIPT), "--carry", carry3, "--pr-ref", head2,
                     "--base", base2, "--report", report2],
                    cwd=d2, capture_output=True, text=True)
check("runs clean on an omitting carry", r3.returncode == 0, r3.stderr)
out2 = json.loads(Path(report2).read_text())
check("a file present at the PR head but not in the carry is OMITTED, not absent",
      out2.get("omitted") == ["win.cmake"] and "win.cmake" not in out2.get("absent", []),
      json.dumps(out2, indent=1))
check("a file the PR deleted stays merely ABSENT",
      out2.get("absent") == ["doomed.txt"], json.dumps(out2, indent=1))
check("an omitted file suppresses the rebuild recommendation",
      "Nothing diverges" not in r3.stdout, r3.stdout[-400:])
check("and says why rebuilding is not equivalent",
      "NOT equivalent" in r3.stdout, r3.stdout[-400:])
check("the omitting carry still writes nothing",
      git(d2, "status", "--porcelain") == "", git(d2, "status", "--porcelain"))

# Take the omitted file, and the rebuild advice is correct again.
write(d2, "win.cmake", "windows-only build tweak\n")
git(d2, "add", "-A")
git(d2, "commit", "-qm", "take win.cmake after all")
r4 = subprocess.run([sys.executable, str(SCRIPT), "--carry", git(d2, "rev-parse", "HEAD"),
                     "--pr-ref", head2, "--base", base2], cwd=d2, capture_output=True, text=True)
check("a PR deletion alone still allows the rebuild",
      r4.returncode == 0 and "Nothing diverges" in r4.stdout, r4.stdout[-400:])

# A multi-commit PR that does not touch every file in its FIRST commit. The
# commits before the one that first changed a path still carry the fork's blob,
# and they are inside fork..head, so a carry deliberately holding that file at
# the base version matches one of them and is called SUPERSEDED -- the same
# wrong "rebuilding is equivalent" answer the fork bound was meant to end, one
# commit further in.
def late_touch_fixture():
    d = tempfile.mkdtemp(prefix="cv_")
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    write(d, "held.txt", "base version\n")
    write(d, "early.txt", "base version\n")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD")

    # c1 touches early.txt only, so held.txt is still the base blob AT c1.
    git(d, "checkout", "-q", "-b", "pr")
    write(d, "early.txt", "upstream v1\n")
    git(d, "commit", "-qam", "pr c1")
    # c2 is the first commit to touch held.txt.
    write(d, "held.txt", "upstream v2\n")
    git(d, "commit", "-qam", "pr c2")
    head = git(d, "rev-parse", "HEAD")

    # The carry took early.txt and holds held.txt at the base version.
    git(d, "checkout", "-q", "-b", "carry", base)
    write(d, "early.txt", "upstream v1\n")
    git(d, "commit", "-qam", "carry")
    return d, base, head, git(d, "rev-parse", "HEAD")


d3, base3, head3, carry4 = late_touch_fixture()
report3 = str(Path(tempfile.mkdtemp(prefix="cvr_")) / "report.json")
r5 = subprocess.run([sys.executable, str(SCRIPT), "--carry", carry4, "--pr-ref", head3,
                     "--base", base3, "--report", report3],
                    cwd=d3, capture_output=True, text=True)
check("runs clean on a late-touch PR", r5.returncode == 0, r5.stderr)
out3 = json.loads(Path(report3).read_text())
check("a file held at base is not superseded by a PR commit that predates its first change",
      "held.txt" in out3["diverged"], json.dumps(out3, indent=1))
check("no in-range commit before the first change counts as a vintage",
      all(e["path"] != "held.txt" for e in out3["superseded"]), json.dumps(out3, indent=1))
check("and the rebuild advice is withheld",
      "Nothing diverges" not in r5.stdout, r5.stdout[-400:])
check("a file the carry really did take is still superseded",
      any(e["path"] == "early.txt" for e in out3["superseded"]), json.dumps(out3, indent=1))
check("the late-touch run still writes nothing",
      git(d3, "status", "--porcelain") == "", git(d3, "status", "--porcelain"))

# A PR that RENAMES a file while the carry deliberately keeps the old path.
# `git diff --name-only` prints only the new name of a detected rename, so the
# old path never entered the file list at all: the new path came back
# SUPERSEDED, nothing diverged, and the summary recommended a rebuild -- which
# deletes the path the carry is holding, unreported.
def rename_fixture():
    d = tempfile.mkdtemp(prefix="cv_")
    git(d, "init", "-q", "-b", "main")
    git(d, "config", "user.email", "t@t")
    git(d, "config", "user.name", "t")
    # Long enough that git scores the move as a rename rather than add+delete.
    write(d, "old.py", "".join(f"line {i}\n" for i in range(40)))
    git(d, "add", "-A")
    git(d, "commit", "-qm", "base")
    base = git(d, "rev-parse", "HEAD")

    git(d, "checkout", "-q", "-b", "pr")
    git(d, "mv", "old.py", "new.py")
    git(d, "commit", "-qm", "pr renames it")
    head = git(d, "rev-parse", "HEAD")

    # The carry takes the new path AND keeps the old one, on purpose.
    git(d, "checkout", "-q", "-b", "carry", base)
    write(d, "new.py", "".join(f"line {i}\n" for i in range(40)))
    git(d, "add", "-A")
    git(d, "commit", "-qm", "carry keeps both")
    return d, base, head, git(d, "rev-parse", "HEAD")


d4, base4, head4, carry5 = rename_fixture()
report4 = str(Path(tempfile.mkdtemp(prefix="cvr_")) / "report.json")
r6 = subprocess.run([sys.executable, str(SCRIPT), "--carry", carry5, "--pr-ref", head4,
                     "--base", base4, "--report", report4],
                    cwd=d4, capture_output=True, text=True)
check("runs clean on a renaming PR", r6.returncode == 0, r6.stderr)
out4 = json.loads(Path(report4).read_text())
check("the old side of a rename is scanned at all",
      "old.py" in out4["diverged"] + out4["absent"] + out4["omitted"]
      or any(e["path"] == "old.py" for e in out4["superseded"]),
      json.dumps(out4, indent=1))
check("a retained old path is reported as diverged, not silently dropped",
      "old.py" in out4["diverged"], json.dumps(out4, indent=1))
check("a rename does not license the rebuild advice",
      "Nothing diverges" not in r6.stdout, r6.stdout[-400:])
check("the renaming run still writes nothing",
      git(d4, "status", "--porcelain") == "", git(d4, "status", "--porcelain"))

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all carry_vintage tests passed")
sys.exit(1 if FAILS else 0)
