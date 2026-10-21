#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Merge the current base tag into each pinned PR branch we control, and repin.

The nightly builds an upstream release tag plus a list of pinned PR commits.
Upstream moves several times a day, so a pin that merged yesterday routinely
stops merging today -- that is what broke four nightlies in a week, and every
fix was the same mechanical merge done by hand.

This does that merge, and only where it is safe to:

  * branches we own (see OWNED). A third-party PR is reported, never pushed to.
  * pins that are still their branch head. If the author has pushed past the
    pin, merging into the branch would silently widen the release to include
    code nobody reviewed, which is the exact property the pin file exists to
    hold. Report it and let a human decide.
  * conflicts that additive_merge.py can prove are pure add/add. Anything else
    is left alone and reported.

Writes the new pins back to pr-set.json and prints a markdown report. Pushing
and opening the PR is the caller's job; nothing here talks to a remote except
to fetch.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Branches we may push to. Anything else gets a report line and no write.
OWNED = ("unslothai/", "danielhanchen/")

PIN_RE = re.compile(
    r"^https://github\.com/([^/]+)/llama\.cpp/pull/(\d+)/commits/([0-9a-f]{40})/?$"
)
HERE = Path(__file__).resolve().parent


def run(args, cwd=None, check=True, quiet=True):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(args[:4])}... failed: {r.stderr.strip()[:400]}")
    if not quiet and r.stdout:
        print(r.stdout.rstrip())
    return r


def gh_json(path):
    r = run(["gh", "api", path], check=False)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def load_pins(pr_set: Path) -> tuple[dict, list[dict]]:
    data = json.loads(pr_set.read_text())
    pins = []
    for entry in data["prs"]:
        url = entry if isinstance(entry, str) else entry["url"]
        required = True if isinstance(entry, str) else entry.get("required", True)
        m = PIN_RE.match(url)
        if not m:
            raise SystemExit(f"malformed pin: {url}")
        pins.append(
            {
                "url": url,
                "required": required,
                "src": f"{m.group(1)}/llama.cpp",
                "num": int(m.group(2)),
                "sha": m.group(3),
            }
        )
    return data, pins


def repin_one(pin: dict, base: str, work: Path) -> dict:
    """Try to bring one pin up to `base`. Never raises for an expected refusal."""
    out = dict(pin, action="skip", note="", new_sha="", files=[], hunks=[])
    pr = gh_json(f"repos/{pin['src']}/pulls/{pin['num']}")
    if pr is None:
        out["note"] = "could not read the PR from the API"
        return out
    if pr.get("state") != "open":
        out["note"] = f"PR is {pr.get('state')}; not repinning a closed PR"
        return out

    head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name")
    head_ref = pr.get("head", {}).get("ref")
    head_sha = pr.get("head", {}).get("sha")
    out.update(head_repo=head_repo, head_ref=head_ref)

    if not head_repo:
        out["note"] = "head repository was deleted"
        return out
    if not head_repo.startswith(OWNED):
        out["action"] = "third-party"
        out["note"] = f"`{head_repo}` is not ours; ask the author to merge master"
        return out
    if head_sha != pin["sha"]:
        out["note"] = (
            f"branch head `{head_sha[:10]}` has moved past the pin `{pin['sha'][:10]}`; "
            "repinning would pull in unreviewed commits"
        )
        return out

    repo = work / f"r{pin['num']}"
    run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
         f"https://github.com/{pin['src']}.git", str(repo)])
    run(["git", "fetch", "-q", "--no-tags", "origin", pin["sha"]], cwd=repo, check=False)
    r = run(["git", "fetch", "-q", "--no-tags",
             "https://github.com/ggml-org/llama.cpp.git",
             f"refs/tags/{base}:refs/tags/{base}"], cwd=repo, check=False)
    if r.returncode != 0:
        out["note"] = f"could not fetch base tag {base}"
        return out
    if run(["git", "rev-parse", "--verify", f"{pin['sha']}^{{commit}}"],
           cwd=repo, check=False).returncode != 0:
        out["note"] = f"pinned commit {pin['sha'][:10]} is gone (force-pushed away)"
        return out

    run(["git", "checkout", "-q", "--detach", pin["sha"]], cwd=repo)
    if run(["git", "merge-base", "--is-ancestor", f"refs/tags/{base}", "HEAD"],
           cwd=repo, check=False).returncode == 0:
        out["note"] = f"already contains {base}"
        return out

    # diff3 is what makes the add/add proof possible: without the base section
    # an edit/edit conflict is indistinguishable from an add/add one.
    git_id = ["-c", "user.name=unsloth-repin-bot",
              "-c", "user.email=unsloth-repin-bot@users.noreply.github.com"]
    m = run(["git", "-c", "merge.conflictStyle=diff3", *git_id, "merge", "--no-ff",
             "--no-edit", "-m", f"Merge {base} into {head_ref}", f"refs/tags/{base}"],
            cwd=repo, check=False)

    if m.returncode != 0:
        report = work / f"res{pin['num']}.json"
        rc = subprocess.run(
            [sys.executable, str(HERE / "additive_merge.py"),
             "--repo", str(repo), "--report", str(report)],
            capture_output=True, text=True,
        ).returncode
        res = json.loads(report.read_text()) if report.exists() else {}
        if rc != 0:
            out["action"] = "conflict"
            refused = res.get("refused", [])
            out["files"] = [x["file"] for x in refused if x["file"] != "-"]
            if out["files"]:
                out["note"] = "; ".join(f"`{x['file']}`: {x['reason']}" for x in refused)
            else:
                # git refused the merge without leaving a single conflicted
                # file, so the conflict report explains nothing. Its stderr is
                # the only thing that does, and discarding it turns a
                # diagnosable failure into "no conflicted files".
                tail = ((m.stderr or "") + (m.stdout or "")).strip().splitlines()
                out["note"] = ("merge failed with no conflicts: " + " / ".join(tail[-3:])
                               if tail else "merge failed and git said nothing")
            run(["git", "merge", "--abort"], cwd=repo, check=False)
            return out
        out["hunks"] = [
            {"file": f["file"], **h} for f in res.get("resolved", []) for h in f["hunks"]
        ]
        out["files"] = [f["file"] for f in res.get("resolved", [])]
        run(["git", *git_id, "commit", "-q", "--no-edit"], cwd=repo)

    new_sha = run(["git", "rev-parse", "HEAD"], cwd=repo).stdout.strip()
    out["action"] = "repin"
    out["new_sha"] = new_sha
    out["repo_path"] = str(repo)
    out["touches_workflows"] = bool(
        run(["git", "diff", "--name-only", f"{pin['sha']}..HEAD", "--",
             ".github/workflows"], cwd=repo).stdout.strip()
    )
    return out


def markdown(base: str, results: list[dict]) -> str:
    L = [f"Base tag: `{base}`", ""]
    L += ["| pin | branch | outcome |", "|---|---|---|"]
    for r in results:
        pin = f"[`{r['src']}#{r['num']}`](https://github.com/{r['src']}/pull/{r['num']})"
        branch = f"`{r.get('head_repo') or '?'}:{r.get('head_ref') or '?'}`"
        if r["action"] == "repin":
            what = f"repinned `{r['sha'][:10]}` to `{r['new_sha'][:10]}`"
            if r["hunks"]:
                what += f" ({len(r['hunks'])} add/add hunk(s) resolved)"
        elif r["action"] == "conflict":
            what = f"**conflict, not resolvable automatically** -- {r['note']}"
        elif r["action"] == "third-party":
            what = f"not ours -- {r['note']}"
        else:
            what = r["note"] or "no change"
        L.append(f"| {pin} | {branch} | {what} |")
    L.append("")

    for r in results:
        if not r.get("hunks"):
            continue
        L += [f"<details><summary>Resolutions for <code>{r['src']}#{r['num']}</code></summary>", ""]
        for h in r["hunks"]:
            L += [f"`{h['file']}`", "", "```diff"]
            L += [f"-{x}" for x in h["ours"].rstrip("\n").split("\n")]
            L += [f"-{x}" for x in h["theirs"].rstrip("\n").split("\n")]
            L += [f"+{x}" for x in h["resolution"].rstrip("\n").split("\n")]
            L += ["```", ""]
        L += ["</details>", ""]

    if any(r.get("touches_workflows") for r in results):
        L += ["Some merges carry upstream changes under `.github/workflows/`, so the "
              "push needs a token with workflow write permission.", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pr-set", required=True)
    ap.add_argument("--base", required=True, help="upstream release tag to merge in")
    ap.add_argument("--work", required=True, help="scratch directory for clones")
    ap.add_argument("--report", help="write a JSON report here")
    ap.add_argument("--markdown", help="write the human-readable report here")
    args = ap.parse_args()

    pr_set = Path(args.pr_set)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    _, pins = load_pins(pr_set)

    results = []
    for pin in pins:
        try:
            r = repin_one(pin, args.base, work)
        except RuntimeError as e:
            r = dict(pin, action="skip", note=f"error: {e}", new_sha="", files=[], hunks=[])
        results.append(r)
        print(f"{r['src']}#{r['num']}: {r['action']} {r['note']}".rstrip())

    # Swap the shas in the raw text rather than re-serialising. Rewriting the
    # JSON would reflow the whole file and bury a four-character change in a
    # whole-file diff, which is the opposite of what a reviewer needs here.
    text = pr_set.read_text()
    changed = 0
    for r in results:
        if r["action"] != "repin":
            continue
        if r["sha"] not in text:
            print(f"::warning::{r['src']}#{r['num']}: pin not found verbatim; not rewritten")
            continue
        text = text.replace(r["sha"], r["new_sha"])
        changed += 1
    if changed:
        pr_set.write_text(text)

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"base": args.base, "changed": changed, "results": results}, indent=2))
    if args.markdown:
        Path(args.markdown).write_text(markdown(args.base, results))

    blocked = [r for r in results if r["action"] in ("conflict", "third-party")]
    print(f"\n{changed} repinned, {len(blocked)} needing a human")
    return 0


if __name__ == "__main__":
    sys.exit(main())
