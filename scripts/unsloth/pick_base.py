#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Pick the newest base tag the pin set merges onto, trying older aged tags when the newest refuses.

Upstream breaks a pin every so often (b12696..b12719 broke #386/#431 against the inkling2 clip and mix changes), and until the pin is repinned every
nightly then failed at resolve and published nothing. The newest aged tag is still tried first and still wins whenever it merges. Only when it fails
the pin merge (a conflict additive_merge.py refuses), merge_checks.py or pin_contract.py does this walk to the next older aged tag, newest first,
at most --depth tags back and never below the base of the last published release, so a fallback can never ship something older than what
users already have. The compile gate is not tried per candidate: it costs a CPU build each, and it still runs on whatever is picked.

If nothing merges, the newest tag is printed unchanged, so the resolve step's own merge loop fails exactly as it did before this existed.

Prints the chosen tag on stdout. Everything else goes to stderr, the job summary ($GITHUB_STEP_SUMMARY) and --note / --report.

Candidates come either from the upstream release list (the workflow: --releases, --first, --depth, --cutoff, --published) or are given
directly (--candidates NAME[=REF] ..., newest first; REF is a commit for a tag that is not published yet, for local dry runs).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG_RE = re.compile(r"^b(\d+)$")
OWN_TAG_RE = re.compile(r"^b(\d+)(?:-mix-[0-9a-f]+)?$")
UPSTREAM = "https://github.com/ggml-org/llama.cpp.git"
BOT = ["-c", "user.name=github-actions[bot]",
       "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com"]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def num(tag: str) -> int:
    m = TAG_RE.match(tag)
    return int(m.group(1)) if m else -1


def epoch(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def candidates_from_releases(releases: list, first: str, depth: int, cutoff: float | None,
                             floor: int | None) -> list[str]:
    """[first] + up to `depth` older upstream b#### tags, newest first, none below `floor`.

    Older is by tag number, which upstream assigns in order. `cutoff` is the aging window: an older tag is still only a candidate once it
    has been public long enough, exactly like the newest one. `floor` None means the published base is unknown, so nothing older is tried.
    """
    older = sorted({r["tag_name"] for r in releases
                    if not r.get("draft") and TAG_RE.match(r.get("tag_name", ""))
                    and num(r["tag_name"]) < num(first)
                    and (cutoff is None or epoch(r["published_at"]) <= cutoff)}, key=num, reverse=True)
    if floor is None:
        return [first]
    return [first] + [t for t in older if num(t) >= floor][:depth]


def published_floor(own: list) -> int | None:
    """Base number of the most recently published release of ours; 0 when there is none yet."""
    rel = [r for r in own if not r.get("draft") and OWN_TAG_RE.match(r.get("tag_name", ""))
           and r.get("published_at")]
    if not rel:
        return 0
    last = max(rel, key=lambda r: epoch(r["published_at"]))
    return int(OWN_TAG_RE.match(last["tag_name"]).group(1))


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", *BOT, *args], cwd=repo, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({r.returncode}): {r.stderr.strip()[-500:]}")
    return r


def have(repo: Path, ref: str) -> bool:
    return git(repo, "rev-parse", "--verify", "-q", f"{ref}^{{commit}}", check=False).returncode == 0


def fetch(repo: Path, *args: str) -> bool:
    """git fetch with the workflows' 3 tries, 15/30 s apart: one dropped fetch must not cost the nightly."""
    for i in (1, 2, 3):
        if git(repo, "fetch", "-q", "--no-tags", *args, check=False).returncode == 0:
            return True
        if i < 3:
            time.sleep(15 * i)
    return False


def ensure_objects(repo: Path, cands: list[tuple[str, str]], pins: list[dict], upstream: str) -> None:
    """Fetch only what is missing, so a local dry run or a test never touches the network."""
    if git(repo, "rev-parse", "--is-shallow-repository").stdout.strip() == "true":
        if not fetch(repo, "--unshallow", "origin"):
            raise RuntimeError("could not unshallow the checkout")
    for name, ref in cands:
        if ref == name and not have(repo, f"refs/tags/{name}"):
            if not fetch(repo, upstream, f"refs/tags/{name}:refs/tags/{name}"):
                raise RuntimeError(f"could not fetch upstream tag {name}")
        elif ref != name and not have(repo, ref) and not fetch(repo, upstream, ref):
            raise RuntimeError(f"could not fetch {ref} for candidate {name}")
    for p in pins:
        if not have(repo, p["sha"]):
            if not (fetch(repo, f"https://github.com/{p['repo']}.git", p["sha"])
                    or fetch(repo, "origin", f"refs/pins/{p['sha']}")):
                raise RuntimeError(f"could not fetch commit {p['sha']} for {p['repo']}#{p['number']}")


def run_check(script: str, *args: str, cwd: Path) -> tuple[bool, str]:
    r = subprocess.run([sys.executable, str(HERE / script), *args], cwd=cwd,
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-2000:]


def trial(repo: Path, ref: str, pins: list[dict], prs_json: str, tmp: Path) -> dict | None:
    """Merge every pin onto `ref` the way resolve does, then run the two tree checks. None = clean, else what refused."""
    git(repo, "checkout", "-q", "--detach", "--force", ref)  # ref is refs/tags/<name> or a commit
    for p in pins:
        name = f"{p['repo']}#{p['number']}"
        r = git(repo, "-c", "merge.conflictStyle=diff3", "merge", "--no-ff", "--no-edit",
                "-m", f"Merge {name} @ {p['sha']}", p["sha"], check=False)
        if r.returncode == 0:
            continue
        ok, _ = run_check("additive_merge.py", cwd=repo)
        if ok and not git(repo, "diff", "--name-only", "--diff-filter=U").stdout.strip():
            git(repo, "commit", "-q", "--no-edit")
            continue
        git(repo, "merge", "--abort", check=False)
        return {"pin": name, "check": "merge", "detail": "does not merge cleanly onto the base + the pins before it"}
    ok, out = run_check("merge_checks.py", "--root", ".", cwd=repo)
    if not ok:
        return {"pin": "", "check": "merge_checks", "detail": out.splitlines()[-1] if out else ""}
    report = tmp / "pin_contract.json"
    ok, out = run_check("pin_contract.py", "--root", ".", "--base", ref, "--prs-json", prs_json,
                        "--report", str(report), cwd=repo)
    if not ok:
        bad = []
        try:
            bad = [e["pin"] for e in json.loads(report.read_text()).get("pins", []) if e.get("problems")]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return {"pin": ", ".join(bad), "check": "pin_contract", "detail": out.splitlines()[-1] if out else ""}
    return None


def pins_from_pr_set(path: Path) -> list[dict]:
    url_re = re.compile(r"^https://github\.com/(ggml-org|unslothai)/llama\.cpp/pull/(\d+)/commits/([0-9a-f]{40})/?$")
    out = []
    for e in json.loads(path.read_text())["prs"]:
        url = e if isinstance(e, str) else e["url"]
        m = url_re.match(url)
        if not m:
            raise SystemExit(f"malformed pin url {url!r}")
        out.append({"repo": f"{m.group(1)}/llama.cpp", "number": int(m.group(2)), "sha": m.group(3), "url": url})
    return out


def describe(chosen: str, first: str, refused: list[dict]) -> str:
    """One line, no third-party text: tag names, pin repo#number and which check refused."""
    why = "; ".join(f"{r['tag']} refused by {r['pin'] or 'the merged tree'} ({r['check']})" for r in refused)
    return f"built {chosen} instead of the newest aged tag {first}: {why}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", type=Path, help="work tree to trial-merge in; restored afterwards")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--prs-json", help="the resolve step's PRS list (repo, number, sha)")
    src.add_argument("--pr-set", type=Path, help="scripts/unsloth/pr-set.json (local dry runs)")
    ap.add_argument("--first", help="the newest aged tag, the one resolve picked")
    ap.add_argument("--releases", type=Path, help="upstream releases JSON (GET repos/ggml-org/llama.cpp/releases)")
    ap.add_argument("--depth", type=int, default=6, help="older tags to try after --first (0 = strict)")
    ap.add_argument("--cutoff", type=float, help="aging window: only older tags published at or before this epoch")
    ap.add_argument("--published", type=Path, help="our releases JSON; the newest one's base is the floor. "
                                                   "Missing or unreadable = floor unknown = no fallback")
    ap.add_argument("--candidates", nargs="+", metavar="NAME[=REF]", help="explicit candidates, newest first")
    ap.add_argument("--floor", help="explicit floor tag (with --candidates)")
    ap.add_argument("--upstream", default=UPSTREAM, help="where missing tags are fetched from")
    ap.add_argument("--note", type=Path, help="write the one-line fallback note here (empty when none)")
    ap.add_argument("--report", type=Path, help="write a JSON report here")
    a = ap.parse_args()

    repo = a.repo.resolve()
    pins = pins_from_pr_set(a.pr_set) if a.pr_set else json.loads(a.prs_json)
    prs_json = json.dumps([{"repo": p["repo"], "number": p["number"], "sha": p["sha"], "url": p.get("url", "")}
                           for p in pins])

    if a.candidates:
        cands = [(c.split("=", 1)[0], c.split("=", 1)[-1]) for c in a.candidates]
        floor = num(a.floor) if a.floor else 0
        cands = cands[:1] + [c for c in cands[1:] if num(c[0]) >= floor][:a.depth]
    else:
        if not (a.first and a.releases):
            ap.error("--first and --releases are required without --candidates")
        floor = None
        if a.depth > 0:
            try:
                floor = published_floor(json.loads(a.published.read_text()))
            except (OSError, ValueError, TypeError, AttributeError) as e:
                log(f"::warning::base-tag fallback disabled: could not read the published releases ({e})")
        releases = json.loads(a.releases.read_text())
        cands = [(t, t) for t in candidates_from_releases(releases, a.first, a.depth, a.cutoff, floor)]
    first = cands[0][0]
    log(f"pick_base: candidates {' '.join(c[0] for c in cands)} (floor b{floor if floor is not None else '?'})")

    orig = git(repo, "rev-parse", "HEAD").stdout.strip()
    tmp = Path(os.environ.get("RUNNER_TEMP", "/tmp")) / f"pick_base_{os.getpid()}"
    tmp.mkdir(parents=True, exist_ok=True)
    refused: list[dict] = []
    chosen = None
    try:
        ensure_objects(repo, cands, pins, a.upstream)  # RuntimeError exits non-zero: resolve then builds --first strictly
        for name, ref in cands:
            why = trial(repo, f"refs/tags/{name}" if ref == name else ref, pins, prs_json, tmp)
            if why is None:
                chosen = name
                log(f"pick_base: {name} merges every pin and passes merge_checks + pin_contract")
                break
            why["tag"] = name
            refused.append(why)
            log(f"pick_base: {name} refused by {why['pin'] or 'the merged tree'} ({why['check']}): {why['detail']}")
    finally:
        git(repo, "merge", "--abort", check=False)
        git(repo, "reset", "-q", "--hard", check=False)
        git(repo, "checkout", "-q", "--detach", "--force", orig, check=False)

    note = ""
    if chosen is None:
        chosen = first
        if len(cands) > 1:
            log(f"::warning::base-tag fallback: none of {', '.join(c[0] for c in cands)} merges the pin set; "
                f"building {first} so the merge fails as before")
    elif chosen != first:
        note = describe(chosen, first, refused)
        log(f"::warning::base-tag fallback: {note}")
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as f:
                f.write(f"### Base-tag fallback\n\n{note}. Repin the refusing pins so the newest tag builds again.\n\n")
    if a.note:
        a.note.write_text(note)
    if a.report:
        a.report.write_text(json.dumps({"first": first, "chosen": chosen, "floor": floor,
                                        "candidates": [c[0] for c in cands], "refused": refused,
                                        "note": note}, indent=2))
    print(chosen)
    return 0


if __name__ == "__main__":
    sys.exit(main())
