#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for finalize_artifacts.py and the Finalize job that uses it.

Self-running like the sibling tests. Besides the decision itself, this runs the
real `keep` step body out of unsloth-prebuilt.yml, evaluates the Finalize step
conditions as written, and walks the case that broke run 45410618008: one leg
fails, `gh run rerun --failed` reruns it, and the rerun publishes.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WF = REPO / ".github" / "workflows"
sys.path.insert(0, str(HERE))
import finalize_artifacts as fa  # noqa: E402

FAILS: list[str] = []
FINALIZE = "Finalize: prune, reclaim, report"
ASSEMBLE = "Assemble metadata + publish"


def check(name: str, ok: bool, out: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        FAILS.append(name)
        if out:
            print("    ---")
            for line in str(out).strip().split("\n"):
                print(f"    {line}")


def load(name: str) -> dict:
    # yaml reads the bare `on:` key as True; nothing here needs it.
    return yaml.safe_load((WF / name).read_text())


PREBUILT = load("unsloth-prebuilt.yml")
RECLAIM = PREBUILT["jobs"]["reclaim"]
STEPS = RECLAIM["steps"]
MAX = int(RECLAIM["env"]["RERUN_MAX_ATTEMPT"])


def step(name_part: str) -> dict:
    hits = [s for s in STEPS if name_part in s.get("name", "")]
    assert len(hits) == 1, (name_part, [s.get("name") for s in hits])
    return hits[0]


LEGS = ["CUDA / x64/cuda12-portable", "CUDA Windows / x64/cuda13-newer", "ROCm / linux/gfx1151",
        "macOS / arm64", "CPU / windows/x64", "CPU / linux/x64 + Vulkan", "Vulkan / windows/x64"]


def jobs_tsv(conclusions: dict[str, str]) -> str:
    """The waiter's jobs.tsv (Finalize itself excluded), every job completed."""
    names = ["Resolve tag", "Prune superseded ccache generations", "CUDA / warm linux-x64/cuda12-portable 1/3"]
    names += LEGS + [ASSEMBLE]
    return "".join(f"{n}\tcompleted\t{conclusions.get(n, 'success')}\n" for n in names)


# ── 1. The decision ──
print("decision")
ok_run = fa.parse_jobs(jobs_tsv({}))
leg_failed = fa.parse_jobs(jobs_tsv({"CPU / windows/x64": "failure", ASSEMBLE: "failure"}))
d = fa.decide(leg_failed, 1, 3, False)
check("a failed leg on attempt 1 keeps and asks for a rerun", d["rerun"] and d["keep"], d)
check("the reason names the failed leg", "CPU / windows/x64" in d["reason"], d)
d = fa.decide(leg_failed, 2, 3, False)
check("attempt 2 of 3 still keeps", d["rerun"] and d["keep"], d)
d = fa.decide(leg_failed, 3, 3, False)
check("the last attempt deletes", not d["rerun"] and not d["keep"], d)
d = fa.decide(leg_failed, 3, 3, True)
check("keep_artifacts still keeps on the last attempt, without a rerun", not d["rerun"] and d["keep"], d)
d = fa.decide(fa.parse_jobs(jobs_tsv({"Resolve tag": "failure", ASSEMBLE: "skipped"} | {n: "skipped" for n in LEGS})), 1, 3, False)
check("a resolve failure deletes", not d["rerun"] and not d["keep"], d)
check("a missing resolve record deletes", not fa.decide([], 1, 3, False)["keep"])
d = fa.decide(ok_run, 1, 3, False)
check("publish:false with every job green deletes (nothing to rerun)", not d["rerun"] and not d["keep"], d)
d = fa.decide(ok_run, 1, 3, True)
check("publish:false with keep_artifacts keeps, without a rerun", not d["rerun"] and d["keep"], d)
d = fa.decide(fa.parse_jobs(jobs_tsv({ASSEMBLE: "failure"})), 1, 3, False)
check("assemble failing alone (waiter deadline) keeps for its rerun", d["rerun"] and d["keep"], d)
d = fa.decide(fa.parse_jobs(jobs_tsv({"macOS / arm64": "cancelled", ASSEMBLE: "cancelled"})), 1, 3, False)
check("a cancelled leg keeps", d["rerun"] and d["keep"], d)
d = fa.decide(fa.parse_jobs(jobs_tsv({"Prune superseded ccache generations": "failure",
                                      "CUDA / warm linux-x64/cuda12-portable 1/3": "failure"})), 1, 3, False)
check("best-effort prune / warm failures alone do not keep", not d["rerun"] and not d["keep"], d)
d = fa.decide(fa.parse_jobs(jobs_tsv({n: "skipped" for n in LEGS[:3]})), 1, 3, False)
check("legs a subset dispatch skipped are not rerunnable", not d["rerun"], d)

# ── 2. The real `keep` step body, run as the runner would ──
print("keep step")
KEEP = step("Decide whether a rerun")


def run_keep(tsv: str | None, attempt: int, keep_requested: str = "false") -> tuple[int, dict, str]:
    tmp = Path(tempfile.mkdtemp())
    if tsv is not None:
        (tmp / "jobs.tsv").write_text(tsv)
    env = dict(os.environ, RUNNER_TEMP=str(tmp), GITHUB_OUTPUT=str(tmp / "out"),
               GITHUB_STEP_SUMMARY=str(tmp / "summary"), ATTEMPT=str(attempt),
               KEEP_REQUESTED=keep_requested, RERUN_MAX_ATTEMPT=str(MAX))
    p = subprocess.run(["bash", "-e", "-c", KEEP["run"]], cwd=REPO, env=env, capture_output=True, text=True)
    outs = {}
    if (tmp / "out").is_file():
        for line in (tmp / "out").read_text().splitlines():
            k, _, v = line.partition("=")
            outs[k] = v
    return p.returncode, outs, p.stdout + p.stderr


rc, outs, log = run_keep(jobs_tsv({"CPU / windows/x64": "failure", ASSEMBLE: "failure"}), 1)
check("step emits rerun=true keep=true for a failed leg", rc == 0 and outs == {"rerun": "true", "keep": "true"}, log)
rc, outs, log = run_keep(jobs_tsv({}), 1)
check("step emits rerun=false keep=false for a green unpublished run", rc == 0 and outs == {"rerun": "false", "keep": "false"}, log)
rc, outs, log = run_keep(jobs_tsv({}), 1, "true")
check("step honours keep_artifacts", rc == 0 and outs == {"rerun": "false", "keep": "true"}, log)
rc, outs, log = run_keep(None, 1)
check("no jobs.tsv at all deletes rather than crashing", rc == 0 and outs.get("keep") == "false", log)
check("the step env passes inputs.keep_artifacts", "inputs.keep_artifacts" in str(KEEP.get("env")))


# ── 3. Finalize's step conditions, as written ──
def evaluate(expr: str, ctx: dict) -> bool:
    """Just enough of GitHub's expression language for the Finalize `if:`s."""
    e = expr.strip()
    if e.startswith("${{") and e.endswith("}}"):
        e = e[3:-2]
    e = e.replace("&&", " and ").replace("||", " or ")
    e = re.sub(r"!(?!=)", " not ", e)
    e = e.replace("always()", "True")

    def ref(m: re.Match) -> str:
        key = m.group(0)
        if key not in ctx:
            raise KeyError(f"no value for {key} in {expr!r}")
        return repr(ctx[key])
    e = re.sub(r"\b(?:steps|needs|inputs|github)\.[\w.-]+", ref, e)
    e = re.sub(r"\btrue\b", "True", re.sub(r"\bfalse\b", "False", e))
    return bool(eval(e, {}))  # noqa: S307 - our own workflow text, not input


def finalize(published: str, attempt: int, tsv: str, event: str = "schedule", keep_req: bool = False) -> dict:
    """Which Finalize steps run, and whether the job ends red."""
    ctx = {"steps.wait.outputs.published": published, "needs.resolve.outputs.exists": "false",
           "github.event_name": event, "inputs.publish": event != "schedule", "inputs.keep_artifacts": keep_req,
           # The base-tag fallback alert's condition; it only reports, so it never changes what Finalize deletes.
           "needs.resolve.result": "success", "inputs.tag": "" if event == "schedule" else "latest",
           "needs.resolve.outputs.base_fallback": ""}
    ran = []
    for s in STEPS:
        cond = s.get("if")
        if cond is None or evaluate(cond, ctx):
            ran.append(s.get("name", s.get("uses")))
            if s.get("id") == "keep":
                rc, outs, log = run_keep(tsv, attempt, "true" if keep_req else "false")
                assert rc == 0, log
                ctx.update({f"steps.keep.outputs.{k}": v for k, v in outs.items()})
        ctx.setdefault("steps.keep.outputs.keep", "")
        ctx.setdefault("steps.keep.outputs.rerun", "")
    red = any(s.get("name", "").startswith("Fail so a rerun") for s in STEPS if s.get("name") in ran)
    return {"ran": ran, "red": red, "deleted_unpublished": any("published nothing" in r for r in ran),
            "deleted_published": any("already published" in r for r in ran)}


print("finalize conditions")
f = finalize("false", 1, jobs_tsv({}), event="workflow_dispatch")
check("publish:false green dispatch still deletes its artifacts", f["deleted_unpublished"] and not f["red"], f)
f = finalize("false", 1, jobs_tsv({}), event="workflow_dispatch", keep_req=True)
check("keep_artifacts dispatch keeps and stays green", not f["deleted_unpublished"] and not f["red"], f)
f = finalize("unknown", 1, jobs_tsv({}))
check("an unknown publish state deletes nothing and stays green",
      not f["deleted_unpublished"] and not f["deleted_published"] and not f["red"], f)
f = finalize("false", 1, jobs_tsv({"Resolve tag": "failure"}))
check("a resolve failure deletes", f["deleted_unpublished"] and not f["red"], f)

# ── 4. Failed leg, then rerun of the failed jobs, then publish ──
print("failed leg -> rerun --failed -> publish")


def needs_graph(doc: dict) -> dict[str, set[str]]:
    g = {}
    for jid, j in doc["jobs"].items():
        n = j.get("needs", [])
        g[jid] = set([n] if isinstance(n, str) else n)
    return g


def rerun_failed(graph: dict[str, set[str]], failed: set[str]) -> set[str]:
    """GitHub's rerun-failed-jobs: the failed or cancelled jobs plus everything that needs them."""
    out = set(failed)
    while True:
        more = {j for j, n in graph.items() if n & out} - out
        if not more:
            return out
        out |= more


G = needs_graph(PREBUILT)
check("Finalize needs only resolve (the reason it must fail to be rerun)", G["reclaim"] == {"resolve"}, G["reclaim"])
check("a green Finalize would be left out of the rerun",
      "reclaim" not in rerun_failed(G, {"build-cpu", "assemble"}), rerun_failed(G, {"build-cpu", "assemble"}))

attempt1 = jobs_tsv({"CPU / windows/x64": "failure", ASSEMBLE: "failure"})
artifacts = {"app-source-T"} | {f"app-T-{n}" for n in LEGS if n != "CPU / windows/x64"}
f1 = finalize("false", 1, attempt1)
check("attempt 1: Finalize keeps every artifact", not f1["deleted_unpublished"] and not f1["deleted_published"], f1)
check("attempt 1: Finalize ends red so the rerun includes it", f1["red"], f1)

failed = {"build-cpu", "assemble"} | ({"reclaim"} if f1["red"] else set())
again = rerun_failed(G, failed)
check("the rerun covers the failed leg, assemble and Finalize, not resolve",
      {"build-cpu", "assemble", "reclaim"} <= again and "resolve" not in again, again)
check("the rerun's legs find the source tarball resolve uploaded on attempt 1", "app-source-T" in artifacts)

# The reran leg uploads its bundle under the same name; a leg that uploaded and then
# failed (runner lost during Save ccache) would 409 without overwrite.
missing = []
for wf in sorted(WF.glob("unsloth-prebuilt-*.yml")):
    for jid, j in (load(wf.name).get("jobs") or {}).items():
        for s in j.get("steps") or []:
            w = s.get("with") or {}
            if "upload-artifact" in str(s.get("uses", "")) and str(w.get("name", "")).startswith("app-") \
                    and w.get("overwrite") is not True:
                missing.append(f"{wf.name}:{jid}:{s.get('name')}")
check("every build-leg bundle upload overwrites on a rerun", not missing, "\n".join(missing))

f2 = finalize("true", 2, jobs_tsv({}))
check("attempt 2 published: Finalize reclaims what the release carries", f2["deleted_published"], f2)
check("attempt 2 published: nothing deleted unpublished, Finalize green",
      not f2["deleted_unpublished"] and not f2["red"], f2)
check("attempt 2 published: the health report runs again", any("Summarise" in r for r in f2["ran"]), f2)

f2b = finalize("false", 2, jobs_tsv({"CPU / windows/x64": "failure", ASSEMBLE: "failure"}))
f3 = finalize("false", 3, jobs_tsv({"CPU / windows/x64": "failure", ASSEMBLE: "failure"}))
check("still failing on attempt 2: kept again", not f2b["deleted_unpublished"] and f2b["red"], f2b)
check(f"still failing on attempt {MAX}: deleted, green", f3["deleted_unpublished"] and not f3["red"], f3)

# ── 5. The retry workflow must not read Finalize's deliberate failure as a cause ──
print("retry workflow")
RETRY = load("unsloth-prebuilt-retry.yml")["jobs"]["retry"]["steps"]
body = next(s["run"] for s in RETRY if "run" in s and "FAILED=" in s["run"])
awk = next(line.strip() for line in body.splitlines() if line.strip().startswith('FAILED="$(awk'))


def failed_of(rows: list[tuple[str, str, str]]) -> str:
    js = "".join(f"{i}\t{c}\t{n}\n" for i, (n, c, _) in enumerate(rows, 1))
    p = subprocess.run(["bash", "-c", f'JOBS="$(cat)"; {awk}; printf "%s" "$FAILED"'], input=js,
                       capture_output=True, text=True)
    return p.stdout


rows = [("CPU / windows/x64", "failure", ""), (ASSEMBLE, "failure", ""), (FINALIZE, "failure", ""),
        ("CUDA / x64/cuda12-portable", "success", "")]
got = failed_of(rows)
check("Finalize is filtered out of the retry's failed set", FINALIZE not in got and "CPU / windows/x64" in got, got)
got = failed_of([("macOS / arm64", "cancelled", ""), (FINALIZE, "failure", ""), ("CPU / windows/x64", "success", "")])
check("a cancelled run whose only failure is Finalize still has no failed job (08-05 retry path)", got == "", got)

lint = (WF / "unsloth-pr-set-lint.yml").read_text()
check("the lint workflow runs this test", "scripts/unsloth/test_finalize_artifacts.py" in lint)

print(f"{len(FAILS)} failure(s)" + (": " + ", ".join(FAILS) if FAILS else ""))
sys.exit(1 if FAILS else 0)
