#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for pin_merge.py. Run: python3 scripts/unsloth/test_pin_merge.py

The first case is the real 08-27 collision: master had moved qwen4exp to
950f135b28 while the GLM-5-Next branch was replacing pin 118 with 125. It was
resolved by hand at the time; this asserts the script reproduces that answer.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "pin_merge.py"
FAILS = []

U = "https://github.com/unslothai/llama.cpp/pull"
BASE_PINS = [
    f"{U}/107/commits/74acc40c37ae2eb36031981feda392b793944f72",
    f"{U}/108/commits/27278df7000ade4a638d044202dbe82975421df6",
    f"{U}/70/commits/edfd4c1a3b7a653303a85257ddac2a1f3ce39a2f",
    f"{U}/91/commits/c86ed269986f2dced6325c5c58bda966a2e2ead1",
    f"{U}/95/commits/3db8cb5b2e9bf291057b9f19960e8601a162da81",
    f"{U}/114/commits/c4ddc4805dbc12727897b354237bfd9225212b06",
    f"{U}/118/commits/3766b41229c20249fd4d83d7ba297499d50e9b80",
]


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def run(base, ours, theirs):
    d = Path(tempfile.mkdtemp(prefix="pm_"))
    paths = []
    for name, pins in (("base", base), ("ours", ours), ("theirs", theirs)):
        p = d / f"{name}.json"
        p.write_text(json.dumps({"prs": pins}, indent=2))
        paths.append(str(p))
    r = subprocess.run([sys.executable, str(SCRIPT), *paths, "--stdout"],
                       capture_output=True, text=True)
    pins = json.loads(r.stdout)["prs"] if r.returncode == 0 else None
    return r.returncode, pins, r.stderr.strip()


def sub(pins, i, sha):
    out = list(pins)
    out[i] = out[i].rsplit("/", 1)[0] + "/" + sha
    return out


def replace(pins, i, url):
    out = list(pins)
    out[i] = url
    return out


# 1. the real 08-27 collision
ours = replace(BASE_PINS, 6, f"{U}/125/commits/f48b99e1fd04628da2a3d4ea5acc335d9ea67f7a")
theirs = sub(BASE_PINS, 5, "950f135b28789057721a65d76de98fbbcd2f7dd6")
rc, pins, err = run(BASE_PINS, ours, theirs)
check("real 08-27 collision resolves", rc == 0, err)
if pins:
    check("takes theirs for the pin only theirs moved", pins[5] == theirs[5], pins[5])
    check("takes ours for the pin only ours moved", pins[6] == ours[6], pins[6])
    check("leaves untouched pins alone", pins[:5] == BASE_PINS[:5])
    check("preserves pin order", [p.split("/pull/")[1].split("/")[0] for p in pins]
          == ["107", "108", "70", "91", "95", "114", "125"])

# 2. both sides repin the same entry differently
rc, _, err = run(BASE_PINS, sub(BASE_PINS, 5, "a" * 40), sub(BASE_PINS, 5, "b" * 40))
check("refuses a genuine two-sided repin", rc == 1, err)
check("says which pin was ambiguous", "pin 5" in err, err)

# 3. a pin added on one side
rc, _, err = run(BASE_PINS, BASE_PINS + [f"{U}/999/commits/{'c' * 40}"], BASE_PINS)
check("refuses an added pin", rc == 1, err)

# 4. a pin removed on one side
rc, _, err = run(BASE_PINS, BASE_PINS[:-1], BASE_PINS)
check("refuses a removed pin", rc == 1, err)

# 5. both sides make the identical repin
same = sub(BASE_PINS, 5, "d" * 40)
rc, pins, err = run(BASE_PINS, same, same)
check("accepts an identical repin on both sides", rc == 0 and pins == same, err)

# 6. neither side changed anything
rc, pins, err = run(BASE_PINS, BASE_PINS, BASE_PINS)
check("no-op merge is a no-op", rc == 0 and pins == BASE_PINS, err)

# 7. object-form entries keep their other fields
objs = [{"url": u, "required": False} for u in BASE_PINS]
o = json.loads(json.dumps(objs)); o[5]["url"] = sub(BASE_PINS, 5, "e" * 40)[5]
d = Path(tempfile.mkdtemp(prefix="pm_"))
for name, pins_ in (("base", objs), ("ours", o), ("theirs", objs)):
    (d / f"{name}.json").write_text(json.dumps({"prs": pins_}, indent=2))
r = subprocess.run([sys.executable, str(SCRIPT), str(d / "base.json"), str(d / "ours.json"),
                    str(d / "theirs.json"), "--stdout"], capture_output=True, text=True)
ok = r.returncode == 0 and all(e.get("required") is False for e in json.loads(r.stdout)["prs"])
check("object-form entries keep their other fields", ok, r.stdout[:200] + r.stderr)


def run_objs(base, ours, theirs):
    d = Path(tempfile.mkdtemp(prefix="pm_"))
    paths = []
    for name, entries in (("base", base), ("ours", ours), ("theirs", theirs)):
        p = d / f"{name}.json"
        p.write_text(json.dumps({"prs": entries}, indent=2))
        paths.append(str(p))
    r = subprocess.run([sys.executable, str(SCRIPT), *paths, "--stdout"],
                       capture_output=True, text=True)
    return r.returncode, (json.loads(r.stdout)["prs"] if r.returncode == 0 else None), r.stderr.strip()


# 8. theirs flips `required` on one entry while ours repins a DIFFERENT one.
# Comparing only urls makes theirs' flip invisible, and the result is rebuilt
# from ours, so the flip is silently dropped by a merge that reports success.
objs = [{"url": u, "required": True} for u in BASE_PINS]
o = json.loads(json.dumps(objs)); o[5]["url"] = sub(BASE_PINS, 5, "e" * 40)[5]
t = json.loads(json.dumps(objs)); t[1]["required"] = False
rc, pins, err = run_objs(objs, o, t)
check("keeps theirs' non-url field change on an entry ours did not touch",
      rc == 0 and pins is not None and pins[1]["required"] is False, err or json.dumps(pins))
check("keeps ours' repin alongside theirs' field change",
      rc == 0 and pins is not None and pins[5]["url"] == o[5]["url"], err)

# 9. both sides touch the SAME entry, but different fields: still mergeable.
o = json.loads(json.dumps(objs)); o[3]["url"] = sub(BASE_PINS, 3, "f" * 40)[3]
t = json.loads(json.dumps(objs)); t[3]["required"] = False
rc, pins, err = run_objs(objs, o, t)
check("merges a repin and a field change on the same entry",
      rc == 0 and pins is not None
      and pins[3]["url"] == o[3]["url"] and pins[3]["required"] is False, err)

# 10. both sides set the same field to different values: still refused.
o = json.loads(json.dumps(objs)); o[2]["required"] = False
t = json.loads(json.dumps(objs)); t[2]["required"] = "maybe"
rc, _, err = run_objs(objs, o, t)
check("refuses a two-sided change to the same field", rc == 1, err)

def run_docs(base, ours, theirs):
    """Like run(), but the caller supplies the whole document, not just pins."""
    d = Path(tempfile.mkdtemp(prefix="pm_"))
    paths = []
    for name, doc in (("base", base), ("ours", ours), ("theirs", theirs)):
        p = d / f"{name}.json"
        p.write_text(json.dumps(doc, indent=2))
        paths.append(str(p))
    r = subprocess.run([sys.executable, str(SCRIPT), *paths, "--stdout"],
                       capture_output=True, text=True)
    return r.returncode, (json.loads(r.stdout) if r.returncode == 0 else None), r.stderr.strip()


# 12. theirs edits a TOP-LEVEL field while ours repins an entry. Rebuilding the
# document from ours drops theirs' edit and still exits 0, and because a merge
# driver replaces git's text merge outright, nothing else ever sees the loss.
doc = {"_doc": ["old doc line"], "prs": list(BASE_PINS)}
o = json.loads(json.dumps(doc)); o["prs"] = sub(BASE_PINS, 5, "a" * 40)
t = json.loads(json.dumps(doc)); t["_doc"] = ["old doc line", "prune closed pins"]
rc, out, err = run_docs(doc, o, t)
check("keeps theirs' top-level field change alongside ours' repin",
      rc == 0 and out is not None and out["_doc"] == t["_doc"], err or json.dumps(out))
check("still takes ours' repin when theirs edited the document",
      rc == 0 and out is not None and out["prs"] == o["prs"], err)
check("keeps .prs in its original key position",
      rc == 0 and out is not None and list(out) == ["_doc", "prs"], json.dumps(list(out or {})))

# 13. theirs ADDS a top-level field ours has never seen: it has to survive.
t = json.loads(json.dumps(doc)); t["base_tag"] = "b10639"
rc, out, err = run_docs(doc, o, t)
check("keeps a top-level field only theirs added",
      rc == 0 and out is not None and out.get("base_tag") == "b10639", err or json.dumps(out))

# 14. both sides set the same top-level field differently: refuse, never guess.
o2 = json.loads(json.dumps(doc)); o2["_doc"] = ["ours' rewrite"]
t2 = json.loads(json.dumps(doc)); t2["_doc"] = ["theirs' rewrite"]
rc, _, err = run_docs(doc, o2, t2)
check("refuses a two-sided change to the same top-level field", rc == 1, err)
check("names the clashing top-level field", "_doc" in err, err)

# 15. a top-level field theirs deleted stays deleted.
t3 = json.loads(json.dumps(doc)); del t3["_doc"]
rc, out, err = run_docs(doc, o, t3)
check("honours a top-level field theirs deleted",
      rc == 0 and out is not None and "_doc" not in out, err or json.dumps(out))

# 16. --help must not crash: argparse %-formats help strings, and the driver
# placeholders %O/%A/%B are literal percents that have to be escaped.
r = subprocess.run([sys.executable, str(SCRIPT), "--help"], capture_output=True, text=True)
check("--help does not crash on the %O/%A/%B placeholders",
      r.returncode == 0 and "%O" in r.stdout, (r.stderr or r.stdout)[-200:])

# 17. one side REORDERS the pins while the other changes a field. Merging by
# position then combines fields belonging to different PRs: base
# [A(required), B(required)] with ours making A optional and theirs swapping
# the two produces B(required=false), so the release skips the wrong PR, and
# the driver exits 0 while doing it. A reorder must be refused instead.
two = [{"url": BASE_PINS[0], "required": True}, {"url": BASE_PINS[1], "required": True}]
o = json.loads(json.dumps(two)); o[0]["required"] = False
t = [two[1], two[0]]
rc, pins, err = run_objs(two, o, t)
check("refuses a reorder that would splice fields across PRs", rc == 1,
      json.dumps(pins) if pins else err)
check("names the reordered position", "reorder" in err, err)
check("never emits a pin carrying another PR's field",
      pins is None or pins[0]["required"] is not False, json.dumps(pins))

# 18. a reorder that also repins the moved entry still has to be refused: the
# url no longer matches, so only the PR number identifies the entry.
t = [dict(two[1]), dict(two[0])]
t[0]["url"] = sub(BASE_PINS, 1, "9" * 40)[1]
rc, _, err = run_objs(two, o, t)
check("refuses a reorder combined with a repin", rc == 1, err)

# 19. a reorder on OUR side is refused too, not just on theirs.
o2 = [two[1], two[0]]
t2 = json.loads(json.dumps(two)); t2[0]["required"] = False
rc, _, err = run_objs(two, o2, t2)
check("refuses a reorder on ours", rc == 1, err)

# 20. swapping a pin for a DIFFERENT PR at the same position is not a reorder
# and must keep merging, which is case 1's real 08-27 resolution.
rc, pins, err = run(BASE_PINS,
                    replace(BASE_PINS, 6, f"{U}/125/commits/{'a' * 40}"),
                    sub(BASE_PINS, 5, "b" * 40))
check("a same-position swap to a new PR is not a reorder", rc == 0, err)

# 21. a plain repin is not a reorder either, on either side.
rc, pins, err = run(BASE_PINS, sub(BASE_PINS, 0, "1" * 40), sub(BASE_PINS, 3, "2" * 40))
check("two repins at different positions still merge", rc == 0, err)

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all pin_merge tests passed")
sys.exit(1 if FAILS else 0)
