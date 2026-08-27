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

print()
print(f"{len(FAILS)} failure(s)" if FAILS else "all pin_merge tests passed")
sys.exit(1 if FAILS else 0)
