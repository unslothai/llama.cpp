#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Three-way merge pr-set.json one pin at a time, and refuse anything ambiguous.

Two repins in flight always collide. Both edit adjacent lines of the same JSON
list, so git merges them as text and reports a conflict over lines that have
nothing to do with each other. On 08-27 that happened twice in one hour, while
landing the qwen4exp, Inkling and GLM-5-Next repins: each merge invalidated the
next, and each one was resolved by hand into exactly what a per-element merge
would have produced.

Pin ORDER is load-bearing. resolve merges pins sequentially, so a later pin sees
the tree the earlier ones produced, and reordering the list silently changes the
composition. This never reorders: it walks the base list positionally and takes
whichever side moved each entry. That also means a pin added or removed on one
side is refused rather than aligned, because guessing where an inserted pin
belongs is exactly the kind of guess that would change composition order.

Usable as a git merge driver:

    git config merge.pinset.name  'pr-set.json pin-wise merge'
    git config merge.pinset.driver 'python3 scripts/unsloth/pin_merge.py %O %A %B'
    echo 'scripts/unsloth/pr-set.json merge=pinset' >> .gitattributes

The driver contract is to write the result over %A (ours) and exit 0, or leave
it alone and exit non-zero to fall back to a normal conflict. That fallback is
the whole safety story: a refusal costs a hand resolution, which is the status
quo, and never a wrong pin set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


class Ambiguous(Exception):
    """A pin set difference this script is not allowed to decide."""


MISSING = object()


def pins(doc: dict) -> list[str]:
    """The pin URLs, in order. Entries are a bare string or {url, required}."""
    return [p if isinstance(p, str) else p["url"] for p in doc["prs"]]


def fields(entry: str | dict) -> dict:
    """An entry in dict form. A bare url string is just {"url": url}."""
    return {"url": entry} if isinstance(entry, str) else dict(entry)


def merge_entry(i: int, b, o, t):
    """Three-way merge one pin ENTRY, not just its url.

    Comparing whole entries matters: an entry carries `required` as well as
    `url`, and comparing only urls makes a `required` flip on one side look
    like "no change", so rebuilding the list from ours drops it silently.
    """
    if o == t:
        return o                       # same on both sides, including untouched
    if o == b:
        return t                       # only theirs touched this entry
    if t == b:
        return o                       # only ours touched this entry
    # Both sides touched it. Merge field-wise, so a repin on one side and a
    # flag change on the other both survive, and refuse only the real clash.
    bd, od, td = fields(b), fields(o), fields(t)
    out: dict = {}
    for k in list(od) + [k for k in td if k not in od] + \
             [k for k in bd if k not in od and k not in td]:
        bv, ov, tv = bd.get(k, MISSING), od.get(k, MISSING), td.get(k, MISSING)
        if ov == tv:
            v = ov
        elif ov == bv:
            v = tv
        elif tv == bv:
            v = ov
        else:
            raise Ambiguous(f"pin {i} field {k!r} changed differently on both "
                            f"sides:\n  ours:   {ov}\n  theirs: {tv}")
        if v is not MISSING:
            out[k] = v
    if "url" not in out:
        raise Ambiguous(f"pin {i} lost its url")
    # Keep the bare-string form when nothing but the url is present, so the
    # file's shape is not rewritten by merging it.
    return out["url"] if list(out) == ["url"] else out


def merge_pins(base: dict, ours: dict, theirs: dict) -> dict:
    b, o, t = base["prs"], ours["prs"], theirs["prs"]
    if not len(b) == len(o) == len(t):
        raise Ambiguous(
            f"pin count differs (base={len(b)} ours={len(o)} theirs={len(t)}); "
            "a pin was added or removed, and placing it is an ordering decision"
        )
    merged = [merge_entry(i, bx, ox, tx)
              for i, (bx, ox, tx) in enumerate(zip(b, o, t))]
    # Keep ours' surrounding document (comments, schema fields) and swap the
    # list, so nothing outside .prs is silently rewritten by this script.
    out = json.loads(json.dumps(ours))
    out["prs"] = merged
    return out


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # argparse %-formats help strings, so a literal percent must be doubled.
    ap.add_argument("base", help="%%O, the merge base")
    ap.add_argument("ours", help="%%A, our version; the result is written here")
    ap.add_argument("theirs", help="%%B, their version")
    ap.add_argument("--stdout", action="store_true",
                    help="print the result instead of writing over `ours`")
    ap.add_argument("--report", metavar="PATH", help="write a JSON summary here")
    a = ap.parse_args()

    report: dict = {"ok": False, "reason": None}
    try:
        merged = merge_pins(load(a.base), load(a.ours), load(a.theirs))
    except (Ambiguous, KeyError, json.JSONDecodeError) as e:
        report["reason"] = str(e)
        if a.report:
            Path(a.report).write_text(json.dumps(report, indent=2))
        print(f"pin_merge: refused: {e}", file=sys.stderr)
        return 1

    text = json.dumps(merged, indent=2) + "\n"
    if a.stdout:
        sys.stdout.write(text)
    else:
        Path(a.ours).write_text(text)
    report["ok"] = True
    report["pins"] = pins(merged)
    if a.report:
        Path(a.report).write_text(json.dumps(report, indent=2))
    # stderr, so --stdout emits nothing but the merged document
    print(f"pin_merge: merged {len(pins(merged))} pins", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
