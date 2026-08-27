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
belongs is exactly the kind of guess that would change composition order. A
side that REORDERS the list is refused for the same reason, and for a sharper
one: position i would no longer name the same pin on both sides, so merging it
field-wise would splice one PR's `required` onto another PR's url.

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
import re
import sys
from pathlib import Path


class Ambiguous(Exception):
    """A pin set difference this script is not allowed to decide."""


MISSING = object()

# The pin url shape repin.py already enforces. The owner matters: ggml-org#125
# and unslothai#125 are different PRs.
PIN_ID = re.compile(r"^https://github\.com/([^/]+)/llama\.cpp/pull/(\d+)/commits/")


def pins(doc: dict) -> list[str]:
    """The pin URLs, in order. Entries are a bare string or {url, required}."""
    return [p if isinstance(p, str) else p["url"] for p in doc["prs"]]


def fields(entry: str | dict) -> dict:
    """An entry in dict form. A bare url string is just {"url": url}."""
    return {"url": entry} if isinstance(entry, str) else dict(entry)


def ident(entry: str | dict) -> str:
    """What a pin IS, independent of which commit it currently points at.

    A repin changes only the sha, so (owner, PR number) is the stable identity.
    Anything that does not look like a pin url is its own identity, which is the
    conservative reading: an unrecognised url can only ever cause a refusal.
    """
    url = fields(entry).get("url")
    m = PIN_ID.match(url) if isinstance(url, str) else None
    return f"{m.group(1)}#{m.group(2)}" if m else repr(url)


def refuse_reorder(b: list, o: list, t: list) -> None:
    """Refuse if either side moved an entry that base holds somewhere else.

    Merging by position assumes position i means the same pin on all three
    sides. A reorder breaks that assumption silently: base [A, B] with ours
    making A optional and theirs reordering to [B, A] merges position 0 as
    "url moved to B, required moved to false" and produces B(required=false),
    so the release skips the wrong PR and the driver still exits 0.

    Realigning by identity instead is not safe. A duplicated entry, or a
    reorder combined with a repin, leaves more than one alignment consistent
    with the diff, and choosing one is a guess about composition order, which
    is load-bearing here. Refusing costs the hand resolution that was the
    status quo; guessing costs a wrong build nothing downstream can see.
    """
    bid = [ident(e) for e in b]
    # Identity is (owner, PR number), so two entries pinning two commits of the
    # SAME PR share one. Swapping those two then satisfies nothing below and
    # goes undetected, and a field the other side changed on the first lands on
    # the second. pr-set.json has never held a duplicate and the lint does not
    # forbid one, so refuse rather than rely on that holding.
    dupes = {k for k in bid if bid.count(k) > 1}
    if dupes:
        raise Ambiguous(
            f"pin set names {', '.join(sorted(dupes))} more than once: two "
            "entries of one PR are indistinguishable here, so a swap between "
            "them cannot be told from no change at all")
    for side, entries in (("ours", o), ("theirs", t)):
        for i, e in enumerate(entries):
            k = ident(e)
            if k != bid[i] and k in bid:
                raise Ambiguous(
                    f"pin {i} on {side} is {k}, which base holds at position "
                    f"{bid.index(k)}: the list was reordered, and merging "
                    "reordered entries by position would take fields from "
                    "different PRs")


def merge_keys(what: str, bd: dict, od: dict, td: dict) -> dict:
    """Three-way merge a mapping key by key, refusing only a real clash.

    A key missing on a side is MISSING rather than absent, so "theirs deleted
    it, ours left it alone" is a deletion both sides agree on, not a no-op.
    Ours' key order is kept, then keys only theirs or only base has.
    """
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
            raise Ambiguous(f"{what} field {k!r} changed differently on both "
                            f"sides:\n  ours:   {ov}\n  theirs: {tv}")
        if v is not MISSING:
            out[k] = v
    return out


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
    # Both sides touched it. Merging field-wise is only meaningful while all
    # three sides name the SAME PR. A repin moves the sha and keeps the
    # identity, which is the case this is for. REPLACING the pin with another
    # PR while the other side edits a field is not: base PR100(required=true),
    # ours PR200, theirs PR100 required=false merges url from ours and
    # required from theirs and yields PR200(required=false), so the release
    # skips a PR nobody made optional and the driver still exits 0.
    #
    # Refused rather than resolved even when both sides agree on the
    # replacement, because base then describes a different PR and every field
    # comparison below is against settings that were never PR200's.
    named = {ident(b), ident(o), ident(t)}
    if len(named) != 1:
        raise Ambiguous(
            f"pin {i} names different PRs across the sides (base {ident(b)}, "
            f"ours {ident(o)}, theirs {ident(t)}) and both sides edited it: "
            "merging their fields would attach one PR's settings to another")
    out = merge_keys(f"pin {i}", fields(b), fields(o), fields(t))
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
    refuse_reorder(b, o, t)
    merged = [merge_entry(i, bx, ox, tx)
              for i, (bx, ox, tx) in enumerate(zip(b, o, t))]
    # Everything outside .prs is three-way merged the same way. Rebuilding the
    # document from ours instead would silently drop a change theirs made to a
    # top-level field -- `_doc`, or any schema field added later -- and this
    # driver REPLACES git's text merge rather than running after it, so nothing
    # downstream would ever notice the loss. A real two-sided clash refuses,
    # which costs a hand resolution and never a wrong document.
    skel = [{k: (None if k == "prs" else v) for k, v in d.items()}
            for d in (base, ours, theirs)]     # .prs is merged positionally
    out = merge_keys("document", *skel)
    out["prs"] = merged                        # keeps ours' key position
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
