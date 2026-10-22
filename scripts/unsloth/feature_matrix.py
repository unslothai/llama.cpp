#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Run the test that proves each shipped feature works, against a built tree.

pin_contract.py proves the merge did not lose a pin's code. That is a different
question from whether the feature works, and neither one implies the other: the
Inkling banded-attention kernel merged against upstream's sparse attention is
thirteen hunks of CUDA template parameter threading, where a mistake gives
wrong attention output and every static check passes.

Keyed by FEATURE, not by pin. When upstream absorbs a feature and the pin is
deleted, removing the check with it would put the blind spot back in a
different place -- the feature is still in the release, it just arrives through
the base tag now. So the manifest binds a feature to its current pin and
survives that pin going away.

A PASS HAS TO BE POSITIVE EVIDENCE. Both harnesses exit 0 having done nothing:

    test-llama-archs -a diffusion-gemma   # excluded -> prints SKIP, exits 0
    test-backend-ops test -o TYPO         # matches nothing, exits 0

so every probe rejects skip markers and requires a non-zero count of cases it
actually ran. Without that this file is decoration.

CPU only under CUDA_VISIBLE_DEVICES="" is what CI can do, since no runner in
the prebuild pipeline has a GPU. Run it with the variable unset on a GPU box to
get the comparison that matters for kernels.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Output that means "this did not run" from a process that exited 0.
SKIP_RE = re.compile(r"\bSKIP\b|not supported|unsupported|no tests|0 tests", re.I)


class Unproven(Exception):
    """The probe exited 0 without demonstrating anything."""


class NeedsGPU(Exception):
    """Nothing is wrong; this check cannot be answered on this machine.

    test-backend-ops compares a backend against the CPU reference, so with no
    accelerator present it has nothing to compare and prints "Skipping CPU
    backend". Reporting that as a pass would be a lie and reporting it as a
    failure would block every nightly, since no runner in the prebuild pipeline
    has a GPU. It is counted and named instead.
    """


def bins(build_dir: Path) -> Path:
    for c in (build_dir / "bin", build_dir):
        if (c / "test-backend-ops").exists() or (c / "test-llama-archs").exists():
            return c
    raise SystemExit(f"no test binaries under {build_dir}")


def run(cmd: list[str], cwd: Path, gpu: bool) -> tuple[int, str]:
    env = None
    if not gpu:
        import os
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def probe_arch(check: dict, b: Path, gpu: bool) -> str:
    """A synthetic model of this architecture decodes, and matches CPU."""
    arch = check["arch"]
    rc, out = run([str(b / "test-llama-archs"), "-a", arch, "-s", "1234"], b, gpu)
    if rc != 0:
        raise Unproven(f"test-llama-archs -a {arch} exited {rc}")
    # The arch's own rows, not the header and not another arch's.
    rows = [ln for ln in out.splitlines() if ln.strip().startswith("|") and f"|{arch:>16}|" in ln
            or (ln.strip().startswith("|") and ln.split("|")[1].strip() == arch)]
    if not rows:
        raise Unproven(f"test-llama-archs printed no row for {arch}; it is not in the harness")
    ok = [r for r in rows if "OK" in r]
    if not ok:
        raise Unproven(f"every {arch} row was skipped, so nothing was decoded: {rows[0].strip()}")
    return f"{len(ok)}/{len(rows)} device rows decoded and matched CPU"


def probe_backend_op(check: dict, b: Path, gpu: bool) -> str:
    """The op exists in the backend and matches the CPU reference."""
    if not gpu:
        raise NeedsGPU("test-backend-ops compares against CPU, so with no "
                       "accelerator it skips every backend and proves nothing")
    cmd = [str(b / "test-backend-ops"), "test", "-o", check["op"]]
    if check.get("params"):
        cmd += ["-p", check["params"]]
    rc, out = run(cmd, b, gpu)
    if rc != 0:
        raise Unproven(f"{' '.join(cmd[1:])} exited {rc}")
    m = re.search(r"(\d+)/(\d+) tests passed", out)
    if not m:
        raise Unproven(f"{check['op']} produced no test count; the filter matched nothing")
    passed, total = int(m.group(1)), int(m.group(2))
    if total == 0:
        raise Unproven(f"{check['op']} matched 0 cases; the op name is stale")
    if passed != total:
        raise Unproven(f"{check['op']}: {passed}/{total} passed")
    return f"{passed}/{total} cases matched the CPU reference"


def probe_mtmd(check: dict, b: Path, gpu: bool) -> str:
    """The projector registry is intact, including this projector's entry."""
    rc, out = run([str(b / "test-mtmd-impl"), "test_projector_registry"], b, gpu)
    if rc != 0:
        raise Unproven(f"test-mtmd-impl exited {rc}")
    m = re.search(r"assertions\s*:\s*(\d+)", out)
    if not m or int(m.group(1)) == 0:
        raise Unproven("test_projector_registry ran no assertions; the filter matched nothing")
    # The registry test walks the whole enum, so it proves the table is sound.
    # That the specific projector is IN the enum is pin_contract.py's job.
    return f"projector registry intact over {m.group(1)} assertions"


PROBES = {"arch": probe_arch, "backend-op": probe_backend_op, "mtmd": probe_mtmd}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--feature-checks", required=True)
    ap.add_argument("--only", help="one feature id")
    ap.add_argument("--gpu", action="store_true",
                    help="let the probes see the GPU; CI has none, so the default "
                         "hides it and the comparison is CPU-only")
    ap.add_argument("--report")
    args = ap.parse_args()

    b = bins(Path(args.build_dir).resolve())
    doc = json.loads(Path(args.feature_checks).read_text())
    report: dict = {"gpu": args.gpu, "features": [], "ok": False, "deferred": 0}
    failed = 0
    deferred = 0

    for name, feat in sorted(doc["features"].items()):
        if args.only and name != args.only:
            continue
        entry = {"feature": name, "owner": feat.get("owner"),
                 "results": [], "problems": [], "deferred": []}
        for check in feat["checks"]:
            kind = check["kind"]
            label = f"{kind}:{check.get('arch') or check.get('op') or check.get('projector')}"
            try:
                if kind not in PROBES:
                    raise Unproven(f"unknown check kind {kind!r}")
                entry["results"].append({"check": label, "evidence": PROBES[kind](check, b, args.gpu)})
            except NeedsGPU as e:
                entry["deferred"].append(f"{label}: {e}")
                deferred += 1
            except Unproven as e:
                entry["problems"].append(f"{label}: {e}")
            except OSError as e:
                entry["problems"].append(f"{label}: cannot run: {e}")
        report["features"].append(entry)
        if entry["problems"]:
            failed += 1
            print(f"FAIL {name}", file=sys.stderr)
            for p in entry["problems"]:
                print(f"     {p}", file=sys.stderr)
        elif entry["results"]:
            print(f"ok   {name}: " + "; ".join(r["evidence"] for r in entry["results"])
                  + (f"  [{len(entry['deferred'])} needs a GPU]" if entry["deferred"] else ""))
        else:
            # Nothing was shown either way. Not a failure here, but it must not
            # read as one of the ok lines.
            print(f"--   {name}: nothing provable without a GPU "
                  f"({len(entry['deferred'])} check(s) deferred)")

    for pin, why in sorted(doc.get("unchecked", {}).items()):
        print(f"note {pin} has no runtime check: {why}")

    report["ok"] = failed == 0
    report["deferred"] = deferred
    if args.report:
        Path(args.report).write_text(json.dumps(report, indent=2))
    if failed:
        print(f"\n{failed} feature(s) could not be shown to work", file=sys.stderr)
        return 1
    # Say what was NOT proven in the same breath as what was. A run that only
    # ever prints a success line teaches the reader that green means covered.
    tail = f", {deferred} check(s) need a GPU and were not run" if deferred else ""
    print(f"\nall {len(report['features'])} features demonstrated"
          + (" on GPU" if args.gpu else " on CPU") + tail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
