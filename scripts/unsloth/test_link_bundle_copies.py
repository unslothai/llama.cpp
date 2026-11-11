#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for link_bundle_copies.sh. Run: python3 scripts/unsloth/test_link_bundle_copies.py"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "link_bundle_copies.sh"
FAILS = []


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def link(path):
    return subprocess.run(["bash", str(SCRIPT), str(path)], capture_output=True, text=True)


with tempfile.TemporaryDirectory() as d:
    b = Path(d)
    for name in ("libfoo.so", "libfoo.so.1", "libfoo.so.1.2.3"):
        (b / name).write_bytes(b"foo")
    (b / "libbar.so.1").write_bytes(b"new")
    (b / "libbar.so").write_bytes(b"old")
    (b / "libbaz.so").write_bytes(b"foo")
    (b / "llama-server").write_bytes(b"foo")
    (b / "libreal.so.1").write_bytes(b"real")
    os.symlink("libreal.so.1", b / "libreal.so")
    (b / "sub").mkdir()
    (b / "sub/libfoo.so").write_bytes(b"foo")

    r = link(b)
    check("runs cleanly", r.returncode == 0, r.stderr)
    check("keeps the longest name as the real file", not (b / "libfoo.so.1.2.3").is_symlink())
    check("links the short names", os.readlink(b / "libfoo.so") == "libfoo.so.1.2.3"
          and os.readlink(b / "libfoo.so.1") == "libfoo.so.1.2.3")
    check("links resolve to the same bytes", (b / "libfoo.so").read_bytes() == b"foo")
    check("different bytes stay separate", (b / "libbar.so").read_bytes() == b"old"
          and not (b / "libbar.so").is_symlink())
    check("another library is never linked", not (b / "libbaz.so").is_symlink())
    check("non-libraries are untouched", not (b / "llama-server").is_symlink())
    check("existing links are untouched", os.readlink(b / "libreal.so") == "libreal.so.1")
    check("subdirectories are untouched", not (b / "sub/libfoo.so").is_symlink())

    r = link(b)
    check("a second run changes nothing", r.returncode == 0
          and os.readlink(b / "libfoo.so") == "libfoo.so.1.2.3", r.stderr)

check("refuses a missing bundle", link("/nonexistent/bundle").returncode != 0)

sys.exit(1 if FAILS else 0)
