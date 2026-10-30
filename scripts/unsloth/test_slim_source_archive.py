#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Tests for slim_source_archive.sh. Run: python3 scripts/unsloth/test_slim_source_archive.py"""
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "slim_source_archive.sh"
FAILS = []
NAME = "llama.cpp-b1-mix-abc"


def check(name, cond, extra=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  :: {extra}" if extra and not cond else ""))
    if not cond:
        FAILS.append(name)


def archive(d, callers=('_set_vocab_builtin("gpt-neox", n)', '_set_vocab_builtin("llama-spm", n)')):
    src = Path(tempfile.mkdtemp(prefix="sa_", dir=d)) / NAME
    files = [
        "CMakeLists.txt", "LICENSE", "gguf-py/gguf/__init__.py", "src/llama.cpp", "ggml/tests/keep.c",
        "docs/ops/a.md", "media/a.png", "tests/a.cpp", "benches/a.txt", "pocs/a.cpp",
        "models/templates/a.jinja", "models/ggml-vocab-gpt-neox.gguf", "models/ggml-vocab-llama-spm.gguf",
        "models/ggml-vocab-qwen2.gguf", "models/ggml-vocab-qwen2.gguf.inp", "models/nested/ggml-vocab-x.gguf",
    ]
    for f in files:
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text(f)
    (src / "conversion").mkdir()
    (src / "conversion/base.py").write_text(
        'def _set_vocab_builtin(self, model_name: Literal["gpt-neox", "llama-spm"], vocab_size: int):\n    pass\n'
    )
    (src / "conversion/mamba.py").write_text("\n".join(callers))
    (src / "convert_hf_to_gguf.py").write_text("pass\n")
    out = Path(d) / f"{src.parent.name}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        t.add(src, arcname=NAME)
    return out


def slim(path):
    return subprocess.run(["bash", str(SCRIPT), str(path)], capture_output=True, text=True)


def names(path):
    with tarfile.open(path) as t:
        return {m.name.removeprefix(f"{NAME}/") for m in t.getmembers() if m.isfile()}


with tempfile.TemporaryDirectory() as d:
    a = archive(d)
    r = slim(a)
    check("slims cleanly", r.returncode == 0, r.stderr)
    got = names(a)
    for f in ("models/ggml-vocab-gpt-neox.gguf", "models/ggml-vocab-llama-spm.gguf",
              "models/templates/a.jinja", "models/nested/ggml-vocab-x.gguf", "CMakeLists.txt",
              "convert_hf_to_gguf.py", "gguf-py/gguf/__init__.py", "conversion/base.py",
              "LICENSE", "src/llama.cpp", "ggml/tests/keep.c", "tests/a.cpp", "pocs/a.cpp"):
        check(f"keeps {f}", f in got)
    for f in ("docs/ops/a.md", "media/a.png", "benches/a.txt",
              "models/ggml-vocab-qwen2.gguf", "models/ggml-vocab-qwen2.gguf.inp"):
        check(f"drops {f}", f not in got)

    b = archive(d)
    c = b.with_name("copy.tar.gz")
    c.write_bytes(b.read_bytes())
    slim(b), slim(c)
    check("a re-run reproduces the same bytes", b.read_bytes() == c.read_bytes())

    new = archive(d, callers=('_set_vocab_builtin("qwen2", n)',))
    check("a vocab conversion starts using survives", slim(new).returncode == 0
          and "models/ggml-vocab-qwen2.gguf" in names(new))

    bad = archive(d, callers=('_set_vocab_builtin("gone", n)',))
    before = bad.read_bytes()
    r = slim(bad)
    check("refuses when conversion needs a missing vocab", r.returncode != 0 and "gone" in r.stderr, r.stderr)
    check("a refused archive is left untouched", bad.read_bytes() == before)

    none = archive(d, callers=())
    r = slim(none)
    check("refuses when no caller is found", r.returncode != 0 and "no _set_vocab_builtin" in r.stderr, r.stderr)

sys.exit(1 if FAILS else 0)
