#!/usr/bin/env python3
# Checks the MoE cache replay of scripts/moe-cache-tune.py on synthetic traces:
# two layers that share their experts (MCT2, a looped layer and its owner) reuse the same slots,
# while the same routing in an MCT1 trace (independent layers) keeps evicting and uploading.
#   python3 tests/test-moe-cache-tune.py

import importlib.util
import os
import struct
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("moe_cache_tune", os.path.join(ROOT, "scripts", "moe-cache-tune.py"))
assert spec is not None and spec.loader is not None
tune = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune)

MIB = 1024 * 1024
N_EXPERT = 4
N_USED = 2
N_TOKENS = 64


def write_trace(path, version, layers):
    """one group of 1 MiB experts, alignment 1; layers: [(layer, owner)]"""
    with open(path, "wb") as f:
        f.write(version)
        f.write(struct.pack("<3I", len(layers), N_USED, 1))
        f.write(struct.pack("<3I", 0, N_EXPERT, len(layers)))
        for il, owner in layers:
            f.write(struct.pack("<I", il))
            if version == b"MCT2":
                f.write(struct.pack("<I", owner))
        n_owners = len({owner for _, owner in layers})
        f.write(struct.pack("<2d", 1.0, 1.0))
        f.write(struct.pack("<3Q", n_owners * N_EXPERT * MIB, n_owners * N_EXPERT * MIB, 1))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<Q", MIB))
        # every token selects experts 0 and 1 in both layers
        for _ in range(N_TOKENS):
            for il, _ in layers:
                f.write(struct.pack("<3i", il, N_USED, N_USED))
                f.write(struct.pack(f"<{N_USED}i", 0, 1))


def main():
    fails = 0
    with tempfile.TemporaryDirectory() as d:
        shared = os.path.join(d, "shared.trace")
        independent = os.path.join(d, "independent.trace")
        write_trace(shared, b"MCT2", [(0, 0), (1, 0)])
        write_trace(independent, b"MCT1", [(0, 0), (1, 1)])

        t_shared = tune.read_trace(shared)
        t_indep = tune.read_trace(independent)
        assert t_shared[2][0].layers == {0: 0, 1: 0}
        assert t_indep[2][0].layers == {0: 0, 1: 1}

        # 3 MiB: 2 slots (plus the extra one), enough for the 2 experts of one owner
        size_mib = 3
        g = t_shared[2][0]
        n_slots = tune.n_slots_of(g, size_mib, 0, N_USED)
        up_shared, _ = tune.replay(t_shared, size_mib, 0.0, N_TOKENS // 2)
        up_indep, _ = tune.replay(t_indep, size_mib, 0.0, N_TOKENS // 2)
        full_shared = tune.full_mib(t_shared, 0.0)
        full_indep = tune.full_mib(t_indep, 0.0)

        checks = [
            ("2 slots at 3 MiB", n_slots == 2),
            ("shared layers: no uploads after warm-up", up_shared == 0.0),
            ("independent layers: uploads every token", abs(up_indep - 4 / 1024) < 1e-12),  # 2 layers x 2 experts x 1 MiB
            ("full cache counts the owner once", full_shared < full_indep),
        ]
        for name, ok in checks:
            print(f"{'ok  ' if ok else 'FAIL'} {name}")
            fails += not ok
        print(f"n_slots {n_slots}, GiB/token uploaded: shared {up_shared}, independent {up_indep}, full MiB: shared {full_shared}, independent {full_indep}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
