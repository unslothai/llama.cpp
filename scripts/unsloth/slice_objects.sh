#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
# Print slice K of N of the objects a configured ninja build compiles for `all`.
#
# Every object, not just ggml-cuda/ggml-hip: after a core-header edit the CPU variants, llama,
# common and the tools miss too (6-7 min of a cold CUDA leg), and slicing them leaves the
# merging job little but hits and the link. Objects come from the compile commands of `all`
# (-o for gcc/clang/nvcc, /Fo for MSVC), so nothing outside the real build is compiled.
# Sorted, then dealt round-robin (line % N == K): the heavy template instances sit together
# in the sorted list and spread evenly across slices.
#
# Usage: slice_objects.sh <build dir> <K> <N>   (0 <= K < N)
set -euo pipefail
dir="${1:?build dir}"; k="${2:?slice}"; n="${3:?slices}"
case "$k$n" in *[!0-9]*) echo "slice and slices must be integers" >&2; exit 2 ;; esac
[ "$k" -lt "$n" ] || { echo "slice $k is outside 0..$(( n - 1 ))" >&2; exit 2; }
objs="$(ninja -C "$dir" -t commands all \
  | grep -oE '(-o |/Fo)[^ "]+\.(o|obj)( |$)' | sed -E 's#^(-o |/Fo)##; s# $##' | sort -u)"
[ -n "$objs" ] || { echo "no object targets found under $dir; refusing an empty slice list" >&2; exit 1; }
printf '%s\n' "$objs" | awk -v k="$k" -v n="$n" 'NR % n == k'
