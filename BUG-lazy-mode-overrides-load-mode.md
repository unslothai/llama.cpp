# `--load-mode none`/`mlock` silently ignored for large "lazy" tensors — undocumented `--lazy-mode` override

## Note on reproducing this

This repo's `mix` tags appear to move (branch is rebased frequently — see e.g. the
"repin #144 ... fixing a brace I dropped in the rebase" commit). `git checkout
b10715-mix-86bd2d3` today does **not** reproduce this — the tag currently resolves to a
commit that predates the `--lazy-mode` code entirely.

The actual evidence is the **release asset**, which is immutable and does contain it:

https://github.com/unslothai/llama.cpp/releases/download/b10715-mix-86bd2d3/llama.cpp-source-commit-92cedc8679d145902ead3f006258e8672eac11e6.tar.gz

(`--lazy-mode` is present in `common/arg.cpp`, `common/common.h`, `include/llama.h`,
`src/llama-model-loader.cpp`/`.h`, `src/llama-model.cpp` inside that tarball.)

## Problem

`--lazy-mode` (default: `auto`) forces mmap-backed, on-demand row-by-row disk reads for
any architecture-marked tensor larger than 4 GiB (e.g. `per_layer_token_embd.weight` in
PLE-style architectures such as `qwen4exp`/gemma3n), regardless of `--load-mode`. In
particular `--load-mode none` — documented as "no special loading mode" — does not
actually prevent this: the tensor is still memory-mapped and read lazily from disk.

This isn't just a documentation gap (the flag also isn't mentioned in this release's
notes at all): users who already relied on `--load-mode none` / `--no-mmap` to get the
whole model fully resident in RAM get a silent behavior change. When the model file is
on rotating storage, the result is a severe, sustained slowdown, because previously
unseen rows are faulted in from disk during generation itself, not only at load time.

## Repro (against the release tarball above)

- `qwen4exp` model (Qwen3.8 Flash Next, `UD-IQ3_XXS`) with `per_layer_token_embd.weight`
  = 27465 MiB, file on HDD
- launch with `--load-mode none`, no `--lazy-mode` passed (defaults to `auto`)
- load log shows:
  ```
  tensor per_layer_token_embd.weight (size = 27465 MiB) lazy read enabled
  ...
  load_tensors:   CPU_Mapped model buffer size = 27465.95 MiB
  ```
  (`CPU_Mapped`, not `CPU` — memory-mapped despite `--load-mode none`)
- generation throughput collapses (observed ~22 t/s -> ~5 t/s on identical
  hardware/model, the only difference being this lazy path engaging) and disk activity
  on the model's drive continues for the entire session, not just at load
- same model/config on SSD: effect is present but minor (SSD random-read latency is
  ~100-200x lower than HDD, so it's much less noticeable — still an unnecessary
  overhead relative to full RAM residency)

## Expected behavior

`--load-mode none` should mean the model — including tensors an architecture marks as
lazy-eligible — is loaded fully into RAM up front. If `--lazy-mode` is meant to be an
independent, higher-priority axis, that should be:
- mentioned in the release notes when introduced, and
- cross-referenced in the `--load-mode`/`--no-mmap` help text, since those flags
  currently imply behavior that `--lazy-mode auto` silently overrides.

## Suggested fix (illustrative — not attached as a diff, see note above on why this
repo's refs aren't stable enough to base a clean PR diff on)

In `src/llama.cpp`, where `ml.lazy.mode` is set from `params.lazy_mode`:

```cpp
ml.lazy.mode    = params.lazy_mode;
ml.model_shared = params.model_shared;

if (ml.lazy.mode != LLAMA_LAZY_MODE_OFF &&
    (params.load_mode == LLAMA_LOAD_MODE_NONE ||
     params.load_mode == LLAMA_LOAD_MODE_MLOCK ||
     params.load_mode == LLAMA_LOAD_MODE_DIRECT_IO)) {
    LLAMA_LOG_WARN("%s: --lazy-mode is '%s' but --load-mode does not request mmap; "
                    "tensors the architecture marks for lazy reading will still be "
                    "memory-mapped and read on demand from disk during inference "
                    "(pass --lazy-mode off to keep them fully resident in RAM)\n",
                    __func__, ml.lazy.mode == LLAMA_LAZY_MODE_ON ? "on" : "auto");
}
```

Either a warning like this, or making `--load-mode none`/`mlock`/`dio` force
`lazy_mode = off` outright, would close the gap.
