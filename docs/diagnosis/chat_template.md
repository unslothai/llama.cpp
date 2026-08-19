# Phase 2 — chat template parity

## TL;DR

The four chat templates considered — (A) GGUF-embedded jinja in
`bartowski/mistralai_Mistral-Medium-3.5-128B-GGUF`, (B) `unsloth/Mistral-Medium-3.5-128B`,
(C) `mistralai/Mistral-Medium-3.5-128B` upstream, (D) the `tokenizer.chat_template`
attribute exposed by HF AutoTokenizer for the upstream model — are
**semantically equivalent for normal multi-turn chat**. The only diff that
could change rendered output is in error-path code that nobody is hitting in
this test. **Templates are not the cause** of the long-context degradation.

## Files

- `outputs/template_llamacpp.jinja` (13281 bytes) — A
- `outputs/template_unsloth.jinja` (14475 bytes) — B
- `outputs/template_mistralai.jinja` (13479 bytes) — C
- `outputs/template_hftokenizer.jinja` (13479 bytes) — D
- `outputs/template_diff_*` — pairwise diffs

## Diff summary

| pair | size | nature of diff |
| --- | --- | --- |
| C vs D (mistralai vs HF tokenizer) | 0 bytes | **identical** |
| A vs C (GGUF vs upstream) | 686 bytes | one hunk: a disabled `{%- if false %}` assertion at line 201 of the GGUF copy where upstream has the real check `(content == '' or content is none) and (no tool calls)` → raise. Both branches are no-ops on valid messages. |
| B vs C (unsloth vs upstream) | 2040 bytes | (1) unsloth uses `strftime_now` for date defaults; upstream hardcodes `today=29-04-2026`, `yesterday=28-04-2026`. Both get overridden by template arguments at render time, so no effect on output. (2) unsloth `arguments\|tojson\|safe` vs upstream `arguments\|tojson` for tool-call serialisation — only relevant when tools are used. |

## Cross-check via tokenization parity (Phase 1)

When all four templates render the same multi-turn fixture (system prompt + 3
user/assistant pairs + final user turn, `reasoning_effort='none'`), they
produce **byte-identical 434-token sequences**, including `<s>`,
`[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[MODEL_SETTINGS]`, `[/MODEL_SETTINGS]`,
`[INST]`, `[/INST]`, and the per-assistant `</s>` markers in the right
positions. See `outputs/diagnosis_tokenization.md`.

## Conclusion

If we replace llama-server's GGUF-embedded template with the upstream mistralai
copy via `--chat-template-file outputs/template_mistralai.jinja`, the rendered
text and resulting token stream are identical to what the GGUF template
produces today. **No improvement is expected from a template swap, and indeed
no improvement was observed empirically** — see Phase 8 matched-sampler runs.
