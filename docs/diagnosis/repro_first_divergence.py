"""Find the EXACT token position where vLLM and llama-server's greedy outputs first disagree.

Generates greedy outputs from both, then walks token-by-token to find the first
mismatch. At that mismatch, prints both servers' top-20 logprobs at that position.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download
from openai import OpenAI


def load_system_prompt() -> str:
    p = hf_hub_download(repo_id="mistralai/Mistral-Medium-3.5-128B", filename="SYSTEM_PROMPT.txt")
    raw = Path(p).read_text()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    return raw.format(name="Mistral-Medium-3.5-128B", today=today, yesterday=yesterday)


SYSTEM_PROMPT = load_system_prompt()
MESSAGES = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "Create a Flappy Bird Python game"},
]


def greedy_with_logprobs(base_url: str, max_tokens: int = 100):
    client = OpenAI(api_key="EMPTY", base_url=base_url)
    model = client.models.list().data[0].id
    r = client.chat.completions.create(
        model=model,
        messages=MESSAGES,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=20,
        extra_body={"reasoning_effort": "none"},
    )
    return r


print("Querying vLLM (max 100 tokens)...")
v = greedy_with_logprobs("http://localhost:8765/v1", max_tokens=100)
print("Querying llama-server (max 100 tokens)...")
l = greedy_with_logprobs("http://localhost:8766/v1", max_tokens=100)

vt = [c.token for c in v.choices[0].logprobs.content]
lt = [c.token for c in l.choices[0].logprobs.content]
v_top = [c.top_logprobs for c in v.choices[0].logprobs.content]
l_top = [c.top_logprobs for c in l.choices[0].logprobs.content]

print(f"vLLM produced {len(vt)} tokens")
print(f"llama produced {len(lt)} tokens")

# vLLM tokens may be in 'token_id:N' form; llama in piece form. Compare via piece by detokenizing vLLM's IDs.
def piece_for_vllm_token(t):
    if t.startswith("token_id:"):
        try:
            tid = int(t.split(":", 1)[1])
            r = requests.post("http://localhost:8766/detokenize", json={"tokens": [tid]}, timeout=30).json()
            return r["content"]
        except Exception:
            return t
    return t


# Walk for first divergence by piece string.
i = 0
n = min(len(vt), len(lt))
while i < n:
    vp = piece_for_vllm_token(vt[i])
    lp = lt[i]
    if vp != lp:
        break
    i += 1

print(f"\nFirst divergence at decoded position {i}")
if i < n:
    vp = piece_for_vllm_token(vt[i])
    lp = lt[i]
    print(f"  vLLM picked: {vp!r} (raw token: {vt[i]!r})")
    print(f"  llama picked: {lp!r}")

    # Show top-20 at divergence position
    print("\nvLLM top-20 at this position:")
    for tlp in (v_top[i] or [])[:10]:
        piece = piece_for_vllm_token(tlp.token)
        print(f"  {piece!r:<25} {tlp.logprob:.4f}")

    print("\nllama top-10 at this position:")
    for tlp in (l_top[i] or [])[:10]:
        print(f"  {tlp.token!r:<25} {tlp.logprob:.4f}")

    # Also: prefix that was generated identically up to this point
    if i > 0:
        # detokenize identical prefix from llama's piece list
        prefix = "".join(lt[:i])
        print(f"\nIdentical prefix ({len(prefix)} chars):\n{prefix!r}")
else:
    print("No divergence in the first 100 tokens.")

out = Path(f"outputs/diag_first_divergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
out.write_text(json.dumps({
    "vllm_tokens": vt,
    "llama_tokens": lt,
    "first_divergence_index": i,
    "vllm_top20_at_divergence": [(t.token, t.logprob) for t in (v_top[i] if i < len(v_top) else [])] if i < n else [],
    "llama_top20_at_divergence": [(t.token, t.logprob) for t in (l_top[i] if i < len(l_top) else [])] if i < n else [],
}, indent=2))
print(f"\nsaved {out}")
