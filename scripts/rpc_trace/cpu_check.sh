#!/bin/bash
# CPU-only smoke test: layers split over two rpc-servers, run with the trace off and on, checking
# the text is identical and that merge.py accepts the result.
#
# pipefail matters here: the completions run as `curl | parser`, and without it the status of a
# failed request is thrown away and two empty output files compare equal, so the test would
# report PASS without ever generating a token. errexit is deliberately not set, because the
# teardown deals in `kill` and `grep -q` calls that are expected to fail; every step that can
# fail is checked by hand instead.
set -u
set -o pipefail

BUILD=${1:?usage: cpu_check.sh <build-dir> <model.gguf> [outdir]}
MODEL=${2:?usage: cpu_check.sh <build-dir> <model.gguf> [outdir]}
OUT=${3:-/tmp/rpc_trace_cpu}

BIN=$BUILD/bin
P1=${P1:-50111}
P2=${P2:-50112}
PORT=${PORT:-8197}

mkdir -p "$OUT"
export LD_LIBRARY_PATH=$BIN
export LLAMA_ARG_OFFLINE=1
export CUDA_VISIBLE_DEVICES=      # CPU backend only

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill -9 "$p" 2>/dev/null; done; }
trap cleanup EXIT

PROMPTS=("the capital of France is" "two plus two equals" "the colour of the sky is")

# complete <tag> <prompt> -- one completion appended to $OUT/$tag.out.txt, non-zero on any failure
complete() {
  local tag=$1 prompt=$2
  local body="$OUT/$tag.resp.json" code rc

  code=$(curl -sS --max-time 300 -o "$body" -w '%{http_code}' \
           http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
           -d "{\"prompt\":\"$prompt\",\"n_predict\":32,\"temperature\":0,\"top_k\":1,\"seed\":1}")
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$tag: request for '$prompt' failed, curl exit $rc (is anything listening on $PORT?)" >&2
    return 1
  fi
  if [ "$code" != "200" ]; then
    echo "$tag: request for '$prompt' returned HTTP $code, body:" >&2
    head -c 400 "$body" >&2; echo >&2
    return 1
  fi
  # a reply without a usable "content" is a failure too, not an empty line in the output file
  python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception as e:
    sys.exit("response is not valid JSON: %s" % e)
if not isinstance(d, dict):
    sys.exit("response is not a JSON object")
if "content" not in d:
    sys.exit("response has no \"content\" (keys: %s)" % ", ".join(sorted(map(str, d))))
if not isinstance(d["content"], str) or not d["content"].strip():
    sys.exit("response \"content\" is empty")
sys.stdout.write("=== %s\n%s\n" % (sys.argv[2], d["content"]))
' "$body" "$prompt" >> "$OUT/$tag.out.txt"
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$tag: could not read a completion for '$prompt' out of the reply" >&2
    return 1
  fi
  return 0
}

# cell <tag>   (PEER_TRACE is the prefix of the peer trace files)
cell() {
  local tag=$1; shift
  rm -f "$OUT/$tag.out.txt"
  local targs1=() targs2=()
  if [ -n "${PEER_TRACE:-}" ]; then targs1=(--trace "$PEER_TRACE.1.jsonl"); targs2=(--trace "$PEER_TRACE.2.jsonl"); fi
  "$BIN/ggml-rpc-server" -H 127.0.0.1 -p $P1 -t 4 ${targs1[@]+"${targs1[@]}"} > "$OUT/$tag.rpc1.log" 2>&1 & pids+=($!)
  "$BIN/ggml-rpc-server" -H 127.0.0.1 -p $P2 -t 4 ${targs2[@]+"${targs2[@]}"} > "$OUT/$tag.rpc2.log" 2>&1 & pids+=($!)
  sleep 3

  "$BIN/llama-server" -m "$MODEL" -ngl 99 --host 127.0.0.1 --port $PORT --no-webui \
      -c 2048 --parallel 2 --rpc 127.0.0.1:$P1,127.0.0.1:$P2 --device RPC0,RPC1 -sm layer \
      --cache-ram 0 -t 4 > "$OUT/$tag.server.log" 2>&1 & local sp=$!
  pids+=($sp)
  for i in $(seq 1 300); do grep -q "listening on" "$OUT/$tag.server.log" && break; sleep 1; done
  if ! grep -q "listening on" "$OUT/$tag.server.log"; then
    echo "$tag: server failed to start"; tail -20 "$OUT/$tag.server.log"; return 1
  fi

  : > "$OUT/$tag.out.txt"
  local ok=0 rc=0
  for p in "${PROMPTS[@]}"; do
    if complete "$tag" "$p"; then ok=$((ok + 1)); else rc=1; fi
  done
  # the point of the test is comparing generated text, so a missing response is a failure and
  # must not be allowed to leave an empty file that would compare equal to another empty file
  if [ $ok -ne ${#PROMPTS[@]} ]; then
    echo "$tag: only $ok of ${#PROMPTS[@]} completions were recorded" >&2
    rc=1
  fi
  if [ ! -s "$OUT/$tag.out.txt" ]; then
    echo "$tag: no generated text at all in $OUT/$tag.out.txt" >&2
    rc=1
  fi

  kill -TERM $sp 2>/dev/null
  for i in $(seq 1 30); do kill -0 $sp 2>/dev/null || break; sleep 1; done
  kill -9 $sp 2>/dev/null
  sleep 1
  for p in "${pids[@]:-}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 2
  for p in "${pids[@]:-}"; do kill -9 "$p" 2>/dev/null; done
  pids=()
  return $rc
}

echo "== trace off"
unset GGML_RPC_TRACE
if ! cell off; then
  echo "the untraced run did not produce all ${#PROMPTS[@]} completions: FAIL"
  exit 1
fi

echo "== trace on"
export GGML_RPC_TRACE=$OUT/on.client.jsonl
if ! PEER_TRACE=$OUT/on.peer cell on; then
  echo "the traced run did not produce all ${#PROMPTS[@]} completions: FAIL"
  exit 1
fi
unset GGML_RPC_TRACE

echo
# belt and braces: never compare two files that hold nothing
for f in "$OUT/off.out.txt" "$OUT/on.out.txt"; do
  if [ ! -s "$f" ]; then echo "no generated text in $f, nothing was compared: FAIL"; exit 1; fi
done
for f in "$OUT/off.out.txt" "$OUT/on.out.txt"; do
  got=$(grep -c '^=== ' "$f")
  if [ "$got" -ne ${#PROMPTS[@]} ]; then
    echo "$f holds $got of ${#PROMPTS[@]} responses: FAIL"; exit 1
  fi
done
echo "compared ${#PROMPTS[@]} completions, $(wc -c < "$OUT/on.out.txt") bytes of generated text"

if cmp -s "$OUT/off.out.txt" "$OUT/on.out.txt"; then
  echo "output identical with the trace off and on: PASS"
else
  echo "output DIFFERS with the trace on: FAIL"
  diff "$OUT/off.out.txt" "$OUT/on.out.txt" | head -20
  exit 1
fi

for f in "$OUT/on.client.jsonl" "$OUT/on.peer.1.jsonl" "$OUT/on.peer.2.jsonl"; do
  if [ ! -s "$f" ]; then echo "missing or empty trace $f: FAIL"; exit 1; fi
  echo "$(basename "$f"): $(wc -l < "$f") lines"
done

python3 "$(dirname "$0")/merge.py" "$OUT/on.client.jsonl" "$OUT/on.peer.1.jsonl" "$OUT/on.peer.2.jsonl" \
    --chrome "$OUT/on.chrome.json" --summary "$OUT/on.summary.txt" || exit 1
cat "$OUT/on.summary.txt"
