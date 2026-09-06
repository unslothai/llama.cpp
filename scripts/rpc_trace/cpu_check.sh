#!/bin/bash
# Smoke test for the event tracer with no GPU involved: two local rpc-servers on the CPU backend
# and one llama-server splitting the layers over them, run once with the trace off and once with
# it on. Checks that the generated text is identical either way and that the merge tool accepts
# the result.
#
#   scripts/rpc_trace/cpu_check.sh <build-dir> <model.gguf> [outdir]
set -u

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

# cell <tag>   (PEER_TRACE, if set, is the prefix of the peer trace files)
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
  for p in "the capital of France is" "two plus two equals" "the colour of the sky is"; do
    curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\":\"$p\",\"n_predict\":32,\"temperature\":0,\"top_k\":1,\"seed\":1}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])' >> "$OUT/$tag.out.txt"
  done

  kill -TERM $sp 2>/dev/null
  for i in $(seq 1 30); do kill -0 $sp 2>/dev/null || break; sleep 1; done
  kill -9 $sp 2>/dev/null
  sleep 1
  for p in "${pids[@]}"; do kill -TERM "$p" 2>/dev/null; done
  sleep 2
  for p in "${pids[@]}"; do kill -9 "$p" 2>/dev/null; done
  pids=()
}

echo "== trace off"
unset GGML_RPC_TRACE
cell off

echo "== trace on"
export GGML_RPC_TRACE=$OUT/on.client.jsonl
PEER_TRACE=$OUT/on.peer cell on
unset GGML_RPC_TRACE

echo
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
