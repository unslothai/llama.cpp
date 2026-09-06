#!/bin/bash
# Strict determinism check: one slot, one request at a time, so the batch composition is
# fixed. The concurrent variant is not usable as a control on this model, a base against base
# run already produces two different md5s.
set -u
D=/home/nvidianew/temp/wt_srvperf
O=$D/bench; mkdir -p $O
M=${M:-/home/nvidianew/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=8193

run_one() {
  local tag=$1; local bin=$2
  LD_LIBRARY_PATH=$bin LLAMA_ARG_OFFLINE=1 $bin/llama-server -m "$M" -ngl 99 -fa on --host 127.0.0.1 --port $PORT \
     --no-webui -c 4096 --parallel 1 --cache-ram 0 -t 6 > "$O/strict_$tag.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 1200); do grep -q "listening on" "$O/strict_$tag.server.log" && break; sleep 1; done
  grep -q "listening on" "$O/strict_$tag.server.log" || { echo "$tag FAILED to start"; tail -5 "$O/strict_$tag.server.log"; kill -9 $pid; return 1; }
  : > "$O/strict_$tag.txt"
  for p in "The capital of France is" "Write a haiku about compilers:" "1, 1, 2, 3, 5, 8," "def fibonacci(n):"; do
    for st in false true; do
      curl -s -N -X POST http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
        -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$p"), \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"seed\": 42, \"cache_prompt\": false, \"stream\": $st}" \
        | python3 -c '
import sys, json
raw = sys.stdin.read()
if raw.lstrip().startswith("{"):
    print(json.loads(raw)["content"])
else:
    out=[]
    for line in raw.splitlines():
        if line.startswith("data:"):
            try: d=json.loads(line[5:])
            except Exception: continue
            out.append(d.get("content",""))
    print("".join(out))' >> "$O/strict_$tag.txt"
    done
  done
  kill -TERM $pid 2>/dev/null; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
  for i in $(seq 1 60); do timeout 2 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null || break; sleep 1; done
  md5sum < "$O/strict_$tag.txt" | awk -v t="$tag" '{print t" md5 "$1}'
}

run_one base "${BASE:-$D/bin_base}"
run_one new  "${NEW:-$D/build/bin}"
