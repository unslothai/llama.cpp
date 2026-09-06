#!/bin/bash
# Same prompt, greedy, through both binaries. The md5 of the generated text must match.
set -u
D=/home/nvidianew/temp/wt_srvperf
O=$D/bench; mkdir -p $O
M=${M:-/home/nvidianew/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf}
PORT=8192
EXTRA=${EXTRA:-}

run_one() {
  local tag=$1; local bin=$2
  LD_LIBRARY_PATH=$bin LLAMA_ARG_OFFLINE=1 $bin/llama-server -m "$M" $EXTRA --host 127.0.0.1 --port $PORT \
     --no-webui -c 8192 --parallel 4 --cache-ram 0 -t 6 > "$O/greedy_$tag.server.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 1200); do grep -q "listening on" "$O/greedy_$tag.server.log" && break; sleep 1; done
  grep -q "listening on" "$O/greedy_$tag.server.log" || { echo "$tag FAILED to start"; tail -5 "$O/greedy_$tag.server.log"; kill -9 $pid; return 1; }
  : > "$O/greedy_$tag.txt"
  # four prompts, two of them concurrent, so the batched path is covered too
  for p in "The capital of France is" "Write a haiku about compilers:" "1, 1, 2, 3, 5, 8," "def fibonacci(n):"; do
    curl -s -X POST http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$p"), \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"seed\": 42, \"cache_prompt\": false, \"stream\": false}" \
      | python3 -c 'import json,sys; print(json.load(sys.stdin)["content"])' >> "$O/greedy_$tag.txt"
  done
  # and one concurrent pair, streamed, which is the path the fix touches
  local cpids=()
  for p in "Once upon a time" "The three laws of robotics are"; do
    curl -s -N -X POST http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "{\"prompt\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$p"), \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"seed\": 42, \"cache_prompt\": false, \"stream\": true}" \
      | python3 -c '
import sys
out=[]
for line in sys.stdin:
    if line.startswith("data:"):
        import json
        try: d=json.loads(line[5:])
        except Exception: continue
        out.append(d.get("content",""))
print("".join(out))' > "$O/greedy_${tag}_s$RANDOM$$.part" &
    cpids+=($!)
  done
  wait "${cpids[@]}"
  cat "$O"/greedy_${tag}_s*.part >> "$O/greedy_$tag.txt" 2>/dev/null; rm -f "$O"/greedy_${tag}_s*.part
  kill -TERM $pid 2>/dev/null; for i in $(seq 1 120); do kill -0 $pid 2>/dev/null || break; sleep 1; done; kill -9 $pid 2>/dev/null
  sort "$O/greedy_$tag.txt" | md5sum | awk -v t="$tag" '{print t" md5 "$1}'
}

run_one base "${BASE:-$D/bin_base}"
run_one new  "${NEW:-$D/build/bin}"
