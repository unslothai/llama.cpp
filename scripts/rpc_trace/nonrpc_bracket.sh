#!/bin/bash
# Single GPU, no RPC backend involved: llama-batched-bench base/new/base plus one traced pass,
# and a greedy md5 with the trace off and on, to show that a workload that does not use RPC is
# not moved by the tracer.
set -u
W=/home/nvidianew/temp/wt_trace; B0=/home/nvidianew/temp/wt_base/build/bin; B1=$W/build/bin
O=$W/bench; mkdir -p $O
M=/home/nvidianew/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $O/bracket.log; }

for pass in base_1 new base_2 new_traced; do
  case $pass in base_*) BIN=$B0;; new*) BIN=$B1;; esac
  tenv=()
  [ "$pass" = new_traced ] && tenv=(GGML_RPC_TRACE=$O/nonrpc_trace.jsonl)
  say "=== batched-bench $pass ($BIN)"
  env ${tenv[@]+"${tenv[@]}"} LD_LIBRARY_PATH=$BIN LLAMA_ARG_OFFLINE=1 \
     $BIN/llama-batched-bench -m "$M" -c 32768 -npp 512 -ntg 128 -npl 1,8,32 \
     -ngl 99 -fa on -t 6 > $O/bb_$pass.log 2>&1
  grep -E "^\|" $O/bb_$pass.log | tail -4 | tee -a $O/bracket.log
done

for pass in base new new_traced; do
  case $pass in base) BIN=$B0;; new*) BIN=$B1;; esac
  tenv=()
  [ "$pass" = new_traced ] && tenv=(GGML_RPC_TRACE=$O/nonrpc_greedy.jsonl)
  PORT=8194
  env ${tenv[@]+"${tenv[@]}"} LD_LIBRARY_PATH=$BIN LLAMA_ARG_OFFLINE=1 \
     $BIN/llama-server -m "$M" -ngl 99 -fa on --host 127.0.0.1 --port $PORT \
     --no-webui -c 8192 --parallel 8 --cache-ram 0 -t 6 > $O/greedy_$pass.srv.log 2>&1 &
  sp=$!
  for i in $(seq 1 600); do grep -q "listening on" $O/greedy_$pass.srv.log && break; sleep 1; done
  : > $O/greedy_$pass.txt
  for p in "The capital of France is" "Explain gravity in one sentence." "def fibonacci(n):" "List three primes:" "Once upon a time"; do
    curl -s http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json,sys; print(json.dumps({'prompt':sys.argv[1],'n_predict':48,'temperature':0,'top_k':1,'seed':1234,'cache_prompt':False}))" "$p")" \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['content'])" >> $O/greedy_$pass.txt
  done
  kill -TERM $sp 2>/dev/null; for i in $(seq 1 60); do kill -0 $sp 2>/dev/null || break; sleep 1; done; kill -9 $sp 2>/dev/null
  say "single GPU greedy $pass: $(md5sum < $O/greedy_$pass.txt)"
done
say "bracket done"
