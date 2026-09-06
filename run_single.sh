#!/bin/bash
# Single node host-stall cells. No RPC, no pipeline groups: every phase measured here is
# pure llama-server host work, so the result applies to one Spark as much as to a pair.
set -u
D=/home/nvidianew/temp/wt_srvperf
O=$D/bench; S=$O/samples
mkdir -p $O $S
M=/home/nvidianew/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf
PORT=8191; PY=/home/nvidianew/temp/llamacpp_pipe/tvenv/bin/python
LOG=$O/run_single.log
NTG=${NTG:-256}; NPP=${NPP:-128}
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

marker() {
  local tag=$1
  local t=$(cat /sys/class/thermal/thermal_zone*/temp | sort -rn | head -1)
  say "  MARK $tag clock=$(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader) temp=$t uptime=[$(uptime)] capped_local=$(cat /tmp/spark_locks/local.capped 2>/dev/null || echo none) capped_peer=$(cat /tmp/spark_locks/peer.capped 2>/dev/null || echo none)"
}
guard() {
  while true; do
    local t=$(cat /sys/class/thermal/thermal_zone*/temp | sort -rn | head -1)
    if [ "$t" -gt 80000 ]; then say "  thermal $t over 80000, waiting 5 min"; sleep 300; else say "  thermal ok $t"; return 0; fi
  done
}

SRVPID=""
stop_srv() {
  [ -n "$SRVPID" ] || return 0
  kill -TERM $SRVPID 2>/dev/null
  for i in $(seq 1 120); do kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  kill -9 $SRVPID 2>/dev/null; SRVPID=""
  for i in $(seq 1 60); do timeout 2 bash -c "</dev/tcp/127.0.0.1/$PORT" 2>/dev/null || return 0; sleep 1; done
}
trap 'stop_srv' EXIT

# cell <tag> <bindir> <trace 0|1> <conc list> [extra server args...]
cell() {
  local tag=$1; local bin=$2; local trace=$3; local conc=$4; shift 4
  guard; marker "$tag"
  local free_g=$(free -g | awk '/^Mem:/{print $7}')
  say "  free mem ${free_g}G"
  [ "$free_g" -lt 25 ] && { say "  !!! low memory, abort"; exit 1; }

  local tracenv=()
  [ "$trace" = 1 ] && tracenv=(GGML_RPC_TRACE=$O/${tag}_client.jsonl)

  ( exec nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 200 | while IFS= read -r l; do echo "$(date +%s.%N),$l"; done > "$S/${tag}.csv" ) & local SL=$!

  env ${tracenv[@]+"${tracenv[@]}"} LD_LIBRARY_PATH=$bin LLAMA_ARG_OFFLINE=1 \
    $bin/llama-server -m "$M" -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-webui --slots \
    -c 16384 --parallel 32 --cache-ram 0 -t 6 "$@" > "$O/$tag.server.log" 2>&1 &
  SRVPID=$!
  for i in $(seq 1 1200); do grep -q "listening on" "$O/$tag.server.log" && break; sleep 1; done
  grep -q "listening on" "$O/$tag.server.log" || { say "  !!! $tag failed to start"; tail -8 "$O/$tag.server.log" | tee -a "$LOG"; stop_srv; kill $SL 2>/dev/null; return 1; }
  say "  $tag up"

  local t0=$(date +%s.%N)
  $PY /home/nvidianew/temp/userscale/bench_users.py --backend http://127.0.0.1:$PORT --label $tag \
     --conc $conc --npp $NPP --ntg $NTG --reqs-per-client 2 --out "$O/$tag.bench.jsonl" > "$O/$tag.bench.log" 2>&1
  echo "$t0 $(date +%s.%N)" > "$O/$tag.window"
  cat "$O/$tag.bench.jsonl" | tee -a "$LOG"

  stop_srv
  for c in $(pgrep -P $SL 2>/dev/null); do kill $c 2>/dev/null; done; kill $SL 2>/dev/null
  marker "$tag-end"

  if [ "$trace" = 1 ]; then
    $PY $D/scripts/rpc_trace/merge.py "$O/${tag}_client.jsonl" \
        --summary "$O/${tag}.summary.txt" 2>>"$LOG"
    say "  --- $tag summary"; cat "$O/${tag}.summary.txt" 2>/dev/null | tee -a "$LOG"
    $PY $D/scripts/phase_table.py "$O/${tag}_client.jsonl" 2>&1 | tee -a "$LOG"
  fi
}

BASE=${BASE:-/home/nvidianew/temp/wt_srvperf/bin_base}
NEW=${NEW:-/home/nvidianew/temp/wt_srvperf/build/bin}

cell base_t32 "$BASE" 1 32
cell new_t32  "$NEW"  1 32

cell base_a "$BASE" 0 1,8,32
cell new_b  "$NEW"  0 1,8,32
cell base_c "$BASE" 0 1,8,32

# what the prefill checkpoints cost, using the knob that already exists
cell base_ckpt0 "$BASE" 0 32 -ctxcp 0

say "run_single done"
