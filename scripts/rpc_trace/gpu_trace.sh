#!/bin/bash
# Traced layer split cells on the Spark pair. Each configuration is run twice, once with the
# tracer off and once with it on, so the overhead of the tracer is measured and not assumed.
set -u
D=/home/nvidianew/temp/wt_trace
O=$D/bench; S=$O/samples
mkdir -p $O $S
BIN=$D/build/bin
PEER=192.168.200.13
PEERDIR=/home/nvidianew/temp/wt_trace_bin
M=/home/nvidianew/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf
RPCPORT=50052; PORT=8188; PY=/home/nvidianew/temp/llamacpp_pipe/tvenv/bin/python
LOG=$O/gpu_trace.log
CONC=${CONC:-32}; NTG=${NTG:-256}; NPP=${NPP:-128}
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8 -o ControlMaster=auto -o ControlPath=/tmp/trace_ssh_%h -o ControlPersist=900 nvidianew@$PEER"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

thermals() {
  local l=$(cat /sys/class/thermal/thermal_zone*/temp | sort -rn | head -1)
  local r=$($SSH 'cat /sys/class/thermal/thermal_zone*/temp | sort -rn | head -1')
  echo "$l $r"
}
guard() {
  while true; do
    read a b < <(thermals)
    if [ "$a" -gt 80000 ] || [ "$b" -gt 80000 ]; then
      say "  thermal $a/$b over 80000, waiting 5 min"; sleep 300
    else
      say "  thermal ok local=$a peer=$b"; return 0
    fi
  done
}
memcheck() {
  local lf=$(free -g | awk '/^Mem:/{print $7}')
  local rf=$($SSH "free -g | awk '/^Mem:/{print \$7}'")
  say "  free mem local=${lf}G peer=${rf}G"
  if [ "$lf" -lt 20 ] || [ "$rf" -lt 20 ]; then say "  !!! not enough free memory, abort"; exit 1; fi
}

export LD_LIBRARY_PATH=$BIN LLAMA_ARG_OFFLINE=1

SRVPID=""; PEERPID=""

# Kill every rpc-server this script started on the peer. By pid, and then by a match scoped to
# our own port, because a server left behind holds the port and the next cell silently talks to
# it instead ("Failed to create server socket" in its log and an empty trace).
stop_peer() {
  # note: matched with pgrep -x on the binary name and then on the port in /proc, never with
  #       pgrep -f or pkill -f, whose pattern also matches the remote shell running it
  $SSH "kill -TERM $PEERPID 2>/dev/null; sleep 2; kill -9 $PEERPID 2>/dev/null;
        for p in \$(pgrep -x ggml-rpc-server 2>/dev/null); do
          if tr '\\0' ' ' < /proc/\$p/cmdline 2>/dev/null | grep -q -- '-p $RPCPORT'; then
            kill -9 \$p 2>/dev/null
          fi
        done; true" 2>/dev/null
  PEERPID=""
  # do not return until the port is free again
  for i in $(seq 1 30); do
    timeout 2 bash -c "</dev/tcp/$PEER/$RPCPORT" 2>/dev/null || return 0
    sleep 1
  done
  say "  !!! peer port $RPCPORT still busy"
}
trap '[ -n "$SRVPID" ] && kill -9 $SRVPID 2>/dev/null; stop_peer' EXIT

# cell <tag> <device order> <trace 0|1> [extra server args...]
cell() {
  local tag=$1; local dev=$2; local trace=$3; shift 3
  guard; memcheck

  local ptrace=""
  [ "$trace" = 1 ] && ptrace="--trace /tmp/trace_${tag}_peer.jsonl"
  stop_peer
  # note: no setsid here. $! would then be the pid of setsid and the server, its child, would
  #       survive every kill. -n so ssh does not hold the terminal open waiting for stdin.
  PEERPID=$($SSH -n "cd $PEERDIR && LD_LIBRARY_PATH=$PEERDIR nohup ./ggml-rpc-server -H 0.0.0.0 -p $RPCPORT $ptrace > /tmp/trace_rpc_$tag.log 2>&1 < /dev/null & echo \$!")
  for i in $(seq 1 60); do timeout 2 bash -c "</dev/tcp/$PEER/$RPCPORT" 2>/dev/null && break; sleep 1; done
  if $SSH -n "grep -q 'Failed to create server socket' /tmp/trace_rpc_$tag.log" 2>/dev/null; then
    say "  !!! peer rpc-server for $tag could not bind $RPCPORT, aborting"; exit 1
  fi
  say "  peer rpc-server $PEERPID up ($tag)"

  ( exec nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 100 | while IFS= read -r l; do echo "$(date +%s.%N),$l"; done > "$S/${tag}_local.csv" ) & local SL=$!
  ( exec $SSH 'exec nvidia-smi --query-gpu=utilization.gpu,clocks.sm,temperature.gpu,power.draw --format=csv,noheader -lms 100 | while IFS= read -r l; do echo "$(date +%s.%N),$l"; done' > "$S/${tag}_peer.csv" ) & local SP=$!

  local tracenv=()
  [ "$trace" = 1 ] && tracenv=(GGML_RPC_TRACE=$O/${tag}_client.jsonl)

  env ${tracenv[@]+"${tracenv[@]}"} $BIN/llama-server -m "$M" -ngl 99 -fa on --host 127.0.0.1 --port $PORT --no-webui --slots \
     -c 16384 --parallel 32 --rpc $PEER:$RPCPORT --device $dev -sm layer --cache-ram 0 -t 6 "$@" \
     > "$O/$tag.server.log" 2>&1 &
  SRVPID=$!
  for i in $(seq 1 900); do grep -q "listening on" "$O/$tag.server.log" && break; sleep 1; done
  grep -q "listening on" "$O/$tag.server.log" || { say "  !!! $tag failed to start"; tail -5 "$O/$tag.server.log" | tee -a "$LOG"; }
  say "  $tag up"

  local t0=$(date +%s.%N)
  $PY /home/nvidianew/temp/userscale/bench_users.py --backend http://127.0.0.1:$PORT --label $tag \
     --conc $CONC --npp $NPP --ntg $NTG --reqs-per-client 1 --out "$O/$tag.bench.jsonl" > "$O/$tag.bench.log" 2>&1
  echo "$t0 $(date +%s.%N)" > "$O/$tag.window"
  tail -1 "$O/$tag.bench.jsonl" | tee -a "$LOG"

  kill -TERM $SRVPID 2>/dev/null; for i in $(seq 1 120); do kill -0 $SRVPID 2>/dev/null || break; sleep 1; done
  kill -9 $SRVPID 2>/dev/null; SRVPID=""
  sleep 2   # let the peer flush the tail of its trace before it is stopped
  stop_peer
  say "  peer compute apps after $tag: $($SSH -n 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr "\n" " "')"
  for p in $SL $SP; do for c in $(pgrep -P $p 2>/dev/null); do pkill -P $c 2>/dev/null; kill $c 2>/dev/null; done; kill $p 2>/dev/null; done

  if [ "$trace" = 1 ]; then
    scp -q -o ControlPath=/tmp/trace_ssh_%h nvidianew@$PEER:/tmp/trace_${tag}_peer.jsonl "$O/${tag}_peer.jsonl" || say "  !!! no peer trace for $tag"
    $PY $D/scripts/rpc_trace/merge.py "$O/${tag}_client.jsonl" "$O/${tag}_peer.jsonl" \
        --chrome "$O/${tag}.chrome.json" --summary "$O/${tag}.summary.txt" 2>>"$LOG"
    say "  --- $tag summary"; cat "$O/${tag}.summary.txt" | tee -a "$LOG"
  fi

  $PY - "$O/$tag.window" "$S/${tag}_local.csv" "$S/${tag}_peer.csv" <<'PYEOF' | tee -a "$LOG"
import sys
t0,t1=[float(x) for x in open(sys.argv[1]).read().split()]
for name,path in (("local",sys.argv[2]),("peer",sys.argv[3])):
    u=[];c=[];t=[]
    for line in open(path):
        p=line.strip().split(",")
        if len(p)<5: continue
        try: ts=float(p[0])
        except: continue
        if ts<t0 or ts>t1: continue
        u.append(float(p[1].split()[0])); c.append(float(p[2].split()[0])); t.append(float(p[3]))
    if u: print("  %s util=%.1f%% clocks.sm=%.0fMHz tmax=%.0fC n=%d"%(name,sum(u)/len(u),sum(c)/len(c),max(t),len(u)))
    else: print("  %s no samples"%name)
PYEOF
}

cell n1_cr_off  CUDA0,RPC0 0
cell n1_cr_on   CUDA0,RPC0 1
cell n1_rc_off  RPC0,CUDA0 0
cell n1_rc_on   RPC0,CUDA0 1
cell n2_rc_off  RPC0,CUDA0 0 --pipeline-groups 2
cell n2_rc_on   RPC0,CUDA0 1 --pipeline-groups 2

say "gpu_trace done"
