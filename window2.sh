#!/bin/bash
set -u
echo "queueing for the pair at $(date)"
for i in 1 2 3 4 5 6; do
  SPARK_LOCK_MAX_WAIT=21600 /home/nvidianew/gpu_lock.sh acquire both "rpc trace" 2>&1 | tail -3
  s=$(/home/nvidianew/gpu_lock.sh status 2>&1 | head -2)
  if echo "$s" | grep -q "local: HELD by rpc trace" && echo "$s" | grep -q "peer: HELD by rpc trace"; then
    echo "HOLDING THE PAIR at $(date)"; break
  fi
  echo "attempt $i did not get the pair, retrying"
  /home/nvidianew/gpu_lock.sh release both "rpc trace" >/dev/null 2>&1
  sleep 60
done
s=$(/home/nvidianew/gpu_lock.sh status 2>&1 | head -2)
if ! (echo "$s" | grep -q "local: HELD by rpc trace" && echo "$s" | grep -q "peer: HELD by rpc trace"); then
  echo "GAVE UP waiting for the pair"; exit 2
fi

/home/nvidianew/link_check.sh --quick 2>&1 | tail -6

echo "=== non RPC bracket (single GPU) ==="
bash /home/nvidianew/temp/wt_trace/scripts/rpc_trace/nonrpc_bracket.sh 2>&1 | tail -60

echo "=== traced layer split cells ==="
bash /home/nvidianew/temp/wt_trace/scripts/rpc_trace/gpu_trace.sh 2>&1 | tail -260
rc=$?
/home/nvidianew/gpu_lock.sh release both "rpc trace" 2>&1 | tail -3
echo "released at $(date), gpu_trace rc=$rc"
