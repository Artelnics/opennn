#!/usr/bin/env bash
# Max-batch search for OpenNN via FRESH PROCESS PER TRIAL.
#
# A CUDA OOM in OpenNN can leave the context corrupted (sticky error 700), so an
# in-process binary search converges on garbage. Here each batch attempt is its
# own ./opennn_rosenbrock_trial process: a fault dies with that process and
# cannot poison later trials. Exit 0 = fit, non-zero = fail.
#
# For each fitting trial we also sample peak GPU memory (background nvidia-smi
# poller) so we can confirm the cliff is VRAM-bound (~6144 MiB) and not WSL
# host-RAM spillover.
#
#   usage: run_maxbatch.sh [inputs] [hidden]

cd "$(dirname "$0")"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
INPUTS=${1:-1000}
HIDDEN=${2:-1000}
BIN=./opennn_rosenbrock_trial

trial() {            # mode batch  -> echoes "ok <peak_mib>" or "fail"
  local mode=$1 batch=$2
  : > /tmp/mbpeak.txt
  ( peak=0; while true; do
      u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
      [ -n "$u" ] && [ "$u" -gt "$peak" ] && peak=$u && echo "$peak" > /tmp/mbpeak.txt
      sleep 0.1
    done ) & local s=$!
  $BIN "$mode" "$batch" "$INPUTS" "$HIDDEN" >/dev/null 2>&1
  local rc=$?
  kill $s 2>/dev/null; wait $s 2>/dev/null
  # wait for the GPU to drain so the next trial starts from a clean slate
  for _ in $(seq 1 50); do
    [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)" = "0" ] && break
    sleep 0.2
  done
  if [ $rc -eq 0 ]; then echo "ok $(cat /tmp/mbpeak.txt 2>/dev/null)"; else echo "fail"; fi
}

search() {           # mode -> prints result line
  local mode=$1
  local lo=0 hi=1024 peak=0
  # exponential grow until first failure
  while :; do
    read -r st pk < <(trial "$mode" "$hi")
    echo "  [$mode b=$hi] $st ${pk:+peak=${pk}MiB}" >&2
    if [ "$st" = "ok" ]; then lo=$hi; peak=${pk:-$peak}; hi=$((hi*2)); else break; fi
  done
  # binary-search the boundary between lo (fits) and hi (fails)
  while [ $((lo+1)) -lt $hi ]; do
    local mid=$(((lo+hi)/2))
    read -r st pk < <(trial "$mode" "$mid")
    echo "  [$mode b=$mid] $st ${pk:+peak=${pk}MiB}" >&2
    if [ "$st" = "ok" ]; then lo=$mid; peak=${pk:-$peak}; else hi=$mid; fi
  done
  echo "mode=$mode max_batch=$lo peak_vram_mib_at_max=$peak"
}

echo "net=${INPUTS}->${HIDDEN}->1 (tanh), fp32  (OpenNN, fresh-process-per-trial)"
echo "total_vram_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
search inference
search train
echo "RESULT=OK"
