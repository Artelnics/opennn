#!/bin/bash
# Lock a sustainable graphics clock chosen for the GPU under test.
# Usage: sudo ./gpu_clocks.sh lock MHz | unlock | status
set -euo pipefail
action=${1:-status}
mhz=${2:-${OPENNN_BENCH_SM_CLOCK_MHZ:-}}

case "$action" in
  lock)
    if [[ ! "$mhz" =~ ^[0-9]+$ || ! "$mhz" =~ [1-9] ]]; then
      echo "Specify a positive integer MHz target: lock MHz or OPENNN_BENCH_SM_CLOCK_MHZ" >&2
      exit 2
    fi
    nvidia-smi -pm 1
    nvidia-smi -lgc "$mhz"
    echo "locked to ${mhz} MHz; release with: sudo $0 unlock"
    ;;
  unlock)
    nvidia-smi -rgc
    nvidia-smi -pm 0
    echo "clocks released"
    ;;
  status)
    nvidia-smi --query-gpu=persistence_mode,clocks.sm,clocks.max.sm,temperature.gpu,power.draw \
               --format=csv
    ;;
  *)
    echo "usage: $0 {lock MHz|unlock|status}" >&2
    exit 2
    ;;
esac
