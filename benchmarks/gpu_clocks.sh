#!/bin/bash
# Lock the GPU to a fixed SM clock for benchmarking, or release it again.
#
# Why: this class of card idles near 400 MHz, takes ~2.5 s of load to reach its
# boost clock, and its sustained boost drifts with ambient temperature over a
# session. Measured on an RTX 5070 Ti, one engine's HIGGS training reading moved
# 8% across a single day while another's held, and three same-session
# interleaved measurements of the same cell gave 1.019x, 0.993x and 0.997x.
# Margins under about 2% are not resolvable while the clock floats, however many
# rounds are averaged, because the drift is slower than a round.
#
# Persistence mode is set too: without it the driver tears down GPU state
# between processes, and every benchmark process is a fresh process.
#
#   sudo ./gpu_clocks.sh lock [MHz]   # default 2700
#   sudo ./gpu_clocks.sh unlock
#   ./gpu_clocks.sh status            # no root needed
#
# Pick a clock the card can hold indefinitely, not its maximum: locking at the
# boost ceiling reintroduces the throttling this is meant to remove. 2700 is
# below the 2835 MHz this card sustained at 230 W of a 300 W limit.

set -euo pipefail
action=${1:-status}
mhz=${2:-2700}

case "$action" in
  lock)
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
    echo "usage: $0 {lock [MHz]|unlock|status}" >&2
    exit 2
    ;;
esac
