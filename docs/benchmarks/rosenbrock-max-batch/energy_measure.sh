#!/usr/bin/env bash
# Measure GPU energy for a benchmark command by integrating nvidia-smi power.draw.
#
# The 6 GB RTX 3060 Laptop has no hardware joule counter (total_energy_consumption
# is unsupported), so energy is integrated from 20 Hz power samples over the run:
#   E = integral(power dt)  [trapezoidal]
# Reports total joules, active joules (minus the idle baseline), average watts,
# and -- given the processed-sample count -- joules/sample and samples/joule.
#
# usage: energy_measure.sh <samples_processed> <label> -- <command...>
#   samples_processed: total samples the command runs through its TIMED loop
#                      (so energy/sample is comparable across engines/batches)
#
# Set ENERGY_IDLE_W to override the measured idle baseline (default 26.8 W).

set -u
IDLE_W=${ENERGY_IDLE_W:-26.8}

samples=$1; label=$2; shift 2
[ "$1" = "--" ] && shift
log=$(mktemp /tmp/energy.XXXXXX.csv)

# Background power logger: "epoch_seconds,watts" at 20 Hz with wall timestamps.
nvidia-smi --query-gpu=timestamp,power.draw --format=csv,noheader,nounits -lms 50 \
  > "$log" 2>/dev/null &
logger_pid=$!
sleep 0.3                         # let the logger spin up

t0=$(date +%s.%N)
"$@" >/tmp/energy_cmd.out 2>&1
rc=$?
t1=$(date +%s.%N)

sleep 0.3
kill "$logger_pid" 2>/dev/null; wait "$logger_pid" 2>/dev/null

# Integrate trapezoidally over the samples whose timestamp falls in [t0, t1].
# nvidia-smi timestamp format: "YYYY/MM/DD HH:MM:SS.mmm" -> seconds-of-day is enough
# for dt between consecutive samples; we just need spacing, so use sample index * dt
# fallback if timestamp parse fails. Here we parse the HH:MM:SS.mmm into seconds.
awk -v idle="$IDLE_W" -v n="$samples" -v lbl="$label" -v rc="$rc" -v wall="$(echo "$t1 - $t0" | bc -l)" '
function tosec(ts,   a, hms, s) {
  # ts like 2026/06/14 02:10:33.456
  split(ts, a, " "); split(a[2], hms, ":")
  return hms[1]*3600 + hms[2]*60 + hms[3]
}
{
  gsub(/^ +| +$/, "", $0)
  split($0, f, ", ")
  t = tosec(f[1]); w = f[2] + 0
  if (prev_t != "") {
    dt = t - prev_t
    if (dt < 0) dt += 86400            # midnight wrap guard
    if (dt > 0 && dt < 2) {            # ignore gaps from logger startup/teardown
      e_total  += 0.5*(w + prev_w)*dt
      e_active += 0.5*((w-idle) + (prev_w-idle))*dt
      span += dt; np++
    }
  }
  prev_t = t; prev_w = w; sumw += w; cnt++
}
END {
  if (span <= 0) { print "ERROR: no usable power samples"; exit 1 }
  avg_w = sumw/cnt
  printf "label=%s\n", lbl
  printf "wall_s=%.3f  power_samples=%d  integ_span_s=%.3f\n", wall, cnt, span
  printf "avg_power_W=%.2f  idle_W=%.2f  active_power_W=%.2f\n", avg_w, idle, avg_w-idle
  printf "energy_total_J=%.2f  energy_active_J=%.2f\n", e_total, e_active
  if (n > 0) {
    printf "J_per_sample_total=%.6f  J_per_sample_active=%.6f\n", e_total/n, e_active/n
    printf "samples_per_J_active=%.0f\n", (e_active>0)? n/e_active : 0
  }
}
' "$log"

rm -f "$log"
exit $rc
