#!/usr/bin/env bash
# Run the three CPU inference-speed benchmarks at a shared config and print a
# comparison. All three run the same 2-layer MLP (F -> F -> 1, tanh then linear)
# forward pass over a Rosenbrock dataset, on the CPU.
#
# OpenNN reads the dataset from a CSV; the Python engines synthesize the same
# shape in-process (the numbers don't matter for a speed benchmark, only the
# work).
#
#   usage:  run_inference.sh <csv_path> <samples> <features> [batch] [reps]

set -u

CSV="${1:?csv path}"
SAMPLES="${2:-8000}"
FEATURES="${3:-1000}"
BATCH="${4:-1000}"
REPS="${5:-30}"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_INFERENCE_BIN:-$HOME/opennn/build/bin/opennn_inference}"

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }

echo "############ inference-speed benchmark (CPU) ############"
echo "samples=$SAMPLES features=$FEATURES batch=$BATCH reps=$REPS"
echo

echo "==== OpenNN (CPU, fp32) ===="
o=$("$OPENNN_BIN" "$CSV" "$FEATURES" "$BATCH" "$REPS" 2>/dev/null)
echo "$o" | grep -E 'samples_per_sec|median_pass_s|ms_per_batch|RESULT'
ONN=$(echo "$o" | extract samples_per_sec)

echo "==== PyTorch (CPU, eval + no_grad) ===="
p=$("$VENV" "$HERE/pytorch_inference.py" "$SAMPLES" "$FEATURES" "$BATCH" "$REPS" 2>/dev/null)
echo "$p" | grep -E 'samples_per_sec|median_pass_s|ms_per_batch|RESULT'
PT=$(echo "$p" | extract samples_per_sec)

echo "==== TensorFlow (CPU, training=False) ===="
t=$("$VENV" "$HERE/tensorflow_inference.py" "$SAMPLES" "$FEATURES" "$BATCH" "$REPS" 2>/dev/null)
echo "$t" | grep -E 'samples_per_sec|median_pass_s|ms_per_batch|RESULT'
TF=$(echo "$t" | extract samples_per_sec)

echo
echo "############ summary (samples/sec, higher = faster) ############"
printf "%-28s %s\n" "OpenNN CPU fp32:"   "${ONN:-FAIL}"
printf "%-28s %s\n" "PyTorch CPU:"       "${PT:-FAIL}"
printf "%-28s %s\n" "TensorFlow CPU:"    "${TF:-FAIL}"
