#!/usr/bin/env bash
# Run the Rosenbrock accuracy-parity benchmark over N seeds for each engine and
# print per-run MSE / R^2 (from the neutral scorer) plus a mean summary. Same
# network, data, optimizer, and epochs in all three; the single score.py is the
# only thing that computes the metric, so no framework's internal reduction can
# bias the result.
#
#   usage:  run_accuracy.sh [runs]

set -u

# Controlled-regression protocol is CPU-side; hide any GPU for determinism.
export CUDA_VISIBLE_DEVICES=""

RUNS="${1:-5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_ACCURACY_BIN:-$HERE/opennn_accuracy}"

cd "$HERE"

# Generate the shared train/test split once if absent.
[ -f rosenbrock_train.csv ] || "$PY" generate_rosenbrock.py

declare -A R2S

record() {  # label, seed, predictions file
    local label="$1" seed="$2" pred="$3"
    local line r2
    line=$("$PY" score.py "$label" "$pred")
    r2=$(echo "$line" | grep -oE 'R2=[-0-9.]+' | cut -d= -f2)
    echo "$label seed=$seed $line"
    R2S[$label]="${R2S[$label]:-} $r2"
}

echo "############ accuracy-parity benchmark ($RUNS seeds each) ############"

for seed in $(seq 0 $((RUNS - 1))); do
    "$OPENNN_BIN" "$seed" >/dev/null 2>&1
    record "OpenNN" "$seed" pred_opennn.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    "$PY" pytorch_accuracy.py "$seed" >/dev/null 2>&1
    record "PyTorch" "$seed" pred_pytorch.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    "$PY" tensorflow_accuracy.py "$seed" >/dev/null 2>&1
    record "TensorFlow" "$seed" pred_tensorflow.txt
done

echo
echo "############ summary (mean R^2 over $RUNS seeds) ############"
for label in OpenNN PyTorch TensorFlow; do
    "$PY" - "$label" "${R2S[$label]:-}" <<'EOF'
import sys
label = sys.argv[1]
r2s = [float(v) for v in sys.argv[2].split()]
if r2s:
    print(f"{label:12s} mean_R2={sum(r2s)/len(r2s):.4f} min={min(r2s):.4f} max={max(r2s):.4f} n={len(r2s)}")
else:
    print(f"{label:12s} NO RESULTS")
EOF
done
