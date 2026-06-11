#!/usr/bin/env bash
# Run the Rosenbrock precision benchmark (Neural Designer blog protocol) over
# N seeds for each engine and print per-run MSE + training time plus a summary.
#
#   usage:  run_precision.sh [runs]

set -u

# CPU-only protocol (the blog benchmark is CPU): hide any GPU from PyTorch/TF.
export CUDA_VISIBLE_DEVICES=""

RUNS="${1:-10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_PRECISION_BIN:-$HERE/opennn_precision}"

cd "$HERE"

[ -f rosenbrock.csv ] || "$PY" generate_rosenbrock.py

declare -A MSES TIMES

record() {  # label, seed, output, prediction file
    local label="$1" seed="$2" out="$3" pred="$4"
    local t mse
    t=$(echo "$out" | grep -oE 'train_time=[0-9.]+' | cut -d= -f2)
    mse=$("$PY" score.py "$label" "$pred" | grep -oE 'MSE=[0-9.]+' | cut -d= -f2)
    echo "$label seed=$seed mse=$mse time=${t}s"
    MSES[$label]="${MSES[$label]:-} $mse"
    TIMES[$label]="${TIMES[$label]:-} $t"
}

echo "############ precision benchmark ($RUNS runs each) ############"

for seed in $(seq 0 $((RUNS - 1))); do
    record "OpenNN-QNM" "$seed" "$("$OPENNN_BIN" "$seed" QuasiNewtonMethod 1000 2>/dev/null)" pred_opennn.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    record "OpenNN-LM" "$seed" "$("$OPENNN_BIN" "$seed" LevenbergMarquardt 1000 2>/dev/null)" pred_opennn.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    record "OpenNN-Adam" "$seed" "$("$OPENNN_BIN" "$seed" AdaptiveMomentEstimation 10000 2>/dev/null)" pred_opennn.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    record "PyTorch" "$seed" "$("$PY" pytorch_precision.py "$seed" 2>/dev/null)" pred_pytorch.txt
done

for seed in $(seq 0 $((RUNS - 1))); do
    record "TensorFlow" "$seed" "$("$PY" tensorflow_precision.py "$seed" 2>/dev/null)" pred_tensorflow.txt
done

echo
echo "############ summary ############"
for label in OpenNN-QNM OpenNN-LM OpenNN-Adam PyTorch TensorFlow; do
    "$PY" - "$label" "${MSES[$label]:-}" "${TIMES[$label]:-}" <<'EOF'
import sys
label = sys.argv[1]
mses = [float(v) for v in sys.argv[2].split()]
times = [float(v) for v in sys.argv[3].split()]
if mses:
    print(f"{label:12s} best_mse={min(mses):.4f} mean_mse={sum(mses)/len(mses):.4f} mean_time={sum(times)/len(times):.1f}s")
else:
    print(f"{label:12s} NO RESULTS")
EOF
done
