#!/usr/bin/env bash
# Run the MNIST CNN GPU training-speed benchmark on all three engines and
# print a summary. The WSL GPU passthrough driver must win over any stale
# libcuda; TensorFlow needs its bundled nvidia-* CUDA libs on LD_LIBRARY_PATH.
#
#   usage:  run_cnn_speed.sh [epochs] [batch]

set -u

EPOCHS="${1:-10}"
BATCH="${2:-128}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_CNN_BIN:-$HERE/opennn_cnn_speed}"
MNIST_DIR="${MNIST_DIR:-$HERE/../../../examples/mnist/data}"
WSL_CUDA="/usr/lib/wsl/lib"
TF_NV=$(find "$(dirname "$PY")/../lib" -path "*nvidia*" -name lib -type d 2>/dev/null | tr '\n' ':')

cd "$HERE"

[ -f mnist_images.npy ] || "$PY" prepare_mnist.py "$MNIST_DIR"

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }

echo "############ MNIST CNN training-speed benchmark (GPU) ############"
echo "epochs=$EPOCHS batch=$BATCH"

echo "==== OpenNN (CUDA, fp32, GPU-resident data) ===="
# OPENNN_GPU_RESIDENT_DATA keeps the dataset on the GPU and gathers batches
# device-side — the analogue of the GPU-resident tensors in the PyTorch script.
o=$(OPENNN_GPU_RESIDENT_DATA=1 LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$MNIST_DIR" "$EPOCHS" "$BATCH" fp32 2>/dev/null)
echo "$o" | grep -E "samples_per_sec|epoch_s|RESULT"
ONN=$(echo "$o" | extract samples_per_sec)

echo "==== PyTorch (CUDA, eager fp32) ===="
p=$("$PY" pytorch_cnn_speed.py "$EPOCHS" "$BATCH" 2>/dev/null)
echo "$p" | grep -E "samples_per_sec|epoch_s|RESULT"
PT=$(echo "$p" | extract samples_per_sec)

echo "==== TensorFlow (CUDA, Keras fit fp32) ===="
t=$(LD_LIBRARY_PATH="$TF_NV$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$PY" tensorflow_cnn_speed.py "$EPOCHS" "$BATCH" 2>/dev/null)
echo "$t" | grep -E "samples_per_sec|epoch_s|RESULT"
TF=$(echo "$t" | extract samples_per_sec)

echo
echo "############ summary (samples/sec, higher = faster) ############"
printf "%-22s %s\n" "OpenNN CUDA fp32:" "${ONN:-FAIL}"
printf "%-22s %s\n" "PyTorch eager fp32:" "${PT:-FAIL}"
printf "%-22s %s\n" "TensorFlow fit fp32:" "${TF:-FAIL}"
