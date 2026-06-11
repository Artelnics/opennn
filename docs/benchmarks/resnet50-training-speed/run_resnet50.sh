#!/usr/bin/env bash
# Run the CIFAR-10 ResNet-50 GPU training-speed benchmark on both engines.
#
#   usage:  run_resnet50.sh [epochs] [batch]

set -u

EPOCHS="${1:-5}"
BATCH="${2:-128}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_RESNET_BIN:-$HERE/opennn_resnet50_speed}"
DATA_DIR="${CIFAR_DIR:-$HERE/cifar10}"
WSL_CUDA="/usr/lib/wsl/lib"

cd "$HERE"

[ -f "$DATA_DIR/cifar_images.npy" ] || "$PY" prepare_cifar10.py "$DATA_DIR"

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }

echo "############ CIFAR-10 ResNet-50 training-speed benchmark (GPU) ############"
echo "epochs=$EPOCHS batch=$BATCH"

echo "==== OpenNN (CUDA, fp32, GPU-resident data) ===="
o=$(OPENNN_GPU_RESIDENT_DATA=1 LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$DATA_DIR/train" "$EPOCHS" "$BATCH" fp32 2>/dev/null)
echo "$o" | grep -E "samples_per_sec|epoch_s|parameters|RESULT"
ONN=$(echo "$o" | extract samples_per_sec)

echo "==== PyTorch (CUDA, eager fp32, GPU-resident data) ===="
p=$("$PY" pytorch_resnet50_speed.py "$EPOCHS" "$BATCH" "$DATA_DIR" 2>/dev/null)
echo "$p" | grep -E "samples_per_sec|epoch_s|parameters|RESULT"
PT=$(echo "$p" | extract samples_per_sec)

echo
echo "############ summary (samples/sec, higher = faster) ############"
printf "%-22s %s\n" "OpenNN CUDA fp32:" "${ONN:-FAIL}"
printf "%-22s %s\n" "PyTorch eager fp32:" "${PT:-FAIL}"
