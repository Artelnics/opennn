#!/usr/bin/env bash
# Run the CIFAR ResNet-50 GPU training-speed benchmark on both engines.
#
#   usage:  run_resnet50.sh [epochs] [batch] [cifar10|cifar100]

set -u

EPOCHS="${1:-5}"
BATCH="${2:-128}"
DATASET="${3:-cifar10}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
OPENNN_BIN="${OPENNN_RESNET_BIN:-$HERE/opennn_resnet50_speed}"
DATA_DIR="${CIFAR_DIR:-$HERE/$DATASET}"
PREP="${PREP_SCRIPT:-prepare_$DATASET.py}"
WSL_CUDA="/usr/lib/wsl/lib"

cd "$HERE"

[ -f "$DATA_DIR/cifar_images.npy" ] || "$PY" "$PREP" "$DATA_DIR"

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }

echo "############ $DATASET ResNet-50 training-speed benchmark (GPU) ############"
echo "epochs=$EPOCHS batch=$BATCH dataset=$DATASET"

echo "==== OpenNN (CUDA, fp32, GPU-resident data, CUDA graph) ===="
o=$(OPENNN_CUDA_GRAPH=1 OPENNN_GPU_RESIDENT_DATA=1 LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$DATA_DIR/train" "$EPOCHS" "$BATCH" fp32 2>/dev/null)
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
