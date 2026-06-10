#!/usr/bin/env bash
# Run the three training-speed benchmarks at the article's config and print a
# comparison. Each engine gets the library paths it needs on WSL:
#   * OpenNN: the WSL GPU passthrough driver (/usr/lib/wsl/lib) must win over the
#     stale /usr/lib/x86_64-linux-gnu/libcuda.so.
#   * TensorFlow: its bundled nvidia-* CUDA libs must be on LD_LIBRARY_PATH.
#
#   usage:  run_speed.sh <csv_path> <samples> <features> [epochs] [batch]

set -u

CSV="${1:?csv path}"
SAMPLES="${2:-1000000}"
FEATURES="${3:-1000}"
EPOCHS="${4:-30}"
BATCH="${5:-1000}"

HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="$HOME/benchenv/bin/python"
OPENNN_BIN="$HOME/opennn-wsl/build-gpu/bin/opennn_speed"
WSL_CUDA="/usr/lib/wsl/lib"
TF_NV=$(find "$HOME/benchenv/lib/python3.12/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }

echo "############ training-speed benchmark ############"
echo "samples=$SAMPLES features=$FEATURES epochs=$EPOCHS batch=$BATCH"
echo

echo "==== OpenNN (CUDA, fp32) ===="
o_fp32=$(LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$CSV" "$FEATURES" "$EPOCHS" "$BATCH" fp32 2>/dev/null)
echo "$o_fp32" | grep -E 'samples_per_sec|median_epoch_s|RESULT'
ONN_FP32=$(echo "$o_fp32" | extract samples_per_sec)

echo "==== OpenNN (CUDA, bf16) ===="
o_bf16=$(LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$CSV" "$FEATURES" "$EPOCHS" "$BATCH" bf16 2>/dev/null)
echo "$o_bf16" | grep -E 'samples_per_sec|median_epoch_s|RESULT'
ONN_BF16=$(echo "$o_bf16" | extract samples_per_sec)

echo "==== PyTorch (CUDA, compile + AMP bf16 + TF32) ===="
p=$("$VENV" "$HERE/pytorch_speed.py" "$SAMPLES" "$FEATURES" "$EPOCHS" "$BATCH" 2>/dev/null)
echo "$p" | grep -E 'samples_per_sec|median_epoch_s|peak_vram_mb|RESULT'
PT=$(echo "$p" | extract samples_per_sec)

echo "==== TensorFlow (CUDA, XLA + mixed bf16) ===="
t=$(LD_LIBRARY_PATH="$TF_NV$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$VENV" "$HERE/tensorflow_speed.py" "$SAMPLES" "$FEATURES" "$EPOCHS" "$BATCH" 2>/dev/null)
echo "$t" | grep -E 'samples_per_sec|median_epoch_s|RESULT'
TF=$(echo "$t" | extract samples_per_sec)

echo
echo "############ summary (samples/sec, higher = faster) ############"
printf "%-28s %s\n" "OpenNN CUDA fp32:" "${ONN_FP32:-FAIL}"
printf "%-28s %s\n" "OpenNN CUDA bf16:" "${ONN_BF16:-FAIL}"
printf "%-28s %s\n" "PyTorch compile+AMP:" "${PT:-FAIL}"
printf "%-28s %s\n" "TensorFlow XLA+mixed:" "${TF:-FAIL}"
