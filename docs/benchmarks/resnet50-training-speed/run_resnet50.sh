#!/usr/bin/env bash
# Run the CIFAR ResNet-50 GPU training-speed benchmark on OpenNN and PyTorch.
#
#   usage:  run_resnet50.sh [epochs] [batch] [cifar10|cifar100]

set -u

EPOCHS="${1:-5}"
BATCH="${2:-128}"
DATASET="${3:-cifar10}"

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${BENCH_PYTHON:-$HOME/benchenv/bin/python}"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"
if [ -n "${OPENNN_RESNET_BIN:-}" ]; then
  OPENNN_BIN="$OPENNN_RESNET_BIN"
else
  OPENNN_BIN="$HERE/opennn_resnet50_speed"
  for candidate in \
    "$HERE/opennn_resnet50_speed" \
    "$REPO_ROOT/build-ninja/bin/opennn_resnet50_speed" \
    "$REPO_ROOT/build-ninja/bin/opennn_resnet50_speed.exe" \
    "$REPO_ROOT/build/bin/opennn_resnet50_speed" \
    "$REPO_ROOT/build/bin/opennn_resnet50_speed.exe"; do
    if [ -x "$candidate" ] || [ -f "$candidate" ]; then
      OPENNN_BIN="$candidate"
      break
    fi
  done
fi
DATA_DIR="${CIFAR_DIR:-$HERE/$DATASET}"
PREP="${PREP_SCRIPT:-prepare_$DATASET.py}"
WSL_CUDA="/usr/lib/wsl/lib"
RESULTS_DIR="${BENCH_RESULTS_DIR:-$HERE/../results}"
RUN_ID="${BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RESULT_FILE="${BENCH_RESULT_FILE:-$RESULTS_DIR/resnet50-training-speed-$DATASET-$RUN_ID.json}"

cd "$HERE"

[ -f "$DATA_DIR/cifar_images.npy" ] || "$PY" "$PREP" "$DATA_DIR"

extract() { grep -oE "$1=[0-9.]+" | head -1 | cut -d= -f2; }
ratio() { awk -v a="${1:-0}" -v b="${2:-0}" 'BEGIN { if (a > 0 && b > 0) printf "%.2f", a / b; }'; }

echo "############ $DATASET ResNet-50 training-speed benchmark (GPU) ############"
echo "epochs=$EPOCHS batch=$BATCH dataset=$DATASET"

echo "==== OpenNN (CUDA, fp32, GPU-resident data, CUDA graph) ===="
o=$(OPENNN_CUDA_GRAPH=1 OPENNN_GPU_RESIDENT_DATA=1 LD_LIBRARY_PATH="$WSL_CUDA:${LD_LIBRARY_PATH:-}" "$OPENNN_BIN" "$DATA_DIR/train" "$EPOCHS" "$BATCH" fp32 2>&1)
echo "$o" | grep -E "samples_per_sec|epoch_s|parameters|RESULT"
ONN=$(echo "$o" | extract samples_per_sec)

echo "==== PyTorch (CUDA, eager fp32, GPU-resident data) ===="
p=$("$PY" pytorch_resnet50_speed.py "$EPOCHS" "$BATCH" "$DATA_DIR" 2>&1)
echo "$p" | grep -E "samples_per_sec|epoch_s|parameters|RESULT"
PT=$(echo "$p" | extract samples_per_sec)

echo "==== PyTorch (CUDA, torch.compile fp32, GPU-resident data) ===="
pc=$("$PY" pt_compile_probe.py "$EPOCHS" "$BATCH" "$DATA_DIR" 2>&1)
echo "$pc" | grep -E "samples_per_sec|epoch_s|parameters|RESULT"
PTC=$(echo "$pc" | extract samples_per_sec)

echo
echo "############ summary (samples/sec, higher = faster) ############"
printf "%-26s %s\n" "OpenNN CUDA fp32:" "${ONN:-FAIL}"
printf "%-26s %s\n" "PyTorch compile fp32:" "${PTC:-FAIL}"
printf "%-26s %s\n" "PyTorch eager fp32:" "${PT:-FAIL}"
ONN_VS_COMPILE=$(ratio "$ONN" "$PTC")
ONN_VS_EAGER=$(ratio "$ONN" "$PT")
printf "%-26s %s\n" "OpenNN vs compile:" "${ONN_VS_COMPILE:+$ONN_VS_COMPILE x}"
printf "%-26s %s\n" "OpenNN vs eager:" "${ONN_VS_EAGER:+$ONN_VS_EAGER x}"

mkdir -p "$(dirname "$RESULT_FILE")"
GIT_COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)
OPENNN_OUTPUT="$o" PYTORCH_EAGER_OUTPUT="$p" PYTORCH_COMPILE_OUTPUT="$pc" \
RESULT_FILE="$RESULT_FILE" RUN_ID="$RUN_ID" GIT_COMMIT="$GIT_COMMIT" \
DATASET="$DATASET" EPOCHS="$EPOCHS" BATCH="$BATCH" ONN="$ONN" PT="$PT" PTC="$PTC" \
ONN_VS_COMPILE="$ONN_VS_COMPILE" ONN_VS_EAGER="$ONN_VS_EAGER" \
"$PY" - <<'PY'
import json
import os
import platform
import subprocess


def command_output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def number(name):
    value = os.environ.get(name, "")
    return float(value) if value else None


result = {
    "schema_version": 1,
    "benchmark_id": "gpu-resnet50-training",
    "run_id": os.environ["RUN_ID"],
    "git_commit": os.environ["GIT_COMMIT"],
    "dataset": os.environ["DATASET"],
    "configuration": {
        "epochs": int(os.environ["EPOCHS"]),
        "batch": int(os.environ["BATCH"]),
        "precision": "fp32",
        "workload": "ResNet-50 on CIFAR",
    },
    "machine": {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": command_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
    },
    "metrics": {
        "samples_per_sec": {
            "opennn_cuda_graph": number("ONN"),
            "pytorch_compile": number("PTC"),
            "pytorch_eager": number("PT"),
        },
        "speedup": {
            "opennn_vs_pytorch_compile": number("ONN_VS_COMPILE"),
            "opennn_vs_pytorch_eager": number("ONN_VS_EAGER"),
        },
    },
    "commands": {
        "opennn": "OPENNN_CUDA_GRAPH=1 OPENNN_GPU_RESIDENT_DATA=1 opennn_resnet50_speed",
        "pytorch_eager": "python pytorch_resnet50_speed.py",
        "pytorch_compile": "python pt_compile_probe.py",
    },
    "raw_output": {
        "opennn": os.environ.get("OPENNN_OUTPUT", ""),
        "pytorch_eager": os.environ.get("PYTORCH_EAGER_OUTPUT", ""),
        "pytorch_compile": os.environ.get("PYTORCH_COMPILE_OUTPUT", ""),
    },
}

with open(os.environ["RESULT_FILE"], "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
    handle.write("\n")
PY

echo "result_json=$RESULT_FILE"
