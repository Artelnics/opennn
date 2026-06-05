#!/usr/bin/env bash
# ----------------------------------------------------------------------------
# Profile the dense-MLP backward GEMMs to decide whether dual-stream
# (delta vs. gradient) overlap can speed up training.
#
# The single question this answers: at your batch size, are the backward
# GEMMs leaving GPU capacity idle? If yes -> dual-stream can help. If the
# GPU is already saturated -> it cannot, and you should not build it.
#
# Usage:
#   1. Set TRAIN_CMD below to the command that runs your 4x300 MLP benchmark.
#   2. ./profile_backward_gemm.sh
#   3. Read the three numbers (see DECISION RULE at the bottom).
# ----------------------------------------------------------------------------

set -euo pipefail

# The HIGGS 5x300 SGD GPU benchmark (blank_cuda). Adjust the path to match
# your CUDA build directory if different.
TRAIN_CMD="./build_cuda/bin/blank_cuda"

OUT_DIR="profile_out"
mkdir -p "$OUT_DIR"

# ----------------------------------------------------------------------------
# PASS 1 - Nsight Systems: timeline + kernel names + H2D-overlap check
#   Look for: (a) backward GEMM kernel names (ampere_sgemm_* / cutlass_* /
#   *gemm*tf32*), (b) are kernels back-to-back or gappy (gaps => launch-bound,
#   graphs help; back-to-back => GPU-bound, occupancy is the question),
#   (c) is the next batch's HtoD copy overlapping compute or sitting in a
#   serial gap (a gap => a COPY STREAM is the better lever).
# ----------------------------------------------------------------------------
echo "=== PASS 1: Nsight Systems timeline ==="
nsys profile \
  -o "$OUT_DIR/opennn_train" \
  --stats=true \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  $TRAIN_CMD

echo
echo "Open $OUT_DIR/opennn_train.nsys-rep in the GUI, or read the stats above."
echo "Note the backward GEMM kernel name(s) for PASS 2 if you want to target them."
echo

# ----------------------------------------------------------------------------
# PASS 2 - Nsight Compute: the decisive occupancy / Tensor Core numbers.
#   --launch-skip 50 skips warmup so we measure a steady-state step.
#   --launch-count 4 grabs a few launches (covers fwd+bwd GEMMs).
# ----------------------------------------------------------------------------
echo "=== PASS 2: Nsight Compute occupancy / Tensor Core utilization ==="
ncu \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active,sm__warps_active.avg.pct_of_peak_sustained_active \
  --launch-skip 50 \
  --launch-count 4 \
  --kernel-name-base demangled \
  -o "$OUT_DIR/opennn_gemm_profile" \
  -f \
  $TRAIN_CMD

echo
echo "============================================================================"
echo "DECISION RULE  (read the per-kernel values for the BACKWARD GEMMs):"
echo
echo "  sm__pipe_tensor_op_hmma...  = Tensor Core utilization  (MOST IMPORTANT;"
echo "                                 your TF32 GEMM is Tensor-Core bound)"
echo "  sm__throughput...           = overall SM throughput"
echo "  sm__warps_active...         = achieved occupancy"
echo
echo "  Tensor Core util OR SM throughput >= 80%  -> GPU SATURATED."
echo "      Dual-stream delta/gradient ~= 0%. DO NOT build it."
echo "      If a stream win is wanted, build an H2D COPY STREAM instead."
echo
echo "  Both in 50-80%   -> moderate headroom. Dual-stream maybe 5-15%. Prototype."
echo
echo "  Both < 50%       -> underutilized. Dual-stream could give 20%+. Worth it."
echo
echo "  ALSO from PASS 1: if there is a serial HtoD gap where the GPU waits for"
echo "  the next batch, a COPY STREAM is the better lever regardless of the above."
echo "============================================================================"
