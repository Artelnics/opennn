# fp32 vs bf16 precision sweep (GPU)

How much does bf16 buy over fp32 for each OpenNN GPU workload, in **inference**
and in **training**? `run_precision_sweep.py` runs each (workload, mode,
precision) cell N times and reports median ± stdev plus the bf16/fp32 speedup.

## Why fp32 is slower (it is not a bug)

Under `Type::FP32` the dense / FFN / projection GEMMs run **TF32**
(`CUBLAS_COMPUTE_DTYPE`) and attention runs through the **fp32-via-bf16** cast
path (Q/K/V cast down to bf16, cuDNN flash-attention in bf16, output cast back).
Under `Type::BF16` every GEMM runs the **bf16 tensor-core** path
(`CUBLAS_COMPUTE_32F_FAST_16BF`). So fp32 is doing genuinely more precise — and
more expensive — math. The sweep quantifies that trade so the precision choice
is data-driven. The gap is largest in training and at large widths, where GEMM
compute dominates.

## Usage

```bash
# build the drivers first (per-benchmark build scripts):
#   ../attention-speed/build.sh  opennn_transformer_resident opennn_transformer_train
#   ../rosenbrock-max-batch/build_resident.sh ; build_tput.sh
# transformer training needs a corpus:
#   ../attention-speed/make_synthetic_corpus.py ../attention-speed/corpus_sweep.txt 10000 256 1024

python run_precision_sweep.py --runs 5
python run_precision_sweep.py --workloads transformer --modes inference
```

Writes `../../results/gpu-precision-sweep-<run_id>.json` with per-cell median ±
stdev, the bf16 speedup, versions, commit, and GPU.

**Measurement note:** the dense-inference cells need enough timed iterations
(use ≥100). A short loop is dominated by warmup and GPU-clock ramp and can give a
spurious bf16 < fp32 reading at small width; the harness uses a longer default
for this reason.
