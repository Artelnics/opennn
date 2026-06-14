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
is data-driven.

The gap is **largest in training and at large widths**, where GEMM compute
dominates: a transformer's non-attention GEMMs are ~2/3 of its work, all of which
move from TF32 to bf16 tensor cores when you switch precision.

## Directional results (WSL2 RTX 3060, 2026-06-15)

Indicative only — **WSL2 degrades OpenNN's bf16 tensor-core path**, so these
understate bf16; native Windows gives larger margins. Re-run there for
investor-grade absolute numbers. bf16 wins **every** cell even under WSL2.

| workload | mode | fp32 | bf16 | bf16 speedup |
|---|---|---|---|---|
| Transformer (d512/6L/seq256) | inference | 82.8k tok/s | 163.8k tok/s | **1.98×** |
| Transformer (d512/6L/seq256) | training | 35.5 samp/s | 144.6 samp/s | **4.07×** |
| Dense MLP (8000×1000→4096) | inference | 2.17M samp/s | 2.95M samp/s | **1.36×** |
| Dense MLP (8000×1000→4096) | training | 97k samp/s | 134k samp/s | **1.38×** |
| Dense MLP (8000×1000→1000) | inference | 12.6M samp/s | 20.8M samp/s | **1.61×** |
| Dense MLP (8000×1000→1000) | training | 199k samp/s | 220k samp/s | **1.11×** |

**Measurement note:** the dense-inference cells need enough timed iterations
(use >=100). A short loop (~50 iters) is dominated by warmup and GPU-clock ramp
and gave a spurious bf16 < fp32 reading at small width; with 200 iters bf16
cleanly wins 1.61×. The harness uses a longer default for this reason.

**Takeaway:** bf16 is the right default for every compute-bound GPU workload.
The win is biggest where GEMM time dominates — transformer training **~4×**
(2/3 of the work is non-attention GEMMs that move from TF32 to bf16 tensor
cores), transformer inference **~2×** — and still solid for dense (**1.4–1.6×**).
It shrinks only when the workload is small/launch-bound (dense small-batch
training 1.11×), where there is little GEMM time to accelerate. **No workload
favors fp32 on throughput** — fp32 is for when you need the precision.

## Usage

```
# build the drivers first (per-benchmark build scripts):
#   ../attention-speed/build.sh  opennn_transformer_resident opennn_transformer_train
#   ../rosenbrock-max-batch/build_resident.sh ; build_tput.sh
# transformer training needs a corpus:
#   ../attention-speed/make_synthetic_corpus.py ../attention-speed/corpus_sweep.txt 10000 256 1024

python run_precision_sweep.py --runs 5
python run_precision_sweep.py --workloads transformer --modes inference
```

Writes `../results/gpu-precision-sweep-<run_id>.json` with per-cell median ±
stdev, the bf16 speedup, versions, commit, and GPU.
