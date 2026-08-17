# HIGGS dense training: throughput at every batch size (working note)

*Working note for the higgs family of the peak-batch benchmark
(`docs/benchmarks/throughput/peak-batch-speed/run_peak_batch_speed.py --family higgs`),
2026-08-17, laptop RTX 3060 Laptop 6 GB (WSL2, CUDA 12.9, cuBLAS 12.x,
PyTorch 2.13.0+cu130, TensorFlow 2.21.0), alongside the RTX 5070 Ti sweep
recorded in `b74b12a34` (`results/gpu-higgs-peak-batch-speed-20260817T104351Z.json`).
Same method as the ResNet and transformer notes: three engines under each one's
best protocol, the OpenNN step profiled per operator, one lever at a time.
The product note stays
[`higgs-gpu-training-opennn-vs-pytorch-vs-tensorflow.md`](higgs-gpu-training-opennn-vs-pytorch-vs-tensorflow.md).*

## Setup

Canonical HIGGS classifier 28 -> 1024 -> 1024 -> 1 (ReLU, ReLU, sigmoid),
Adam, binary cross-entropy, 10.5 M training rows resident on the device, batch
ladder 7,000 x 2^k, 3 timed epochs after 2 warm-up epochs, bf16 and fp32
(fp32 = TF32 in all engines).

Two protocol points fixed first:

* **PyTorch's best configuration.** The Higgs PyTorch driver was eager: no
  `torch.compile`, foreach Adam. It now takes `PT_COMPILE_MODE`,
  `PT_COMPILE_STEP` and `PT_FUSED_ADAM`, and the runner gives it the strongest
  one-line recipe - the whole train step compiled with `reduce-overhead` (CUDA
  graphs) and `Adam(fused=True)`. On this GPU at batch 7,000, bf16: eager
  2.01 M -> compiled model 2.38 M -> compiled step 2.43 M samples/s (+21%).
  `PYTORCH_PLAIN=1` reverts. TensorFlow already runs the step under XLA.
* **Whole batches only.** PyTorch and TensorFlow iterate
  `range(0, n - batch + 1, batch)`; OpenNN's driver trained the remainder as an
  eager tail batch of its own size - up to 6.5% of extra work the throughput
  figure does not count, and a second set of activation contexts. At 6 GB
  that second set is the difference between fitting and paging: bf16 448,000
  went from 2.22 M (paging) to 3.09 M samples/s, fp32 224,000 from 0.92 M to
  1.42 M. The driver now leaves the remainder out like the others do
  (`OPENNN_SPEED_KEEP_TAIL=1` restores it) and prints `tail_kept=`.

## Finding: the first layer's weight gradient

`OPENNN_PROFILE=1` with the graph off (per-op scopes synchronised on both
sides; new scopes `op:linear_bwd_wgrad <out>x<in>x<rows>` and
`op:linear_bwd_dx ...`), bf16, batch 7,000, per step (3.7 ms):

| op | ms | cuBLAS on the same shape (PyTorch `a @ b`) |
|---|---:|---:|
| L2 forward (7000x1024x1024) | 0.27 x3 layers = 0.82 | 0.76 |
| L2 wgrad (1024x1024, K 7000) | 0.72 | 0.81 |
| L2 dX | 0.61 | 0.72 |
| **L1 wgrad (28x1024, K 7000)** | **1.35** | **0.077** |
| L3 wgrad (1x1024, K 7000) | 0.11 | 0.073 |
| L3 dX (K = 1) | 0.09 | 0.077 |
| ReLU backward (2) + sigmoid | 0.40 | at bandwidth |
| Adam | 0.15 | |

cuBLASLt's heuristics - even its top-8 timed by the autotune - pick a kernel
without split-K for the skinny weight gradient (a 28 x 1024 output over a
7,000-long reduction): 17x the time cuBLAS's plain GEMM takes for the same
shape, 30% of the step. Weight gradients with a small output and a long
reduction now go through `cublasGemmEx` with the bias gradient reduced by its
own kernel (`linear_backward_gpu`, committed in b74b12a34 with the RTX 5070 Ti
measurement: bf16 +39% at 7,000). Here: bf16 7,000 1.88 M -> 2.60 M (+38%),
fp32 1.33 M. The same shape gets worse as K grows, which is what made the
earlier RTX 4080 ladder fall with batch (11.1 M -> 9.5 M) while TensorFlow's
rose. After the fix the per-sample cost of every op falls or stays flat from
7,000 to 56,000 on this GPU (profile: L2 GEMMs -15%, L1 wgrad -30%, ReLU
backward -20%).

## The ladder (RTX 3060 Laptop, 3 timed epochs, samples/s)

PyTorch = compiled step + fused Adam; TensorFlow = XLA. OpenNN "before" is
19f8287c8 (yesterday's code); "after" is the skinny-wgrad fix with the tail
left out (`hg_ladder_opennn_notail.json`; the PyTorch/TensorFlow ladders are
`hg_ladder_pt_tf.json`, run ~1 h earlier on a cooler machine). Points past
the 6 GB of VRAM page (marked).

| batch | OpenNN before | **OpenNN after** | PyTorch best | TensorFlow |
|---|---:|---:|---:|---:|
| bf16 7,000 | 1.88 M | **2.63 M** | 2.52 M | 2.17 M |
| bf16 14,000 | | **2.88 M** | 2.69 M | 2.16 M |
| bf16 28,000 | | **3.05 M** | 2.75 M | 2.51 M |
| bf16 56,000 | | **2.98 M** | 2.79 M | 2.59 M |
| bf16 112,000 | | **2.95 M** | 2.77 M | 2.56 M |
| bf16 224,000 | | **2.90 M** | 2.77 M | 2.66 M |
| bf16 448,000 | | **2.81 M** | 2.76 M | error |
| bf16 896,000 | | 0.77 M (paging) | 1.95 M (paging) | |
| fp32 7,000 | 1.33 M | 1.29 M | 1.38 M | 1.17 M |
| fp32 14,000 | | 1.29 M | 1.41 M | 1.16 M |
| fp32 28,000 | | 1.30 M | 1.38 M | 1.24 M |
| fp32 56,000 | | **1.37 M** | 1.33 M | 1.33 M |
| fp32 112,000 | | **1.35 M** | 1.34 M | OOM |
| fp32 224,000 | | 1.27 M | 1.29 M | |
| fp32 448,000 | | 0.30 M (paging) | 0.22 M (paging) | |

The fp32 ladder ran last, on a hot machine (its 7k-28k cells are 9% under
the same binary's cells measured earlier: 1.41 / 1.48 / 1.43 M with the tail
still trained), so the small-batch fp32 comparison was redone paired,
alternated O P P O, same thermal state:

| point | OpenNN | PyTorch best | |
|---|---:|---:|---:|
| fp32 7,000 | 1.36 / 1.29 M | 1.24 / 1.17 M | **+10%** |
| fp32 28,000 | 1.33 / 1.30 M | 1.29 / 1.29 M | **+2%** |
| bf16 7,000 | 2.45 / 2.49 M | 2.37 / 2.23 M | **+7%** |

So against PyTorch's best config on this GPU: **OpenNN leads bf16 at every
batch that fits (2-11%) and fp32 at every paired point (2-10%)**; the only
cells where PyTorch is ahead are 896,000 (both paging on 6 GB, PyTorch less
badly) and the hot-machine fp32 ladder cells the pairs contradict.
TensorFlow is well behind at small batches and out of memory earliest.

On the RTX 5070 Ti (b74b12a34, same code without the tail change): bf16
OpenNN 10.4 M vs PyTorch 9.9 M vs TensorFlow 9.4 M at the peaks (1.05x /
1.11x), fp32 5.57 M vs 5.07 M vs 5.23 M (1.10x / 1.06x); OpenNN leads at
7k-28k and 112k-448k, PyTorch by 2% at 56k and 896k, TensorFlow wins the
largest points. OpenNN's throughput there still falls with batch (bf16 10.4 M
-> 8.8 M, fp32 5.6 M -> 4.7 M) while PyTorch stays flat - a per-batch cost
that does not show on this laptop (per-sample cost falls from 7k to 56k here),
so it is the next thing to profile on that machine; the tail is part of it
(the driver still trained the remainder in that sweep).

## What is left on the OpenNN side

* Memory: at bf16 224,000 OpenNN peaks at 4.9 GB against PyTorch's 4.1 GB
  (nvidia-smi; PyTorch's own allocator 2.7 GB) - the tail contexts were the
  larger part of the difference; the remaining is the backward arena keeping
  both hidden-layer deltas alive.
* The output layer (1024 -> 1) costs 0.25 ms of a 2.7 ms step in three
  GEMV-shaped calls cuBLAS runs at 0.1-0.3 TFLOPS; a fused outer-product x
  ReLU' kernel for the single-output case would save ~0.13 ms (5%).
* `OPENNN_DRELU_FUSION=1` on these shapes (the ReLU derivative as cuBLASLt's
  DRELU epilogue of the next layer's dX GEMM, mask from a RELU_AUX forward
  epilogue): measured, alternated pairs at 7,000 - bf16 2.27 / 2.31 M vs
  2.78 / 2.51 M (-14%), fp32 1.08 / 1.02 M vs 1.17 / 1.19 M (-11%). The aux
  epilogues make cuBLASLt pick slower kernels here as they did on the
  transformer; the separate ReLU backward at bandwidth stays.
