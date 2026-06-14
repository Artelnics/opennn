# GPU energy consumption: OpenNN vs PyTorch (dense MLP)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-14. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU (6 GB), CUDA 12.9, cuDNN 9.23.*

The [speed note](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) shows
OpenNN running the same dense network faster than PyTorch. Speed and energy are
not the same thing — a faster engine can finish sooner but draw more power while
it runs, so the energy bill is an independent question. This note measures it:
**how many joules each engine spends per sample**, for inference and for
training, on the same GPU and the same 1000 → 1000 (tanh) → 1 network.

## The result

GPU energy at batch 8000, fp32, integrated from 20 Hz `nvidia-smi` power samples
over a ≥8 s steady-state window (idle baseline 26.8 W subtracted to give the
workload's *active* energy):

| Workload | Engine | Avg power | Active power | **Energy / sample** | OpenNN advantage |
|---|---|---:|---:|---:|---|
| **Inference** | OpenNN | 97.3 W | 70.5 W | **20.4 µJ** | — |
| | PyTorch | 86.8 W | 60.0 W | 28.2 µJ | **1.44× less energy** |
| **Training** | OpenNN | 57.8 W | 31.0 W | **24.0 µJ** | — |
| | PyTorch | 80.2 W | 53.4 W | 58.2 µJ | **2.42× less energy** |

**OpenNN spends 1.44× less energy per inference and 2.42× less energy per
training sample** than PyTorch on this network. The two wins come from two
different mechanisms — which is the interesting part.

## Inference: more power, less energy

OpenNN draws *more* power during inference than PyTorch (97 W vs 87 W): its
device-resident inference path keeps the GPU busier, with fewer idle gaps. But it
finishes each sample so much faster that the **energy per sample is 28 % lower**.
Efficiency here is bought with speed, not with throttling — the energy ratio
(28.2 / 20.4 = 1.38×) tracks the throughput ratio (≈1.43×) almost exactly, which
is the signature of a workload whose energy is dominated by *how long the GPU
runs*, not *how hard*.

## Training: less power *and* less energy

Training is the larger and more surprising win. OpenNN draws **less** power than
PyTorch (58 W vs 80 W) *and* spends **2.4× less energy per sample**. PyTorch
holds the GPU at ~80 W for longer per sample; OpenNN's faster GEMMs and lower
sustained draw compound into under half the energy.

This win is not an artifact of a favorable batch schedule. The speed note shows
OpenNN's per-step throughput depends on how many mini-batches an epoch has (its
data-pipeline coordination is per-step). Measuring training energy at the *hard*
end of that range — 50 mini-batches per epoch, where OpenNN is slowest on raw
speed — it still spends **33.0 µJ/sample** versus PyTorch's 58.2, a **1.76×**
energy win. So across the whole regime OpenNN is between **1.76× and 2.42×** more
energy-efficient at training; the headline uses the 12-batch/epoch point, and the
floor (1.76×) is stated here so the number is not cherry-picked. The reason is
consistent: OpenNN trains at 46–58 W where PyTorch sits at 80 W.

## Setup

| | Value |
|---|---|
| Network | 1000 → 1000 (tanh) → 1, dense; MSE, Adam, fp32 |
| Inference | device-resident forward loop, batch 8000 |
| Training | resident dataset + CUDA-graph mega-launch (`OPENNN_GPU_RESIDENT_DATA=1 OPENNN_CUDA_GRAPH=1`), batch 8000 |
| Power source | `nvidia-smi --query-gpu=power.draw`, 20 Hz (`-lms 50`), trapezoidal integration |
| Idle baseline | 26.8 W (mean over 5 s idle), subtracted for *active* energy |
| Window | ≥ 8 s steady state per run (150–330 power samples each); warmup excluded |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, driver 555.42)
under WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 +
CUDA 12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124 wheels) on CPython 3.12.

## Caveats

* **This is GPU energy only.** The board's power sensor (`power.draw`) covers the
  GPU; CPU/system energy is *not* included. Intel RAPL (the CPU energy counter)
  is virtualized away under WSL2, so a whole-system figure is not available on
  this setup. For a GPU-bound workload the GPU is the dominant term, but the
  number is "GPU energy," not "wall energy."
* **Energy is integrated from sampled power, not read from a hardware counter.**
  This consumer GPU does not expose a cumulative joule counter
  (`total_energy_consumption` is unsupported), so energy is ∫power dt at 20 Hz.
  That is accurate over a multi-second window (hundreds of samples) and is
  applied identically to both engines; it would be less reliable for sub-second
  runs, which is why every run is sized to ≥ 8 s.
* **Active vs total energy.** The table's "energy/sample" is *active* (idle
  baseline removed) so it reflects the workload's marginal cost. Total-energy
  ratios are smaller because both engines share the same ~27 W idle floor; the
  raw totals are in the reproduction output.
* Single consumer laptop GPU under WSL2; absolute watts and the idle floor shift
  with hardware, driver, and power policy. The *ratios* are the portable result,
  and they follow directly from the speed and utilization differences in the
  [speed note](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md).

## Reproducing

The energy harness wraps any benchmark command, logs GPU power alongside it, and
integrates. It and the benchmark programs are in
[`docs/benchmarks/rosenbrock-max-batch/`](rosenbrock-max-batch/):

```bash
# energy_measure.sh <samples_processed> <label> -- <command...>

# Inference energy (samples = batch * iters)
ENERGY_IDLE_W=26.8 ./energy_measure.sh 40000000 opennn_infer -- \
  ./opennn_rosenbrock_resident_infer 8000 5000 1000 1000
ENERGY_IDLE_W=26.8 ./energy_measure.sh 28000000 pytorch_infer -- \
  python pytorch_rosenbrock_throughput.py inference 8000 3500 1000 1000

# Training energy (samples = dataset_size * epochs, or batch * iters for PyTorch)
ENERGY_IDLE_W=26.8 ./energy_measure.sh 10000000 opennn_train -- \
  env OPENNN_GPU_RESIDENT_DATA=1 OPENNN_CUDA_GRAPH=1 \
  ./opennn_rosenbrock_throughput train 100000 8000 100 1000 1000
ENERGY_IDLE_W=26.8 ./energy_measure.sh 12800000 pytorch_train -- \
  python pytorch_rosenbrock_throughput.py train 8000 1600 1000 1000
```

Measure your own idle baseline first (`ENERGY_IDLE_W`):

```bash
timeout 5 nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -lms 50 \
  | awk '{s+=$1;n++} END{printf "idle_W=%.2f\n", s/n}'
```
