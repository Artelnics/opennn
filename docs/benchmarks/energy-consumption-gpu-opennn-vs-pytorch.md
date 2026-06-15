# GPU energy consumption: OpenNN vs PyTorch vs TensorFlow (dense MLP)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-14. Linux x86_64 (WSL2), NVIDIA RTX 3060 Laptop GPU (6 GB), CUDA 12.9, cuDNN 9.23.*

**Status:** current GPU energy result. It is a **sampled-power estimate**, not a
hardware joule counter: energy is the time-integral of NVIDIA `power.draw`
sampled at 20 Hz (GPU-board only, not whole-system wall power). Read the
**ratios** between engines as the result, not the absolute watts — those shift
with hardware, driver, and power policy. Every number is the **median of 5 runs
(± population stdev)**; the raw per-run data, versions, power traces, and the
**per-run GPU state** (SM/memory clocks, temperature, power limit, and active
throttle reasons before/after each run, so a reviewer can rule out thermal or
power-limit artifacts behind the watt figures) are in
[`results/`](results/) (`gpu-dense-rosenbrock-energy-*.json`).

The [speed note](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) shows
OpenNN running the same dense network faster than PyTorch. Speed and energy are
not the same thing — a faster engine can finish sooner but draw more power while
it runs, so the energy bill is an independent question. This note measures it:
**how many joules each engine spends per sample**, for inference and for
training, on the same GPU and the same 1000 → 1000 (tanh) → 1 network, against
**both** PyTorch and TensorFlow running their fair fast paths.

## The result

GPU energy per sample at batch 8000, fp32, integrated from 20 Hz `nvidia-smi`
power samples over a steady-state loop of 2000 iterations (so each run is many
seconds; the idle baseline of 27.2 W is subtracted to give the workload's
*active* energy). All three engines run the **identical workload** — one
forward (inference) or forward+backward+Adam step (training) on a fixed batch,
repeated — so energy per sample is apples-to-apples. PyTorch and TensorFlow run
their compiled fast paths (`torch.compile` / XLA `jit_compile`).

**Inference — energy per sample (µJ), lower is better:**

| Engine | Total energy | Active energy | Avg power | vs OpenNN (total) |
|---|---:|---:|---:|---|
| **OpenNN** | **25.9 ± 2.8** | 17.9 ± 2.1 | 88 W | — |
| TensorFlow (XLA) | 29.8 ± 3.0 | 18.3 ± 2.1 | 68 W | OpenNN **1.15× less** |
| PyTorch (compile) | 43.5 ± 2.2 | 30.6 ± 1.3 | 91 W | OpenNN **1.68× less** |

**Training — energy per sample (µJ), lower is better:**

| Engine | Total energy | Active energy | Avg power | vs OpenNN (total) |
|---|---:|---:|---:|---|
| TensorFlow (XLA) | **60.5 ± 3.5** | 38.6 ± 2.5 | 74 W | OpenNN 0.93× (TF lower) |
| **OpenNN** | **64.9 ± 3.9** | 46.1 ± 2.8 | 92 W | — |
| PyTorch (compile) | 89.1 ± 3.2 | 62.7 ± 2.7 | 91 W | OpenNN **1.37× less** |

**OpenNN spends the least energy per inference of the three — 1.15× less than
TensorFlow and 1.68× less than PyTorch.** For training, OpenNN spends **1.37×
less energy than PyTorch**, but here **TensorFlow's XLA path is the most
energy-efficient** (≈7 % below OpenNN), because TF holds the GPU at a markedly
lower average power (74 W vs 92 W). We report that honestly: OpenNN is the
inference-energy leader and beats PyTorch on both, and TF's compiled training
step is the one place a competitor edges it.

## Why the picture differs between inference and training

The energy bill is power × time, and the three engines trade those off
differently:

* **OpenNN runs the GPU hot and short.** It sustains the highest average power
  (88–92 W) but finishes each sample fastest, so on inference its short runtime
  wins outright. Its energy tracks its speed lead, the signature of a
  runtime-dominated workload.
* **TensorFlow runs the GPU cooler.** Its XLA-compiled step sits at 68–74 W —
  ~20 % below OpenNN/PyTorch. On inference that isn't enough to overcome
  OpenNN's speed; on training, where the step is heavier, the lower sustained
  power makes TF the energy leader despite not being the fastest.
* **PyTorch draws high power without the matching speed**, so it is the least
  energy-efficient of the three on both workloads even with `torch.compile`.

## Setup

| | Value |
|---|---|
| Network | 1000 → 1000 (tanh) → 1, dense; MSE, Adam, fp32 |
| Workload | identical across engines: fixed batch 8000, 2000 steps, warmup excluded |
| OpenNN | device-resident; training uses `OPENNN_GPU_RESIDENT_DATA=1 OPENNN_CUDA_GRAPH=1` |
| PyTorch | `torch.compile`, GPU-resident tensors |
| TensorFlow | `@tf.function(jit_compile=True)` (XLA), GPU-resident tensors |
| Power source | `nvidia-smi --query-gpu=power.draw`, 20 Hz (`-lms 50`), trapezoidal integration |
| Idle baseline | 27.2 W (measured fresh at start), subtracted for *active* energy |
| Statistics | median of 5 runs, ± population stdev; raw runs in `results/*.json` |

Hardware/software: NVIDIA GeForce RTX 3060 Laptop GPU (6 GB, driver 555.85)
under WSL2 Ubuntu 24.04 on Windows 11 (i7-12700H). OpenNN built with g++ 13.3 +
CUDA 12.9.86 + cuDNN 9.23; PyTorch 2.6.0 (cu124), TensorFlow 2.21.0, CPython 3.12.

## Caveats

* **This is GPU energy only.** The board power sensor (`power.draw`) covers the
  GPU; CPU/system energy is *not* included (Intel RAPL is virtualized away under
  WSL2). For a GPU-bound workload the GPU is the dominant term, but the number is
  "GPU energy," not "wall energy." A whole-system claim needs a wall-power meter.
* **Energy is integrated from sampled power, not a hardware joule counter.** This
  consumer GPU does not expose `total_energy_consumption`, so energy is ∫power dt
  at 20 Hz — accurate over a multi-second window (hundreds of samples) and
  applied identically to all three engines.
* **Active vs total energy.** The headline uses *total* energy (it includes the
  shared idle floor, so it is the conservative framing). *Active* energy (idle
  removed) is also reported; on it the OpenNN/TF inference gap narrows to a tie
  while PyTorch remains the outlier.
* **Run-to-run variance is real** (±3–4 µJ on a noisy 20 Hz signal and a laptop
  GPU under thermal variation), which is why every figure is a 5-run median with
  its stdev. The *ranking* is stable across runs; the exact ratios shift slightly.
* Single consumer laptop GPU under WSL2; absolute watts and the idle floor shift
  with hardware, driver, and power policy.

## Reproducing

The energy harness runs all three engines on the identical workload, samples GPU
power while each runs, integrates, repeats N times, and writes a result JSON. It
and the benchmark programs are in
[`rosenbrock-max-batch/`](rosenbrock-max-batch/):

```bash
cd docs/benchmarks/rosenbrock-max-batch
# build the OpenNN binaries first (build_tput.sh / build_resident.sh)
python run_energy.py --mode both --batch 8000 --iters 2000 --runs 5
# -> writes ../results/gpu-dense-rosenbrock-energy-<timestamp>.json
# engines: --engines opennn,pytorch,tensorflow   idle override: --idle 27.2
```
