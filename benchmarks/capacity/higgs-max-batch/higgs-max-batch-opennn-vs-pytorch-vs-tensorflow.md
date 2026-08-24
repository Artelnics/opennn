# HIGGS dense max batch (GPU and CPU): OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-11. GPU: NVIDIA GeForce RTX 4080 (16 GB), driver 595.84. CPU: Intel Core i9-12900K under a hard 8 GiB `RLIMIT_DATA` cap. Artifacts: [`results/gpu-higgs-max-batch-20260811T114222Z.json`](../../results/), [`results/cpu-higgs-max-batch-20260811T114022Z.json`](../../results/) (OpenNN cells; PyTorch/TensorFlow cells from the 2026-08-10 run — their trials are unchanged).*

Largest batch of the canonical HIGGS dense classifier (28 → 1024 × 2 → 1) that
completes one **monolithic** step — train: forward + backward + one Adam update
with activations O(batch); infer: one forward — inside the memory budget. The
same protocol in all three engines. Fresh process per candidate, exponential
growth then binary search, synthetic contract-shaped data.

> **2026-08-11 protocol correction.** Earlier revisions of this note reported
> OpenNN ceilings of ≥67M (GPU) and 60.9M (CPU inference). Those came from a
> different protocol: the OpenNN trial transparently split the logical batch
> into tiles (training via gradient accumulation, inference via row tiling), so
> its activations were O(tile) while PyTorch/TensorFlow ran the monolithic step
> with activations O(batch). The numbers were real but not comparable, and the
> tiling machinery existed only to serve this benchmark. It has been **removed
> from the library and the trial**; every cell below is the monolithic step,
> like-for-like across engines.

## GPU result (VRAM cap: total − 512 MiB)

| Engine | fp32 train | fp32 infer | bf16 train | bf16 infer |
|---|---:|---:|---:|---:|
| **OpenNN** | **961,536** | **1,914,880** | **1,921,024** | **3,828,736** |
| PyTorch | 769,024 | 1,908,736 | 1,520,640 | 3,715,072 |
| TensorFlow | 869,376 | 1,730,560 | 1,001,472 | 1,718,272 |
| OpenNN / PyTorch | 1.25× | 1.00× | 1.26× | 1.03× |
| OpenNN / TensorFlow | 1.11× | 1.11× | **1.92×** | **2.23×** |

OpenNN leads every cell. The margins are physically meaningful: inference
capacity is almost pure activation size, so OpenNN and PyTorch tie within 1%
(both saturate the same 15.9 GB frontier); in training OpenNN's co-planned
forward+delta arena fits ~25% more batch than PyTorch's per-op allocator, and
TensorFlow's bf16 cells trail well behind both. Peak VRAM at every OpenNN
frontier is 15.87 GB — a genuine out-of-memory boundary. The frontier carries a
few-MB run-to-run jitter from ambient VRAM (desktop compositor), so reproduce
via the runner's search, not single trials at the boundary.

## CPU result (hard 8 GiB `RLIMIT_DATA` per trial, fp32)

| Engine | train | infer |
|---|---:|---:|
| **OpenNN** | **488,448** | **980,992** |
| PyTorch | 367,616 | 750,592 |
| TensorFlow | 295,936 | 524,288 |
| OpenNN / PyTorch | 1.33× | 1.31× |
| OpenNN / TensorFlow | 1.65× | 1.87× |

The 2026-08-10 revision reported a training **loss** for OpenNN here (131,072,
0.36× PyTorch). That number was an artifact of the retired tiling machinery,
which capped the search at its own tile size; the monolithic step fits 3.7×
more. With the machinery gone, OpenNN wins both CPU cells.

## Setup

| | Value |
|---|---|
| Model | 28 → 1024 → 1024 → 1, ReLU, sigmoid, BCE, Adam |
| Step | monolithic in all engines: activations O(batch), one optimizer update |
| Data | synthetic, contract-shaped (28 float32 features); `--higgs-bin` switches to real rows |
| Search | fresh process per candidate; exponential then binary (`--min-step 1024`) |
| GPU budget | physical VRAM − 512 MiB reserve, peak via `nvidia-smi` at 20 Hz |
| CPU budget | `RLIMIT_DATA` 8 GiB (brk + anonymous mmap) per trial; peak = RSS |
| Execution | OpenNN batch pool 1, CUDA graph off (capacity, not speed) |

## Reproducing

```bash
cmake --build build --target opennn_higgs_maxbatch_trial -j
# GPU, 12 cells:
python benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 --modes train,infer --min-step 1024
# CPU, 8 GiB cap:
python benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --device cpu --modes train,infer --mem-cap-gib 8 --min-step 1024
```
