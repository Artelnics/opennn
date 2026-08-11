# HIGGS dense max batch (GPU and CPU): OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-10, commit 52e21e15d. GPU: NVIDIA GeForce RTX 4080 (16 GB), driver 595.84. CPU: Intel Core i9-12900K under a hard 8 GiB `RLIMIT_DATA` cap. Artifacts: [`results/gpu-higgs-max-batch-20260810T125159Z.json`](../../results/), [`results/cpu-higgs-max-batch-20260810T211407Z.json`](../../results/).*

Largest batch of the canonical HIGGS dense classifier (28 → 1024 × 2 → 1) that
completes one full step — train: forward + backward + Adam; infer: forward —
inside the memory budget. Fresh process per candidate, exponential growth then
binary search, synthetic contract-shaped data.

## GPU result (VRAM cap: total − 512 MiB)

**OpenNN saturates the search ceiling of 2²⁶ = 67,108,864 samples in all four
cells without exhausting VRAM** — the recorded value is a censored lower bound,
not a boundary. PyTorch and TensorFlow hit genuine out-of-memory limits:

| Engine | fp32 train | fp32 infer | bf16 train | bf16 infer |
|---|---:|---:|---:|---:|
| **OpenNN** | **≥ 67,108,864** | **≥ 67,108,864** | **≥ 67,108,864** | **≥ 67,108,864** |
| PyTorch | 769,024 | 1,908,736 | 1,520,640 | 3,715,072 |
| TensorFlow | 869,376 | 1,730,560 | 1,001,472 | 1,718,272 |

OpenNN's peak VRAM at the ceiling is 1.3–8.6 GiB depending on the cell — the
streaming batch pipeline never materializes the whole dataset's activations at
once, so the ceiling is the search limit, ≥ **18× PyTorch's best cell and
≥ 39× TensorFlow's**.

## CPU result (hard 8 GiB `RLIMIT_DATA` per trial, fp32)

| Engine | train | infer |
|---|---:|---:|
| OpenNN | 131,072 | **60,948,480** |
| PyTorch | **367,616** | 750,592 |
| TensorFlow | 295,936 | 524,288 |

Two opposite stories, reported as measured:

* **Inference**: OpenNN's row-tiled resident path holds **81× PyTorch's and
  116× TensorFlow's** batch under the same cap — its memory ceiling is inputs
  plus outputs, not a full activation arena. 60.9M samples of 28 floats is the
  claim that used to live in the retired Windows *data-capacity* benchmark,
  now measured cross-platform under a deterministic cap.
* **Training**: OpenNN fits **0.36× PyTorch** (131k vs 368k). The training
  arena plans forward and delta storage together and its CPU path keeps the
  whole co-planned block under `RLIMIT_DATA`, while PyTorch's allocator grows
  and frees per-op. A capacity gap on the CPU training side, recorded as the
  honest current state.

## Setup

| | Value |
|---|---|
| Model | 28 → 1024 → 1024 → 1, ReLU, sigmoid, BCE, Adam |
| Data | synthetic, contract-shaped (28 float32 features); `--higgs-bin` switches to real rows |
| Search | fresh process per candidate; exponential then binary (`--min-step 1024`) |
| GPU budget | physical VRAM − 512 MiB reserve, peak via `nvidia-smi` at 20 Hz |
| CPU budget | `RLIMIT_DATA` 8 GiB (brk + anonymous mmap) per trial; peak = RSS |
| Execution | OpenNN batch pool 1, CUDA graph off (capacity, not speed) |

## Reproducing

```bash
cmake --build build --target opennn_higgs_maxbatch_trial -j
# GPU, 12 cells:
python docs/benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 --modes train,infer --min-step 1024
# CPU, 8 GiB cap:
python docs/benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --device cpu --modes train,infer --mem-cap-gib 8 --min-step 1024
```
