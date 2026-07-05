# HIGGS dense max-batch (GPU and CPU) — OpenNN vs PyTorch vs TensorFlow

Capacity benchmark for the canonical HIGGS dense classifier
([`../higgs/README.md`](../higgs/README.md)): the largest batch that completes
one step on the same device within a fixed memory cap, per framework, per
precision (fp32, bf16), per mode:

- **train** — one full-batch training step: forward + backward + Adam update
- **infer** — one forward pass: no gradients, no optimizer state (OpenNN
  device-resident path `calculate_outputs_resident`, PyTorch
  `torch.inference_mode()`, TensorFlow `training=False`)

This suite replaces the historical Rosenbrock dense max-batch probe
([`../rosenbrock-max-batch/`](../rosenbrock-max-batch/)) per the
[HIGGS migration plan](../DENSE_HIGGS_MIGRATION.md).

## Model and data

Every engine builds the identical canonical network:

| Item | Value |
|---|---|
| Network | 28 → hidden → hidden → 1 (default hidden 1024, 2 hidden layers) |
| Hidden activation | ReLU |
| Objective | binary cross-entropy (OpenNN: sigmoid output + CrossEntropy; PyTorch: BCEWithLogitsLoss; TF: BinaryCrossentropy(from_logits=True)) |
| Optimizer | Adam, learning rate 0.001 |
| Data | synthetic with the HIGGS contract shapes (28 standardized features, {0,1} label) |

The data is synthetic on purpose: capacity depends on the shapes and the
training/inference step, not on the feature values, and a fresh process per
candidate would otherwise re-parse a multi-gigabyte CSV dozens of times per
search. Quality-gated HIGGS numbers (accuracy / log loss / AUC on the real
split) come from the training-speed suite, not from this one.

## Protocol

- Fresh process per batch candidate (a CUDA OOM can leave the context with a
  sticky error, and allocator state must not leak between candidates).
- Exponential growth then binary search; the artifact records the largest
  passing batch and, when probed, the next failing batch.
- A candidate passes only if the process exits 0, prints `RESULT=OK`, reports
  finite loss/outputs, and its observed `nvidia-smi` peak stays under
  `total VRAM - reserve` (default reserve 512 MiB). The cap matters: under
  WSL2 the driver silently spills GPU allocations into system RAM, which
  would otherwise report meaningless oversized batches.
- OpenNN runs with prefetch-pool depth 1 (`set_batch_pool_size(1)`; the
  default pool of 3 holds extra device batch copies and is a throughput
  feature, not a capacity one) and CUDA graph off.
- **CPU mode** (`--device cpu`, fp32 only): the same matrix on the CPU, with
  each trial process under a hard `RLIMIT_AS` address-space cap
  (`--mem-cap-gib`, default 8 — the same budget as the published
  [data-capacity benchmark](../data-capacity-opennn-vs-pytorch-vs-tensorflow.md)),
  so the out-of-memory boundary is deterministic instead of swap-dependent.
  Linux only; on Windows use a Job Object wrapper as in
  [`../capacity/`](../capacity/). PyTorch/TF run with the GPU hidden.

## Files

| File | Purpose |
|---|---|
| `opennn_higgs_maxbatch_trial.cpp` | OpenNN trial: one (mode, batch, precision) attempt |
| `pytorch_higgs_maxbatch.py` | PyTorch counterpart (TF32, fused Adam, autocast bf16, optional `PT_COMPILE=1`) |
| `tensorflow_higgs_maxbatch.py` | TensorFlow counterpart (graph mode, XLA off, `mixed_bfloat16`) |
| `run_higgs_maxbatch.py` | Driver: fresh-process exponential + binary search, VRAM cap, JSON artifact |

## How to run

```bash
# 1. Build the OpenNN trial (a benchmarks target).
cmake -S . -B build -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build build --target opennn_higgs_maxbatch_trial -j

# 2. Run the full comparison (torch + TF live in the ml venv; see BENCH_PYTHON).
python docs/benchmarks/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow \
    --precisions fp32,bf16 --modes train,infer

# 3. CPU capacity (fp32, 8 GiB RLIMIT_AS cap per trial; Linux only).
python docs/benchmarks/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow \
    --device cpu --modes train,infer --mem-cap-gib 8
```

The driver writes a JSON artifact to `docs/benchmarks/results/`
(`gpu-higgs-max-batch-<timestamp>.json` / `cpu-higgs-max-batch-<timestamp>.json`)
unless `--no-result-json` is passed.

## Status

Harness ready; not yet measured. Run on the reference GPU and archive the
result JSON before quoting any number.
