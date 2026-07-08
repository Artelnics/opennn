# HIGGS dense max-batch (GPU and CPU) — OpenNN vs PyTorch vs TensorFlow

Capacity benchmark for the canonical HIGGS dense classifier
([`../../throughput/higgs/README.md`](../../throughput/higgs/README.md)): the
largest batch that completes one step on the same device within a fixed memory
cap, per framework, per precision (fp32, bf16), per mode:

- **train** — one full-batch training step: forward + backward + Adam update
- **infer** — one forward pass: no gradients, no optimizer state (OpenNN
  device-resident path `calculate_outputs_resident`, PyTorch
  `torch.inference_mode()`, TensorFlow `training=False`)

This is the dense capacity probe of the suite.

## Model and data

Every engine builds the identical canonical network:

| Item | Value |
|---|---|
| Network | 28 → hidden → hidden → 1 (default hidden 1024, 2 hidden layers) |
| Hidden activation | ReLU |
| Objective | binary cross-entropy (OpenNN: sigmoid output + CrossEntropy; PyTorch: BCEWithLogitsLoss; TF: BinaryCrossentropy(from_logits=True)) |
| Optimizer | Adam, learning rate 0.001 |
| Data | real prepared HIGGS rows via `--higgs-bin` (recommended), else synthetic with the contract shapes |

With `--higgs-bin` every engine reads the same float32 binary (rows × 29:
28 standardized features then the {0,1} label), and rows repeat modulo when
the candidate batch exceeds the file — the same convention as the ResNet-50
capacity runner repeating CIFAR-10. Prepare it once from a prepared HIGGS
CSV (see [`../../throughput/higgs/README.md`](../../throughput/higgs/README.md)):

```bash
python - <<'EOF'
import numpy as np, os
root = os.environ["OPENNN_BENCH_DATA"]
a = np.loadtxt(f"{root}/higgs/higgs_train.csv", delimiter=",", dtype=np.float32)
a.tofile(f"{root}/higgs/higgs_train_f32.bin")
EOF
```

Without `--higgs-bin` the trials fall back to synthetic contract-shaped data
(capacity depends on the shapes and the step, not the values). Either way,
quality-gated HIGGS numbers (accuracy / log loss / AUC on the real split) come
from the training-speed suite, not from this one.

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
  each trial process under a hard `RLIMIT_DATA` cap (`--mem-cap-gib`,
  default 8 — the same budget as the
  [data-capacity benchmark](../data-capacity/README.md)), so the out-of-memory
  boundary is deterministic instead of swap-dependent. `RLIMIT_DATA` charges
  brk + anonymous mmap (the tensor allocations) but not file-backed library
  mappings — PyTorch/TF map several GiB of runtime libraries, so an `RLIMIT_AS`
  address-space cap would measure code size, not data capacity. Trials report
  `peak_rss_mib` / `vm_peak_mib` for context. Linux only (kernel ≥ 4.7); on
  Windows use a Job Object wrapper. PyTorch/TF run with the GPU hidden.

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
python docs/benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow \
    --precisions fp32,bf16 --modes train,infer

# 3. CPU capacity (fp32, 8 GiB RLIMIT_DATA cap per trial; Linux only).
python docs/benchmarks/capacity/higgs-max-batch/run_higgs_maxbatch.py \
    --engines opennn,pytorch,tensorflow \
    --device cpu --modes train,infer --mem-cap-gib 8
```

The driver writes a JSON artifact to `docs/benchmarks/results/`
(`gpu-higgs-max-batch-<timestamp>.json` / `cpu-higgs-max-batch-<timestamp>.json`)
unless `--no-result-json` is passed.
