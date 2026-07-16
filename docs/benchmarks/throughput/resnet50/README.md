# ResNet-50 Speed (GPU)

This folder contains the ResNet-50 **training** and **inference** speed harnesses
(OpenNN vs PyTorch vs TensorFlow, fp32 and bf16). Treat this README as the
entrypoint. Large datasets and generated image folders must live outside the
repository; see [`../DATA_POLICY.md`](../../DATA_POLICY.md).

## Current Status

The CIFAR-geometry inference result now has a five-run canonical JSON with the
optimized OpenNN, PyTorch, and TensorFlow paths in the same harness. The
ImageNet-geometry work remains a caveat track: it tests whether the small-image
launch-overhead win survives at 224x224 inputs.

## Canonical Runner

Prepare datasets under `OPENNN_BENCH_DATA`. The prepare scripts write to
`$OPENNN_BENCH_DATA/<dataset>` (fallback `~/opennn-benchmark-data/<dataset>`)
so no dataset ever lands inside the benchmark folder:

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python prepare_cifar10.py        # -> $OPENNN_BENCH_DATA/cifar10
python prepare_cifar100.py       # -> $OPENNN_BENCH_DATA/cifar100
python prepare_imagenet_like.py  # -> $OPENNN_BENCH_DATA/imagenet_like
```

Each prepare script also accepts an explicit output directory as its first
positional argument to override the default.

`prepare_imagenet_like.py` creates a 224x224 ImageNet-geometry caveat dataset
from CIFAR-10 content. It is intentionally large and should stay in the external
cache.

Use the Python harness for new measurements:

```bash
python run_resnet50.py --dataset cifar10 --epochs 5 --batch 128 --runs 5 --precision both
```

By default, the harness reads `$OPENNN_BENCH_DATA/cifar10` or
`$OPENNN_BENCH_DATA/cifar100`. Use `--data-root <dir>` to change the dataset
root (default `$OPENNN_BENCH_DATA`, fallback `~/opennn-benchmark-data`), or
`--data-dir <dir>` to point directly at a dataset directory.

It compares:

- OpenNN with CUDA graph and GPU-resident data.
- PyTorch with channels-last, `torch.compile`, TF32, and optional bf16.
- TensorFlow with XLA and optional bf16.

Set `OPENNN_RESNET_BIN` if the OpenNN executable is not
`./opennn_resnet50_speed`.

## Inference — `gpu-resnet50-inference-speed`

Forward-only twin of the training benchmark, same CIFAR data:

```bash
python run_resnet50_infer.py --dataset cifar10 --batch 128 --runs 5 --precision both
```

Same `--data-root` / `--data-dir` flags as the training harness; by default it
reads `$OPENNN_BENCH_DATA/<dataset>`.

Writes `../../results/gpu-resnet50-inference-speed-<dataset>-<run_id>.json` with
`samples_per_sec` and `ms_per_batch`. OpenNN runs the device-resident forward
path; PyTorch uses `.eval()` + `no_grad()` (+ bf16 autocast); TensorFlow runs
`training=False` under XLA (+ bf16).

## File Map

| File | Role |
|------|------|
| `run_resnet50.py` | Current 3-way training harness; writes result JSON under `../../results/`. |
| `run_resnet50_infer.py` | Forward-only inference harness (`gpu-resnet50-inference-speed`). |
| `prepare_cifar10.py`, `prepare_cifar100.py` | Dataset preparation for CIFAR geometry. |
| `prepare_imagenet_like.py`, `pytorch_resnet50_lazy.py` | ImageNet-geometry caveat track. |

## Before Citing

1. Run with an idle GPU and archive the generated JSON.
2. Confirm the result JSON includes git commit, machine metadata, commands, and
   raw output.
3. Add or cite the quality/convergence gate before presenting throughput as a
   training result.
4. Keep the CIFAR-vs-ImageNet geometry caveat explicit.
