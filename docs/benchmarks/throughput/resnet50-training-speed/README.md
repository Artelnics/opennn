# ResNet-50 Training Speed

This folder contains the ResNet-50 training-speed harnesses. Treat this README as
the entrypoint. Large datasets and generated image folders must live outside the
repository; see [`../DATA_POLICY.md`](../../DATA_POLICY.md).

## Current Status

The CIFAR-geometry result is promising, but it is not a headline claim until it
has a fresh result JSON with repeated runs, quality metadata, and the optimized
PyTorch/TensorFlow paths in the same harness. The ImageNet-geometry work is a
caveat track: it tests whether the small-image launch-overhead win survives at
224x224 inputs.

## Canonical Runner

Prepare datasets under `OPENNN_BENCH_DATA`:

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python prepare_cifar10.py
python prepare_cifar100.py
python prepare_imagenet_like.py
```

`prepare_imagenet_like.py` creates a 224x224 ImageNet-geometry caveat dataset
from CIFAR-10 content. It is intentionally large and should stay in the external
cache.

Use the Python harness for new measurements:

```bash
python run_resnet50.py --dataset cifar10 --epochs 5 --batch 128 --runs 5 --precision both
```

By default, the harness reads `$OPENNN_BENCH_DATA/cifar10` or
`$OPENNN_BENCH_DATA/cifar100`. Use `--data-root` or `--data-dir` to override.

It compares:

- OpenNN with CUDA graph and GPU-resident data.
- PyTorch with channels-last, `torch.compile`, TF32, and optional bf16.
- TensorFlow with XLA and optional bf16.

Set `OPENNN_RESNET_BIN` if the OpenNN executable is not
`./opennn_resnet50_speed`.

## File Map

| File | Role |
|------|------|
| `run_resnet50.py` | Current 3-way harness; writes result JSON under `../../results/`. |
| `run_resnet50.sh` | Older shell harness kept for compatibility. |
| `prepare_cifar10.py`, `prepare_cifar100.py` | Dataset preparation for CIFAR geometry. |
| `prepare_imagenet_like.py`, `pytorch_resnet50_lazy.py` | ImageNet-geometry caveat track. |
| `pt_compile_probe.py`, `pt_conv_budget.py` | Diagnostic probes. |

## Before Citing

1. Run with an idle GPU and archive the generated JSON.
2. Confirm the result JSON includes git commit, machine metadata, commands, and
   raw output.
3. Add or cite the quality/convergence gate before presenting throughput as a
   training result.
4. Keep the CIFAR-vs-ImageNet geometry caveat explicit.
