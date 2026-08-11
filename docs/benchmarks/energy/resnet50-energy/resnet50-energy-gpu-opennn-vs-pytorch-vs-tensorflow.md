# ResNet-50 fixed-work GPU energy: OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-11. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-resnet50-energy-20260811T131041Z.json`](../../results/).*

OpenNN spends the least GPU energy of the three engines to train ResNet-50 for
identical fixed work in both precisions: **1.32× less than PyTorch and 1.44×
less than TensorFlow in fp32** (1.35× / 1.13× in bf16).

## The result

Every engine trains the identical ResNet-50 v1.5 at CIFAR-10 geometry for
exactly **20 epochs** (50,000 images, batch 128, cross-entropy + Adam), each on
its fastest path (OpenNN cuDNN + CUDA graph, PyTorch channels_last +
`torch.compile` + TF32, TensorFlow XLA). GPU board power is sampled at 20 Hz
and trapezoid-integrated over each engine's `TRAIN_START`/`TRAIN_END` window
(the 2 warmup epochs and data loading stay outside). Median of 3 runs:

| Precision | Engine | Energy (J) | µJ/sample | Avg power | Train window |
|---|---|---:|---:|---:|---:|
| fp32 | **OpenNN** | **10,362** | **10,362** | 236 W | **43.9 s** |
| fp32 | PyTorch | 13,637 | 13,637 | 229 W | 59.6 s |
| fp32 | TensorFlow | 14,951 | 14,951 | 222 W | 67.2 s |
| bf16 | **OpenNN** | **7,249** | **7,249** | 225 W | **32.2 s** |
| bf16 | PyTorch | 9,759 | 9,759 | 208 W | 46.9 s |
| bf16 | TensorFlow | 8,200 | 8,200 | 168 W | 48.7 s |

| Comparison | fp32 | bf16 |
|---|---:|---:|
| PyTorch vs OpenNN | 1.32× more energy | 1.35× |
| TensorFlow vs OpenNN | 1.44× more energy | 1.13× |

Run-to-run dispersion is under 1% for every cell. All three engines hold the
same ~2,835-2,850 MHz SM clock; the energy differences are wall-time
differences (OpenNN finishes the same work 1.3-1.5× sooner), except
TensorFlow bf16, which runs longer at markedly lower power (168 W) and
partially compensates on joules while being 1.5× slower.

Versus the 2026-08-10 snapshot, OpenNN's bf16 window shrank 34.7 → 32.2 s —
the hybrid batch-norm forward (native bf16 tensor IO, no fp32 staging casts)
does the same work in fewer joules — which widened the bf16 lead from
1.24×/1.04× to 1.35×/1.13×.

## Caveats

* GPU board energy only (`nvidia-smi power.draw`, sampled, not a hardware
  joule counter); CPU/DRAM/PSU losses are excluded for every engine equally.
* Fixed work at CIFAR geometry: the small spatial size keeps launch overhead
  visible; a 224×224 run would shift all engines toward the conv-FLOP floor.
* Same fixed-work semantics as [`../higgs-dense-energy/`](../higgs-dense-energy/)
  and [`../transformer-energy/`](../transformer-energy/).

## Reproducing

```bash
cmake --build build --target opennn_resnet50_speed -j
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/throughput/resnet50/prepare_cifar10.py "$OPENNN_BENCH_DATA/cifar10"
python docs/benchmarks/energy/resnet50-energy/run_resnet50_energy.py \
    --epochs 20 --batch 128 --precision both --runs 3
```
