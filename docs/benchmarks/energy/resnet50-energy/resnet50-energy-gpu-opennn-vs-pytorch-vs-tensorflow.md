# ResNet-50 fixed-work GPU energy: OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-10. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-resnet50-energy-20260810T142601Z.json`](../../results/).*

OpenNN spends the least GPU energy of the three engines to train ResNet-50 for
identical fixed work in both precisions: **1.32× less than PyTorch and 1.41×
less than TensorFlow in fp32** (1.24× / 1.04× in bf16).

## The result

Every engine trains the identical ResNet-50 v1.5 at CIFAR-10 geometry for
exactly **20 epochs** (50,000 images, batch 128, cross-entropy + Adam), each on
its fastest path (OpenNN cuDNN + CUDA graph, PyTorch channels_last +
`torch.compile` + TF32, TensorFlow XLA). GPU board power is sampled at 20 Hz
and trapezoid-integrated over each engine's `TRAIN_START`/`TRAIN_END` window
(the 2 warmup epochs and data loading stay outside). Median of 3 runs:

| Precision | Engine | Energy (J) | µJ/sample | Avg power | Train window |
|---|---|---:|---:|---:|---:|
| fp32 | **OpenNN** | **10,472** | **10,472** | 235 W | **44.6 s** |
| fp32 | PyTorch | 13,819 | 13,819 | 231 W | 59.8 s |
| fp32 | TensorFlow | 14,807 | 14,807 | 223 W | 66.6 s |
| bf16 | **OpenNN** | **7,915** | **7,915** | 229 W | **34.7 s** |
| bf16 | PyTorch | 9,810 | 9,810 | 210 W | 46.8 s |
| bf16 | TensorFlow | 8,212 | 8,212 | 169 W | 48.4 s |

| Comparison | fp32 | bf16 |
|---|---:|---:|
| PyTorch vs OpenNN | 1.32× more energy | 1.24× |
| TensorFlow vs OpenNN | 1.41× more energy | 1.04× |

Run-to-run dispersion is under 1% for every cell. All three engines hold the
same ~2,835-2,850 MHz SM clock; the energy differences are wall-time
differences (OpenNN finishes the same work 1.3-1.4× sooner), except
TensorFlow bf16, which runs longer at markedly lower power (169 W) and lands
close to OpenNN on joules while being 1.4× slower.

> The OpenNN figures use the current, numerically-correct CUDA-graph training
> path (see [the ResNet-50 training note](../../throughput/resnet50/resnet50-training-speed-gpu-opennn-vs-pytorch.md)
> for the open speed-headroom investigation on that path); any future graph
> speedup shortens the train window and widens the energy lead accordingly.

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
