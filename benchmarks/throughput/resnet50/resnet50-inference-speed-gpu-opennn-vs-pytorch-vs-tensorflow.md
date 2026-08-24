# GPU ResNet-50 inference: OpenNN vs PyTorch vs TensorFlow

OpenNN reaches a five-run median of 111,936 samples/s in fp32 and 185,903 samples/s in bf16 on an NVIDIA GeForce RTX 4080. In the same forward-only ResNet-50 workload, that is 1.296x and 1.480x PyTorch throughput, respectively.

> Results are medians across five runs, with the standard deviation reported for every framework. This benchmark uses ResNet-50 v1.5 with CIFAR geometry and 32x32 CIFAR-10 inputs; it does not represent ImageNet-resolution inference.

## Contents

- [Introduction](#introduction)
- [Benchmark application](#benchmark-application)
- [Reference computer](#reference-computer)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [Reproducing](#reproducing)
- [References](#references)

## Introduction

ResNet-50 remains a useful test of convolution, normalization, residual connections, and repeated GPU kernel execution. The CIFAR-10 adaptation keeps the ResNet-50 v1.5 bottleneck structure while using the smaller spatial geometry required by 32x32 images.

This benchmark compares sustained fp32 and bf16 forward-pass throughput in OpenNN, PyTorch, and TensorFlow. It measures the optimized inference path of each framework rather than preprocessing, model construction, or cold start.

## Benchmark application

| Item | Configuration |
|---|---|
| Dataset | CIFAR-10 |
| Input | 32x32 RGB images |
| Model | ResNet-50 v1.5 bottleneck, CIFAR geometry |
| Batch size | 128 |
| Precisions | fp32 and bf16 |
| Runs | 5 per framework and precision |
| Timed work | Forward inference only |
| Metrics | Median samples/s and milliseconds/batch; higher throughput is better |

## Reference computer

| Component | Value |
|---|---|
| CPU | Intel Core i9-12900K |
| GPU | NVIDIA GeForce RTX 4080, 16 GB |
| NVIDIA driver | 595.84 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| TensorFlow | 2.21.0 |
| OpenNN | commit 52e21e15d |
| Run ID | 20260810T095235Z |

## Methodology

All three frameworks use the same CIFAR-10 input geometry, ResNet-50 v1.5 bottleneck topology, batch size, precision, and forward-only timing scope. Five successful runs are recorded for each framework and precision. The tables report the median throughput and batch time, plus the standard deviation of throughput across runs.

- OpenNN keeps inputs on the GPU and uses its captured resident-output inference path.
- PyTorch runs in evaluation mode under `no_grad` with `PT_FAST=1`, channels-last tensors, TF32 for fp32, and `torch.compile(mode="reduce-overhead")`. This is PyTorch's compile-plus-CUDA-Graphs inference path.
- TensorFlow runs with `training=False` under XLA and uses its bf16 policy for the bf16 cell.

Dataset preparation, process startup, model construction, host-to-device setup, and file loading are outside the measured region. GPU synchronization is included before timing is finalized.

## Results

| Precision | Framework | Median throughput | Standard deviation | Median batch time | OpenNN speedup |
|---|---|---:|---:|---:|---:|
| fp32 | OpenNN | **111,936 samples/s** | **284 samples/s** | **1.144 ms** | **1.000x** |
| fp32 | PyTorch | 86,357 samples/s | 486 samples/s | 1.482 ms | **1.296x** |
| fp32 | TensorFlow | 53,669 samples/s | 1,249 samples/s | 2.385 ms | **2.086x** |
| bf16 | OpenNN | **185,903 samples/s** | **819 samples/s** | **0.689 ms** | **1.000x** |
| bf16 | PyTorch | 125,592 samples/s | 1,913 samples/s | 1.019 ms | **1.480x** |
| bf16 | TensorFlow | 83,683 samples/s | 4,072 samples/s | 1.530 ms | **2.222x** |

OpenNN leads PyTorch by 29.6% in fp32 and 48.0% in bf16. Against TensorFlow, the corresponding leads are 108.6% and 122.2%.

## Discussion

The five-run result separates the fp32 paths clearly: OpenNN records 111.9k samples/s with a standard deviation of 284 samples/s, while PyTorch reaches 86.4k samples/s and TensorFlow processes 53.7k samples/s in the same cell.

The bf16 path raises OpenNN throughput to 185.9k samples/s, a 1.66x increase over its own fp32 result. OpenNN also retains the highest throughput in this precision, reaching 1.480x PyTorch and 2.222x TensorFlow.

Relative dispersion is lowest for OpenNN in both cells: 0.25% in fp32 and 0.44% in bf16. PyTorch records 0.56% and 1.52%, while TensorFlow records 2.33% and 4.87%. These values describe repeatability in this five-run sample; they do not by themselves identify which runtime mechanism causes the difference.

The small CIFAR spatial geometry makes launch scheduling and framework overhead more visible than an ImageNet-resolution workload. The results therefore support this exact CIFAR-10 deployment shape and should not be generalized to 224x224 ResNet-50 without a separate measurement.

## Conclusions

- OpenNN reaches 111,936 ± 284 samples/s in fp32 and 185,903 ± 819 samples/s in bf16.
- OpenNN is 1.296x PyTorch in fp32 and 1.480x PyTorch in bf16.
- OpenNN is 2.086x TensorFlow in fp32 and 2.222x TensorFlow in bf16.
- Five-run dispersion remains below 0.5% for OpenNN in both precision cells.
- The result applies to ResNet-50 v1.5 with CIFAR-10 geometry and batch size 128.

## Reproducing

The canonical runner is `benchmarks/throughput/resnet50/run_resnet50_infer.py`:

```bash
python benchmarks/throughput/resnet50/run_resnet50_infer.py \
  --dataset cifar10 --batch 128 --runs 5 --precision both
```

The immutable result artifact is `benchmarks/results/gpu-resnet50-inference-speed-cifar10-20260810T095235Z.json`.

## References

- [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
- [NVIDIA CUDA](https://developer.nvidia.com/cuda-zone).
