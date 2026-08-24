# ResNet-50 fixed-work GPU energy — OpenNN vs PyTorch vs TensorFlow

GPU energy for **identical fixed work**: every engine trains the same
ResNet-50 v1.5 at CIFAR-10 geometry for the same number of epochs at the same
batch size, and we integrate GPU board power over each engine's timed training
window. Same protocol as [`../higgs-dense-energy/`](../higgs-dense-energy/):
`nvidia-smi power.draw` sampled at 20 Hz, trapezoid-integrated between the
`TRAIN_START_UNIX` / `TRAIN_END_UNIX` markers each driver prints around its
timed loop, idle baseline subtracted for the "active" figures. The 2 warmup
epochs and one-time data loading stay outside the window.

The engine drivers are the ones from the ResNet speed track
([`../../throughput/resnet50/`](../../throughput/resnet50/)): OpenNN
`opennn_resnet50_speed` (cuDNN + CUDA graph), `pytorch_resnet50_speed.py`
(channels_last + `torch.compile` + TF32), `tensorflow_resnet50_speed.py`
(XLA train step). Each runs its fastest path; the work is identical.

Headline metric: **microjoules per nominal epoch-sample**
(`energy / (samples x epochs)`, same divisor for every engine), plus total and
active Wh, average power, and the training window wall time.

## How to run

```bash
# 1. Build the OpenNN driver (registered in benchmarks/CMakeLists.txt).
cmake --build build --target opennn_resnet50_speed -j

# 2. Prepare CIFAR-10 once (writes BMPs for OpenNN + npy for PyTorch/TF).
python benchmarks/throughput/resnet50/prepare_cifar10.py

# 3. Run (fp32 + bf16, 3 runs each, 20 epochs of fixed work).
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
export BENCH_PYTHON="$HOME/.venvs/opennn-bench/bin/python"
python benchmarks/energy/resnet50-energy/run_resnet50_energy.py \
    --epochs 20 --batch 128 --precision both --runs 3
```

Writes `../../results/gpu-resnet50-energy-<run_id>.json` (immutable, per
`../../results/README.md`). Run on a quiet GPU: the runner measures the idle
power baseline at startup and waits for cooldown between runs.
