# Rosenbrock GPU benchmark: OpenNN vs PyTorch — max batch & speed

A focused GPU benchmark on a shallow Rosenbrock MLP (**1000 → 1000 (tanh) → 1**,
MSE, Adam, fp32) comparing **OpenNN** and **PyTorch** on four axes:

1. **Inference speed** at equal batch
2. **Training speed** at equal batch
3. **Maximum training batch** that fits in VRAM
4. **Maximum inference batch** that fits in VRAM

All numbers below are on an **RTX 3060 Laptop (6 GB), WSL2, fp32**. Run each
engine **alone** with the GPU idle in between (`nvidia-smi` shows 0 MiB) — on a
6 GB card, overlapping two GPU jobs causes spurious OOMs.

## Results summary

| Axis | OpenNN | PyTorch | Winner |
|------|--------|---------|--------|
| Inference speed (b≥2000) | 3.8–4.4 M samples/s | 2.5–2.8 M | **OpenNN 1.43–1.56×** |
| Training speed (few steps/epoch) | 1.7 M samples/s | 1.3 M | **OpenNN 1.30–1.35×** |
| Training speed (many steps/epoch) | 0.9 M samples/s | 1.3 M | PyTorch (see note) |
| Max train batch | 482,344 (`OPENNN_BATCH_POOL=1`) | 399,507 | **OpenNN 1.21×** |
| Max inference batch (VRAM-bound) | ~524 K | ~535 K | ~tie |

### The inference-speed win
The default `NeuralNetwork::calculate_outputs(MatrixR)` path re-uploads all
parameters, allocates a fresh `ForwardPropagation`, copies the input H2D and the
output D2H — **every call**. The new `NeuralNetwork::calculate_outputs_resident`
keeps weights, activations, input and output GPU-resident (params uploaded once),
so a repeated-inference loop pays only the forward kernels. That is **4–6.5×
faster than the old path** and beats PyTorch 1.43–1.56×. See
`opennn_rosenbrock_resident_infer.cpp`.

### The max-batch win
OpenNN's GPU prefetch pool holds ≥3 `Batch` objects, each a full input+target
copy on the GPU. `OPENNN_BATCH_POOL=1` drops this to one copy → **+57% max batch**
(482 K vs PyTorch's 399 K) at ~6% throughput cost on GPU-resident data. Default
stays 3 (keeps prefetch overlap, important for disk-streamed data like ResNet).

### The training-speed note (important)
At **equal batch and equal step**, OpenNN's three GEMMs are actually **faster**
than PyTorch's (3.69 ms vs 4.98 ms at b=8000). OpenNN **wins training speed
1.30–1.35×** when there are few batches per epoch. The apparent loss at *many*
steps/epoch is **not compute** — it is per-step host-side pipeline coordination
(gather + transfer-stream events) that compounds with batch count. The resident
mega-graph (`OPENNN_GPU_RESIDENT_DATA=1 OPENNN_CUDA_GRAPH=1`) recovers ~+23% of
it. Ruled out as causes (measured): cuBLASLt algorithm choice, graph group size,
and the gather kernel. Full closure would require gathering *inside* the captured
graph (reading the resident dataset directly) — a larger re-architecture.

### Energy
`energy_measure.sh` wraps any run, logs GPU power at 20 Hz, and integrates it to
joules/sample (idle-subtracted). On this card OpenNN spends **1.44× less energy
per inference** and **2.42× less per training sample** (1.76× even at 50
mini-batches/epoch) than PyTorch — see the
[energy article](../energy-consumption-gpu-opennn-vs-pytorch.md). Inference: more
power but faster → less energy; training: lower power *and* faster.

## Files

| File | Purpose |
|------|---------|
| `energy_measure.sh` | Wrap a run, integrate GPU power → J/sample + avg W |

| File | Purpose |
|------|---------|
| `opennn_rosenbrock_resident_infer.cpp` | Device-resident inference throughput (the win) |
| `opennn_rosenbrock_throughput.cpp` | OpenNN train/inference throughput, any batch |
| `opennn_rosenbrock_trial.cpp` | One max-batch attempt (one process = one batch) |
| `run_maxbatch.sh` | Max-batch search driver (fresh process per trial) |
| `pytorch_rosenbrock_throughput.py` | PyTorch train/inference throughput counterpart |
| `pytorch_rosenbrock_maxbatch.py` | PyTorch max-batch probe (VRAM-capped) |
| `build_*.sh` | Hand-link recipes (paths are machine-specific; edit for your tree) |

## How to run

The `build_*.sh` scripts hand-link each benchmark against a prebuilt
`libopennn.a` — **edit the absolute paths** (CUDA, cuDNN, eigen, the build dir)
to match your machine, then:

```bash
# OpenNN inference speed (resident path)
./build_resident.sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_rosenbrock_resident_infer 8000 500 1000 1000

# OpenNN train/inference throughput  (args: mode samples batch iters inputs hidden)
# Set GPU_RESIDENT for a fair train number; for the mega-graph add OPENNN_CUDA_GRAPH=1.
./build_tput.sh
OPENNN_GPU_RESIDENT_DATA=1 ./opennn_rosenbrock_throughput train 100000 8000 20 1000 1000
./opennn_rosenbrock_throughput inference 8000 8000 500 1000 1000

# OpenNN max batch (fresh process per trial; auto VRAM-bound)
./build_trial.sh
./run_maxbatch.sh 1000 1000          # inputs hidden

# PyTorch counterparts (VRAM-capped to physical 6 GB)
python pytorch_rosenbrock_throughput.py train 8000 200 1000 1000
python pytorch_rosenbrock_maxbatch.py 1000 1000
```

## Methodology gotchas (these cost real time)

- **WSL silently spills GPU allocations into system RAM** (shared GPU memory),
  so an uncapped max-batch probe measures VRAM+RAM, not GPU capacity. The PyTorch
  probe caps with `torch.cuda.set_per_process_memory_fraction(1.0)`. OpenNN spills
  too — detect it by watching whether peak VRAM plateaus while the batch keeps
  growing.
- **OpenNN faults (CUDA error 700, sticky) on OOM** instead of throwing cleanly,
  which corrupts the context — so an in-process binary search converges on
  garbage. The driver runs **one fresh process per batch trial**.
- **Always set `OPENNN_GPU_RESIDENT_DATA=1` for training**, or every batch does a
  host gather + H2D copy and starves the GPU; PyTorch keeps data on-device by
  default, so omitting it makes the comparison unfair to OpenNN.
- **OpenNN block-buffers `std::cout` to a pipe** — add `std::cout << std::unitbuf`
  in benchmarks, and never pipe a watched run through `grep` (it buffers too).
