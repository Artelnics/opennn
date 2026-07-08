# Rosenbrock GPU benchmark: OpenNN vs PyTorch — max batch & speed

A focused GPU benchmark on a shallow Rosenbrock MLP (**1000 → 1000 (tanh) → 1**,
MSE, Adam, fp32) comparing **OpenNN** and **PyTorch** on four axes:

1. **Inference speed** at equal batch
2. **Training speed** at equal batch
3. **Maximum training batch** that fits in VRAM
4. **Maximum inference batch** that fits in VRAM

Run each engine **alone** with the GPU idle in between (`nvidia-smi` shows 0 MiB)
— on a small card, overlapping two GPU jobs causes spurious OOMs.

## What each path does

- **Device-resident inference** (`opennn_rosenbrock_resident_infer.cpp`):
  `NeuralNetwork::calculate_outputs_resident` keeps weights, activations, input
  and output GPU-resident (params uploaded once), so a repeated-inference loop
  pays only the forward kernels — versus the default
  `calculate_outputs(MatrixR)` path, which re-uploads parameters and copies input
  H2D / output D2H on every call.
- **Max-batch control**: OpenNN's GPU prefetch pool holds several `Batch` objects,
  each a full input+target copy on the GPU. `set_batch_pool_size(1)` drops this to
  one copy to maximize the fitting batch; the default of 3 keeps prefetch overlap
  (important for disk-streamed data like ResNet).
- **GPU-resident training data**: `opennn_rosenbrock_throughput.cpp` enables
  `set_storage_mode(GPUPersistantData)` in code so every batch avoids a host
  gather + H2D copy; the resident mega-graph (CUDA graph, on by default) reduces
  per-step host-side coordination.
- **Energy**: `run_energy.py` runs the identical workload on all three engines,
  logs GPU power at 20 Hz, and integrates it to joules/sample (trapezoidal; total
  and idle-subtracted), reporting a median ± stdev over N runs. These are
  sampled-power estimates (`nvidia-smi`, not a hardware joule counter), so treat
  the ratios, not the absolute watts, as the result. (`energy_measure.sh` is the
  older single-run wrapper.)

## Files

| File | Purpose |
|------|---------|
| `opennn_rosenbrock_resident_infer.cpp` | Device-resident inference throughput |
| `opennn_rosenbrock_throughput.cpp` | OpenNN train/inference throughput, any batch |
| `opennn_rosenbrock_trial.cpp` | One max-batch attempt (one process = one batch) |
| `run_maxbatch.sh` | Max-batch search driver (fresh process per trial) |
| `run_energy.py` | 3-way GPU energy harness |
| `pytorch_rosenbrock_throughput.py` | PyTorch train/inference throughput counterpart |
| `pytorch_rosenbrock_maxbatch.py` | PyTorch max-batch probe (VRAM-capped) |
| `tensorflow_rosenbrock_throughput.py` | TensorFlow throughput counterpart |
| `build_*.sh` | Hand-link recipes (paths are machine-specific; edit for your tree) |

## How to run

The `build_*.sh` scripts hand-link each benchmark against a prebuilt
`libopennn.a` — **edit the absolute paths** (CUDA, cuDNN, eigen, the build dir)
to match your machine, then:

```bash
# OpenNN inference speed (resident path)
./build_resident.sh
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_rosenbrock_resident_infer 8000 500 1000 1000

# OpenNN train/inference throughput  (args: mode samples batch iters inputs hidden [cuda_graph 0|1])
# Train data is kept GPU-resident in code; the mega-graph is on by default (pass 0 to disable).
./build_tput.sh
./opennn_rosenbrock_throughput train 100000 8000 20 1000 1000
./opennn_rosenbrock_throughput inference 8000 8000 500 1000 1000

# OpenNN max batch (fresh process per trial; auto VRAM-bound)
./build_trial.sh
./run_maxbatch.sh 1000 1000          # inputs hidden

# PyTorch counterparts (VRAM-capped to physical VRAM)
python pytorch_rosenbrock_throughput.py train 8000 200 1000 1000
python pytorch_rosenbrock_maxbatch.py 1000 1000

# Energy (3-way)
python run_energy.py --mode both --batch 8000 --iters 2000 --runs 5
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
- **Training data must stay GPU-resident**, or every batch does a host gather +
  H2D copy and starves the GPU; PyTorch keeps data on-device by default, so not
  doing it makes the comparison unfair to OpenNN.
- **OpenNN block-buffers `std::cout` to a pipe** — add `std::cout << std::unitbuf`
  in benchmarks, and never pipe a watched run through `grep` (it buffers too).
