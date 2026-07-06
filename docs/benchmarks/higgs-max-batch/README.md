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
| Data | real prepared HIGGS rows via `--higgs-bin` (recommended), else synthetic with the contract shapes |

With `--higgs-bin` every engine reads the same float32 binary (rows × 29:
28 standardized features then the {0,1} label), and rows repeat modulo when
the candidate batch exceeds the file — the same convention as the ResNet-50
capacity runner repeating CIFAR-10. Prepare it once from a prepared HIGGS
CSV (see [`../higgs/README.md`](../higgs/README.md)):

```bash
python - <<'EOF'
import numpy as np, os
root = os.environ["OPENNN_BENCH_DATA"]
a = np.loadtxt(f"{root}/higgs/higgs_train.csv", delimiter=",", dtype=np.float32)
a.tofile(f"{root}/higgs/higgs_train_f32.bin")
EOF
```

Without `--higgs-bin` the trials fall back to synthetic contract-shaped
data (capacity depends on the shapes and the step, not the values; the
2026-07-05 CPU artifacts above were measured in synthetic mode). Either
way, quality-gated HIGGS numbers (accuracy / log loss / AUC on the real
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
  each trial process under a hard `RLIMIT_DATA` cap (`--mem-cap-gib`,
  default 8 — the same budget as the published
  [data-capacity benchmark](../data-capacity-opennn-vs-pytorch-vs-tensorflow.md)),
  so the out-of-memory boundary is deterministic instead of swap-dependent.
  `RLIMIT_DATA` charges brk + anonymous mmap (the tensor allocations) but not
  file-backed library mappings — PyTorch/TF map ~5 GiB of runtime libraries,
  so an `RLIMIT_AS` address-space cap would measure code size, not data
  capacity, while the Windows Job Object cap used by the data-capacity
  benchmark is committed memory, which `RLIMIT_DATA` approximates far
  better. Trials report `peak_rss_mib` / `vm_peak_mib` for context. Linux
  only (kernel ≥ 4.7); on Windows use a Job Object wrapper as in
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

## Results — CPU inference, fp32, 8 GiB data cap (internal, WSL2)

Internal runs (2026-07-05, WSL2 Ubuntu 24.04, 20-thread desktop CPU,
`RLIMIT_DATA` 8 GiB per trial). Artifacts:
`results/cpu-higgs-max-batch-20260705T115414Z.json` (3-way, before the
`linear_forward_cpu` fix),
`results/cpu-higgs-max-batch-20260705T124425Z.json` (OpenNN Eigen rerun
after it), and `results/cpu-higgs-max-batch-20260705T131138Z.json`
(OpenNN MKL build):

| Engine | Max inference batch | Peak RSS at max | Anonymous data at import |
|---|---:|---:|---:|
| OpenNN, MKL build, row-tiled resident inference | **62,914,560** | 8,011 MiB | ~30 MiB |
| OpenNN, Eigen build, untiled (after `linear_forward_cpu` fix) | 1,013,085 | 8,042 MiB | ~20 MiB |
| OpenNN, MKL build, untiled | 1,012,070 | 8,043 MiB | ~30 MiB |
| PyTorch 2.5.1 (cu121 wheel) | 837,263 | 7,072 MiB | 1,474 MiB |
| TensorFlow 2.21 | 524,288 | 4,934 MiB | 965 MiB |

The tiled row (artifact `cpu-higgs-max-batch-20260705T143356Z.json`, boundary
resolved to 1 M: 62,914,560 passes, 63,963,136 fails) uses row-tiled
inference: CPU inference is batch-separable — with `is_training == false` no
layer mixes samples — so the forward runs in 131,072-row tiles through
reused workspaces. Activation memory is then a fixed ~1.07 GiB regardless
of batch, and the ceiling becomes the caller's own input + output data
(~116 B/sample), which is why capacity jumps 62×. The tile size is the
measured MKL speed-parity point: 3-run medians 168 k (tiled) vs 171.6 k
(untiled) samples/s at batch 262,144 (Eigen: 160 k vs 154 k) — within this
machine's run-to-run noise, while smaller tiles cost real throughput
(−15 % at 64 k rows, −21 % at 32 k). `NeuralNetwork::calculate_outputs`
applies the same tiling by default on CPU (1 GiB activation budget, row
count adapted to the network's width, engaged only when the batch would
exceed the budget — so it never uses more memory than the untiled path),
and the trial reuses tile workspaces across iterations, the CPU analogue
of the GPU resident protocol. PyTorch/TF comparisons run their default
eager paths, which materialize full-batch activations; chunking there is
user code, not engine behavior.

OpenNN measured 615,712 before the fix — the same 3-way run that produced
the PyTorch and TensorFlow rows (their engines are unaffected by the fix).
The MKL build (the production CPU configuration used by the HIGGS speed
benchmarks) lands within 0.1 % of the Eigen build — its `cblas_sgemm` path
always wrote directly into the output slot, and the difference is MKL's
~9 MiB internal buffer pool — while raising steady-state inference
throughput a further ~16 % (154 k → 178 k samples/s at batch 262,144).

Reading it:

- **OpenNN fits the largest forward batch: 1.21× PyTorch, 1.93× TF.** Its
  per-sample footprint is 8.3 KB — the live-set minimum (input + two
  1024-wide activations + output) — with a ~20 MiB process baseline. The
  margin over PyTorch is entirely the baseline: both engines sit at the
  per-sample minimum.
- **The fix behind it** (`tensor_operations.cpp`, `linear_forward_cpu`):
  `(input * weights).rowwise() + bias` made Eigen materialize the whole GEMM
  result in a heap temporary before the bias add — an extra batch × 1024
  float allocation per hidden layer (~5 KB/sample at peak). Now the GEMM
  writes straight into the output slot and the bias is added in place. Side
  effect: steady-state CPU inference throughput rose ~6.4× at batch 262,144
  (24 k → 154 k samples/s) because the temporary was mmap'd and freed every
  pass, re-faulting gigabytes of pages each iteration. MKL builds always
  took the direct `cblas_sgemm` path and are unaffected. Test suite after
  the fix: 592/594 pass; the 2 failures (Scaling/Unscaling ForwardPropagate,
  "network has no trainable layers") predate the change and involve no
  dense layer.
- **PyTorch is also at the per-sample minimum** (`inference_mode` frees each
  intermediate as the next layer computes, ~8.1 KB/sample) but pays a
  1.5 GiB import baseline at this budget — and proportionally more at
  smaller ones (at 2 GiB: OpenNN ~240 k vs PyTorch ~67 k).
- **TensorFlow's ceiling is the 2 GiB tensor boundary, then its allocator.**
  Its largest passing batch (524,288) puts the first hidden activation at
  exactly 2^31 bytes. One sample more passes uncapped (RSS 5.1 GiB) but
  exceeds the 8 GiB anonymous-data budget — past the 2 GiB boundary its
  allocator reserves roughly double the working set.
- Caveats: single run per candidate (the boundary is deterministic under the
  hard cap), WSL2, and the venv's torch is the CUDA wheel — a CPU-only wheel
  has a smaller baseline, which would raise PyTorch's number somewhat (it
  cannot pass OpenNN: even at zero baseline its ceiling is ~1.03 M).
  Engineering evidence, not a public headline.

## Results — GPU inference, real HIGGS rows, VRAM-capped (internal, WSL2)

RTX 3060 Laptop GPU (6,144 MiB, cap 5,632 MiB), WSL2, real HIGGS rows via
`--higgs-bin` (100 k rows, repeated modulo). Artifacts:
`results/gpu-higgs-max-batch-20260705T151042Z.json` (3-way baseline) and
`results/gpu-higgs-max-batch-20260705T153749Z.json` (OpenNN tiled):

| Engine | fp32 max batch | bf16 max batch |
|---|---:|---:|
| OpenNN, row-tiled resident inference | **45,203,456** | **48,234,496** |
| OpenNN, untiled resident | 695,980 | 1,352,555 |
| PyTorch 2.5.1 | 691,200 | 1,344,512 |
| TensorFlow 2.21 | 451,260 | 448,237 |

The OpenNN rows are post-audit (2026-07-06, artifacts
`results/gpu-higgs-max-batch-20260706T001426Z.json` tiled /
`...T001525Z.json` untiled; earlier same-protocol runs measured tiled
40,370,176 / 44,957,696 with a 131,072-row tile and 694,700 / 1,349,996
untiled). The gains came from a 40-agent adversarial code audit plus a
tile-size sweep: (a) GPU tile 65,536 — measured *faster* than untiled in
fp32 (5.22 vs 5.09 M samples/s) with a cuBLASLt algorithm cliff at 32,768
(−21 %), auto-selected per device (CPU keeps 131,072); (b) the tail
propagation now overlays the tile arena instead of owning a second one
(up to 512 MiB; previously boundaries snapped to tile multiples);
(c) the output assembly buffer is bf16-width in bf16 mode and skipped
entirely in single-tile runs; (d) the legacy cuBLAS handle initializes
lazily like cuDNN (the inference path is pure cuBLASLt + custom kernels).
The audit also *formally confirmed the fp32 untiled floor* (GEMMs cannot
run in place; h2 must fully materialize; the fp32 input feeds TF32 GEMMs
with no staged copy). Follow-up HIGGS BF16 inference now stores the
resident input as BF16 by default
(`OPENNN_HIGGS_BF16_RESIDENT_INPUT=0` restores the fp32-resident input
protocol, where `OPENNN_HIGGS_ALIAS_BF16_INPUT=0` also disables the
dead-h2 input-cast alias). The trial also releases the fp32 parameter
master before allocating the large inference activations
(`OPENNN_HIGGS_RELEASE_BF16_FP32_MASTER=0` disables that A/B). On this
6 GiB GPU, the capped bf16 untiled ceiling rose from 1,351,680 to
1,390,936 rows at a 512-row search step, with no measured speed loss.

Reading it:

- **The untiled row is the apples-to-apples comparison and it is already at
  the theoretical floor.** The measured VRAM slope is fully accounted:
  fp32 8,302 B/sample = activations (8,196) + fp32 input (112); bf16
  is bf16 activations + bf16 input in the resident-input path. Restoring
  fp32-resident input adds ~56 B/sample; disabling the dead-h2 alias in
  that fallback adds another ~56 B/sample.
  There is no waste to remove — the GPU `linear_forward` writes through
  cuBLASLt's fused bias epilogue straight into the output slot. OpenNN
  edges PyTorch (+0.5 %, both at the same floor since PyTorch's GPU
  allocator frees intermediates); TensorFlow's own allocator plateaus at
  ~3.7 GiB and its bf16 max batch gains nothing over fp32.
- **The tiled rows are the engine-capability demonstration**, the GPU twin
  of the CPU protocol: input and assembled outputs stay device-resident,
  131,072-row tile workspaces are reused, activations are O(tile). The
  ceiling becomes the resident input data (112 B/sample), which is why
  fp32 jumps 58×. Speed is unchanged — 3-run A/B at batch 524,288:
  fp32 4.99 M samples/s tiled vs 4.99 M untiled, bf16 9.93 M vs 9.94 M.
- **Fixed-overhead audit (batch 1024, activations ≈ 0)**: OpenNN 109 MiB,
  PyTorch 175 MiB, TensorFlow 3,705 MiB (its allocator grabs the pool up
  front). OpenNN's figure was 135 MiB until the eager cuDNN handle —
  ~26 MiB of kernel images a dense network never uses — was made lazy
  (`Backend::cudnn()`, created on first cuDNN op; conv/attention pay it
  when they first need it). On this WSL2 box the untiled ceiling did not
  move with the fix (rerun 694,444 fp32 / 1,348,973 bf16, within noise of
  the baseline — artifact `results/gpu-higgs-max-batch-20260705T230710Z.json`):
  near the cap the driver already demotes cold pages to system RAM, so the
  handle's VRAM was effectively reclaimed at the boundary anyway. On native
  Linux (no spill) the ~26 MiB should convert to ~+3 k fp32 samples.
- Caveats: WSL2 (VRAM-capped on purpose; the driver spills to system RAM
  past physical VRAM), 6 GiB laptop card, single run per candidate.
  bf16 *throughput* on this box is known to be WSL2-penalized (see the
  rosenbrock note); capacity is unaffected. Engineering evidence — rerun
  on the reference machine before public use. Known GPU-suite failure on
  this box, pre-existing and unrelated (verified against the pre-change
  build): `DeviceBackendTest.SetZeroAsyncClearsHostBuffer` fails under
  WSL2 and its sticky async error poisons four later GPU tests in
  same-process runs; all four pass in isolation.

## Results — GPU training, real HIGGS rows, VRAM-capped (internal, WSL2)

Same machine and protocol; one optimizer step = forward + backward + one
Adam update over the batch. Artifacts:
`results/gpu-higgs-max-batch-20260706T002825Z.json` (3-way monolithic
baseline) and `results/gpu-higgs-max-batch-20260706T004843Z.json`
(OpenNN gradient-accumulated):

| Engine | fp32 max train batch | bf16 max train batch |
|---|---:|---:|
| OpenNN, gradient-accumulated step (tile 65,536) | **≥ 67,108,864** (search limit; VRAM constant 1,183 MiB) | **≥ 67,108,864** (constant 695 MiB) |
| OpenNN, monolithic step | 348,245 | 674,998 |
| PyTorch 2.5.1 | 346,112 | 650,834 |
| TensorFlow 2.21 | 290,757 | 261,618 |

Reading it:

- **The monolithic row is the like-for-like comparison and OpenNN leads
  it** (+0.6 % fp32, +3.7 % bf16 over PyTorch; TensorFlow's allocator caps
  itself again). The boundary profile shows the monolithic footprint at
  its floor: forward activations + backward deltas (the delta pool carries
  its own liveness analysis and sits within 4 B/sample of optimal) + the
  batch data. A follow-up training path now converts bf16 inputs in pinned
  host memory and uploads bf16 directly by default
  (`OPENNN_BF16_HOST_INPUT_CAST=0` restores the old device-staging path),
  and no longer reserves gather-index device buffers for non-gather batches;
  on this 6 GiB GPU, the capped bf16 monolithic ceiling rose from 679,936
  to 694,912 rows at a 512-row search step with no measured speed loss.
- **The accumulated row is the engine capability**: Adam's new
  `set_update_period(K)` runs one optimizer step over a virtual batch of
  K × batch_size by averaging K equal mini-batch gradients into a
  parameters-sized accumulator (4.3 MB) and applying a single update —
  mathematically the full-batch step (sub-batch gradients are per-batch
  means; exactness verified single-threaded on CPU: 0.69913 monolithic vs
  0.699131 accumulated, reduction-order agreement). The mini-batch
  machinery streams tiles from host (pool 1), so VRAM holds only tile
  buffers: the measured peak does not grow with the virtual batch, and the
  ceiling moves to host RAM / the search limit. It is also *faster* than
  the monolithic step at equal batch: fp32 1.19 M vs 1.03 M samples/s
  (+16 %), bf16 1.97 M vs 1.75 M (+13 %) — better GEMM shapes plus
  batch-pool overlap. PyTorch/TF users get this via hand-written
  accumulation loops; here it is one setter on the optimizer.
- The monolithic-vs-accumulated `final_loss` prints differ on GPU only
  because the pre-existing CUDA warmup performs a real update on the
  monolithic path while an accumulation period longer than the warmup does
  not — a warmup bookkeeping asymmetry, not a math difference (the CPU
  check above has no warmup and agrees).

## Results — CPU training, real HIGGS rows, 8 GiB RLIMIT_DATA (internal, WSL2)

Same protocol as the CPU inference rows; fp32 (CPU mode drops bf16).
Artifact: `results/cpu-higgs-max-batch-20260706T083207Z.json` (monolithic
3-way; the gradient-accumulated OpenNN search runs separately):

| Engine | Max train batch (monolithic step) |
|---|---:|
| **OpenNN (MKL)** | **501,651** |
| PyTorch 2.5.1 | 411,423 |
| TensorFlow 2.21 | 331,784 |

OpenNN +22 % over PyTorch and +51 % over TensorFlow — the widest
like-for-like margin in the matrix, because every engine shares the
~16.5 KB/sample training slope while PyTorch's ~1.5 GiB runtime baseline
costs proportionally more under an 8 GiB budget than it did in VRAM.
CPU gradient accumulation is exact (single-thread check 0.69913 vs
0.699131) and at speed parity at the 131,072-row tile (−1.3 %, within
noise; 65,536 costs 6.6 % — unlike the GPU, MKL gains nothing from
smaller GEMMs and there is no transfer to overlap, so tiling on CPU buys
capacity only).

## Status

CPU and GPU inference and training measured (tables above; internal).
Still to run: the CPU gradient-accumulated training search (in progress),
and reference-machine reruns — archive the result JSONs.
