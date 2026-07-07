# OpenNN benchmarks: OpenNN vs PyTorch vs TensorFlow

*For [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-15.*

These notes compare the **deployment characteristics** of OpenNN against PyTorch and
TensorFlow — how much a trained model costs to ship, start, and run, not how fast it trains.
The theme is the same throughout: OpenNN is a native C++ library that links into your
executable, so its footprint is a small binary; PyTorch and TensorFlow are large general-purpose
frameworks with a runtime that must be loaded, installed, or shipped alongside the model.

This page is the **current headline index**. Detailed benchmark notes may mention earlier
runs when they explain an optimization path, but the table below is the single source of
truth for current headline numbers. Results are measured on the platform stated in each
linked note; several newer GPU notes use Linux x86_64 under WSL2, while the CPU inference
note is a Windows result that is marked for Linux re-measurement before public use.

## Consolidation rules

- `benchmark_manifest.json` is the machine-readable inventory for every
  top-level benchmark note.
- `TRACKS.md` is the readable active/support/internal/hold-back/historical
  track map.
- `BENCHMARK_RUNBOOK.md` is the operational map for what to run next.
- `PRESENTATION_CLAIMS.md` gates what can be used in slides.
- Folder-level `README.md` files explain the local runner entry point.
- Run `python tools/validate_benchmarks.py` after adding, renaming, or retiring
  any benchmark note or runner folder.

## Summary

| Benchmark | OpenNN | PyTorch | TensorFlow |
|---|---:|---:|---:|
| [Accuracy (R², Rosenbrock)](quality/accuracy/accuracy-opennn-vs-pytorch-vs-tensorflow.md) | **0.988** | 0.988 | 0.987 |
| [Training precision (MSE / time)](quality/precision/precision-opennn-vs-pytorch-vs-tensorflow.md) | **0.109 / 2 s** | 0.162 / 42 s | 0.156 / 310 s |
| [GPU CNN training (samples/s)](throughput/cnn-training-speed/cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md) | **211,904** | 111,241 | 55,339 |
| [GPU ResNet-50 training (samples/s)](throughput/resnet50-training-speed/resnet50-training-speed-gpu-opennn-vs-pytorch.md) | **8,433** | 5,268 (`torch.compile`) / 3,960 eager | — |
| [GPU ResNet-50 max train batch](capacity/resnet50-max-batch/resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md) | 4,752 | 9,216 (`torch.compile`) / 8,704 eager | **11,760** |
| [GPU dense inference (samples/s)](capacity/rosenbrock-max-batch/rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) | **3.99 M** | 2.80 M | — |
| [GPU dense max train batch](capacity/rosenbrock-max-batch/rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) | **482,344** | 399,507 | — |
| [GPU energy / inference sample (µJ)](capacity/rosenbrock-max-batch/energy-consumption-gpu-opennn-vs-pytorch.md) | **25.9** | 43.5 | 29.8 |
| [GPU Transformer energy to target (Wh)](energy/transformer-energy/transformer-energy-to-target-gpu-opennn-vs-pytorch-vs-tensorflow.md) | **24.1** | 33.2 | 39.8 (XLA) |
| [GPU Transformer inference, bf16, seq 512 (tok/s)](throughput/attention-speed/transformer-inference-gpu-opennn-vs-pytorch.md) | **160,128** | 84,511 | 101,400 |
| [CPU inference speed (samples/s; Windows tuned)](throughput/inference-speed/inference-speed-opennn-vs-pytorch-vs-tensorflow.md) | **466,837** | 372,780 | 375,409 |
| [CPU runtime size](footprint/size-cpu-opennn-vs-pytorch-vs-tensorflow.md) | **3.2 MB** | 442 MB | 752 MB |
| [GPU (CNN) deployment](footprint/size-gpu-opennn-vs-pytorch-vs-tensorflow.md) | **~1.3 GB** | ~5.0 GB | ~6.2 GB |
| [Startup latency](footprint/startup/startup-latency-opennn-vs-pytorch-vs-tensorflow.md) | **36 ms** | 1,005 ms | 1,685 ms |
| [Baseline memory footprint](footprint/memory/peak-memory-opennn-vs-pytorch-vs-tensorflow.md) | **195.2 MB RAM / 119.0 MB GPU-ready VRAM** | 516.2 MB / 155.0 MB | 871.2 MB / 121.0 MB |
| [Data capacity (8 GB cap)](capacity/data-capacity/data-capacity-opennn-vs-pytorch-vs-tensorflow.md) | **16M samples** | 6M samples | (same path¹) |
| [Install size / packages](footprint/dependencies-opennn-vs-pytorch-vs-tensorflow.md) | **1 file, 0 pkgs** | 946 MB, 12 pkgs | 1.6 GB, 33 pkgs |
| [Native source LOC](footprint/loc-opennn-vs-pytorch-vs-tensorflow.md) | **34,926** | 834,319 | 1,792,182 |
| [Application LOC](footprint/application-loc/application-lines-of-code-opennn-vs-pytorch-vs-tensorflow.md) | **14** | 43 | 23 |
| [Standalone code export](footprint/export/code-export-opennn-vs-pytorch-vs-tensorflow.md) | **C/Py/JS/PHP source** | needs a runtime | needs a runtime |

## Evidence status

| Benchmark group | Status | Before public headline use |
|---|---|---|
| Accuracy, precision, footprint, startup, memory, capacity, dependencies, LOC, code export | Current benchmark notes | Keep raw command output, version metadata, and commit hash with the published numbers |
| CPU inference speed | Valid Windows tuned result; not yet final public headline | Re-measure on the reference Linux x86_64 box and keep the Windows result as a platform-specific note |
| GPU CNN training | Current WSL2 laptop GPU result | Add an optimized PyTorch/`torch.compile` and TensorFlow/XLA run or state clearly that the comparison is eager/Keras fp32 |
| GPU ResNet-50 training | Current WSL2 laptop GPU result; previous 2,912 samples/s is historical | Keep the `torch.compile` result in the official runner output, not only in the compile probe |
| GPU ResNet-50 max batch | Current RTX 4080 capacity result; TensorFlow and PyTorch fit larger batches than OpenNN | Treat as memory-regression evidence, not as an OpenNN headline claim; repeat before public use |
| GPU dense speed, max batch, and energy | Current WSL2 laptop GPU result | Replace machine-specific build scripts with CMake targets and attach raw logs/power traces |
| CPU HIGGS dense training/inference | Internal WSL2 100k/20k subset measured with MKL; OpenNN wins training and the batch-1024 inference rerun | Rerun the full HIGGS split and add measured CPU joules outside WSL before public energy claims |
| GPU Transformer inference and training | Current WSL2 laptop GPU result | Add repeated-run statistics and exact cross-framework correctness/quality gates before using as a flagship claim |
| Recurrent/LSTM forecasting | Harness packaged; Linux result pending | Run the JSON harness on the reference Linux GPU and archive raw output |
| GPU Transformer energy to target | Current RTX 4080 result (JSON artifact in `results/`) | Repeat on the reference machine; investigate the occasional convergence collapse (OpenNN 2/4, TF 3/4 seeds converge at lr 1e-4) before flagship use |

## Internal HIGGS snapshot

These rows are local migration evidence for the dense HIGGS track. They are not
part of the public headline table until the full HIGGS split is rerun with
committed result artifacts and measured CPU energy outside WSL.

| Benchmark | OpenNN | PyTorch | TensorFlow | Status |
|---|---:|---:|---:|---|
| [CPU HIGGS dense training, samples/s](throughput/higgs/higgs-cpu-training-opennn-vs-pytorch-vs-tensorflow.md) | **59,372** | 25,651 | 49,101 | WSL2 100k/20k subset, MKL-linked OpenNN |
| [CPU HIGGS dense inference, samples/s](throughput/higgs/higgs-cpu-inference-opennn-vs-pytorch-vs-tensorflow.md) | **209,774** | 166,003 | 153,281 | WSL2 20k test subset, batch 1024, MKL-linked OpenNN |

Top-level benchmark notes are the public evidence layer unless they are marked
internal or historical. The
`CONTINUE_HERE.md` files and machine-specific build notes in subdirectories are
internal lab logs until their commands and raw artifacts are folded into the consolidated
benchmark harness.

The machine-readable benchmark inventory is
[`benchmark_manifest.json`](benchmark_manifest.json). New published runs should write
result artifacts under [`results/`](results/) using the format described there.
The result artifact rules in [`results/README.md`](results/README.md) define the
suite's MLPerf-inspired protocol. These are not official MLPerf results unless
submitted through MLCommons.

OpenNN is the smallest on every footprint axis, starts an order of magnitude faster, is the
only one that can export a trained model as dependency-free source — and reaches the **same
accuracy** as PyTorch and TensorFlow on an identical task.

## The benchmarks

* **[Numerical accuracy](quality/accuracy/accuracy-opennn-vs-pytorch-vs-tensorflow.md)** — same network, data, and
  optimizer in all three: OpenNN matches PyTorch and TensorFlow (R² ≈ 0.988) on the Rosenbrock
  approximation benchmark. The lean footprint costs nothing in accuracy.
* **[Training precision](quality/precision/precision-opennn-vs-pytorch-vs-tensorflow.md)** — the best MSE each
  tool reaches with the optimizers it ships. Every second-order run (OpenNN's native
  quasi-Newton and Levenberg-Marquardt, and PyTorch's built-in LBFGS) hits the same ~0.108
  error floor; first-order Adam is stuck at 0.14–0.20. OpenNN reaches it in one line and ~1 s;
  PyTorch's LBFGS needs a closure-rewritten loop; core Keras has no second-order option at all.
* **[GPU CNN training speed](throughput/cnn-training-speed/cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md)** — a
  minimal CNN (one conv + one pooling layer) on MNIST, batch 128, fp32: OpenNN trains 1.9×
  faster than PyTorch and 3.8× faster than TensorFlow on the same GPU (cudnn-frontend
  convolutions + GPU-resident dataset mode).
* **[GPU ResNet-50 training speed](throughput/resnet50-training-speed/resnet50-training-speed-gpu-opennn-vs-pytorch.md)** — the
  same question at real-architecture scale: ResNet-50 on CIFAR-10, batch 128, fp32. OpenNN
  trains 1.6× faster than `torch.compile` and 2.1× faster than eager PyTorch after the
  full cudnn-frontend and resident CUDA-graph optimization pass.
* **[GPU ResNet-50 max training batch](capacity/resnet50-max-batch/resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md)** — a
  capacity benchmark on ResNet-50/CIFAR-10, fp32, one full training step. On the current RTX 4080
  run, TensorFlow XLA fits the largest batch (11,760), PyTorch `torch.compile` fits 9,216, and
  OpenNN with the `ImageDataset` BinaryFile path and prefetch-pool depth 1 (`set_batch_pool_size(1)`) fits 4,752. This is
  useful memory-regression evidence, not an OpenNN headline win.
* **[GPU dense max batch & speed](capacity/rosenbrock-max-batch/rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md)** — a
  purely dense 1000→1000→1 MLP on the GPU: OpenNN runs inference 1.43–1.56× faster than PyTorch
  (device-resident path), trains 1.30–1.35× faster at low mini-batch counts, and fits a 1.21×
  larger training batch on the same 6 GB card.
* **[GPU energy consumption](capacity/rosenbrock-max-batch/energy-consumption-gpu-opennn-vs-pytorch.md)** — joules per sample on
  the same dense MLP, integrated from GPU power (3-way, 5-run medians). OpenNN spends the least energy
  per inference of the three: 25.9 µJ vs TensorFlow 29.8 (1.15×) and PyTorch 43.5 (1.68×). On training
  it beats PyTorch 1.37× but TensorFlow's XLA path edges it — reported honestly.
* **[GPU Transformer inference](throughput/attention-speed/transformer-inference-gpu-opennn-vs-pytorch.md)** — the *Attention
  Is All You Need* forward pass (d_model 512, 8 heads, FF 2048, 6+6 layers). In bf16 (the precision
  transformers run in, where OpenNN's fused flash-attention engages), OpenNN is **the fastest of the
  three and the lead grows with sequence length: 1.10→1.58× over TensorFlow, 1.37→1.89× over PyTorch**
  (seq 128→512, 5-run medians). bf16 output validated against the fp32 CPU reference.
* **[GPU Transformer training](throughput/attention-speed/transformer-training-gpu-opennn-vs-pytorch.md)** — the same
  encoder-decoder Transformer shape in fp32 training, where OpenNN is 1.04–1.69× faster than
  PyTorch across the tested sequence lengths and spends 23% less GPU energy per sample at seq 256.
* **[Recurrent vs LSTM forecasting](quality/recurrent-lstm-forecasting/recurrent-lstm-forecasting-opennn.md)** - an OpenNN
  sequence-model coverage benchmark on UCI Beijing PM2.5 forecasting. It compares OpenNN's recurrent
  layer against its LSTM layer on CPU and GPU, reporting test RMSE, training time, winners by
  scenario, and CPU/GPU speedups. Harness is ready; Linux headline run pending.
* **[CPU inference speed](throughput/inference-speed/inference-speed-opennn-vs-pytorch-vs-tensorflow.md)** —
  a tuned dense MLP inference run. The Windows result puts OpenNN ahead of PyTorch/TensorFlow,
  but it needs the planned Linux re-measurement before becoming a public headline.
* **[Deployment size on CPU](footprint/size-cpu-opennn-vs-pytorch-vs-tensorflow.md)** — the runtime library a CPU app
  ships: a 3.2 MB OpenNN executable vs the 442 MB `libtorch_cpu` / 752 MB `libtensorflow_cc`.
* **[Deployment size on GPU](footprint/size-gpu-opennn-vs-pytorch-vs-tensorflow.md)** — a CNN's CUDA footprint. Here the
  gap narrows because NVIDIA's cuBLAS/cuDNN dominate all three; OpenNN still wins by shipping
  only the libraries its model loads.
* **[Startup latency](footprint/startup/startup-latency-opennn-vs-pytorch-vs-tensorflow.md)** — time-to-first-prediction. A
  native binary starts in tens of milliseconds; `import torch` / `import tensorflow` cost ~1–1.7 s.
* **[Baseline memory footprint](footprint/memory/peak-memory-opennn-vs-pytorch-vs-tensorflow.md)** — minimum RAM after
  empty training objects, plus GPU-ready VRAM after one tiny matrix multiply in a WSL2 CUDA run.
* **[Data capacity](capacity/data-capacity/data-capacity-opennn-vs-pytorch-vs-tensorflow.md)** — how much data fits and trains in a
  fixed RAM budget: for quote-free numeric CSV (this benchmark's data) OpenNN's memory-mapped,
  compact-matrix loader trains ~2.7× more samples than the standard `pandas.read_csv` path before
  running out of memory. (On CSVs with quoted text fields the loader falls back to a full in-memory
  copy.) ¹TensorFlow loads CSVs via the same `pandas.read_csv` path as PyTorch, so its capacity is the
  same; a `tensorflow_capacity.py` measurement is pending (the table no longer prints an unmeasured TF
  number).
* **[Dependencies & install friction](footprint/dependencies-opennn-vs-pytorch-vs-tensorflow.md)** — one self-contained
  file that runs on a clean machine, vs Python plus a 12–33 package tree.
* **[Source lines of code](footprint/loc-opennn-vs-pytorch-vs-tensorflow.md)** — the size of the native library layer
  behind each project.
* **[Application lines of code](footprint/application-loc/application-lines-of-code-opennn-vs-pytorch-vs-tensorflow.md)** — the amount
  of user code needed for the same Iris classification workflow. OpenNN uses 14 logical source
  lines, PyTorch uses 43, and TensorFlow/Keras uses 23 through its high-level `fit` API.
* **[Model export to standalone code](footprint/export/code-export-opennn-vs-pytorch-vs-tensorflow.md)** — OpenNN emits a
  trained model as compilable C/Python/JavaScript/PHP; the frameworks export model files that
  still need their runtime.

## Scope and fairness

* Most of these are **size, startup, memory, and packaging** benchmarks; the training
  precision and GPU notes add controlled training comparisons. PyTorch and TensorFlow are
  general-purpose frameworks with vast operator sets, autograd, JIT, and large ecosystems;
  OpenNN is a focused native library. The footprint differences follow directly from that
  difference in scope.
* The comparisons aim to be like-for-like on the same hardware, data shape, model, and precision;
  where a note uses a platform-specific setup, an eager/Keras competitor path, or a resident-data
  optimization, it says so explicitly. Each note states its method and how to reproduce it.
* Where a figure favors OpenNN by a smaller margin (GPU size) or carries a methodology caveat
  (LOC test files), the note says so explicitly.
