# Presentation claim matrix

*Internal planning document. Use this file to decide which benchmark claims are
ready for high-stakes presentations, which need a fresh run, and which should
remain supporting context.*

## Decision levels

| Level | Meaning |
|---|---|
| Lead now | Strong enough for a main slide with current benchmark notes, provided raw commands, versions, and artifacts are attached. |
| Lead after rerun | Strong result, but it needs a fresh provenance-rich result JSON before use as a main slide. |
| Supporting only | Useful context, but not a primary proof point. |
| Internal only | Useful for migration, development, or draft slides; not a public claim. |
| Hold back | Do not use as a slide headline until the listed work is complete. |
| Historical | Retained as engineering history; superseded for new claims. |

These levels mirror the lifecycle labels in
[`benchmark_manifest.json`](benchmark_manifest.json).

## Claim matrix

| Level | Area | Safe wording | Evidence | Required before use | Caveat |
|---|---|---|---|---|---|
| Lead now | CPU deployment size | A CPU OpenNN application ships as a small native binary: about 3.2 MB in this benchmark, versus 22 MB for ONNX Runtime, 442 MB for PyTorch, and 752 MB for TensorFlow runtime libraries. | [CPU size](size-cpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Attach `ls`/`du` output, `ldd`, build flags, versions, and commit. | Size is not capability; PyTorch and TensorFlow carry much broader runtimes. |
| Lead now | Startup latency | OpenNN reaches first prediction in tens of milliseconds: 36 ms here, compared with 237 ms for ONNX Runtime, 1,005 ms for PyTorch, and 1,685 ms for TensorFlow. | [Startup latency](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Attach timing logs, command lines, warmup protocol, versions, and commit. | Measures cold start plus one tiny model, not steady-state throughput. |
| Lead now | Baseline memory footprint | In the 2026-07-05 WSL2 CUDA run, OpenNN used 195.2 MB baseline RAM versus 516.2 MB for PyTorch and 871.2 MB for TensorFlow; GPU-ready VRAM after one tiny matrix multiply was 119.0 MB, 155.0 MB, and 121.0 MB respectively. | [Memory footprint](peak-memory-opennn-vs-pytorch-vs-tensorflow.md) | Attach raw runner output, `nvidia-smi`, framework versions, and `results/baseline-memory-footprint-wsl2-20260705T110753Z.json`. | GPU-ready VRAM mostly reflects CUDA runtime/context and first math-backend setup before tensors are resident. |
| Lead now | Install and deployment friction | OpenNN deploys as one native executable with no Python package tree; the comparison installs thousands of files for the Python stacks. | [Dependencies](dependencies-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Attach package lists, file counts, `du` output, and clean-machine run proof. | Build-time dependencies are separate from deployed artifact dependencies. |
| Lead now | Standalone code export | OpenNN can export a trained model as standalone source code in C, Python, JavaScript, and PHP; the other paths require a runtime. | [Code export](code-export-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Store the generated source, compile command, run output, and dependency check. | Best for supported feed-forward style models; very large or exotic networks may be better served by a runtime. |
| Supporting only | Legacy accuracy parity | On the historical Rosenbrock accuracy test, OpenNN, PyTorch, and TensorFlow reach statistically similar R2 values around 0.987 to 0.988. | [Accuracy](accuracy-opennn-vs-pytorch-vs-tensorflow.md) | Replace with HIGGS accuracy, log loss, and ROC AUC before making a dense quality claim. | Keep as engineering history; do not use as the active dense headline. |
| Lead now | GPU deployment size | For the measured CNN GPU deployment, OpenNN ships a smaller CUDA footprint than ONNX Runtime, PyTorch, and TensorFlow by linking only the needed NVIDIA components plus a small binary. | [GPU size](size-gpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Attach `ldd` output, resolved library list, package/archive sizes, CUDA/cuDNN versions. | NVIDIA libraries dominate every GPU deployment, so the gap is smaller than on CPU. |
| Lead after rerun | Transformer inference | On bf16 Transformer inference, OpenNN is the fastest of the three in the current run, and the lead grows with sequence length. | [Transformer inference](transformer-inference-gpu-opennn-vs-pytorch.md), [current JSON](results/gpu-transformer-inference-20260614T211356Z.json) | Rerun with `run_transformer.py` so the result JSON includes full git metadata, raw per-run output, commands, hardware, framework versions, and dirty status. | Current JSON lacks a real commit; do not use the old artifact as final evidence. |
| Supporting only | Native source size | OpenNN has a much smaller native implementation surface than the compared framework runtimes. | [LOC](loc-opennn-vs-pytorch-vs-tensorflow.md) | Attach `cloc` commands and pinned upstream revisions. | Smaller source size is context, not proof of better quality or broader capability. |
| Supporting only | Application code size | On a small Iris workflow, OpenNN uses 14 logical source lines versus 43 for PyTorch and 23 for TensorFlow/Keras. | [Application LOC](application-lines-of-code-opennn-vs-pytorch-vs-tensorflow.md) | Keep snippets aligned to the same workflow and rerun `count_lsloc.py` after changes. | LOC depends on coding style; Keras concision does not imply a smaller runtime. |
| Supporting only | Data capacity | For quote-free numeric CSV using the default in-memory paths, OpenNN fits about 2.7x more samples than the common pandas-to-tensor path under the same memory cap. | [Data capacity](data-capacity-opennn-vs-pytorch-vs-tensorflow.md) | Store the capped-run CSV, cap method, generated data command, and versions. | Python frameworks can stream data with custom pipelines, which changes the ceiling. |
| Supporting only | Second-order precision workflow | On the historical Rosenbrock precision task, second-order methods reach the lower error floor; OpenNN exposes them as native one-line options, while PyTorch LBFGS needs a closure loop and core Keras lacks this path. | [Precision](precision-opennn-vs-pytorch-vs-tensorflow.md) | Reframe or rerun on HIGGS with classification metrics before using it with dense claims. | Narrow small-parameter CPU task; do not generalize to deep-learning scale. |
| Supporting only | Native Windows GPU capability | OpenNN has a complete native CUDA path on Windows; competing stacks have platform-specific gaps depending on training, inference, or compiler features. | [GPU on Windows](gpu-on-windows-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Cite current vendor docs and keep the date visible. | Capability matrix, not a throughput benchmark. |
| Hold back | GPU energy inference | The current dense MLP sampled-power run is Rosenbrock-based and should be treated as historical. | [Energy](energy-consumption-gpu-opennn-vs-pytorch.md), [current JSON](results/gpu-dense-rosenbrock-energy-20260614T195119Z.json) | Rerun with the HIGGS dense contract and a fresh artifact containing a real commit. | GPU board sampled power only; TensorFlow is lower on dense training energy in this run. |
| Hold back | CPU inference speed | Do not headline the tuned CPU inference number yet. | [CPU inference](inference-speed-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Remeasure on the reference Linux x86_64 box, attach raw logs and checksum/quality gates. | Current Windows result is valid but platform-specific; ONNX Runtime is essentially tied. |
| Hold back | CNN training speed | Do not headline the MNIST CNN speedup yet. | [CNN training](cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md) | Add optimized PyTorch `torch.compile` and TensorFlow XLA paths or label the result as plain-loop only. | Current comparison is useful but not the fastest fair competitor path. |
| Hold back | ResNet-50 training speed | Do not headline the CIFAR ResNet-50 speedup until the evidence is broadened. | [ResNet-50 training](resnet50-training-speed-gpu-opennn-vs-pytorch.md) | Add repeated-run artifacts, keep `torch.compile` in the official runner, add TensorFlow or explain scope, and test real image geometry. | CIFAR 32x32 is launch-overhead heavy; the margin can narrow at 224x224. |
| Hold back | Dense GPU speed and max batch | Do not headline dense GPU speed or max-batch numbers yet. | [Dense GPU](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md), [HIGGS migration](DENSE_HIGGS_MIGRATION.md) | Rerun on HIGGS, native Windows, and reference Linux; archive VRAM-cap evidence and replace hand-link scripts with CMake targets. | WSL2 affects OpenNN bf16 dense behavior; training throughput depends heavily on batches per epoch. |
| Hold back | Transformer training | Do not headline Transformer training yet. | [Transformer training](transformer-training-gpu-opennn-vs-pytorch.md) | Add repeated-run JSON, raw logs, TensorFlow where applicable, and a quality gate beyond decreasing loss. | Current result is OpenNN vs PyTorch only and uses a synthetic corpus. |
| Hold back | Training energy | Do not headline training energy as an OpenNN win. | [Energy](energy-consumption-gpu-opennn-vs-pytorch.md) | If used, show all three rows and state that TensorFlow is lower on dense training energy in the current sampled-power run. | The honest result is mixed: OpenNN beats PyTorch, while TensorFlow leads dense training energy. |

## Artifact rule

No numeric claim should move from this file into a presentation unless the linked
benchmark has:

- a result artifact or raw command log,
- a git commit and dirty status,
- exact command lines and environment flags,
- framework, CUDA/cuDNN, driver, CPU/GPU, and OS versions,
- per-run values, not only a final median,
- the relevant caveat shown on the slide or in speaker notes.
