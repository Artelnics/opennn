# GPU on Windows: OpenNN vs PyTorch vs TensorFlow

Most GPU comparisons assume Linux. On Linux every major framework has a complete CUDA story,
so the only question is which is fastest. **Windows is different.** Here the question comes
*before* speed: on Windows, the other frameworks' GPU paths are **incomplete** — a feature is
missing, disabled, or only reachable through a Linux virtual machine. OpenNN's GPU path is the
same complete, native-CUDA path it is on Linux. This note is about that gap in *capability*,
not throughput.

The short version:

| On Windows, with an NVIDIA GPU | Trains on GPU? | Its fast path works? |
|---|---|---|
| **OpenNN** | **Yes, natively** | **Yes** — CUDA graphs, capturable optimizer, all native |
| PyTorch | Yes, natively | **No** — torch.compile (its kernel-fusion path) does not work on Windows |
| TensorFlow | **No** (native GPU dropped after 2.10) | — only via WSL2 (a Linux VM) or a limited plugin |

## What is missing on Windows, framework by framework

### PyTorch — loses its kernel-fusion compiler

PyTorch trains on the GPU on Windows. What it loses is **`torch.compile`**, the feature that
gives modern PyTorch much of its speed. `torch.compile` fuses a model's many small operations
into a few large GPU kernels through its Triton backend — and Triton has no working toolchain on
Windows. So on Windows, PyTorch falls back to **eager mode**: every operation is launched
separately, with no fusion.

This is not a minor footnote. On a small, memory-bound training step, the cost is dominated by
how many separate kernels you launch and how many times intermediate tensors travel to and from
GPU memory — exactly what fusion removes. `torch.compile` is the headline reason PyTorch is fast
on Linux; on Windows that headline feature is simply absent.

OpenNN has **no equivalent dependency**. Its overhead-reducing path — capturing the whole
training step into a CUDA graph and replaying it, with a purpose-built capturable optimizer —
is plain CUDA-runtime C++. It behaves identically on Windows and Linux.

### TensorFlow — no native GPU at all

This is the starkest gap: **TensorFlow dropped native Windows GPU support after version 2.10.**
On current TensorFlow, a Windows machine with an NVIDIA GPU has three options, and none is a
native Windows GPU build:

* **CPU-only** TensorFlow, or
* GPU only inside **WSL2** — a Linux virtual machine running on Windows, or
* the third-party **DirectML** plugin, which is limited and not the mainline CUDA path.

So "TensorFlow GPU training on Windows" is not slow — for the native toolchain, it does not
exist. OpenNN runs CUDA directly on Windows with no virtual-machine layer.

## The cost of the workarounds

The usual way to recover the missing features on Windows is WSL2 — run Linux inside Windows to
get `torch.compile` and native TensorFlow GPU back. That path is real, but it is not free, and
it is not Windows:

* GPU access goes through a paravirtualization layer (GPU-PV), which adds overhead and breaks
  some tooling — NVIDIA's `nsys`/CUPTI profiling, for example, does not work under WSL2.
* The Linux guest needs its own memory budget carved out of the host.
* It is a Linux environment. A result obtained in WSL2 is a Linux result running on Windows
  hardware — it does not make the framework's *Windows* GPU path any more complete.

OpenNN needs none of this. It is a native Windows executable that links CUDA directly.

## Inference on Windows

For *inference* — running a trained model — the picture changes. The relevant comparison there is
not "can it run on the GPU" but the deployment characteristics OpenNN's other benchmarks already
cover — runtime size, startup latency, dependencies — where a self-contained native binary still
has the edge. See the
[deployment benchmarks](../README.md) for those head-to-heads.

OpenNN trains and infers the same native model end-to-end with no export step, avoiding the
friction (operator coverage, opset versions, dynamic shapes) that a conversion to a separate
inference format introduces.

## Why OpenNN's path is complete on Windows

The features doing the work on Windows are all owned by OpenNN at the CUDA level, with no reliance
on a fusion compiler, a Python runtime, or a Linux VM:

* **Native CUDA, native binary** — no interpreter, no framework runtime to load, the same build
  on Windows and Linux.
* **Whole-step CUDA-graph capture and replay** — the step's kernel sequence is recorded once and
  replayed, removing per-launch overhead. This is the eager-mode cost PyTorch cannot escape on
  Windows.
* **A capturable optimizer** — the optimizer step is written so a single captured graph replays
  correctly each iteration (iteration-dependent values live on the device), which is what makes
  the graph path usable for training, not just inference.
* **GPU-resident data** — the dataset can live on the device, so training steps read it in place
  instead of copying batches from the host each iteration.

None of these depend on anything Windows withholds. That is the whole point: OpenNN's GPU
capability does not regress when you move it from Linux to Windows, and the other frameworks' do.

## Scope and fairness

* This note is about **capability and completeness on Windows**, not measured throughput. It makes
  no samples-per-second claim. Where the frameworks' GPU paths *are* complete (notably Linux),
  PyTorch's `torch.compile` and TensorFlow's native GPU are strong — and on Linux the comparison
  is a genuine speed race, not an availability one.
* The limitations described are platform facts (Triton/`torch.compile` on Windows; TensorFlow
  native GPU dropped after 2.10), not measurements — they hold regardless of hardware.
* WSL2 is a legitimate way to run these frameworks' full GPU paths on Windows hardware. The point
  here is narrower: it is a Linux environment, so it does not make the framework's *native Windows*
  GPU support complete.
