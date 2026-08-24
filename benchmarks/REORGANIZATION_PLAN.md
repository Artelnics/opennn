# Benchmark suite reorganization plan

Written 2026-08-24. Working document: follow it from the Linux side of this
machine, tick things off, and edit it as reality disagrees.

The suite is not too broad — it already covers every model family, both devices,
both precisions, training and inference. It is too **redundant**: each family is
implemented three to six times over, once per metric directory, and every copy is
a place the comparison can quietly stop being like-for-like.

**48 model definitions today. 12 is the target.** Of those twelve, eleven already
exist somewhere and need reconciling rather than writing; only the TensorFlow
ResNet-50 is genuinely new.

---

## 0. Prerequisites on the Linux machine

Do these before anything else. Step 0.2 is a hard blocker.

### 0.1 Confirm the hardware and driver

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
```

Expected: `NVIDIA GeForce RTX 5070 Ti, 16303 MiB, 610.x, 12.0`.

### 0.2 BLOCKER — TensorFlow has no sm_120 kernels

The TensorFlow currently installed (2.21.0, built against CUDA 12.5.1) is
compiled for:

```
sm_60, sm_70, sm_80, sm_89, compute_90
```

There is **no `sm_120` cubin**, and this GPU is compute capability 12.0. TensorFlow
therefore reaches the card only through the `compute_90` PTX, JIT-compiled by the
driver at runtime. PyTorch 2.13.0+cu130 ships native Blackwell kernels and OpenNN
compiles directly for `sm_120`.

Comparing those three is not fair: it is two frameworks running native machine
code against one running driver-JIT-ed PTX from a two-generations-older target.
Any TensorFlow deficit measured this way is partly an artifact of the build.

Fix before measuring anything. Blackwell support lands in CUDA 12.8+, so a newer
`tensorflow[and-cuda]` wheel or an NVIDIA TF container is needed. Verify with:

```bash
python -c "
from tensorflow.python.platform import build_info as b
print(b.build_info['cuda_compute_capabilities'])
print(b.build_info['cuda_version'], b.build_info['cudnn_version'])
"
```

`sm_120` or `compute_120` must appear. Until it does, **TensorFlow numbers on this
card are provisional** — including the 6 existing RTX 5070 Ti results, which were
all produced with this build and should be re-baselined rather than trusted.

### 0.3 Confirm PyTorch really has Blackwell

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_arch_list())"
```

`sm_120` should be in the arch list.

### 0.4 Build OpenNN for this card

```bash
cmake -S . -B build-bench -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenNN_BUILD_BENCHMARKS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-bench --target benchmarks
```

`CMAKE_CUDA_ARCHITECTURES=native` also works now that the GPU is visible at
configure time. Note `OPENNN_HAS_CUDA` is a non-FORCE cache entry, so a build
directory keeps whichever CUDA decision its first configure made — to flip a tree
between CPU and CUDA, delete it and configure again.

**FlashAttention-2 is unavailable on this GPU.** `cmake/flash_attention.cmake`
covers `sm_80 86 89 90`; Blackwell has no FA2 kernel in that release, so the
target is skipped entirely. Keep `OpenNN_WITH_FLASH_ATTENTION=OFF`.

---

## 1. Reference machine — lock this

Every published number is regenerated here. Other machines may replicate, but
their numbers are never compared against these.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti · sm_120 · 16,303 MiB · 300 W cap |
| Driver | 610.43.02 |
| OS | Linux 7.0.0-28-generic x86_64, glibc 2.39 — **native, not WSL** |
| Python | 3.12.3 (`/home/artelnics/benchenv`) |
| PyTorch | 2.13.0+cu130 · CUDA 13.0 · cuDNN 9.23.02 |
| TensorFlow | 2.21.0 · CUDA 12.5.1 — **must be upgraded, see 0.2** |
| FlashAttention-2 | unavailable (Blackwell) |

Native Linux rather than WSL is deliberate, and not only preference: the ImageNet
ResNet-50 benchmark lazy-loads images per batch, so it measures conv throughput
**plus input-pipeline efficiency**. WSL's filesystem layer would contaminate
exactly that. Windows also has no native TensorFlow GPU support after 2.10, and
this machine's Windows side currently cannot link C++ at all (`rc.exe` and
`mt.exe` missing from the Windows SDK).

---

## 2. What is wrong today

All measured from the tree and the result corpus on 2026-08-24, not estimated.

| Finding | Evidence |
|---|---|
| **The dense family is defined 18 times** | 6 OpenNN + 6 PyTorch + 6 TensorFlow, across 30 files in 6 directories. Transformer 15, CNN 12, recurrent 3. |
| **Duplication has already drifted** | `opennn_higgs_maxbatch_trial.cpp` seeds with `0`; the other five dense definitions seed with `42`. The capacity benchmark has never used the same initialized network as the speed and quality ones. |
| **PyTorch and TF are not measured at their best** | Of 107 result artifacts, exactly one carries `pytorch_best` and one `tensorflow_xla`. Everything else is a precision variant, while OpenNN is timed with CUDA graphs and resident data. |
| **Results span three GPUs** | RTX 3060 Laptop (32 runs), RTX 4080 (8), RTX 5070 Ti (6); 5 benchmark ids hold results from more than one card. The audit measured a 24% swing on one machine an hour apart. |
| **Most results cannot be regenerated** | 39 of 107 record `git.dirty = true`; 42 more do not record the field. Only 26 are known clean. |
| **Scratch work sits beside evidence** | 14 `gpu-higgs-infer-sweep-*` artifacts plus 5 benchmark ids that appear in results but not in the manifest. |
| **The transformer family uses three datasets** | `attention-speed` runs on a synthetic LCG corpus, capacity on WMT14, energy on `chat_pairs`. |
| **CNN has no TensorFlow at all** | `run_imagenet_resnet50.py` supports `opennn`, `pytorch_fast`, `pytorch_eager`. It is a two-way benchmark. |

---

## 3. Decisions

The cost that matters is **preparation, not runtime** — a HIGGS training epoch is
1.77 s, so runs are cheap; what hurts is how many distinct things a person must
author, verify and keep correct. In that unit the dimensions split sharply:

| Free — a flag on existing code | Expensive — code to author and verify |
|---|---|
| precision (fp32 / bf16) | **a family** = 3 definitions that must be provably equivalent |
| device (CPU / GPU) | **a protocol** = 1 driver, reused by every family |
| mode (train / inference) | **an engine invocation path** = 1 per execution mode |
| batch size, repeat count | |

| Dimension | Decision |
|---|---|
| Framework | Keep all three. |
| Model family | **Keep all four.** Breadth was never the cost; redundancy was. |
| Precision / device / mode / batch | Keep — they are flags, cutting them saves nothing to prepare. |
| Execution mode | **Compiled only in the matrix**: OpenNN, `torch.compile`, `jit_compile`. Eager becomes a once-per-release reference number, not a cell. Three engines per cell, not five. |
| GPU model | **One reference GPU** (section 1). Others replicate; never compared. |
| Operating system | **Linux only** for published numbers. Zero of 107 existing results are Windows, so nothing is lost. |
| Batch as an axis | Removed — but **both batch benchmarks survive** as their own tracks (section 4). |
| Energy | Separate opt-in track; needs an idle machine. |

---

## 4. Tracks

| Track | Question | Batches | Cadence |
|---|---|---|---|
| **Speed** | Throughput at a common batch, engine vs engine | 1 fixed | after any meaningful change |
| **max-batch** | Largest batch that trains under a memory cap, and its memory cost | ~9, ascending to OOM | before publishing |
| **peak-batch** | Where each engine's throughput peaks when free to choose | ~9, geometric | before publishing |

The sweeps run less often because their answers barely move. The fixed-batch speed
track is the sensitive regression detector. To cover the gap, the speed run records
peak memory at its fixed batch — the peak monitor already samples it, so it costs
nothing and warns when the ceiling shifts.

`max-batch` is the most defensible three-way claim in the suite, because its metric
is measured identically for every engine by an external instrument.

Evidence that peak-batch earns its place, from the last HIGGS run: OpenNN and
PyTorch peak at batch 14,000 while TensorFlow peaks at 896,000 — its last working
batch. OpenNN's curve *declines* after its peak, 10.38M → 8.83M samples/s. Any
single fixed batch therefore flatters somebody.

---

## 5. Datasets

One real dataset per family for published numbers, one synthetic stand-in for CI.

| Family | Headline | CI / smoke |
|---|---|---|
| Dense | HIGGS | — |
| CNN | **ImageNet subset**, 1000 classes × N images, 224×224 | `imagenet_like` proxy |
| Transformer | **WMT14 EN-DE** (`prepare_wmt14.py` already exists) | synthetic LCG corpus |
| Recurrent | Beijing PM2.5 | — |

**ImageNet without shipping ImageNet.** ILSVRC2012 needs registration and cannot be
redistributed, and `DATA_POLICY.md` forbids committing datasets while allowing small
metadata. So `prepare_imagenet_subset.py` takes a path to *your own* ImageNet tree,
selects deterministically (sorted synsets, seeded pick), and writes a **manifest of
synset + filename + sha256** into `$OPENNN_BENCH_DATA/imagenet_subset/`. Only the
manifest is committed.

Keep 1000 classes so the true 2048×1000 head is preserved and numbers stay
comparable to published ResNet-50. A 10-class subset changes the network.

**A subset cannot carry an accuracy claim.** At 50–100 images per class ResNet-50
will not reach a meaningful top-1, so the CNN quality gate is **cross-engine loss
agreement**, not an accuracy target. That still catches a precision or fusion
difference masquerading as a speed win, which is the gate's real job here.

Moving ResNet-50 from CIFAR-10 (32×32) to 224×224 makes it a substantially heavier
benchmark — runs become minutes, not seconds. Still the right call: ResNet-50 on
32×32 was never really ResNet-50.

---

## 6. Metric definitions

**Speed.** Samples/s, warmup excluded, median of repeated runs, spread reported
alongside so a thermally-drifting session is visible in the artifact rather than
averaged into a claim. Ratios within one session are the durable output; absolute
samples/s is provenance and is never compared across sessions.

**Memory.** One definition wherever a memory number appears in a three-way table:

- **GPU** — `nvidia-smi` device-used memory sampled at 50 ms by a peak monitor,
  minus the idle reading taken before the run.
- **CPU** — `ru_maxrss`, which all three engines already self-report, including
  OpenNN's C++ trial.

Both are peak, not average; the artifact records which via `peak_metric`.
Whole-device is the honest choice — it counts the CUDA context and the allocator's
cached blocks, because that memory really is unavailable to anything else.

> **`torch.cuda.max_memory_allocated()` must never appear in a comparison table.**
> It is allocator-internal, excludes the CUDA context and freed-but-cached blocks,
> and has no OpenNN or TensorFlow equivalent, so it flatters PyTorch by
> construction. Keep it as a labelled PyTorch diagnostic only.

---

## 7. The contract — goes into `suite.json`

Adding a family or protocol later must never invalidate what came before. That only
holds if these are fixed now and never quietly change. Changing any one of them
forces a full re-run.

1. **The reference machine** — section 1, including driver and framework versions.
2. **Framework build targets** — TensorFlow and PyTorch must both carry `sm_120`.
   This is the one that silently regresses on a `pip install --upgrade`; assert it
   at runtime and fail the run if it is missing.
3. **How each engine is invoked at its best** — OpenNN with CUDA graphs and resident
   data; PyTorch with `torch.compile`; TensorFlow with `jit_compile`. Written as
   commands, not intentions.
4. **The fixed reference batch per family** — chosen once and never re-tuned to
   flatter a result. Bootstrap rule needed: see open question 9.2.
5. **The dataset, pinned by manifest** — changing which images are in the subset
   changes the benchmark.
6. **The two metric definitions** — section 6, including the `max_memory_allocated`
   ban.
7. **The quality gate and its tolerance** — see open question 9.1.
8. **The result schema, with a session id** — comparisons are valid only among
   numbers sharing a session id. A dirty tree writes to `results/scratch/`, never to
   the evidence store.

---

## 8. Target structure

Organize by **family**, not by metric. A family owns one definition per framework;
protocols are family-agnostic drivers that consume them. Drift becomes structurally
impossible rather than something to remember.

```text
benchmarks/
├── suite.json                 the contract, and which cells are official
├── common/                    one shared harness
│   ├── gpu.py                 PeakMonitor, nvidia_used_mib, measure_idle, cooldown
│   ├── provenance.py          git commit, dirty gate, session id, framework versions
│   └── metrics.py             currently duplicated by hand in two copies
├── families/                  3 definitions each — 12 files total
│   ├── dense/                 ← 18 definitions across 6 directories today
│   │   ├── model_opennn.cpp
│   │   ├── model_pytorch.py
│   │   ├── model_tensorflow.py
│   │   ├── prepare_higgs.py
│   │   └── quality_gate.py
│   ├── cnn/                   ← 12 today; model_tensorflow.py does not exist yet
│   │   ├── prepare_imagenet_subset.py
│   │   ├── imagenet_subset.manifest      synset + filename + sha256, committed
│   │   └── prepare_imagenet_like.py      CI proxy, no licence wall
│   ├── transformer/           ← 15 today, across 3 datasets
│   │   ├── prepare_wmt14.py              headline; already exists
│   │   └── make_synthetic_corpus.py      CI proxy, deterministic LCG
│   └── recurrent/             ← 3 today, already flat
├── protocols/                 family-agnostic; each works for all four
│   ├── speed.py               fixed batch · train|infer · cpu|gpu · fp32|bf16 · +peak memory
│   ├── capacity.py            max batch under a stated memory cap
│   ├── quality.py             accuracy and convergence
│   ├── peak_batch.py          throughput swept over batch        (add when wanted)
│   └── energy.py              idle-delta, GPU only               (add when wanted)
├── footprint/                 unchanged — LOC, export, baseline memory, startup
├── results/                   reviewed artifacts only
│   └── scratch/               gitignored — probes, sweeps, dirty-tree runs
└── reports/                   the published notes
```

On the C++ side the equivalent of a protocol is a **subcommand**, not a separate
binary: six OpenNN dense programs become one with `train | infer | capacity |
quality` modes. `opennn_higgs_cpu.cpp` already uses that pattern (`train`/`infer`),
so it is the template.

**About what gets deleted.** The six OpenNN dense programs are 1,717 lines, but the
duplicated *model construction* is only about 100 of them. The other ~1,600 are
genuinely different driver logic — subcommands, OOM handling, convergence targets —
which becomes the protocol layer. So the deletion target is the ~80 lines of
duplicated model construction plus five redundant `main()` wrappers, not five whole
files.

---

## 9. Open questions — settle before writing code

### 9.1 The quality gate is not yet operational

Contract item 7 says a cell failing the gate reports no speed number. But loss
trajectories across three frameworks will never match exactly: different RNG
algorithms, different initialization, different data ordering, nondeterministic
GPU kernels. Unanswered: what tolerance counts as agreement, and is initialization
*matched* or merely seeded?

**Proposed answer — matched initialization via weight exchange.** It converts an
unanswerable statistical question into a deterministic one. OpenNN has
`save_parameters_binary()` / `load_parameters_binary()`, and the format is a fixed
header (magic, float count, payload bytes, layout fingerprint, FNV-1a payload hash)
followed by a flat float dump. There is **no ONNX support**, so this is the exchange
path.

The trick that avoids reimplementing the fingerprint: have OpenNN write a reference
file first, then swap **only the payload** and recompute the FNV-1a hash, leaving
magic and fingerprint bytes untouched. `set_parameters_pytorch()` also already
exists in the library, so PyTorch's initialization convention is partly implemented.

**Spike to run once the build works:** build the dense model in OpenNN, save
parameters, swap in PyTorch weights in OpenNN's parameter order, load back, run one
forward pass on one fixed batch in all three engines, compare outputs. If they agree
to ~1e-6 in fp32 the gate becomes deterministic and the tolerance is measured rather
than guessed. If they do not, you have found where the definitions differ — which is
the reconciliation work anyway, surfaced early and cheaply.

This spike doubles as the **acceptance test for consolidation**: "does the merged
definition reproduce the same forward pass as the one it replaced?"

### 9.2 The reference batch is circular

Contract item 4 says choose it from the peak-batch curve, but peak-batch is a
per-release track that runs after the matrix exists. Needs a bootstrap rule —
probably: start from the current fixed batches, and re-pin once after the first
peak-batch run, then freeze.

### 9.3 Unplaced files

- `benchmarks/analysis/analyze_joint_arena.py` — in no manifest and no track.
- `benchmarks/energy/max-batch-to-target/` — empty leftover directory.
- The `footprint/` track has been carried along as "unchanged" without ever being
  examined.

---

## 10. Sequence

Nothing published changes meaning until step 6.

**Step 0 — prerequisites.** Section 0, especially the TensorFlow sm_120 fix.

**Step 1 — the duplication ledger.** Before deleting anything, diff the duplicates
per family and record, for each: what is byte-identical, what silently diverges
(like that `set_seed(0)` vs `set_seed(42)`), and what is genuinely distinct driver
logic. Needs no compiler. This is what makes deletion safe rather than hopeful — if
a number moves later, the ledger says why.

| Family | Definitions | Ledger |
|---|---|---|
| dense | 18 → 3 | |
| transformer | 15 → 3 | |
| CNN | 12 → 3 | |
| recurrent | 3 → 3 | already flat — but `run_forecasting.py` defaults to `--frameworks opennn`, so it is **not** currently running as a three-way comparison |

**Step 2 — lock the contract.** Fill in `suite.json` from section 7, having settled
9.1 and 9.2. Move the 14 sweep artifacts and 5 unregistered ids to
`results/scratch/`.

**Step 3 — extract `common/`.** Lift the GPU monitor and provenance helpers the
earlier audit identified as ~350 duplicated lines. Port one runner at a time and
diff a JSON artifact before and after.

**Step 4 — consolidate, worst first.** dense 18 → 3, then transformer 15 → 3 (and
settle it on WMT14; retire `chat_pairs`), then CNN 12 → 3 (and write the missing
TensorFlow ResNet-50, plus `prepare_imagenet_subset.py`). Use the 9.1 spike as the
acceptance test for each merge. Recurrent needs its `--frameworks` default fixed.

**Step 5 — write the drivers.** `speed.py`, `capacity.py`, `quality.py` against the
twelve definitions. This is where precision, device and mode become flags instead of
files.

**Step 6 — re-baseline in one session.** Clean tree, reference GPU, one sitting, one
session id. The first numbers publishable as a coherent comparison.

**Step 7 — retire the non-conforming results.** They stay in git history and leave
the reports. Any claim still resting on a pre-baseline number gets restated or
withdrawn — including the 6 existing RTX 5070 Ti results, whose TensorFlow numbers
came from the sm_120-less build.

---

## Notes on measurement, carried over from the engineering audit

- Absolute throughput moves a lot with GPU clock and thermal state: the same bf16
  configuration measured 6,994 and 8,682 samples/s in two sessions an hour apart.
  Only A/B comparisons taken back-to-back in one session mean anything.
- The cuDNN fusion probe found **0 of 9 engines** for three fusion patterns — but
  that was measured on **sm_86**. The audit says explicitly to re-run before
  concluding anything about different hardware. Worth re-running
  `cudnn_fusion_probe` on sm_120.
- The conv workspace cap is already tuned: `auto` beat every fixed cap.
- Before any dead-code deletion in the library itself, read
  [Before deleting anything: Neural Designer](../docs/status/engineering-audit.md#before-deleting-anything-neural-designer).
