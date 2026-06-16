# Benchmark audit and fix plan for investor presentation

*Internal working document — not for investors. Written 2026-06-14 after a deep audit
of every benchmark in `docs/benchmarks/`. Goal: make every claim we show investors
survive a skeptical technical due-diligence reviewer, and add TensorFlow everywhere
PyTorch appears. Reference rigor target: MLPerf.*

---

## 0. UPDATE (2026-06-14, after fair-baseline re-runs + native-Windows profiling)

Three findings supersede parts of the audit below:

1. **TensorFlow added everywhere (fair XLA path).** New TF scripts with
   `@tf.function(jit_compile=True)`: `tensorflow_rosenbrock_throughput.py`,
   `tensorflow_resnet50_speed.py`, `tensorflow_transformer_infer.py`. PyTorch conv
   baselines got a `PT_FAST=1` mode (channels_last + `torch.compile`).

2. **OpenNN's real wins vs TF, measured 3-way (WSL):**
   - **Transformer bf16 inference — OpenNN WINS: 1.25–1.29× vs TF, 1.4–2.2× vs
     PyTorch; the lead GROWS with sequence length** (seq 768–1024 = the LLM
     regime). Fused cuDNN flash-attention is the moat XLA can't out-fuse. *This is
     the fundable headline.*
   - **Inference energy — OpenNN WINS** (25.9 vs TF 29.8, PyTorch 43.5 µJ/sample;
     5-run median, hardened harness `run_energy.py`, results JSON committed).
   - Footprint/startup/deps/LOC/export — OpenNN dominates structurally (XLA
     can't touch these).

3. **The dense "loss to TF" was a WSL2 ARTIFACT, not a real weakness.** Under WSL2
   OpenNN's bf16 dense got 0 speedup; on **native Windows** (same RTX 3060) it runs
   **1.8× faster than fp32, ≈8.3 M samples/s, matching TF bf16** — confirmed by
   Nsight Systems (`cutlass_80_tensorop_bf16_s16816gemm_relu` tensor-core kernel,
   one-time input cast). **WSL2 specifically degrades OpenNN's bf16 tensor-core
   path while leaving the frameworks unaffected, so every WSL OpenNN-vs-TF number
   understates OpenNN.** Banked dense optimizations: ReLU activation (fuses into
   the cuBLASLt epilogue, +10%), opt-in CUDA-graph inference path, GEMM-autotune
   (no effect — cuBLASLt already optimal). The fp32 two-GEMM-fusion gap remains
   (needs a fused-MLP kernel) but is moot once bf16 is the comparison.

**Net strategy:** lead with attention + energy + footprint (real, measured wins);
state dense as parity in bf16 on native Windows; re-measure GPU headlines on
native Windows (not WSL) before investor use, since WSL biases against OpenNN.

---

## 1. Executive summary

The benchmark suite is broad and the prose is unusually candid. The **structural
footprint wins** (binary size, startup, memory, dependencies, LOC, code export) are
**solid and defensible** — they reflect a real native-C++-vs-Python-framework
difference and are honestly caveated. They should be the backbone of the investor
story.

The **speed, precision, and energy claims** are where the risk is, and it is exactly
where a technical reviewer will dig. Three of them would not survive an audit as
currently published. None of the problems are fatal — the fixes are known, and in
several cases the *fair* baseline already exists in this repo and simply isn't the
one we published.

**Two factual errors must be fixed before anything is shown** (see §4.0).

---

## 2. Per-benchmark verdict

Severity: **SOLID** (show as-is) · **NEEDS-WORK** (fixable, re-run needed) ·
**WOULD-NOT-SURVIVE** (do not show until reworked).

| Benchmark | Claim | Verdict | Core issue |
|---|---|---|---|
| CPU runtime size | 3.2 MB vs 442/752 MB | **SOLID** | Honest; framework figures are full-install vs OpenNN minimal — disclosed. Add raw `du`/`ls` logs. |
| GPU deploy size | ~1.3 vs 5–6 GB | **SOLID** | Mixed measurement basis (OpenNN `ldd`-trimmed vs framework full archive) — state it explicitly. |
| Startup latency | 36 ms vs 1–1.7 s | **SOLID** | Whole-process timed, same job. OpenNN model has *more* layers (handicaps OpenNN). Add raw distribution. |
| Peak memory | 9 vs 295/521 MB | **SOLID** | Like-for-like, `ru_maxrss`, single-threaded. Add variance. |
| Dependencies | 1 file / 0 pkgs | **SOLID** | Strongest structural win. `env -i` portability proof is legit. |
| LOC | 35k vs 834k/1.8M | **SOLID** | Carefully caveated, commit-pinned, `cloc` commands shown. Confirm OpenNN's own tests are/aren't counted. |
| Code export | C/Py/JS/PHP source | **SOLID** | Real capability. Commit the generated `model.c` + compile/run log (currently absent). |
| GPU on Windows | capability, no speed claim | **SOLID** | Dated vendor facts — add citations + "valid as of" stamp. |
| Accuracy parity | R² ≈ 0.988 | **SOLID (narrow)** | Clean neutral scorer, matched setup. But toy task; headline "on par with major frameworks" overreaches. No `run_accuracy.sh` despite "5-seed avg." |
| Data capacity | 2.7× more samples | **NEEDS-WORK** | README "TF 6M" is **unmeasured** (no TF script). float64 pandas vs float32 mmap inflates gap. "memory-mapped" claim unconditional but false for quoted CSVs. |
| CPU inference | 1.25× torch/TF, ~tie ORT | **NEEDS-WORK** | Already self-flagged "Windows, re-measure on Linux." 0.4% over ORT = noise. MKL fast-VML reduced-accuracy tanh competitors don't get. |
| CNN training speed | 1.9× torch, 3.8× TF | **NEEDS-WORK** | OpenNN NHWC vs PyTorch **NCHW** (undisclosed). TF host-streamed + un-XLA'd. Mean-vs-median asymmetry. |
| ResNet-50 training | 1.6× compile, 2.1× eager | **NEEDS-WORK** | Same NHWC/NCHW layout handicap. Headline stitches runner (eager-only) + a separate "not headline" compile probe. No TF. |
| Transformer inference | bf16 1.21–1.37× | **NEEDS-WORK** | Best GPU win, defensible in bf16. fp32 column overstates (it's bf16-attention internally). Single runs, no TF. |
| Transformer training | 1.04–1.69× | **NEEDS-WORK** | Same fp32-via-bf16 caveat. 1.69× is seq-384-only (cherry-pick risk). No TF. |
| GPU energy | 1.44× infer, 2.42× train | **WOULD-NOT-SURVIVE** | `nvidia-smi` 20 Hz power proxy (not HW joules), different step regimes per engine, active-energy framing, single run. |
| Precision (2nd-order) | 1.3–1.5× MSE, 21–155× | **WOULD-NOT-SURVIVE** | OpenNN 2nd-order vs **Adam** (1st-order) everywhere; never vs PyTorch L-BFGS (exists, unused). "21–155×" mixes stopping criteria; admittedly dominated by Keras overhead. 131-param regime favors 2nd-order; inverts at scale. |

---

## 3. The cross-cutting problems (hit many benchmarks at once)

**P1 — Convolution layout asymmetry (CNN + ResNet).** OpenNN runs cuDNN in
NHWC/channels_last (`opennn/convolution_operator.cpp:46`); the published PyTorch
scripts feed contiguous NCHW (`cnn-training-speed/pytorch_cnn_speed.py:24`,
`resnet50-training-speed/pytorch_resnet50_speed.py:82`). NHWC is the fast tensor-core
layout. The repo's own probes (`pt_nhwc_probe.py`, `pt_conv_budget.py`) use
channels_last — so we know it matters and didn't apply it to the published baseline.
**A reviewer's first move is `.to(memory_format=torch.channels_last)`; this must be in
the baseline.**

**P2 — Optimized-OpenNN vs un-optimized-framework, generally.** Several GPU
comparisons pit OpenNN's best path (resident + CUDA-graph + NHWC + fused) against
PyTorch eager / TF `fit()` with no `torch.compile` / XLA / AMP. The *fair* baselines
already exist: `training-speed/pytorch_speed.py` (compile + AMP + TF32) and
`training-speed/run_speed.sh` (XLA + bf16 for TF). The published articles just don't
use them.

**P3 — No convergence/quality target (MLPerf's core).** We time fixed epochs and
assert "loss descends." Throughput without "reaches accuracy X" is gameable. Every
training benchmark needs a fixed quality gate.

**P4 — No reproducibility artifacts.** No committed raw logs, no variance/error bars,
no pinned-environment files, hand-link build scripts with machine-specific paths.
Single runs on one laptop GPU.

**P5 — Missing TensorFlow.** ResNet, Rosenbrock (max-batch/speed), both transformer
notes, and energy are OpenNN-vs-PyTorch only. **The investor ask is explicitly
OpenNN vs PyTorch *and* TensorFlow** — TF must be added to all of them.

**P6 — Non-standard scale (MLPerf gap).** ResNet-50 is on 32×32 CIFAR, not 224×224
ImageNet. Our wins live in the small-input, launch-overhead-dominated regime — the
opposite of what MLPerf measures, and our own paused ImageNet note concedes the lead
fades at real geometry. A reviewer who knows MLPerf will notice.

---

## 4. The plan (phased, in priority order)

### 4.0 — Immediate factual fixes (do first, ~1 hour, no GPU)
1. **Remove or measure the "TensorFlow 6M samples" data-capacity number** (`README.md`).
   No TF capacity script exists; the number is inferred. Either write
   `capacity/tensorflow_capacity.py` and measure, or drop the column and scope the
   claim to "vs the pandas load path."
2. **Scope the "memory-mapped loader" claim** (data-capacity article + `README.md`)
   to "quote-free numeric CSV (this benchmark's data)"; it falls back to a full heap
   copy on quoted files (`opennn/io_utilities.cpp:371-414`).

### 4.1 — Build the fair, TF-inclusive baseline harness (the core work)
For every GPU speed benchmark, produce a single reproducible harness that runs
**OpenNN, PyTorch, and TensorFlow** under each engine's *fair fast path*:
- **PyTorch:** `channels_last` for convs + `torch.compile` + AMP(bf16) + TF32
  (promote the logic already in `training-speed/pytorch_speed.py` and the conv probes).
- **TensorFlow:** `tf.function` + XLA (`jit_compile=True`) + mixed-precision policy +
  NHWC (TF's native layout). Add TF scripts where missing (ResNet, Rosenbrock,
  transformer).
- **OpenNN:** its existing fast path (already in place).
- Run **fp32 and bf16** side-by-side so the precision is explicit and matched.
- Each harness: pinned versions printed at runtime, warmup excluded, **N≥5 runs with
  median + stdev**, raw logs written to a committed `results/` dir.

Deliverables per benchmark: one runnable script, a `results/*.json` with raw numbers
+ versions + GPU + driver, and a regenerated article whose table shows
median±stdev for all three engines in both precisions.

### 4.2 — Add convergence/quality gates (MLPerf-style)
Convert each training benchmark from "fixed epochs, loss descends" to **"wall-clock to
reach a fixed target metric"** (e.g. ResNet to a target top-1 on a held-out set;
regression to a target R²/MSE). This is the single biggest credibility upgrade and
directly answers "are you fast because you don't actually learn?"

### 4.3 — Fix the precision benchmark's framing
- Add `torch.optim.LBFGS` (and a TF second-order or documented absence) to the table,
  so it's 2nd-order-vs-2nd-order, not 2nd-order-vs-Adam.
- Report **time-to-fixed-MSE** on both sides (same stopping rule), not
  OpenNN-early-stop vs frameworks-fixed-10k-epochs.
- Explicitly scope the second-order advantage to small-parameter / full-batch
  problems and note it inverts at deep-learning scale.
- Keep the (genuine) library-bug-fix disclosures.

### 4.4 — Rework or retire the energy benchmark
Either (a) re-measure with matched step regimes + N runs + report total (not just
active) energy and label it clearly as a sampled-power *estimate* with stated error,
or (b) demote it from a headline to a supporting "speed → proportional energy"
observation derived from the validated throughput numbers. Do not show 2.42× as a
hard standalone figure.

### 4.5 — Real-scale validation (MLPerf credibility)
Run at least one benchmark at standard scale: ResNet-50 on 224×224 ImageNet (or a
documented ImageNet subset) to a fixed accuracy target. Report it honestly even if the
margin narrows — a smaller, real-geometry win is far more credible than a large
toy-geometry one, and pre-empts the "you only win on 32×32" objection. (The paused
`resnet50-training-speed/IMAGENET_GEOMETRY_CONTINUE.md` is the starting point.)

### 4.6 — Reproducibility pass (applies to all)
- Commit raw result logs + per-run numbers for every published figure.
- Add pinned-environment files (`requirements.txt` / `environment.yml`, CUDA/cuDNN/
  driver versions) per benchmark.
- Replace machine-specific hand-link build scripts with a documented, parameterized
  build (or CMake target) a reviewer can run.
- One unified "results manifest" mapping every README number to its log + commit.

---

## 5. What to put in the investor deck (recommended cut)

**Lead with the structural moat** (all SOLID, all defensible):
deploy a 3 MB self-contained binary vs a multi-GB Python stack; start in 36 ms vs
~1.7 s; 9 MB training memory; zero install dependencies; export a trained model as
dependency-free C source. This is OpenNN's real, unique, un-spoofable advantage and
it needs no caveats.

**Support with the defensible compute wins** *after* §4.1 re-runs land:
bf16 transformer inference, and whichever conv/dense numbers survive the fair-baseline
re-run, each shown as median±stdev for OpenNN/PyTorch/TF in matched precision, with a
convergence gate.

**Hold back until reworked:** the 21–155× precision claim and the 2.42× energy claim
in their current form.

**Frame honestly:** OpenNN is a focused native library, not a general framework. The
footprint wins follow from that focus; the compute wins are real on the workloads
shown, at the stated scale, with stated methodology. That framing *is* the credible
pitch — and it survives diligence.

---

## 6. Suggested order of execution

1. §4.0 factual fixes (now).
2. §4.1 fair + TF-inclusive harness for **transformer inference (bf16)** first — it's
   the strongest GPU win and the cleanest to make bulletproof. Proves out the harness
   pattern.
3. §4.1 for **dense (Rosenbrock)** and **CNN/ResNet with channels_last + compile + XLA**.
4. §4.2 convergence gates + §4.3 precision rework.
5. §4.5 ImageNet-scale validation.
6. §4.4 energy decision + §4.6 reproducibility pass.
7. Regenerate `README.md` as the single source of truth, three-way (OpenNN/PyTorch/TF),
   median±stdev, every number linked to a committed log.
