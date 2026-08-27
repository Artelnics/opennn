# The contract

Fixed. Changing any item here invalidates every result taken under the old
version, so a change means a re-run, not a footnote.

## 1. The reference machine

Every published number comes from this machine. Others may replicate; their
numbers are never compared against these.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti · `sm_120` · 16,303 MiB · 300 W |
| Driver | 610.43.02 |
| CPU | Intel i7-14700F |
| OS | Linux, native — not WSL |
| CUDA · cuDNN | 13.3.73 · 9.25.0 |
| PyTorch | 2.13.0+cu130, arch list includes `sm_120` |
| MKL | 2026.0.1 for OpenNN; PyTorch's wheel bundles its own |
| Python | 3.12.3 |

Native Linux is deliberate. The CNN family lazy-loads images per batch, so it
measures convolution throughput *plus* input-pipeline efficiency, and WSL's
filesystem layer would contaminate exactly that.

## 2. TensorFlow is not in the matrix

No TensorFlow build ships `sm_120` kernels — not 2.21.0, the current release,
and not the nightly, which pulls CUDA 12.9 and still targets only `sm_60`
through `sm_90`. It runs here on driver-JIT'd PTX and says so at import.

Measured cost: ~25% on a 4096³ matmul. That figure is *unattributable*, because
there is no native build to difference against, and being unattributable is
what disqualifies it — a published deficit would partly measure NVIDIA's
release schedule and would move on a rebuild.

Every cell records `engines: [opennn, pytorch]` until an NGC container changes
this. Verify any candidate build by dumping its kernels, not by reading
`build_info`, which the nightly leaves empty:

```bash
cuobjdump --list-elf .../tensorflow/libtensorflow_cc.so.2 | grep -oE "sm_[0-9]+" | sort -u
```

## 3. Each engine at its best

Not "as it comes". An engine measured below its own ceiling makes the other
look good for the wrong reason, and this is the item most likely to drift
silently.

| | GPU | CPU |
|---|---|---|
| OpenNN | captured CUDA graph, device-resident split, whole batches only | same definition, `Device::CPU`, **MKL** |
| PyTorch | `torch.compile`, `channels_last`, TF32 enabled | **eager**, MKL |

**PyTorch runs eager on CPU on purpose, and it is measured, not assumed.**
Inductor's CPU codegen loses on a small stack of GEMMs: dense CPU training at
batch 4,096 gives eager 93,156 samples/s against compiled 83,722 here, and the
previous suite measured the same ordering on different hardware (41,523 against
29,449). Compiling anyway would hand OpenNN a win against a PyTorch nobody
would ship. `PT_COMPILE_MODE` overrides either way, so the choice stays
measurable rather than baked in.

This is not hypothetical. Measured against an *eager* PyTorch, dense training
read 1.29×; against a compiled one, 1.06×. The first number was not a lie about
OpenNN — it was a lie about PyTorch.

**Both engines run on MKL, and every run says which.** OpenNN's CPU kernels
dispatch to Eigen or MKL, chosen at runtime by `Configuration::set_blas`; Eigen
is the library default, so a plain build behaves like a plain build, and the
benchmark drivers opt in. PyTorch's wheels are MKL-backed and report
`BLAS_INFO=mkl`.

Leaving OpenNN on Eigen would measure the BLAS and call it the framework. The
same dense CPU inference reads 169,152 samples/s on Eigen and 213,422 on MKL —
against PyTorch's ~167,000 that is the difference between 1.02× and 1.28×, and
neither number is wrong about OpenNN. Both engines print a `blas=` line and the
artifact records it per launch, so the backend is a property of the *run* and
readable from the result, rather than something inferred from which build
directory the binary came out of.

One consequence to keep in view: linking MKL also makes Eigen use it for
BLAS/LAPACK, independently of the dispatch flag. An MKL-linked build fails
`CorrelationsTest.LogisticCorrelation` on both paths and
`GeneticAlgorithmTest.SelectsParsimoniousSubset` on the Eigen one; both pass
with MKL absent. That is a linkage sensitivity, not a benchmark result, but it
is why MKL is opted into rather than defaulted to.

## 4. Same work, or no comparison

Both gates run before any number is reported, and both are recorded in the
artifact.

**Shape.** Sample count, sequence length, vocabulary and parameter count must
agree across engines. Two findings that arrived this way:

- OpenNN's tokeniser emitted 158-position sequences where PyTorch's emitted
  128 — 23% more arithmetic for the same corpus, invisible in samples/s. Fixed
  by pre-tokenising with OpenNN's own rule, so both read the same stream.
- `nn.Transformer` appends a final `LayerNorm` to encoder and decoder, 2,048
  parameters OpenNN's model does not have. Fixed by building the stacks
  directly with `norm=None`.

Current parameter counts, matching exactly: dense 1,080,321 · CNN 25,557,032 ·
transformer 74,878,496 · LSTM 73,857.

**Quality.** Accuracies must agree within tolerance, per batch, across engines.
A speed win bought by computing something different is not a speed win.

## 5. Metrics

**Throughput.** Samples/s, warmup excluded, median of repeated rounds, every
round kept in the artifact so a drifting session is visible rather than
averaged into a claim. Ratios within one session are the durable output;
absolute samples/s is provenance.

**Peak memory.** On GPU, `nvidia-smi` device-used sampled at 20 ms, minus the
idle reading taken before the run. Whole-device on purpose: it counts the CUDA
context and the allocator's cached blocks, because that memory really is
unavailable to anything else. On CPU, the process's peak *anonymous*
resident set, sampled from `RssAnon`, with `RssFile` recorded beside it.

Anonymous, not total, and the distinction decides the result. `RssFile` counts
pages backed by a mapped file, and OpenNN's reader mmaps its CSV rather than
copying it -- so total RSS charges it ~150 MiB for a file it never allocated,
while an engine that reads the same file onto the heap is charged the same for
memory the kernel cannot reclaim. Counting mapped pages penalises the cheaper
strategy. On one 500k-row cell: by total RSS, OpenNN 428 MiB and PyTorch 875;
by anonymous, 270 and 563. Same runs; only the second pair answers "how much
memory does this demand".

`RssAnon` has no kernel-maintained high-water mark, so unlike `VmHWM` it is
sampled rather than read once at exit. Each artifact records which metric was
used via `memory_metric`, because the CPU and GPU quantities are not the same
thing and must not share a column.

> `torch.cuda.max_memory_allocated()` never appears in a comparison. It excludes
> both, has no OpenNN equivalent, and flatters PyTorch by construction. Keep it
> as a labelled PyTorch diagnostic if wanted.

**Energy.** Board power integrated over the engine's own timed window, between
the marks it prints. Reported only when the window held at least four samples;
otherwise the artifact says so. A short run has no energy figure, and `0.0000
Wh` would be a claim rather than an absence.

The dense family's epochs are milliseconds, so its energy cell needs `--epochs`
raised until the window clears about a second. That is a per-family setting,
not a matter of taste.

**CPU runs have no energy figure at all.** It would need a RAPL counter, which
is not wired up, and the GPU's draw during a CPU run is an idle card. The
artifact says `energy_measurable: false` rather than reporting that idle draw
as though it were the workload's.

## 6. CPU cells

The CPU has three sources of variance where the GPU has one, and only one of
them can be removed without root.

**Removed, by the runner.** This is a hybrid part: 8 performance cores at
5,400 MHz and 12 efficiency cores at 4,200, and the scheduler decides which a
thread gets. That is worse than clock drift — it is discrete and per-thread, so
two identical runs can differ by ~22% purely on placement, with nothing in the
result to say so. Every CPU launch is therefore `taskset`-pinned to the P-cores,
with a thread count set identically for both engines at one per *physical*
core, since SMT siblings share execution units. The artifact records the core
span, the thread count and the excluded E-cores.

**Not removed, and recorded instead.** The governor and turbo state need root:

```bash
sudo cpupower frequency-set -g performance
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo
```

Until those are set, cores scale between 800 and 5,400 MHz and turbo boosts
until the package hits a limit and then drops — so the first seconds of a run
are faster than the rest, and which engine gets them depends on ordering. Each
artifact carries `cpu.governor` and `cpu.turbo_enabled` so a reader can tell an
opportunistic run from a pinned-down one.

**No energy figure on CPU.** It needs a RAPL counter, which is not wired up.
The artifact says `energy_measurable: false` rather than reporting the idle
GPU's draw.

## 7. Sessions

**Lock the clock before measuring anything published:**

```bash
sudo ./gpu_clocks.sh lock 2700
```

This card idles near 400 MHz and drifts about 8% across a day. Margins under
~2% are not resolvable while it floats, however many rounds are taken. A
transformer run on unlocked clocks read 986/s and 482/s for identical work
fifteen minutes apart.

**The machine must be quiet, and the runner checks.** A competing process does
not slow both engines equally, so rotation does not cancel it. One browser tab
at a single core cost 35% of achievable memory bandwidth here: bandwidth-bound
steps moved by that much while cache-resident ones did not move at all. That is
worse than a uniform slowdown, because the result it produces has the shape of
a real finding. In the session that found it, it cost two measurement windows
and presented as a 24% regression in a binary that had not changed.

The runner samples the non-idle fraction across all hardware threads before the
first launch, records it as `machine_quiet`, and files the artifact in
`results/scratch/` above 3%. Deliberately not the load average, which decays
over minutes and reads 21 on an idle machine that was busy a moment ago. Quiet
measures under 1% here and one saturated core is about 3.6%, so the threshold
trips on roughly a single competing process. `OPENNN_BENCH_BUSY_THRESHOLD`
moves it.

The check is necessary, not sufficient: it sees CPU time, not memory bandwidth.
When a number moves and nothing in the tree explains it, re-run a fixed
reference kernel whose figures are known before suspecting the code. A GEMM at
a known shape told machine from binary in minutes, after an hour of reading
diffs that were not the cause.

**One session id per sitting**, exported so every launch under it agrees:

```bash
export OPENNN_BENCH_SESSION=2026-08-25-baseline
```

Numbers compare only within a session. An artifact whose session id nothing
else shares is an un-anchored run, and the `adhoc-<pid>` default is deliberately
awkward for that reason.

**A dirty tree writes to `results/scratch/`.** Enforced in code, because the
previous suite stated this rule in prose and checked it nowhere — 39 of its 107
artifacts were dirty-tree results filed as reproducible ones.
