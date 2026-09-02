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
| CUDA · cuDNN | 13.3.73 · 9.25.1 system, which OpenNN links |
| PyTorch | 2.13.0+cu130, arch list includes `sm_120`; bundles its own cuDNN 9.23.2 |
| MKL | 2026.0.1 for OpenNN; PyTorch's wheel bundles its own |
| Python | 3.12.3 |

**The two engines do not share a cuDNN.** OpenNN links the system
`libcudnn.so.9` at 9.25.1; PyTorch loads the 9.23.2 its wheel ships. Two minor
versions apart, and the CNN and transformer families are cuDNN-bound, so this
is an asymmetry in the same class as the OpenMP-runtime one that inverted the
CPU LSTM cell twice before item 6 pinned it down. It is recorded rather than
corrected: neither engine can be moved to the other's cuDNN without building it
from source, and the effect here is unmeasured. Do not attribute a CNN or
transformer margin under a few percent to the framework without checking it.

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
transformer 120,245,792 · LSTM 73,857.

The transformer figure read 74,878,496 here until 2026-09-01, which no run has
ever reported: every artifact in the store records 120,245,792, and the gate
passed all along because it compares the engines against *each other*, not
against this document. A stale number in the contract is worse than none --
anyone checking the gate by hand would have concluded it was broken.

**Quality.** Accuracies must agree within tolerance, per batch, across engines.
A speed win bought by computing something different is not a speed win.

## 5. Metrics

**Throughput.** Samples/s, warmup excluded, median of repeated rounds, every
round kept in the artifact so a drifting session is visible rather than
averaged into a claim. Ratios within one session are the durable output;
absolute samples/s is provenance.

**Peak memory.** On GPU, device-used memory (`nvmlDeviceGetMemoryInfo`, the
figure `nvidia-smi` prints) sampled at 20 ms, minus the idle reading taken
before the run. Whole-device on purpose: it counts the CUDA
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
the marks it prints. The power is the driver's own sampling of the board -- one
instantaneous reading every 20 ms, timestamped by the driver and drained from
its ring through NVML (`nvmlDeviceGetSamples`) once a second; the ring holds
about 2.4 s, so nothing is lost, and the drain rate does not touch the result
(20 ms, 1 s and no draining at all read the same throughput on a launch-bound
cell, interleaved). The figure is reported only when the window held at least
fifty samples, one second; otherwise the artifact says so. A short run has no
energy figure, and `0.0000 Wh` would be a claim rather than an absence.

Why not `nvidia-smi --query-gpu=power.draw`, which the suite used until
2026-09-02: on Ampere and later that field is the driver's *one-second moving
average*, so polling it at 20 ms cannot see inside a second. Every sub-second
cell taken with it -- the dense and LSTM CUDA cells, windows of 70 ms to 1.3 s
-- reported a mean of about 45 W, which is an idle card with a burst of GEMMs
averaged into it, not the 200 W and more the burst draws, and the old rule of
four samples let those through as figures. The other readings were measured
and rejected too: `power.draw.instant` refreshes every 500 ms, and
`nvmlDeviceGetTotalEnergyConsumption` is a synchronous firmware query that
stalls the accumulator it reads -- polled every 10 ms it under-counts a steady
233 W by 60%, every 100 ms by 10%. That counter is read exactly twice per run,
once at each end, and the whole-run difference is filed beside the window's
figure as `run_energy_joules`: an independent instrument that agrees with the
integral to 0.1% on a steady load, and a bound the window can never exceed.

The window is a per-cell setting, not a matter of taste: a cell whose timed
window would fall under two seconds runs more epochs or repeats until it clears
them. On the reference machine that is cuda-dense-train at 100 epochs,
cuda-dense-infer at 200 repeats, cuda-lstm-train at 20 epochs and
cuda-lstm-infer at 50 repeats; every other cell clears two seconds at its
default. The footprint family prints no marks: its process is the operation
being timed, its figure is the whole process, and a process that lives under a
second reports none.

**CPU energy is the RAPL package counter,** integrated over the same timed
window, and never the GPU's draw -- which during a CPU run is an idle card.

`package-0`, not `core`. The package domain covers the cores *and* the uncore:
the memory controller, the ring, the last-level cache. A bandwidth-bound run
pays for those as surely as it pays for the multipliers, so charging it only
for arithmetic would flatter exactly the workloads that move the most data.
Each artifact records `energy_domain` and `energy_metric`, so a CPU figure is
never silently read as the GPU's whole-board one -- they are different
quantities and must not share a column.

RAPL reports cumulative energy, not power, so the window is taken by
*differencing* the counter with interpolated endpoints, not by the trapezoid
rule the GPU path applies to power samples. Steps are accumulated modulo
`max_energy_range_uj`, because the register wraps -- 262 kJ on this part -- and
an unhandled wrap reads as one large negative step that would cancel most of a
long run's energy.

On a stock kernel `energy_uj` is `0400` root-only, hardened after CVE-2020-8694
(PLATYPUS). Where nobody has granted read access the artifact says
`energy_measurable: false` and gives the reason, rather than inventing a figure:

```bash
sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
```

That permission does not survive a reboot, so a CPU energy cell taken after one
will report the counter as unreadable until it is granted again. Machines
without the counter at all -- AMD without the module, most VMs -- report the
same absence.

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

**One OpenMP wait policy for both engines, and the artifact records it.** The
two engines do not share an OpenMP runtime: OpenNN links the system libgomp,
PyTorch's wheel carries its own. GCC 14's libgomp recognises a hybrid CPU and
stops spinning at barriers (`GOMP_SPINCOUNT` falls to 1; every fork/join
sleeps in the kernel), while PyTorch's older copy still spins 300,000 times.
oneDNN's LSTM opens about thirty regions per batch, so this alone moved the
identical primitive from 3.14 ms to 3.45 ms and turned a won cell into a lost
one. The runner sets `GOMP_SPINCOUNT=300000` — libgomp's own documented
default — for both engines; measured, it is a no-op for PyTorch (69.0k →
69.3k samples/s) and restores OpenNN (62.6k → 72.3k). The `pinning.omp_wait`
field says so in every CPU artifact. Intel's runtime (`libiomp5`) preloaded
under both engines gives the same ordering with a wider margin; it is not the
protocol default because neither engine ships it.

**Spinning has a cost of its own, and OpenNN pays it on one pool.** The
process holds two pools on the same sixteen CPUs: Eigen's, for tensor
expressions, and libgomp's, for `#pragma omp`, MKL's threaded calls and
oneDNN. A libgomp worker that has just finished a region spins for those
300,000 iterations — several milliseconds — on a logical CPU an Eigen thread
may need next. The first run under the wait policy above showed exactly that:
the dense forward GEMM ran on Eigen's pool, the one OpenMP region per batch
(MKL's `sgemv` for the 1024→1 layer) released fifteen spinners into it, 62% of
the process's samples were libgomp's `do_spin`, and cpu-dense-infer read 117k
samples/s against 222k with the spin disabled — 0.70× of PyTorch instead of
1.29×. OpenNN now runs its blocked GEMM on the libgomp pool by default
(`OPENNN_GEMM_MODE`, in `tensor_operations.cpp`), so the threads that spin are
the threads that work: 221k spinning, 217k not. The rule this leaves behind is
that no cell may keep steady work on Eigen's pool while libgomp spins; a
profile with `libgomp.so` above a few percent means it has crept back.

**Energy on CPU is RAPL `package-0`,** not the idle GPU's draw. See item 5 for
the domain, the wrap handling and the permission it needs.

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
first launch and after the last, and in between it watches every second of
every launch: the CPU time of everything that is not the launch -- not the
runner, not the launched process or its children, not kernel threads --
charged from `/proc/<pid>/stat` once a second and judged over the engine's
timed window. All three readings go into `machine_quiet` (`busy_before`,
`busy_after`, `busy_during_max` with the second it happened), and the artifact
is filed in `results/scratch/` when any of them is above 3%. Deliberately not
the load average, which decays over minutes and reads 21 on an idle machine
that was busy a moment ago. Quiet measures under 1% here and one saturated core
is about 3.6%, so the threshold trips on roughly a single competing process.
`OPENNN_BENCH_BUSY_THRESHOLD` moves it.

The per-second watch exists because the edge samples were not enough. On
2026-09-02 an editor drawing a conversation on the E-cores, while 2.4 s
dense-training windows ran on the P-cores, cost the launch-bound step 4-12%
on both engines -- and three foreign cores took PyTorch's from 10.2M to 4.7M
samples/s while OpenNN's lost 1% -- with both edge samples reading quiet. A
disturbance that starts after the first launch and ends before the last is
exactly the one a long cell is exposed to, and it does not slow both engines
equally.

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
