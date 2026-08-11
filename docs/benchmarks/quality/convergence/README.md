# Convergence gate: wall-clock time to a fixed held-out quality (MLPerf-style)

The other speed benchmarks time a **fixed number of epochs** and report
throughput. That is gameable: an engine can look "fast" by doing less useful
work per step. This benchmark instead measures the metric that actually matters
— **how long until the model reaches a fixed quality target** — and gates on a
**held-out** metric so a model that overfits the training split cannot pass.

This directly answers the technical review question *"are you fast because you
don't actually learn?"*

## Task

HIGGS binary classification (UCI HIGGS, 28 numeric features, signal-vs-background
target). All three engines train the **canonical HIGGS dense classifier** and
stop at the **identical** held-out target:

- **Model:** MLP `28 -> 1024 -> 1024 -> 1`, ReLU hidden, sigmoid output.
- **Optimizer / loss:** Adam (no weight decay, no gradient clipping), binary
  cross entropy, batch 1024.
- **Precision / device:** fp32, CPU.
- **Gate:** train until the **held-out test log-loss** ≤ `--target`, evaluated
  after each epoch (OpenNN: after each training chunk). The clock counts
  **training time only** — per-epoch / per-chunk held-out evaluation is
  excluded.
- **Reported:** median ± stdev wall-clock time-to-target over `--runs`, the
  held-out test log-loss at the stopping point, and epochs taken. A run that
  fails to reach the target within `--max-epochs` is recorded `reached_goal=0`
  and excluded from the timing median.

Gating on the **held-out** metric (not the training loss) is the crucial design
choice: a training-loss gate lets an engine "pass" before it generalizes. Gating
on the test log-loss removes that artifact and makes the time comparison fair.

The model, split, and file layout follow the shared HIGGS dense contract in
[`../../throughput/higgs/README.md`](../../throughput/higgs/README.md).

## Data

HIGGS files live **outside** the repository. Set `OPENNN_BENCH_DATA` and prepare
the split as documented in [`../DATA_POLICY.md`](../../DATA_POLICY.md) and
[`../../throughput/higgs/README.md`](../../throughput/higgs/README.md). The
runner reads:

```text
$OPENNN_BENCH_DATA/higgs/higgs_train.csv
$OPENNN_BENCH_DATA/higgs/higgs_test.csv
```

Each row is `feature_0,...,feature_27,label` with no header.

## Build

The OpenNN engine (`opennn_convergence`) is a CMake benchmark target — it is
built by the benchmark build, not by a hand-written link line:

```bash
cmake -S . -B build-benchmarks \
  -DOpenNN_BUILD_EXAMPLES=OFF \
  -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build build-benchmarks --config Release --target opennn_convergence
```

The runner finds the binary automatically in `build-benchmarks/bin` or
`build/bin`; override with `OPENNN_CONVERGENCE_BIN=/path/to/opennn_convergence`.

## Usage

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"

python run_convergence.py                          # target 0.47, 5 runs, 8 threads
python run_convergence.py --engines opennn,pytorch
```

Flags: `--engines`, `--target` (held-out test log-loss, default 0.47),
`--max-epochs`, `--runs`, plus `--batch`, `--hidden`, `--hidden-layers`,
`--threads` (default 8 — the suite's fair CPU protocol, applied to all three
engines; `0` = engine defaults), and `--train` / `--test` overrides.

Writes `../../results/convergence-higgs-<run_id>.json`
(`benchmark_id: "convergence-higgs"`) with per-engine median ± stdev
time-to-target, held-out test log-loss, epochs, convergence count, per-run
detail, framework versions, git commit, and dataset file info.

## Notes

- The gate is on **held-out** log-loss, so pick `--target` from a real reference
  run (a random classifier sits near `0.693`; a trained HIGGS MLP reaches
  well below that). Use accuracy or AUC instead by swapping the held-out metric
  in each engine if a different quality contract is required.
- Each engine seeds deterministically (seed 42) so the number of epochs to the
  target is comparable; only the wall-clock time differs across engines.

## Latest result (2026-08-11, i9-12900K CPU, commit 52e21e15d)

Target hardened from the historical 0.60 (which every engine reached in one
epoch, degenerating into "time per epoch") to **held-out log-loss ≤ 0.47**;
per-run process timeout 1800 s. 5 runs per engine, all converged
(artifact `results/convergence-higgs-20260811T064906Z.json`):

| Engine | Median time to target | Epochs to target | Final test log-loss |
|---|---:|---:|---:|
| **OpenNN** | **200.0 s** | 2 | 0.4680 |
| TensorFlow | 225.3 s | 2 | 0.4674 |
| PyTorch | 323.8 s | 3 | 0.4660 |

OpenNN reaches the held-out target first — 1.13× before TensorFlow and 1.62×
before PyTorch (which needs an extra epoch). Per-epoch times match the HIGGS
CPU throughput benchmark for all three engines. The clock counts training time
only (held-out evaluation excluded on every engine).

### 2026-08-11 protocol fixes

The 2026-08-10 run (`convergence-higgs-20260810T174215Z.json`, TensorFlow
179.3 s < OpenNN 280.4 s < PyTorch 513.9 s) was measured under **engine-default
threading** — the only CPU benchmark in the suite not applying the fair
8-thread protocol. On this hybrid CPU the defaults punished OpenNN (24 threads
across HT + E-cores: 140 s/epoch vs 98 under the protocol) and PyTorch (171 vs
105) while favoring TensorFlow's own thread pools (90 vs 103), inverting the
ranking. Two fixes, superseding that artifact:

1. `run_convergence.py` now applies the same thread protocol as
   `../../throughput/higgs/run_higgs_cpu.py` to all three engines
   (default `--threads 8`; `0` restores engine defaults).
2. `opennn_convergence` now trains in a single `train()` call gated by the
   optimizer's post-epoch callback, so Adam moment state persists across
   epochs exactly as in the PyTorch/TF drivers (the previous chunked driver
   reset it each epoch — a quality handicap the other engines did not pay).
