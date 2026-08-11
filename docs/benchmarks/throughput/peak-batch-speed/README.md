# Peak-batch training throughput: each engine at its own best batch

The fixed-batch throughput benchmarks ([higgs-gpu](../higgs-gpu/README.md),
[resnet50](../resnet50/README.md), [attention-speed](../attention-speed/README.md))
compare the engines at one common batch size. The capacity benchmarks
([higgs-max-batch](../../capacity/higgs-max-batch/README.md),
[resnet50-max-batch](../../capacity/resnet50-max-batch/README.md),
[transformer-max-batch](../../capacity/transformer-max-batch/README.md)) showed
the engines' memory ceilings differ by large factors. This benchmark combines
the two: **how fast can each engine train when it is allowed to pick its own
batch size?**

For every (engine, precision) the runner sweeps the batch geometrically upward
from the standard size (doubling, ending exactly at the training-set size),
measures training samples/s at every point with the **same speed drivers the
fixed-batch benchmarks use**, and reports:

- the full throughput-vs-batch **curve**,
- the **peak samples/s** and the batch where it happens,
- the **frontier** — the batch where the engine's curve dies (OOM), which for
  the smaller-ceiling engines arrives before their curve has flattened.

Each point runs in a **fresh process** (the capacity-suite isolation pattern),
so an OOM never poisons the next point; the first failed point ends that
engine's ascent.

## What this does and does not claim

- It measures **hardware/runtime saturation** — throughput at each engine's own
  best operating point. It is the fair headline when the deployment question is
  "how fast can this GPU train with engine X, period".
- It does **not** gate on model quality: very large batches change convergence
  behaviour at a fixed learning rate. The quality-gated comparison is
  [quality/convergence](../../quality/convergence/README.md).
- Batches are full (drop-last), matching the rest of the suite.

## Families and defaults

| `--family` | Drivers reused from | Ladder | Timed epochs/point |
|---|---|---|---|
| `higgs` | [`../higgs-gpu/`](../higgs-gpu/) | 7,000 ×2 → 10,500,000 (full batch) | 3 |
| `resnet50` | [`../resnet50/`](../resnet50/) | 128 ×2 → 50,000 (full batch) | 3 |
| `transformer` | [`../attention-speed/`](../attention-speed/) | 32 ×2 → 4,096 (full batch) | 5 |

There is **no engine code in this folder** — the OpenNN binaries and the
PyTorch/TensorFlow scripts are the sibling speed drivers, unchanged, so any
driver improvement automatically flows into this benchmark.

## Running

```bash
# build the OpenNN drivers once
cmake --build build --target opennn_speed opennn_resnet50_speed opennn_transformer_train

export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
export BENCH_PYTHON="$HOME/.venvs/opennn-bench/bin/python"

cd docs/benchmarks/throughput/peak-batch-speed
python run_peak_batch_speed.py --family higgs
python run_peak_batch_speed.py --family resnet50
python run_peak_batch_speed.py --family transformer

# inspect the ladder and exact per-engine commands without running anything:
python run_peak_batch_speed.py --family higgs --dry-run
```

Writes `../../results/gpu-<family>-peak-batch-speed-<run_id>.json` with the per-engine
curves, peaks, frontiers, and `opennn_vs_*` ratios on the peaks.

Budget: the full three-family, two-precision matrix is several GPU-hours
(dominated by the HIGGS CSV parse per fresh process and TensorFlow's ResNet
trials). `--precisions bf16` halves it; `--engines`, `--max-batch`, and
`--timeout-s` bound it further.

## Status

Benchmark scaffolding added 2026-08-11; **not yet executed** — no result
artifact or note exists yet. First full run pending after the throughput
drivers' 5-run re-measurement pass.
