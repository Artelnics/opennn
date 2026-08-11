# Predictive accuracy parity: OpenNN vs PyTorch vs TensorFlow (HIGGS)

*Last updated 2026-08-10, commit 52e21e15d. Linux x86_64, Intel Core i9-12900K (CPU, fp32). Artifact: [`results/accuracy-higgs-20260810T170812Z.json`](../../results/).*

OpenNN matches — and in this run marginally leads — the predictive quality of
PyTorch and TensorFlow on the canonical HIGGS dense classifier, while keeping
the smaller native footprint described in the rest of this benchmark series.

> The other notes in this series compare speed, capacity, energy, deployment
> size, startup latency, memory use, and export friction. This one asks the
> natural follow-up question: does OpenNN's lighter native design cost anything
> in predictive quality? It does not.

## The result

Held-out test metrics after an identical fixed budget (5 epochs, batch 1024)
on the full HIGGS split (10.5M train / 500k test rows), same architecture
(28 → 1024 → 1024 → 1, ReLU, sigmoid, BCE, Adam) and the same prepared,
normalized CSVs for every engine:

| Framework | Accuracy | Log loss | ROC AUC |
| --- | ---: | ---: | ---: |
| **OpenNN** | **0.7782** | **0.4579** | **0.8632** |
| PyTorch | 0.7766 | 0.4599 | 0.8617 |
| TensorFlow | 0.7774 | 0.4584 | 0.8628 |

The spread across engines is 0.16 accuracy points, 0.002 log loss and 0.0015
AUC — the three land in the same quality band, with OpenNN at the top of it in
this run. OpenNN's training is numerically on par with the major frameworks at
a fraction of their footprint.

## Benchmark setup

| Item | Value |
| --- | --- |
| Task | HIGGS binary classification (UCI, 11M rows, 28 features) |
| Split | 10.5M train / 500k held-out test, prepared once, shared by all engines |
| Architecture | 28 → 1024 → 1024 → 1, ReLU hidden, sigmoid output |
| Loss / optimizer | Binary cross-entropy, Adam lr 0.001 |
| Budget | 5 epochs, batch 1024, CPU fp32 (`CUDA_VISIBLE_DEVICES=""`) |
| Runner | [`run_accuracy.py`](README.md) — one process per engine, metrics parsed from each engine's own evaluation |

## Why this matters

Speed and capacity claims are only meaningful if the trained model is as good.
This note pins that equivalence on the same contract (HIGGS, dense, Adam) the
throughput, capacity and energy benchmarks use, so the whole matrix shares one
quality baseline.

## Caveats

- Single run per engine at a fixed budget (the historical protocol); the
  band is narrow but no seed statistics are claimed. `--runs 3` extends it.
- Quality parity at 5 epochs does not certify convergence behavior — that is
  [the convergence benchmark](../convergence/README.md)'s job.
- An earlier version of this note reported the legacy Rosenbrock regression
  parity (R² ≈ 0.987-0.988 for all three engines); that experiment predates
  the HIGGS contract and its artifacts are no longer generated.

## References

- [UCI HIGGS dataset](https://archive.ics.uci.edu/dataset/280/higgs)
