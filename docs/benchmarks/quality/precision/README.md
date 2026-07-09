# Precision Benchmark

Compares the lowest error floor each framework reaches with the optimizers it
ships, on the Rosenbrock approximation task: OpenNN's second-order
Quasi-Newton and Levenberg-Marquardt versus first-order Adam (all three
engines) and PyTorch's LBFGS.

## Documented HIGGS exception

Every other dense benchmark uses **HIGGS** (BCE classification) — see
[`../DATA_POLICY.md`](../../DATA_POLICY.md). **This is the one intentional
exception.** It stays on the **Rosenbrock regression** task on purpose, because
it exists to document OpenNN's **second-order optimizers** —
Levenberg-Marquardt and Quasi-Newton — which are **least-squares methods that do
not apply to HIGGS/BCE classification**. Regression is the only task where these
optimizers are defined, so this benchmark keeps it.

## Method

All engines train the **identical** model on the **identical** data:

- **Task:** Rosenbrock 10-input regression (`generate_rosenbrock.py`, a
  deterministic, pre-normalized 10,000-sample set; no train/test split — the
  blog trains and evaluates on the full dataset).
- **Model:** MLP 10 → 10 (tanh) → 1 (linear), all parameters `U(-1, 1)`.
- **Loss:** MSE, no regularization.
- **Optimizers (each engine, its own):**
  - OpenNN — Levenberg-Marquardt, Quasi-Newton (1000 epochs), Adam (10,000).
  - PyTorch — LBFGS (its only built-in second-order method, 1000 epochs),
    Adam (10,000).
  - TensorFlow — Adam only (`keras.optimizers` has no second-order option).
- **Scoring:** the neutral scorer `score.py` computes the full-dataset MSE the
  same way for every engine, from the shared targets and each engine's
  predictions.
- **Reported:** per engine/optimizer best MSE, mean MSE, and mean training time
  over N seeds.

## Build

The OpenNN driver is the CMake target `opennn_precision` (source
`opennn_precision.cpp`), built with the rest of the benchmark drivers; the
binary lands in `build/bin/`.

```bash
cmake --build build --target opennn_precision
```

## Usage

```bash
python run_precision.py                       # 10 seeds, all engines
python run_precision.py --runs 3 --engines opennn,pytorch
python run_precision.py --seeds 0,1,2
```

Binary discovery uses no hardcoded machine paths:

- OpenNN — `OPENNN_PRECISION_BIN`, else `opennn_precision` found in the
  benchmark build dir (`build/bin`, …).
- Python — `BENCH_PYTHON` (default `python3`).

Writes `../../results/precision-rosenbrock-<run_id>.json`
(`benchmark_id: "precision-rosenbrock"`) with per-engine/optimizer best/mean MSE
and mean time, framework versions, and the git commit.
