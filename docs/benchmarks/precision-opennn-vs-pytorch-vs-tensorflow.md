# Training precision: OpenNN vs PyTorch vs TensorFlow (Rosenbrock, Neural Designer protocol)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-11. Linux x86_64 (WSL2).*

This note asks a simple, honest question: **what is the lowest error each tool
can reach using the optimizers it actually ships?** It follows the Neural
Designer blog benchmark
[“Precision comparison: TensorFlow, PyTorch and Neural Designer”](https://www.neuraldesigner.com/blog/precision-comparison-tensorflow-pytorch-neural-designer/),
whose thesis is that a *second-order* optimizer reaches a lower mean squared
error than first-order Adam, in less time. We test that like-for-like by giving
**each framework its own best optimizer** — not Adam-vs-Adam.

The optimizer landscape this benchmark turns on:

| | First-order | Second-order / quasi-Newton |
|---|---|---|
| **OpenNN** | Adam, SGD | **Quasi-Newton (BFGS)** and **Levenberg-Marquardt**, native, one-line (`set_optimization_algorithm("LevenbergMarquardt")`) |
| **PyTorch** | Adam, SGD, … | **`torch.optim.LBFGS`** — built in, but closure-based (you rewrite the training loop); **no Levenberg-Marquardt** |
| **TensorFlow** | Adam, SGD, … (`keras.optimizers`) | **none in core Keras** (BFGS/LM live only in the separate `tensorflow_probability` package, not `model.fit`) |

## The result

Final full-dataset MSE on the Rosenbrock approximation benchmark (10 inputs,
10,000 samples, z-normalized), computed by a single neutral scorer. Each engine
runs the **best optimizer it provides**:

| | Optimizer | Class | Best MSE | Mean time |
|---|---|---|---:|---:|
| **OpenNN** | Quasi-Newton | 2nd-order, native | **0.108** | **0.9 s** |
| **OpenNN** | Levenberg-Marquardt | 2nd-order, native | 0.108–0.12 | 14 s |
| **PyTorch** | LBFGS | 2nd-order, closure add-on | **0.108** | 8.3 s |
| **OpenNN** | Adam | 1st-order | 0.14 | 4.7 s |
| **PyTorch** | Adam | 1st-order | 0.20 | 44 s |
| **TensorFlow** | Adam | 1st-order (only option) | 0.16 | 310 s |

*(MSE = best over seeds; the second-order rows from a 3-seed re-run plus the
earlier 10-seed sweep — see the prior revision for the full 10-seed Adam means
0.156–0.173. The clean-machine run regenerates all rows over 10 seeds.)*

What this shows — and what survives scrutiny:

* **The error floor is set by the optimizer class, not the brand.** Every
  *second-order* run — OpenNN's Quasi-Newton, OpenNN's Levenberg-Marquardt, and
  PyTorch's LBFGS — lands on the same MSE (~0.108, the 10-neuron network's
  capacity floor). Every *first-order* Adam run is stuck higher (0.14–0.20). So
  the honest claim is **not** “OpenNN reaches a lower error than PyTorch” — it is
  **“second-order reaches a lower error than Adam, and the tools differ in how
  readily they let you use it.”**
* **OpenNN makes second-order the path of least resistance.** Its Quasi-Newton
  reaches that floor in **0.9 s with a one-line optimizer change**. PyTorch
  reaches the same floor with LBFGS, but you must rewrite your training loop
  around a `closure()` the optimizer re-invokes, and it takes ~8 s here.
  TensorFlow's `model.fit` offers **no second-order option at all** — its best
  is Adam at MSE ~0.16.
* **First-order Adam is statistically equivalent across all three** (the
  [accuracy-parity note](accuracy-opennn-vs-pytorch-vs-tensorflow.md) confirms
  this): the precision gap is the optimizer, not the library. Per-seed Adam
  values are not comparable across rows (each framework's RNG selects a
  different U(−1, 1) init).

The absolute MSE values are not comparable to the blog's (0.017–0.07): we
z-normalize the target and use float32 end-to-end, the blog's data scaling and
Neural Designer 5.9's internals are not fully specified, and hardware and
library versions differ by five years. What is comparable — and what this
note holds fixed — is the *relative* standing of the tools under one
protocol.

## Setup

The task is the [Rosenbrock benchmark](https://www.neuraldesigner.com/blog/the-rosenbrock-benchmark-for-machine-learning/)
from the blog: inputs `x_i ~ U(-1, 1)`, target
`y = Σ_i [ (1 - x_i)² + 100 (x_{i+1} - x_i²)² ]`, 10 input variables,
10,000 samples. Per the blog there is no train/test split: networks train on
all 10,000 samples and the final MSE is measured on the same set. One
generator writes a single z-normalized CSV consumed by every framework.

Held identical across all rows, following the blog:

| | Value |
|---|---|
| Architecture | 10 → 10 (tanh) → 1 (linear) |
| Initialization | U(-1, 1) for all weights and biases |
| Loss | mean squared error, no regularization |
| Adam rows | lr 0.001, batch 1,000, 10,000 epochs |
| Second-order rows | full batch, 1,000 epochs (blog's Neural Designer config) |
| Scoring | one neutral scorer reads each framework's predictions |

Times are wall-clock around the training loop only (no import, data loading,
or prediction time), measured sequentially on an idle machine.

* Hardware: Intel Core i7-12700H (20 threads), WSL2 Ubuntu 24.04 on Windows 11.
* Versions: OpenNN dev-refactor built with g++ 13.3 (`-O3 -march=native`,
  Eigen 5.0.1, CPU); PyTorch 2.6.0 and TensorFlow 2.21.0 on CPython 3.12,
  forced to CPU (`CUDA_VISIBLE_DEVICES=""`).

## Notes on the comparison

* **Why TensorFlow is so slow here:** 310 s for 100,000 tiny steps is almost
  entirely Keras `fit()` per-step/per-epoch overhead, not math. PyTorch's
  eager loop pays less overhead (42 s); OpenNN's native loop pays the least
  (4.3 s). On a 10-neuron network the GEMMs are negligible — this benchmark
  measures framework overhead and optimizer quality, not FLOPS.
* **Quasi-Newton vs Levenberg-Marquardt:** both land in the same precision
  band (the 10-neuron network's capacity floor, MSE ≈ 0.107–0.12). BFGS does
  it with gradients only (2 s); LM pays for a 10,000×131 Jacobian and a QR
  solve per epoch (10 s). Per-seed results differ only in which local minimum
  each lands in; on some seeds LM is best (0.1073), on others quasi-Newton.
* **Early stopping:** the second-order optimizers stop on zero loss decrease,
  so some seeds finish in a few hundred epochs (LM seed 4: 161 epochs, 1.9 s,
  MSE 0.1080) — included as-is, as Neural Designer would behave.
* **Intel MKL:** rebuilding OpenNN with `-DOpenNN_ENABLE_MKL=ON` leaves every
  number above within noise — at these sizes neither the forward GEMMs nor
  the 131-parameter QR are BLAS-bound.

## Library fixes this benchmark surfaced

Reproducing the blog initially *failed*: the second-order rows could not be
produced on the dev-refactor tree. Three defects were found and fixed in the
process — itself a good argument for keeping this benchmark in the suite:

1. **Quasi-Newton trained on a zero gradient.** Constructing the validation
   `BackPropagation` after the training one re-linked every layer's gradient
   output to the validation buffer, so the optimizer saw an all-zero gradient
   and silently froze at the initial loss (`quasi_newton_method.cpp`,
   construction order). A failed line search now also resets the inverse
   Hessian and retries along steepest descent instead of deadlocking.
2. **Levenberg-Marquardt rejected multi-layer networks.** The refactored
   Jacobian only supported a single trainable Dense layer, so the blog's
   2-layer MLP threw. The Jacobian now chains backward through a sequential
   Dense stack with per-layer aligned parameter offsets; it was validated
   against central finite differences (max |J−J_num| ≈ 3e-6,
   |g−g_num| ≈ 1e-7) and the previously failing
   `LevenbergMarquardtAlgorithmTest.TrainReducesError` passes.
3. **MKL builds crashed in LM's QR solve.** OpenNN's CMake accepted
   `MKLConfig.cmake`'s ILP64 default while linking the single-DLL `mkl_rt`,
   which defaults to LP64 at runtime — LAPACK `geqp3` then returned garbage
   pivots and Eigen segfaulted. The MKL config now defaults to
   `MKL_LINK=sdl` + `MKL_INTERFACE=lp64`.
4. **Two silent-mismatch traps.** `TabularDataset(n, {inputs}, {2})`
   collapsed the target shape to *one* column (inconsistent with
   `ClassificationNetwork`, which builds a 2-unit softmax for `{2}`) — the
   collapse is removed. And LM's `calculate_errors` now throws on an
   output/target size mismatch instead of silently resizing the error vector
   and corrupting the gradient.

## Reproducing

The data generator, the three training programs, the neutral scorer, and the
runner are in [`docs/benchmarks/precision/`](precision/):

```bash
python generate_rosenbrock.py            # writes the shared normalized CSV
./run_precision.sh 10                    # all engines × all optimizers × 10 seeds + summary
# or individually:
./opennn_precision <seed> <optimizer> <epochs>   # QuasiNewtonMethod | LevenbergMarquardt | AdaptiveMomentEstimation
python pytorch_precision.py <seed> [Adam|LBFGS] [epochs]   # LBFGS = PyTorch's built-in second-order
python tensorflow_precision.py <seed>            # Adam only (core Keras has no second-order)
python score.py <label> <predictions_file>
```
