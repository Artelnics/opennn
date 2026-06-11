# Training precision: OpenNN vs PyTorch vs TensorFlow (Rosenbrock, Neural Designer protocol)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-11. Linux x86_64 (WSL2).*

This note reproduces the Neural Designer blog benchmark
[“Precision comparison: TensorFlow, PyTorch and Neural Designer”](https://www.neuraldesigner.com/blog/precision-comparison-tensorflow-pytorch-neural-designer/)
with OpenNN in the Neural Designer role. The blog's thesis is that a
second-order optimizer (Neural Designer's Levenberg-Marquardt) reaches a lower
mean squared error than TensorFlow and PyTorch's Adam, in far less time.
OpenNN provides the same second-order optimizers natively, so the claim can be
tested like-for-like.

## The result

Final full-dataset MSE on the Rosenbrock approximation benchmark (10 inputs,
10,000 samples, z-normalized), 10 random seeds per row, computed by a single
neutral scorer:

| | Optimizer | Best MSE | Mean MSE | Mean time |
|---|---|---:|---:|---:|
| **OpenNN** | Quasi-Newton (1,000 epochs) | 0.1082 | **0.1091** | **2.0 s** |
| **OpenNN** | Levenberg-Marquardt (1,000 epochs) | **0.1073** | 0.1183 | 10.0 s |
| **OpenNN** | Adam (10,000 epochs, batch 1,000) | 0.1350 | 0.1725 | 4.3 s |
| **PyTorch** | Adam (10,000 epochs, batch 1,000) | 0.1245 | 0.1622 | 42.3 s |
| **TensorFlow** | Adam (10,000 epochs, batch 1,000) | 0.1349 | 0.1562 | 310.0 s |

The blog's qualitative claim **reproduces**:

* **Second-order beats Adam on precision.** OpenNN's quasi-Newton and
  Levenberg-Marquardt runs reach a mean MSE ~1.3–1.5× lower than any Adam
  configuration — TensorFlow's (0.156), PyTorch's (0.162), or OpenNN's own
  (0.173). The blog reported ×1.91 vs TensorFlow and ×1.27 vs PyTorch for
  Neural Designer's LM; we measure ×1.43 and ×1.49 for quasi-Newton.
* **And it is faster, not slower.** Quasi-Newton converges in 2 seconds —
  21× faster than PyTorch and 155× faster than TensorFlow take to run their
  10,000 Adam epochs (the blog measured 5.7× and 8.2× for Neural Designer).
* All three frameworks' **Adam runs are statistically equivalent**
  (mean 0.156–0.173 with overlapping seed spreads), as in the
  [accuracy-parity note](accuracy-opennn-vs-pytorch-vs-tensorflow.md) —
  the precision gap comes from the optimizer, not the library. OpenNN's
  slightly higher 10-seed mean is one unlucky initialization (seed 1,
  MSE 0.283); extending to 30 seeds gives mean 0.170 ± 0.038 (median 0.152),
  indistinguishable from PyTorch's 0.162 ± 0.031. Seeds select different
  U(−1, 1) initializations in each framework's RNG, so per-seed values are
  not comparable across rows.

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
./run_precision.sh 10                    # all engines × 10 seeds + summary
# or individually:
./opennn_precision <seed> <optimizer> <epochs>   # QuasiNewtonMethod | LevenbergMarquardt | AdaptiveMomentEstimation
python pytorch_precision.py <seed>
python tensorflow_precision.py <seed>
python score.py <label> <predictions_file>
```
