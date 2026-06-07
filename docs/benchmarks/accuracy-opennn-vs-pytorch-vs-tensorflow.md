# Numerical accuracy: OpenNN vs PyTorch vs TensorFlow (Linux)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-08. Linux x86_64.*

The other notes in this series show OpenNN is far smaller, faster to start, and lighter to
deploy than PyTorch and TensorFlow. The natural question is whether that lightness costs
anything in **accuracy**. This benchmark answers it directly: train the *same* network on the
*same* data with the *same* optimizer in all three, and compare how well each fits.

## The result

Final test-set accuracy on the Rosenbrock approximation benchmark (10 inputs), averaged over
5 random seeds:

| | Mean MSE | Mean R² |
|---|---:|---:|
| **OpenNN** | 0.0116 | **0.9879** |
| **PyTorch** | 0.0114 | **0.9882** |
| **TensorFlow** | 0.0129 | **0.9867** |

The three are **statistically indistinguishable** — all reach R² ≈ 0.987–0.988. OpenNN's
training is numerically on par with the major frameworks, at a fraction of their footprint.

## Setup

The task is the [Rosenbrock benchmark](https://www.neuraldesigner.com/blog/the-rosenbrock-benchmark-for-machine-learning/):
inputs `x_i ~ U(-1, 1)`, target `y = Σ_i [ (1 - x_i)² + 100 (x_{i+1} - x_i²)² ]`. We generate
10,000 samples with 10 input variables, z-normalize once, and split 80/20 into train/test.
The exact same normalized files feed all three frameworks.

Everything that affects the result is held identical:

| | Value |
|---|---|
| Architecture | 10 → 50 (tanh) → 50 (tanh) → 1 (linear) |
| Initialization | Glorot/Xavier uniform |
| Loss | mean squared error |
| Optimizer | Adam (lr 0.001, β₁ 0.9, β₂ 0.999) |
| Epochs / batch | 200 / 64 |
| Data | one shared normalized train/test split |

Accuracy is computed by a **single neutral scorer** that reads each framework's test-set
predictions and computes MSE and R² the same way — so no framework's internal loss
definition or reduction convention can bias the comparison.

## Methodology note: matching the objective

For a fair raw-accuracy comparison, all three must minimize the *same* objective. By default
OpenNN ships with **production-sensible training defaults that PyTorch and TensorFlow do not**:

* **L2 regularization** (weight 0.001) on by default — PyTorch/TF `Adam` use no weight decay.
* An automatic **60/20/20 train/validation/test split** with validation-based early stopping —
  the framework training loops we wrote use the whole training set.

Those defaults are good practice for real modeling (they curb overfitting and give an honest
validation signal out of the box), but they mean OpenNN would otherwise be solving a
*different, regularized* problem on *less* data than the bare PyTorch/TensorFlow scripts. For
this benchmark we therefore disable L2 regularization and put all samples in the training role,
so every framework minimizes the same pure-MSE objective on the same data. With that alignment,
the accuracies match.

> This is worth emphasizing: out of the box, OpenNN defaults to a *regularized, validated*
> training setup, whereas the minimal PyTorch/TensorFlow scripts do not. The parity above is
> measured after stripping OpenNN's extra safeguards to match the others — not by adding
> anything to OpenNN.

## Why this matters

A small, native library is only useful if it is also *correct*. This benchmark shows OpenNN's
forward pass, backpropagation, and Adam optimizer produce the same quality of fit as two
independently-developed, industry-standard frameworks — which agree with each other and with
OpenNN to within seed-to-seed noise. The footprint and startup advantages in the other notes
therefore come with no accuracy penalty.

## Caveats

* This is an **accuracy-parity** check on a controlled regression task, not a claim that any
  framework is more accurate — the point is that they are equivalent.
* Measured on Linux x86_64: OpenNN built with g++ 13.3 (CPU); PyTorch 2.12.0+cpu and
  TensorFlow 2.21.0 on CPython 3.12. Five seeds each; the spread is small (OpenNN R²
  0.985–0.991).
* Different default initializations and Adam epsilon (OpenNN uses `FLT_EPSILON`, the
  frameworks 1e-8) introduce tiny per-seed differences that average out; they do not change
  the conclusion.

## Reproducing

The data generator, the three training programs, and the neutral scorer are in
[`docs/benchmarks/accuracy/`](accuracy/):

```bash
python generate_rosenbrock.py                 # writes shared train/test CSVs
./opennn_accuracy <seed>                       # writes pred_opennn.txt
python pytorch_accuracy.py <seed>              # writes pred_pytorch.txt
python tensorflow_accuracy.py <seed>           # writes pred_tensorflow.txt
python score.py OpenNN pred_opennn.txt         # MSE + R2, computed identically
```
