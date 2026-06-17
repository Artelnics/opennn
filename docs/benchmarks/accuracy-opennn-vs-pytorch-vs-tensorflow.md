# Numerical accuracy: OpenNN vs PyTorch vs TensorFlow

OpenNN matches the predictive accuracy of PyTorch and TensorFlow on a controlled nonlinear
regression benchmark, while keeping the smaller native footprint described in the rest of this
benchmark series.

> The other notes in this series compare deployment size, startup latency, dependencies, memory use,
> and export friction. This one asks the natural follow-up question: does OpenNN’s lighter native
> design cost anything in numerical accuracy?

## Contents

- [The result](#result)
- [Benchmark setup](#setup)
- [Why this matters](#why-it-matters)
- [Caveats](#caveats)
- [References](#references)

## The result

Final test-set accuracy on the Rosenbrock approximation benchmark, averaged over 5 random seeds:

| Framework | Mean MSE | Mean R^2 |
| --- | --- | --- |
| **OpenNN** | 0.0116 | **0.9879** |
| **PyTorch** | 0.0114 | **0.9882** |
| **TensorFlow** | 0.0129 | **0.9867** |

The three results are statistically indistinguishable: all reach R^2 around 0.987-0.988. OpenNN’s
training is numerically on par with the major frameworks, at a fraction of their footprint.

## Benchmark setup

The task is the Rosenbrock approximation benchmark: inputs `x_i ~ U(-1, 1)`, with target `y = Σ_i
[(1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2]`. We generate 10,000 samples with 10 input variables,
z-normalize the dataset once, and split it 80/20 into train and test sets.

The exact same normalized files feed OpenNN, PyTorch, and TensorFlow. This removes per-framework
preprocessing differences from the comparison.

| Item | Value |
| --- | --- |
| Architecture | 10 -> 50 tanh -> 50 tanh -> 1 linear |
| Initialization | Glorot/Xavier uniform |
| Loss | Mean squared error |
| Optimizer | Adam, learning rate 0.001, beta1 0.9, beta2 0.999 |
| Epochs / batch size | 200 / 64 |
| Data | One shared normalized train/test split |
| Runs | 5 random seeds per framework |

Accuracy is computed by a single neutral scorer that reads each framework’s test-set predictions and
computes MSE and R^2 in the same way. That prevents differences in internal loss definitions or
reduction conventions from affecting the comparison.

## Why this matters

A small native library is only useful if it is also correct. This benchmark shows that OpenNN’s
forward pass, backpropagation, and Adam optimizer produce the same quality of fit as two
independently developed, industry-standard frameworks.

That matters for deployment: the footprint and startup advantages in the other benchmark notes come
with no accuracy penalty on this controlled regression task.

## Caveats

- This is an accuracy-parity check on a controlled regression task, not a claim that one framework
  is universally more accurate than another.
- The benchmark was measured on Linux x86_64. OpenNN was built with g++ 13.3 on CPU. PyTorch was
  2.12.0+cpu and TensorFlow was 2.21.0 on CPython 3.12.
- The reported values are averaged over five seeds. The spread is small; OpenNN’s R^2 range was
  approximately 0.985-0.991.
- Different Adam epsilon defaults and tiny initialization differences can create per-seed
  variation, but they do not change the conclusion.

## References

- [The Rosenbrock benchmark for machine
  learning](https://www.neuraldesigner.com/blog/the-rosenbrock-benchmark-for-machine-learning/),
  Neural Designer.
- [OpenNN](https://www.opennn.net/).
- [PyTorch](https://pytorch.org/).
- [TensorFlow](https://www.tensorflow.org/).
