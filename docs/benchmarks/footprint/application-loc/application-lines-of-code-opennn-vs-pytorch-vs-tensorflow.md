# Application lines of code: OpenNN vs PyTorch vs TensorFlow

This note compares the amount of user application code needed to implement the same small Iris
classification workflow in OpenNN, PyTorch, and TensorFlow/Keras. It is a developer-experience
metric, not a runtime-size or speed benchmark.

The workload follows the OpenNN API comparison article: load the Iris data, build a 4 -> 16 -> 3
classification network, train it, print a confusion matrix, run one deployment prediction, and save
the trained model.

## Contents

- [Results](#results)
- [What is counted](#what-is-counted)
- [How to reproduce](#how-to-reproduce)
- [Interpretation](#interpretation)
- [Caveats](#caveats)
- [References](#references)

## Results

| Framework | Language/API path | Logical source lines of code |
| --- | --- | ---: |
| OpenNN | C++ | 14 |
| PyTorch | Python | 43 |
| TensorFlow | Python / Keras | 23 |

OpenNN is the most concise of the three on this workflow: about **67% fewer logical instructions
than PyTorch** and **39% fewer than TensorFlow/Keras**.

## What is counted

The metric is **logical source lines of code (LSLOC)**:

- Blank lines and comments are ignored.
- C++ brace-only lines are ignored.
- Multi-line statements count as one logical instruction.
- Imports/includes are counted because the user must write them.

The counted snippets are stored in:

```text
docs/benchmarks/footprint/application-loc/opennn_iris.cpp
docs/benchmarks/footprint/application-loc/pytorch_iris.py
docs/benchmarks/footprint/application-loc/tensorflow_iris.py
```

The snippets implement the same application workflow with each framework's normal high-level API.

## How to reproduce

Run:

```bash
cd docs/benchmarks/footprint/application-loc
python count_lsloc.py
```

Current output:

```json
{
  "opennn_cpp": 14,
  "pytorch_python": 43,
  "tensorflow_python": 23
}
```

## Interpretation

This benchmark measures how much application glue code a developer writes for a small supervised
learning workflow.

- OpenNN is concise in native C++ because dataset handling, model construction, training strategy,
  testing analysis, deployment inference, and model serialization are represented by high-level
  library classes without requiring a Python runtime.
- PyTorch is more explicit: the user writes tensor conversion, the L-BFGS closure, the training
  loop, and the confusion-matrix accumulation.
- TensorFlow/Keras is also concise for this standard workflow because Keras provides a high-level
  `compile`/`fit` API. That concision does not imply a smaller runtime footprint; the deployment
  still depends on the TensorFlow/Keras Python runtime stack.

## Caveats

- This is not a capability comparison. LOC depends on the exact style of code chosen.
- The TensorFlow/Keras snippet uses the standard Keras `Adam` training path. Core Keras does not
  provide an L-BFGS optimizer equivalent to the PyTorch snippet without adding TensorFlow
  Probability or custom optimizer code.
- Fewer lines do not automatically mean better software. The metric is useful only as context for
  API ergonomics and deployment-code discussions.

## References

- [OpenNN API comparison](https://www.opennn.net/tutorials/api-comparison/).
- [OpenNN](https://github.com/Artelnics/OpenNN).
- [PyTorch](https://github.com/pytorch/pytorch).
- [TensorFlow](https://github.com/tensorflow/tensorflow).
