# Source lines of code: OpenNN vs PyTorch vs TensorFlow

This note compares the amount of first-party native source code behind OpenNN, PyTorch, and
TensorFlow. It is not a speed benchmark, and it is not a complete capability comparison. It is a
scope measurement: how much C/C++ implementation source is in the native library layer that an
application depends on.

## Contents

- [Results](#results)
- [What is counted](#what-is-counted)
- [What is not counted](#what-is-not-counted)
- [PyTorch breakdown](#pytorch-breakdown)
- [Conclusion](#conclusion)
- [Why fewer lines can matter](#why-fewer-lines-can-matter)
- [References](#references)

## Results

| Scope | OpenNN | PyTorch | TensorFlow |
| --- | --- | --- | --- |
| Native C/C++ source LOC | 34,926 | 834,319 | 1,792,182 |
| Native C/C++ plus CUDA/HIP LOC | 37,069 | 920,576 | 1,792,182 |
| vs OpenNN (C/C++) | 1× | 24× | 51× |

The counts use `cloc` and count code lines only (blank and comment lines excluded). Two methodology
notes for the TensorFlow figure, in the interest of accuracy:

## What is counted

For OpenNN, the counted source is:

```
opennn/
```

For PyTorch, the counted source is the first-party native runtime subset from PyTorch `v2.12.0`:

```
aten/
c10/
torch/csrc/
caffe2/
```

This scope is chosen to compare native library implementation against native library implementation.
It includes PyTorch's tensor/runtime core (`aten`, `c10`), C++ bindings and autograd/runtime glue
(`torch/csrc`), and the remaining native runtime surface in `caffe2`.

For TensorFlow, the counted source is the first-party native runtime subset from TensorFlow
`v2.21.0` (revision `a481b10260dfdf833a1b16007eead49c1d7febf3`), excluding `third_party/`:

```
tensorflow/core/
tensorflow/c/
tensorflow/cc/
tensorflow/compiler/
tensorflow/lite/
```

This mirrors the PyTorch scope: TF's runtime core (`core`), its C and C++ APIs (`c`, `cc`), the
compiler/XLA-facing native code (`compiler`), and the TFLite runtime (`lite`).

## What is not counted

The main comparison excludes:

```
tests
examples
documentation
build outputs
generated build directories
third-party dependencies
PyTorch's Python frontend
```

The Python frontend is important to PyTorch, but OpenNN does not have an equivalent Python layer, so
including it in the headline number would mix native runtime scope with frontend ecosystem scope. If
a separate "full PyTorch repository" comparison is desired, it should be reported as a different
metric.

## PyTorch breakdown

For the main C/C++ count:

| PyTorch directory | Code LOC |
| --- | --- |
| `aten/` | 418,975 |
| `c10/` | 50,487 |
| `torch/csrc/` | 351,557 |
| `caffe2/` | 13,303 |
| **Total** | **834,319** |

## Conclusion

OpenNN's native library source is much smaller because it is a focused C++ neural-network library.
PyTorch's native runtime is much broader: it includes a general-purpose tensor engine, dispatch
system, autograd infrastructure, many operators, and runtime integration layers.

That broader scope is valuable for PyTorch's use cases. The LOC comparison should therefore not be
read as "smaller is always better." It should be read as a concrete source-size context for
deployment-size and embedded/edge discussions: OpenNN has a smaller native implementation surface,
while PyTorch carries a much larger general-purpose runtime.

## Why fewer lines can matter

Fewer lines of production code can have practical benefits, especially for native deployment:

- **Smaller audit surface.** There is less implementation to inspect when reviewing numerical
  behavior, memory ownership, threading, error handling, or security-sensitive paths.
- **Lower maintenance burden.** A smaller codebase is usually easier to refactor, port, document,
  and keep consistent across compilers and platforms.
- **Faster onboarding.** Engineers can understand a focused library more quickly than a broad
  runtime with many subsystems and historical layers.
- **Less build complexity.** Fewer components and dependencies generally mean simpler builds,
  fewer generated artifacts, and fewer platform-specific integration problems.
- **Smaller bug surface.** LOC does not predict bugs by itself, but every subsystem adds possible
  interactions. A focused implementation has fewer interactions to reason about.
- **Better fit for embedded and edge use.** When the target is a small native application, a
  compact source and runtime footprint can make the difference between a deployable solution and
  one that needs a separate trimming/export pipeline.

The tradeoff is scope. PyTorch's larger native runtime buys a very broad operator set, dynamic
dispatch, extensive integration points, and a large ecosystem. OpenNN's smaller codebase is an
advantage when the task fits its focused C++ library model.

## References

- [OpenNN source repository](https://github.com/Artelnics/OpenNN).
- [PyTorch source repository](https://github.com/pytorch/pytorch).
- [TensorFlow source repository](https://github.com/tensorflow/tensorflow).
- [cloc source line counter](https://github.com/AlDanial/cloc).
