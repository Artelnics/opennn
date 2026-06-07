# Source lines of code: OpenNN vs PyTorch

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-07.*

This note compares the amount of first-party native source code behind OpenNN, PyTorch, and
TensorFlow. It is not a speed benchmark, and it is not a complete capability comparison. It is
a scope measurement: how much C/C++ implementation source is in the native library layer that
an application depends on.

The short version:

| Scope | OpenNN | PyTorch | TensorFlow |
|---|---:|---:|---:|
| Native C/C++ source LOC | 34,926 | 834,319 | 1,792,182 |
| Native C/C++ plus CUDA/HIP LOC | 37,069 | 920,576 | 1,792,182 |
| vs OpenNN (C/C++) | 1× | 24× | 51× |

The counts use `cloc` and count code lines only (blank and comment lines excluded). Two
methodology notes for the TensorFlow figure, in the interest of accuracy:

* **Tests are co-located in TensorFlow's source.** Of TF's 1,792,182 native LOC, ~507,000
  are `*_test.*` files sitting alongside the implementation. Excluding them gives ~1,285,000
  non-test native LOC — still ~37× OpenNN. PyTorch's counted directories separate most tests,
  so its figure is closer to non-test already; the headline row counts each project's native
  directories as-is.
* **CUDA:** TF's CUDA/HIP total equals its C/C++ total here because the counted directories
  contain no `.cu` files — TF generates much of its GPU code through other mechanisms — so the
  `.cu/.cuh` extensions add nothing to the count for TF.

## What is counted

For OpenNN, the counted source is:

```text
opennn/
```

For PyTorch, the counted source is the first-party native runtime subset from PyTorch
`v2.12.0`:

```text
aten/
c10/
torch/csrc/
caffe2/
```

The PyTorch revision used was:

```text
0d62256a2b23365f8e1604297eb23a6545102aa8
```

This scope is chosen to compare native library implementation against native library
implementation. It includes PyTorch's tensor/runtime core (`aten`, `c10`), C++ bindings
and autograd/runtime glue (`torch/csrc`), and the remaining native runtime surface in
`caffe2`.

For TensorFlow, the counted source is the first-party native runtime subset from TensorFlow
`v2.21.0` (revision `a481b10260dfdf833a1b16007eead49c1d7febf3`), excluding `third_party/`:

```text
tensorflow/core/
tensorflow/c/
tensorflow/cc/
tensorflow/compiler/
tensorflow/lite/
```

This mirrors the PyTorch scope: TF's runtime core (`core`), its C and C++ APIs (`c`, `cc`),
the compiler/XLA-facing native code (`compiler`), and the TFLite runtime (`lite`).

## What is not counted

The main comparison excludes:

```text
tests
examples
documentation
build outputs
generated build directories
third-party dependencies
PyTorch's Python frontend
```

The Python frontend is important to PyTorch, but OpenNN does not have an equivalent Python
layer, so including it in the headline number would mix native runtime scope with frontend
ecosystem scope. If a separate "full PyTorch repository" comparison is desired, it should
be reported as a different metric.

## PyTorch breakdown

For the main C/C++ count:

| PyTorch directory | Code LOC |
|---|---:|
| `aten/` | 418,975 |
| `c10/` | 50,487 |
| `torch/csrc/` | 351,557 |
| `caffe2/` | 13,303 |
| **Total** | **834,319** |

## Method

Install `cloc`:

```powershell
winget install --id AlDanial.Cloc --exact --accept-package-agreements --accept-source-agreements --disable-interactivity
```

Fetch a sparse PyTorch checkout:

```bash
git clone --filter=blob:none --sparse --depth 1 --branch v2.12.0 https://github.com/pytorch/pytorch.git pytorch-v2.12.0
cd pytorch-v2.12.0
git sparse-checkout set aten c10 torch/csrc caffe2
git rev-parse HEAD
```

Count OpenNN native C/C++ LOC:

```bash
cloc opennn \
  --include-ext=c,h,cc,cpp,cxx,hpp,hxx,hh \
  --exclude-dir=build,_deps
```

Result:

```text
Language       files   blank   comment   code
C++               68    7562       839   29146
C/C++ Header      72    2324       732    5780
SUM:             140    9886      1571   34926
```

Count OpenNN native C/C++ plus CUDA/HIP LOC:

```bash
cloc opennn \
  --include-ext=c,h,cc,cpp,cxx,hpp,hxx,hh,cu,cuh,hip \
  --exclude-dir=build,_deps
```

Result:

```text
Language       files   blank   comment   code
C++               68    7562       839   29146
C/C++ Header      72    2324       732    5780
CUDA               6     445        13    2143
SUM:             146   10331      1584   37069
```

Count PyTorch native C/C++ LOC:

```bash
cloc pytorch-v2.12.0/aten pytorch-v2.12.0/c10 pytorch-v2.12.0/torch/csrc pytorch-v2.12.0/caffe2 \
  --include-ext=c,h,cc,cpp,cxx,hpp,hxx,hh \
  --exclude-dir=test,tests,benchmark,benchmarks,docs,examples,third_party,build,generated
```

Result:

```text
Language       files    blank   comment    code
C++             1605    64005     56262   552947
C/C++ Header    2133    46067     57749   252974
C                185     2890      2104    28398
SUM:            3923   112962    116115   834319
```

Count PyTorch native C/C++ plus CUDA/HIP LOC:

```bash
cloc pytorch-v2.12.0/aten pytorch-v2.12.0/c10 pytorch-v2.12.0/torch/csrc pytorch-v2.12.0/caffe2 \
  --include-ext=c,h,cc,cpp,cxx,hpp,hxx,hh,cu,cuh,hip \
  --exclude-dir=test,tests,benchmark,benchmarks,docs,examples,third_party,build,generated
```

Result:

```text
Language       files    blank   comment    code
C++             1605    64005     56262   552947
C/C++ Header    2133    46067     57749   252974
CUDA             387    11217      8141    86257
C                185     2890      2104    28398
SUM:            4310   124179    124256   920576
```

## Interpretation

OpenNN's native library source is much smaller because it is a focused C++ neural-network
library. PyTorch's native runtime is much broader: it includes a general-purpose tensor
engine, dispatch system, autograd infrastructure, many operators, and runtime integration
layers.

That broader scope is valuable for PyTorch's use cases. The LOC comparison should therefore
not be read as "smaller is always better." It should be read as a concrete source-size
context for deployment-size and embedded/edge discussions: OpenNN has a smaller native
implementation surface, while PyTorch carries a much larger general-purpose runtime.

## Why fewer lines can matter

Fewer lines of production code can have practical benefits, especially for native
deployment:

* **Smaller audit surface.** There is less implementation to inspect when reviewing
  numerical behavior, memory ownership, threading, error handling, or security-sensitive
  paths.
* **Lower maintenance burden.** A smaller codebase is usually easier to refactor, port,
  document, and keep consistent across compilers and platforms.
* **Faster onboarding.** Engineers can understand a focused library more quickly than a
  broad runtime with many subsystems and historical layers.
* **Less build complexity.** Fewer components and dependencies generally mean simpler
  builds, fewer generated artifacts, and fewer platform-specific integration problems.
* **Smaller bug surface.** LOC does not predict bugs by itself, but every subsystem adds
  possible interactions. A focused implementation has fewer interactions to reason about.
* **Better fit for embedded and edge use.** When the target is a small native application,
  a compact source and runtime footprint can make the difference between a deployable
  solution and one that needs a separate trimming/export pipeline.

The tradeoff is scope. PyTorch's larger native runtime buys a very broad operator set,
dynamic dispatch, extensive integration points, and a large ecosystem. OpenNN's smaller
codebase is an advantage when the task fits its focused C++ library model.
