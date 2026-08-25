# OpenNN code organization

The conventions this codebase actually follows, with the reasoning behind the ones
that look arbitrary. Split out of `AGENTS.md` on 2026-08-24; it is the same text.

Most rules here exist because breaking them broke something specific, and those
cases are recorded inline — the exceptions are the load-bearing part, so read them
before "simplifying" an enum position or a `std::` qualification.

## Folder layout

The library is split by responsibility, and the folders are ordered by
dependency — each one may include the ones above it, never the ones below:

```text
opennn/core/                    types, tensors, device backend, memory, utilities
opennn/core/cuda/               .cu/.cuh kernels
opennn/neural_network/          network, propagation, expression export
opennn/neural_network/layers/
opennn/neural_network/operators/
opennn/models/                  ready-to-use task, vision and language networks
opennn/dataset/                 tabular, image, language, time series, YOLO
opennn/training_strategy/       losses and optimizers
opennn/model_selection/
opennn/testing_analysis/
opennn/                         pch.h and registry.{h,cpp} only
```

Models sit above the network primitives: they assemble layers and operators into
ready-to-use networks without depending on datasets or training. Datasets sit
above both on purpose: the language datasets need `tokenizer_operator.h` and
`yolo_dataset` needs the convolutional layer, while neither `neural_network/` nor
`models/` includes a dataset.

`models/` is deliberately flat. `models.h` is its public header; `models.cpp`,
`vision_models.cpp` and `language_models.cpp` divide the implementations without
creating a directory or class hierarchy for every architecture. The old
`neural_network/standard_networks.h` remains a compatibility include only.

`registry.{h,cpp}` stays at the root because it constructs `Layer`, `Optimizer`
and `InputsSelection` — it spans three folders and belongs above all of them.

`tests/` mirrors those folders one for one, so a test sits at the same
relative path as what it exercises — `dense_layer.cpp` is tested by
`tests/neural_network/layers/dense_layer_test.cpp`. Only the harness stays at
`tests/`: `pch`, `numerical_derivatives`, `test.cpp`, and `registry_test.cpp`
beside the `registry` it covers. A new test goes in the folder of the thing it
tests; CMake globs recursively, so nothing else needs touching.

Every include names its folder, from the repo root, with no exceptions:
`#include "opennn/neural_network/layers/dense_layer.h"`, and
`#include "tests/pch.h"` for the harness. Bare neighbour includes do not
resolve — only the repo root is on the include path.

Two known upward includes remain, both `.cpp`-only, both deliberate:
`back_propagation.cpp` needs `Loss` (its header only forward-declares it), and
`correlations.cpp` trains a small network to get a nonlinear correlation.

One ordering rule, so a reader meets concepts in the same sequence in every file.

## Header layout

```text
license/title comment
#pragma once
includes            (own header first in .cpp, then C, C++ std, third-party, project)
forward declarations
namespace-scope constants
namespace-scope enums
types (structs, classes)
free-function declarations
```

Enums go at the **top of the scope that owns them** — first thing inside
`namespace opennn`, or first thing inside the class — *when the enum is bare
vocabulary*. Two deliberate exceptions, both about keeping an enum next to its
meaning:

- **An enum with an attached helper cluster stays with it.** `variable.h`
  declares `VariableType`, then its `EnumMap` + `to_string` + `from_string`,
  then `ScalerMethod` and its cluster. Hoisting the enums to the top would
  separate each from its own converters. Concept grouping wins.
- **A sub-topic enum in a multi-topic header stays local.** `io_utilities.h`
  keeps `DateFormat` beside `detect_date_format`/`date_to_timestamp` rather
  than 190 lines away from its only users.

So: never sweep enums to the top mechanically. Hoist one only when it is
loose vocabulary with no attached helpers, as in `chat.h` and `memory_pool.h`.

## Class members

`public:` → `protected:` → `private:`, each appearing once. Within a section:

1. types and aliases (`using`, nested `enum`/`struct`)
2. static constants
3. factory functions
4. constructors and assignment operators
5. destructor
6. all other member functions
7. **data members last**

This is the Google C++ Style Guide order; `dense_layer.h` is the reference
example in this repo.

## Do not reorder data members mechanically

C++ initializes non-static data members in **declaration order**, not in
constructor-initializer-list order. Moving a data member can therefore change
initialization order and behavior, and will trip `-Wreorder`. Reorder members
only deliberately, with the constructors in view.

## `std::` is sometimes load-bearing

`opennn_types.h` does `using namespace std` globally and `using namespace
Eigen` inside `namespace opennn`, so unqualified names can resolve to Eigen or
to a class member instead of `std`. Keep the qualification when the name is
shadowed — `std::swap` inside a member named `swap`, `std::fill` inside a class
with a `fill` member, `std::set` inside a class with a `set` method,
`std::copy` where `opennn::copy` is the tensor overload, and `std::array`
anywhere `Eigen::array` is visible.

Never add `using namespace std` to a header. In `kernel.cuh`/`kernel_common.cuh`
it preceded Eigen's Tensor includes and broke nvcc's parse of them entirely.
