# OpenNN — instructions for Codex

See [README.md](README.md) for what this project is and generic build instructions.

## Build environment on this machine (Windows)

`cl.exe` and `ninja.exe` are not on PATH in a plain shell. Ninja ships bundled with
Visual Studio at:
`C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`

To get a working MSVC environment (`cl`, `link`, `INCLUDE`/`LIB`), source one of:
- `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat`
- `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`

### Existing build directories

| Dir | Generator | Config | Tests | Notes |
|---|---|---|---|---|
| `build` | Visual Studio 18 2026 | multi-config | OFF | needs `--config Release` on every `cmake --build` |
| `build-ninja` | Ninja | Release | ON | has a working `bin/run_tests.exe` already built; single-config, no `--config` flag needed |
| `build-fresh` | Ninja | Release | OFF | |
| `build-codex-tests` | Visual Studio | — | — | has a `RUN_TESTS` project but no `CMakeCache.txt` found; treat as possibly stale before relying on it |
| `build-cpu-audit`, `build-cuda-audit` | — | — | — | not inspected; audit/benchmark builds, check before assuming purpose |

Prefer `build-ninja` when you need both tests and a single-config build. The
`run-examples` skill assumes this directory.

## Project-local skills

`.agents/skills/` in this repo holds project-specific skills:
- `run-examples` — run the example matrix across CPU/GPU FP32/GPU BF16.

## Code organization

One ordering rule, so a reader meets concepts in the same sequence in every file.

### Header layout

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

### Class members

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

### Do not reorder data members mechanically

C++ initializes non-static data members in **declaration order**, not in
constructor-initializer-list order. Moving a data member can therefore change
initialization order and behavior, and will trip `-Wreorder`. Reorder members
only deliberately, with the constructors in view.

### `std::` is sometimes load-bearing

`opennn_types.h` does `using namespace std` globally and `using namespace
Eigen` inside `namespace opennn`, so unqualified names can resolve to Eigen or
to a class member instead of `std`. Keep the qualification when the name is
shadowed — `std::swap` inside a member named `swap`, `std::fill` inside a class
with a `fill` member, `std::set` inside a class with a `set` method,
`std::copy` where `opennn::copy` is the tensor overload, and `std::array`
anywhere `Eigen::array` is visible.

Never add `using namespace std` to a header. In `kernel.cuh`/`kernel_common.cuh`
it preceded Eigen's Tensor includes and broke nvcc's parse of them entirely.
