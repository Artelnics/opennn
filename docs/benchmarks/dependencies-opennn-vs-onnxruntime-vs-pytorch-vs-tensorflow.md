# Dependencies & install friction: OpenNN vs PyTorch vs TensorFlow

Size, startup time, and memory all describe a running model. Before any of that, there is the
question of **getting it onto the machine**: how many packages must be installed, how many files
land on disk, and — the real test for a locked-down or air-gapped target — whether the deployed
artifact runs at all on a clean system with nothing pre-installed.

## Contents

- [The numbers](#the-numbers)
- [What each requires](#what-each-requires)
- [Why it matters](#why-it-matters)
- [Caveats](#caveats)
- [References](#references)

## The numbers

|  | OpenNN | ONNX Runtime | PyTorch | TensorFlow |
| --- | --- | --- | --- | --- |
| **Packages to install** | **0** | **6** | **12** | **33** |
| **Files installed** | **1** (the executable) | **2,997** | **21,755** | **21,213** |
| **On-disk install** | **~6 MB** | **~140 MB** | **~946 MB** | **~1.6 GB** |
| **Extra runtime required** | none | a Python interpreter | a Python interpreter | a Python interpreter |
| **Runs on a clean machine (no toolchain, no Python)?** | **yes** | no | no | no |

The OpenNN deployment is a single native executable. ONNX Runtime, PyTorch, and TensorFlow are
Python package trees plus the interpreter to run them. ONNX Runtime is the lightest of the three
frameworks (it is inference-only), but still 6 packages and ~140 MB versus OpenNN's single file —
TensorFlow's is the largest, at 33 packages and ~1.6 GB.

## What each requires

**OpenNN — a single binary, zero installs.** OpenNN links statically into the executable, so
deployment is *one file*. Its only runtime dependencies are shared libraries that are already
present on any Linux system (or part of the GCC runtime) — confirmed with `ldd`:

```
libstdc++.so.6   libc.so.6   libm.so.6   libgcc_s.so.1   libgomp.so.1   libtbb.so.12
```

No package manager step, no Python, no virtual environment. As a portability check, the binary
**runs in a completely empty environment with a scrubbed `PATH`** (`env -i PATH=/nonexistent./app`)
— proof it depends on nothing you have to install.

**PyTorch — 12 packages, ~22k files, plus an interpreter.** A CPU-only `pip install torch` pulls
**12 packages**:

```
torch  numpy  sympy  networkx  jinja2  markupsafe  mpmath
fsspec  filelock  typing_extensions  setuptools  pip
```

…landing **21,755 files** and **~946 MB** in `site-packages`. And that is on top of a working
**Python interpreter**, which is itself a prerequisite (and not counted above). On an air-gapped or
minimal host you must provision Python and all of these before the model runs.

**TensorFlow — 33 packages, ~21k files, ~1.6 GB, plus an interpreter.** A `pip install
tensorflow-cpu` pulls **33 packages** (TF plus abseil, protobuf, gRPC, Keras, h2/grpcio, ml-dtypes,
and more), landing **21,213 files** and **~1.6 GB** in `site-packages` — the largest of the three —
again on top of a Python interpreter.

**ONNX Runtime — 6 packages, ~3k files, ~140 MB, plus an interpreter.** A `pip install onnxruntime`
pulls **6 packages** (onnxruntime, numpy, protobuf, flatbuffers, packaging, pip), landing **2,997
files** and **~140 MB** — the lightest framework here, since ORT only runs inference. It still needs
a Python interpreter and a model trained elsewhere, whereas OpenNN's one file trains and runs the
model itself.

## Why it matters

- **Air-gapped / locked-down targets:** copying one file beats provisioning Python plus a 900 MB
  package tree, especially where a package manager isn't available.
- **Containers:** an OpenNN image can be `FROM scratch` + one binary; a PyTorch image needs a
  Python base layer and the full dependency tree, dominating image size and build time.
- **Supply-chain surface:** 0 third-party packages versus 11 transitive dependencies is a smaller
  set of things to vet, pin, and keep patched.
- **Reproducibility:** a single self-contained binary has no "works on my machine" dependency
  drift.

## Caveats

- This counts the **runtime/deployment** dependencies, not build-time ones. Building OpenNN from
  source needs a C++ toolchain and CMake; the point here is what the *deployed* artifact requires,
  which is nothing extra.
- PyTorch's 12 packages / ~946 MB are the CPU-only install measured in a clean virtual environment
  (torch 2.12.0+cpu, CPython 3.12). A CUDA install adds the `nvidia-*` packages and is far larger.
- OpenNN's listed shared libraries ship with essentially every Linux distribution and the GCC
  runtime; on an unusual minimal image they can be bundled, or the binary linked more statically.
- Numbers measured on Linux x86_64; the structure (one native binary vs. an interpreted package
  tree) is the same across platforms.

## References

- [OpenNN](https://www.opennn.net/).
- [PyTorch installation guide](https://pytorch.org/get-started/locally/).
- [TensorFlow installation guide](https://www.tensorflow.org/install).
- [ONNX Runtime installation documentation](https://onnxruntime.ai/docs/install/).
