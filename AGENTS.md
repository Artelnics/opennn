# OpenNN — instructions for coding agents

See [README.md](README.md) for what this project is and generic build instructions.
This file is the single entry point for agent-facing documentation; everything else
is linked from here.

| Topic | Where |
| --- | --- |
| Code organization, header layout, class member order, `std::` caveats | [docs/architecture.md](docs/architecture.md) |
| Current engineering status, audit findings, YOLO roadmap | [docs/status/engineering-audit.md](docs/status/engineering-audit.md) |
| YOLO implementation notes, session by session | [docs/status/yolo-session-log.md](docs/status/yolo-session-log.md) |
| CUDA-graph topology dumps (training) | [docs/uml/cuda-graph/](docs/uml/cuda-graph/) |
| Project-local skills | [.agents/skills/](.agents/skills/) |

Before deleting anything that looks unused, read
[Before deleting anything: Neural Designer](docs/status/engineering-audit.md#before-deleting-anything-neural-designer).
Neural Designer links against this library and uses many symbols that look orphaned
from inside this repo, so dead-code analysis run only here produces false positives.

## Build environment on this machine (Windows)

`cl.exe` and `ninja.exe` are not on PATH in a plain shell. Both ship inside the Visual
Studio install, so locate it rather than hard-coding a version — this repo is used from
more than one machine and the VS version differs between them:

```bat
for /f "usebackq tokens=*" %i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath`) do set VSROOT=%i
call "%VSROOT%\VC\Auxiliary\Build\vcvars64.bat"
set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
```

`vcvars64.bat` needs `vswhere.exe` on PATH to resolve the toolset; if it prints
`'vswhere.exe' is not recognized`, prepend
`%ProgramFiles(x86)%\Microsoft Visual Studio\Installer` first.

`VsDevCmd.bat` (in `%VSROOT%\Common7\Tools\`) is the equivalent for a non-x64 default.

**Known gap on the Windows box as of 2026-08-24:** VS 2022 Community 17.14 is installed
with the MSVC toolset (14.44.35207), but the Windows SDK resource tools are missing —
`rc.exe` is not found and CMake reports `CMAKE_MT-NOTFOUND`. `cl` compiles, but linking
fails, so CMake cannot get past its own compiler probe. Install the Windows SDK
component before expecting a local build here.

### Creating the two build directories

No build directory is checked in, and they are all gitignored — expect to create
these yourself. Two configurations cover the work; both are Ninja + Release +
single-config, so `cmake --build <dir>` needs no `--config` flag, and both produce
`bin/opennn_tests.exe`.

The fast CPU check:

```sh
cmake -S . -B build-consolidated -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON
cmake --build build-consolidated
```

Anything touching GPU paths, plus the benchmark targets:

```sh
cmake -S . -B build-resnet-capacity -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build build-resnet-capacity
```

On Windows, cuDNN is installed outside the CUDA toolkit and `FindCUDNN.cmake`
only hints at `CUDAToolkit_INCLUDE_DIRS` and the Linux paths, so the CUDA
configure above fails with `Could NOT find CUDNN` until it is told where to
look:

```sh
      -DCUDNN_INCLUDE_DIR="C:/Program Files/NVIDIA/CUDNN/v9.19/include/13.1"       -DCUDNN_LIBRARY="C:/Program Files/NVIDIA/CUDNN/v9.19/lib/13.1/x64/cudnn.lib"
```

The trailing directory is the CUDA major version cuDNN was built for, not the
cuDNN version: a v9.19 install for CUDA 13 puts its headers under `include/13.1`.
Running the tests needs `bin/13.1/x64` ahead of `bin/12.9/x64` on PATH for the
same reason — with the 12.9 directory first the binary loads a cuBLASLt built
for CUDA 12 and dies part-way through the suite.

`OpenNN_BUILD_TESTS` and `OpenNN_BUILD_EXAMPLES` default to `ON`, so neither needs a
flag; `OpenNN_BUILD_BENCHMARKS` defaults to `OFF`. `CMAKE_CUDA_ARCHITECTURES` defaults
to `native`, which is right whenever the GPU is visible at configure time — if it is
not, CMake falls back to a value that cannot compile the packed-bf16 kernels, so pass
it explicitly (`-DCMAKE_CUDA_ARCHITECTURES=89` for Ada, `86` for Ampere).

`OPENNN_HAS_CUDA` is set from a non-FORCE cache entry, so **a reconfigure keeps
whichever CUDA decision the directory made first**. To flip a tree between CPU and
CUDA, delete it and configure again rather than re-running `cmake` over it.

A library change should be built and run in **both** before you call it done.

Directory names referred to in older notes (`build-ninja`, `build-fresh`,
`build-cpu-audit`, `build-std-cleanup`, `build_cmake`, `build-benchmarks`,
`build-mkl`, `build-cpu-verification`, ...) do not exist; do not go looking for them.

## Project-local skills

`.agents/skills/` in this repo holds project-specific skills:
- `run-examples` — run the example matrix across CPU/GPU FP32/GPU BF16.

## Code organization

Moved to [docs/architecture.md](docs/architecture.md) — folder layout and the
dependency order between folders, header layout, class member order, the two
deliberate upward includes, and the places where `std::` qualification is
load-bearing. Read it before moving a file, hoisting an enum, or reordering data
members.
