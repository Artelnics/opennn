# OpenNN — instructions for coding agents

See [README.md](README.md) for what this project is and generic build instructions.
This file is the single entry point for agent-facing documentation; everything else
is linked from here.

| Topic | Where |
| --- | --- |
| Code organization, header layout, class member order, `std::` caveats | [docs/architecture.md](docs/architecture.md) |
| Current engineering status, audit findings, YOLO roadmap | [docs/status/engineering-audit.md](docs/status/engineering-audit.md) |
| YOLO implementation notes, session by session | [docs/status/yolo-session-log.md](docs/status/yolo-session-log.md) |
| Project-local skills | [.agents/skills/](.agents/skills/) |

Before deleting anything that looks unused, read
[Before deleting anything: Neural Designer](docs/status/engineering-audit.md#before-deleting-anything-neural-designer).
Neural Designer links against this library and uses many symbols that look orphaned
from inside this repo, so dead-code analysis run only here produces false positives.

## Build environment on this machine (Windows)

`cl.exe` and `ninja.exe` are not on PATH in a plain shell. Ninja ships bundled with
Visual Studio at:
`C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe`

To get a working MSVC environment (`cl`, `link`, `INCLUDE`/`LIB`), source one of:
- `C:\Program Files\Microsoft Visual Studio\18\Community\Common7\Tools\VsDevCmd.bat`
- `C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat`

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
