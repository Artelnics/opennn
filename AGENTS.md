# OpenNN — instructions for coding agents

See [README.md](README.md) for what this project is and generic build instructions.
Project-local skills live in [.agents/skills/](.agents/skills/).

Before deleting anything that looks unused, remember that Neural Designer links
against this library and uses many symbols that look orphaned from inside this
repository. Dead-code analysis performed only here therefore produces false
positives.

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

### Fast verification (preferred)

Use the cross-platform wrappers while editing. They keep persistent CPU and CUDA
builds outside the OneDrive checkout, build only `opennn_tests`, and accept a
GoogleTest filter for fast feedback:

```powershell
.\tools\verify.ps1 quick -Filter 'Dense.*:DenseNoBiasTest.*'
.\tools\verify.ps1 quick -Backend cuda -Filter '*Gpu*:*CUDA*'
.\tools\verify.ps1 full
```

```bash
./tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*'
./tools/verify.sh quick --backend cuda --filter '*Gpu*:*CUDA*'
./tools/verify.sh full
```

Focused checks are the edit loop; `full` remains the final gate and runs both
complete suites once. The wrappers expose their cache, backend and compiler-cache
options through their command-line help.

### Creating the two build directories manually

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

`OpenNN_ENABLE_ONEDNN` is `AUTO`: oneDNN is used when found, skipped with a status
line when not, and `ON` turns a missing one into a configure error. It matters more
than most flags — without it the CPU recurrent layers fall back to the built-in path,
which measured **2.9x slower** on the benchmark's LSTM training cell and 1.9x on
inference, against a PyTorch whose wheel bundles oneDNN regardless. `AUTO` finds an
*installed* oneDNN; a prefix outside the standard paths and outside `$ONEDNN_ROOT`
still needs `-DOpenNN_ONEDNN_ROOT=...`:

```sh
      -DOpenNN_ONEDNN_ROOT=/home/artelnics/onednn-omp
```

Prefer an **OpenMP** build of oneDNN. A TBB-threaded one works but gives oneDNN its
own thread pool beside OpenNN's OpenMP one over the same cores; the configure warns
when it detects that. Downstream code must branch on `OpenNN_ONEDNN_FOUND`, never on
`OpenNN_ENABLE_ONEDNN` — the latter is a tri-state whose default is the string `AUTO`,
and `if("AUTO")` is true in CMake.

`OPENNN_HAS_CUDA` is set from a non-FORCE cache entry, so **a reconfigure keeps
whichever CUDA decision the directory made first**. To flip a tree between CPU and
CUDA, delete it and configure again rather than re-running `cmake` over it.

A library change should be built and run in **both** before you call it done.
That requirement applies to the completed batch, not to every intermediate edit.

Directory names referred to in older notes (`build-ninja`, `build-fresh`,
`build-cpu-audit`, `build-std-cleanup`, `build_cmake`, `build-benchmarks`,
`build-mkl`, `build-cpu-verification`, ...) do not exist; do not go looking for them.

## Project-local skills

`.agents/skills/` in this repo holds project-specific skills:
- `run-examples` — run the example matrix across CPU/GPU FP32/GPU BF16.

## Code organization

Follow the organization and naming of neighboring files before moving code,
hoisting an enum or reordering data members. Some `std::` qualifications and
include directions are load-bearing, so validate structural changes in both CPU
and CUDA builds.
