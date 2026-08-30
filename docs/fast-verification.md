# Fast local verification

OpenNN keeps the final quality gate unchanged: a library change is complete only
after the CPU and CUDA builds and their complete test suites pass. The fast
workflow avoids paying for that gate after every intermediate edit.

## Commands

On Windows, the PowerShell entry point finds Visual Studio, imports its x64
environment, and locates the bundled Ninja executable when they are not already
on `PATH`:

```powershell
# Normal edit/compile/test loop (CPU unless -Backend cuda is supplied).
.\tools\verify.ps1 quick -Filter 'Dense.*:DenseNoBiasTest.*'
.\tools\verify.ps1 quick -Backend cuda -Filter '*Gpu*:*CUDA*'

# Checkpoints and the final gate.
.\tools\verify.ps1 cpu
.\tools\verify.ps1 cuda
.\tools\verify.ps1 full
```

On Linux:

```bash
./tools/verify.sh quick --filter 'Dense.*:DenseNoBiasTest.*'
./tools/verify.sh quick --backend cuda --filter '*Gpu*:*CUDA*'

./tools/verify.sh cpu
./tools/verify.sh cuda
./tools/verify.sh full
```

The modes have deliberately different jobs:

| Mode | Builds | Tests | When to use it |
| --- | --- | --- | --- |
| `quick` | Incremental `opennn_tests` target for one backend | Required GoogleTest filter, fail-fast | After an edit |
| `cpu` | Incremental CPU `opennn_tests` target | Complete CPU suite | CPU checkpoint |
| `cuda` | Incremental CUDA `opennn_tests` target | Complete CUDA suite | CUDA checkpoint |
| `full` | CPU, then CUDA | Both complete suites | Once, before declaring the batch done |

`quick` rejects an empty filter and asks GoogleTest to fail if the filter selects
no tests. This prevents a typo from producing a false green result. GoogleTest
filters accept `*` and `?`, with positive patterns separated by `:`.

## Build cache

The wrappers keep generated files outside the source checkout by default:

- Windows: `%LOCALAPPDATA%\OpenNN\build\<checkout-hash>\{cpu,cuda}`
- Linux: `$XDG_CACHE_HOME/opennn/build/<checkout-hash>/{cpu,cuda}`, or
  `~/.cache/opennn/build/...` when `XDG_CACHE_HOME` is unset

The checkout hash prevents two clones from sharing an incompatible CMake cache.
Set `OPENNN_BUILD_ROOT`, or pass `-BuildRoot`/`--build-root`, to choose another
cache root. Use `-Reconfigure`/`--reconfigure` after deliberately changing a
cached toolchain option. Ordinary source and CMake changes are detected by the
generated Ninja build without this flag.

Only `opennn_tests` and its dependencies are built. Examples and benchmarks are
excluded from these verification configurations. Build their targets separately
when they change or when a public API change can affect them.

On Windows, CUDA verification selects the newest installed CUDA toolkit and then
selects a cuDNN installation built for the same CUDA major version. It fails with
a diagnostic instead of mixing incompatible runtime families. Set both
`OPENNN_CUDNN_INCLUDE_DIR` and `OPENNN_CUDNN_LIBRARY` to override discovery.

## Compiler cache

If `sccache` is on `PATH` when a verification directory is first configured, the
workflow automatically assigns it as the C++ and CUDA compiler launcher. This is
most useful after branch switches, repeated clean builds, and when several
checkouts compile the same revision.

On Windows, an official package is available through:

```powershell
winget install Mozilla.sccache
.\tools\verify.ps1 quick -Reconfigure -Filter 'ConfigurationTest.*'
sccache --show-stats
```

Install `sccache` through the distribution package manager or an official release
on Linux, then use `--reconfigure` once. Pass `-NoSccache` or `--no-sccache` to
disable it for diagnosis.

## Working rule

Use focused tests while code is changing. Group related header and public-signature
edits before compiling because those edits necessarily invalidate many translation
units. Run `full` once after the coherent batch is finished; do not replace the
final gate with focused tests.
