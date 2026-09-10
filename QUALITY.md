# Engineering quality gates

OpenNN's release branch uses one required CI workflow for supported builds and
runtime checks. A candidate commit must pass all jobs before promotion.

## Runtime-neutral build profiles

`OpenNN_CPU_TARGET=NATIVE` is the default and retains the existing local
throughput settings (`-march=native`, `/arch:AVX2`, or the Apple Silicon target).
Binary release packages use `OpenNN_CPU_TARGET=PORTABLE`; the selected value is
recorded in `build-info.json` and the archive name. The portable profile removes
only explicit host-specific ISA flags. It does not change tensor layouts,
allocations, algorithms, thread counts, CUDA kernels, or the native default.

CI installs and runs static and shared-library consumers on Linux, a portable
AppleClang consumer on macOS, and static consumers on GCC, Clang and MSVC. CUDA
is compiled on a hosted runner and executed on the registered GPU runner.

## Analysis, coverage and fuzzing

The static-analysis job treats selected Clang analyzer ownership defects and
use-after-move findings as errors in central persistence, dataset and device
modules. This focused blocking set avoids accepting a large unreviewed warning
baseline.

The coverage job executes the CPU unit suite with GCC instrumentation and
requires at least 25% line and 15% branch coverage across non-CUDA library
sources. Its JSON and HTML reports are retained as CI artifacts. These are
initial repository-wide floors, not a claim that every subsystem is adequately
covered; thresholds should rise as tests are added.

The sanitizer build also links `opennn_json_fuzz` with libFuzzer, ASan and
UBSan. CI runs 20,000 mutations from the checked-in corpus. Longer local runs
can use:

```sh
cmake --preset verify-sanitizers -B ../build-fuzz -DOpenNN_BUILD_FUZZERS=ON
cmake --build ../build-fuzz --target opennn_json_fuzz --parallel 2
../build-fuzz/bin/opennn_json_fuzz -max_total_time=3600 tools/fuzz/corpus
```

The JSON parser rejects non-standard numbers, unescaped control characters,
invalid surrogate pairs, nesting beyond 256 containers and inputs over 256 MiB.
