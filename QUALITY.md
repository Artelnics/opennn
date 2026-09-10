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

## Maintainability ratchets

`python tools/check_code_quality.py` measures first-party C++ code while
excluding the vendored FlashAttention shim. `CODE_QUALITY.json` is the reviewed
upper bound for logical and physical lines, duplicate code, oversized functions
and cyclomatic complexity. A change must simplify a regression or update the
baseline explicitly during review; ordinary feature work cannot silently grow
these measures.

`python tools/check_architecture.py` enforces the dependency direction between
core, datasets, networks, training, evaluation, model selection and response
optimization. A small path-specific exception list records existing cycles so
they cannot spread to other files.

Coverage is gated per module as well as repository-wide. Performance changes
can be checked with `python benchmarks/compare.py baseline.json candidate.json`;
the default controlled-machine gate permits 5% measurement variation in
throughput and memory and rejects lower confirmed batch capacity.
