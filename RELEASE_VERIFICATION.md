# Release hardening verification

Date: 2026-09-07. Base: `dev` at `48efcc03f`, after fetching all remotes and
fast-forwarding `origin/dev`. Local hardening work was reconciled with the
incoming Qwen, YOLO and anomaly-detection changes.

## Changes

- Order default CUDA copies and clears on the active compute stream. Compute
  uses nonblocking streams, so a default-stream copy could read gradients before
  their kernels completed. Default copies still wait before returning; callers
  supplying a stream retain asynchronous behavior.
- Add three stream-ordering regression tests. The buffer-migration regression
  failed before the fix and all three passed afterward. The focused CUDA suite
  passed 82 tests, with one optional FlashAttention test skipped. No numerical
  tolerances were relaxed. The C2PSA comparison now uses the same ordered copy API.
- Restore each optimizer test fixture's previous thread count, preventing it
  from overriding `OPENNN_THREADS` for subsequent suites.
- Preserve retained matrix rows when the feasibility-study helper shrinks its
  result cloud. Assigning an unevaluated view back to its resized source caused
  corrupted rows and a Linux integration failure.
- Allow static GPU buffers to release storage after the backend or block cache
  has shut down, without accessing destroyed streams or cache tables. The Linux
  response executable initially passed all 26 assertions and then crashed while
  destroying its static network. A fresh-process exit regression failed before
  this fix and passes afterward.
- Include the 26 response-optimization scenarios in full verification and CTest.
  Require a CUDA build and an available GPU when the CUDA verification mode runs.
- Provide configurable logging with callback lifetime protection, concurrent
  sink replacement, recursion protection and exception containment.
- Fix the installed JPEG dependency and build configuration so a separate
  `find_package(OpenNN)` consumer works after moving the installation prefix.
- Use the Windows system-directory search for NVML, guard GPU synchronization on
  the CPU loss path, and support spaces in header-check include paths.

## Verification results

| Platform and backend | Unit tests passed | Skipped | Integration scenarios |
| --- | ---: | ---: | --- |
| Windows CPU | 1,020 | 41 | 26/26 passed |
| Windows CUDA | 1,171 | 8 | 26/26 passed |
| Linux (WSL) CPU | 1,018 | 43 | 26/26 passed |
| Linux (WSL) CUDA | 1,167 | 12 | 26/26 passed |

All four backend/platform gates completed without failures. Both CUDA response
executables exited successfully after the shutdown fix. Their final integration
runs took 1,167 seconds on Windows and 1,379 seconds on WSL while sharing the GPU.

Windows uses MSVC 19.50, CUDA 13.3 and cuDNN 9.19. Linux uses GCC 13.3,
CUDA 12.9 and cuDNN 9.10. Both GPU runs use an NVIDIA RTX 3060 Laptop GPU.
CPU builds explicitly disable CUDA. All runs set `OPENNN_THREADS=4`.

The ordinary suite skips five opt-in export generators, GPU tests in CPU-only
builds, unavailable FlashAttention kernels, and two oneDNN tests when that
optional backend is absent. Linux cuDNN 9.10 also skips two BF16 recurrent
comparisons that require newer cuDNN support. Two Python execution checks skipped
because the Linux installation exposed `python3` but not `python`; supplemental
checks use a virtual environment with NumPy, and CI installs Python 3.12 and
NumPy 2.4.4 explicitly. Skips are not
counted as passes. One timing benchmark is disabled by default.

Additional checks completed:

- Windows and Linux: install, move the prefix, build and run the independent
  package consumer, including inference and JPEG linkage.
- Linux: all OpenNN public headers compile in isolation.
- Windows: all five optional expression/topology dump tests pass when enabled.
- Linux CPU and CUDA: all 10 expression execution tests pass with Python 3.12.3
  and NumPy 2.4.4, including the Python checks skipped in the main runs.
- Windows: the normally disabled LSTM timing benchmark passes when explicitly
  enabled. It remains disabled in the ordinary correctness suite.
- Requesting CUDA while explicitly disabling it fails configuration as intended.

## Continuous integration

The previous CI workflow was removed in `55d285b2f` on 2026-09-04. Repository
Actions were disabled and no self-hosted runners were registered when checked.
Actions have been enabled. CPU workflows cover Linux GCC 13,
Linux Clang 17 and Windows MSVC 2022, including the integration scenarios and
relocated-package checks. Hosted Linux CI also compiles the CUDA library and
both test executables without requiring a GPU. It uses CUDA 12.9 and cuDNN
9.10.2 from [NVIDIA's Ubuntu 24.04 repository](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/).
The Linux CUDA runtime workflow uses the `opennn-wsl-cuda` runner, registered
and brought online during the follow-up audit on the same day. It has the
`self-hosted`, `linux`, and `cuda` labels. Availability and service operations
are documented in [tools/CI_RUNNER.md](tools/CI_RUNNER.md).

The first hosted run exposed Clang 17 portability issues in the vision stage
table and optional output-window construction, inconsistent cuDNN package
versions, and missing pandas for Python exports. These were fixed without
weakening test assertions. Hosted CI now also includes ASan and UBSan.

The follow-up GPU memory check exposed a separate shutdown-order leak:
lazily initialized CUDA library runtimes could run their exit handlers before
the backend released its handles. Cleanup is now registered after each lazy
library initialization and is idempotent. The focused cuDNN handle test reported
5,768 leaked bytes before the fix and zero leaks and zero memory errors after
it, using Compute Sanitizer 13.3 and a cuDNN 9.20 runtime. A standalone cuDNN
create/destroy reproducer outside OpenNN was clean, isolating the lifetime issue.

The broader run then found 272 cached allocations left at exit. The CUDA block
cache now frees retained blocks and destroys its pooled events while the runtime
is still available. Repeating the backend, GPU comparison, and C2PSA suites with
Compute Sanitizer from CUDA 13.3 and a clean cuDNN 9.25.1 runtime passed 86 tests,
skipped the unavailable FlashAttention kernel, and reported **zero leaked bytes
and zero memory errors** (66.9 seconds). The normal CUDA configuration remains
cuDNN 9.10; that older runtime could not initialize under the WSL memory checker.
The process-exit death test is covered by ordinary CUDA verification because
child-process instrumentation hung on the tested WSL stack.

The zlib fossil server intermittently returned content that failed the pinned
SHA-256 check in fresh hosted jobs. The primary URL now uses the upstream GitHub
release asset, with the fossil URL as fallback. Both serve the same verified
1.3.2 archive; the dependency version and expected hash are unchanged.

Clang's Release static archive also required LTO flags on downstream linkers.
The exported target now supplies the compiler's flags. An independent minimal
archive/consumer reproduced the original link failure and passed with the fix.
Linux CI checks both Release and Debug consumers of the relocated installation.

ASan keeps leak detection and fail-on-error enabled, but permits `malloc` to
return null so the existing impossible-allocation regression can verify
`std::bad_alloc`. The sanitizer preset gives the exhaustive numerical unit suite
1,800 seconds; ordinary unit verification retains its 600-second limit.
Current hosted results are available in the repository's
[Actions runs](https://github.com/Artelnics/opennn/actions).

Publication also requires the compatibility and merge review described in
`RELEASE_READINESS.md`; these hardening checks do not constitute publication of
a master release.

Raw build and test logs and generated export files are retained outside the
checkout. Optional backend skips are reported separately from passing tests.
