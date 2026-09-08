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

UBSan also found a histogram bin estimate converting NaN to an integer for a
zero-width range. The estimate is now bounded before conversion, retaining the
subsequent boundary refinement. New zero-width and subnormal-width regressions
are included. The 36 statistics/testing-analysis checks pass under ASan/UBSan
and in the Windows CPU build after this fix.

The next sanitizer pass reached optimizer serialization and exposed another
out-of-range cast. JSON stores numbers as doubles, so the maximum signed integer
used for unlimited validation failures rounds to 2^63. Integer conversion now
recovers that existing boundary value explicitly and rejects nonfinite or larger
values before casting. Boundary regressions and the optimizer settings round-trip
assertion pass in the focused ASan/UBSan and Windows CPU checks (nine tests).

A CUDA-enabled Windows build was also exercised with `CUDA_VISIBLE_DEVICES=-1`.
The network and device-query checks passed 33 tests and skipped two GPU snapshot
round trips. Those two tests now check device availability before requesting
CUDA; both still execute and pass with the GPU visible (31 network tests pass).

Publication also requires the compatibility and merge review described in
`RELEASE_READINESS.md`; these hardening checks do not constitute publication of
a master release.

Raw build and test logs and generated export files are retained outside the
checkout. Optional backend skips are reported separately from passing tests.

## 2026-09-08: reconciled 9.0.0 candidate

Candidate `c8d919c72914c0ff61f91788be00002a496c08b9` merges development
`5840396e9` and master `efd566b38`. All 98 conflicted paths have recorded
resolutions in `MERGE_RECONCILIATION.md`. Master is an ancestor of the
reconciled development branch. Versioning targets 9.0.0; no tag was published.

- Windows MSVC 19.50 CPU: 1,071 unit tests ran; 1,030 passed and 41 skipped
  (GPU/optional facilities). All 26 response-optimization scenarios passed.
- Windows relocated-package consumer: configured, built and ran against
  OpenNN 9.0. This local build reused an external Eigen package, supplied
  explicitly to the consumer; hosted CI checks the fetched-dependency case.
- Linux CPU on the GPU runner: 1,030 passed, 41 skipped.
- Linux CUDA: 1,189 tests ran; 1,179 passed and 10 skipped for optional
  facilities. All 26 CUDA response-optimization scenarios also passed.
  One explicitly disabled test remains in the suite.
- JavaScript is executed with Node 24: all dense activations, difficult feature
  labels, categorical one-hot controls and six-output dropdown updates agree
  with native inference. Exporting does not rename the source network.
- New regressions cover all-missing histogram bins, genetic initialization
  JSON/legacy defaults, and atomic class labels with sequence truncation.
- The final documentation inventory passes for 14 groups / 10,131 indexed
  files, including three added per-dataset attribution notices. The separate
  `--release` clearance gate remains intentionally failing for 11 groups.
  Airfoil, both breast-cancer CSVs and all 10,000 MNIST test images have verified sources, transformations
  and attribution. Other sources/derivatives still need the records in DATASETS.md.

Local Clang 17 ASan/UBSan verification also passed both executables:
unit suite 340.55 seconds, integration suite 301.03 seconds, with
`detect_leaks=1:halt_on_error=1:allocator_may_return_null=1` and
`halt_on_error=1:print_stacktrace=1`. This reused the external WSL build cache
against the same candidate source.

The first hosted sanitizer attempt was cancelled after a 44-minute build
with several six-to-eight-minute gaps compiling unchanged files and no
compiler error. Only that job was retried on a fresh worker; the other
successful jobs were retained.

Hosted results for code commit `c8d919c72`:

- [Main CI, run 34167608153](https://github.com/Artelnics/opennn/actions/runs/34167608153):
  GCC 13 CPU, Clang 17 CPU, Windows MSVC 2022 CPU and CUDA 12.9 compilation
  passed. The CPU jobs include relocated-package consumers.
- [Linux CUDA, run 34167608171](https://github.com/Artelnics/opennn/actions/runs/34167608171):
  full CPU and CUDA verification passed, including both integration runs.
- Hosted ASan/UBSan retry
  [job 101888844903](https://github.com/Artelnics/opennn/actions/runs/34167608153/job/101888844903)
  was still building when this record was written. Its final result is pending;
  the successful local sanitizer run is not represented as a hosted CI pass.
  Check the linked run before promoting the candidate.

The attribution/manifest follow-up `adad276c3` changes documentation and data
notices only. Its 14-group inventory was checked locally. That commit and this
verification record use `[skip ci]`; the executable code verified above is
unchanged. These records do not claim a completed six-job green matrix.

Neural Designer validation is deferred at the owner's request. No general
8.x XML/NDM conversion or representative production-model compatibility is
claimed. The merge retains 9.x time-series window-start indexing, documented
as a behavior change. These limitations and dataset clearance remain release
publication work, independent of the engineering test results.

The prior targeted CUDA memory checks apply to their recorded earlier commit;
this reconciliation did not change CUDA memory-management code. They were not
rerun as part of this batch. Raw logs remain outside Git.
