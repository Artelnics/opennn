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
The Linux CUDA runtime workflow requires an online runner
labelled `self-hosted`, `linux`, `cuda`, with CUDA and cuDNN 9 installed.

Local WSL CUDA verification does not register a GitHub runner. Until one is
provided, that workflow will queue rather than execute. The release still needs
a green hosted CI run and the remaining release work in `RELEASE_PLAN.md`;
these hardening checks do not constitute publication of a master release.

Raw build and test logs and generated export files are retained outside the
checkout. Optional backend skips are reported separately from passing tests.
