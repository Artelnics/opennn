# OpenNN: from `dev` to a professional-grade `master` release

Historical audit notes for `dev` at `c19e76a64` (2026-09-07), and the plan
for a release. Counts, timings and passing-test claims below describe that
checkout and environment; they are not a release certification. The current
hardening results are recorded in `RELEASE_VERIFICATION.md`.

## 1. What the package is today

**Code.** `opennn/` is 253 files, 83,000 lines of C++20 (GCC 13+, Clang 17+;
`std::format`), in eight modules: core 18.9k lines, network 32.5k,
dataset 13.1k, training_strategy 9.3k, model_selection 2.5k,
response_optimization 2.7k, models 2.7k, testing_analysis 1.2k. CUDA is 21
`.cu` and 24 `.cuh` files under `core/cuda`. Eight source files exceed 2,000
lines (`network.cpp` 2,990, `tabular_dataset.cpp` 2,853,
`device_backend.cpp` 2,756, `optimizer.cpp` 2,421, `tensor_operations.cpp`
2,368, `model_expression.cpp` 2,238, `yolo_dataset.cpp` 2,231,
`long_short_term_memory_layer.cpp` 2,146). There are no TODO/FIXME markers
left in the library. One header still has `using namespace std`. All 107
public headers compile in isolation (`tools/check_headers.sh`, run on this
tree with the build's Eigen): the include discipline is already there.

**Configuration surface.** The library reads **75** `OPENNN_*` environment
variables (27 `getenv` sites plus the `env_flag_enabled`/`env_int_or`
helpers). They mix supported settings (threads, BLAS, TF32, CUDA graphs,
autotune on/off, plan-cache directory), tuning policy knobs with measured
defaults (tile tolerance, cross-source gain, energy tolerance), diagnostics
(`OPENNN_PROFILE`, `OPENNN_MEMORY_DEBUG`, `*_VERBOSE`), and experiments whose
outcome is already decided (`OPENNN_RNN_*` layout switches,
`OPENNN_SAVE_TRANSACTION_V1`, `OPENNN_CUTLASS_NARROW_K_VARIANT`). None are
documented outside source comments and the benchmark reports. The library
writes to `cerr`/`cout` at 130 sites; there is no logging facility.

**Build.** CMake 3.24 with `project(OpenNN VERSION 8.0.1)`, options for tests,
examples, benchmarks, header checking and `OpenNN_DISABLE_CUDA`; static or
shared; `install()`, an exported target set and a package-version file, so
`find_package(OpenNN)` is designed in; `CMakePresets.json` with verify
presets; `-Wall -Wextra -Wpedantic -Wswitch-enum` on. Dependencies: Eigen 5
(`find_package` with FetchContent fallback), MKL (required on the CPU path),
oneDNN, optional TBB, libjpeg-turbo 3.1.4 fetched by URL, cudnn_frontend
fetched, a flash-attention shim, nlohmann json, CUTLASS; no third-party
notices file. Residual warnings: `-Wswitch-enum` (32 in one translation
unit, a CUDA data-type switch) and `[[nodiscard]]` in the cuDNN frontend
calls. The README says CMake 3.18+; the build requires 3.24. MSVC is claimed
and `tools/verify.ps1` exists; nothing here exercises it.

**Tests.** 1,149 tests in 7 suites, all passing as of `c19e76a64` (1,141
passed, 8 skipped), about 200 s on this machine including CUDA. 77
skip/disable sites, almost all "no CUDA device", one `DISABLED_` benchmark
boundary test; no `tests/models`; no coverage measurement; no sanitizer run;
**no CI** — `.github/workflows` was removed in the last clean-up.

**Examples.** 16 targets. 14 run from the build directory and print a
credible metric (this session, one run each); `yolo` needs a dataset
argument that does not exist on this machine; `qwen3` downloads an 8.2 GB
model on first run with no resume or progress contract. Example data is
tracked in git: 10,102 BMP files (melanoma), 40 MB of MNIST, 7.6 MB ECG,
4.1 MB SST-2 — the pack is **647 MB**.

**Benchmarks.** A serious harness (`run.py`, protocol, gates for machine
quietness, clocks, quality; per-launch artifacts) with six reviewed reports
in `benchmarks/reports/`. `results/` is local by design. Fine as it is; it
is the strongest part of the repository's evidence.

**Documentation.** `README.md` (207 lines) and `AGENTS.md`. `docs/` was
removed. 0 of 107 public headers carry API documentation comments. No
CHANGELOG, no CONTRIBUTING, no API reference, no user guide. License:
LGPL-3.0.

**Git.** `dev` is 1,309 commits ahead of `master` (last commit 2026-08-11);
`master` has 64 commits `dev` does not (to be reconciled before any merge).
Seven remote branches besides the two, the oldest from 2025-07. Tags to
`v8.0.1`. Nine `build-*` directories sit in the working tree (ignored, but
clutter). The checkout lives in a OneDrive-synced folder with `.git`
excluded from sync.

## 2. What "professional grade" will mean, concretely

The release is done when all of these hold and CI proves the ones it can:

1. A clean clone builds on Linux (GCC 13, Clang 17), CPU-only and CUDA, and
   on Windows (MSVC 2022), with **zero warnings** under the project's flags.
2. `cmake --install` produces a package that `find_package(OpenNN 9.0)`
   consumes from a separate project; versioned SONAME; SemVer.
3. Every test passes on CI for CPU, and on a CUDA runner nightly; the CPU
   suite is clean under ASan/UBSan; the CUDA subset under compute-sanitizer.
4. Public API documented and published; a user guide; a changelog; the
   environment-variable reference is short and complete.
5. Every example runs from a clean clone, data fetched by a script with
   checksums (or LFS), no example larger than a few MB in git.
6. A clone is under 100 MB; `master` is protected and only receives release
   merges from `dev`; stale branches are gone.

## 3. Phases

### Phase 0 — Reconcile and freeze (1 day)
- Diff the 64 `master`-only commits against `dev`; merge what is not
  superseded, then merge `master` into `dev` so the histories join.
- Decide the version: the configuration API, layer naming and precision
  model changed since 8.0.1 — this is **9.0.0**.
- Freeze feature work on `dev`; everything below is release engineering.
- Exit: `git log dev..master` is empty; `RELEASE_PLAN.md` agreed.

### Phase 1 — Repository hygiene (2–3 days)
- Move example data out of git: a `examples/prepare_data.py` that downloads
  archives from a release asset (or Git LFS) with checksums; keep only tiny
  CSVs in-tree. Decide whether to rewrite history to drop the 10k BMPs
  (`git filter-repo`; 647 MB → ~50 MB; every clone re-fetches) or only stop
  adding — recommendation: rewrite once, now, before the 9.0 tag.
- Delete the seven stale remote branches after confirming nothing unmerged.
- Add `.clang-format`, `.editorconfig`, and format the tree in one commit;
  add `.clang-tidy` with a modest baseline (modernize, bugprone, readability
  identifiers off).
- `.gitignore` for `build-*`, `__pycache__`, scratch; remove the local
  build directories from the working tree.
- Exit: clone under 100 MB; `git status` clean after a build; formatting CI
  check green.

### Phase 2 — Build and packaging (3–5 days)
- Make dependency choices explicit options with graceful fallback:
  `OpenNN_WITH_CUDA`, `OpenNN_WITH_MKL`, `OpenNN_WITH_ONEDNN`, `OpenNN_WITH_TBB`,
  `OpenNN_BUILD_SHARED`; an Eigen-only CPU build must configure and pass the
  CPU tests.
- Pin every FetchContent by tag and hash; write `THIRD_PARTY_NOTICES.md`
  (Eigen, libjpeg-turbo, cudnn_frontend, nlohmann json, CUTLASS, the
  flash-attention shim, oneDNN, MKL redistribution terms).
- Export macros (`OPENNN_EXPORT`) and default hidden visibility for the
  shared build; SOVERSION 9.
- Fix the README requirement (3.24), and add an `install` + `find_package`
  smoke project under `tests/package/` that CI builds against the installed
  tree.
- Optional: CPack archives; vcpkg/conan manifest.
- Exit: install/find_package smoke green on Linux and Windows.

### Phase 3 — Code quality (1–2 weeks)
- Zero warnings: the `-Wswitch-enum` cases in `device_backend.cpp`, the
  `[[nodiscard]]` frontend calls, then `-Werror` in CI.
- Split the eight files over 2,000 lines along their existing seams:
  `device_backend.cpp` → backend/streams, cuBLASLt policy, cuDNN plan
  cache; `network.cpp` → parameter storage vs propagation vs
  serialisation; `tabular_dataset.cpp` → reading vs statistics vs
  splitting; `optimizer.cpp` → per-optimizer files.
- Logging: one `opennn::log` facility (levels, off by default, sink
  redirectable) replacing the 130 stream writes; library code never prints
  unless asked.
- Environment variables: classify the 75 into (a) **supported settings**
  exposed through `Configuration`/`device::` setters with documented
  `OPENNN_*` fallbacks — about 15 (threads, BLAS, TF32, CUDA graphs, conv
  and attention autotune, plan-cache directory, profile, memory debug);
  (b) **tuning policy** with measured defaults kept but documented as
  advanced (tile tolerance, cross-source gain, energy tolerance and window,
  candidate limits); (c) **experiments already decided** — removed, with
  the decision recorded in the commit (the RNN layout switches, save
  transaction v1, narrow-K variants, GEMM mode knobs that lost).
- Header hygiene: no `using namespace std` at global scope in headers.
  Measured: scoping the two directives in `opennn_types.h` to
  `namespace opennn` produces 425 errors in that header and dozens in
  `statistics.h`, `dataset.h` and `tensor_operations.h`, all from templates
  and aliases declared at global scope that name `std`, `Eigen::Index` and
  `span` unqualified -- so this is a sweep of every header's global-scope
  code (move it into the namespace or qualify it), a day or two, not a
  two-line change. `pch.h` stays an internal convenience, not the public
  include; a public umbrella `opennn/opennn.h`.
- Sanitizers: ASan/UBSan job on the CPU suite; compute-sanitizer on a CUDA
  subset; fix what they find.
- Exit: `-Werror` build green; clang-tidy baseline green; sanitizer jobs
  green; no file over 2,000 lines without a written reason.

### Phase 4 — Tests and CI (1 week)
- GitHub Actions: Linux GCC 13 and Clang 17 (CPU: build, tests, header
  check, install smoke, fast examples); Windows MSVC 2022 (build, CPU
  tests); formatting and tidy checks; a nightly job on a self-hosted CUDA
  runner (the RTX 5070 Ti machine) for the CUDA suite and the examples that
  need a GPU.
- Split the CUDA tests into their own binary so a CPU job reports 0 skips
  rather than 77.
- Coverage (llvm-cov) with a floor per module; add `tests/models`; retire or
  fix the one `DISABLED_` test.
- Exit: badges green on `dev`; a PR cannot merge red.

### Phase 5 — Documentation (1 week, parallel with 3–4)
- API reference: Doxygen over the 107 public headers, user-facing classes
  first (datasets, `Network`, layers, `TrainingStrategy` and
  optimizers, `TestingAnalysis`, model selection, `Configuration`,
  `ModelExpression`); published with the docs site.
- User guide: install; data → model → training → testing → export;
  CPU/CUDA and FP32/BF16/INT8 settings; the environment-variable reference;
  a performance chapter drawn from `benchmarks/reports/`.
- `CHANGELOG.md` for 8.0.1 → 9.0.0 (configuration API, graph capture, memory
  attribution, energy-aware autotune, export); `CONTRIBUTING.md`; README
  rewritten with a CI-verified quick start and the benchmark summary table
  linking to the reports.
- Exit: every public class has a reference page; the quick start runs as
  written in CI.

### Phase 6 — Examples (3 days)
- Every example runs from a clean clone after `prepare_data.py`; `yolo`
  ships a tiny sample dataset; `qwen3`, `gpt2`, `bert` download models with
  progress, resume and a documented cache directory, and accept a prepared
  model path.
- The fast examples run in CI; the GPU ones nightly.
- Exit: the example matrix (CPU FP32, CUDA FP32/BF16/INT8) has a recorded
  result for every cell.

### Phase 7 — Release (2 days)
- Version 9.0.0, SONAME 9, CHANGELOG final, release notes.
- Final matrix: CPU and CUDA suites, examples, package smoke on all three
  compilers. The benchmark numbers stand at the commits that measured them
  and are not re-run for the release.
- `dev` → `master` by pull request; tag `v9.0.0`; GitHub release with the
  notes and, optionally, prebuilt archives; protect `master`.
- Exit: `master` = `v9.0.0`; `dev` reopened for features.

## 4. Effort and order

About five to six weeks for one engineer; three to four with two, since
Phases 4–5 run alongside 3. Phases 0–2 first (they change the shape of the
repository and are cheap to do before the code work), then 3 with 4 and 5
in parallel, then 6 and 7.

## 5. Quick wins available immediately

- Fix the README's CMake version, add `THIRD_PARTY_NOTICES.md`, add
  `.gitignore` entries, delete the local `build-*` directories: an hour.
- The zero-warning build: a day.
- A first GitHub Actions workflow (Linux CPU build + tests + header check):
  a day, and it turns every later step into something CI verifies.


## Appendix A. The 64 commits on `master` that `dev` does not have

For Phase 0. `dev` rewrote most of these areas since; each row is a
candidate to cherry-pick, confirm superseded, or drop. 528 of the files
they touch are `.cache/clangd/*` (editor cache committed by mistake), which
is reason enough to finish the reconciliation with `git merge -s ours
master` on `dev` once the rows are decided, so that `master` can be
fast-forwarded from `dev` afterwards.

| commit | date | subject | files | areas |
|---|---|---|---|---|
| `efd566b38` | 2026-08-11 | CUDA fixes | 2 | .gitignore, cuda.pri |
| `13d826be8` | 2026-07-07 | clean | 2 | blank |
| `03e2bb204` | 2026-07-07 | clean and ctest | 7 | CMakeLists.txt, blank, tests |
| `fb7ca4b81` | 2026-07-07 | Merge pull request #342 from arkadesOrg/hotfix-tests | 0 |  |
| `0c43b1b25` | 2026-07-03 | pch.h: re-enable OPENNN_CUDA when WITH_CUDA is defined | 1 | opennn |
| `9c2f2ed67` | 2026-07-02 | remove 6th assert (out of 5), as it causes segfault | 1 | tests |
| `fdd0446c7` | 2026-07-02 | adapt expected layers number result to 5 | 1 | tests |
| `aa5fe9a8a` | 2026-07-02 | adapt expected mutations result | 1 | tests |
| `6eba95493` | 2026-07-01 | OpenNN v8.0.1 | 5 | examples/forecasting, opennn |
| `f0f000be9` | 2026-07-01 | pch.h: disable OPENNN_CUDA for the macOS CPU-only build | 1 | opennn |
| `8423473d4` | 2026-06-26 | ImageDataset::read_bmp: replace per-image progress bar with a single "Loading images..." message | 1 | opennn |
| `097a7f508` | 2026-06-24 | Left-align categorical combo boxes in the exported JavaScript form | 1 | opennn |
| `58a7b4234` | 2026-06-23 | Fall back to CPU when no usable CUDA device is present | 3 | opennn |
| `eb31bcb1c` | 2026-06-18 | ImageDataset: honor the display flag in read_bmp | 1 | opennn |
| `cfb8dc1c3` | 2026-06-18 | LanguageDataset: classification-target labels + cap input sequence length | 2 | opennn |
| `e6e60ad9d` | 2026-06-11 | Conv fixed LR, text-classification encoding, JS export sanitizing | 4 | opennn |
| `f8b2250d2` | 2026-06-10 | Growing neurons: score candidates by best-epoch validation error | 1 | opennn |
| `423857101` | 2026-06-03 | fix | 5 | blank_cuda, opennn |
| `5d785dc30` | 2026-06-03 | fix | 0 |  |
| `d729bbd7e` | 2026-06-02 | merge | 1 | opennn |
| `fcfb23285` | 2026-06-02 | image scaling fix | 2 | opennn |
| `f90854369` | 2026-05-29 | Fixes | 2 | opennn |
| `b3a3a3a7d` | 2026-05-27 | Fixes correlations(Spearman), Genetic algorithm | 4 | opennn |
| `8c705585f` | 2026-05-25 | Fix parameters_bin | 1 | opennn |
| `c7c0fec6a` | 2026-05-22 | Fix genetic algorithm: proper fitness ranking, scaling restoration per individual, real elite preservation | 6 | examples/forecasting, opennn |
| `65d9cad1b` | 2026-05-21 | Publish opennn C++ code rules via CLAUDE.md | 2 | .gitignore, CLAUDE.md |
| `aebfb7b89` | 2026-05-19 | fix scaling layer | 1 | opennn |
| `0ab385649` | 2026-05-18 | UI report uses real variable names; native file pickers; reset sample labels on model switch; Refactor ImageClassification .ndm | 16 | cuda.pri, opennn |
| `9149a5552` | 2026-05-13 | Cleaning-blank | 6 | blank |
| `43132f445` | 2026-05-12 | Fix response_optimization | 3 | opennn |
| `7ea2edb7a` | 2026-05-11 | Bounding layer change | 2 | opennn |
| `b425ec37c` | 2026-05-06 | fix forecasting errors | 6 | opennn |
| `a578b2c1f` | 2026-05-06 | modify example no2 forecasting | 1 | examples/n2o_forecast |
| `c5f748d65` | 2026-05-05 | fix timestampt error | 1 | opennn |
| `d9c4d465f` | 2026-05-04 | fix forecasting LM and SGD | 11 | opennn |
| `7b0fefe25` | 2026-04-30 | LM, SGD forecasting | 4 | opennn |
| `625706c66` | 2026-04-29 | delete wrong example | 7 | .gitignore, examples/kohle_eaf |
| `f315a0c36` | 2026-04-29 | Prevent ResponseOptimization crashes empty inputs | 3 | opennn |
| `172cd9977` | 2026-04-28 | Affine formaula constraints and examples | 31 | examples, examples/kohle_eaf, examples/n2o_forecast, examples/wwt_optimization, opennn … |
| `b44d3b39d` | 2026-04-28 | Clean Windows | 1 | opennn |
| `7aa38090b` | 2026-04-28 | Clean & Exceptions | 5 | opennn |
| `f8acd016b` | 2026-04-28 | final forecasting fixes | 10 | opennn |
| `303b5de0b` | 2026-04-21 | forecasting fixes | 4 | examples/forecasting, opennn |
| `c7450567d` | 2026-04-16 | fix forecasting networks | 5 | examples/forecasting, opennn |
| `0f8a68242` | 2026-04-14 | Clean | 2 | opennn |
| `597407d80` | 2026-04-14 | TextClasificationModel Fixes | 3 | opennn |
| `f6cdcdd3f` | 2026-04-13 | adding example and cleaning | 15 | .gitignore, blank, examples/kohle_eaf, examples/wwt_optimization, opennn |
| `1de2dd465` | 2026-04-10 | Text Class Network fix & Binary Parameters on load/save | 3 | opennn |
| `6756716c1` | 2026-04-09 | Binary Parameters (X_Params.bin, X_Data.bin) | 4 | opennn |
| `0e6b08a81` | 2026-04-07 | fixing little logic response_optimization | 8 | .gitignore, examples/wwt_optimization, opennn |
| `52a99c1eb` | 2026-04-07 | OpenNN 8.0.0 RC-2 | 11 | blank, examples/melanoma_cancer, opennn, tests |
| `4a97e5934` | 2026-04-07 | merge | 2 | opennn |
| `f8a204c42` | 2026-04-07 | Merge branch 'dev' of https://github.com/Artelnics/opennn into dev | 8 | opennn |
| `d5d4514df` | 2026-04-07 | OpenNN 8.0.0 RC | 60 | blank, blank_cuda, examples/airfoil_self_noise, examples/amazon_reviews, examples/breast_cancer … |
| `2f8a0cf0e` | 2026-04-07 | Update CNN | 15 | examples/mnist, opennn |
| `ade7cc748` | 2026-04-06 | Traslation example and clean | 555 (528 .cache) | CMakeLists.txt, blank_cuda, examples/concrete, examples/translation, examples/wwt_optimization … |
| `9d88c6da8` | 2026-04-01 | Dense clean | 1 | opennn |
| `219167780` | 2026-04-01 | Transformers CUDA optimizations | 11 | opennn, tests |
| `229195427` | 2026-03-31 | merge | 2 | opennn |
| `917ab5d16` | 2026-03-31 | Merge branch 'dev' of https://github.com/Artelnics/opennn into dev | 2 | opennn |
| `30ff7b865` | 2026-03-31 | Transformers CUDA | 11 | blank_cuda, opennn, tests |
| `d9271c165` | 2026-03-30 | merge | 2 | opennn |
| `155a52774` | 2026-03-30 | Clean response optimization | 3 | opennn |
| `3c7958258` | 2026-03-30 | updating response optimization forecasting | 22 | examples, examples/concrete, examples/wwt_optimization, opennn |
