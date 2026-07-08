---
name: run-examples
description: Run an OpenNN example category across CPU / GPU FP32 / GPU BF16, restore each example file afterwards, and report a pass/fail table with key metrics. Accepts an optional `basic`/`image`/`text` argument to filter — omit for all categories. Use when the user wants end-to-end verification that nothing regressed after a refactor.
---

# Run example matrix across CPU / GPU FP32 / GPU BF16

Build dir: pick whichever exists — commonly `build/` (Unix Makefiles/Release, single-config; the `cmake --build .` commands below take no `--config` flag) or a Ninja `build-ninja/`; if it's a multi-config Visual Studio solution instead, add `--config Release` to every build command. Each example's `main.cpp` lives in `examples/<name>/main.cpp`; the resulting binary lands in `<build>/bin/<name>`.

### Environment prerequisites (check these FIRST — they cost a full failed build to discover otherwise)

- **CUDA arch must be sm_80+.** The packed-bf16 kernels (`activation_*_kernel_bf162`) call `__bfloat1622float2`, which the CUDA headers only declare under `__CUDA_ARCH__ >= 800`. If the CMake cache has `CMAKE_CUDA_ARCHITECTURES=52` (CMake's fallback when the GPU isn't visible at configure time), the CUDA build fails with `identifier "__bfloat1622float2" is undefined` in `kernel_layers.cu`. Fix once: `cmake -DCMAKE_CUDA_ARCHITECTURES=89 .` from the build dir (89 = RTX 4070/Ada; use 80/86/etc. for your GPU), then rebuild. Check with `grep CUDA_ARCHITECTURES <build>/CMakeCache.txt`.
- **WSL GPU visibility.** On WSL the CUDA driver stub lives in `/usr/lib/wsl/lib`; if it's not on the loader path every GPU run dies. Ensure `export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH` is in `~/.bashrc` (above any interactivity guard) and that runs happen through a login shell (`wsl bash -lc "..."`).

## Argument

The skill accepts one optional argument: `basic`, `image`, or `text`. With no argument, run all three categories.

| Arg | Examples |
|---|---|
| `basic` | airfoil_self_noise, iris_plant, breast_cancer |
| `image` | mnist, melanoma_cancer |
| `text`  | amazon_reviews, emotion_analysis, translation |

`blank_cuda` is explicitly NOT in this skill — it's a benchmark, run it manually when measuring performance.

## The matrix

| Example | CPU | GPU FP32 | GPU BF16 | Category | Notes |
|---|---|---|---|---|---|
| airfoil_self_noise | ✓ | ✓ | — | basic | CSV (TabularDataset), regression |
| breast_cancer | ✓ | ✓ | — | basic | CSV, binary classification |
| iris_plant | ✓ | ✓ | — | basic | CSV, multi-class |
| amazon_reviews | ✓ | ✓ | ✓ | text | LanguageDataset, default is BF16 train + FP32 inference |
| emotion_analysis | **skip** | ✓ | ✓ | text | LanguageDataset; CPU too slow — feedback_dont_test_emotion_cpu |
| translation | **skip** | ✓ | ✓ | text | LanguageDataset + Transformer; CPU too slow |
| mnist | ✓ | ✓ | ✓ | image | ImageDataset, default is BF16 both |
| melanoma_cancer | **skip** | ✓ | ✓ | image | ImageDataset; CPU too slow — feedback_dont_test_melanoma_cpu |

The CSV examples have no BF16 column because they use Type::FP32 by default — no point varying.

## How each run works

Default `Configuration::instance().set(...)` line lives in each example's `main.cpp`. The current API is two-arg — `set(Device, Type)` (no separate inference type; there used to be a third `inference_type` arg, now gone). `Device` is `Auto`/`CPU`/`CUDA`, `Type` is `Auto`/`FP32`/`BF16`. `Auto` resolves at runtime (CUDA+BF16 if a GPU is present). For each (example, mode) pair:

1. **Edit** the Configuration line in-place with the right `(Device, Type)` pair.
2. **Build** that target only: `cmake --build . --target <example> -j2` from `build/`.
3. **Run** from `build/bin/`. Use a generous timeout: 60s for CSV, 300s for text, 600s for image/transformer.
4. **Capture** the key metric (see "Key metric" table).
5. **Restore** the original Configuration line. After all runs of an example are done, run a final `cmake --build` so the file matches its on-disk default again.

Edit/restore are idempotent text swaps — do them with Edit (not sed) because the line wording differs per example.

### Default Configuration lines (restore to these)

| Example | Default line |
|---|---|
| airfoil_self_noise | `Configuration::instance().set(Device::CUDA, Type::FP32);` |
| breast_cancer | `Configuration::instance().set(Device::CPU, Type::FP32);` |
| iris_plant | `Configuration::instance().set(Device::CPU, Type::FP32);` |
| amazon_reviews | `Configuration::instance().set(Device::Auto, Type::Auto);` |
| emotion_analysis | `Configuration::instance().set(Device::Auto, Type::FP32);` |
| translation | `Configuration::instance().set(Device::Auto, Type::FP32);` |
| mnist | `Configuration::instance().set(Device::Auto, Type::Auto);` |
| melanoma_cancer | `Configuration::instance().set(Device::Auto, Type::Auto);` |

The defaults differ per example (some ship as `CPU`, some `CUDA`, some `Auto`) — grep the real line before editing rather than trusting this table; it drifts. Restore to whatever was actually there.

### Mode lines

For each mode, substitute the FULL line (same for every example now that there's no inference-type arg):

- **CPU**: `Configuration::instance().set(Device::CPU, Type::FP32);`
- **GPU FP32**: `Configuration::instance().set(Device::CUDA, Type::FP32);`
- **GPU BF16**: `Configuration::instance().set(Device::CUDA, Type::BF16);`

### Key metric to capture per example

| Example | Capture |
|---|---|
| airfoil_self_noise | `Determination: <value>` (R²) |
| breast_cancer | `Classification accuracy : <value>` |
| iris_plant | The 4×4 "Confusion matrix:" block + `Class probabilities:` line |
| amazon_reviews | Last `Validation error: <value>` + 3×3 confusion + `Prediction for ...:` |
| emotion_analysis | Last 7×7 `Confusion matrix:` + `Prediction for ...:` |
| translation | The four `Source/Predicted` blocks (look for `Sample 0..3`) |
| mnist | Last `Validation error:` + 11×11 confusion |
| melanoma_cancer | `Classification accuracy : <value>` + 3×3 confusion |

The CSV examples are fast (~5-30 s); text takes 1-5 min per run; image runs are 1-10 min. Don't tolerate timeouts — if a run times out, mark the cell ❌ in the report and continue.

## Concurrency caveat

The build uses serial LTO link. Don't try to compile two examples in parallel; the link will serialize and you waste time. One example at a time, `-j2` is enough.

## Final report format

Table with one row per example, one column per mode. Cells use the captured metric + ✓ / ❌:

```text
| Example              | CPU                  | GPU FP32             | GPU BF16            |
|----------------------|----------------------|----------------------|---------------------|
| airfoil_self_noise   | R²=0.75 ✓            | R²=0.60 ✓            | —                   |
| breast_cancer        | acc=0.971 ✓          | acc=0.971 ✓          | —                   |
| iris_plant           | diag confusion ✓     | diag confusion ✓     | —                   |
| amazon_reviews       | val=0.428 ✓          | val=0.427 ✓          | val=0.427 ✓         |
| emotion_analysis     | (skipped)            | acc≈0.88 ✓           | acc≈0.88 ✓          |
| translation          | (skipped)            | 4/4 traduce ✓        | 4/4 traduce ✓       |
| mnist                | acc≈0.81 ✓           | acc≈0.65 ✓           | acc≈0.68 ✓          |
| melanoma_cancer      | (skipped)            | acc=0.75 ✓           | acc=0.75 ✓          |
```

After the table, a "Failures and notes" section listing any cell marked ❌, the wall-clock per example, and anything that looked off (NaN, divergent loss, segfault during testing analysis).

## Practical tips

- **Don't blow away binary caches between modes**: `images.bin` / `tokens.bin` are precision-independent. Reusing them across CPU/GPU FP32/BF16 saves the cache-build time. Only delete if you suspect a cache is stale.
- **Restore on failure**: if a build or run fails mid-sweep, still restore the example's Configuration line before bailing. Otherwise the user is left with a broken example file.
- **Counter-checks for refactors**: if every CSV stays identical (deterministic seed), but image/text shift slightly, that's expected — different paths and OMP thread counts give different RNG sequences. The CSV deterministic match is the strongest signal that the dataset/optimizer plumbing didn't regress.
- **Inference path quirks for image examples**: `TestingAnalysis::calculate_confusion` requires the binary cache to have valid categories (placeholder names ok) — if you ever see a segfault right after training in mnist/melanoma, suspect the cache-hit path of ImageDataset.
- **Shared `build/data` pollution (bites melanoma after mnist)**: both image examples read `ImageDataset("../data")` = `<build>/data`, and `ImageDataset` treats **every** subdirectory there as a class folder. Two ways this goes wrong: (1) mnist's `zero`..`nine` folders stay behind and get merged with melanoma's `benign`/`malignant`; (2) the text examples' `*.txt.cache` directories (`amazon_cells_labelled.txt.cache`, `emotion_analysis.txt.cache`, `ES-EN-small.txt.cache`) are ALSO scanned as (empty) image classes. Either way melanoma runs as a 4-6-class problem, its `print_binary_classification_tests()` sees a non-2×2 confusion, and every binary metric (accuracy/sensitivity/specificity) prints **0** even though training converged and the real 2-class accuracy is ~0.8. Before each image run: wipe `<build>/data/.cache/images.bin` (the dot-`.cache` dir is itself skipped by the scan, but the stale `images.bin` inside would be loaded verbatim) AND ensure only the intended class folders are present — move the digit folders and the `*.txt.cache` dirs out of `<build>/data` for the melanoma run, then move them back. Symptom to recognize: binary metrics all `0` with a confusion matrix wider than 2×2, populated only at the alphabetical indices of `benign`/`malignant`.
- **mnist GPU BF16 can fail in the conv path at large batch**: the packed-bf16 convolution requests a cuDNN-frontend workspace sized by batch; at mnist's `batch_size 2020` it fails with `ConvolutionOperator: cudnn-frontend path unavailable (CUDA Error: 2 ...device_backend.cpp:51)` then a sticky error at `:297`, even with GPU memory free. mnist GPU FP32 and melanoma GPU BF16 (batch 10) are unaffected — so this reads as a large-batch BF16 conv workspace bug, not a config or OOM problem. Mark the mnist GPU BF16 cell ❌ and keep going.

## When to stop the sweep early

Stop and report immediately if:
- A CSV example fails (means basic plumbing broke).
- The same example fails in two different modes (means a layer/operator issue, not a config issue).
- A build error appears that's not a typo from your own edit — could indicate the refactor didn't compile cleanly.

Otherwise, push through all cells; partial table + flagged cells is more useful than nothing.
