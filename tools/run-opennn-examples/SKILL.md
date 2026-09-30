---
name: run-opennn-examples
description: Build and run every CMake-registered OpenNN example across the explicit CPU and CUDA precision matrix, preserving the working tree and reporting every pass, failure, unsupported cell and missing prerequisite. Use for full end-to-end example validation, not for unit tests or benchmarks.
---

# Run the OpenNN example matrix

Exercise every target registered through `opennn_example(...)` in
`examples/CMakeLists.txt`. Treat that file as the source of truth; do not keep a
separate hard-coded target list in this skill.

## Matrix

Use the current two-argument configuration API from
`opennn/core/configuration.h`:

| Cell | Configuration |
| --- | --- |
| CPU FP32 | `Configuration::instance().set(Device::CPU, Type::FP32);` |
| CUDA FP32 | `Configuration::instance().set(Device::CUDA, Type::FP32);` |
| CUDA BF16 | `Configuration::instance().set(Device::CUDA, Type::BF16);` |
| CUDA INT8 | `Configuration::instance().set(Device::CUDA, Type::INT8);` |

Do not pass a third inference-precision argument; it no longer exists. Do not
add `Auto` rows to the comparison: `Device::Auto` and `Type::Auto` resolve to
one of the explicit cells according to available hardware. CPU with BF16 or
INT8 is invalid by design. CUDA BF16 and INT8 require compute capability 8.0 or
newer.

## Prepare

1. Read `AGENTS.md`, `README.md`, `examples/README.md`, `examples/CMakeLists.txt`
   and the current `Configuration` declaration and implementation. The code
   overrides this document if the API changes again.
2. Record `git status --short`. Never overwrite or restore changes that were
   already present.
3. Discover the example targets from CMake and inspect each entry point for
   required arguments, model downloads, interactive input and an existing
   `Configuration::instance().set(...)` call. Resolve current default filenames,
   labels and sample counts from the entry point and catalog;
   do not reuse filenames or expected counts from an earlier dataset.
4. Use separate Release build directories outside the checkout for CPU and
   CUDA. Configure with examples enabled and benchmarks/tests disabled unless
   the user requested otherwise.
5. Confirm required datasets and model assets before starting a long cell.
   Missing data or model assets is `BLOCKED`, not a code failure.

## Apply a matrix cell

Set the configuration before constructing a dataset, network, model or session.
For an example that already calls `Configuration::set`, temporarily replace
that complete call. For an example without a call, temporarily add the direct
`opennn/core/configuration.h` include and one call at the start of its `try`
block. The `blank` target is an empty template that only prints two lines; do
not edit it, and report it as `N/A` in every cell.

Before editing, save the exact original contents outside the repository. Restore
them in a `finally`-style cleanup path after every example, including build
failure, timeout or interruption. Verify restoration by comparing the file with
the saved original; do not use `git checkout`, `git restore` or another command
that could discard the user's changes.

Build only the target being exercised. Run it from the build's runtime-output
directory so paths such as `../data/<example>` resolve to the data copied or unpacked by
CMake. Use the current bundled data and check the loaded sample count and
classes against the catalog when the example prints them. Supply deterministic,
minimal input to interactive programs; for Qwen,
pipe a prompt followed by `exit` and use a user-provided/prepared model
directory. Apply a finite timeout suited to the workload.

Use supported command-line flags for focused checks. `emotion_analysis` accepts
`--device cpu|cuda|auto`, `--seed N`, `--epochs E` and `--data FILE`; verify these
against its current entry point. `--epochs 1` checks loading, training and
evaluation quickly, but report it as a one-epoch smoke run. Its CLI selects FP32;
the CUDA BF16 and INT8 cells still require the temporary precision edit above.

`electrocardiogram_anomaly_detection` takes no arguments and trains for 20
epochs. Use its bundled waveform CSV and testing indices from the current
entry point. A successful run prints the reconstruction threshold, a confusion
matrix, binary classification metrics and ROC AUC, then exits with code zero.

`melanoma_cancer` takes no arguments and loads 102 RGB images of 300 x 300
from `../data/melanoma_cancer/isic2016`, in the `benign` and `malignant` class
folders. Check that the loader sees 102 samples and two classes, then that
training and binary evaluation finish with finite metrics. For a focused smoke
run, a temporary `set_maximum_epochs(1)` on its optimizer limits training;
save and restore the entry point as above, and report the one-epoch limit.

`translation` takes no arguments, loads 1,000 Spanish-English pairs from
`../data/translation/tatoeba_es_en.tsv`, and trains for 50 epochs in batches
of 16 before translating `Vete.`. Autoregressive generation requires
CUDA; a CPU check limited to loading or training does not pass the complete
example. For a focused CUDA smoke run, temporarily limit the optimizer to one
epoch, restore the entry point afterwards, and report that limit. Require
finite training results, a non-empty generated translation and exit code zero.

`bert` accepts optional review-file and model-cache arguments, defaulting to
`../data/bert/amazon_cells_labelled.txt` and `../data/bert`. It loads 1,000
reviews with `Bad`/`Good` labels and a WordPiece vocabulary of 30,522 tokens,
uses sequences of length 64, and trains for three epochs in batches of 32.
The first run downloads about 437 MB from `Artelnics/bert-base-uncased-opennn`
on Hugging Face; record download time separately from
execution with a populated cache. Require finite training and binary
classification metrics, the final `Good bye!` message and exit code zero.
For a focused smoke run, temporarily set one epoch, restore the entry point
afterwards, and report that limit.

`gpt2` accepts one optional quoted prompt argument, defaults to
`Artificial intelligence`, and generates at most 40 new tokens. Its model
cache is `../data/gpt2`; the first run downloads about 650 MB from
`Artelnics/gpt2-small-opennn` on Hugging Face. Autoregressive
generation requires CUDA. Require a non-empty continuation after the
`GPT-2 generated text:` label and exit code zero; record the prompt and any
download time separately from a run using the populated cache.

`qwen3` accepts one optional model-directory argument and selects the 4B
model. Its default cache is `../data/qwen3`; pass an existing cache explicitly
to reuse it. The initial BF16 weight download is about 8.83 GB, plus tokenizer
files. Downloading a 0.6B cache does not change the example's selected model.
Pipe a short prompt such as `Say hello in one short sentence. /no_think`
followed by `exit`. Require a non-empty `Response:` and exit code zero;
opening the chat and immediately exiting checks loading only. Record the
actual resolved device/precision and separate download time from execution.
The CUDA binary loader explicitly requires BF16 or INT8; report a rejected
CUDA FP32 cell as `UNSUPPORTED`, retaining the diagnostic. Do not substitute
a BF16 run for a CUDA FP32 result.

`yolo [experiment]` currently selects `v3-pretrained` when called without
arguments. It needs an external synthetic dataset at `SYNTHETIC_YOLO_IMAGES`
and `SYNTHETIC_YOLO_LABELS`, falling back to `synthetic_yolo/images` and
`synthetic_yolo/labels`. It expects 416 x 416 images and detection labels;
record the actual loaded sample count and classes. The internal coloured-block
generator is not called by this entry point. A missing dataset is `BLOCKED`.
`v3-scratch` skips pretrained-backbone loading, but an existing experiment
checkpoint can still be resumed; run in a new output directory when checking
training from scratch. A missing pretrained file causes scratch initialization,
so record that fallback instead of counting it as a pretrained run.

An experiment name containing `voc` selects `VOC_ROOT` and optional
`VOC12_ROOT`; one containing `bccd` selects `BCCD_IMAGES` and `BCCD_LABELS`.
The `use_coco` and `use_raccoon` source switches are false by default.
`v8-pretrained` can import a supplied `yolov8s.onnx`; `v8-scratch` skips the
pretrained import. There is no `--epochs` CLI option. For a focused training
check, temporarily shorten the actual `lr_schedule` and any applicable
backbone-freeze warmup, restore the source afterwards, and report the limits.
Do not rely on the `quick_test` comment to cap training: inspect its uses.
Require finite training results, saved checkpoints, completed detection/mAP
output and exit code zero; report loading-only or shortened runs separately.

## Classify results

Every target must have a result for every matrix column:

- `PASS`: exit code zero and the expected final output or metric is present.
- `FAIL`: it built and ran with its prerequisites, but compilation, execution,
  numeric validity or output validation failed.
- `UNSUPPORTED`: the code explicitly rejects that device/type combination.
- `BLOCKED`: required hardware, data, model assets or external tooling is
  unavailable.
- `N/A`: the target does not instantiate OpenNN. Currently only `blank`; every
  other example target requires a result for each requested cell.

Do not silently convert failures to unsupported results. Record the command,
exit code, elapsed time and a short diagnostic for every non-pass cell. Capture
the example's meaningful final metric when it has one, and reject NaN/Inf or an
unexpectedly empty result.

## Finish

Restore every temporarily edited source and verify that `git status --short`
differs from the initial snapshot only by artifacts explicitly requested by the
user. Delete temporary Python runners created for the task after using them.
Report a table with one row per CMake example and these columns:

```text
| Example | CPU FP32 | CUDA FP32 | CUDA BF16 | CUDA INT8 |
```

Follow it with prerequisites, failures, unsupported paths, per-example elapsed
time, run arguments and the exact build configuration. A partial matrix is
useful, but never describe it as complete.
