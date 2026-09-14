# OpenNN examples

Start with [minimal inference](blank/main.cpp), then
[Iris training](#train-your-first-model). Build and run one target at a time;
some language and detection examples require large downloads or a CUDA device.

## Choose an example

| Target | What it demonstrates | Inputs and runtime notes |
| --- | --- | --- |
| [`blank`](blank/main.cpp) | A two-input dense network and one prediction | CPU FP32; fixed demo weights; no data or downloads |
| [`iris_plant`](iris_plant/main.cpp) | Classification, evaluation and C/Python export | Bundled small CSV; explicitly CPU FP32 |
| [`airfoil_self_noise`](airfoil_self_noise/main.cpp) | Tabular regression | Bundled CSV |
| [`yacht_hydrodynamics`](yacht_hydrodynamics/main.cpp) | Regression with neuron selection, quasi-Newton training and test-set regression analysis | Bundled CSV; explicitly CPU FP32 with one thread |
| [`breast_cancer`](breast_cancer/main.cpp) | Binary classification | Bundled CSV; explicitly CPU FP32 |
| [`concrete`](concrete/main.cpp) | Constrained response optimization | Bundled saved network; loads from the source tree |
| [`ecg5000_anomaly_detection`](ecg5000_anomaly_detection/main.cpp) | Autoencoder anomaly detection with a reconstruction-error threshold | Bundled ECG CSV and stored test indices; explicitly CPU FP32 |
| [`forecasting_tinyml`](forecasting_tinyml/main.cpp) | RNN/LSTM export parity | Constructs inputs; CPU FP32; optional emulator tools |
| [`amazon_reviews`](amazon_reviews/main.cpp) | Sentiment classification | Bundled labelled text |
| [`emotion_analysis`](emotion_analysis/main.cpp) | Multi-class text classification | Bundled labelled text |
| [`mnist`](mnist/main.cpp) | Image classification | Bundled `data/images.zip`; unpacked automatically |
| [`melanoma_cancer`](melanoma_cancer/main.cpp) | Binary image classification | Bundled `data/images.zip`; unpacked automatically; attribution review pending |
| [`bert`](bert/main.cpp) | Fine-tuning a pretrained text classifier | Bundled SST-2 text; downloads model weights; accepts text and model-directory arguments |
| [`translation`](translation/main.cpp) | Encoder-decoder text translation | Bundled text; explicitly requests CUDA |
| [`gpt2`](gpt2/main.cpp) | Pretrained text generation | Downloads weights; explicitly requests CUDA; optional prompt argument |
| [`qwen3`](qwen3/main.cpp) | Interactive pretrained chat | Downloads Qwen3-4B assets; optional data-directory argument; provision suitable memory/backend |
| [`yolo`](yolo/main.cpp) | Object-detection training experiments | Explicitly requests CUDA; requires separately prepared detection data and backbone weights for pretrained modes |

Targets without an explicit device selection use the configured library backend;
inspect their source before a controlled CPU/GPU comparison. The maintained target
list is in [CMakeLists.txt](CMakeLists.txt). Unsupported 8.x applications are
kept together in [a reference archive](#preserved-8x-examples).

## Folder layout

Each maintained example has a `main.cpp` and, when needed, a `data/` directory.
MNIST and melanoma keep their images in one ZIP per dataset, with the original
class folders and filenames inside. Building either target extracts its ZIP
outside the checkout; no separate download or Python setup is required.

The additional folders serve specific purposes:

| Folder | Purpose |
| --- | --- |
| `concrete/nn/` | Saved network and parameters shared with response-optimization tests |
| `iris_plant/tinyml/` | C/Python export checks and shared PC, AVR and ARM harnesses |
| `legacy_8/` | One historical archive; excluded from current builds |

`CMakeLists.txt` selects targets and `prepare_data.cmake` stages their data.
Both are build support files; users run the commands below.

## Build and run

From the repository root, enable examples in an external build and select a target:

```sh
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=ON
cmake --build ../opennn-build --config Release --target iris_plant --parallel
```

Run examples from their executable directory. Build rules copy or unpack bundled data at
`../data/<example>` relative to that directory:

```sh
cd ../opennn-build/bin
./iris_plant
```

For Visual Studio, run from `../opennn-build/bin/Release` instead. On Windows,
the executable is `iris_plant.exe`. Generated models and exports stay in the
build directory. Replace `iris_plant` with another target from the table;
CUDA-only examples require a CUDA build and a working GPU.

## Train your first model

The [Iris program](iris_plant/main.cpp) shows the complete training flow:

1. Load the tabular CSV into `TabularDataset`.
2. Create a `ClassificationNetwork` matching its inputs and targets.
3. Connect `Training` to the network and dataset, then call `train()`.
4. Use `Evaluation` to print classification results.
5. Export the model and compare predictions with `ModelExpression`.

It writes the saved JSON/binary model, C expression and CEmbedded exports,
a Python model, and reference predictions beside the executable. The
[TinyML procedure](#tinyml-export-parity) checks the exported implementations.

For complete CPU/CUDA and precision coverage, use the
[example matrix skill](../tools/run-opennn-examples/SKILL.md).
[DATASETS.md](../DATASETS.md) records asset sources, reproduction procedures
and unresolved permissions; keep each dataset's `SOURCE.md` notice with its data.

Further examples: [concrete optimization](#concrete-response-optimization),
[ECG anomaly detection](#ecg5000-anomaly-detection),
[TinyML export parity](#tinyml-export-parity), [8.x reference code](#preserved-8x-examples).

## Concrete response optimization

This example uses the UCI Concrete Compressive Strength dataset and a small pretrained OpenNN surrogate from the IDC paper companion material.

Decision variables:

```text
cement, slag, fly_ash, water, sp, coarse_agg, fine_agg, age
```

Response:

```text
strength
```

Objectives:

```text
maximize strength
minimize cement
```

Simplex / mass-balance expression:

```text
cement + slag + fly_ash + water + sp + coarse_agg + fine_agg = 2325.012558
```

With nonnegative ingredient bounds, that equality places the seven ingredient masses on a fixed-density simplex. The example writes it directly as a `ResponseOptimization::add_constraint(...)` expression.

Other affine constraints shown in [`concrete/main.cpp`](concrete/main.cpp):

```text
cement + slag + fly_ash >= 200
water - 0.30 * cement >= 0
water - 0.70 * cement <= 0
slag - 0.70 * (cement + slag + fly_ash) <= 0
fly_ash - 0.40 * (cement + slag + fly_ash) <= 0
age = 28
```

The exact IDC paper, model source and training provenance are still unresolved;
the source attribution above is the historical description, not clearance of
the bundled model. See [the data review](../DATASETS.md).

## ECG5000 anomaly detection

This example trains an autoencoder to recognize normal ECG signals and flags
signals with unusually large reconstruction errors:

- the 1,000 testing signals are the rows listed in
  `ecg5000_anomaly_detection/data/test_indices.csv`; the other 3,998 rows are the
  development set;
- all 140 values of every signal are scaled to [0, 1] with one global minimum and
  maximum taken from the development rows;
- a 140-32-16-8-16-32-140 autoencoder with ReLU hidden layers and a sigmoid output
  is trained only on the normal development signals, minimizing the mean absolute
  error with Adam (learning rate 0.001, batch size 512) for 20 epochs;
- a signal raises an alert when its mean absolute reconstruction error reaches the
  mean plus one population standard deviation of the normal training errors.

The tutorial data is stored in `ecg5000_anomaly_detection/data/ecg.csv` from the tutorial's
[official download](https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv).
Although the dataset is commonly called ECG5000, this prepared CSV contains
4,998 rows. Its SHA-256 checksum is:

```text
72ce7b040ca0c6ed36c3368e570c6ac4ddf20100476e47373c63b2395e012df1
```

The executable prints the threshold and, with anomaly as the positive class, the
test confusion matrix, accuracy, precision, sensitivity, specificity, F1 score and
ROC AUC.

## TinyML export parity

These Linux/WSL pipelines compare exported C with OpenNN reference predictions
on a native PC, an ATmega328P/Arduino Uno in simavr, and optionally an ARM
Cortex-M3 in QEMU (`mps2-an385`). They test inference parity, not training quality.

`ProgrammingLanguage::C` generates unrolled expressions; `CEmbedded` uses
float32 weight tables and generic loops without heap allocation. On AVR,
`PROGMEM` and `pgm_read_*` keep tables in flash. On other targets, `NN_FLASH` and
`NN_READ_*` become ordinary constant reads. Approximate weight storage is
50 bytes per weight for expressions versus 4 for tables; actual totals depend
on the model and compiler.

### Prerequisites and commands

Run these commands from the repository root. `BUILD` selects the external OpenNN
build, and `WORK` can override each script's external output directory:

```sh
cmake -S . -B "$HOME/opennn-build" -DCMAKE_BUILD_TYPE=Release -DOpenNN_DISABLE_CUDA=ON -DOpenNN_BUILD_EXAMPLES=ON
cmake --build "$HOME/opennn-build" --target iris_plant forecasting_tinyml --parallel
arduino-cli core install arduino:avr
```

Install simavr in user space, for example from Ubuntu packages:

```sh
mkdir -p "$HOME/simavr-local"
cd "$HOME/simavr-local"
apt-get download simavr libsimavr2 libelf1t64
for f in *.deb; do dpkg -x "$f" root; done
```

Optional xPack `arm-none-eabi-gcc` and QEMU Arm distributions belong under
`~/arm-tools`. Without them, the ARM stage reports a skip. Return to the
repository root before running:

```sh
BUILD="$HOME/opennn-build" bash examples/iris_plant/tinyml/run_tinyml_test.sh
BUILD="$HOME/opennn-build" bash examples/forecasting_tinyml/run_forecasting_test.sh
```

Use `--skip-training` for Iris or `--skip-generation` for forecasting to reuse
exports. When piping the Iris script into Bash, set `TINYML_DIR` to the absolute
`examples/iris_plant/tinyml` directory first. These scripts are available locally;
the former `tinyml-parity.yml` workflow is not present in the current CI tree.

### Harness and acceptance

The shared harness in [`iris_plant/tinyml/`](iris_plant/tinyml/) contains:

| File | Purpose |
| --- | --- |
| `avr_harness.c`, `pc_harness.c` | Execute the same exported model and emit hex float bits; AVR uses UART0. |
| `make_test_vectors.py` | Convert OpenNN reference CSVs into `test_vectors.h`. |
| `compare_outputs.py` | Compare native, emulator and OpenNN results, tolerating simavr's coloured stderr echo. |
| `check_python_export.py` | Check the Iris Python export with a NumPy shim. |
| `arm/arm_harness.c`, `arm/vectors.c`, `arm/mps2.ld` | Semihosting, vector table and linker script for Cortex-M. |

The model-agnostic harness takes `NN_MODEL_FILE` and `test_vectors.h`.
`OPENNN_EXPORT_NO_MAIN` disables the generated demo entry point. An AVR pass
requires unmodified C to compile with `avr-gcc -mmcu=atmega328p`, fit 32 KB flash
and 2 KB RAM, and match the reference within the default absolute tolerance
`1e-4`. Iris also requires the predicted class to match on every test vector.

### Dated measurements

These July 6, 2026 observations retain their original scope. They were measured
with avr-gcc 7.3.0 and simavr 1.6 under WSL2 Ubuntu 24.04; forecasting ARM checks
used arm-none-eabi-gcc 15.2.1 and QEMU 9.2.4. They are not a fresh validation of
the current release.

Iris used a 4-16-3 classifier with 131 parameters:

| Backend | AVR flash (text + data) | AVR RAM (data + bss) | Maximum absolute difference |
| --- | --- | --- | --- |
| C expression | 6,776 B | 186 B | About 1e-7 |
| CEmbedded tables | 3,450 B | 302 B | About 1e-7 |
| Python | PC only | — | About 1e-7 |

All variants agreed on 9/9 predicted classes. The recorded maximum numerical
error was about `9e-8`, within the configured tolerance.

Forecasting used LSTM and simple recurrent networks with 6 steps, 2 features and
hidden size 6, retaining Glorot initialization. The driver in
[`forecasting_tinyml/main.cpp`](forecasting_tinyml/main.cpp) generates both export
backends and reference vectors; it reuses the Iris harness.

| Model/backend | AVR flash | Cortex-M3 | Maximum absolute difference |
| --- | --- | --- | --- |
| LSTM/expression | 42,538 B; does not fit | Passed | About 2e-7 on PC/ARM |
| LSTM/tables | 4,368 B; passed | Passed | About 2e-7; PC and ARM bit-identical |
| RNN/expression | 12,206 B; passed | Passed | About 5e-7 |
| RNN/tables | 3,434 B; passed | Passed | About 5e-7; PC and ARM bit-identical |

The LSTM expression variant passed only the targets where it fitted; it was not
an AVR pass. Expression size grows with time steps and hidden size squared.
The table variants keep weights in flash and used about 0.6 KB of working RAM.
The float32 PC/ARM table implementations used the same operation order.

## Preserved 8.x examples

The [reference archive](legacy_8/reference.zip) preserves the 12 source, project
and data files recovered during the 9.0 reconciliation. Every member is identical
to [the pre-cleanup revision](https://github.com/Artelnics/opennn/tree/89d185d75fcd04070484b50389f1214ced792f21/examples/legacy_8).
They use the flat 8.x headers and XML-era APIs and are **not supported 9.x targets**.
Keeping them in one archive preserves their unresolved data records without
presenting obsolete project files as runnable examples.

Browse the historical revision online, or extract the archive into a separate
directory with a ZIP tool. No additional Git checkout is needed.

- `n2o_forecast`: action-conditioned wastewater forecasting; bundled CSV
  retained pending verification of its transformations and attribution.
- `wwt_optimization`: requires `WWTP_PO4_NH4_removal.csv`, which was not present
  in the merged master tree. The results file is not a replacement dataset.
- `forecasting`: Madrid NO2 binary artifacts retained for provenance review;
  their original producer and binary layout are not established here.

For maintained code, see [`forecasting_tinyml/`](forecasting_tinyml/), the time-series dataset
tests, and the response-optimization integration scenarios. A port of these
historical applications must explicitly translate window indices and verify
predictions before transferring their optimization settings. See
[the migration guide](../CHANGELOG.md#migrating-from-8x-to-90) and [DATASETS.md](../DATASETS.md).
