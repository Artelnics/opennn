# Migrating from OpenNN 8.x to 9.0

9.0.0 is a major source and model-format migration. Recompile consumers;
replacing an 8.x shared library in place is not supported. The candidate is
unreleased. Neural Designer validation is explicitly deferred at the owner's
request and is not implied by OpenNN's test results.

## Build and headers

Use CMake 3.24 or newer and a C++20 compiler. Install the library and consume
its exported target so that Eigen, OpenMP and the image dependencies follow
the package configuration:

```cmake
find_package(OpenNN 9.0 CONFIG REQUIRED)
target_link_libraries(my_application PRIVATE OpenNN::opennn)
target_compile_features(my_application PRIVATE cxx_std_20)
```

`tools/package_smoke` is an executable example of this contract, checked by
CI after moving the installation prefix. The package version file requires
the same major version: an application requesting 8.x must migrate explicitly.

| Former header | Current header |
| --- | --- |
| `opennn/neural_network.h` | `opennn/network/network.h` |
| `opennn/neural_network/neural_network.h` (earlier 9.0 candidate) | `opennn/network/network.h` |
| `opennn/dataset.h` | `opennn/dataset/dataset.h` and `opennn/dataset/tabular_dataset.h` |
| `opennn/standard_networks.h` | `opennn/models/models.h` |
| `opennn/training_strategy.h` | `opennn/training/training.h` |
| `opennn/training_strategy/training_strategy.h` (earlier 9.0 candidate) | `opennn/training/training.h` |
| `opennn/dense_layer.h` | `opennn/network/layers/dense_layer.h` |
| `opennn/bounding_layer.h` | `opennn/network/layers/clamping_layer.h` |
| `opennn/response_optimization.h` | `opennn/response_optimization/response_optimization.h` |
| `opennn/variable.h` | `opennn/core/variable.h` |

Changing includes alone is insufficient. Use `TabularDataset` for tabular
data and the specialized image, language and time-series dataset classes for
their respective storage contracts. Use `opennn::Dense` and `Clamping` for the
current layer classes. The installed-package smoke program demonstrates adding
layers, compiling a network, and running a row-major `MatrixR` batch.

## Network naming

The public class is now `opennn::Network`, declared in
`opennn/network/network.h`. Replace `NeuralNetwork` in constructors, pointers,
references and derived classes with `Network`. The former
`opennn/neural_network/` directory is now `opennn/network/`, including layer and
operator headers. There is no old-name alias or forwarding header. Rebuild all
consumers because the C++ binary symbols have changed.

Generated Python models also expose `Network`; update callers to
`module.Network()` and regenerate exports. JSON models use the top-level key
`"Network"`. For models from the earlier 9.0 candidate, rename only the previous
top-level `"NeuralNetwork"` key to `"Network"`, retaining the contents and matching
binary file. The old root is rejected. Parameter layouts and binary snapshot
formats are unchanged by this naming change. This root-key edit is not an 8.x
model conversion; follow the procedure below for historical formats.

## Training naming

The training coordinator is now `opennn::Training`, declared in
`opennn/training/training.h`. Replace `TrainingStrategy` with `Training` and
change includes from `opennn/training_strategy/` to `opennn/training/`. Model
selection accessors are now `get_training()` and `set_training()` where
provided. There is no old-name alias or forwarding header; rebuild consumers
for the renamed C++ symbols.

Training configuration JSON uses the top-level key `"Training"`. For an earlier
9.0 configuration, rename its `"TrainingStrategy"` root to `"Training"`. For
Quasi-Newton configurations, also update the optimizer names described below.
The former root is rejected. The
training algorithms, ownership and buffer layouts are unchanged.

Optimizer classes are now `LevenbergMarquardt` and `QuasiNewton`, replacing
`LevenbergMarquardtAlgorithm` and `QuasiNewtonMethod` without aliases. Their
headers are `opennn/training/levenberg_marquardt.h` and
`opennn/training/quasi_newton.h`. Update constructor calls, type references and
includes, then rebuild consumers.

The optimizer factory and JSON name for `QuasiNewtonMethod` is now `QuasiNewton`.
Update both the `OptimizationMethod` value and its nested object key in saved
training configurations; standalone optimizer JSON uses the new root as well.
The old factory name is rejected. The existing `LevenbergMarquardt` factory/JSON
name stays the same.

## Further naming simplifications

The following public C++ names replace the earlier 9.0 candidate names:

| Previous class | Current class | Current header |
| --- | --- | --- |
| `AdaptiveMomentEstimation` | `Adam` | `opennn/training/adam.h` |
| `StochasticGradientDescent` | `SGD` | `opennn/training/sgd.h` |
| `LongShortTermMemory` | `LSTM` | `opennn/network/layers/lstm_layer.h` |
| `AutoencoderNetwork` | `Autoencoder` | `opennn/models/models.h` |
| `YoloNetwork` | `Yolo` | `opennn/models/models.h` |
| `InputsSelection` | `InputSelection` | `opennn/model_selection/input_selection.h` |
| `TestingAnalysis` | `Evaluation` | `opennn/evaluation/evaluation.h` |

Rename the corresponding old header basenames and change the
`opennn/testing_analysis/` include directory to `opennn/evaluation/`.
`LongShortTermMemoryOperator` becomes `LSTMOperator`, and
`LayerType::LongShortTermMemory` becomes `LayerType::LSTM`.
`InputsSelectionResult` becomes `InputSelectionResult`; rename
`get_inputs_selection_name` and `create_inputs_selection`
to their singular `input_selection` spellings. In model-selection JSON, rename
the `InputsSelection` object to `InputSelection` and its `InputsSelectionMethod`
field to `InputSelectionMethod`.

For saved training configurations, replace the optimizer `OptimizationMethod`
value and its nested object key from `AdaptiveMomentEstimation` to `Adam`, or
from `StochasticGradientDescent` to `SGD`. Standalone optimizer JSON uses the
new root names too. In network JSON, rename each LSTM configuration object key
inside `Network.Layers.Items` from `LongShortTermMemory` to `LSTM`; standalone
LSTM layer JSON also uses the new root. Former factory names are rejected.
Keep custom layer labels and their connection references consistent; the
default label for newly constructed LSTM layers is now `lstm_layer`.

There are no old-name aliases or forwarding headers. Rebuild consumers for
the new symbols. These renames do not change tensor layouts, weights, numerical
algorithms or memory ownership. `ModelExpression` keeps its existing name.

## Models and parameters

9.x uses JSON configuration and binary parameter storage. It does not provide
a general 8.x XML/NDM converter. Renaming an XML file to `.json`, or loading an
old raw parameter file into a new topology, is not a migration procedure.

For each production model:

1. Keep the original artifacts and an 8.x environment that can load them.
2. Record the topology, activation functions, feature order, category order,
   scaling/unscaling descriptives, tensor shapes, and output interpretation.
   Export representative inputs and expected outputs from that environment.
3. Reconstruct the model using current APIs. Transfer parameters only after
   checking the layout of each layer; equal parameter counts do not establish
   equal ordering. If the old environment supports an executable expression
   export, keep it as an independent reference for supported architectures.
4. Compare predictions against the recorded cases, including missing values,
   categories, boundary values and forecasting windows. Choose numerical
   tolerances appropriate to the deployment precision.
5. Save with `Network::save` and `save_parameters_binary`, reload using
   the matching current APIs, and repeat prediction comparisons. Keep paired
   architecture and parameter files together.

Current-format round trips and malformed binary rejection are tested. Those
tests do not certify arbitrary historical models. Real 8.x artifact conversion
remains application-specific work; no converter is claimed by this release.

Raw parameter snapshots use the compiled storage layout, including alignment
padding. `get_parameters_number()` reports the logical count;
`get_parameters_buffer_size()` reports the size expected by `set_parameters`.
Prefer the paired model save/load APIs for persistence. Buffer sizes alone
still do not establish that two models use the same parameter ordering.

`Network::load(path)` requires the matching `.bin` file or embedded JSON
parameter values. A missing binary with no embedded weights raises an error
before clearing the existing network. Saved model pairs must be kept together.
Nonempty embedded parameter text must contain exactly the compiled buffer's
number of values, including alignment padding. Too few or too many values now
raise an error before any embedded weights are copied; the previous warning and
partial-copy behavior is no longer accepted. Whitespace-only text has zero values
and is rejected for a model requiring parameters.
Embedded parameter values must also be finite: NaN and positive or negative
infinity now raise an error identifying the zero-based buffer index before any
embedded weights are copied. This check applies to alignment padding as well.
For intentional architecture-only loading, use a new network explicitly:

```cpp
Network network;
network.from_JSON(load_json_file(path));
// Initialize or load matching parameters before using the model for inference.
```

This check adds no inference/training operations or parameter buffers. It does
not provide rollback after every possible error later in model construction or
binary loading.

### Available historical artifacts reviewed

The example assets in tags `v8.0.0`, `v8.0.1` and master `efd566b38` were
inspected during the release review. Their breast-cancer, Iris and MNIST `.bin`
files are dataset caches. `time_series_data_set.xml` describes a dataset, not
a neural-network topology. The Madrid `_Params.bin` has no matching saved
network architecture, feature/scaling contract and reference predictions in
those assets. These files do not form a complete production-model migration case.
The owner confirmed that no complete external 8.x reference model is available
for this review, so production-model migration remains unverified.

The current concrete JSON/binary pair is exercised by response-optimization
integration tests, but that is current-format validation. The independently
trained models in [tools/REPRODUCTION.md](tools/REPRODUCTION.md) likewise test
training, save/reload and exported predictions, rather than 8.x conversion.
To close production migration validation, provide a complete 8.x topology and
weights plus representative inputs and predictions from the original environment.
Neural Designer remains outside this review.

## Time-series indexing

The reconciled master implementation treated sample `i` as a present instant:
inputs were rows `[i-past, i-1]`, and targets began at `i+1`, leaving row `i`
out of both. The 9.x implementation treats the sample index as a window start:
inputs are `[i, i+past-1]`; targets begin at `i+past`. With a single target
window, the selected target is `i+past+future-1`; multi-target windows include
all future steps. This is a behavior change, not just an index rename.

Regenerate chronological splits and compare windows explicitly when migrating
forecasting applications. Do not reuse old sample indices or blindly copy
master's action-conditioned optimization constraints. The legacy wastewater
examples are preserved under `examples/legacy_8` and are not 9.x targets.

## Training, selection and device configuration

The optimizer prepares scaling artifacts for each training run. Input
selection changes the active feature set and recompiles the model. Retain
explicit optimizer settings: the old convolution-specific Adam learning-rate
override is not imposed on the current optimizer.

New genetic algorithms default to `Random` initialization, carrying master's
bounded Bernoulli initialization forward. Select `Correlations` explicitly
with `set_initialization_method` to keep that strategy. JSON now records
`InitializationMethod`; older development JSON without the field retains its
historical correlation initialization. Reproducibility requires recording
the initialization method, seed, dataset split, precision and backend.

Configure CPU-only builds with `OpenNN_DISABLE_CUDA=ON`; require a CUDA build
with `OpenNN_REQUIRE_CUDA=ON`. `OpenNN_ENABLE_MKL=OFF` selects the Eigen CPU
path. Use `Configuration::instance().set(Device::CPU, Type::FP32)` when CPU
execution is required. The CUDA verification wrapper requires a working GPU;
a successful CPU fallback is not recorded as a passing CUDA run.

JavaScript export now sanitizes feature identifiers before generating formulas,
preserves display labels with HTML escaping, and uses a one-hot dropdown for
categorical inputs. Re-export HTML models to obtain these fixes.
