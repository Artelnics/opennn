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
| `opennn/neural_network.h` | `opennn/neural_network/neural_network.h` |
| `opennn/dataset.h` | `opennn/dataset/dataset.h` and `opennn/dataset/tabular_dataset.h` |
| `opennn/standard_networks.h` | `opennn/models/models.h` |
| `opennn/training_strategy.h` | `opennn/training_strategy/training_strategy.h` |
| `opennn/dense_layer.h` | `opennn/neural_network/layers/dense_layer.h` |
| `opennn/bounding_layer.h` | `opennn/neural_network/layers/clamping_layer.h` |
| `opennn/response_optimization.h` | `opennn/response_optimization/response_optimization.h` |
| `opennn/variable.h` | `opennn/core/variable.h` |

Changing includes alone is insufficient. Use `TabularDataset` for tabular
data and the specialized image, language and time-series dataset classes for
their respective storage contracts. Use `opennn::Dense` and `Clamping` for the
current layer classes. The installed-package smoke program demonstrates adding
layers, compiling a network, and running a row-major `MatrixR` batch.

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
5. Save with `NeuralNetwork::save` and `save_parameters_binary`, reload using
   the matching current APIs, and repeat prediction comparisons. Keep paired
   architecture and parameter files together.

Current-format round trips and malformed binary rejection are tested. Those
tests do not certify arbitrary historical models. Real 8.x artifact conversion
remains application-specific work; no converter is claimed by this release.

### Available historical artifacts reviewed

The example assets in tags `v8.0.0`, `v8.0.1` and master `efd566b38` were
inspected during the release review. Their breast-cancer, Iris and MNIST `.bin`
files are dataset caches. `time_series_data_set.xml` describes a dataset, not
a neural-network topology. The Madrid `_Params.bin` has no matching saved
network architecture, feature/scaling contract and reference predictions in
those assets. These files do not form a complete production-model migration case.

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
