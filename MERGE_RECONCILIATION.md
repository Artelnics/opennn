# dev/master reconciliation for 9.0

Parents: development `5840396e9` and master `efd566b38`; common ancestor
`fc16fe9ce`. Master contributed 64 commits absent from dev by ancestry.
The trial merge produced 98 conflicted paths. Resolution keeps the current
module layout, CMake package, GoogleTest harness and device backend, and ports
missing behavior into those implementations.

## Behavioral decisions

| Master change | Reconciled implementation and verification |
| --- | --- |
| Genetic selection ranking, correlation weighting and elitism | Current `model_selection/genetic_algorithm.cpp` already maps sorted ranks back to individuals/features, preserves the best quarter with partial sorting, and excludes elites from mutation. Existing selection and cross-validation tests retained. |
| Random population initialization | Ported master's bounded Bernoulli initializer and Random default. Added an explicit method setter and JSON field; old development JSON retains Correlations. Added initialization round-trip regression. |
| Per-candidate scaling and best-epoch neuron selection | Current optimizer prepares training artifacts each run; selection utilities choose best validation history. Retained current input-shape recompilation and final scaling installation. |
| Catch all candidate-training exceptions | Not ported: invalid structural configurations remain visible exceptions. Nonfinite returned errors already receive worst fitness. |
| Convolution-specific forced Adam rate | Not ported: current explicit optimizer settings remain authoritative. |
| JavaScript numeric/overlapping names and categorical UI | Ported identifier sanitation before expression generation without temporarily renaming the source network. Added categorical one-hot selects, HTML label escaping, and output-dropdown value updates. Node executes generated formulas against native network predictions, including every dense activation and difficult labels. |
| Spearman logistic matrix sizing | Current `fit_logistic_correlation` constructs a sized two-column matrix; the broken old comma initialization is absent. |
| Missing-value filtering and histograms | Current generic row filtering checks all columns and dimensions; masked extrema ignore NaNs. Ported master's all-missing histogram contract (requested bins, zero frequencies, NaN locations) and added regression coverage. |
| Image loading/scaling and compact metadata | Current image cache and channel-level scaling supersede the old per-pixel XML/data-matrix implementation. Image tests cover cache reads, raw versus scaled inputs, categories and concurrent loads. Old XML-specific serializers are not reinstated. |
| Atomic text classification and sequence cap | Current tokenizer preserves each lowercased class label as one vocabulary token; sequence cap is already present. Added coverage for `Sci_Tech`, `World News`, and a capped input window. Reserved vocabulary tokens remain part of the current tokenizer contract. |
| Forecasting windows, missing values and timestamps | Retained current window-start indexing, chronological splitting, field parser and current snapshot handling. This deliberately differs from master's present-instant indexing; MIGRATION.md describes both formulas. Existing time-series tests and timestamp parser tests retained. |
| Response optimization, affine constraints and empty inputs | Retained current response-optimization solver and expression evaluator, with its unit tests and application scenarios. The old `formula_expression` implementation is superseded. Invalid or empty domains are diagnosed rather than silently accepted. |
| CUDA build guards and no-device fallback | Retained current device backend, CMake CUDA detection and strict CUDA verification preflight. Runtime CUDA and sanitizer gates must run on the resulting candidate. |

The flat source files, qmake scaffolding, master-only compiler cache, dummy
file and stale flat-layout CLAUDE instructions are excluded. Their removal
does not indicate that public symbols were assessed as unused; current module
implementations remain the migration target. No Neural Designer code was
changed or validated.

The apparent translation-to-ECG rename was a Git similarity inference. The
ECG file is retained. Master's MNIST binary cache is generated from the retained
BMP files and is excluded. Madrid artifacts and wastewater applications/data
are preserved under `examples/legacy_8`, with explicit limitations. They are
not added as broken targets to the current example build.

The following inventory records the structural resolution of every conflicted
path. Behavioral equivalence is limited by the explicit decisions above;
9.0 is not represented as an 8.x binary-compatible release.

## Conflicted paths

| Conflict path | Structural resolution |
| --- | --- |
| `.gitignore` | Retain current module/build/example implementation |
| `CMakeLists.txt` | Retain current module/build/example implementation |
| `blank/main.cpp` | Retain removal of obsolete flat source or qmake target |
| `blank_cuda/main.cpp` | Retain removal of obsolete flat source or qmake target |
| `cuda.pri` | Retain removal of obsolete flat source or qmake target |
| `examples/CMakeLists.txt` | Retain current module/build/example implementation |
| `examples/airfoil_self_noise/main.cpp` | Retain current module/build/example implementation |
| `examples/amazon_reviews/main.cpp` | Retain current module/build/example implementation |
| `examples/breast_cancer/main.cpp` | Retain current module/build/example implementation |
| `examples/concrete/CMakeLists.txt` | Retain removal of obsolete flat source or qmake target |
| `examples/concrete/main.cpp` | Retain current module/build/example implementation |
| `examples/ecg5000_anomaly_detection/data/ecg.csv` | Retain current module/build/example implementation |
| `examples/emotion_analysis/main.cpp` | Retain current module/build/example implementation |
| `examples/examples.pro` | Retain removal of obsolete flat source or qmake target |
| `examples/forecasting/main.cpp` | Retain removal of obsolete flat source or qmake target |
| `examples/melanoma_cancer/main.cpp` | Retain current module/build/example implementation |
| `examples/mnist/main.cpp` | Retain current module/build/example implementation |
| `examples/translation/main.cpp` | Retain current module/build/example implementation |
| `examples/translation/translation.pro` | Retain removal of obsolete flat source or qmake target |
| `opennn/CMakeLists.txt` | Retain current module/build/example implementation |
| `opennn/adaptive_moment_estimation.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/adaptive_moment_estimation.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/addition_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/bounding_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/convolutional_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/correlations.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/cross_entropy_error_3d.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/dataset.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/dataset.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/dense_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/embedding_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/flatten_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/flatten_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/genetic_algorithm.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/growing_inputs.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/growing_neurons.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/image_dataset.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/image_dataset.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/image_utilities.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/kernel.cu` | Retain removal of obsolete flat source or qmake target |
| `opennn/kernel.cuh` | Retain removal of obsolete flat source or qmake target |
| `opennn/language_dataset.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/language_dataset.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/levenberg_marquardt_algorithm.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/loss.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/loss.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/mean_squared_error.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/mean_squared_error.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/minkowski_error.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/model_expression.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/multihead_attention_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/multihead_attention_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/neural_network.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/neural_network.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/normalization_layer_3d.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/normalized_squared_error.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/optimizer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/pch.h` | Retain current module/build/example implementation |
| `opennn/pooling_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/pooling_layer_3d.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/quasi_newton_method.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/random_utilities.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/recurrent_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/recurrent_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/registry.h` | Retain current module/build/example implementation |
| `opennn/response_optimization.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/response_optimization.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/scaling_layer.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/standard_networks.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/statistics.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/stochastic_gradient_descent.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/stochastic_gradient_descent.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/string_utilities.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/string_utilities.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/tensor_utilities.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/tensor_utilities.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/testing_analysis.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/testing_analysis.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/time_series_dataset.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/time_series_dataset.h` | Retain removal of obsolete flat source or qmake target |
| `opennn/tinyxml2.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/training_strategy.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/unscaling_layer.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/variable.cpp` | Retain removal of obsolete flat source or qmake target |
| `opennn/variable.h` | Retain removal of obsolete flat source or qmake target |
| `tests/CMakeLists.txt` | Retain current module/build/example implementation |
| `tests/cross_entropy_error_3d_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/cross_entropy_error_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/data_set_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/dataset/correlations_test.cpp` | Retain current module/build/example implementation |
| `tests/genetic_algorithm_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/growing_inputs_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/mean_squared_error_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/minkowski_error_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/neural_network_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/response_optimization_test.cpp` | Retain removal of obsolete flat source or qmake target |
| `tests/test.cpp` | Retain current module/build/example implementation |
