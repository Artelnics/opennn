# OpenNN engineering audit — 2026-08-22

Scope: `opennn/` (80,476 raw / 61,562 effective lines, 219 source files), `tests/`, build and CI; `core/` read most closely. Method: 20 scoped auditors + 3 cross-cutting lenses, one adversarial verifier per scope, a completeness critic and a partial second round (50 agents, 1,914 tool calls). Kept 379 findings (273 confirmed, 88 confirmed with corrections, 3 verifier-added, 15 unverified; 2 refuted). 115 are bugs/UB; severity 21 high / 129 medium / 229 low; per-item line estimates sum to -3,374 (-4,556 removed, +1,182 added). Line numbers are as of commit 509857699. Neural Designer (external consumer) was not available: grep its tree before any deletion.

Companion HTML report with a filterable explorer of every finding: see the published artifact (same date).

## Fix first

### 1. Unbreak the Windows CPU build

Twelve #pragma omp simd sites in long_short_term_memory_layer.cpp need OpenMP 3+; the MSVC flag is plain -openmp (OpenMP 2.0), which rejects them with C7660, stopping the build. Corrected attribution: the auditor blamed commit 4ccb88dea, but git show HEAD has zero omp simd - all twelve are uncommitted working-tree edits from the other machine, which builds on WSL/GCC where they compile fine. FIXED 2026-08-22 in opennn/CMakeLists.txt: MSVC now gets -openmp:experimental, patched on the OpenMP::OpenMP_CXX interface too (the plain flag arrived twice and would otherwise win). Build green, suite unchanged at 929/2/19.

- `opennn/neural_network/layers/long_short_term_memory_layer.cpp:372-437` — '#pragma omp simd' in the LSTM CPU kernels breaks every MSVC build (C7660) (high, loc +3, S)

### 2. Restore the CPU nonlinearities lost on 2026-08-20

Commit 65f2afbda made ActivationOperator skip its pass whenever forward_fused is set, reasoning only about Dense-with-bias. Three CPU paths now have no nonlinearity: Convolutional+ReLU without batch-norm (the CPU im2col path never applies a ReLU), bias-less Dense+ReLU (linear_forward_cpu only honours RELU_BIAS), and Dense+GELUTanh with width %8==0 (the Output slot is never written: zeros). The suite is green because the conv test feeds all-positive inputs. Fix each path where the fusion is claimed, and add forward-value tests with negative inputs.

- `opennn/neural_network/operators/activation_operator.cpp:30-39` — Convolutional ReLU is never applied on CPU: 'forward_fused' skips the activation but no CPU epilogue exists (high, loc +8, S)
- `opennn/core/tensor_operations.cpp:1275-1296` — Bias-free fused-ReLU Dense on CPU skips the ReLU: output is the raw pre-activation (high, loc +1, S)
- `opennn/neural_network/layers/dense_layer.cpp:199-233` — Fused GELUTanh Dense on CPU never writes its Output slot: the layer returns zeros (high, loc +2, S)

### 3. Decide the conv activation policy (a test already fails)

Convolutional::set demotes GELU/SiLU to Identity with a cerr warning while set_activation_function throws; ActivationsTest.ConvolutionalRejectsInputDerivativeActivations fails on a rebuilt binary, and a Darknet53+SiLU YoloNetwork silently builds linear neck convolutions.

- `opennn/neural_network/layers/convolutional_layer.cpp:249-263` — Convolutional::set silently demotes unsupported activations and prints to std::cerr (medium, loc -8, S)
- `opennn/neural_network/standard_networks.cpp:494-580` — YoloNetwork with BodyActivation::SiLU builds Identity convolutions on non-V8 backbones (medium, loc +4, S)
- `opennn/neural_network/layers/convolutional_layer.cpp:246-263` — Convolutional constructor demotes GELU/SiLU to Identity with a warning while set_activation_function throws; a test now fails (medium, loc -10, S)

Current CPU suite on a rebuilt binary: 929 passed, 2 failed: ActivationsTest.ConvolutionalRejectsInputDerivativeActivations, ScalingTest.UnscaleDataMeanStandardDeviation, 19 skipped (no GPU)

## Bugs and undefined behaviour

### High

- `opennn/core/device_backend.cpp:511-528` — CudaBlockCache::give can throw from Buffer destructors -> std::terminate masks the real CUDA error (high, loc -2, S)
- `opennn/core/device_backend.cpp:1214-1225` — set_threads_number destroys the ThreadPool that tensor_operations caches for the process lifetime (UAF) (high, loc -1, S)
- `opennn/core/string_utilities.h:27-58` — Apple from_chars shim calls itself for integral T: infinite recursion on macOS builds (high, loc 0, S)
- `opennn/core/string_utilities.cpp:147-216` — Quoted-field tokenizer deletes every ',' and ';' inside quotes regardless of the separator (high, loc -2, S)
- `opennn/dataset/tabular_dataset.cpp:1342-1382` — BinaryFile storage: analysis methods index the empty `data` matrix (null-pointer reads) (high, loc +18, M)
- `opennn/neural_network/layers/concatenation_layer.cpp:34-50` — Float-only layers accept a BF16 compute dtype and reinterpret BF16 buffers as float (high, loc +12, S)
- `opennn/neural_network/layers/long_short_term_memory_layer.cpp:983-1012` — LSTM on CUDA has no FP32 guard: BF16 networks feed BF16 slots to CUDNN_DATA_FLOAT descriptors (high, loc +4, S)
- `opennn/neural_network/standard_networks.cpp:1951-1983` — load_darknet_backbone_v11 targets c11_* labels the builder no longer emits; always loads 0 layers (high, loc -56, S)
- `opennn/neural_network/neural_network.cpp:1002-1017` — set_parameters / load_parameters_binary on a released fp32 master overflow the compact bf16 mirror (high, loc +5, S)
- `opennn/neural_network/model_expression.cpp:1993-2008` — Logarithm scaler/unscaler exports to Python (NameError) and JavaScript (ReferenceError) (high, loc -3, S)
- `opennn/neural_network/operators/dropout_operator.cpp:117-127` — Dropout seed is baked into captured CUDA graphs: every replay reuses the same mask (high, loc +20, M)
- `opennn/neural_network/forward_propagation.cpp:1165-1190` — CPU valid-length record is frozen after the first forward pass; later padded batches use stale masks (high, loc -8, S)
- `opennn/training_strategy/optimizer.cpp:1606-1610` — run_graph_epoch dereferences a null pipeline slot when grouped slots exist but a post_batch_callback is set (high, loc +2, S)
- `opennn/testing_analysis/testing_analysis.cpp:264-292` — calculate_errors divides the Minkowski error by the unrelated batch_size member (default 0 -> +inf) (high, loc 0, S)
- `opennn/training_strategy/error_functions.cpp:201-217` — NormalizedSquaredError drops the total/batch scaling its sibling WSE and the old implementation apply (high, loc +6, S)

### Medium, by area

**core**

- `opennn/core/statistics.cpp:690-702` — calculate_rank sorts with a NaN-unsafe comparator; NaN correlations reach it (medium, loc +3, S)
- `opennn/core/json.cpp:226-239` — JSON number dump casts double to long long before range check; NaN/inf become unparsable tokens (medium, loc +3, S)
- `opennn/core/statistics.cpp:228-255` — variance() lacks the negative-variance clamp its sibling has; three different variance formulas (medium, loc -6, S)
- `opennn/core/random_utilities.cpp:250-280` — Global mutex-serialized RNG makes OpenMP callers slower than serial and non-reproducible under set_seed (medium, loc -1, S)
- `opennn/core/cuda/cudnn_frontend_utilities.h:394-418` — Plan-cache key omits OPENNN_SDPA_AUTOTUNE: attention autotune silently never runs once the cache is warm (medium, loc +2, S)
- `opennn/core/cuda/cudnn_frontend_utilities.h:376-391` — plan_cache_directory uses the throwing temp_directory_path: a bad TMP disables the cuDNN frontend for the process (medium, loc +4, S)
- `opennn/core/cuda/kernel_attention.cu:677-801` — GPU sampler silently ignores every logit beyond index 262,144 (no host check) (medium, loc +8, S)

**datasets**

- `opennn/dataset/tabular_dataset.cpp:1342-1382` — BinaryFile storage: analysis methods index the empty `data` matrix (null-pointer reads) (high, loc +18, M)
- `opennn/dataset/tabular_dataset.cpp:2519-2551` — impute_missing_values_interpolate interpolates against a phantom point (sample 0, value 0) (medium, loc -4, S)
- `opennn/dataset/tabular_dataset.cpp:619-683` — unuse_collinear_variables leaves input_shape stale; shape/role resync is hand-copied in 5 places (medium, loc -2, M)
- `opennn/dataset/tabular_dataset.cpp:1822-1823` — read_csv keeps stale sample roles/ids from a previous file (resize instead of assign) (medium, loc 0, S)
- `opennn/dataset/field_parsing.cpp:29-35` — CsvReader trims tabs/spaces from every line, so TSV/space files lose leading/trailing empty fields (medium, loc +2, S)
- `opennn/dataset/tabular_dataset.cpp:942-964` — Input-input correlations include unused (None) samples while input-target correlations exclude them (medium, loc +2, S)
- `opennn/dataset/time_series_dataset.cpp:258-275` — impute_missing_values_unuse checks a lags+1 window but targets read past+future rows (medium, loc 0, S)
- `opennn/dataset/image_processing.cpp:128-198` — 8-bit BMP: palette sized by biClrUsed but indexed by raw pixel byte (out-of-bounds read) (medium, loc 0, S)
- `opennn/dataset/yolo_dataset.cpp:1505-1510` — YOLO cache accepted with a stale classes_number: .names file is not in the sources hash (medium, loc -2, S)

**network, layers, operators**

- `opennn/neural_network/neural_network.cpp:1500-1501` — to_JSON stages states based on the parameters' device, not the states' device (medium, loc -6, S)
- `opennn/neural_network/neural_network.cpp:2632-2642` — upload_parameters_bf16_inference silently migrates a CPU-configured network's parameters to the GPU (medium, loc +3, S)
- `opennn/neural_network/operators/embedding_lookup_operator.cpp:279-285` — A second compile() zeroes the sinusoidal positional-encoding table (medium, loc -4, S)
- `opennn/neural_network/operators/batch_norm_operator.cpp:292-298` — Running variance uses the biased estimator on CPU but the unbiased one on GPU (medium, loc 0, S)
- `opennn/neural_network/operators/batch_norm_operator.cpp:123-135` — set_parameters_random/glorot throw when the network's states live on the device (medium, loc -1, S)
- `opennn/neural_network/operators/attention_operator.cpp:1074-1131` — CPU attention backward infers padding from trailing zeros of head 0/row 0 - softmax underflow zeroes real gradients (medium, loc -6, S)
- `opennn/neural_network/operators/attention_operator.cpp:653-787` — CPU causal/dropout attention ignores exported sequence lengths the CUDA path honours - CPU/GPU divergence after layer 1 (medium, loc +10, S)
- `opennn/neural_network/layers/embedding_layer.h:26-29` — Embedding::set_input_shape is accepted for rank 1 but silently changes nothing (medium, loc +2, S)
- `opennn/neural_network/layers/recurrent_layer.cpp:103-134` — Recurrent back_propagate indexes scratch slots that do not exist for a frozen layer (medium, loc +2, S)
- `opennn/neural_network/layers/dense_layer.cpp:295-302` — DReLU wiring is reset one-sided: reconfiguring the consumer after compile drops the producer's ReLU backward (medium, loc +6, S)

**builders, chat, export**

- `opennn/neural_network/standard_networks.cpp:1898-1987` — Both Darknet loaders leak the FILE* when load_darknet_weights throws; header reading duplicated (medium, loc -14, S)
- `opennn/neural_network/standard_networks.cpp:483-484` — DarknetTinyV3+FPN accepts 9 anchors but is 2-head: 6 anchors land on a 3-anchor logits conv (medium, loc +2, S)
- `opennn/neural_network/chat.cpp:156-167` — sample_token ignores repetition_penalty when temperature == 0; DecoderSampler applies it (medium, loc 0, S)
- `opennn/neural_network/model_expression.cpp:677-701` — emit_c_main injects raw output names into a printf format string (and HTML/PHP literals) (medium, loc -16, S)
- `opennn/neural_network/model_expression.cpp:1528-1543` — Embedded Logarithm scaling is unclamped; library computes log(max(x, EPSILON)) (medium, loc -2, S)
- `opennn/neural_network/model_expression.cpp:1905-1962` — JS category selector compares raw option values against sanitized output ids (medium, loc 0, S)
- `opennn/neural_network/model_expression.cpp:2101-2107` — Python emitter bypasses fix_names: unnamed inputs become `variable`, and fix_names never dedupes (medium, loc +2, S)

**training**

- `opennn/training_strategy/error_functions.cpp:201-217` — NormalizedSquaredError drops the total/batch scaling its sibling WSE and the old implementation apply (high, loc +6, S)
- `opennn/training_strategy/loss.h:216-224` — YoloLambdas positional init shifted when `dfl` was inserted; tests now run with lambda_class = 0 (medium, loc +2, S)
- `opennn/training_strategy/adaptive_moment_estimation.cpp:142-185` — Adam uses two unrelated step counters when CUDA graphs are on: device step for whole batches, host iteration for the tail (medium, loc -2, S)
- `opennn/training_strategy/optimizer.h:52-74` — set_display_period(0) / set_validation_period(0) / JSON DisplayPeriod 0 cause integer modulo by zero (medium, loc +4, S)
- `opennn/dataset/batch.cpp:269-272` — Batch::~Batch calls a throwing CUDA sync; after a sticky CUDA error unwinding ends in std::terminate (medium, loc +2, S)
- `opennn/training_strategy/loss.cpp:986-1038` — 16 raw CUDA copy/memset/sync sites outside device::, 13 unchecked; loss.cpp relies on legacy-stream cudaMemcpy (medium, loc -4, S)

**model selection, testing analysis, response optimization**

- `opennn/model_selection/growing_neurons.cpp:211-230` — GrowingNeurons never reports MaximumEpochs: exhausting the loop leaves stopping_condition empty and elapsed_time blank (medium, loc +2, S)
- `opennn/testing_analysis/testing_analysis.cpp:869-873` — Matthews correlation multiplies four Index counts; signed overflow at ~110k balanced testing samples (medium, loc 0, S)
- `opennn/testing_analysis/testing_analysis.cpp:775-803` — calculate_multiple_classification_rates writes out of bounds when targets have one column (medium, loc -2, S)
- `opennn/model_selection/growing_inputs.cpp:72-79` — GrowingInputs::set_maximum_inputs_number dereferences a null training strategy; ModelSelection::load() without a strategy segfaults (medium, loc 0, S)
- `opennn/response_optimization/response_optimization.cpp:1139-1154` — Domain::reshape pins Binary scalar inputs to 1 whenever any nearby point had a 1 (medium, loc +2, S)
- `opennn/response_optimization/response_optimization.cpp:82-97` — Univariate EqualTo on an output is filtered with a 1e-6 absolute band, so it is effectively infeasible (medium, loc 0, S)

**build & CI**

- `CMakeLists.txt:93-134` — OPENNN_HAS_CUDA is a non-FORCE cache set: a reconfigure keeps the stale CUDA decision (medium, loc -2, S)
- `.github/workflows/tinyml-parity.yml:12-35` — tinyml-parity workflow path filters point at files that no longer exist (medium, loc 0, S)
- `examples/airfoil_self_noise/main.cpp:32` — airfoil_self_noise hard-codes Device::CUDA and therefore throws on every CPU-only build (medium, loc 0, S)
- `examples/yolo/main.cpp:364-369` — yolo example leaves std::cout pointing at a destroyed TeeBuf on any exception path (medium, loc +3, S)

## Runtime overhead

**Inference API cliff (per-call arena)**

- `opennn/neural_network/neural_network.h:219-226` — GPU 'buffer-reusing' calculate_outputs still allocates a ForwardPropagation arena per call (medium, loc +15, S)
- `opennn/neural_network/neural_network.h:226-228` — calculate_outputs_resident defaults upload_parameters=true, which discards the CUDA graph every call (medium, loc 0, S)
- `opennn/neural_network/neural_network.cpp:1202-1217` — calculate_outputs tail tile builds a second ForwardPropagation with its own arena instead of sharing the tile's (medium, loc +1, S)
- `opennn/neural_network/neural_network.cpp:1291-1292` — Every forward call re-materialises all parameter specs just to compare one integer (low, loc +2, S)
- `opennn/neural_network/neural_network.cpp:1311` — Unconditional copy_states_device() per GPU forward rebuilds every operator's state views each batch (low, loc +1, S)

**Training loop**

- `opennn/training_strategy/optimizer.cpp:2126-2162` — Validation remainder batch allocates a Batch and a ForwardPropagation arena every validation epoch; the allocation guard is disabled to allow it (medium, loc +12, M)
- `opennn/training_strategy/optimizer.cpp:2134-2152` — Validation tail builds a Batch (cudaMalloc + cudaMallocHost + event) and a ForwardPropagation arena every epoch, and disables the allocation guard for the whole run (medium, loc +12, M)
- `opennn/training_strategy/optimizer.cpp:1762-1770` — GPU validation loop host-syncs the compute stream after every batch; the Batch event machinery already covers the hazard (medium, loc -1, S)
- `opennn/training_strategy/optimizer.cpp:1723-1774` — Default (non-graph) GPU training path has zero H2D/compute overlap: one fixed device slot, copy waits for the previous step (medium, loc +15, M)
- `opennn/dataset/batch.cpp:336-348` — BF16 host cast of the whole input batch runs on the GPU-feeding main thread inside upload_to_device_batch_async instead of in the prefetch worker (medium, loc 0, S)
- `opennn/training_strategy/loss.cpp:1373-1376` — Per-batch regularization penalty (full-parameter reduction + cuBLAS host sync) whose result the optimizer overwrites (medium, loc -12, S)
- `opennn/training_strategy/levenberg_marquardt_algorithm.cpp:278-312` — LM allocates a validation Jacobian (V*outputs x P) and a P x P Hessian that validation never reads (medium, loc -3, S)
- `opennn/neural_network/forward_propagation.cpp:832-836` — ForwardPropagation::set rewrites a process-global conv workspace cap that keys the cuDNN plan cache (medium, loc +6, S)

**CPU kernels**

- `opennn/core/tensor_operations.cpp:1174-1246` — CPU activation forward/backward run single-threaded while sibling ops in the file parallelise at 65536 (medium, loc +12, S)
- `opennn/neural_network/operators/convolution_operator.cpp:526-533` — CPU convolution allocates and zeroes a workers x weights gradient buffer plus im2col scratch on every call (medium, loc -2, M)
- `opennn/neural_network/layers/addition_layer.cpp:18-26` — AdditionOperator forward spends a full copy pass before the first add (medium, loc 0, S)
- `opennn/neural_network/layers/long_short_term_memory_layer.cpp:728-737` — LSTM CPU backward (H<96) allocates and zero-fills per-thread gradient scratch for omp_get_max_threads() every call (medium, loc 0, S)
- `opennn/neural_network/layers/long_short_term_memory_layer.cpp:234-249` — LSTM fused path rebuilds Wcat/Ucat(/bcat) identically in forward and backward, per call (medium, loc -18, S)
- `opennn/neural_network/operators/c2psa_operator.cpp:358-384` — C2PSA CPU back_propagate heap-allocates six Eigen matrices per batch element per call although a backward scratch slot is already planned (medium, loc -4, S)
- `opennn/neural_network/layers/pooling_layer.cpp:558-567` — Pooling/Convolutional apply_input_shape skip the geometry validation their set() performs (low, loc -10, S)
- `opennn/neural_network/operators/convolution_operator.h:63-67` — ConvolutionOperator::set takes ten positional Index arguments, has one caller, and only copies public fields (low, loc -13, S)

**GPU kernels and cuDNN**

- `opennn/training_strategy/error_functions.cpp:383-421` — CE3d GPU forward still uses the serial one-thread-per-token argmax kernel the metrics path already replaced (medium, loc -60, M)
- `opennn/training_strategy/loss.cpp:970-1099` — YOLO v8 GPU round-trip runs twice per batch: TAL assignment, D2H of targets and every head repeated (medium, loc -15, M)
- `opennn/training_strategy/loss.cpp:807-968` — MSVC compiles the v8 TAL gradient kernel at /Od, and that kernel is the GPU training path too (medium, loc -4, S)
- `opennn/neural_network/operators/cudnn_rnn.cpp:230-277` — cuDNN RNN re-derives weight-region pointers on every forward and backward: 2 descriptor create/destroy + N cudnnGetRNNWeightParams per call (medium, loc +6, S)
- `opennn/neural_network/operators/kernel_c2psa.cu:28-110` — C2PSA ships its own one-thread-per-row softmax fwd/bwd kernels; library cuDNN softmax and a shared softmax_backward would replace them and 3 hand loops (medium, loc -15, M)
- `opennn/neural_network/layers/long_short_term_memory_layer.cpp:1096-1127` — LSTM allocates six unused B×T×H gate tensors on CUDA; cuDNN-only scratch allocated on CPU (medium, loc +4, S)
- `opennn/core/cuda/cudnn_frontend_utilities.h:679-712` — autotune_with_scratch sizes every scratch tensor as fp32: 2x the real transient on bf16 graphs (medium, loc +6, S)
- `opennn/core/cuda/kernel_activation.cu:82-200` — Activation fwd/bwd carry three hand-written kernels each; one VecIO<T,vec16<T>> kernel covers both dtypes (medium, loc -45, M)

**Data pipeline**

- `opennn/dataset/tabular_dataset.cpp:398-421` — Per-batch training scaling runs a per-element switch through scale_value in a strided loop (medium, loc +6, M)
- `opennn/dataset/tabular_dataset.cpp:257-287` — fill_from_binary_cache issues one ReadFile/pread per row, serially, and duplicates read_int32_batch (medium, loc -20, M)
- `opennn/dataset/tabular_dataset.cpp:897-909` — Correlation loops re-gather variable columns O(inputs x targets) and O(inputs^2) (medium, loc +2, S)
- `opennn/dataset/tabular_dataset.cpp:1822-1823` — read_csv keeps stale sample roles/ids from a previous file (resize instead of assign) (medium, loc 0, S)
- `opennn/model_selection/cross_validation.cpp:138-160` — k-fold finalisation re-trains all k folds just to recover an epoch count the selection loop already had (medium, loc +6, M)
- `opennn/model_selection/genetic_algorithm.cpp:185-238` — GeneticAlgorithm keeps a full parameter vector for every individual of every generation; only the best is ever read (medium, loc -8, S)

## Progress

Closed so far: **all 115 bug/UB findings**, **all 21 high-severity findings**, and 48 quality
items. Remaining: ~215, all medium or low.

Two golden harnesses were added to make the larger refactors verifiable rather than
argued about. Both skip unless their environment variable is set, so neither costs
anything in CI:

- `tests/neural_network/model_expression_golden_test.cpp` -- dumps all five language
  emitters for three networks to `OPENNN_EXPRESSION_DUMP_DIR` (15 files).
- `tests/neural_network/network_topology_golden_test.cpp` -- dumps every layer's label,
  type, shapes, parameter count and source indices for 17 configurations to
  `OPENNN_TOPOLOGY_DUMP_DIR` (1,961 lines), covering each branch `YoloNetwork` can take.

Every emitter and builder change since has been verified byte-identical against them.

Declined, with reasons in the commit messages: `nn-expression-19` (rewriting the name
mapping as a tokenizer risks the multi-word and `scaled_` cases for 8 lines in a cold
export path) and `nn-builders-chat-8` (the audit's own fix note says it changes
classic-session sampling behaviour).

Largest remaining single items: `nn-expression-13` (-120, one `LanguageSyntax`-driven
emitter replacing four per-neuron exporters), `nn-builders-chat-6`'s remaining half
(lifting the `YoloNetwork` helpers to file scope as a `YoloBuilder`, which is what would
actually shorten the 745-line constructor), and `xcut-build-tests-16` (+400, the 24
library files with no mirrored test).

## Uninitialised device memory: the poison mode

`device::allocate` promises nothing about contents -- `set_zero` is a separate call --
but fresh CUDA pages arrive zeroed from the driver, so a site that reads before writing
is correct until the block cache hands it a recycled block instead. That is a class of
defect no amount of reading the source finds, and the cuDNN RNN weight-space bug
(`ebc6cb14e`) was a real instance: it silently corrupted GPU training results and
survived twenty audit agents.

`OPENNN_DEVICE_POISON` makes the class visible. It is off by default and costs nothing.

| value | behaviour |
|---|---|
| `1` | recycled blocks come back as `0xFF` -- NaN as fp32 and as bf16 |
| `2` | recycled blocks come back as zeros; **the control** |
| `3` | as `1`, and fresh `cudaMalloc` too, so a site fails on its own allocation rather than on the pool's history |

**Always read mode 1 against mode 2.** The fill has to be ordered against all work or the
diagnostic accuses the wrong code -- a memset landing after the kernel that filled a
buffer looks exactly like a read of uninitialised memory. An unordered fill reported 16
failures where 3 were real; the control passing at 1051/0 is what makes a mode 1 result
trustworthy. `compute-sanitizer --tool initcheck` (with `OPENNN_DEVICE_CACHE=0` and no
poison) is the independent second opinion.

### Open leads

Three tests fail under mode 1 with the control clean. All three involve a network whose
parameters are set on the host and used on the GPU:

- `GpuComparison.RnnDescriptorCacheSupportsConcurrentMixedBatches`
- `AdaptiveMomentEstimationTest.TrainForecastingGPU`
- `MeanSquaredErrorTest.GpuWorkspaceIsForwardPropagationOwned` -- was missing
  `copy_parameters_device()` after writing through `get_parameters_map()`; added, which
  fixes it in isolation but not in a full-suite run, so there is a second cause too.

Already ruled out: `OptimizerData::set` zeroes Adam's moments, `cudnn_unpack_gradients_`
zeroes the RNN backward weight space before `cudnnRNNBackwardWeights_v8` accumulates into
it, and both propagation arenas zero themselves at construction.

With the fill correctly ordered all three now reproduce **in isolation** under mode 1,
which is the part that was hard before. What is still unexplained:

    poison off, cache on      pass
    poison off, cache off     pass
    NaN-poison, cache on      FAIL
    zero-poison, cache on     pass
    initcheck, cache off      0 uninitialised reads

A plain read-before-write should show up in initcheck. It does not, and the failure needs
the block cache to be on. Two candidates: a buffer read before being written that happens
to be correct at zero, or a block being read after it was handed back -- a use-after-free,
and much the more serious of the two.

**Settled by mode 4, and it is the benign one.** Mode 4 poisons on `give()` and zeroes on
`take()`, which separates them: a new owner reading before writing gets zeros and is
happy, while an old owner still holding the block sees NaN. Mode 4 passes the whole suite,
1052/0, where mode 1 fails three. So nothing reads a block after returning it, and the
cache's event guard is doing its job.

That leaves the three as buffers read before being written whose contents happen to be
correct at zero. It also means the zeroing guarantee is semantically the right fix for
them -- what defeated it earlier was stream ordering, not the idea (see the poison fill
note: OpenNN's streams are non-blocking, so a fill has to be ordered against all of them).

Still unexplained: why initcheck reports nothing. Lower priority now that the serious
possibility is ruled out.

Not urgent -- the library is green with poison off. Worth revisiting if anyone reports
non-reproducible GPU training results, or if the block cache is made to recycle more
aggressively. Worth adding mode 1 as a CI job once the three are cleared.

## Benchmarks rot because nothing compiles them with CUDA

CI compiles the benchmark set, but with `OpenNN_DISABLE_CUDA=ON`, so every target
inside the `if(OPENNN_HAS_CUDA)` block -- `cudnn_fusion_probe`, `pooling_probe`, the
transformer resident/train/energy drivers -- is never built by anything automatic.
`cudnn_fusion_probe` had accumulated 36 calls to `Buffer::data` without parentheses and
a `finalize()` call carrying an argument removed some time ago.

Closing this needs a CUDA runner, which the CI file's own comment already notes.

One correction to a claim in commit 60782dc47: that message wonders whether the CI
benchmark job is green, because `opennn_higgs_cpu` is CPU-only and was broken. It was
not a CI failure. `git log -S` puts the deletion of `Shape(size_t, Index)` in b6d49f4b9,
earlier the same day -- the audit called it zero-caller, it had two in a file no local
build compiles, and the benchmark set was not rebuilt after the deletion. CI would have
caught it on that push. Verify locally with:

    cmake --build build-<cuda-dir> --target benchmarks

## Where the GPU time actually goes, and the cuDNN fusion question

Measured on ResNet-50 / CIFAR-10 (the datasets live in `$OPENNN_BENCH_DATA`, default
`~/opennn-benchmark-data`; the benchmark binaries take positional arguments, not the
`--flags` the READMEs show). Baseline batch 64: 3,224 samples/s fp32, 5,794 bf16.

Steady-state epoch profile (`OPENNN_PROFILE=1`), 405 ms epoch:

| scope | ms | % |
|---|---|---|
| `bwd:Convolutional` | 108.6 | 26.8 |
| `fwd:Convolutional` | 41.7 | 10.3 |
| `op:bn_bwd` + `op:bn_fwd` | 28.2 | 6.9 |
| `step:wait_fill` | 0.2 | 0.1 |

**The input pipeline is not a bottleneck.** `worker:queue_wait` looks enormous (>100%)
but that is the worker idling because it is ahead of the consumer; the main thread's
wait is `step:wait_fill`, 0.2 ms.

**Most overhead findings are not where the time is.** `operators-b-6` measures 0.6% of
RNN time; `dataset-a-9` is a `TabularDataset` path ResNet never enters. Measure before
taking one on.

### Fusion: measured and negative on sm_86

`cudnn_fusion_probe` (bit-rotted, repaired) tests the premise behind the planned Phase 3
MLPerf fusion architecture -- that cuDNN's fused engines run near plain-conv speed on
CIFAR geometry. On an RTX 3060 (sm_86) with cuDNN 9.19, batch 256 bf16, nine real
ResNet-50 shapes:

| pattern | engines found | speed vs plain |
|---|---|---|
| `fprop + genstats` | **0 of 9** | -- |
| `SBRCS` (BN-apply+ReLU prologue+genstats) | **0 of 9** | -- |
| `DBAR` (dgrad+dReLU+dbn_weight) | **0 of 9** | -- |
| `dgrad + dReLU` | 9 of 9 | 0.88-7.13x, median ~1.03x |

Three of the four patterns have no engine at all; the one that exists is a wash at best
and 7x worse on one shape. **Phase 3 is not viable on this hardware.** The result is
hardware-specific -- the fused engines very likely exist on A100/H100 -- so re-run the
probe before concluding anything about a datacenter target.

### Plan selection and layout: also already tuned

Sweeping the conv workspace cap (bf16, batch 64, 3 runs each, median samples/s):

| cap | auto | 16 MiB | 64 MiB | 256 MiB | 1024 MiB |
|---|---|---|---|---|---|
| samples/s | **8682** | 8676 | 8603 | 8655 | 8489 |

`auto` is the best of them and larger caps are slightly worse, which is what the note in
`finalize` already claims. Nothing to gain here. Layout is already NHWC.

So "call cuDNN better" is answered, and the answer is mostly no: fusion has no engines on
this hardware, the workspace cap is tuned, the layout is right. The convolution time is
cuDNN's own.

**Measurement caution.** Absolute throughput moves a lot with GPU clock and thermal state
-- the same bf16 configuration measured 6,994 and 8,682 samples/s in two sessions an hour
apart. Only A/B comparisons taken back-to-back in one session mean anything; never
compare a number against one from an earlier run.

## Fewer effective lines

| Item | Kind | Lines | Effort | Risk |
|---|---|---:|---|---|
| `opennn/dataset/dataset.cpp:227` 115 strictly trivial getters/setters still defined out-of-line in 25 .cpp files (~400 lines) | boilerplate | -350 | M | low |
| `tests/training_strategy/yolo_loss_test.cpp:24` write_bmp_24/TempDir/write_label/write_classes copy-pasted into 7 test files (+yolo example); copies already diverged | duplication | -340 | M | low |
| `examples/blank/main_cuda.cpp:50` blank_cuda is 230 lines entirely under #if 0 with hard-coded /home/artelnics paths, plus its own CMake/option plumbing | dead code | -250 | S | medium |
| `tests/test.cpp:20` 177 Configuration::set calls in tests are redundant with the per-test listener reset | boilerplate | -174 | S | low |
| `opennn/neural_network/model_expression.cpp:589` Four per-neuron exporters repeat one traversal; a LanguageSyntax-driven emitter removes ~180 lines | design | -120 | L | medium |
| `tests/numerical_derivatives.cpp:155` Numerical Hessian helper returns a zero matrix; its only consumer asserts nothing; input-deltas helper has no callers | dead code | -95 | S | low |
| `opennn/neural_network/standard_networks.cpp:460` YoloNetwork ctor (731 lines) duplicates SPPF, FPNv8 prior-bias, v8 det head, class-activation, PAN/neck blocks | duplication | -90 | L | medium |
| `opennn/dataset/tabular_dataset.cpp:398` Per-batch training scaling runs a per-element switch through scale_value in a strided loop | overhead | +6 | M | low |
| `opennn/neural_network/neural_network.cpp:513` About 108 lines of one-expression wrappers defined out-of-line in neural_network.cpp | boilerplate | -85 | S | low |
| `opennn/neural_network/model_expression.cpp:896` 124-line embedded scaler switch re-derives slope/offset that core/scaling.h scaling_affine() provides | duplication | -85 | M | low |
| `tests/neural_network/int8_inference_test.cpp:22` Qwen3/INT8 test helpers are still duplicated verbatim between two files | duplication | -75 | S | low |
| `opennn/neural_network/layers/grouped_query_attention_layer.cpp:110` rotary_backward / rotary_backward_cpu / rope_backward_gpu are never called by the library (GQA is inference-only) | dead code | -90 | S | medium |
| `opennn/response_optimization/response_optimization.h:240` Forecasting path (fixed_history, time_roles, combine_input) is unreachable: no setter exists | dead code | -75 | S | medium |
| `opennn/neural_network/neural_network.h:70` add_layer returns void: 71 'x = get_layers_number() - 1' lines in standard_networks.cpp (134 repo-wide) | boilerplate | -71 | M | low |
| `opennn/neural_network/chat.cpp:145` Two parallel sampling implementations (sample_token_with_workspace vs DecoderSampler::sample_host) with diverging semantics | duplication | -70 | M | medium |
| `opennn/training_strategy/loss.cpp:982` Seven raw, unchecked CUDA calls in the YOLO drivers (7 of the 8 in all opennn .cpp files) | bug | 0 | S | low |
| `examples/yolo/main.cpp:1141` yolo example duplicates its FPN-head collection/decode block, GtBox, and box-to-pixel math inside one 1818-line main | duplication | -70 | M | low |
| `opennn/training_strategy/error_functions.cpp:383` CE3d GPU forward still uses the serial one-thread-per-token argmax kernel the metrics path already replaced | overhead | -60 | M | medium |
| `opennn/core/json.cpp:608` 73 two-line `if (has) read_json` guards exist only because read_json_* has no fallback parameter | boilerplate | -65 | S | low |
| `opennn/neural_network/standard_networks.cpp:1951` load_darknet_backbone_v11 targets c11_* labels the builder no longer emits; always loads 0 layers | bug | -56 | S | medium |
| `opennn/neural_network/model_expression.cpp:1234` Embedded Recurrent/LSTM packing loops are element-wise identity copies of row-major views | duplication | -55 | S | low |
| `opennn/neural_network/model_expression.cpp:677` emit_c_main injects raw output names into a printf format string (and HTML/PHP literals) | bug | -16 | S | low |
| `opennn/training_strategy/levenberg_marquardt_algorithm.cpp:31` set_default() bodies duplicate or contradict header default member initializers in 7 classes | design | -55 | S | medium |
| `opennn/dataset/tabular_dataset.cpp:2698` infer_column_types duplicates infer_dataset_date_format and then calls it anyway | duplication | -52 | S | low |
| `opennn/neural_network/layers/grouped_query_attention_layer.cpp:898` GQA project→qk_norm→rope→attend→o_proj pipeline is written four times (CPU/GPU × batch==1/batch>1) | duplication | -50 | M | medium |
| `opennn/neural_network/layers/scaling_layer.cpp:245` Scaling/Unscaling/Flatten apply_input_shape reset the label (and Scaling's statistics) | bug | -8 | S | low |
| `opennn/core/cuda/kernel_activation.cu:82` Activation fwd/bwd carry three hand-written kernels each; one VecIO<T,vec16<T>> kernel covers both dtypes | duplication | -45 | M | medium |
| `opennn/training_strategy/optimizer.cpp:1780` train_epoch and evaluate_epoch are the same epoch skeleton written twice (~80 lines removable with one driver) | duplication | -45 | M | medium |
| `opennn/dataset/text_generation_dataset.cpp:43` TextGenerationDataset re-implements the tokenizer's vocabulary counting and rebuilds its lookup map | duplication | -35 | M | low |
| `opennn/dataset/yolo_dataset.cpp:558` Two uint8 bilinear resizers (bilinear_resize_uint8 vs blit_resized_into_canvas) with different sampling conventions | duplication | -35 | S | medium |
| `opennn/training_strategy/loss.cpp:1428` Single-head YOLO CPU entry points duplicate the multi-head path, which only differs by an identity copy | boilerplate | -35 | S | low |
| `opennn/neural_network/layers/pooling_layer.h:149` Pooling and Convolutional mirror every geometry field of their operator | duplication | -35 | M | medium |
| `opennn/core/string_utilities.cpp:303` replace() and replace_all_appearances() are the same function except a hidden '_' rule | duplication | -35 | M | medium |
| `opennn/training_strategy/loss.cpp:1034` v8 delta upload relies on legacy-stream implicit ordering with the compute stream | bug | 0 | S | low |
| `opennn/core/string_utilities.cpp:47` Free function tokenize(const string&) has no library caller; tokenize_views is the live variant | dead code | -33 | S | medium |
| `opennn/neural_network/model_expression.cpp:589` 44 call sites recompute the same input/output/fixed name vectors across the emitters | boilerplate | -32 | S | low |
| `opennn/dataset/tabular_dataset.cpp:2287` read_csv's post-parse type refinement re-implements infer_variable_types_from_data; both entry points are unused | duplication | -30 | S | medium |
| `opennn/dataset/yolo_dataset.cpp:1390` Target-cache writer duplicated in try_rebuild_target_from_boxes and build_cache, with a v8 divergence | duplication | -30 | M | low |
| `opennn/training_strategy/loss.cpp:741` DFL box decode, DFL target computation and reg_max softmax written three, two and three times | duplication | -30 | S | low |
| `opennn/CMakeLists.txt:241` Arch/warning/optimisation flags are set twice (root add_compile_options and opennn target) and partly contradict | boilerplate | -30 | S | medium |
| `opennn/dataset/tabular_dataset.cpp:979` calculate_correlations_rank, get_used_variables_indices and both set_variable_type overloads have no callers in the repo | dead code | -30 | S | medium |
| `opennn/model_selection/growing_neurons.h:46` GrowingNeurons re-declares InputsSelection's whole configuration block (7 knobs, 8 setters, save/load) | duplication | -28 | M | medium |
| `opennn/training_strategy/error_functions.cpp:87` Ten hand-written template CUDA stubs, one of them unreachable | boilerplate | -24 | S | low |
| `opennn/dataset/tabular_dataset.cpp:257` fill_from_binary_cache issues one ReadFile/pread per row, serially, and duplicates read_int32_batch | overhead | -20 | M | low |
| `opennn/neural_network/layers/activation_layer.cpp:35` apply_input_shape overrides and set() bodies repeat the rank check the base already enforces | boilerplate | -22 | S | low |
| `opennn/core/statistics.cpp:151` Two Histogram constructors are unused; one re-implements histogram() | dead code | -22 | S | medium |
| `tests/numerical_derivatives.cpp:43` Four numerical-derivative helpers repeat the same 14-line batch-building preamble | boilerplate | -25 | S | low |
| `opennn/neural_network/layers/unscaling_layer.cpp:106` Unscaling::write_expression is a hand-rolled twin of Scaling's affine export | duplication | -25 | M | low |
| `opennn/training_strategy/adaptive_moment_estimation.cpp:25` Static update_parameters_cuda wrapper + CUDA stub duplicated in Adam and SGD; the Capturable branch already calls the kernel inline | boilerplate | -22 | S | low |
| `opennn/neural_network/model_expression.cpp:623` Activation-body emission loop written four times | duplication | -22 | S | low |
| `opennn/dataset/batch.cpp:35` Batch::set is only ever called by the constructor; its reset/shrink logic (~20 lines) never executes on a live object | dead code | -20 | S | medium |
| **Sum** | | **-3161** | | |

## Design and public API

- `opennn/neural_network/neural_network.h:226-228` — calculate_outputs_resident defaults upload_parameters=true, which discards the CUDA graph every call (medium, loc 0, S)
- `opennn/neural_network/neural_network.h:219-226` — GPU 'buffer-reusing' calculate_outputs still allocates a ForwardPropagation arena per call (medium, loc +15, S)
- `opennn/neural_network/neural_network.h:278-297` — Residency has ~10 public entry points; most are implementation steps with 0-1 external callers (low, loc +2, S)
- `opennn/training_strategy/optimizer.h:185-309` — The (batches, input, decoder, target) index tuple is threaded positionally through eight signatures (low, loc -20, M)
- `opennn/neural_network/layers/convolutional_layer.h:103-108` — Class layout depends on OPENNN_HAS_CUDA: data members wrapped in #ifdef in 5 headers (medium, loc -4, S)
- `opennn/training_strategy/levenberg_marquardt_algorithm.cpp:31-49` — set_default() bodies duplicate or contradict header default member initializers in 7 classes (medium, loc -55, S)
- `opennn/response_optimization/response_optimization.cpp:1644-1694` — ResponseOptimization writes 36 unconditional cout messages and has no display flag (medium, loc +10, S)
- `opennn/neural_network/forward_propagation.cpp:832-836` — ForwardPropagation::set rewrites a process-global conv workspace cap that keys the cuDNN plan cache (medium, loc +6, S)
- `opennn/neural_network/forward_propagation.h:127-140` — Input-staging state is public only because NeuralNetwork and ChatSession hand-roll the staging (medium, loc -10, M)
- `opennn/neural_network/back_propagation.cpp:106-115` — Joint-plan handshake fails silently in both directions (inference-mode lifetimes, arena without offsets) (medium, loc +6, S)
- `opennn/neural_network/layers/clamping_layer.cpp:127-139` — Clamping and Unscaling claim ranks 1-3 but size their state from dimension 0 only (low, loc +1, S)
- `opennn/neural_network/layers/convolutional_layer.cpp:249-263` — Convolutional::set silently demotes unsupported activations and prints to std::cerr (medium, loc -8, S)
- `opennn/neural_network/operators/batch_norm_operator.cpp:123-135` — set_parameters_random/glorot throw when the network's states live on the device (medium, loc -1, S)
- `opennn/response_optimization/response_optimization.h:84-103` — Objectives encodes bool/Index/sense per objective as floats in three 2xN matrices plus three vectors (medium, loc -15, M)
- `opennn/response_optimization/response_optimization.h:240-242` — Forecasting path (fixed_history, time_roles, combine_input) is unreachable: no setter exists (medium, loc -75, S)
- `opennn/core/tensor_types.h:192-199` — Shape(size_t rank, Index value) has zero callers and is the vector(n, v) trap next to Shape{a, b} (medium, loc -8, S)
- `opennn/neural_network/neural_network.h:126-142` — noexcept accessors call validators that throw (get_parameters_data const, flat_rows, get_inputs_number) (low, loc 0, S)
- `opennn/neural_network/neural_network.h:52` — HostParametersGuard/HostStatesGuard destructors call throwing uploads (implicit noexcept -> terminate) (low, loc +6, S)
- `opennn/core/device_backend.h:363-365` — get_compute_stream() resolves to the active lane; capture/sync points assume lane 0 without checking (low, loc +4, S)
- `opennn/training_strategy/training_strategy.cpp:123-147` — TrainingStrategy::to_JSON/save dereference loss and optimizer without a check (low, loc +2, S)
- `opennn/model_selection/growing_neurons.h:46-92` — GrowingNeurons re-declares InputsSelection's whole configuration block (7 knobs, 8 setters, save/load) (medium, loc -28, M)
- `opennn/neural_network/standard_networks.cpp:478-492` — YoloNetwork silently ignores head_style/reg_max/use_sppf/model_size for unsupported combinations (medium, loc +12, S)
- `opennn/neural_network/standard_networks.cpp:1269-1272` — Seven trivial constructors defined out-of-line (Transformer, TextGenerationNetwork, Qwen3, Bert, BertForSequenceClassification) (low, loc -14, S)

## Structural splits

- `opennn/neural_network/forward_propagation.cpp:124-847` — ForwardPropagation::set is 724 lines with four member-independent seams (medium, loc +12, M)
- `opennn/dataset/tabular_dataset.cpp:1618-2330` — read_csv is 712 lines with six self-contained seams (medium, loc +12, L)
- `opennn/neural_network/model_expression.cpp:713-1672` — ModelExpression::get_expression_c_embedded is a single 960-line function (medium, loc 0, L)
- `opennn/neural_network/model_expression.cpp:713-1672` — get_expression_c_embedded: 960-line function with ~155 lines of constant C runtime text inlined (medium, loc -25, M)
- `opennn/neural_network/standard_networks.cpp:460-1192` — YoloNetwork constructor is 732 lines with a verbatim 13-line duplicate and PRIOR_BIAS defined three times (medium, loc -20, M)
- `opennn/training_strategy/optimizer.cpp:1780-2277` — train_epoch and evaluate_epoch are the same epoch skeleton written twice (~80 lines removable with one driver) (medium, loc -45, M)
- `opennn/neural_network/operators/batch_norm_operator.cpp:626-843` — BatchNormalizationOperator::apply_delta_gpu is a 218-line function with four obvious seams (medium, loc +6, M)
- `opennn/training_strategy/loss.cpp:807-968` — MSVC compiles the v8 TAL gradient kernel at /Od, and that kernel is the GPU training path too (medium, loc -4, S)
- `opennn/response_optimization/response_optimization.cpp:2323-2607` — perform_response_optimization is a 283-line function with four nested lambdas and two save/restore layers (medium, loc +10, M)
- `opennn/neural_network/layers/grouped_query_attention_layer.cpp:898-1250` — GQA project→qk_norm→rope→attend→o_proj pipeline is written four times (CPU/GPU × batch==1/batch>1) (medium, loc -50, M)

## Build, tests and CI

- `CMakeLists.txt:93-134` — OPENNN_HAS_CUDA is a non-FORCE cache set: a reconfigure keeps the stale CUDA decision (medium, loc -2, S)
- `.github/workflows/tinyml-parity.yml:12-35` — tinyml-parity workflow path filters point at files that no longer exist (medium, loc 0, S)
- `CMakeLists.txt:75-80` — Global -Wno-unused-result silences every [[nodiscard]] in the library on the CI compilers (medium, loc -2, S)
- `tests/CMakeLists.txt:12-26` — Tests link bare `gtest gtest_main`, which only exist in the FetchContent path, and gtest_main is redundant (medium, loc -1, S)
- `examples/airfoil_self_noise/main.cpp:32` — airfoil_self_noise hard-codes Device::CUDA and therefore throws on every CPU-only build (medium, loc 0, S)
- `tests/neural_network/operators:1` — 24 library .cpp files have no mirrored test; 9 of them are never even included from tests/ (medium, loc +400, L)
- `tests/neural_network/memory_audit_test.cpp:4-149` — MemoryAudit: seven assertion-free 'TEMPORARY DRIVER' tests run in every suite invocation (medium, loc 0, S)
- `tests/neural_network/qwen3_network_test.cpp:372-380` — Nine CUDA test files set Device::CUDA with no runtime device guard, unlike the four that GTEST_SKIP (low, loc +40, S)
- `opennn/core/string_utilities.cpp:115` — Library .cpp files compile only through the forced-include PCH (<ranges> etc.); clang-tidy/IDEs without it fail (low, loc +40, M)
- `opennn/core/statistics.cpp:452-454` — No .clang-tidy/.clang-format/.editorconfig; a first clang-tidy pass on 5 core files yields real narrowing and rounding hits plus known-noise categories to disable (low, loc +25, S)
- `opennn/CMakeLists.txt:241-267` — Arch/warning/optimisation flags are set twice (root add_compile_options and opennn target) and partly contradict (low, loc -30, S)
- `opennn/CMakeLists.txt:169-180` — GCC-only raw `gomp pthread` OpenMP linking forces an export special-case and leaves consumer pragmas serial (low, loc -12, S)
- `examples/CMakeLists.txt:2-48` — examples/CMakeLists.txt: re-declared option, double gating, phantom subdirectory, duplicated example list (low, loc -15, S)
- `opennn/training_strategy/loss.h:216-224` — YoloLambdas positional init shifted when `dfl` was inserted; tests now run with lambda_class = 0 (medium, loc +2, S)
- `opennn/core/memory_pool.cpp:135-165` — find_memory_pool_overlay has no direct test; the only pin is an end-to-end overfit that the code comment says strategy can break (low, loc +35, M)

## Folder scorecard

| Scope | eff. lines | findings | bugs | high | overhead | lines |
|---|---:|---:|---:|---:|---:|---:|
| core: types, tensors, ops | 7909 | 12 | 3 | 0 | 3 | -43 |
| core: io, json, strings, stats | 7909 | 18 | 10 | 2 | 0 | -157 |
| core: device backend, cuDNN frontend | 7909 | 14 | 5 | 2 | 2 | +7 |
| core/cuda: kernels | 3890 | 12 | 2 | 0 | 5 | -23 |
| neural_network: NeuralNetwork, Fwd/BackPropagation | 9987 | 17 | 4 | 1 | 4 | -134 |
| neural_network: model expression export | 9987 | 19 | 7 | 1 | 1 | -388 |
| neural_network: builders, chat | 9987 | 16 | 5 | 1 | 2 | -339 |
| layers: base + simple layers | 8988 | 16 | 8 | 1 | 2 | -91 |
| layers: recurrent, attention, detection | 8988 | 16 | 4 | 1 | 4 | -205 |
| operators: base + simple | 6746 | 14 | 7 | 2 | 2 | +4 |
| operators: attention, tokenizer, cuDNN RNN, C2PSA | 6746 | 12 | 3 | 0 | 5 | -11 |
| dataset: base, tabular, batch, correlations | 10140 | 23 | 10 | 1 | 3 | -166 |
| dataset: image, language, time series, YOLO | 10140 | 14 | 5 | 0 | 3 | -119 |
| training_strategy: optimizers | 6972 | 18 | 5 | 0 | 5 | -72 |
| training_strategy: loss, error functions | 6972 | 19 | 5 | 1 | 5 | -194 |
| model_selection, testing_analysis, registry | 1682 | 21 | 10 | 1 | 3 | -80 |
| response_optimization | 4189 | 20 | 7 | 0 | 3 | -112 |
| cross-cutting: boilerplate sweep |  | 14 | 0 | 0 | 0 | -542 |
| cross-cutting: build, tests, CI |  | 28 | 5 | 2 | 0 | -614 |
| cross-cutting: public API |  | 16 | 2 | 0 | 1 | -34 |
| round 2: batch pipeline, gather, optimizer kernels |  | 14 | 2 | 1 | 7 | -46 |
| round 2: duplicated kernels |  | 8 | 1 | 0 | 2 | -31 |
| round 2: arena planner, propagation structs (unverified) |  | 10 | 1 | 1 | 3 | +8 |
| round 2: set() vs compile() device ordering (unverified) |  | 8 | 4 | 3 | 1 | +8 |

## Execution plan

0. Today: unbreak the MSVC build, restore the three CPU nonlinearities with negative-input tests, settle the conv activation policy, add the CPU build to CI.
1. Correctness: high bugs, then medium by area, each with its named test (~40 PRs of 1–30 lines).
2. Mechanical line reduction: out-of-line accessors per folder, `add_layer` returning `Index`, JSON fallback overloads, defaults into member initializers, shared test fixtures, redundant `Configuration::set`, dead forecasting/Darknet-v11/blank_cuda code (~-2,000 lines).
3. Hot paths, benchmark-gated: public inference overload + resident default, validation tail context and per-batch sync, OpenMP CPU activations, conv CPU scratch, YOLO v8 single pass, CE3d kernel, Addition copy pass.
4. Structural splits, one seam per commit.
5. Tests and tooling: operator tests for the nine never-included files, CPU/GPU parity tests per drifting twin, `.clang-tidy`, CI path filter, `GTest::gtest`, drop `-Wno-unused-result`.

## All findings

| id | sev | kind | location | title | lines | eff/risk | verdict |
|---|---|---|---|---|---:|---|---|
| nn-builders-chat-1 | high | bug | `opennn/neural_network/standard_networks.cpp:1951-1983` | load_darknet_backbone_v11 targets c11_* labels the builder no longer emits; always loads 0 layers | -56 | S/medium | confirmed |
| r2-arena-planner-and-propagation-structs-1 | high | bug | `opennn/neural_network/forward_propagation.cpp:1165-1190` | CPU valid-length record is frozen after the first forward pass; later padded batches use stale masks | -8 | S/low | unverified |
| nn-expression-2 | high | bug | `opennn/neural_network/model_expression.cpp:1993-2008` | Logarithm scaler/unscaler exports to Python (NameError) and JavaScript (ReferenceError) | -3 | S/low | confirmed |
| core-device-1 | high | bug | `opennn/core/device_backend.cpp:511-528` | CudaBlockCache::give can throw from Buffer destructors -> std::terminate masks the real CUDA error | -2 | S/low | confirmed |
| core-utils-2 | high | bug | `opennn/core/string_utilities.cpp:147-216` | Quoted-field tokenizer deletes every ',' and ';' inside quotes regardless of the separator | -2 | S/medium | confirmed |
| core-device-2 | high | bug | `opennn/core/device_backend.cpp:1214-1225` | set_threads_number destroys the ThreadPool that tensor_operations caches for the process lifetime (UAF) | -1 | S/low | confirmed |
| core-utils-1 | high | bug | `opennn/core/string_utilities.h:27-58` | Apple from_chars shim calls itself for integral T: infinite recursion on macOS builds | 0 | S/low | confirmed |
| selection-testing-1 | high | bug | `opennn/testing_analysis/testing_analysis.cpp:264-292` | calculate_errors divides the Minkowski error by the unrelated batch_size member (default 0 -> +inf) | 0 | S/low | confirmed |
| r2-set-vs-compile-device-ordering-2 | high | bug | `opennn/core/tensor_operations.cpp:1275-1296` | Bias-free fused-ReLU Dense on CPU skips the ReLU: output is the raw pre-activation | +1 | S/low | confirmed |
| r2-set-vs-compile-device-ordering-1 | high | bug | `opennn/neural_network/layers/dense_layer.cpp:199-233` | Fused GELUTanh Dense on CPU never writes its Output slot: the layer returns zeros | +2 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-1 | high | bug | `opennn/training_strategy/optimizer.cpp:1606-1610` | run_graph_epoch dereferences a null pipeline slot when grouped slots exist but a post_batch_callback is set | +2 | S/low | confirmed |
| layers-b-1 | high | bug | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:983-1012` | LSTM on CUDA has no FP32 guard: BF16 networks feed BF16 slots to CUDNN_DATA_FLOAT descriptors | +4 | S/low | partial |
| nn-core-1 | high | bug | `opennn/neural_network/neural_network.cpp:1002-1017` | set_parameters / load_parameters_binary on a released fp32 master overflow the compact bf16 mirror | +5 | S/low | confirmed |
| training-loss-1 | high | bug | `opennn/training_strategy/error_functions.cpp:201-217` | NormalizedSquaredError drops the total/batch scaling its sibling WSE and the old implementation apply | +6 | S/medium | confirmed |
| operators-a-1 | high | bug | `opennn/neural_network/operators/activation_operator.cpp:30-39` | Convolutional ReLU is never applied on CPU: 'forward_fused' skips the activation but no CPU epilogue exists | +8 | S/low | confirmed |
| layers-a-1 | high | bug | `opennn/neural_network/layers/concatenation_layer.cpp:34-50` | Float-only layers accept a BF16 compute dtype and reinterpret BF16 buffers as float | +12 | S/low | confirmed |
| dataset-a-2 | high | UB | `opennn/dataset/tabular_dataset.cpp:1342-1382` | BinaryFile storage: analysis methods index the empty `data` matrix (null-pointer reads) | +18 | M/low | confirmed |
| operators-a-2 | high | bug | `opennn/neural_network/operators/dropout_operator.cpp:117-127` | Dropout seed is baked into captured CUDA graphs: every replay reuses the same mask | +20 | M/medium | partial |
| xcut-build-tests-5 | high | duplication | `tests/training_strategy/yolo_loss_test.cpp:24-104` | write_bmp_24/TempDir/write_label/write_classes copy-pasted into 7 test files (+yolo example); copies already diverged | -340 | M/low | confirmed |
| xcut-build-tests-8 | high | dead code | `examples/blank/main_cuda.cpp:50-218` | blank_cuda is 230 lines entirely under #if 0 with hard-coded /home/artelnics paths, plus its own CMake/option plumbing | -250 | S/medium | confirmed |
| r2-set-vs-compile-device-ordering-3 | high | build/test | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:372-437` | '#pragma omp simd' in the LSTM CPU kernels breaks every MSVC build (C7660) | +3 | S/low | confirmed |
| nn-expression-1 | medium | bug | `opennn/neural_network/model_expression.cpp:677-701` | emit_c_main injects raw output names into a printf format string (and HTML/PHP literals) | -16 | S/low | partial |
| nn-builders-chat-2 | medium | bug | `opennn/neural_network/standard_networks.cpp:1898-1987` | Both Darknet loaders leak the FILE* when load_darknet_weights throws; header reading duplicated | -14 | S/low | confirmed |
| core-utils-12 | medium | bug | `opennn/core/statistics.cpp:228-255` | variance() lacks the negative-variance clamp its sibling has; three different variance formulas | -6 | S/low | confirmed |
| nn-core-2 | medium | bug | `opennn/neural_network/neural_network.cpp:1500-1501` | to_JSON stages states based on the parameters' device, not the states' device | -6 | S/low | confirmed |
| operators-b-1 | medium | bug | `opennn/neural_network/operators/attention_operator.cpp:1074-1131` | CPU attention backward infers padding from trailing zeros of head 0/row 0 - softmax underflow zeroes real gradients | -6 | S/low | partial |
| dataset-a-3 | medium | bug | `opennn/dataset/tabular_dataset.cpp:2519-2551` | impute_missing_values_interpolate interpolates against a phantom point (sample 0, value 0) | -4 | S/low | confirmed |
| operators-a-3 | medium | bug | `opennn/neural_network/operators/embedding_lookup_operator.cpp:279-285` | A second compile() zeroes the sinusoidal positional-encoding table | -4 | S/low | confirmed |
| r2-duplicated-kernels-across-folders-4 | medium | bug | `opennn/training_strategy/loss.cpp:986-1038` | 16 raw CUDA copy/memset/sync sites outside device::, 13 unchecked; loss.cpp relies on legacy-stream cudaMemcpy | -4 | S/low | partial |
| xcut-build-tests-2 | medium | bug | `CMakeLists.txt:93-134` | OPENNN_HAS_CUDA is a non-FORCE cache set: a reconfigure keeps the stale CUDA decision | -2 | S/low | confirmed |
| dataset-a-4 | medium | bug | `opennn/dataset/tabular_dataset.cpp:619-683` | unuse_collinear_variables leaves input_shape stale; shape/role resync is hand-copied in 5 places | -2 | M/low | partial |
| dataset-b-4 | medium | bug | `opennn/dataset/yolo_dataset.cpp:1505-1510` | YOLO cache accepted with a stale classes_number: .names file is not in the sources hash | -2 | S/low | confirmed |
| nn-expression-3 | medium | bug | `opennn/neural_network/model_expression.cpp:1528-1543` | Embedded Logarithm scaling is unclamped; library computes log(max(x, EPSILON)) | -2 | S/low | confirmed |
| selection-testing-4 | medium | bug | `opennn/testing_analysis/testing_analysis.cpp:775-803` | calculate_multiple_classification_rates writes out of bounds when targets have one column | -2 | S/low | confirmed |
| training-optimizers-1 | medium | bug | `opennn/training_strategy/adaptive_moment_estimation.cpp:142-185` | Adam uses two unrelated step counters when CUDA graphs are on: device step for whole batches, host iteration for the tail | -2 | S/low | confirmed |
| core-utils-4 | medium | bug | `opennn/core/random_utilities.cpp:250-280` | Global mutex-serialized RNG makes OpenMP callers slower than serial and non-reproducible under set_seed | -1 | S/low | partial |
| operators-a-5 | medium | bug | `opennn/neural_network/operators/batch_norm_operator.cpp:123-135` | set_parameters_random/glorot throw when the network's states live on the device | -1 | S/low | confirmed |
| xcut-build-tests-1 | medium | bug | `.github/workflows/tinyml-parity.yml:12-35` | tinyml-parity workflow path filters point at files that no longer exist | 0 | S/low | confirmed |
| xcut-build-tests-3 | medium | bug | `examples/airfoil_self_noise/main.cpp:32` | airfoil_self_noise hard-codes Device::CUDA and therefore throws on every CPU-only build | 0 | S/low | confirmed |
| dataset-b-3 | medium | UB | `opennn/dataset/image_processing.cpp:128-198` | 8-bit BMP: palette sized by biClrUsed but indexed by raw pixel byte (out-of-bounds read) | 0 | S/low | confirmed |
| dataset-a-5 | medium | bug | `opennn/dataset/tabular_dataset.cpp:1822-1823` | read_csv keeps stale sample roles/ids from a previous file (resize instead of assign) | 0 | S/low | confirmed |
| dataset-b-2 | medium | bug | `opennn/dataset/time_series_dataset.cpp:258-275` | impute_missing_values_unuse checks a lags+1 window but targets read past+future rows | 0 | S/low | confirmed |
| selection-testing-6 | medium | bug | `opennn/model_selection/growing_inputs.cpp:72-79` | GrowingInputs::set_maximum_inputs_number dereferences a null training strategy; ModelSelection::load() without a strategy segfaults | 0 | S/low | confirmed |
| nn-builders-chat-4 | medium | bug | `opennn/neural_network/chat.cpp:156-167` | sample_token ignores repetition_penalty when temperature == 0; DecoderSampler applies it | 0 | S/low | confirmed |
| nn-expression-4 | medium | bug | `opennn/neural_network/model_expression.cpp:1905-1962` | JS category selector compares raw option values against sanitized output ids | 0 | S/low | confirmed |
| operators-a-4 | medium | bug | `opennn/neural_network/operators/batch_norm_operator.cpp:292-298` | Running variance uses the biased estimator on CPU but the unbiased one on GPU | 0 | S/medium | confirmed |
| response-opt-4 | medium | bug | `opennn/response_optimization/response_optimization.cpp:82-97` | Univariate EqualTo on an output is filtered with a 1e-6 absolute band, so it is effectively infeasible | 0 | S/low | confirmed |
| selection-testing-3 | medium | UB | `opennn/testing_analysis/testing_analysis.cpp:869-873` | Matthews correlation multiplies four Index counts; signed overflow at ~110k balanced testing samples | 0 | S/low | confirmed |
| core-device-3 | medium | bug | `opennn/core/cuda/cudnn_frontend_utilities.h:394-418` | Plan-cache key omits OPENNN_SDPA_AUTOTUNE: attention autotune silently never runs once the cache is warm | +2 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-2 | medium | bug | `opennn/dataset/batch.cpp:269-272` | Batch::~Batch calls a throwing CUDA sync; after a sticky CUDA error unwinding ends in std::terminate | +2 | S/low | confirmed |
| dataset-a-1 | medium | bug | `opennn/dataset/field_parsing.cpp:29-35` | CsvReader trims tabs/spaces from every line, so TSV/space files lose leading/trailing empty fields | +2 | S/low | partial |
| dataset-a-6 | medium | bug | `opennn/dataset/tabular_dataset.cpp:942-964` | Input-input correlations include unused (None) samples while input-target correlations exclude them | +2 | S/low | partial |
| selection-testing-2 | medium | bug | `opennn/model_selection/growing_neurons.cpp:211-230` | GrowingNeurons never reports MaximumEpochs: exhausting the loop leaves stopping_condition empty and elapsed_time blank | +2 | S/low | confirmed |
| layers-a-2 | medium | bug | `opennn/neural_network/layers/embedding_layer.h:26-29` | Embedding::set_input_shape is accepted for rank 1 but silently changes nothing | +2 | S/low | partial |
| layers-b-3 | medium | UB | `opennn/neural_network/layers/recurrent_layer.cpp:103-134` | Recurrent back_propagate indexes scratch slots that do not exist for a frozen layer | +2 | S/low | confirmed |
| nn-expression-5 | medium | bug | `opennn/neural_network/model_expression.cpp:2101-2107` | Python emitter bypasses fix_names: unnamed inputs become `variable`, and fix_names never dedupes | +2 | S/low | confirmed |
| nn-builders-chat-3 | medium | bug | `opennn/neural_network/standard_networks.cpp:483-484` | DarknetTinyV3+FPN accepts 9 anchors but is 2-head: 6 anchors land on a 3-anchor logits conv | +2 | S/low | confirmed |
| response-opt-3 | medium | bug | `opennn/response_optimization/response_optimization.cpp:1139-1154` | Domain::reshape pins Binary scalar inputs to 1 whenever any nearby point had a 1 | +2 | S/medium | confirmed |
| xcut-build-tests-4 | medium | UB | `examples/yolo/main.cpp:364-369` | yolo example leaves std::cout pointing at a destroyed TeeBuf on any exception path | +3 | S/low | confirmed |
| core-utils-5 | medium | UB | `opennn/core/json.cpp:226-239` | JSON number dump casts double to long long before range check; NaN/inf become unparsable tokens | +3 | S/low | confirmed |
| core-utils-3 | medium | UB | `opennn/core/statistics.cpp:690-702` | calculate_rank sorts with a NaN-unsafe comparator; NaN correlations reach it | +3 | S/low | confirmed |
| nn-core-3 | medium | bug | `opennn/neural_network/neural_network.cpp:2632-2642` | upload_parameters_bf16_inference silently migrates a CPU-configured network's parameters to the GPU | +3 | S/low | confirmed |
| core-device-4 | medium | bug | `opennn/core/cuda/cudnn_frontend_utilities.h:376-391` | plan_cache_directory uses the throwing temp_directory_path: a bad TMP disables the cuDNN frontend for the process | +4 | S/low | confirmed |
| layers-a-14-extra-1 | medium | bug | `opennn/neural_network/standard_networks.cpp:494-580` | YoloNetwork with BodyActivation::SiLU builds Identity convolutions on non-V8 backbones | +4 | S/medium | verifier-added |
| training-optimizers-7 | medium | UB | `opennn/training_strategy/optimizer.h:52-74` | set_display_period(0) / set_validation_period(0) / JSON DisplayPeriod 0 cause integer modulo by zero | +4 | S/low | confirmed |
| r2-set-vs-compile-device-ordering-5 | medium | bug | `opennn/neural_network/layers/dense_layer.cpp:295-302` | DReLU wiring is reset one-sided: reconfiguring the consumer after compile drops the producer's ReLU backward | +6 | S/low | unverified |
| core-kernels-1 | medium | bug | `opennn/core/cuda/kernel_attention.cu:677-801` | GPU sampler silently ignores every logit beyond index 262,144 (no host check) | +8 | S/low | partial |
| operators-b-2 | medium | bug | `opennn/neural_network/operators/attention_operator.cpp:653-787` | CPU causal/dropout attention ignores exported sequence lengths the CUDA path honours - CPU/GPU divergence after layer 1 | +10 | S/low | partial |
| xcut-boilerplate-1 | medium | boilerplate | `opennn/dataset/dataset.cpp:227-235` | 115 strictly trivial getters/setters still defined out-of-line in 25 .cpp files (~400 lines) | -350 | M/low | partial |
| xcut-build-tests-7 | medium | boilerplate | `tests/test.cpp:20-27` | 177 Configuration::set calls in tests are redundant with the per-test listener reset | -174 | S/low | partial |
| nn-expression-13 | medium | design | `opennn/neural_network/model_expression.cpp:589-2187` | Four per-neuron exporters repeat one traversal; a LanguageSyntax-driven emitter removes ~180 lines | -120 | L/medium | partial |
| xcut-build-tests-13 | medium | dead code | `tests/numerical_derivatives.cpp:155-240` | Numerical Hessian helper returns a zero matrix; its only consumer asserts nothing; input-deltas helper has no callers | -95 | S/low | confirmed |
| nn-builders-chat-6 | medium | duplication | `opennn/neural_network/standard_networks.cpp:460-1190` | YoloNetwork ctor (731 lines) duplicates SPPF, FPNv8 prior-bias, v8 det head, class-activation, PAN/neck blocks | -90 | L/medium | confirmed |
| nn-expression-8 | medium | duplication | `opennn/neural_network/model_expression.cpp:896-1019` | 124-line embedded scaler switch re-derives slope/offset that core/scaling.h scaling_affine() provides | -85 | M/low | confirmed |
| nn-core-9 | medium | boilerplate | `opennn/neural_network/neural_network.cpp:513-2349` | About 108 lines of one-expression wrappers defined out-of-line in neural_network.cpp | -85 | S/low | confirmed |
| response-opt-2 | medium | dead code | `opennn/response_optimization/response_optimization.h:240-242` | Forecasting path (fixed_history, time_roles, combine_input) is unreachable: no setter exists | -75 | S/medium | confirmed |
| xcut-build-tests-6 | medium | duplication | `tests/neural_network/int8_inference_test.cpp:22-110` | Qwen3/INT8 test helpers are still duplicated verbatim between two files | -75 | S/low | confirmed |
| nn-builders-chat-5 | medium | boilerplate | `opennn/neural_network/neural_network.h:70` | add_layer returns void: 71 'x = get_layers_number() - 1' lines in standard_networks.cpp (134 repo-wide) | -71 | M/low | confirmed |
| xcut-build-tests-23 | medium | duplication | `examples/yolo/main.cpp:1141-1240` | yolo example duplicates its FPN-head collection/decode block, GtBox, and box-to-pixel math inside one 1818-line main | -70 | M/low | confirmed |
| nn-builders-chat-8 | medium | duplication | `opennn/neural_network/chat.cpp:145-231` | Two parallel sampling implementations (sample_token_with_workspace vs DecoderSampler::sample_host) with diverging semantics | -70 | M/medium | partial |
| xcut-boilerplate-3 | medium | boilerplate | `opennn/core/json.cpp:608-634` | 73 two-line `if (has) read_json` guards exist only because read_json_* has no fallback parameter | -65 | S/low | confirmed |
| training-loss-6 | medium | overhead | `opennn/training_strategy/error_functions.cpp:383-421` | CE3d GPU forward still uses the serial one-thread-per-token argmax kernel the metrics path already replaced | -60 | M/medium | confirmed |
| nn-expression-7 | medium | duplication | `opennn/neural_network/model_expression.cpp:1234-1278` | Embedded Recurrent/LSTM packing loops are element-wise identity copies of row-major views | -55 | S/low | confirmed |
| xcut-boilerplate-2 | medium | design | `opennn/training_strategy/levenberg_marquardt_algorithm.cpp:31-49` | set_default() bodies duplicate or contradict header default member initializers in 7 classes | -55 | S/medium | confirmed |
| dataset-a-11 | medium | duplication | `opennn/dataset/tabular_dataset.cpp:2698-2757` | infer_column_types duplicates infer_dataset_date_format and then calls it anyway | -52 | S/low | confirmed |
| layers-b-4 | medium | duplication | `opennn/neural_network/layers/grouped_query_attention_layer.cpp:898-1250` | GQA project→qk_norm→rope→attend→o_proj pipeline is written four times (CPU/GPU × batch==1/batch>1) | -50 | M/medium | partial |
| core-kernels-3 | medium | duplication | `opennn/core/cuda/kernel_activation.cu:82-200` | Activation fwd/bwd carry three hand-written kernels each; one VecIO<T,vec16<T>> kernel covers both dtypes | -45 | M/medium | confirmed |
| training-optimizers-5 | medium | duplication | `opennn/training_strategy/optimizer.cpp:1780-2277` | train_epoch and evaluate_epoch are the same epoch skeleton written twice (~80 lines removable with one driver) | -45 | M/medium | partial |
| dataset-b-6 | medium | duplication | `opennn/dataset/text_generation_dataset.cpp:43-295` | TextGenerationDataset re-implements the tokenizer's vocabulary counting and rebuilds its lookup map | -35 | M/low | confirmed |
| dataset-b-7 | medium | duplication | `opennn/dataset/yolo_dataset.cpp:558-601` | Two uint8 bilinear resizers (bilinear_resize_uint8 vs blit_resized_into_canvas) with different sampling conventions | -35 | S/medium | confirmed |
| layers-a-8 | medium | duplication | `opennn/neural_network/layers/pooling_layer.h:149-175` | Pooling and Convolutional mirror every geometry field of their operator | -35 | M/medium | confirmed |
| nn-expression-10 | medium | boilerplate | `opennn/neural_network/model_expression.cpp:589-2187` | 44 call sites recompute the same input/output/fixed name vectors across the emitters | -32 | S/low | confirmed |
| dataset-a-12 | medium | duplication | `opennn/dataset/tabular_dataset.cpp:2287-2324` | read_csv's post-parse type refinement re-implements infer_variable_types_from_data; both entry points are unused | -30 | S/medium | confirmed |
| selection-testing-8 | medium | duplication | `opennn/model_selection/growing_neurons.h:46-92` | GrowingNeurons re-declares InputsSelection's whole configuration block (7 knobs, 8 setters, save/load) | -28 | M/medium | confirmed |
| nn-expression-9 | medium | design | `opennn/neural_network/model_expression.cpp:713-1672` | get_expression_c_embedded: 960-line function with ~155 lines of constant C runtime text inlined | -25 | M/low | confirmed |
| r2-duplicated-kernels-across-folders-1 | medium | duplication | `opennn/core/cuda/kernel_common.cuh:304-345` | block_reduce_sum/_sum2 are twin bodies and kernel_losses/kernel_tensor/kernel_embedding re-roll them | -20 | M/low | partial |
| dataset-a-9 | medium | overhead | `opennn/dataset/tabular_dataset.cpp:257-287` | fill_from_binary_cache issues one ReadFile/pread per row, serially, and duplicates read_int32_batch | -20 | M/low | confirmed |
| xcut-boilerplate-5 | medium | design | `opennn/neural_network/standard_networks.cpp:460-1192` | YoloNetwork constructor is 732 lines with a verbatim 13-line duplicate and PRIOR_BIAS defined three times | -20 | M/medium | confirmed |
| layers-b-7 | medium | duplication | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:234-249` | LSTM fused path rebuilds Wcat/Ucat(/bcat) identically in forward and backward, per call | -18 | S/low | confirmed |
| selection-testing-9 | medium | duplication | `opennn/model_selection/growing_inputs.cpp:287-305` | GrowingInputs and GeneticAlgorithm end with the same 15-line install-optimal-inputs tail | -16 | S/low | confirmed |
| operators-b-4 | medium | duplication | `opennn/neural_network/operators/kernel_c2psa.cu:28-110` | C2PSA ships its own one-thread-per-row softmax fwd/bwd kernels; library cuDNN softmax and a shared softmax_backward would replace them and 3 hand loops | -15 | M/medium | partial |
| response-opt-9 | medium | design | `opennn/response_optimization/response_optimization.h:84-103` | Objectives encodes bool/Index/sense per objective as floats in three 2xN matrices plus three vectors | -15 | M/low | confirmed |
| training-loss-2 | medium | overhead | `opennn/training_strategy/loss.cpp:970-1099` | YOLO v8 GPU round-trip runs twice per batch: TAL assignment, D2H of targets and every head repeated | -15 | M/medium | confirmed |
| dataset-a-13 | medium | boilerplate | `opennn/dataset/tabular_dataset.cpp:989-1070` | Scaler methods round-trip enum -> string -> enum at four sites; unscale formulas hand-written despite scaling.h helpers | -12 | S/low | confirmed |
| training-loss-8 | medium | overhead | `opennn/training_strategy/loss.cpp:1373-1376` | Per-batch regularization penalty (full-parameter reduction + cuBLAS host sync) whose result the optimizer overwrites | -12 | S/low | partial |
| r2-arena-planner-and-propagation-structs-2 | medium | design | `opennn/neural_network/forward_propagation.h:127-140` | Input-staging state is public only because NeuralNetwork and ChatSession hand-roll the staging | -10 | M/low | unverified |
| r2-set-vs-compile-device-ordering-8 | medium | build/test | `opennn/neural_network/layers/convolutional_layer.cpp:246-263` | Convolutional constructor demotes GELU/SiLU to Identity with a warning while set_activation_function throws; a test now fails | -10 | S/low | unverified |
| core-types-6 | medium | API | `opennn/core/tensor_types.h:192-199` | Shape(size_t rank, Index value) has zero callers and is the vector(n, v) trap next to Shape{a, b} | -8 | S/medium | confirmed |
| selection-testing-7 | medium | overhead | `opennn/model_selection/genetic_algorithm.cpp:185-238` | GeneticAlgorithm keeps a full parameter vector for every individual of every generation; only the best is ever read | -8 | S/low | confirmed |
| layers-a-14 | medium | design | `opennn/neural_network/layers/convolutional_layer.cpp:249-263` | Convolutional::set silently demotes unsupported activations and prints to std::cerr | -8 | S/medium | confirmed |
| r2-duplicated-kernels-across-folders-3 | medium | duplication | `opennn/neural_network/operators/attention_operator.cpp:100-108` | Softmax single-home: CPU row softmax and three CPU/cuDNN softmax-backward copies belong in tensor_operations | -8 | M/low | partial |
| xcut-boilerplate-4 | medium | API | `opennn/neural_network/layers/convolutional_layer.h:103-108` | Class layout depends on OPENNN_HAS_CUDA: data members wrapped in #ifdef in 5 headers | -4 | S/medium | partial |
| operators-b-5 | medium | overhead | `opennn/neural_network/operators/c2psa_operator.cpp:358-384` | C2PSA CPU back_propagate heap-allocates six Eigen matrices per batch element per call although a backward scratch slot is already planned | -4 | S/low | confirmed |
| training-loss-3 | medium | overhead | `opennn/training_strategy/loss.cpp:807-968` | MSVC compiles the v8 TAL gradient kernel at /Od, and that kernel is the GPU training path too | -4 | S/medium | partial |
| training-optimizers-3 | medium | overhead | `opennn/training_strategy/levenberg_marquardt_algorithm.cpp:278-312` | LM allocates a validation Jacobian (V*outputs x P) and a P x P Hessian that validation never reads | -3 | S/low | confirmed |
| xcut-build-tests-10 | medium | build/test | `CMakeLists.txt:75-80` | Global -Wno-unused-result silences every [[nodiscard]] in the library on the CI compilers | -2 | S/low | confirmed |
| operators-a-6 | medium | overhead | `opennn/neural_network/operators/convolution_operator.cpp:526-533` | CPU convolution allocates and zeroes a workers x weights gradient buffer plus im2col scratch on every call | -2 | M/low | partial |
| training-optimizers-6 | medium | overhead | `opennn/training_strategy/optimizer.cpp:1762-1770` | GPU validation loop host-syncs the compute stream after every batch; the Batch event machinery already covers the hazard | -1 | S/medium | confirmed |
| xcut-build-tests-9 | medium | build/test | `tests/CMakeLists.txt:12-26` | Tests link bare `gtest gtest_main`, which only exist in the FetchContent path, and gtest_main is redundant | -1 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-6 | medium | overhead | `opennn/dataset/batch.cpp:336-348` | BF16 host cast of the whole input batch runs on the GPU-feeding main thread inside upload_to_device_batch_async instead of in the prefetch worker | 0 | S/low | confirmed |
| xcut-boilerplate-7 | medium | design | `opennn/dataset/tabular_dataset.cpp:1618-2330` | TabularDataset::read_csv is a single 713-line function with a 230-line OpenMP region inside | 0 | L/medium | confirmed |
| layers-a-10 | medium | overhead | `opennn/neural_network/layers/addition_layer.cpp:18-26` | AdditionOperator forward spends a full copy pass before the first add | 0 | S/low | confirmed |
| layers-b-6 | medium | overhead | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:728-737` | LSTM CPU backward (H<96) allocates and zero-fills per-thread gradient scratch for omp_get_max_threads() every call | 0 | S/low | confirmed |
| xcut-boilerplate-6 | medium | design | `opennn/neural_network/model_expression.cpp:713-1672` | ModelExpression::get_expression_c_embedded is a single 960-line function | 0 | L/low | confirmed |
| xcut-api-2 | medium | API | `opennn/neural_network/neural_network.h:226-228` | calculate_outputs_resident defaults upload_parameters=true, which discards the CUDA graph every call | 0 | S/medium | confirmed |
| xcut-build-tests-15 | medium | build/test | `tests/neural_network/memory_audit_test.cpp:4-149` | MemoryAudit: seven assertion-free 'TEMPORARY DRIVER' tests run in every suite invocation | 0 | S/low | confirmed |
| r2-arena-planner-and-propagation-structs-3 | medium | overhead | `opennn/neural_network/neural_network.cpp:1202-1217` | calculate_outputs tail tile builds a second ForwardPropagation with its own arena instead of sharing the tile's | +1 | S/low | unverified |
| r2-batch-pipeline-and-device-gather-4 | medium | overhead | `opennn/training_strategy/optimizer.cpp:1761-1770` | Validation loop does a device-wide stream synchronize after every batch to protect pool device buffers; an event on the batch would do | +1 | S/medium | partial |
| dataset-a-7 | medium | overhead | `opennn/dataset/tabular_dataset.cpp:897-909` | Correlation loops re-gather variable columns O(inputs x targets) and O(inputs^2) | +2 | S/low | confirmed |
| training-loss-7 | medium | build/test | `opennn/training_strategy/loss.h:216-224` | YoloLambdas positional init shifted when `dfl` was inserted; tests now run with lambda_class = 0 | +2 | S/low | confirmed |
| layers-b-2 | medium | overhead | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:1096-1127` | LSTM allocates six unused B×T×H gate tensors on CUDA; cuDNN-only scratch allocated on CPU | +4 | S/low | confirmed |
| core-device-5 | medium | overhead | `opennn/core/cuda/cudnn_frontend_utilities.h:679-712` | autotune_with_scratch sizes every scratch tensor as fp32: 2x the real transient on bf16 graphs | +6 | S/low | confirmed |
| dataset-a-8 | medium | overhead | `opennn/dataset/tabular_dataset.cpp:398-421` | Per-batch training scaling runs a per-element switch through scale_value in a strided loop | +6 | M/low | confirmed |
| selection-testing-5 | medium | overhead | `opennn/model_selection/cross_validation.cpp:138-160` | k-fold finalisation re-trains all k folds just to recover an epoch count the selection loop already had | +6 | M/medium | partial |
| r2-arena-planner-and-propagation-structs-4 | medium | API | `opennn/neural_network/back_propagation.cpp:106-115` | Joint-plan handshake fails silently in both directions (inference-mode lifetimes, arena without offsets) | +6 | S/low | unverified |
| nn-core-6 | medium | design | `opennn/neural_network/forward_propagation.cpp:832-836` | ForwardPropagation::set rewrites a process-global conv workspace cap that keys the cuDNN plan cache | +6 | S/medium | confirmed |
| operators-a-7 | medium | design | `opennn/neural_network/operators/batch_norm_operator.cpp:626-843` | BatchNormalizationOperator::apply_delta_gpu is a 218-line function with four obvious seams | +6 | M/low | partial |
| operators-b-6 | medium | overhead | `opennn/neural_network/operators/cudnn_rnn.cpp:230-277` | cuDNN RNN re-derives weight-region pointers on every forward and backward: 2 descriptor create/destroy + N cudnnGetRNNWeightParams per call | +6 | S/low | confirmed |
| response-opt-8 | medium | design | `opennn/response_optimization/response_optimization.cpp:2323-2607` | perform_response_optimization is a 283-line function with four nested lambdas and two save/restore layers | +10 | M/low | confirmed |
| xcut-boilerplate-8 | medium | API | `opennn/response_optimization/response_optimization.cpp:1644-1694` | ResponseOptimization writes 36 unconditional cout messages and has no display flag | +10 | S/low | partial |
| core-types-4 | medium | overhead | `opennn/core/tensor_operations.cpp:1174-1246` | CPU activation forward/backward run single-threaded while sibling ops in the file parallelise at 65536 | +12 | S/low | confirmed |
| dataset-a-10 | medium | design | `opennn/dataset/tabular_dataset.cpp:1618-2330` | read_csv is 712 lines with six self-contained seams | +12 | L/low | confirmed |
| nn-core-8 | medium | design | `opennn/neural_network/forward_propagation.cpp:124-847` | ForwardPropagation::set is 724 lines with four member-independent seams | +12 | M/low | confirmed |
| nn-builders-chat-7 | medium | API | `opennn/neural_network/standard_networks.cpp:478-492` | YoloNetwork silently ignores head_style/reg_max/use_sppf/model_size for unsupported combinations | +12 | S/low | confirmed |
| training-optimizers-2 | medium | overhead | `opennn/training_strategy/optimizer.cpp:2126-2162` | Validation remainder batch allocates a Batch and a ForwardPropagation arena every validation epoch; the allocation guard is disabled to allow it | +12 | M/medium | confirmed |
| r2-batch-pipeline-and-device-gather-3 | medium | overhead | `opennn/training_strategy/optimizer.cpp:2134-2152` | Validation tail builds a Batch (cudaMalloc + cudaMallocHost + event) and a ForwardPropagation arena every epoch, and disables the allocation guard for the whole run | +12 | M/medium | confirmed |
| xcut-api-1 | medium | overhead | `opennn/neural_network/neural_network.h:219-226` | GPU 'buffer-reusing' calculate_outputs still allocates a ForwardPropagation arena per call | +15 | S/low | partial |
| r2-batch-pipeline-and-device-gather-5 | medium | overhead | `opennn/training_strategy/optimizer.cpp:1723-1774` | Default (non-graph) GPU training path has zero H2D/compute overlap: one fixed device slot, copy waits for the previous step | +15 | M/medium | partial |
| response-opt-1 | medium | overhead | `opennn/response_optimization/response_constraints.cpp:1366-1391` | FD output-constraint repair runs a full network forward per row, per pass, per constraint | +25 | M/medium | partial |
| xcut-build-tests-16 | medium | build/test | `tests/neural_network/operators:1` | 24 library .cpp files have no mirrored test; 9 of them are never even included from tests/ | +400 | L/low | confirmed |
| response-opt-5 | low | bug | `opennn/response_optimization/response_constraints.cpp:1145-1172` | GreaterThan/LessThan are strict in the linear filter path but inclusive-with-tolerance elsewhere | -20 | S/low | confirmed |
| dataset-b-9 | low | bug | `opennn/dataset/time_series_dataset.cpp:510-524` | calculate_cross_correlations lacks the lag guards its twin calculate_autocorrelations has | -12 | S/low | confirmed |
| nn-core-7 | low | UB | `opennn/neural_network/neural_network.cpp:1683-1703` | from_JSON lets layers through with no sources; invariants validated in add_layer are not re-established | -11 | S/low | partial |
| core-utils-6 | low | bug | `opennn/core/json.cpp:357-389` | JSON \u escapes do not combine surrogate pairs and duplicate append_utf8 | -10 | S/low | confirmed |
| layers-a-6 | low | bug | `opennn/neural_network/layers/pooling_layer.cpp:558-567` | Pooling/Convolutional apply_input_shape skip the geometry validation their set() performs | -10 | S/low | confirmed |
| layers-a-3 | low | bug | `opennn/neural_network/layers/scaling_layer.cpp:245-264` | Scaling/Unscaling/Flatten apply_input_shape reset the label (and Scaling's statistics) | -8 | S/low | partial |
| training-loss-14 | low | bug | `opennn/training_strategy/error_functions.cpp:275-305` | CPU cross-entropy masks NaN/inf to 10.0 and clamps outputs; GPU and the CPU gradient do neither | -4 | S/low | confirmed |
| layers-b-11 | low | bug | `opennn/neural_network/layers/detection_layer.cpp:61-76` | Detection uploads anchors with a raw, unchecked cudaMemcpyAsync from a pageable local vector | -3 | S/low | confirmed |
| nn-expression-6 | low | bug | `opennn/neural_network/model_expression.cpp:2230-2242` | replace_reserved_keywords duplicates names that start with '$' | -3 | S/low | confirmed |
| xcut-build-tests-11-extra-1 | low | bug | `CMakeLists.txt:34-44` | LTO block references undefined LIBOMP_* variables and reports 'OpenMP not found' when IPO is unsupported | -2 | S/low | verifier-added |
| core-utils-17 | low | bug | `opennn/core/io_utilities.cpp:446-455` | FileWriter::open calls create_directories on an empty parent path; sibling in same file guards it | -2 | S/low | confirmed |
| dataset-b-1 | low | bug | `opennn/dataset/time_series_dataset.cpp:265-275` | impute_missing_values_unuse calls set_sample_role inside an OpenMP loop: data race + O(N^2) cache refresh | -2 | S/low | partial |
| response-opt-7 | low | bug | `opennn/response_optimization/response_optimization.cpp:1172-1200` | filter_feasible_points indexes outputs and the output Domain by variable index, not feature column | -2 | S/low | confirmed |
| core-types-2 | low | bug | `opennn/core/tensor_types.cpp:78-88` | TensorView::fill on host ignores dtype: BF16/INT8 host views overflow their storage | -1 | S/low | partial |
| core-kernels-5 | low | bug | `opennn/core/cuda/kernel_normalization.cu:18-22` | Block-per-row norm kernels and rope_apply_kernel compute row offsets in int; overflow past 2^31 elements | 0 | S/low | partial |
| core-device-7 | low | bug | `opennn/core/device_backend.cpp:567-596` | allocate() turns the growth-guard diagnostic into a cache flush (device sync + cudaFree) before rethrowing | 0 | S/low | confirmed |
| core-types-5 | low | UB | `opennn/core/tensor_operations.cpp:1580-1582` | multiply_gpu computes strides and batch_count in int: signed overflow at 2^31 elements per matrix | 0 | S/low | confirmed |
| core-types-1 | low | bug | `opennn/core/tensor_types.cpp:53-59` | TensorView::fill(0) / Buffer::setZero zero CUDA memory on the legacy stream, unordered with the lane streams | 0 | S/low | partial |
| dataset-a-21 | low | bug | `opennn/dataset/dataset.cpp:548-552` | get_features_number() accumulates Index into an int | 0 | S/low | partial |
| selection-testing-12 | low | bug | `opennn/model_selection/growing_inputs.cpp:104-105` | GrowingInputs ranks candidate inputs by correlation with the first target only | 0 | S/low | confirmed |
| layers-a-7 | low | bug | `opennn/neural_network/layers/kernel_upsampling.cu:56-71` | Upsampling launch size computed in int before the overflow check | 0 | S/low | confirmed |
| r2-set-vs-compile-device-ordering-7 | low | bug | `opennn/neural_network/layers/multihead_attention_layer.cpp:221-240` | apply_input_shape on a configured cross-attention layer overwrites the source sequence length with the query length | 0 | S/low | unverified |
| nn-expression-16 | low | bug | `opennn/neural_network/model_expression.cpp:149-227` | Generated HTML: CSS selectors use attribute `float` instead of `type`; PHP header uses `source` instead of `src` | 0 | S/low | confirmed |
| xcut-api-5 | low | bug | `opennn/neural_network/neural_network.h:126-142` | noexcept accessors call validators that throw (get_parameters_data const, flat_rows, get_inputs_number) | 0 | S/low | confirmed |
| operators-a-10 | low | bug | `opennn/neural_network/operators/layer_normalization_operator.cpp:51-58` | LayerNorm variance computed as E[x^2] - mean^2 in float loses all precision for offset inputs | 0 | S/low | confirmed |
| operators-a-9 | low | bug | `opennn/neural_network/operators/multihead_projection_operator.cpp:185-188` | Projection scratch is reshaped with input_features where the GEMM output width is output_features | 0 | S/low | confirmed |
| operators-b-8 | low | bug | `opennn/neural_network/operators/tokenizer_operator.cpp:317-332` | next_utf8_codepoint returns the masked lead bits (not the lead byte) for a sequence truncated at end of text | 0 | S/low | partial |
| selection-testing-13 | low | UB | `opennn/testing_analysis/testing_analysis.cpp:664-680` | Cumulative gain accumulates 0.05f twenty times; the last bucket index exceeds the sample count for large testing sets, and positives are recounted from zero 20 times | 0 | S/low | confirmed |
| selection-testing-20 | low | bug | `opennn/testing_analysis/testing_analysis.cpp:229-248` | Unchecked preconditions: error-data descriptives dereference before check(); GoodnessOfFitAnalysis::save ignores a failed open | 0 | S/low | partial |
| training-loss-4 | low | bug | `opennn/training_strategy/loss.cpp:982-1200` | Seven raw, unchecked CUDA calls in the YOLO drivers (7 of the 8 in all opennn .cpp files) | 0 | S/low | partial |
| training-loss-5 | low | bug | `opennn/training_strategy/loss.cpp:1034-1039` | v8 delta upload relies on legacy-stream implicit ordering with the compute stream | 0 | S/low | confirmed |
| layers-a-4 | low | bug | `opennn/neural_network/layers/clamping_layer.cpp:127-139` | Clamping and Unscaling claim ranks 1-3 but size their state from dimension 0 only | +1 | S/low | confirmed |
| layers-b-15 | low | bug | `opennn/neural_network/layers/kernel_recurrent.cu:116-126` | rnn_copy_regions_cuda silently drops regions beyond RNN_COPY_MAX_REGIONS | +1 | S/low | partial |
| training-loss-18 | low | UB | `opennn/training_strategy/kernel_losses.cu:301-330` | CE3d backward kernel declares outputs/output_deltas __restrict__ but the caller aliases them in place | +1 | S/low | confirmed |
| training-optimizers-9 | low | bug | `opennn/training_strategy/training_result.cpp:143-145` | write_override_results emits the literal text "QUIET_NAN" into saved result files; get_training_error reads out of bounds on empty history | +1 | S/low | confirmed |
| core-utils-18 | low | bug | `opennn/core/json.cpp:469-481` | Json::parse rejects a UTF-8 BOM, so files re-saved by Windows tools fail to load | +2 | S/low | confirmed |
| core-utils-11 | low | UB | `opennn/core/statistics.cpp:367-492` | histogram()/histogram_centered() with bins_number <= 0 hit std::clamp UB and out-of-range writes | +2 | S/low | confirmed |
| layers-a-5 | low | bug | `opennn/neural_network/layers/convolutional_layer.cpp:115-131` | Same-padding maths yields negative padding when stride exceeds kernel+1 | +2 | S/low | confirmed |
| response-opt-17 | low | bug | `opennn/response_optimization/response_optimization.cpp:2352-2353` | perform_response_optimization/solve_once dereference a null neural_network without a check | +2 | S/low | confirmed |
| training-optimizers-16 | low | bug | `opennn/training_strategy/training_strategy.cpp:123-147` | TrainingStrategy::to_JSON/save dereference loss and optimizer without a check | +2 | S/low | confirmed |
| selection-testing-11 | low | bug | `opennn/model_selection/growing_inputs.cpp:92-95` | Inputs selection permanently switches the user's optimizer display off (and GrowingNeurons does not) | +3 | S/low | confirmed |
| selection-testing-2-extra-1 | low | bug | `opennn/model_selection/growing_neurons.cpp:255-288` | GrowingNeurons JSON round-trip silently drops maximum_epochs | +3 | S/low | verifier-added |
| dataset-a-16 | low | bug | `opennn/dataset/dataset.cpp:256-279` | set_sample_roles(vector<string>) / set_sample_roles(indices) / set_sample_role write unchecked; JSON path can overrun | +4 | S/low | confirmed |
| dataset-a-18 | low | bug | `opennn/dataset/dataset.cpp:810-877` | Preview rows are joined with "," on write and split on "," on read, so cells containing commas do not round-trip | +4 | S/medium | confirmed |
| nn-builders-chat-12 | low | bug | `opennn/neural_network/standard_networks.cpp:1198-1202` | TextClassificationNetwork and 3-arg AutoAssociationNetwork index Shape without rank checks (unchecked operator[]) | +4 | S/low | confirmed |
| response-opt-13 | low | bug | `opennn/response_optimization/network_differential.h:42-48` | NetworkDifferential disagrees with Scaling on constant features, so the analytic Jacobian is rejected for them | +4 | S/low | confirmed |
| response-opt-6 | low | bug | `opennn/response_optimization/response_constraints.cpp:1186-1193` | Multivariate AllowedSet constraints are silently treated as satisfied outside the branching driver | +5 | S/low | confirmed |
| dataset-a-17 | low | bug | `opennn/dataset/tabular_dataset.cpp:2590-2595` | variables_missing_values_number changes meaning from per-variable to per-feature-column | +6 | S/low | confirmed |
| xcut-api-4 | low | bug | `opennn/neural_network/neural_network.h:52` | HostParametersGuard/HostStatesGuard destructors call throwing uploads (implicit noexcept -> terminate) | +6 | S/low | confirmed |
| training-optimizers-14 | low | bug | `opennn/training_strategy/optimizer.cpp:909-934` | Async next-epoch shuffle makes seeded GPU runs with dropout non-reproducible; the comment claims draw order is unchanged | +6 | S/low | confirmed |
| layers-b-5 | low | dead code | `opennn/neural_network/layers/grouped_query_attention_layer.cpp:110-160` | rotary_backward / rotary_backward_cpu / rope_backward_gpu are never called by the library (GQA is inference-only) | -90 | S/medium | partial |
| dataset-a-15 | low | boilerplate | `opennn/dataset/dataset.cpp:222-1015` | ~28 one-line forwarding members defined out-of-line across dataset.cpp / tabular_dataset.cpp | -60 | M/low | partial |
| core-utils-7 | low | duplication | `opennn/core/string_utilities.cpp:303-364` | replace() and replace_all_appearances() are the same function except a hidden '_' rule | -35 | M/medium | confirmed |
| training-loss-9 | low | boilerplate | `opennn/training_strategy/loss.cpp:1428-1442` | Single-head YOLO CPU entry points duplicate the multi-head path, which only differs by an identity copy | -35 | S/low | confirmed |
| core-utils-8 | low | dead code | `opennn/core/string_utilities.cpp:47-77` | Free function tokenize(const string&) has no library caller; tokenize_views is the live variant | -33 | S/medium | confirmed |
| xcut-build-tests-11 | low | boilerplate | `opennn/CMakeLists.txt:241-267` | Arch/warning/optimisation flags are set twice (root add_compile_options and opennn target) and partly contradict | -30 | S/medium | confirmed |
| dataset-a-19 | low | dead code | `opennn/dataset/tabular_dataset.cpp:979-987` | calculate_correlations_rank, get_used_variables_indices and both set_variable_type overloads have no callers in the repo | -30 | S/medium | confirmed |
| dataset-b-5 | low | duplication | `opennn/dataset/yolo_dataset.cpp:1390-1427` | Target-cache writer duplicated in try_rebuild_target_from_boxes and build_cache, with a v8 divergence | -30 | M/low | partial |
| training-loss-17 | low | duplication | `opennn/training_strategy/loss.cpp:741-906` | DFL box decode, DFL target computation and reg_max softmax written three, two and three times | -30 | S/low | confirmed |
| core-kernels-10 | low | boilerplate | `opennn/core/cuda/kernel_attention.cu:313-321` | Launch-scaffolding helper quantified for this scope: ~25 lines over 9 sites, no dispatch_rows_cols needed | -25 | S/low | confirmed |
| layers-b-16 | low | duplication | `opennn/neural_network/layers/recurrent_layer.cpp:733-790` | Recurrent and LongShortTermMemory layer wrappers repeat the same shape/return_sequences/JSON plumbing; Recurrent also mirrors Layer::input_shape | -25 | M/medium | partial |
| layers-a-12 | low | duplication | `opennn/neural_network/layers/unscaling_layer.cpp:106-160` | Unscaling::write_expression is a hand-rolled twin of Scaling's affine export | -25 | M/low | confirmed |
| xcut-build-tests-14 | low | boilerplate | `tests/numerical_derivatives.cpp:43-122` | Four numerical-derivative helpers repeat the same 14-line batch-building preamble | -25 | S/low | confirmed |
| training-loss-11 | low | boilerplate | `opennn/training_strategy/error_functions.cpp:87-125` | Ten hand-written template CUDA stubs, one of them unreachable | -24 | S/low | confirmed |
| core-utils-9 | low | dead code | `opennn/core/statistics.cpp:151-178` | Two Histogram constructors are unused; one re-implements histogram() | -22 | S/medium | partial |
| layers-a-9 | low | boilerplate | `opennn/neural_network/layers/activation_layer.cpp:35-40` | apply_input_shape overrides and set() bodies repeat the rank check the base already enforces | -22 | S/low | partial |
| nn-expression-11 | low | duplication | `opennn/neural_network/model_expression.cpp:623-631` | Activation-body emission loop written four times | -22 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-11 | low | boilerplate | `opennn/training_strategy/adaptive_moment_estimation.cpp:25-61` | Static update_parameters_cuda wrapper + CUDA stub duplicated in Adam and SGD; the Capturable branch already calls the kernel inline | -22 | S/low | partial |
| r2-batch-pipeline-and-device-gather-7 | low | dead code | `opennn/dataset/batch.cpp:35-62` | Batch::set is only ever called by the constructor; its reset/shrink logic (~20 lines) never executes on a live object | -20 | S/medium | confirmed |
| xcut-api-11 | low | design | `opennn/training_strategy/optimizer.h:185-309` | The (batches, input, decoder, target) index tuple is threaded positionally through eight signatures | -20 | M/low | confirmed |
| core-types-7 | low | design | `opennn/core/variable.h:118-148` | Variable ctor/set take role and scaler as strings, with defaults that contradict the in-class initialisers | -18 | S/low | partial |
| operators-b-11 | low | boilerplate | `opennn/neural_network/operators/attention_operator.cpp:276-292` | heads_input/heads_output are pure forwarding wrappers and both SDPA backward exits repeat the same cast-back epilogue | -18 | S/low | confirmed |
| r2-duplicated-kernels-across-folders-6 | low | duplication | `opennn/neural_network/operators/kernel_c2psa.cu:13-67` | C2PSA copy_right and scatter_dx kernels are strided column copies slice_channels/cudaMemcpy2DAsync already express | -18 | S/medium | partial |
| xcut-boilerplate-11 | low | boilerplate | `opennn/training_strategy/error_functions.cpp:87-125` | Residual hand-written CUDA stubs left after the OPENNN_CUDA_STUB pass (3 sites) | -18 | S/low | confirmed |
| training-optimizers-12 | low | boilerplate | `opennn/training_strategy/stochastic_gradient_descent.cpp:67-82` | SGD/QN/LM set_default() re-state defaults that member initializers express (or should) | -18 | S/medium | confirmed |
| selection-testing-10 | low | duplication | `opennn/model_selection/selection_utilities.cpp:21-59` | evaluate_candidate skips on_trial on the fold path, so GrowingInputs and GrowingNeurons each carry a second record-optimum block | -16 | M/medium | confirmed |
| xcut-build-tests-20 | low | boilerplate | `examples/CMakeLists.txt:2-48` | examples/CMakeLists.txt: re-declared option, double gating, phantom subdirectory, duplicated example list | -15 | S/low | confirmed |
| core-utils-13 | low | boilerplate | `opennn/core/statistics.cpp:531-583` | descriptives(matrix, rows, cols) splits one per-column job into two OMP loops and five scratch vectors | -15 | S/low | confirmed |
| xcut-boilerplate-9 | low | duplication | `opennn/dataset/image_dataset.cpp:618-627` | OpenMP first-exception capture block copied four times | -15 | S/low | partial |
| r2-arena-planner-and-propagation-structs-5 | low | duplication | `opennn/neural_network/forward_propagation.cpp:43-50` | ForwardPropagation builds the consumer-edge map a fourth and fifth time; hoist it to NeuralNetwork | -15 | S/low | unverified |
| response-opt-15 | low | API | `opennn/response_optimization/response_constraints.h:168-211` | Header exports nine repair_* entry points with colliding names; four are internal-only | -15 | S/medium | confirmed |
| response-opt-10 | low | duplication | `opennn/response_optimization/response_optimization.cpp:2186-2201` | Finite-difference Jacobian written twice; the validation copy issues 2n single-row forwards per probe | -15 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-12 | low | API | `opennn/training_strategy/kernel_optimizers.cuh:9-18` | adam_update_cuda/sgd_update_cuda declared with runs of unnamed positional floats; each has a near-identical capturable twin | -15 | S/low | confirmed |
| xcut-build-tests-24 | low | API | `opennn/training_strategy/training_strategy.h:33-34` | Every example/test re-implements 'dynamic_cast the optimizer and hope': a typed accessor would remove 13 casts and 8 unchecked derefs | -15 | S/low | confirmed |
| xcut-api-14 | low | boilerplate | `opennn/core/tensor_types.h:74-110` | to_cudnn / to_cuda / type_bytes are three copies of the same switch over TypeInfo | -14 | S/low | confirmed |
| dataset-a-14 | low | boilerplate | `opennn/dataset/dataset.cpp:659-767` | Separator uses a hand-rolled tuple table with four linear searches; check_separators allocates two strings per CSV line | -14 | S/low | confirmed |
| nn-builders-chat-10 | low | overhead | `opennn/neural_network/chat.cpp:446-459` | Non-incremental GenerationParser re-decodes the whole channel per token (O(n^2) per response) | -14 | S/low | confirmed |
| nn-builders-chat-9 | low | duplication | `opennn/neural_network/standard_networks.cpp:342-351` | ResNet and YoloNetwork each define the same add_conv lambda | -14 | S/low | confirmed |
| nn-builders-chat-15 | low | boilerplate | `opennn/neural_network/standard_networks.cpp:1269-1272` | Seven trivial constructors defined out-of-line (Transformer, TextGenerationNetwork, Qwen3, Bert, BertForSequenceClassification) | -14 | S/low | confirmed |
| operators-a-11 | low | API | `opennn/neural_network/operators/convolution_operator.h:63-67` | ConvolutionOperator::set takes ten positional Index arguments, has one caller, and only copies public fields | -13 | S/medium | confirmed |
| xcut-build-tests-12 | low | boilerplate | `opennn/CMakeLists.txt:169-180` | GCC-only raw `gomp pthread` OpenMP linking forces an export special-case and leaves consumer pragmas serial | -12 | S/medium | confirmed |
| core-utils-14 | low | boilerplate | `opennn/core/json.h:106-130` | add_json_field is a three-layer forwarding chain; the templates add nothing | -12 | S/low | confirmed |
| core-utils-16 | low | duplication | `opennn/core/random_utilities.cpp:282-409` | Three fill loops hand-roll what fill_random already abstracts | -12 | S/low | partial |
| core-types-8 | low | duplication | `opennn/core/scaling.h:69-92` | scale_value re-implements the guards and formulas that scaling_affine already encodes | -12 | S/medium | confirmed |
| xcut-boilerplate-12 | low | boilerplate | `opennn/dataset/dataset.cpp:658-680` | Three residual hand-written enum<->string converters outside EnumMap | -12 | S/low | confirmed |
| xcut-boilerplate-10 | low | duplication | `opennn/model_selection/growing_inputs.cpp:311-347` | InputsSelection common JSON fields repeated in GrowingInputs, GeneticAlgorithm and GrowingNeurons | -12 | S/low | partial |
| nn-core-13 | low | duplication | `opennn/neural_network/back_propagation.cpp:60-77` | The consumer-edge map of source_layers is built three times | -12 | S/low | confirmed |
| layers-b-8 | low | overhead | `opennn/neural_network/layers/recurrent_layer.cpp:206-320` | Recurrent CPU forward/backward copy a B×H block per time step instead of using the strided maps directly | -12 | S/low | partial |
| nn-core-12 | low | API | `opennn/neural_network/neural_network.cpp:1749-1767` | Legacy JSON 'Parameters/Values' read is asymmetric and silently truncates on size mismatch | -12 | S/medium | confirmed |
| nn-core-15 | low | dead code | `opennn/neural_network/neural_network.cpp:967-976` | get_layers_number(const string&) and get_layers_number(LayerType) have no callers | -12 | S/medium | confirmed |
| nn-core-14 | low | duplication | `opennn/neural_network/neural_network.h:44-59` | HostParametersGuard and HostStatesGuard are the same RAII struct written twice | -12 | S/low | confirmed |
| operators-b-10 | low | duplication | `opennn/neural_network/operators/tokenizer_operator.cpp:173-216` | encode_sequence overloads and WordLevel tokenize/encode duplicate the same bodies | -12 | S/low | confirmed |
| response-opt-19 | low | boilerplate | `opennn/response_optimization/response_optimization.h:57-82` | Small boilerplate: virtual Domain dtor, UnivariateConstraint ctor, append_rows vs stack_rows, std::function find | -12 | S/low | partial |
| training-loss-12 | low | dead code | `opennn/training_strategy/error_functions.cpp:490-508` | cross_entropy_3d_gradient_device_count CPU fallback is unreachable | -12 | S/medium | confirmed |
| xcut-api-7 | low | dead code | `opennn/training_strategy/optimizer.h:73-80` | Eleven accessors with zero callers in opennn/, tests/, examples/ and docs/benchmarks/ | -12 | S/medium | confirmed |
| core-utils-15 | low | duplication | `opennn/core/string_utilities.cpp:428-453` | env_flag_enabled(name) duplicates env_flag_enabled(name, false) | -11 | S/low | confirmed |
| response-opt-14 | low | boilerplate | `opennn/response_optimization/network_differential.cpp:39-57` | Scaling/Unscaling snapshot branch duplicated although Unscaling derives from Scaling | -11 | S/low | confirmed |
| core-device-8 | low | duplication | `opennn/core/cuda/cudnn_frontend_utilities.h:512-622` | finalize and finalize_attention duplicate the validate/load/build/workspace/store skeleton | -10 | M/medium | partial |
| core-device-11 | low | boilerplate | `opennn/core/device_backend.cpp:1289-1300` | LtMatmulPlan hand-swaps nine members to move into the cache; preference handle leaks on a CHECK_CUBLAS throw | -10 | S/low | confirmed |
| dataset-a-20 | low | duplication | `opennn/dataset/correlations.cpp:47-100` | fit_softmax_correlation and fit_logistic_correlation repeat the dataset/loss/train/correlate sequence | -10 | S/low | partial |
| nn-expression-12 | low | duplication | `opennn/neural_network/model_expression.cpp:1806-1815` | get_expression_javascript re-implements split_expression_lines/prepare_body_lines with drifted rules | -10 | S/low | confirmed |
| nn-expression-15 | low | boilerplate | `opennn/neural_network/model_expression.cpp:759-780` | activation_constant_for is a hand-written enum->string ladder; EnumMap is the sanctioned helper | -10 | S/low | confirmed |
| core-types-9 | low | dead code | `opennn/core/opennn_types.h:312-321` | Dead umbrella aliases and duplicated using-directives in opennn_types.h / tensor_types.h | -9 | S/medium | confirmed |
| core-types-3 | low | overhead | `opennn/core/tensor_types.cpp:44-51` | uses_cuda_fill queries the driver per fill and ignores TensorView::device | -9 | S/medium | confirmed |
| layers-b-9 | low | design | `opennn/neural_network/layers/recurrent_layer.h:21-27` | Recurrent forward slot numbering is declared twice (operator enum and private layer enum) with no link between them | -9 | S/low | confirmed |
| core-device-9 | low | duplication | `opennn/core/device_backend.cpp:193-262` | Compute capability and device presence are queried three different ways with two encodings | -8 | S/low | partial |
| dataset-b-13 | low | duplication | `opennn/dataset/language_dataset.cpp:153-196` | LanguageDataset::read_txt repeats record packing that write_binary_cache already implements | -8 | S/low | partial |
| dataset-b-12 | low | duplication | `opennn/dataset/yolo_dataset.cpp:1902-1908` | Batch seed derivation duplicated verbatim in fill_inputs and fill_targets (must stay identical) | -8 | S/low | confirmed |
| layers-b-13 | low | boilerplate | `opennn/neural_network/layers/detection_v8_layer.cpp:155-174` | DetectionV8 JSON body re-derives input_shape from fields from_JSON already applied and hand-rolls an optional read | -8 | S/low | confirmed |
| nn-expression-19 | low | overhead | `opennn/neural_network/model_expression.cpp:1691-1702` | Name mapping is O(names x lines x line_length): a whole-string replace per name per line | -8 | M/low | confirmed |
| nn-core-11 | low | boilerplate | `opennn/neural_network/neural_network.cpp:1602-1767` | from_JSON: capture-free lambda, two identical Items walks, and a legacy block with its own error policy | -8 | S/low | confirmed |
| operators-b-12 | low | boilerplate | `opennn/neural_network/operators/cudnn_rnn.h:37-44` | CudnnRnnConfig is a one-field struct wrapping cudnnRNNMode_t | -8 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-8 | low | duplication | `opennn/training_strategy/optimizer.cpp:1494-1495` | The H2D event handshake is re-implemented inline five times in optimizer.cpp instead of using Batch's helpers; one site bypasses record_h2d_done | -8 | S/low | confirmed |
| training-optimizers-4 | low | overhead | `opennn/training_strategy/quasi_newton_method.cpp:192-252` | QN builds a full validation BackPropagation (gradient + delta arena) just to hold three metric floats | -8 | S/low | confirmed |
| selection-testing-17 | low | boilerplate | `opennn/testing_analysis/testing_analysis.cpp:55-59` | Constructor/initializer boilerplate: TestingAnalysis ctor out-of-line, NeuronsSelectionResult defaults overridden by its ctor, pch.h double include guard | -7 | S/low | partial |
| r2-duplicated-kernels-across-folders-2 | low | duplication | `opennn/core/cuda/kernel_tensor.cu:55-70` | Column-sum-over-rows reduction planned twice with opposite zeroing contracts; callers must pre-zero | -6 | S/low | partial |
| core-utils-10 | low | design | `opennn/core/statistics.h:16-32` | Descriptives member initializers are dead and contradict the constructor; `name` is unused | -6 | S/medium | confirmed |
| r2-batch-pipeline-and-device-gather-14 | low | design | `opennn/dataset/batch.cpp:119-139` | For GPU batches input_views_host_cache wraps a CUDA pointer in a TensorView labelled Device::CPU | -6 | S/low | confirmed |
| xcut-api-12 | low | design | `opennn/dataset/dataset.h:272-288` | 'int contiguous = -1' tri-state is converted from and back to optional<bool> at both ends | -6 | S/low | confirmed |
| nn-builders-chat-13 | low | design | `opennn/neural_network/chat.cpp:1133-1140` | chat.cpp resizes ForwardPropagation::staged_input_storage directly (only external user of the staging internals) | -6 | S/low | confirmed |
| layers-a-15 | low | boilerplate | `opennn/neural_network/layers/dense_layer.cpp:326-378` | Dense::set duplicates set_activation_function's Softmax-to-Sigmoid demotion and skips the label on empty shapes | -6 | S/low | confirmed |
| operators-a-12 | low | duplication | `opennn/neural_network/operators/dropout_operator.cpp:102-113` | Dropout CPU backward duplicates the forward's mask loop, serially, re-evaluating delta.size() per iteration | -6 | S/low | confirmed |
| nn-builders-chat-11 | low | design | `opennn/neural_network/standard_networks.cpp:1735-1756` | get_tokenizer_layer uses try/catch around get_layer(label) as control flow; chat.cpp:911 calls get_layer_index for its throw | -6 | S/low | confirmed |
| selection-testing-15 | low | boilerplate | `opennn/registry.cpp:60-75` | registry: construct_layer<T> duplicates construct<Layer,T>; two hand-rolled 'Component not found' throws bypass throw_if and mislabel vision-disabled layers | -6 | S/low | confirmed |
| selection-testing-14 | low | overhead | `opennn/testing_analysis/testing_analysis.cpp:497-560` | ROC analysis makes 100 full passes over the testing set plus two redundant confusion passes | -6 | M/low | partial |
| training-loss-15 | low | design | `opennn/training_strategy/loss.cpp:1904-1971` | Loss JSON serialises regularization twice and the error name three times, with asymmetric readers | -6 | S/medium | confirmed |
| training-optimizers-17 | low | duplication | `opennn/training_strategy/optimizer.cpp:1104-1112` | train_full_batch re-implements the per-epoch display that display_epoch_results already provides | -6 | S/low | confirmed |
| training-optimizers-11 | low | dead code | `opennn/training_strategy/optimizer.h:46-321` | Dead or duplicate Optimizer members: restore_best has no setter, print() has no override or caller, set_loss duplicates set, get_batch_workers_number ignores its argument | -6 | S/medium | confirmed |
| training-optimizers-18 | low | design | `opennn/training_strategy/optimizer.h:205-208` | display_epoch_results takes 9 positional scalars including 3 adjacent bools; train() keeps the same four floats as loose locals | -6 | S/low | confirmed |
| training-optimizers-15 | low | design | `opennn/training_strategy/training_result.h:23-40` | OptimizerData is a union of every optimizer's private scalars, declared in training_result.h | -6 | M/medium | confirmed |
| operators-a-14 | low | boilerplate | `opennn/neural_network/operators/batch_norm_operator.cpp:115-121` | link_states hand-rolls what the promoted link_views helper already does | -5 | S/low | confirmed |
| xcut-api-9 | low | duplication | `opennn/training_strategy/adaptive_moment_estimation.h:27` | set_batch_size duplicated in Adam and SGD while batch_size/get_batch_size live on the base; forces downcasts | -5 | S/low | confirmed |
| xcut-build-tests-19 | low | dead code | `tests/CMakeLists.txt:18-22` | tests/pch.cpp and opennn/pch.cpp are vestigial one-line TUs; tests/CMakeLists exists partly to exclude one | -5 | S/low | confirmed |
| core-types-10 | low | design | `opennn/core/tensor_operations.cpp:1117-1124` | multiply accepts the flattened rank-3 x rank-2 product only on CUDA, so linear_forward_transposed keeps a CPU-only Eigen branch | -4 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-9 | low | overhead | `opennn/dataset/dataset.cpp:1083-1087` | Device gather re-allocates DeviceGather::row_indices on every fill and memcpy's it into the pinned index buffer at upload time | -4 | S/low | confirmed |
| dataset-b-8 | low | overhead | `opennn/dataset/yolo_dataset.cpp:1292-1308` | Cold YOLO start hashes every image and label file three times and lists the directory five times | -4 | S/low | confirmed |
| r2-arena-planner-and-propagation-structs-6 | low | API | `opennn/neural_network/forward_propagation.h:58-74` | ForwardPropagation ctor and set() disagree on arity/position of the same positional bool | -4 | S/low | unverified |
| layers-a-13 | low | boilerplate | `opennn/neural_network/layers/embedding_layer.cpp:59-91` | Embedding and Normalization3d re-parse dimension fields the base already applied; Embedding writes OutputDimensions twice | -4 | S/low | confirmed |
| nn-core-16 | low | overhead | `opennn/neural_network/neural_network.cpp:1464-1495` | Line-search forward_propagate(inputs, parameters, fp) does three full parameter transfers and three link_parameters walks per evaluation | -4 | S/medium | partial |
| response-opt-12 | low | design | `opennn/response_optimization/response_optimization.cpp:741-757` | resolve_cardinality_columns keeps a process-wide static warning set and an unused parameter | -4 | S/low | confirmed |
| training-optimizers-10 | low | design | `opennn/training_strategy/quasi_newton_method.cpp:226-240` | QN records the pre-update training error in the history but displays and stops on the post-update one | -4 | S/medium | confirmed |
| xcut-build-tests-18 | low | build/test | `tests/pch.h:5-8` | tests/pch.h force-defines NDEBUG and re-defines Eigen macros the opennn target already exports | -4 | S/low | partial |
| xcut-build-tests-21 | low | boilerplate | `opennn/CMakeLists.txt:6-9` | OneDrive '-DESKTOP-' conflict-copy filters are machine-specific hygiene baked into three GLOBs | -3 | S/low | confirmed |
| r2-duplicated-kernels-across-folders-8 | low | duplication | `opennn/core/cuda/kernel_pooling.cuh:32-41` | MaxPoolGeometry::decompose re-implements kernel_common's nhwc_decompose with a channel-group factor | -3 | S/low | partial |
| core-device-14 | low | boilerplate | `opennn/core/device_backend.cpp:35-41` | get_op_tensor_add_descriptor takes the lane mutex and touches the cuDNN handle for a ctor-created descriptor | -3 | S/low | confirmed |
| dataset-b-14 | low | design | `opennn/dataset/yolo_dataset.cpp:1744-1779` | set_multi_scale_heads and set_v8_mode update target_shape/target_record_floats but only one updates the target Variable | -3 | S/low | partial |
| selection-testing-18 | low | boilerplate | `opennn/model_selection/genetic_algorithm.cpp:494-508` | GeneticAlgorithm loop leftovers: per-generation history growth on a pre-sized history and a duplicated role restore | -3 | S/low | confirmed |
| layers-b-14 | low | overhead | `opennn/neural_network/layers/recurrent_layer.cpp:405-411` | Recurrent cuDNN path writes y into HiddenStates and then copies B×T×H into Output; LSTM writes y into Output directly | -3 | S/medium | confirmed |
| training-loss-13 | low | API | `opennn/training_strategy/error_functions.cpp:350-375` | Minkowski pair detects GPU by two ad-hoc signals instead of input.is_cuda(), and p is unvalidated | -3 | S/low | confirmed |
| r2-batch-pipeline-and-device-gather-13 | low | overhead | `opennn/training_strategy/stochastic_gradient_descent.cpp:131-148` | SGD CPU update forks an OpenMP region per step unconditionally; Adam guards it with if(parameters_size > 4096) | -3 | S/low | confirmed |
| xcut-build-tests-22 | low | build/test | `.github/workflows/ci.yml:30-51` | CI dependency cache key ignores the files that pin the dependencies; stale '../datasets' comment | -2 | S/low | confirmed |
| core-kernels-4 | low | overhead | `opennn/core/cuda/kernel_pooling.cuh:26-34` | 64-bit div/mod per thread in index decomposition although the element count is already checked_int | -2 | S/low | confirmed |
| selection-testing-16 | low | dead code | `opennn/model_selection/inputs_selection.h:68` | InputsSelection::print() virtual no-op and the NeuronSelection alias have no users in opennn/, tests/, examples/ or docs/ | -2 | S/medium | confirmed |
| nn-builders-chat-14 | low | overhead | `opennn/neural_network/chat.cpp:1124-1131` | Per-token host-to-device 4-byte copies from stack locals force an extra stream sync each step | -2 | S/low | confirmed |
| r2-duplicated-kernels-across-folders-5 | low | overhead | `opennn/neural_network/layers/grouped_query_attention_layer.cpp:329-339` | GQA value-tail zeroing issues one cudaMemsetAsync per batch-head instead of one pitched memset | -2 | S/low | confirmed |
| xcut-api-10 | low | boilerplate | `opennn/neural_network/layers/layer.h:169-181` | Operator::compute_dtype is propagated by hand per layer while weights_dtype is propagated generically | -2 | S/low | confirmed |
| xcut-api-15 | low | design | `opennn/neural_network/layers/layer.h:187-191` | Non-const get_parameter_views()/get_parameter_scales() hand out the layer's view vectors for resizing | -2 | S/medium | partial |
| nn-expression-18 | low | API | `opennn/neural_network/model_expression.cpp:1280-1283` | Recurrent::get_activation_function returns string, LSTM's returns the enum; exporter round-trips through strings | -2 | S/medium | confirmed |
| nn-core-10 | low | dead code | `opennn/neural_network/neural_network.cpp:1448-1451` | Pre-scaled input branch in forward_propagate duplicates passthrough_overrides and uses the wrong index | -2 | S/low | partial |
| operators-b-13 | low | overhead | `opennn/neural_network/operators/tokenizer_operator.cpp:1030-1044` | Byte-level decode does a hash lookup per byte through unordered_map byte_decoder although codepoints are < 324 | -2 | S/low | confirmed |
| response-opt-16 | low | overhead | `opennn/response_optimization/response_optimization.cpp:1842-1871` | reselect_pareto_front computes the full n x n distance matrix twice | -2 | S/low | confirmed |
| training-loss-10 | low | overhead | `opennn/training_strategy/loss.cpp:641-660` | TAL assignment and DFL gradient allocate heap vectors per (sample, ground-truth) pair and per positive cell | -2 | S/low | confirmed |
| xcut-api-8 | low | design | `opennn/training_strategy/optimizer.h:46-48` | Optimizer::set(Loss*) and virtual set_loss(Loss*) are twins; the virtual one is bypassed and uncalled | -2 | S/medium | confirmed |
| core-kernels-11 | low | design | `opennn/core/cuda/flash_attention.cu:93-98` | Flash_bwd_params is default-initialised; set_parameters zeroes only the Flash_fwd_params base | -1 | S/low | confirmed |
| xcut-boilerplate-14 | low | dead code | `opennn/core/opennn_types.h:170` | `using type = float;` declared twice in opennn_types.h (global and inside namespace opennn) | -1 | S/low | confirmed |
| r2-arena-planner-and-propagation-structs-10 | low | boilerplate | `opennn/neural_network/back_propagation.h:44` | BackPropagation declares a virtual destructor but nothing derives from it | -1 | S/medium | unverified |
| operators-a-8 | low | overhead | `opennn/neural_network/operators/pool3d_operator.cpp:72-79` | Max pooling CPU forward heap-allocates a bool array per (sequence, step) via .eval() | -1 | S/low | confirmed |
| r2-set-vs-compile-device-ordering-9 | low | build/test | `AGENTS.md:20-25` | AGENTS.md's CPU check directory has no test binary; no CPU suite has run since the change that broke findings 1-2 | 0 | S/low | unverified |
| core-kernels-12 | low | boilerplate | `opennn/core/cuda/kernel_normalization.cu:1090-1095` | norm_backward_launch calls raw, unchecked cudaMemsetAsync where device::set_zero_async is the sanctioned helper | 0 | S/low | partial |
| core-types-11 | low | overhead | `opennn/core/tensor_operations.cpp:1600-1609` | softmax_gpu and add_gpu build the same cuDNN descriptor two or three times per call | 0 | S/low | confirmed |
| xcut-api-13 | low | API | `opennn/core/tensor_types.h:192-199` | Shape(rank, fill) constructor reads like Shape{a, b}: Shape(2, 3) is [3,3], Shape{2, 3} is [2,3] | 0 | S/medium | confirmed |
| r2-batch-pipeline-and-device-gather-10 | low | overhead | `opennn/dataset/kernel_gather.cu:56-62` | gather_rows_cuda caps column threads at 32, so wide rows (images) launch only batch/8 blocks | 0 | S/low | confirmed |
| xcut-boilerplate-13 | low | design | `opennn/dataset/tabular_dataset.cpp:316-368` | Index/size_t cast churn has doubled since the prior audit (1412 casts); 108 in one file | 0 | S/low | partial |
| selection-testing-19 | low | API | `opennn/model_selection/model_selection.h:43-52` | ModelSelection exposes no way to choose GeneticAlgorithm except through JSON | 0 | S/low | confirmed |
| nn-builders-chat-16 | low | API | `opennn/neural_network/chat.h:58-60` | Public sample_token mutates the caller's probability vector in place without signalling it | 0 | S/low | confirmed |
| r2-arena-planner-and-propagation-structs-7 | low | overhead | `opennn/neural_network/forward_propagation.cpp:746-850` | memory_debug::record arguments are formatted eagerly on every ForwardPropagation construction | 0 | S/low | unverified |
| layers-b-10 | low | API | `opennn/neural_network/layers/long_short_term_memory_layer.cpp:47-57` | RecurrentOperator::set and LongShortTermMemoryOperator::set take the same three Index parameters in different orders | 0 | S/low | confirmed |
| nn-expression-14 | low | API | `opennn/neural_network/model_expression.cpp:328-331` | ModelExpression takes const NeuralNetwork* but const_casts to migrate parameters and relink layers | 0 | S/medium | confirmed |
| response-opt-11 | low | overhead | `opennn/response_optimization/response_constraints.cpp:797-803` | evaluate_rpn heap-allocates its stack per call; nonlinear filter path copies each row twice | 0 | S/low | confirmed |
| training-loss-16 | low | design | `opennn/training_strategy/loss.cpp:29-1238` | YOLO block still inline: 1210 of loss.cpp's 1977 lines (61%) under one #ifndef OPENNN_NO_VISION | 0 | S/low | confirmed |
| training-optimizers-13 | low | overhead | `opennn/training_strategy/optimizer.cpp:956-960` | Validation batch index lists are rebuilt from the same unshuffled indices on every validation epoch | 0 | S/low | confirmed |
| xcut-build-tests-17 | low | design | `tests/neural_network/response_optimization_test.cpp:1` | Test files that violate the one-for-one mirror rule by name or folder | 0 | S/low | partial |
| nn-core-5 | low | overhead | `opennn/neural_network/neural_network.cpp:1311` | Unconditional copy_states_device() per GPU forward rebuilds every operator's state views each batch | +1 | S/low | partial |
| nn-core-17 | low | overhead | `opennn/neural_network/neural_network.cpp:1178-1182` | CPU calculate_outputs computes get_forward_specs(1) on every call even when tiling cannot apply | +1 | S/low | confirmed |
| core-device-13 | low | API | `opennn/core/device_backend.cpp:1147-1160` | lane_stream(int) has no range check: a lane >= MAX_LANES indexes the std::array out of bounds | +2 | S/low | partial |
| layers-b-12 | low | API | `opennn/neural_network/layers/kernel_detection.cu:19-32` | Anchor-based Detection GPU path rejects non-square grids that the CPU path and DetectionV8 (CPU+GPU) accept | +2 | S/low | confirmed |
| nn-core-4 | low | overhead | `opennn/neural_network/neural_network.cpp:1291-1292` | Every forward call re-materialises all parameter specs just to compare one integer | +2 | S/low | partial |
| xcut-api-3 | low | API | `opennn/neural_network/neural_network.h:278-297` | Residency has ~10 public entry points; most are implementation steps with 0-1 external callers | +2 | S/medium | partial |
| xcut-api-6 | low | API | `opennn/neural_network/neural_network.h:152` | get_layer(Index) is unchecked while its sibling get_layer(string) throws 'Layer not found' | +2 | S/low | confirmed |
| operators-a-13 | low | boilerplate | `opennn/neural_network/operators/embedding_lookup_operator.cpp:341-355` | Inline #ifdef islands with a dangling else break the file-wide static _gpu twin + OPENNN_CUDA_STUB pattern | +2 | S/low | confirmed |
| core-device-12 | low | API | `opennn/core/device_backend.h:140-143` | set_conv_workspace_cap(int64_t mode) encodes auto/unlimited/bytes in one undocumented integer | +3 | S/low | confirmed |
| nn-expression-17 | low | duplication | `opennn/neural_network/model_expression.cpp:497-502` | LeakyReLU slope hard-coded as 0.1 in four activation bodies while embedded uses LEAKY_RELU_SLOPE | +3 | S/low | confirmed |
| response-opt-20 | low | API | `opennn/response_optimization/response_optimization.cpp:177-182` | set(NeuralNetwork*) keeps compiled formula constraints whose column indices belong to the old network | +3 | S/medium | confirmed |
| core-kernels-8 | low | overhead | `opennn/core/cuda/kernel_quantization.cu:29-65` | W8A16 GEMV keeps acc[] in local memory (runtime-bounded loop) and loads x as four scalar BF16 loads | +4 | S/low | confirmed |
| xcut-api-16 | low | design | `opennn/core/device_backend.h:363-365` | get_compute_stream() resolves to the active lane; capture/sync points assume lane 0 without checking | +4 | S/low | confirmed |
| dataset-a-23 | low | design | `opennn/dataset/tabular_dataset.cpp:219-225` | BinaryFile mode: every single set_sample_role call triggers a full streaming pass over the cache file | +4 | S/medium | partial |
| r2-arena-planner-and-propagation-structs-8 | low | overhead | `opennn/neural_network/forward_propagation.cpp:163-176` | set() discards every owned buffer even when the new layout fits the existing ones | +4 | S/low | unverified |
| training-loss-19 | low | design | `opennn/training_strategy/error_functions.cpp:129-147` | MSE/NSE/WSE/CE skip the shape and empty checks that MAE performs | +4 | S/low | confirmed |
| dataset-b-10 | low | overhead | `opennn/dataset/image_dataset.cpp:85-111` | ImageDataset::enable_device_residency materialises the whole dataset twice on the host | +5 | M/medium | partial |
| core-kernels-6 | low | overhead | `opennn/core/cuda/kernel_activation.cu:202-224` | Dropout forward initialises one Philox state per element and discards 3 of every 4 random numbers | +6 | S/low | confirmed |
| core-types-12 | low | design | `opennn/core/opennn_types.h:179-192` | throw_if drops the source location whenever the message has format arguments | +6 | S/low | confirmed |
| layers-a-11 | low | overhead | `opennn/neural_network/layers/concatenation_layer.cpp:52-66` | Concatenation CPU loops re-resolve views and shapes per pixel and use two different channel sources | +6 | S/low | partial |
| r2-set-vs-compile-device-ordering-6 | low | overhead | `opennn/neural_network/layers/recurrent_layer.cpp:728-765` | Recurrent plans cuDNN-only and fused-step-only slots on both devices | +6 | S/low | unverified |
| core-kernels-7 | low | overhead | `opennn/core/cuda/kernel_activation.cu:47-80` | SwiGLU forward/backward are scalar per element (2-byte BF16 accesses across 3-5 streams) | +8 | S/low | confirmed |
| dataset-a-22 | low | design | `opennn/dataset/tabular_dataset.cpp:1489-1496` | Unparseable numeric/binary tokens become NaN silently and are not counted as missing | +8 | S/low | confirmed |
| operators-b-9 | low | overhead | `opennn/neural_network/operators/tokenizer_operator.cpp:981-1005` | tokenize_into rescans the whole remaining text for every special token at every segment boundary | +8 | S/low | confirmed |
| response-opt-18 | low | design | `opennn/response_optimization/response_optimization.cpp:858-1068` | calculate_random_inputs is 209 lines: three sampling lambdas plus a discrete-repair tail | +8 | S/low | confirmed |
| training-optimizers-8 | low | API | `opennn/training_strategy/optimizer.cpp:771-777` | Optimizer::train() silently returns a NaN-filled result without loss/network/dataset, while QN/LM null-deref in the same state | +8 | S/medium | partial |
| core-device-10 | low | overhead | `opennn/core/cuda/cudnn_frontend_utilities.h:82-104` | execute_graph passes the shared_ptr-keyed VariantPack: the frontend rebuilds a uid map on every execute | +10 | S/low | partial |
| core-kernels-13 | low | overhead | `opennn/core/cuda/kernel_attention.cu:450-491` | Generic grouped_attention_kernel re-reads the query row from global memory for every key and keeps acc[256] in local memory | +12 | M/low | confirmed |
| core-kernels-9 | low | design | `opennn/core/cuda/kernel_tensor.cu:41-70` | bias_grad_sum_cuda is the only reduction in the scope that uses float atomics (nondeterministic bias gradient) | +12 | M/medium | partial |
| core-device-6 | low | design | `opennn/core/device_backend.cpp:504-509` | 14 runtime knobs in this scope, 4 documented nowhere, 3 parsed by hand beside env_int_or | +14 | S/low | partial |
| dataset-b-11 | low | overhead | `opennn/dataset/image_dataset.cpp:524-537` | Image cache build decodes every image serially while every other image loop is OpenMP | +15 | M/low | confirmed |
| xcut-build-tests-26 | low | build/test | `opennn/core/statistics.cpp:452-454` | No .clang-tidy/.clang-format/.editorconfig; a first clang-tidy pass on 5 core files yields real narrowing and rounding hits plus known-noise categories to disable | +25 | S/low | partial |
| r2-duplicated-kernels-across-folders-7 | low | overhead | `opennn/core/cuda/kernel_attention.cu:509-519` | grouped_attention_softmax_kernel reads each score row three times and computes every exp twice | +30 | M/low | partial |
| operators-b-7 | low | overhead | `opennn/neural_network/operators/tokenizer_operator.cpp:820-852` | BytePairTokenizer::bpe is O(n^2) per piece and Qwen3's pre-tokenizer makes any non-space run (CJK, URLs, base64) a single piece | +30 | M/low | partial |
| r2-arena-planner-and-propagation-structs-9 | low | build/test | `opennn/core/memory_pool.cpp:135-165` | find_memory_pool_overlay has no direct test; the only pin is an end-to-end overfit that the code comment says strategy can break | +35 | M/low | unverified |
| xcut-build-tests-25 | low | build/test | `opennn/core/string_utilities.cpp:115` | Library .cpp files compile only through the forced-include PCH (<ranges> etc.); clang-tidy/IDEs without it fail | +40 | M/low | confirmed |
| xcut-build-tests-27 | low | build/test | `tests/neural_network/qwen3_network_test.cpp:372-380` | Nine CUDA test files set Device::CUDA with no runtime device guard, unlike the four that GTEST_SKIP | +40 | S/low | partial |

### Refuted

- core-kernels-2 `opennn/core/cuda/kernel_attention.cu:688` GPU sampler can never emit token id 0; the CPU sampler can (and tests assert it) — The GPU fast path's actual host fallback is ChatSession::sample_host (chat.cpp ~612-690), which begins with `adjusted[0] = NEG_INFINITY;` and builds its candidate list from `token = 1`, so that path excludes id 0 for…
- operators-b-3 `opennn/neural_network/layers/multihead_attention_layer.cpp:184` SDPA choice is evaluated once against the Layer's default CPU device and never re-applied when compile() changes the device — NeuralNetwork::compile (neural_network.cpp:579-583) calls layer->set_compute_device(get_device()) and then layer->set_compute_dtype(get_training_type()). Layer::set_compute_dtype (layer.h:169-177) unconditionally calls…

### Finding details

#### nn-builders-chat-1 — load_darknet_backbone_v11 targets c11_* labels the builder no longer emits; always loads 0 layers

`opennn/neural_network/standard_networks.cpp:1951-1983` · high · bug · lines -56 · effort S · risk medium · confirmed

The CSPDarknet53v11 backbone now labels every layer with the c8_ prefix (line 699: 'Layer prefix "c8_" distinguishes from the old C3k2 ("c11_") implementation'); grep finds no c11_ label generated anywhere in opennn/ or tests/. load_darknet_backbone_v11 still looks up c11_stem, c11_s1_down ... c11_s5_down, so every lookup hits the 'not found — skipping' branch, the function prints six warnings, returns 0, and the skip offsets (42368, 79872, 811520, 3228672 floats) describe the dead C3k2 layout, not the C2f one. examples/yolo/main.cpp:846-855 calls it, prints 'Loaded 0 backbone layers' and still sets backbone_pretrained_loaded = true, so a user asking for pretrained yolov4.conv.137 weights…

**Fix:** Either delete load_darknet_backbone_v11 (and its header declaration) and route the example through load_darknet_backbone, or retarget it to the c8_ labels with offsets recomputed for the C2f layout; in both cases make the loader throw (not print) when a target label is missing so the example cannot report success with 0 layers. Verify against Neural Designer before deleting the symbol.

*Verifier:* standard_networks.cpp:1951-1958 targets c11_stem..c11_s5_down; the only c11_ strings in the repo are those six, the comment at 699 ('Layer prefix "c8_" distinguishes from the old C3k2 ("c11_")') and examples/yolo/main.cpp:877 (freeze prefix). The builder emits c8_stem (764), c8_s1_down..c8_s4_down (770-784). Every lookup at 1972-1977 therefore misses, prints and continues; loaded stays 0.…

#### r2-arena-planner-and-propagation-structs-1 — CPU valid-length record is frozen after the first forward pass; later padded batches use stale masks

`opennn/neural_network/forward_propagation.cpp:1165-1190` · high · bug · lines -8 · effort S · risk low · unverified

inherit_valid_lengths copies the source's host record into valid_lengths[layer] on the first pass and then returns early forever because the copy is non-empty. The Embedding rewrites only its own entry each pass (compute_token_valid_lengths at embedding_lookup_operator.cpp:353-354); nothing clears the downstream copies (no other writer exists in the repo, and set() only clears at construction). Concrete failure: CPU transformer Embedding(export) -> MHA1 -> Addition -> LN -> MHA2. Pass 1 with lengths {8,8,8,8} propagates copies into MHA1/Addition/LN. Pass 2 with lengths {8,5,3,1}: MHA1 reads the Embedding's fresh record (direct source), but MHA2 reads LN's frozen {8,8,8,8}…

**Fix:** Delete the early return at line 1170 (and its comment) so every pass re-inherits; the Embedding is already protected because its source is a network input. Better and one PR: replace the per-layer copies and cached device pointers with one `vector<Index> valid_lengths_origin` resolved in inherit_valid_lengths (origin = source's origin, or the layer itself when it exported), and have input_valid_lengths/input_device_valid_lengths index valid_lengths[origin] / device_valid_length_storage[origin];…

*Verifier:* Code-consistent: forward_propagation.cpp:1169 returns early once valid_lengths[layer] is non-empty, so a downstream copy is never refreshed. Needs a two-batch CPU test to confirm the observable effect.

#### nn-expression-2 — Logarithm scaler/unscaler exports to Python (NameError) and JavaScript (ReferenceError)

`opennn/neural_network/model_expression.cpp:1993-2008` · high · bug · lines -3 · effort S · risk low · confirmed

Scaling::write_expression emits `scaled_x = log(x);` for ScalerMethod::Logarithm and Unscaling emits `y = exp(x);`. The JS emitter rewrites only exp/tanh/max/min to Math.*, so `log(` is left bare and the exported page throws ReferenceError at runtime. The Python emitter never qualifies exp/log at all (the header imports `math` and `numpy as np`, neither makes `log`/`exp` bare names), so calculate_outputs raises NameError for any network with a Logarithm scaler or unscaler. Concrete: a network whose scaling layer uses "Logarithm" for one feature, exported with ProgrammingLanguage::Python, fails on the first call. Neither test file exercises Logarithm (grep: no match in…

**Fix:** Add "log" to math_keywords. In emit_python_calculate_outputs' transform, map the words exp/log to np.exp/np.log with the same replace_all_word_appearances call (or add `from math import exp, log` to emit_python_class_header). Extend build_network() in expression_execution_test.cpp with a Logarithm feature so the Python run covers it.

*Verifier:* scaling_layer.cpp:511 emits bare log(...) and unscaling_layer.cpp:145 emits bare exp(...). emit_js_runtime math_keywords (line 1993) is {exp, tanh, max, min} - no log, so JS throws ReferenceError. The Python transform (2111-2124) only maps activation names to self.<name> and strips ';'; emit_python_class_header imports only math, numpy as np, pandas as pd (2065-2068), so bare log/exp raise…

#### core-device-1 — CudaBlockCache::give can throw from Buffer destructors -> std::terminate masks the real CUDA error

`opennn/core/device_backend.cpp:511-528` · high · bug · lines -2 · effort S · risk low · confirmed

device::deallocate is reached from Buffer::free_buffer() inside ~Buffer (implicitly noexcept). deallocate -> CudaBlockCache::give -> record_pending -> create_event_handle (CHECK_CUDA(cudaEventCreateWithFlags)) and record_event (throw_if + CHECK_CUDA(cudaEventRecord)) all throw on failure. Concrete scenario: a kernel faults (cudaErrorIllegalAddress, sticky). check_last_error throws a runtime_error; while that exception unwinds, the first local Buffer destroyed calls give(), cudaEventRecord returns the sticky error, CHECK_CUDA throws inside a noexcept destructor -> std::terminate. The user never sees the original diagnostic (MSVC abort()s without printing what()). Second scenario: any Buffer…

**Fix:** Make the deallocate path non-throwing: in record_pending call cudaEventCreateWithFlags/cudaEventRecord directly, on any non-success status call cudaGetLastError(), push the event back to event_pool, recycle the block's events and have give() return false so the caller falls through to cudaFree (whose status is already ignored). Mark device::deallocate noexcept and replace its throw_if_auto with an assert-style early return. Add a test that destroys a Buffer after device::reset_last_error() with…

*Verifier:* tensor_types.h:379 `~Buffer() { free_buffer(); }` (implicitly noexcept) -> tensor_types.h:537-539 device::deallocate -> device_backend.cpp:599-607 throw_if_auto + CudaBlockCache::give (423-448) -> record_pending (511-529) -> create_event_handle (817-822, CHECK_CUDA) and record_event (941-946, throw_if + CHECK_CUDA). Every throw on that path lands in a noexcept destructor. deallocate's cudaFree…

#### core-utils-2 — Quoted-field tokenizer deletes every ',' and ';' inside quotes regardless of the separator

`opennn/core/string_utilities.cpp:147-216` · high · bug · lines -2 · effort S · risk medium · confirmed

get_token_views_maybe_quoted (line 178) and first_token_maybe_quoted (line 211) skip any ',' or ';' while in_quote, independent of the file separator. Quotes exist to protect the separator; instead the field content is silently altered. Concrete: language_dataset.cpp:331-352 defaults field_separator to '\t' and passes result.has_quotes, so a tab-separated corpus whose text column is quoted ("Hello, world; bye") is tokenized as "Hello world bye" -- every comma and semicolon in the training text disappears before the tokenizer sees it. For comma-separated tabular files a quoted categorical "Smith; John" becomes "Smith John" and a quoted "1,5" (European decimal) becomes "15" even though…

**Fix:** Delete both `if (in_quote && (c == ',' || c == ';')) continue;` lines so quoted fields keep their content verbatim (the separator inside quotes is already protected by the `!in_quote && c == separator` test); add a test that a quoted field containing the separator and other punctuation round-trips. While there, implement first_token_maybe_quoted by stopping the shared loop at the first unquoted separator instead of duplicating it.

*Verifier:* string_utilities.cpp:178 and :211 both read `if (in_quote && (c == ',' || c == ';')) continue;` independent of `separator`; the separator itself is already protected at :172/:210 by `!in_quote && c == separator`. Callers: dataset.cpp:936/942, language_dataset.cpp:351 (field_separator defaults to '\t' at :331-332, passes result.has_quotes), tabular_dataset.cpp:1696 (first_token_maybe_quoted for…

#### core-device-2 — set_threads_number destroys the ThreadPool that tensor_operations caches for the process lifetime (UAF)

`opennn/core/device_backend.cpp:1214-1225` · high · bug · lines -1 · effort S · risk low · confirmed

Backend::set_threads_number replaces thread_pool (unique_ptr<ThreadPool>) and thread_pool_device. tensor_operations.cpp:237-243 builds a function-local static Eigen::ThreadPoolDevice from get_device().getPool() the first time the Contract GEMM path runs (the default mode: gemm_parallelism() returns Contract when OPENNN_GEMM_MODE is unset). After any later set_threads_number() call that static holds a pointer to a deleted ThreadPool; the next large CPU dense forward with bias (k >= 64, flops >= gemm_contract_flops) enqueues work on freed memory. set_threads_number is public API (device_backend.h:379), called by four test files (e.g. stochastic_gradient_descent_test.cpp:826…

**Fix:** ThreadPoolDevice is a three-pointer handle: make contraction_device() return Eigen::ThreadPoolDevice by value built from the current get_device() on every call (the static scratch allocator stays). Additionally document on get_device() that the pool is replaced by set_threads_number and must not be cached; optionally have set_threads_number keep the previous pool alive in a retired vector so any other cached handle degrades to 'stale thread count' instead of UAF.

*Verifier:* tensor_operations.cpp:237-243 caches `static Eigen::ThreadPoolDevice device(get_device().getPool(), ...)` once; device_backend.cpp:1222-1223 `thread_pool = make_unique<ThreadPool>(...)` deletes the previous pool on every set_threads_number. gemm_parallelism() defaults to Contract when OPENNN_GEMM_MODE is unset (tensor_operations.cpp:125-126), and the Contract branch at 554-560 routes to…

#### core-utils-1 — Apple from_chars shim calls itself for integral T: infinite recursion on macOS builds

`opennn/core/string_utilities.h:27-58` · high · bug · lines 0 · effort S · risk low · confirmed

On __APPLE__ the header defines opennn::from_chars<T> as a shim for floating-point parsing and, for integral T, intends to forward to std::from_chars. But the forward is written unqualified inside namespace opennn, where unqualified lookup finds opennn::from_chars itself and stops (the global 'using namespace std' is never reached, and ADL contributes nothing for const char*/int&). So opennn::from_chars<int>/<long>/<long long>/<Index> recurses until the stack overflows. Reached by parse_number<T> (string_utilities.h:104-115) for every integer field: parse_int (time_series_dataset.cpp:212-213), parse_long (tabular_dataset.cpp:2401), parse_number<Index> (tabular_dataset.cpp:2391,…

**Fix:** Change line 55 to `return std::from_chars(first, last, value);`. Optionally add a static_assert-free unit test that calls parse_int on the shim path (the existing parse tests only run on the non-Apple branch).

*Verifier:* string_utilities.h:27-58: the shim is defined inside namespace opennn and line 55 calls unqualified `from_chars(first, last, value)`. The only `using namespace std` is at global scope (opennn_types.h:84/136); there is no `using std::from_chars` inside namespace opennn (grep), so unqualified lookup finds opennn::from_chars in the enclosing namespace and stops; ADL adds nothing for const…

#### selection-testing-1 — calculate_errors divides the Minkowski error by the unrelated batch_size member (default 0 -> +inf)

`opennn/testing_analysis/testing_analysis.cpp:264-292` · high · bug · lines 0 · effort S · risk low · confirmed

TestingAnalysis::calculate_errors(targets, outputs) computes four errors from samples_number = targets.rows(), but the fifth (Minkowski, p=1.5) divides by the class member batch_size, which is the inference chunk size used by get_targets_and_outputs and defaults to 0. Scenario: `TestingAnalysis ta(&nn, &ds); ta.calculate_errors("Testing")` -> errors(4) = sum/0.0f = +inf. If the user called set_batch_size(256) on a 10,000-sample testing set, errors(4) is 39x too large. calculate_classification_errors also runs this computation and discards it. No test covers errors(4) (testing_analysis_test.cpp only tests reconstruction errors), and Neural Designer consumes this vector.

**Fix:** Replace `static_cast<float>(batch_size)` with `float(samples_number)` (already computed at the top of the function). Add a one-line assertion in testing_analysis_test.cpp that errors(4) is finite for a 2x2 case.

*Verifier:* testing_analysis.cpp:264-292 read: samples_number = targets.rows() is used for errors(1..3) but errors(4) divides by the member batch_size (testing_analysis.h:151 `Index batch_size = 0;`), which get_targets_and_outputs (cpp:121-125) treats as 'use default chunk'. Default construction therefore yields sum/0 = +inf; a user-set batch_size scales the Minkowski error by samples/batch.…

#### r2-set-vs-compile-device-ordering-2 — Bias-free fused-ReLU Dense on CPU skips the ReLU: output is the raw pre-activation

`opennn/core/tensor_operations.cpp:1275-1296` · high · bug · lines +1 · effort S · risk low · confirmed

CombinationOperator picks the epilogue as `use_bias ? (relu ? RELU_BIAS : BIAS) : (relu ? RELU : DEFAULT)` (combination_operator.cpp:153-156), and Dense sets activation_operator.forward_fused = true for any ReLU without batch norm regardless of use_bias. linear_forward_cpu only honours RELU_BIAS: `const bool fuse_relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS;` so for CUBLASLT_EPILOGUE_RELU (no bias) neither the MKL path nor the Eigen fallback clamps, and ActivationOperator returns early. A Dense("ReLU") with set_use_bias(false) - JSON "UseBias": false - therefore behaves as Identity on CPU. The ActivationOperator comment's claim that "the CPU epilogue applies the ReLU in add_bias exactly…

**Fix:** In linear_forward_cpu: `const bool fuse_relu = epilogue == CUBLASLT_EPILOGUE_RELU_BIAS || epilogue == CUBLASLT_EPILOGUE_RELU;` (try_linear_forward already rejects an empty bias and the Eigen fallback applies the clamp). Add a bias-free ReLU case to dense_no_bias_test.cpp.

*Verifier:* Confirmed by hand: combination_operator.cpp:153-155 picks CUBLASLT_EPILOGUE_RELU for a bias-less Dense; linear_forward_cpu (tensor_operations.cpp:1280) only honours RELU_BIAS.

#### r2-set-vs-compile-device-ordering-1 — Fused GELUTanh Dense on CPU never writes its Output slot: the layer returns zeros

`opennn/neural_network/layers/dense_layer.cpp:199-233` · high · bug · lines +2 · effort S · risk low · confirmed

Dense::configure_operators decides fuse_gelu_tanh with no device knowledge (activation == GELUTanh, no batch norm, output_features % 8 == 0) and then sets combination.output_slots = {CombinationView, Output} and activation_operator.forward_fused = true. CombinationOperator::forward_propagate only takes the GELU_AUX_BIAS epilogue branch when output.is_cuda(); on CPU it falls through and writes the plain Wx+b into get_output() == slots[CombinationView], leaving Output untouched. ActivationOperator::forward_propagate then returns immediately because forward_fused is true (activation_operator.cpp:38, the is_cuda() guard was removed on 2026-08-20 with a comment that reasons only about ReLU). Net…

**Fix:** In Dense::configure_operators gate the fusion on the device: `const bool fuse_gelu_tanh = ... && get_compute_device() == Device::CUDA;`. configure_operators is re-run from on_compute_dtype_changed, which compile() calls after set_compute_device, so the flag is correct at compile time (same mechanism MultiHeadAttention::should_use_sdpa relies on). Add a forward-value test for a width-multiple-of-8 GELUTanh Dense on CPU (the gradient check does not catch it). Longer term see finding 10.

*Verifier:* Code-confirmed by hand: dense_layer.cpp:199-203 sets fuse_gelu_tanh with no device test; combination_operator.cpp:103-110 takes the GELU_AUX_BIAS branch only when output.is_cuda(), so on CPU it writes Wx+b into CombinationView and the activation pass is skipped.

#### r2-batch-pipeline-and-device-gather-1 — run_graph_epoch dereferences a null pipeline slot when grouped slots exist but a post_batch_callback is set

`opennn/training_strategy/optimizer.cpp:1606-1610` · high · bug · lines +2 · effort S · risk low · confirmed

setup_batch_pools allocates pipelines[1].slots[0] only on the non-grouped path (`training_batches > 0 && !grouped_batches`, line 281) and allocates pipelines[1] for the grouped path only when `training_batches >= slots_count` (16, line 288-291). With 8 <= training_batches < 16 the grouped slots pipelines[0].slots[1..7] exist, has_graph_batches() is true, cuda_graph_capture_allowed becomes true (line 899) and train_epoch enters run_graph_epoch. There, `can_group_batches = !post_batch_callback && batches_number >= group_size` (line 1524): any user-installed post_batch_callback (a public member; examples/yolo/main.cpp:920 installs one) sends the epoch to the single-slot loop, which indexes…

**Fix:** In the single-slot loop of run_graph_epoch compute the usable pipeline count once: `const size_t usable_pipelines = pipelines[1].slots[0] ? pipelines.size() : 1;` and index `pipelines[size_t(iteration) % usable_pipelines]`. Alternatively make setup_batch_pools always allocate pipelines[1].slots[0] on GPU (one extra batch of VRAM). Add a test: GPU + set_cuda_graph(true) + 10 complete batches + post_batch_callback.

*Verifier:* Read optimizer.cpp 268-295 (setup_batch_pools), 899, 1524, 1616-1660 and optimizer.h 109-147. grouped_batches is true for training_batches >= group_size (8); pipelines[1].slots[0] is only created on the !grouped path (281) or, grouped, when training_batches >= slots_count (16, lines 288-291). has_graph_batches() (optimizer.h 123-126) is true via pipelines[0].slots[1], so…

#### layers-b-1 — LSTM on CUDA has no FP32 guard: BF16 networks feed BF16 slots to CUDNN_DATA_FLOAT descriptors

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:983-1012` · high · bug · lines +4 · effort S · risk low · partial

LongShortTermMemory::get_forward_specs/get_backward_specs (lines 1096-1127) allocate every slot with compute_dtype, and the layer does not override allows_bf16_input_cast, so under Configuration::instance().set(Device::CUDA, Type::BF16) its input, gate and output slots are BF16 (B*T*H*2 bytes). apply_gpu/apply_delta_gpu then hand input.get_data()/output.as<float>() straight to cuDNN, whose x/y descriptors are built with CUDNN_DATA_FLOAT (cudnn_rnn.cpp:178-201). cuDNN reads the BF16 input as float (half garbage, and B*T*F*4 bytes from a B*T*F*2 allocation) and writes B*T*H floats into a B*T*H*2-byte output slot, clobbering the neighbouring pooled slot. No error is raised; the network trains…

**Fix:** Minimal: at the top of apply_gpu and apply_delta_gpu add throw_if(!input.is_fp32() || !output.is_fp32(), "LongShortTermMemory CUDA: cuDNN LSTM runs FP32 only; compile the network with Type::FP32."), mirroring RecurrentOperator::require_same_recurrent_dtype. Fuller (same PR if desired): make get_forward_specs/get_backward_specs use Type::FP32 like parameter_specs already does, and override allows_bf16_input_cast to return false, so an LSTM inside a BF16 network runs in FP32 instead of throwing.…

*Verifier:* Substance confirmed. long_short_term_memory_layer.cpp:983-1012 apply_gpu and 1021-1060 apply_delta_gpu hand input.get_data()/output.as<float>() to cudnn_rnn_forward_/backward_ with no dtype check (TensorView::as<T> at tensor_types.h:593 is a blind reinterpret_cast); cudnn_rnn.cpp:184/189 build x/y descriptors with CUDNN_DATA_FLOAT; get_forward_specs/get_backward_specs (1096-1130) use…

#### nn-core-1 — set_parameters / load_parameters_binary on a released fp32 master overflow the compact bf16 mirror

`opennn/neural_network/neural_network.cpp:1002-1017` · high · bug · lines +5 · effort S · risk low · confirmed

After upload_parameters_bf16_inference() or release_bf16_fp32_parameter_master_for_inference(), `parameters` is a non-owning CUDA view over parameters_bf16_mirror (set_view at 2626-2628 / 2084) and parameters_bf16_mirror_compact is true, so the mirror holds only totals.bf16_elements (2063-2064), fewer than parameters.size_in_floats() whenever the network has any FP32 slot (biases, norms - always). Scenario A: set_parameters(v) -> upload_host_vector: Buffer::resize_bytes on a view does not early-return (owns_allocation false) and allocates a fresh owning fp32 buffer -> returns true -> cast_parameters_to_bf16() writes size_in_floats bf16 elements into the smaller compact mirror (device…

**Fix:** Add `throw_if(fp32_master_released(), "NeuralNetwork::set_parameters: the fp32 parameter master was released for quantized inference; reload the model before replacing parameters.");` at the top of set_parameters, and the same guard (with its own message) at the top of load_parameters_binary, mirroring save_parameters_binary:2093 and copy_parameters_host:2769. Then delete line 1011, which only existed for the released state and becomes dead. Add a regression test:…

*Verifier:* Read neural_network.cpp 978-1017, 2062-2068, 2084-2091, 2550-2556, 2622-2628, 1900-1947, tensor_types.h 422-463. After upload_parameters_bf16_inference -> use_compact_parameter_storage (2084-2091) `parameters` is a non-owning CUDA view of master_bytes over the compact mirror sized totals.bf16_elements (2063). set_parameters: the size check at 1007 passes (size_in_floats is the master size),…

#### training-loss-1 — NormalizedSquaredError drops the total/batch scaling its sibling WSE and the old implementation apply

`opennn/training_strategy/error_functions.cpp:201-217` · high · bug · lines +6 · effort S · risk medium · confirmed

normalization_coefficient is the sum of squared deviations over the WHOLE training set (loss.cpp:1299-1313), but the per-batch error divides the batch's sum of squares by that full-set coefficient with no total_samples/batch_samples factor. The optimizer averages per-batch errors (optimizer.cpp:99 `sums.error /= float(batches_number)`), so with k mini-batches the reported epoch NSE is ~NSE_true/k, and the validation error (same Loss, same coefficient, different sample count) is not comparable to the training error. The gradient (line 216) and the device-metrics path (loss.cpp:1600-1603, `1.0f / (normalization_coefficient + EPSILON)`) carry the same omission, so the effective step size also…

**Fix:** Generalize Loss::get_weighted_coefficient into a `batch_scale(batch)` = training_samples/batch_samples and fold it into the NSE coefficient passed to normalized_squared_error / normalized_squared_error_gradient and into the device-metrics scale at loss.cpp:1603 (coefficient * batch/total). Add a two-batch test asserting that the mean over batches equals the full-batch NSE.

*Verifier:* error_functions.cpp:201-217 divide the batch sum of squares by `coefficient + EPSILON` with no samples factor; loss.cpp:1299-1313 compute that coefficient over ALL training rows; loss.cpp:1600-1603 device-metrics path uses `1.0f / (normalization_coefficient + EPSILON)` likewise. WSE applies `float(total) / (float(samples) * ...)` via get_weighted_coefficient (loss.cpp:1381-1388). optimizer.cpp:99…

#### operators-a-1 — Convolutional ReLU is never applied on CPU: 'forward_fused' skips the activation but no CPU epilogue exists

`opennn/neural_network/operators/activation_operator.cpp:30-39` · high · bug · lines +8 · effort S · risk low · confirmed

ActivationOperator::forward_propagate returns early whenever forward_fused is set, on either device. Convolutional::update_convolution_operator (convolutional_layer.cpp:206-211) sets activation_operator.forward_fused = relu unconditionally, but the only ReLU epilogues are GPU-side: ConvolutionOperator::apply_cpu (convolution_operator.cpp:468-509) has no ReLU, and BatchNormalizationOperator::apply_training_cpu/apply_inference_cpu (batch_norm_operator.cpp:265-310) ignore fuse_relu. The CPU backward still applies the ReLU derivative (backward_fused only short-circuits when output_delta.is_cuda()), masking on a sign the forward never enforced. Confirmed with a probe linked against a fresh CPU…

**Fix:** Make the CPU bodies honour the flag the layer already sets, mirroring the GPU epilogues: in ConvolutionOperator::apply_cpu add `if (fuse_relu) output_matrix = output_matrix.cwiseMax(0.0f);` inside the per-image loop; in BatchNormalizationOperator::forward_propagate's CPU branches apply `if (fuse_relu) activation_forward(output, ActivationFunction::ReLU);` after the optional residual add (training and inference). Add a ForwardPropagate test case with negative inputs and an initialized…

*Verifier:* Confirmed by hand: convolutional_layer.cpp:210 sets forward_fused = relu on every device; ConvolutionOperator::apply_cpu (convolution_operator.cpp:470-515) never applies a ReLU; the conv test feeds all-positive inputs so EXPECT_GE(…,0) cannot see it. Substance confirmed from the code: activation_operator.cpp:38 returns on `forward_fused` regardless of device (the comment at 30-37 justifies it…

#### layers-a-1 — Float-only layers accept a BF16 compute dtype and reinterpret BF16 buffers as float

`opennn/neural_network/layers/concatenation_layer.cpp:34-50` · high · bug · lines +12 · effort S · risk low · confirmed

NeuralNetwork::set(config) calls layer->set_compute_dtype(get_training_type()) on every layer (neural_network.cpp:580-584) with no per-layer capability check. Concatenation, Upsampling (upsampling_layer.cpp:32-42, 70-80) and Detection do not override get_forward_specs, so with CUDA + BF16 (or Type::Auto, which the run-examples skill documents as resolving to CUDA+BF16 when a GPU is present) their forward/backward slots are allocated as BF16 views (forward_propagation.cpp:881 builds views from the spec dtype) and their inputs arrive as BF16 from the preceding Convolutional. The operators then call inputs[i].as<float>() / output.as<float>() (TensorView::as is an unchecked reinterpret_cast,…

**Fix:** Minimal: in Concatenation, Upsampling and Detection/DetectionV8 override on_compute_dtype_changed() with throw_if(compute_dtype != Type::FP32, "{} layer does not support {} activations; use Type::FP32.", get_name(), ...). Better for Concatenation: the BF16 instantiations of slice_channels_cuda<T,Scatter> already exist in kernel_concat.cu:39-44, so replace the two float-only wrappers with output.dispatch([&]<typename T>{ slice_channels_cuda<T,true>(...); }) and delete…

*Verifier:* Layer::set_compute_dtype (layer.h:169-176) assigns compute_dtype unconditionally and NeuralNetwork::compile (neural_network.cpp:580-584) calls it on every layer with no capability check; no supports_bf16 gating exists anywhere (grep over neural_network.cpp/configuration.*). Convolutional::get_forward_specs emits its output with compute_dtype (convolutional_layer.cpp:136), so under CUDA+BF16 the…

#### dataset-a-2 — BinaryFile storage: analysis methods index the empty `data` matrix (null-pointer reads)

`opennn/dataset/tabular_dataset.cpp:1342-1382` · high · UB · lines +18 · effort M · risk low · confirmed

Dataset::set_storage_mode(BinaryFile) does data.resize(0,0) and TabularDataset::get_samples_number() then returns ssize(sample_roles), so any method that loops samples_number and reads data(i, j) dereferences a null Eigen buffer in release builds (eigen_assert is compiled out). Affected public methods: calculate_target_distribution (1342-1382), calculate_variable_distributions (685-772, `data(used_sample_indices, feature_index)`), calculate_variables_box_plots / calculate_Tukey_outliers / replace_Tukey_outliers_with_NaN (774-796, 1384-1458), Dataset::filter_data (dataset.cpp 281-319), impute_missing_values_* / unuse_samples_with_missing_targets (2405-2555; only scrub_missing_values guards),…

**Fix:** Add one protected helper `void require_in_memory_data(string_view what) const { throw_if(storage_mode == StorageMode::BinaryFile, "{} is not available with BinaryFile storage.", what); }` and call it on entry of each listed method (one line each). Make get_variable_data(Index, rows) go through fill_features so the correlation path works on both storages. Add a test that each method throws instead of crashing on the binary fixture.

*Verifier:* dataset.cpp:130-135 set_storage_mode(BinaryFile) does data.resize(0,0); tabular_dataset.h:39-44 get_samples_number returns ssize(sample_roles) in that mode. calculate_target_distribution (tabular_dataset.cpp:1342-1382) loops samples_number reading data(i, target_feature) from a 0x0 matrix; calculate_variable_distributions (711, 730, 750) and calculate_variables_box_plots (786) index data…

#### operators-a-2 — Dropout seed is baked into captured CUDA graphs: every replay reuses the same mask

`opennn/neural_network/operators/dropout_operator.cpp:117-127` · high · bug · lines +20 · effort M · risk medium · partial

dropout_forward_gpu draws one host seed per call (random_integer) and passes it by value to dropout_forward_cuda, whose kernel does curand_init(seed, idx, 0). Optimizer::run_graph_epoch captures run_compute_step (forward + backward + update) once per GraphPipeline (optimizer.cpp:1441-1470, capture_or_run: `if (exec) return device::launch_graph(exec, compute);`) and replays it for every later group and every epoch; nothing in can_use_cuda_graph() excludes networks with dropout. So a Dense or Embedding layer with rate > 0 trained with CUDA graphs on Adam/SGD gets at most pipelines_count (= 2) distinct masks for the entire run: the regularizer degenerates into a fixed sparse sub-network,…

**Fix:** Give the kernel a device-resident seed/counter instead of an immediate: keep one `unsigned long long` in a small device buffer (a GraphWorkspaceKind slot), have dropout_forward_kernel read it, and advance it inside the captured step with a one-thread kernel appended after the forward (so each replay sees a new value). Alternatively, until that lands, make can_use_cuda_graph() return false when any layer has an active DropoutOperator so graph training falls back to eager. Add a test that runs…

*Verifier:* Mechanism confirmed: dropout_operator.cpp:82 draws the seed on the host (`random_integer(0, 1<<30)`) and passes it by value to dropout_forward_cuda; kernel_activation.cu:211-212 does `curand_init(seed, idx, 0)` so the mask is a pure function of the baked seed. optimizer.cpp:1429-1438 run_compute_step captures forward+backward+update; capture_or_run at 1441-1447 replays `exec` whenever it is set,…

#### xcut-build-tests-5 — write_bmp_24/TempDir/write_label/write_classes copy-pasted into 7 test files (+yolo example); copies already diverged

`tests/training_strategy/yolo_loss_test.cpp:24-104` · high · duplication · lines -340 · effort M · risk low · confirmed

Still pending from the prior audit, with more copies than it counted. `write_bmp_24` is defined in yolo_dataset_test.cpp:17-55, yolo_loss_test.cpp:24-60, yolo_fpn_test.cpp:22-58, yolo_overfit_test.cpp:25-57, image_dataset_test.cpp:32-81, adaptive_moment_estimation_test.cpp:32-79 and stochastic_gradient_descent_test.cpp:34-81 (plus write_u16/write_u32 helpers in the last three), `TempDir` in four files (yolo_dataset 69-94, yolo_loss 74-104, yolo_fpn 66-96, yolo_overfit 60-78), `write_label` in two, `write_classes` in three, and examples/yolo/main.cpp:47-96 has a third BMP writer/reader pair. The copies have drifted: yolo_loss_test.cpp:41-42 and yolo_fpn_test.cpp:39-40 write only the low byte…

**Fix:** Add `tests/vision_test_helpers.{h,cpp}` (the GLOB picks it up) with one `write_bmp_24` (full 4-byte width/height), `write_label`, `write_classes` and `TempDir(string_view prefix)`; delete the seven local copies. Keep `TempDir` movable-deleted as today.

*Verifier:* Grep confirms write_bmp_24 definitions in yolo_dataset_test:17, yolo_loss_test:24, yolo_fpn_test:22, yolo_overfit_test:25, image_dataset_test:32, adaptive_moment_estimation_test:32, stochastic_gradient_descent_test:34 (7 files) plus write_u16/write_u32 in the last three; TempDir in yolo_dataset:69, yolo_loss:74, yolo_fpn:66, yolo_overfit:60; write_label in yolo_dataset:57, yolo_loss:62…

#### xcut-build-tests-8 — blank_cuda is 230 lines entirely under #if 0 with hard-coded /home/artelnics paths, plus its own CMake/option plumbing

`examples/blank/main_cuda.cpp:50-218` · high · dead code · lines -250 · effort S · risk medium · confirmed

`main_cuda.cpp` has two `#if 0` blocks (50-115 ResNet/Imagenette, 117-218 WMT14 transformer) with absolute `/home/artelnics/Documents/datasets/...` paths and 25 includes that serve only that dead code; the live body is an empty try/catch identical to blank/main.cpp. Around it: the `OpenNN_BUILD_BLANK_CUDA` option (root CMakeLists.txt:157-159), examples/blank/CMakeLists.txt:11-26 (a nested `project(blank)` and a `blank_cuda` target that re-links CUDA::cudart/cublas and `${CUDNN_LIBRARY}` although opennn already exports them PUBLIC), and `target_include_directories(... ${CMAKE_CURRENT_SOURCE_DIR}/../opennn)` (lines 6-8, 13-15) which puts `opennn/` itself on the include path, contradicting…

**Fix:** Delete main_cuda.cpp, the blank_cuda target and the OpenNN_BUILD_BLANK_CUDA option; replace examples/blank/CMakeLists.txt with `opennn_example(blank)` in examples/CMakeLists.txt (drop the `../opennn` include dir). If a GPU scratch target is wanted, keep it untracked (.gitignore) or move the two experiments to docs/benchmarks where such drivers live.

*Verifier:* examples/blank/main_cuda.cpp is 231 lines: `#if 0` at 50-115 and 117-218 with /home/artelnics/... paths at 58, 125, 128, 172; 25 includes at 9-40. examples/blank/CMakeLists.txt: nested project(blank) line 3, `../opennn` include dir on both targets (lines 6-8, 13-15) violating AGENTS.md 'only the repo root is on the include path', blank_cuda re-links CUDA::cudart/cublas and ${CUDNN_LIBRARY}. Root…

#### r2-set-vs-compile-device-ordering-3 — '#pragma omp simd' in the LSTM CPU kernels breaks every MSVC build (C7660)

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:372-437` · high · build/test · lines +3 · effort S · risk low · confirmed

long_short_term_memory_layer.cpp is the only file in the library using `#pragma omp simd` (12 sites: 372, 389, 403, 415, 429, 437, 786, 830, 855, 899, 905, 911). MSVC only accepts that pragma with /openmp:experimental; opennn/CMakeLists.txt passes ${OpenMP_CXX_FLAGS}, which on MSVC is plain -openmp (both build-cpu-verification/build.ninja and build-resnet-capacity/build.ninja contain only `-openmp`). Rebuilding the test target in build-cpu-verification from the current sources stops at `error C7660: 'simd': requires '-openmp:experimental' command line option(s)`; the CUDA tree compiles the same file with the same flag, so neither tree currently builds. The file was modified after the last…

**Fix:** Either add `if(MSVC) target_compile_options(opennn PRIVATE /openmp:experimental)` next to line 172 (also for the test/example targets that compile with the PCH, to avoid C4652 PCH-mismatch warnings), or define an `OPENNN_OMP_SIMD` macro in opennn_types.h that expands to `_Pragma("omp simd")` on GCC/Clang and to nothing on MSVC, and use it at the 12 sites. Add the CPU build to CI (the prior audit already noted the benchmark compile gap).

*Verifier:* Reproduced: a clean `cmake --build build-cpu-verification --target opennn_tests` stops with C7660 at long_short_term_memory_layer.cpp:909/915 (12 `#pragma omp simd` sites; MSVC flag is plain -openmp).

#### nn-expression-1 — emit_c_main injects raw output names into a printf format string (and HTML/PHP literals)

`opennn/neural_network/model_expression.cpp:677-701` · medium · bug · lines -16 · effort S · risk low · partial

The C and CEmbedded drivers write each raw output feature name inside the printf format string. A column name containing '%' (e.g. "Humidity (%)", common in tabular datasets) yields printf("Humidity (%): %f \n", ...) whose "%)" is an invalid conversion specification: undefined behaviour in the exported program, and it is the firmware path the docs advertise. A name containing '"' or '\' makes the exported C fail to compile. The same unescaped insertion happens in the PHP response (line 1786: a single quote in a name breaks the PHP array literal), and in JS HTML (lines 1873, 1912, 1927: '<', '&', '"' in names corrupt the markup). Only the identifier-ish names go through…

**Fix:** Emit printf("%s: %f\n", <escaped literal>, outputs[i]) where the literal comes from a 6-line static `c_string_literal(string_view)` that escapes \ " and newlines (or simply print the sanitized fixed_output_names[i], which is already a valid identifier). Add a 6-line `html_escape` for the three JS sites and a 2-line single-quote escape for the PHP site. Add a test exporting a network whose output is named "Humidity (%)" to C and compiling it (the ExpressionExecution harness already compiles C).

*Verifier:* Code confirmed: emit_c_main (lines 698-699) streams output_names[i] raw into the printf format string; emit_php_response (1786) streams it inside a single-quoted PHP literal; emit_js_outputs_html (1925-1927, 1940) and emit_js_inputs_html (1873) stream raw names into HTML. No escaping anywhere (replace_reserved_keywords is only applied to identifiers). Overstated on one point: main() is wrapped in…

#### nn-builders-chat-2 — Both Darknet loaders leak the FILE* when load_darknet_weights throws; header reading duplicated

`opennn/neural_network/standard_networks.cpp:1898-1987` · medium · bug · lines -14 · effort S · risk low · confirmed

Convolutional::load_darknet_weights (convolutional_layer.cpp:370,387,393) throws on any short read. Both loaders open with fopen and only fclose after the loop, so a truncated or mismatched .weights file (the common failure when the architecture and the file disagree, e.g. the c11_ case above) leaks the handle; on Windows the file then stays locked for the process lifetime. The two functions also duplicate the 14-line header read + cout banner verbatim (1902-1915 vs 1936-1949), and the library prints progress to cout.

**Fix:** Add a file-local `unique_ptr<FILE, int(*)(FILE*)> open_darknet_weights(const filesystem::path&, const char* who)` that opens, reads and validates the header once and returns the RAII handle; both loaders call it and drop their fclose. Replace the cout banners with nothing (or a single line behind the existing display flag).

*Verifier:* convolutional_layer.cpp:369-393 throws via throw_if on every short fread; both loaders (1902-1930, 1936-1987) fopen and only fclose after the loop with no RAII, so any throw from load_darknet_weights leaks the FILE*. Header read + cout banner at 1902-1915 and 1936-1949 are verbatim twins apart from the message prefix. A unique_ptr<FILE,int(*)(FILE*)> helper is the natural fix; -14 LOC is…

#### core-utils-12 — variance() lacks the negative-variance clamp its sibling has; three different variance formulas

`opennn/core/statistics.cpp:228-255` · medium · bug · lines -6 · effort S · risk low · confirmed

variance(const VectorR&) uses the one-pass E[x^2]-E[x]^2 form in double with no clamp; variance(const VectorR&, const VectorI&) uses a third algebraic arrangement in long double; descriptives(matrix, rows, cols) (line 572-573) uses the same one-pass form but clamps with max(0.0, variance) -- the authors evidently hit the negative case. For a constant or near-constant column whose value is not exactly representable (0.1f repeated over 1e5 rows), cancellation between squared_sum and sum*sum/count yields a tiny negative number; variance() returns it, standard_deviation() takes sqrt and returns NaN, and vector_descriptives (line 516) -> descriptives(matrix) (tabular_dataset.cpp:807) hands NaN…

**Fix:** Add one file-local helper `double sample_variance(double sum, double squared_sum, Index count)` that returns max(0.0, (squared_sum - sum*sum/count) / (count-1)) (or a two-pass mean-then-deviation for the vector overload), and use it in all three places; delete the two ad-hoc formulas.

*Verifier:* statistics.cpp:228-240 variance(VectorR) returns `(squared_sum - sum*sum/count)/(count-1)` with no clamp; :242-255 variance(VectorR, VectorI) uses a third rearrangement in long double; :572-573 in descriptives(matrix,rows,cols) uses the same one-pass form but `sqrt(max(0.0, variance))`. tests/core/statistics_test.cpp:231/236 wrap variance() in abs(), consistent with the negative case having been…

#### nn-core-2 — to_JSON stages states based on the parameters' device, not the states' device

`opennn/neural_network/neural_network.cpp:1500-1501` · medium · bug · lines -6 · effort S · risk low · confirmed

to_JSON constructs HostStatesGuard with the predicate `parameters.get_device() == Device::CUDA`, but the guard moves *states*. Every GPU forward_propagate calls copy_states_device() (1311), while copy_parameters_host() (public; called by selection_utilities.cpp:70, loss.cpp:1879, numerical_derivatives.cpp:124) moves only parameters. So a GPU-trained network with BatchNorm whose parameters were staged to host and then save()d has states on CUDA and a guard that does not stage them; BatchNormalizationOperator::to_JSON calls running_mean.as_vector() (batch_norm_operator.cpp:144) which goes through require_host_fp32 and throws "requires CPU FP32 storage" - save() fails (and the transactional…

**Fix:** Use the one-argument constructor: `const HostStatesGuard guard(*const_cast<NeuralNetwork*>(this));` and delete the two-argument constructor (header 352-356), which then has no users. Add a test: compile on CUDA, forward once, copy_parameters_host(), save() on a BatchNorm network.

*Verifier:* Read neural_network.cpp 1497-1501, 1311, 2769-2782, 2783-2796, header 44-60 and 347-364, batch_norm_operator.cpp 137-148, tensor_types.h 670-672/713-717, loss.cpp 1874-1880, optimizer.cpp 1282-1306. The guard moves states but tests parameters' device. copy_parameters_host (2763-2782) migrates only parameters; every GPU forward_propagate calls copy_states_device (1311). A concrete path: GPU…

#### operators-b-1 — CPU attention backward infers padding from trailing zeros of head 0/row 0 - softmax underflow zeroes real gradients

`opennn/neural_network/operators/attention_operator.cpp:1074-1131` · medium · bug · lines -6 · effort S · risk low · partial

apply_delta_cpu decides the per-batch valid source length by calling infer_attention_prefix_length, which counts trailing exact-0.0f entries of the FIRST attention row (head 0, query 0) of each batch element. That is only a proxy for padding: softmax_rows_prefix/softmax produce exact zeros whenever exp(score - max) underflows float (logit gap > ~87, or ~104 without FTZ), which happens in sharply-peaked heads of trained models and in unstable training. Concrete failure: batch with no padding, head 0 query 0 attends sharply so its last 3 columns underflow to 0 while head 1 attends broadly. infer returns L = S-3, has_padding becomes true, the fast path runs for ALL heads with attention_valid =…

**Fix:** In back_propagate, recompute the lengths exactly as forward_propagate does (forward_propagation.input_sequence_lengths(layer, last_input).host, falling back to get_contiguous_source_lengths on the same source_input view) and pass the vector<Index> into apply_delta_cpu; take the fast path only when those lengths show padding. Delete infer_attention_prefix_length (declaration + 15-line definition). Add a CPU test: no padding, one head with a logit gap > 110 in row 0, compare dK/dV against the…

*Verifier:* Code as cited (attention_operator.cpp:110-121 infer_attention_prefix_length counts trailing exact 0.0f of head 0/query 0; apply_delta_cpu 1057-1131 takes the fast path when has_padding and zeroes key_delta/value_delta.bottomRows for every head). The mechanism is real: softmax_cpu (tensor_operations.cpp:1150-1158) and softmax_rows_prefix produce exact zeros when exp underflows, and nothing else…

#### dataset-a-3 — impute_missing_values_interpolate interpolates against a phantom point (sample 0, value 0)

`opennn/dataset/tabular_dataset.cpp:2519-2551` · medium · bug · lines -4 · effort S · risk low · confirmed

left_sample_index/right_sample_index/left_value/right_value default to 0, and 'no neighbour found' is indistinguishable from 'neighbour is sample 0 with value 0'. When the NaN is the first used value of a column, left stays (0, 0.0) and the code interpolates between (0,0) and the right neighbour; when it is the last, right stays (0, 0.0) and the formula extrapolates toward zero. Scenario: column [1, 2, 3, NaN] (used samples 0..3): left = (2, 3.0), right = (0, 0.0) -> interpolated = 3 + (3-2)*(0-3)/(0-2) = 4.5 instead of 3 (or 4). Column [NaN, 10, 10]: sample 0 is imputed 0 instead of 10, and because imputed values are written back immediately, later NaNs in the same column interpolate from…

**Fix:** Track the neighbours as optional<pair<Index,float>>; if both exist interpolate, if only one exists copy its value, if neither leave NaN (the row is then unused by unuse_samples_with_missing_targets / downstream). Read neighbours from a snapshot of the column so earlier imputations do not feed later ones. Add a unit test with leading, trailing and interior NaNs.

*Verifier:* tabular_dataset.cpp:2519-2551: left/right indices and values default to 0; the neighbour scans (2523-2537) leave them untouched when none is found; the branch at 2539-2548 then interpolates against (0, 0.0). Recomputed the scenarios: [1,2,3,NaN] gives 3 + (3-2)*(0-3)/(0-2) = 4.5; [NaN,10,10] gives 0. Values are written back at 2550 so later NaNs see imputed values. No test references…

#### operators-a-3 — A second compile() zeroes the sinusoidal positional-encoding table

`opennn/neural_network/operators/embedding_lookup_operator.cpp:279-285` · medium · bug · lines -4 · effort S · risk low · confirmed

NeuralNetwork::compile() resizes and setZero()s the state buffer, relinks states, then calls Operator::initialize_states() on every operator (neural_network.cpp:593-603). EmbeddingLookupOperator does not override initialize_states; it initializes the table only inside link_states, and only when positional_encoding.get_data() was null, i.e. on the very first link. After any later compile() (model_selection growing_neurons.cpp:129/242, inputs_selection.cpp:28, standard_networks.cpp:449/539/868/1080, or a user changing shapes and recompiling) the table is all zeros and the model silently loses positional information. Confirmed with a probe: Embedding(Shape{10,4}, 6) with…

**Fix:** Use the hook that exists for exactly this: override `void initialize_states() override { init_positional_encoding(); }` in EmbeddingLookupOperator, and reduce link_states to `if (!positional_trainable) link_views(views, {&positional_encoding});`. load_state_from_JSON can then drop its CUDA early-return special case since compile() already initialized the table on the host. Add a compile-twice assertion to embedding_layer_test.

*Verifier:* Read embedding_lookup_operator.cpp:279-285: `needs_init = !positional_encoding.get_data()` so only the first link initializes; neural_network.cpp:593-603 does `states.resize_bytes(...); states.setZero(); link_states();` then `op->initialize_states()` on every compile, and grep shows only BatchNormalizationOperator overrides initialize_states (operator.h:50 default is empty). On a second compile…

#### r2-duplicated-kernels-across-folders-4 — 16 raw CUDA copy/memset/sync sites outside device::, 13 unchecked; loss.cpp relies on legacy-stream cudaMemcpy

`opennn/training_strategy/loss.cpp:986-1038` · medium · bug · lines -4 · effort S · risk low · partial

Repo-wide count of raw cudaMemcpy*/cudaMemset*/cudaStreamSynchronize outside the sanctioned device_backend.cpp helpers (opennn/ only; tests/examples/docs hold 26 more in harness code): loss.cpp x10 (986 cudaStreamSynchronize, 987 cudaMemcpy D2H, 1014 cudaMemcpy D2H, 1036 cudaMemcpy H2D, 1145 cudaMemcpyAsync H2D on the compute stream, 1169 cudaMemsetAsync, 1198 cudaStreamSynchronize, 1200 cudaMemcpy D2H) - ALL unchecked; kernel_normalization.cu:1093-1094 cudaMemsetAsync x2 - unchecked, although device::set_zero_async is already declared in kernel_common.cuh:18 and used by kernel_pool3d.cu:163 and kernel_losses.cu:793; detection_layer.cpp:71-75 cudaMemcpyAsync H2D from a local pageable vector…

**Fix:** One mechanical PR: loss.cpp 987/1014/1036/1200 -> device::copy_async(dst, src, bytes, CopyKind::..., device::get_compute_stream()) followed by device::synchronize(stream) where the host reads the result (drops the separate raw cudaStreamSynchronize at 986/1198); 1145 -> device::copy_async(..., HostToDevice, stream); 1169 -> device::set_zero_async(error_accum, sizeof(float), stream); kernel_normalization.cu:1093-1094 -> opennn::device::set_zero_async; detection_layer.cpp:71 ->…

*Verifier:* Verified by grep over opennn/ (excluding device_backend.cpp): loss.cpp has 8 raw calls (986 sync, 987 D2H, 1014 D2H, 1036 H2D, 1145 async H2D, 1169 memsetAsync, 1198 sync, 1200 D2H), all unchecked; kernel_normalization.cu:1093-1094 two unchecked cudaMemsetAsync; detection_layer.cpp:71-75 one unchecked cudaMemcpyAsync from a pageable local vector; grouped_query_attention_layer.cpp:337…

#### xcut-build-tests-2 — OPENNN_HAS_CUDA is a non-FORCE cache set: a reconfigure keeps the stale CUDA decision

`CMakeLists.txt:93-134` · medium · bug · lines -2 · effort S · risk low · confirmed

`set(VAR value CACHE BOOL "")` without FORCE is a no-op when the cache entry already exists. Scenario A: configure a build dir with CUDA (cache ON), then re-run `cmake -DOpenNN_DISABLE_CUDA=ON .` in the same dir: line 133 does nothing, `OPENNN_HAS_CUDA` stays ON, `enable_language(CUDA)` is skipped, and opennn/CMakeLists.txt:191-197 still globs the .cu sources and links `CUDA::cudart` etc. -> generate-time error. Scenario B (worse, silent): a dir first configured CPU-only (cache OFF) later gets CUDA enabled: line 101 cannot flip it, the user gets a CPU build while believing CUDA is on. The same block also runs `check_language(CUDA)` even when OpenNN_DISABLE_CUDA=ON, paying the nvcc probe on…

**Fix:** Use plain normal variables: `set(OPENNN_HAS_CUDA ON)` / `set(OPENNN_HAS_CUDA OFF)` (subdirectories inherit them; nothing external reads the cache). Wrap the probe: `if(NOT OpenNN_DISABLE_CUDA) check_language(CUDA) endif()` and test `CMAKE_CUDA_COMPILER` only inside it.

*Verifier:* CMakeLists.txt:93-94 `include(CheckLanguage) check_language(CUDA)` runs unconditionally; line 96 tests `CMAKE_CUDA_COMPILER AND NOT OpenNN_DISABLE_CUDA`; line 103 `set(OPENNN_HAS_CUDA ON CACHE BOOL "")` and line 134 `set(OPENNN_HAS_CUDA OFF CACHE BOOL "")`, both non-FORCE, so the first configure's value sticks. Readers are all configure-time: opennn/CMakeLists.txt:191/282/333,…

#### dataset-a-4 — unuse_collinear_variables leaves input_shape stale; shape/role resync is hand-copied in 5 places

`opennn/dataset/tabular_dataset.cpp:619-683` · medium · bug · lines -2 · effort M · risk low · partial

input_shape/target_shape are cached separately from the variable roles. Every sibling that changes roles resyncs them by hand (set(): 522-523, unuse_uncorrelated_variables: 562-563, unuse_least_correlated_variables: 613-614, from_JSON: 1338-1339, Dataset::set_variable_indices: dataset.cpp 593-594) but unuse_collinear_variables (used by Neural Designer per the prior audit) does not. After it runs, get_shape(Input) still reports the old feature count while get_feature_indices(Input) is shorter; Batch::set sizes the input slot from get_shape("Input"), so fill_inputs writes fewer values than the slot holds and the network's declared input width disagrees with the batch. Any direct…

**Fix:** Add a protected virtual hook `on_variable_roles_changed()` in Dataset (mirroring on_used_samples_changed), call it from set_variable_role/set_variable_roles/set_variable_indices, and let TabularDataset implement it as the two-line shape resync. Delete the five hand-written resync pairs and the missing one is fixed for free. ImageDataset keeps its multi-dim shapes by not overriding.

*Verifier:* True: unuse_collinear_variables (619-683) ends at 'return unused_variables' with no shape resync, whereas unuse_uncorrelated_variables (562-563), unuse_least_correlated_variables (613-614), set (522-523), from_JSON (1338-1339) and Dataset::set_variable_indices (593-594) do; Batch::set sizes slots from dataset->get_shape(role) (batch.cpp:79-86), so the mismatch is real. Fix needs a different…

#### dataset-b-4 — YOLO cache accepted with a stale classes_number: .names file is not in the sources hash

`opennn/dataset/yolo_dataset.cpp:1505-1510` · medium · bug · lines -2 · effort S · risk low · confirmed

try_open_cache validates images and per-image label .txt files through hash_sources (191-248), which never looks at the .names file, then unconditionally takes classes_number from the target-cache header and only fills class_names if empty. class_names, however, was already read from the .names file in set() (line 1265) and get_classes_number() returns ssize(class_names). Scenario: a user adds a class line to voc.names (or swaps the file) without editing any label -> cache accepted -> member classes_number = old N (target_shape = {G,G,bpc*(5+N)}, cached targets have N class slots) while get_classes_number() = N+1, which examples/yolo/main.cpp:697 uses to size the network head ->…

**Fix:** In try_open_cache, reject the cache when `!class_names.empty() && Index(target_header.classes_number) != ssize(class_names)` (2 lines) so open_or_build_cache falls through to try_rebuild_target_from_boxes, which recomputes classes_number from class_names. Optionally also mix the .names file into hash_sources.

*Verifier:* set() reads class_names from the .names file (yolo_dataset.cpp:1265) before open_or_build_cache; hash_sources (191-248) mixes only image paths and label .txt files, never the .names file. try_open_cache at 1508-1510 then sets classes_number = target_header.classes_number and assign_default_class_names (127-134) is a no-op when class_names is non-empty, so member classes_number (used for…

#### nn-expression-3 — Embedded Logarithm scaling is unclamped; library computes log(max(x, EPSILON))

`opennn/neural_network/model_expression.cpp:1528-1543` · medium · bug · lines -2 · effort S · risk low · confirmed

The library's Scaling forward does `column.max(EPSILON).log()` (scaling_layer.cpp:59) and scale_value does `log(max(value, EPSILON))` (core/scaling.h), so an input of 0 or a negative value scales to log(1.19e-7) = -15.9. The generated nn_affine_flags_forward emits logf(inputs[i]) with no clamp, so the same input produces -inf/NaN in firmware and every downstream output becomes NaN. The exported model therefore disagrees with the network it came from exactly on the inputs (zeros, sensor dropouts) where a log scaler is most fragile. The same divergence exists in Scaling::write_expression's `log(x)` text for the four per-neuron exporters (out of this scope, same one-line fix there).

**Fix:** Emit `#define NN_LOG_EPSILON <c_float_literal(EPSILON)>` in the embedded prelude and change the snippet to `logf(inputs[i] > NN_LOG_EPSILON ? inputs[i] : NN_LOG_EPSILON)`. Apply the same clamp in Scaling::write_expression (`log(max(x, 1.1920929e-07))`; the C prelude already defines max()). Add a Logarithm feature with a zero input to the degenerate execution test.

*Verifier:* model_expression.cpp:1536 emits `logf(inputs[i])` with no clamp; scaling_layer.cpp:59 does `column.max(EPSILON).log()` and core/scaling.h:88 `log(max(value, EPSILON))`. Scaling::write_expression (scaling_layer.cpp:511) also emits an unclamped log(x). Divergence on inputs <= 0 is real; fix as proposed is sound (C prelude already defines max(), line 616).

#### selection-testing-4 — calculate_multiple_classification_rates writes out of bounds when targets have one column

`opennn/testing_analysis/testing_analysis.cpp:775-803` · medium · bug · lines -2 · effort S · risk low · confirmed

The rates tensor is sized targets.cols() x targets.cols() and each cell is resized to confusion(i,j). calculate_confusion treats a one-column problem as 2 classes (thresholded at 0.5), so for targets.cols()==1 the tensor is 1x1 and rates(0,0) is resized to confusion(0,0) == number of true positives only. The sample loop then uses maximal_index of a 1-element row, which is always 0, and writes all samples_number indices into rates(0,0). Scenario: binary targets/outputs (1 column), 100 testing samples, 60 true positives -> 40 writes past the end of a 60-element VectorI (heap overflow in release; Eigen assert in debug). The public wrapper calculate_multiple_classification_rates() forwards…

**Fix:** Add `throw_if(targets.cols() < 2 || outputs.cols() != targets.cols(), "calculate_multiple_classification_rates needs one column per class; use calculate_binary_classification_rates for one output.");` at the top of the (targets, outputs, indices) overload. Optionally size the tensor from confusion.rows()-1 so it cannot disagree with calculate_confusion.

*Verifier:* testing_analysis.cpp:776-803 read together with calculate_confusion (445-475): for outputs.cols()==1 num_classes is 2 and confusion is 3x3, but the rates tensor is sized targets_number x targets_number = 1x1 and rates(0,0) is resized to confusion(0,0) (true positives only). maximal_index (statistics.cpp:659-665) of a 1-element row is always 0, so every sample writes into rates(0,0); after TP…

#### training-optimizers-1 — Adam uses two unrelated step counters when CUDA graphs are on: device step for whole batches, host iteration for the tail

`opennn/training_strategy/adaptive_moment_estimation.cpp:142-185` · medium · bug · lines -2 · effort S · risk low · confirmed

With set_cuda_graph(true) the whole batches run through update_parameters(..., UpdateMode::Capturable) (optimizer.cpp:1438), whose bias correction comes from the device counter advanced by adam_prepare_kernel (kernel_optimizers.cu:150-151). The remainder batch runs through the Standard path (optimizer.cpp:1886, default mode), which advances and uses the host counter optimization_data.iteration instead. Both counters start at 0 after setup_optimizer_data (OptimizerData::set zeroes the slot buffer; iteration = 0), so the host counter only counts tails: one per epoch. Concrete scenario: 1000 training samples, batch 128 (7 whole batches + a 104-sample tail), Adam defaults. First epoch after…

**Fix:** Make the device counter the single source of truth whenever graph scalars exist: in AdaptiveMomentEstimation::update_parameters take the capturable branch when `mode == UpdateMode::Capturable || (neural_network->is_gpu() && optimization_data.views.size() > GraphScalars && optimization_data.views[GraphScalars].size() >= 4)`. The tail then calls adam_update_capturable_cuda too, which increments the same device step, and the host `iteration` is only used on CPU / non-graph GPU runs. Add a parity…

*Verifier:* adaptive_moment_estimation.cpp:142-177: Capturable branch calls adam_update_capturable_cuda with graph_step (device counter, incremented in kernel_optimizers.cu:150-151 adam_prepare_kernel); the Standard branch does optimization_data.iteration++ and derives bias_correction_1/2 from the host counter, then update_parameters_cuda on GPU. optimizer.cpp:1438 run_compute_step uses…

#### core-utils-4 — Global mutex-serialized RNG makes OpenMP callers slower than serial and non-reproducible under set_seed

`opennn/core/random_utilities.cpp:250-280` · medium · bug · lines -1 · effort S · risk low · partial

All random_* functions lock one global mutex around one mt19937. Any OpenMP region that calls them is serialized on the lock (contention cost, no parallel speedup) and, worse, the draw order depends on thread scheduling, so set_seed() no longer gives reproducible results. A concrete caller: TabularDataset::set_data_binary_classification (tabular_dataset.cpp:1472-1474) runs `#pragma omp parallel for` over samples calling random_bool() per element -- with set_seed(1) two runs produce different target columns, and the loop is slower than a plain loop because every iteration takes the mutex. The seeding-determinism problem is exactly what the prior audit hit with the unseeded…

**Fix:** Minimal: drop the `#pragma omp parallel for` at tabular_dataset.cpp:1472 (the loop is memory-trivial) and state in random_utilities.h that the per-draw API must not be called from parallel regions; the bulk fillers (set_random_*) already draw under one lock in deterministic order. If parallel draws are wanted later, give each thread a thread_local engine seeded from (global seed, omp_get_thread_num()).

*Verifier:* Line numbers are wrong: random_utilities.cpp is 186 lines; the mutex/generator are at :21-22 and random_bool at :46-51 (not 250-280). Substance holds: every per-draw function takes rng_mutex around one shared mt19937, and tabular_dataset.cpp:1472-1474 runs `#pragma omp parallel for` calling random_bool() per row, so draw order depends on scheduling and set_seed is not reproducible there;…

#### operators-a-5 — set_parameters_random/glorot throw when the network's states live on the device

`opennn/neural_network/operators/batch_norm_operator.cpp:123-135` · medium · bug · lines -1 · effort S · risk low · confirmed

BatchNormalizationOperator::set_parameters_random/glorot call init_defaults(), which calls initialize_states() and writes running_mean/running_variance through as_vector(). NeuralNetwork::initialize_parameters (neural_network.cpp:1037-1044) takes a HostParametersGuard only; states are not staged. After any GPU inference the states stay on CUDA (copy_states_device at neural_network.cpp:1311 and 2853; nothing copies them back until Optimizer::teardown_device_training), so a user or Neural Designer calling set_parameters_glorot() after predicting on the GPU hits `TensorView::as_vector requires host` from require_host_fp32_data and the re-initialization aborts. The CPU-only parameters are…

**Fix:** Stage both buffers for parameter initialization: add `const HostStatesGuard states_guard(*this);` next to the HostParametersGuard in NeuralNetwork::initialize_parameters (the guard already exists and is what to_JSON uses). Keep init_defaults as is. Add a test: compile, calculate_outputs on CUDA, then set_parameters_glorot() must not throw and must reset the running statistics.

*Verifier:* batch_norm_operator.cpp:123-135 init_defaults -> initialize_states -> running_mean.as_vector(), and tensor_types.h:672 as_vector calls require_host_fp32_data. neural_network.cpp:1037-1044 initialize_parameters takes only HostParametersGuard (neural_network.h:44-58, parameters only). calculate_outputs at neural_network.cpp:1311 and calculate_outputs_resident at 2853 call copy_states_device…

#### xcut-build-tests-1 — tinyml-parity workflow path filters point at files that no longer exist

`.github/workflows/tinyml-parity.yml:12-35` · medium · bug · lines 0 · effort S · risk low · confirmed

The push/pull_request `paths:` filters still use the pre-reorganisation flat layout (`opennn/model_expression.*`, `opennn/dense_layer.*`, `opennn/recurrent_layer.*`, `opennn/long_short_term_memory_layer.*`, `opennn/scaling_layer.*`, `opennn/unscaling_layer.*`). Those files now live under `opennn/neural_network/` and `opennn/neural_network/layers/` (verified with the source tree: only `opennn/neural_network/layers/clamping_layer.*` in the list is a real path). Consequence: a change to Dense/Recurrent/LSTM/ModelExpression, exactly the code whose C export the AVR/Cortex-M parity job exists to guard, never triggers the job; only example or clamping edits do. Silent loss of CI coverage.

**Fix:** Replace both path lists with `opennn/neural_network/model_expression.*`, `opennn/neural_network/layers/{scaling,unscaling,clamping,dense,recurrent,long_short_term_memory}_layer.*` (or simply `opennn/neural_network/**`). Consider a YAML anchor so the two lists cannot drift again.

*Verifier:* tinyml-parity.yml lines 12-21 and 24-33 list opennn/model_expression.*, opennn/scaling_layer.*, opennn/dense_layer.*, etc. Verified with ls: those files live at opennn/neural_network/model_expression.{h,cpp} and opennn/neural_network/layers/{scaling,unscaling,dense,recurrent,long_short_term_memory}_layer.{h,cpp}; only the clamping_layer path is real. A dense_layer edit cannot trigger the job. Fix…

#### xcut-build-tests-3 — airfoil_self_noise hard-codes Device::CUDA and therefore throws on every CPU-only build

`examples/airfoil_self_noise/main.cpp:32-32` · medium · bug · lines 0 · effort S · risk low · confirmed

`Configuration::resolve_effective` throws `"Configuration: CUDA requested but no GPU detected."` when Device::CUDA is requested without a GPU (opennn/core/configuration.cpp:66-69). The 12-neuron tabular airfoil example, the simplest example in the repo, requests Device::CUDA unconditionally, so on the CI configuration (OpenNN_DISABLE_CUDA=ON) or any laptop without an NVIDIA GPU it aborts before loading data. Every other tabular/image example uses Device::Auto or Device::CPU. gpt2/main.cpp:71 and yolo/main.cpp:303 also hard-code CUDA; gpt2 documents `--int8` as CUDA-only but its FP32 path could be Auto too. CI builds examples OFF, so nobody sees this.

**Fix:** Change airfoil to `Configuration::instance().set(Device::Auto, Type::FP32);` (one token). Do the same in gpt2 (INT8 branch keeps CUDA) and yolo unless GPU-only is intended, in which case print a clear message instead of the generic exception. Optionally build examples in the CPU CI job (OpenNN_BUILD_EXAMPLES=ON) so link/compile rot is caught too.

*Verifier:* examples/airfoil_self_noise/main.cpp:32 `Configuration::instance().set(Device::CUDA, Type::FP32);` and configuration.cpp:65-68 `case Device::CUDA: throw_if(!device::has_cuda_device(), "Configuration: CUDA requested but no GPU detected.")`. Grep of all examples: airfoil, gpt2:71, yolo:290 (not 303), blank/main_cuda hard-code CUDA; the rest use Auto/CPU. ci.yml configures with…

#### dataset-b-3 — 8-bit BMP: palette sized by biClrUsed but indexed by raw pixel byte (out-of-bounds read)

`opennn/dataset/image_processing.cpp:128-198` · medium · UB · lines 0 · effort S · risk low · confirmed

parse_bmp_header resizes h.palette to biClrUsed entries when biClrUsed is non-zero (e.g. 16 for a 16-colour image saved at 8 bpp). decode_bmp_pixels then does h.palette[row_ptr[x]] with the raw 0..255 pixel byte and no bound check. Scenario: an 8-bit BMP with biClrUsed=16 and any pixel value >= 16 (common in files from editors that write biClrUsed but do not clamp indices, and trivially in a corrupted file) -> vector operator[] out of range -> UB / heap over-read. The PNG palette path (line 401) does have the check; BMP does not.

**Fix:** Allocate the palette with 256 entries (`h.palette.assign(256, RGBQuad{})`) and fill only the first num_palette_colors from the file; unused indices then decode as black instead of reading past the vector. One-line change, no per-pixel branch.

*Verifier:* image_processing.cpp:128-149: h.palette.resize(num_palette_colors) with num_palette_colors = biClrUsed ? biClrUsed : 256; decode_bmp_pixels 184-198 indexes h.palette[row_ptr[x]] with the raw byte and no bound check in both the 1-channel and 3-channel branches. Any index >= biClrUsed reads past the vector. Filling a fixed 256-entry palette (assign(256, RGBQuad{})) and reading only…

#### dataset-a-5 — read_csv keeps stale sample roles/ids from a previous file (resize instead of assign)

`opennn/dataset/tabular_dataset.cpp:1822-1823` · medium · bug · lines 0 · effort S · risk low · confirmed

read_csv explicitly supports re-reading into the same object (it snapshots previous_variables to restore roles/scalers by name, lines 1741 and 1805-1820), but it sizes sample_roles with vector::resize, which preserves existing elements. split_samples_random only reassigns samples that are not None (dataset.cpp:350). Scenario: load file A (1000 rows), scrub_missing_values marks rows 5 and 7 None; call set(file_B, ...) or read_csv() on the same object with file B of 1000 complete rows -> rows 5 and 7 of B stay None and never train. sample_ids has the same problem (stale ids survive when has_sample_ids flips to false).

**Fix:** Use `sample_roles.assign(size_t(samples_number), SampleRole::Training);` and `sample_ids.assign(size_t(samples_number), {});` (binary mode then marks incomplete rows None as it already does). Add a test that re-reads a second file into a dataset whose roles were modified.

*Verifier:* tabular_dataset.cpp:1822-1823 use vector::resize for sample_roles and sample_ids; previous_variables snapshot at 1741 and the role/scaler restore loop at 1805-1820 prove re-reading into a live object is intended. Binary mode only marks rows None at 2145-2149 for rows with missing tokens; split_samples (dataset.cpp:350) skips None rows; so stale None roles from a previous file survive. assign()…

#### dataset-b-2 — impute_missing_values_unuse checks a lags+1 window but targets read past+future rows

`opennn/dataset/time_series_dataset.cpp:258-275` · medium · bug · lines 0 · effort S · risk low · confirmed

The NaN scan covers rows i..i+lags (lags+1 rows) and the tail marks only the last `lags` samples as None. fill_targets (line 355) reads row sample+past_time_steps+(future_time_steps-1), and the multi-target branch reads rows sample+past..sample+past+future-1; refresh_forecasting_roles (128-139) correctly uses window_span = past+future. Scenario: future_time_steps=2 (used by tests/dataset/timeseries_dataset_test.cpp:151,175,215 and the LSTM forecasting benchmark), a NaN at row r: sample r-lags-1 passes the check (its window ends at r-1) but its target row is r -> NaN target -> NaN loss. Likewise sample samples_number-lags-1 is kept although its target row is out of range (filled with 0.0f…

**Fix:** Use `const Index window_span = past_time_steps + future_time_steps;` as refresh_forecasting_roles does: num_sequences = samples_number - window_span + 1, scan `first, first + window_span`, and mark samples from num_sequences onward as None. Combine with dataset-b-1 in the same PR.

*Verifier:* Read 253-275 and fill_targets 355-392: single-target row = sample + past_time_steps + (future_time_steps - 1); multi-target rows sample+past .. sample+past+future-1. The scan covers only rows i..i+lags (any_of over lags+1 chars) and the tail unuses only N-lags.. N-1, whereas refresh_forecasting_roles (128-139) uses window_span = past+future and last_valid_start = hi - window_span + 1. With…

#### selection-testing-6 — GrowingInputs::set_maximum_inputs_number dereferences a null training strategy; ModelSelection::load() without a strategy segfaults

`opennn/model_selection/growing_inputs.cpp:72-79` · medium · bug · lines 0 · effort S · risk low · confirmed

GrowingInputs::set_maximum_inputs_number does `training_strategy->get_dataset()->get_variables_number(...)` unconditionally. The sibling GeneticAlgorithm::set_maximum_inputs_number (genetic_algorithm.cpp:69-76) null-checks both pointers. Scenario: `ModelSelection model_selection; model_selection.load("model_selection.json")` -> from_JSON -> set_inputs_selection("GrowingInputs") -> GrowingInputs::from_JSON -> set_maximum_inputs_number -> null dereference. Same for `GrowingInputs gi; gi.load(path)` or `gi.set_maximum_inputs_number(5)` before set(). Also note the two siblings clamp differently (GA clamps to [1, inputs], GI only to <= inputs), so a JSON MaximumInputsNumber of 0 is accepted by…

**Fix:** Copy the GeneticAlgorithm body: `const Dataset* dataset = training_strategy ? training_strategy->get_dataset() : nullptr; const Index inputs_number = dataset ? dataset->get_variables_number(VariableRole::Input) : 0; maximum_inputs_number = inputs_number == 0 ? new : clamp(new, Index(1), inputs_number);`. Better: move that single body to InputsSelection as a protected helper `clamp_to_input_count(Index)` and have both setters call it.

*Verifier:* growing_inputs.cpp:72-79 dereferences training_strategy->get_dataset() unconditionally. GeneticAlgorithm::set_maximum_inputs_number (genetic_algorithm.cpp:69-76) null-checks both and clamps to [1, inputs]. Reachable path: ModelSelection::from_JSON (model_selection.cpp:101-102) -> set_inputs_selection -> inputs_selection->set(training_strategy) (nullptr for `ModelSelection ms; ms.load(..)` since…

#### nn-builders-chat-4 — sample_token ignores repetition_penalty when temperature == 0; DecoderSampler applies it

`opennn/neural_network/chat.cpp:156-167` · medium · bug · lines 0 · effort S · risk low · confirmed

sample_token_with_workspace returns maximal_index(probabilities) before the repetition-penalty loop runs, so greedy decoding in the classic sessions (Transformer / TextGenerationNetwork, which route through this function at line 1339) never applies the penalty. DecoderSampler::sample_host (619-632), used by the generic session, applies the penalty first and then takes the argmax. Concrete scenario: ChatSession(text_generation_network).send(prompt, {.sampling = SamplingConfig{.temperature = 0, .repetition_penalty = 1.5}}) loops on the same token exactly as with penalty 1.0, while the same config on ChatSession(network, tokenizer, template) demotes repeats. No test pins the greedy+penalty…

**Fix:** Move the `temperature == 0` early return below the repetition-penalty loop (after line 167) so greedy and stochastic paths share the same penalised distribution, matching DecoderSampler::sample_host; add a test case with temperature 0 and a history token.

*Verifier:* chat.cpp:156-157 returns maximal_index(probabilities) before the penalty loop at 164-167. DecoderSampler::sample_host (616-631) applies the penalty first and only then the temperature==0 argmax. ClassicDecodeLoop::sample_at (1336-1341) routes classic sessions through sample_token_with_workspace, so greedy+penalty is a no-op there. tokenizer_layer_test.cpp:85-96 uses temperature 1.0 / top_k 1, so…

#### nn-expression-4 — JS category selector compares raw option values against sanitized output ids

`opennn/neural_network/model_expression.cpp:1905-1962` · medium · bug · lines 0 · effort S · risk low · confirmed

When a network has more than 5 outputs (use_category_select), the <option value="..."> is written with the raw output name but updateSelectedCategory() compares selectedCategory against fixes_output_names[i] (the replace_reserved_keywords form). Any output name that is not already a valid identifier ("Iris-setosa", "class A", "yes/no", or a reserved word) never matches, so the Value box stays empty after clicking calculate outputs. Multi-class classifiers with descriptive class labels are exactly the case that triggers the selector.

**Fix:** Write `<option value="` << fixes_output_names[i] << `">` << html-escaped output_names[i] << `</option>` so the value is the same id the script compares and the label stays human-readable.

*Verifier:* Line 1927 writes `<option value="` << output_names[i] (raw) while line 1962-1963 compares selectedCategory === fixes_output_names[i] and reads the hidden input with id fixes_output_names[i] (1911). Any output name altered by replace_reserved_keywords never matches. Fix is the obvious one-line change.

#### operators-a-4 — Running variance uses the biased estimator on CPU but the unbiased one on GPU

`opennn/neural_network/operators/batch_norm_operator.cpp:292-298` · medium · bug · lines 0 · effort S · risk medium · confirmed

apply_training_cpu folds the population (biased) batch variance into running_variance, and tests/neural_network/operators/batch_norm_operator_test.cpp:253 (RunningVarianceUsesBiasedEstimate) pins that convention with the rationale that changing it would rescale saved models. The library's own GPU kernel does the opposite on purpose: batchnorm_forward_finalize_kernel multiplies by `unbias = rows/(rows-1)` (kernel_normalization.cu:347, 430, comment at 323-325 'the running variance keeps the sample variance cuDNN's forward stores'), and both cuDNN paths store the sample variance too. Same model, same data, same seed: CPU and GPU training leave running variances that differ by M/(M-1) (14% at…

**Fix:** Pick one convention deliberately and apply it on both devices. Aligning the CPU to the GPU/cuDNN/PyTorch convention is one line (`* (N > 1 ? N / (N - 1.0f) : 1.0f)` on the variance term, with N = effective batch rows) plus flipping the pinned test; aligning the GPU instead means passing unbias = 1 in batchnorm_forward_fused_cuda and leaving the cuDNN paths mismatched, which is worse. Document the choice in the test comment either way.

*Verifier:* batch_norm_operator.cpp:292-297: running_variances fold `inverse_variances` which at that point holds the population (colwise().mean()) variance. kernel_normalization.cu:323-325 comment and :347 `var * unbias * momentum`, with :430 `unbias = rows/(rows-1)`, so the own kernel stores the sample variance on purpose; the cudnnBatchNormalizationForwardTraining fallback…

#### response-opt-4 — Univariate EqualTo on an output is filtered with a 1e-6 absolute band, so it is effectively infeasible

`opennn/response_optimization/response_optimization.cpp:82-97` · medium · bug · lines 0 · effort S · risk low · confirmed

set_constraint("y", EqualTo, v) on a target goes Domain::bound -> [v, v] -> filter_feasible_points -> filter_selected_indices_by_column(outputs, ..., v, v), which accepts only |y - v| <= 1e-6 absolute. A float network output landing inside a 2e-6 band by random sampling essentially never happens, so the loop prints 'Zero feasible points' and returns an empty matrix. Every other equality in this module uses a relative tolerance: formula constraints use bound_tolerance(bound) = max(EPSILON, |bound|*1e-4) (constraint_is_satisfied, build_linear_constraint_set), and Sense::Fixed objectives are expanded into a relative band (expand_fixed_objectives:2306-2310). Scenario: MinimalApproximation with…

**Fix:** Use the module's own tolerance: `value >= minimum - bound_tolerance(minimum) && value <= maximum + bound_tolerance(maximum)` in filter_selected_indices_by_column (move bound_tolerance's include if needed — response_optimization.h already includes response_constraints.h). Add a test for set_constraint(output, EqualTo).

*Verifier:* filter_selected_indices_by_column (82-97) uses a fixed 1e-6 absolute band; filter_feasible_points (1187-1194) feeds it the output Domain frontiers, which Domain::bound sets to [low_bound, low_bound] for EqualTo via interval_from_comparison (response_constraints.h:90). Formula constraints use bound_tolerance(bound)=max(EPSILON,|bound|*1e-4) in constraint_is_satisfied (1189-1193) and…

#### selection-testing-3 — Matthews correlation multiplies four Index counts; signed overflow at ~110k balanced testing samples

`opennn/testing_analysis/testing_analysis.cpp:869-873` · medium · UB · lines 0 · effort S · risk low · confirmed

matthews_denominator_squared = (TP+FP)(TP+FN)(FP+TN)(TN+FN) is computed in Index (int64). With n testing samples and roughly balanced classes each factor is ~n/2, so the product reaches 2^63 at n ~ 110,000 (n^4/16 > 9.22e18). Signed overflow is UB; in practice the value wraps negative, `== 0` is false, sqrt of a negative gives NaN and MCC is reported as NaN (or a garbage positive value if the wrap lands positive). 100k+ testing rows is routine for a 1M-row tabular dataset with a 10% test split. The numerator `true_positive * true_negative - false_positive * false_negative` overflows much later (n^2/4) and is fine.

**Fix:** Compute the denominator as `const double matthews_denominator = sqrt(double(tp_plus_fp) * double(tp_plus_fn) * double(fp_plus_tn) * double(tn_plus_fn));` and test `matthews_denominator == 0.0`; cast the numerator to double too. Same change, same lines, no API impact.

*Verifier:* testing_analysis.cpp:869-873 read; Index is Eigen::Index (opennn_types.h:137, ptrdiff_t/int64). The product of four ~n/2 terms exceeds 2^63 at n ≈ 110k balanced samples; signed overflow is UB and in practice gives a negative value whose sqrt() is NaN (the `== 0` guard does not catch it). The numerator TP*TN - FP*FN is ~n^2/4, safe. The double-based fix at the same lines is correct and has no API…

#### core-device-3 — Plan-cache key omits OPENNN_SDPA_AUTOTUNE: attention autotune silently never runs once the cache is warm

`opennn/core/cuda/cudnn_frontend_utilities.h:394-418` · medium · bug · lines +2 · effort S · risk low · confirmed

plan_cache_file hashes the conv knobs (workspace cap, conv autotune, candidate limits, heuristic modes, engine notes) but not sdpa_autotune_enabled(). finalize_attention calls load_cached_plan before looking at allow_autotune && sdpa_autotune_enabled(). Scenario: run 1 with OPENNN_SDPA_AUTOTUNE unset stores the heuristic-A plan for an SDPA graph; run 2 with OPENNN_SDPA_AUTOTUNE=1 loads that file under the identical key, returns false, and autotune_now is never reached - the A/B the knob exists for measures nothing and the documented 'measured neutral' claim cannot be distinguished from a warm cache. The comment in autotune_now ('or, worse, settling for the heuristic one under the same key')…

**Fix:** Fold sdpa_autotune_enabled() (and the allow_autotune flag, since only some attention graphs may tune) into the selection hash, e.g. selection = selection * 31 + (sdpa_autotune_enabled() ? 3 : 0) + (allow_autotune ? 5 : 0) passed through a small parameter to plan_cache_file; use hash_combine from tensor_types.h instead of the hand-rolled XOR/shift mixing while there.

*Verifier:* cudnn_frontend_utilities.h:404-416: selection hashes candidate limits, heuristic_modes(), conv_engine_notes(), workspace cap and conv_autotune_enabled() only; sdpa_autotune_enabled() (506-510) and the allow_autotune parameter are absent. finalize_attention (512-541) calls load_cached_plan at 519 before the `allow_autotune && sdpa_autotune_enabled()` gate at 523, and attention_operator.cpp:409/464…

#### r2-batch-pipeline-and-device-gather-2 — Batch::~Batch calls a throwing CUDA sync; after a sticky CUDA error unwinding ends in std::terminate

`opennn/dataset/batch.cpp:269-272` · medium · bug · lines +2 · effort S · risk low · confirmed

~Batch() calls wait_h2d_complete(), which calls device::synchronize_event -> CHECK_CUDA(cudaEventSynchronize) -> check_cuda_status throws runtime_error on any non-zero status. Destructors are implicitly noexcept, so the throw terminates the process. Concrete scenario: a kernel faults (sticky cudaErrorIllegalAddress) during a training step; check_last_error throws from forward/back-propagation; stack unwinding destroys the pool Batches (unique_ptr in BatchPools), the TrainingSession slots, or the local `Batch batch(tail_size, ...)` in evaluate_tail; each ~Batch with h2d_done_recorded == true calls cudaEventSynchronize, which returns the sticky error, CHECK_CUDA throws inside the destructor…

**Fix:** Make the destructor non-throwing: in ~Batch() call cudaEventSynchronize directly and ignore the status (or add a `device::synchronize_event_noexcept` used only by destructors), i.e. `Batch::~Batch() { if (h2d_done_recorded && h2d_done_event) (void)cudaEventSynchronize(h2d_done_event.get()); }` under OPENNN_HAS_CUDA. Keep the throwing wait_h2d_complete() for the worker path.

*Verifier:* batch.cpp 269-272 (~Batch calls wait_h2d_complete), 411-418 (wait_h2d_complete -> device::synchronize_event), device_backend.cpp 952-958 (CHECK_CUDA(cudaEventSynchronize)), opennn_types.h 87-95 (check_cuda_status throws runtime_error on any non-zero status). Destructor is implicitly noexcept; a sticky error during unwinding (pool batches in BatchPools, TrainingSession slots, the local Batch in…

#### dataset-a-1 — CsvReader trims tabs/spaces from every line, so TSV/space files lose leading/trailing empty fields

`opennn/dataset/field_parsing.cpp:29-35` · medium · bug · lines +2 · effort S · risk low · partial

CsvReader::parse applies trim_view to each line before tokenising, and trim_view's whitespace set is " \t\n\r\f\v\b". For Separator::Tab or Separator::Space (both reachable via set_separator/JSON "Separator", and the default for BertDataset/LanguageDataset which use the same reader), an empty first or last field is the separator itself and is stripped. Scenario: a tab-separated file with header "a\tb\tc" and row "1\t2\t" (c missing) tokenises to ["1","2"], so read_csv throws "Row N has fewer columns than expected (2)" for a valid file; a row "\t2\t3" (a missing) is read as ["2","3"] with the same error, or, when the file also has extra columns, silently shifts every value one column left.…

**Fix:** Only strip the carriage return; use trim_view solely to decide whether the line is blank (`if (trim_view(line).empty()) continue; out.lines.push_back(line);`). If trailing spaces on comma files must still be dropped, trim only the characters that are not the configured separator. Add a field_parsing_test case with a tab-separated row whose first and last field are empty.

*Verifier:* True: field_parsing.cpp:29-35 applies trim_view (string_utilities.cpp:220 whitespace set includes \t and space) before pushing the line; split_views (string_utilities.cpp:128-145) keeps empty tokens, so for Separator::Tab a row '1\t2\t' loses its trailing empty field and read_csv's required_tokens check (tabular_dataset.cpp:2119-2131, 2249-2256) throws 'fewer columns'. Tab is reachable via…

#### dataset-a-6 — Input-input correlations include unused (None) samples while input-target correlations exclude them

`opennn/dataset/tabular_dataset.cpp:942-964` · medium · bug · lines +2 · effort S · risk low · partial

calculate_input_target_variable_correlations gathers both columns with get_used_sample_indices() (893, 901, 906), but calculate_input_variable_correlations calls get_variable_data(Index) which returns data.block over every row (99-104). Samples excluded by filter_data, Tukey outlier handling, impute_missing_values_unuse (rows re-marked None) or by the user therefore influence the collinearity analysis (unuse_collinear_variables) but not the relevance analysis. Scenario: mark the 10% outlier rows None, then compare correlations(i,j) from the two entry points for the same pair of numeric inputs: the input-input value still reflects the outliers. TimeSeriesDataset's…

**Fix:** Gather with get_used_sample_indices() in calculate_input_variable_correlations (same call the twin already uses), and do the same in the time-series loops. Combine with dataset-a-7 so each column is gathered once.

*Verifier:* True for the tabular twins: calculate_input_target_variable_correlations gathers with get_used_sample_indices() (893, 901, 906) while calculate_input_variable_correlations (942-964) calls get_variable_data(Index) which is data.block over every row (99-104). Mis-scoped for time series: calculate_autocorrelations/calculate_cross_correlations (time_series_dataset.cpp:489, 532-540) operate on lagged…

#### selection-testing-2 — GrowingNeurons never reports MaximumEpochs: exhausting the loop leaves stopping_condition empty and elapsed_time blank

`opennn/model_selection/growing_neurons.cpp:211-230` · medium · bug · lines +2 · effort S · risk low · confirmed

The for-loop runs epoch < maximum_epochs, but the first_stopping_condition list only has MaximumTime, ValidationErrorGoal, MaximumValidationFailures and MaximumNeurons. elapsed_time and resize_history are only assigned inside `if (stopping_condition)`. Scenario: set_maximum_epochs(3), set_maximum_neurons(100), neurons_increment 1 -> three epochs run, loop ends normally, results.stopping_condition == nullopt and results.elapsed_time == "" (never set). The enum value GrowingNeurons::StoppingCondition::MaximumEpochs is declared (growing_neurons.h:28) but unreachable. GeneticAlgorithm handles the same case correctly with `{epoch >= maximum_epochs - 1, StoppingCondition::MaximumEpochs, ...}`…

**Fix:** Add `{epoch + 1 >= maximum_epochs, StoppingCondition::MaximumEpochs, format("Epoch {}\nMaximum number of epochs reached.\n", epoch)}` to the check list (mirrors genetic_algorithm.cpp:548), so elapsed_time and the history trim always run. Add a growing_neurons_test case with maximum_epochs=2 asserting stopping_condition == MaximumEpochs.

*Verifier:* growing_neurons.cpp:117 `for (Index epoch = 0; epoch < maximum_epochs; ++epoch)`; the first_stopping_condition list at 211-221 has only MaximumTime/ValidationErrorGoal/MaximumValidationFailures/MaximumNeurons; elapsed_time string and resize_history(epoch+1) only run inside `if (stopping_condition)` at 223-230. When the loop exhausts, results.stopping_condition stays nullopt, elapsed_time stays ""…

#### layers-a-2 — Embedding::set_input_shape is accepted for rank 1 but silently changes nothing

`opennn/neural_network/layers/embedding_layer.h:26-29` · medium · bug · lines +2 · effort S · risk low · partial

Embedding declares accepts_input_rank(1) and overrides get_input_shape() to return {sequence_length}, but it does not override apply_input_shape. The base default (layer.h:213-216) assigns the protected input_shape member, which Embedding never reads. So Embedding e(Shape{100, 7}, 8); e.set_input_shape(Shape{12}); e.get_input_shape() still returns {7} and embedding_lookup keeps sequence_length 7. NeuralNetwork::set_input_shape (neural_network.cpp:775-800) propagates shapes layer by layer via set_input_shape, so a Tokenizer->Embedding network resized to a new sequence length keeps the old embedding geometry and fails later with a shape mismatch in propagation, far from the cause.…

**Fix:** Add in embedding_layer.h: void apply_input_shape(const Shape& s) override { set(vocabulary_size, s.dim_or_zero(0), embedding_dimension, label); }. Layer::from_JSON calls set_input_shape before read_JSON_body, so the override runs with vocabulary_size/embedding_dimension still 0 and read_JSON_body then sets the real values (embedding_lookup.set(0, seq, 0) is what the default constructor already does). Extend the LayerInputShape Embedding test to assert get_input_shape() == Shape{new_len} after…

*Verifier:* True in substance: embedding_layer.h:26-29 declares accepts_input_rank(1) and get_input_shape() returning {sequence_length}, there is no apply_input_shape override in the header (only set/on_compute_dtype_changed/read/write JSON), so the base default at layer.h:213-216 writes the unused input_shape member; embedding_lookup.set(vocabulary_size, sequence_length, ...) in Embedding::set…

#### layers-b-3 — Recurrent back_propagate indexes scratch slots that do not exist for a frozen layer

`opennn/neural_network/layers/recurrent_layer.cpp:103-134` · medium · UB · lines +2 · effort S · risk low · confirmed

Recurrent::get_backward_specs returns {} when !is_trainable, and BackPropagation sizes slots[i] to backward_specs[i].size() + 1 (back_propagation.cpp:520), i.e. one entry. Loss::back_propagate (loss.cpp:1785-1795) calls back_propagate on every layer between the first and last trainable layer, so a Recurrent layer with set_is_trainable(false) sitting between two trainable layers (Trainable=false is a public JSON field read in Layer::from_JSON) executes backward_slots[SequenceDeltaScratchSlot] (index 7) / backward_slots[StepInputScratchSlot..CudnnInputDeltaScratchSlot] on a size-1 vector: out-of-bounds vector access, UB. LongShortTermMemoryOperator::back_propagate guards exactly this (`if…

**Fix:** Add the same early return at the top of RecurrentOperator::back_propagate: if (backward_slots.size() <= CudnnInputDeltaScratchSlot) return;. Add a test that freezes a Recurrent layer between two Dense layers and runs one back-propagation.

*Verifier:* RecurrentOperator::back_propagate (recurrent_layer.cpp:101-134) indexes backward_slots[StepInputScratchSlot..CudnnInputDeltaScratchSlot] (CUDA) and backward_slots[SequenceDeltaScratchSlot] (CPU, line 133) unconditionally; only input_delta goes through slot_or (112-113). Recurrent::get_backward_specs returns {} when !is_trainable (692-694), BackPropagation sizes slots[i] to…

#### nn-expression-5 — Python emitter bypasses fix_names: unnamed inputs become `variable`, and fix_names never dedupes

`opennn/neural_network/model_expression.cpp:2101-2107` · medium · bug · lines +2 · effort S · risk low · confirmed

build_expression names an empty input `input_{i}` and the C/JS/PHP emitters derive the same identifier through fix_names(..., "input_"). emit_python_calculate_outputs instead runs replace_reserved_keywords on the raw names, which turns an empty name into the literal `variable`. A network built like examples/forecasting_tinyml (`set_input_variables(vector<Variable>(FEATURES))`, no set_input_names) exported to Python therefore emits `variable = inputs[0]`, `variable = inputs[1]` while the body references input_0/input_1: NameError on first call. Independently, replace_reserved_keywords drops every character it does not know ('(', ')', '[', ']', '%', non-ASCII), so "Temp (C)" and "Temp [C]"…

**Fix:** Use `fix_names(input_names, "input_")` in emit_python_calculate_outputs (delete python_mapped) and in emit_python_class_header. In fix_names, keep an unordered_set of emitted identifiers and append `_<i>` on collision (5 lines). Add a test exporting a network with two unnamed inputs to Python and one with "Temp (C)"/"Temp [C]" to C.

*Verifier:* Lines 2103-2107: python_mapped = replace_reserved_keywords(raw) and replace_reserved_keywords returns "variable" for an empty name (2244-2245), while build_expression (340-344) names empty inputs input_{i} and fix_names (2335-2338) uses default_prefix+i. The same python_mapped is passed to process_body_line (2113), so body references stay input_N -> NameError.…

#### nn-builders-chat-3 — DarknetTinyV3+FPN accepts 9 anchors but is 2-head: 6 anchors land on a 3-anchor logits conv

`opennn/neural_network/standard_networks.cpp:483-484` · medium · bug · lines +2 · effort S · risk low · confirmed

The FPN validation admits 6 or 9 anchors and says '9 anchors (3-head)', but the DarknetTinyV3 branch is the only 2-head FPN and slices anchors_large = [begin+3, end) — with 9 anchors that is 6 anchors — while add_det_head always builds the logits conv with 3*(5+classes) channels. DetectionOperator::set (detection_layer.cpp:39-44) then requires channels % boxes_per_cell == 0 and classes = channels/boxes - 5 > 0: for classes in {1,2,3,4,5,6} it throws with the unrelated message 'channels must be divisible by boxes_per_cell' / 'classes_number must be positive'; for odd classes >= 7 (e.g. 7: 36/6-5 = 1) it passes and silently builds a large head with 6 boxes of 6 channels and the wrong class…

**Fix:** In the DarknetTinyV3 FPN branch add `throw_if(ssize(anchors) != 6, "YoloNetwork: DarknetTinyV3 FPN (2-head) requires exactly 6 anchors.")` (mirroring lines 966 and 1137) and drop the '9 anchors' alternative from the generic FPN message, or make add_det_head size the logits conv from head_anchors.size() instead of the literal 3.

*Verifier:* 483-484 admits 6 or 9 anchors for any FPN backbone; DarknetTinyV3 branch (669-690) has no anchor-count check and slices anchors_large = [begin+3, end), i.e. 6 anchors when 9 are passed. add_det_head (538-556) always builds 3*(5+classes) logits channels. DetectionOperator::set (detection_layer.cpp:36-44) sets boxes_per_cell = 6 and checks channels % 6 == 0 and classes = channels/6 - 5 > 0: for…

#### response-opt-3 — Domain::reshape pins Binary scalar inputs to 1 whenever any nearby point had a 1

`opennn/response_optimization/response_optimization.cpp:1139-1154` · medium · bug · lines +2 · effort S · risk medium · confirmed

reshape routes Binary scalars into the categorical (one-hot) branch. categories_to_save(c) = max(colwise max over points, center) is then written to BOTH frontiers: inferior = max(cts, inferior), superior = min(cts, superior). For one-hot columns this is harmless because sample_categorical only reads superior_frontier (>0.5). For a Binary scalar, sample_scalar reads both (lo = ceil(inferior), hi = floor(superior)), so the box becomes [1,1] as soon as a single point in points_inputs (the 85%-nearest subset in single-objective, best+Pareto in multi-objective) has value 1 — i.e. practically always after iteration 1 — and it never reopens, because the exploit rows are now all 1. Scenario:…

**Fix:** For Binary scalars keep the values actually seen: inferior = max(inferior, colwise min over points (and center)), superior = min(superior, colwise max) — or simply treat Binary like Integer in the continuous branch (the lattice snap already handles it). Add a test with a plain Binary input whose optimum is at 0 and assert the returned point has b=0.

*Verifier:* reshape (1139-1154): Binary scalars are excluded from the continuous branch and take the categorical branch where inferior = max(categories_to_save, inferior) with categories_to_save = max(colwise max over points_inputs, center) (1128-1132). sample_scalar for Binary reads lo = ceil(inferior), hi = floor(superior) and clamps the rounded draw into [lo,hi] (881-885). Since inferior is only ever…

#### xcut-build-tests-4 — yolo example leaves std::cout pointing at a destroyed TeeBuf on any exception path

`examples/yolo/main.cpp:364-369` · medium · UB · lines +3 · effort S · risk low · confirmed

`log_file` and `tee_buf` are locals inside the `try` block; `cout.rdbuf(&tee_buf)` redirects the global stream and the original buffer is restored only at the successful end of main (line 1806 `cout.rdbuf(old_rdbuf); return 0;`). When anything throws (e.g. running `yolo v3-pretrained` without a dataset: YoloDataset ctor throws), stack unwinding destroys `tee_buf` and `log_file` while `cout` still holds the dangling streambuf pointer. The catch block uses cerr, but at exit `ios_base::Init` flushes `cout`, which calls `pubsync()` on the destroyed `TeeBuf` -> use-after-scope UB (crash or hang on exit instead of a clean error). Any library `cout <<` during unwinding hits the same dangling…

**Fix:** Make the redirect RAII: `struct RdbufGuard { streambuf* old = cout.rdbuf(); ~RdbufGuard(){ cout.rdbuf(old); } } guard;` declared right after `tee_buf` (so it is destroyed before it), and delete the manual restore at line 1806.

*Verifier:* Line range is 362-369 (log_file at 362, TeeBuf 363-368, `auto* old_rdbuf = cout.rdbuf(&tee_buf)` at 369), all inside the `try` opened at 284; restore only at 1806 before `return 0`, catch at 1809 returns 1 without restoring. Unwinding destroys tee_buf/log_file while cout still points at them; the ios_base::Init flush at exit calls into the dead streambuf. RAII guard fix is correct (also note…

#### core-utils-5 — JSON number dump casts double to long long before range check; NaN/inf become unparsable tokens

`opennn/core/json.cpp:226-239` · medium · UB · lines +3 · effort S · risk low · confirmed

dump_value computes `static_cast<long long>(number)` unconditionally and only afterwards tests `std::abs(number) < 1e15`. For NaN, +/-inf or |x| >= 2^63 the cast itself is undefined behaviour (the check is too late). In the non-integer branch std::to_chars writes "nan"/"inf", which is not JSON and which this library's own parser rejects (parse_value, json.cpp:417-428: 'n' only matches "null"). Any float field that turns NaN -- e.g. a user-set LearningRate/DropoutRate/MinRange (adaptive_moment_estimation.cpp:222, dropout_operator.cpp:133, scaling_layer.cpp:426) or a divergent metric stored through Json(float) -- produces a model file that save() writes happily and load() refuses with…

**Fix:** Reorder: `if (!std::isfinite(number)) { out += "null"; return; }` first, then `if (std::abs(number) < 1e15 && number == std::trunc(number)) snprintf("%lld", (long long)number)` else to_chars. Add a test that Json(NAN).dump() round-trips through Json::parse.

*Verifier:* json.cpp:226-237: `static_cast<long long>(number)` executes before the `< 1e15` check, so NaN/inf/|x|>=2^63 hit UB (Json(1e300).dump()). In the else branch std::to_chars emits "nan"/"inf", and parse_value (json.cpp:417-428) only accepts 'n' as "null", so the round-trip fails with 'unexpected character n'. Json(float) ctor (json.h:46) makes every float field a Number; dropout_operator.cpp:131-134…

#### core-utils-3 — calculate_rank sorts with a NaN-unsafe comparator; NaN correlations reach it

`opennn/core/statistics.cpp:690-702` · medium · UB · lines +3 · effort S · risk low · confirmed

calculate_rank feeds std::sort (sort_parallel_if_large) a comparator `vector[i] < vector[j]`. With a NaN in the data the comparator is not a strict weak ordering (NaN is 'equivalent' to every value while those values are ordered), which is undefined behaviour for std::sort; libstdc++'s unguarded insertion sort can walk off the array and crash, MSVC debug asserts 'invalid comparator'. NaNs do arrive: correlations.cpp:444-449 and :206-211 return coefficient = QUIET_NAN for a constant column or a non-positive target, get_correlation_values (correlations.cpp:221-231) copies them unchanged, and TabularDataset::calculate_correlations_rank (tabular_dataset.cpp:980-987) takes rowwise().mean()…

**Fix:** Make the comparator total: treat NaN as the worst rank, e.g. `const auto key = [&](Index i){ return isnan(vector[i]) ? (ascending ? POS_INFINITY : NEG_INFINITY) : vector[i]; }; ... key(i) < key(j)` (and tie-break on index for determinism), or throw_if the input has NaN. Add a test with a NaN entry.

*Verifier:* statistics.cpp:690-702 comparator is `vector[i] < vector[j]` passed to sort_parallel_if_large -> std::sort (parallel_algorithms.h:40-48). NaN reaches it: correlations.cpp:209-211 and :444-449 return coefficient = QUIET_NAN; get_correlation_values (:221-231) copies coefficients; tabular_dataset.cpp:980-986 takes rowwise().mean() of abs values and calls calculate_rank; genetic_algorithm.cpp:137-140…

#### nn-core-3 — upload_parameters_bf16_inference silently migrates a CPU-configured network's parameters to the GPU

`opennn/neural_network/neural_network.cpp:2632-2642` · medium · bug · lines +3 · effort S · risk low · confirmed

When its preconditions fail, upload_parameters_bf16_inference falls back to copy_parameters_device(), which has no `config.device == Device::CUDA` check and unconditionally does parameters.migrate_to(Device::CUDA) + link_parameters(). On a CUDA build, a network compiled for Device::CPU (Configuration default) whose user calls upload_parameters_bf16_inference()/upload_parameters_int8_inference() (the documented inference entry points, see examples/gpt2/main.cpp:86) ends up with its parameter views pointing at device memory; the next CPU forward dereferences them on the host and segfaults instead of reporting the misconfiguration. The non-CUDA stubs are inconsistent among themselves too:…

**Fix:** At the top of upload_parameters_bf16_inference add `throw_if(config.device != Device::CUDA, "NeuralNetwork::upload_parameters_bf16_inference: the network is compiled for the CPU.");` and give copy_parameters_device the same guard (it is only meaningful for CUDA-configured networks; every in-repo caller already satisfies it). Make the two silent non-CUDA stubs use OPENNN_CUDA_STUB_BODY like their sibling so CPU builds fail loudly and identically.

*Verifier:* Read neural_network.cpp 2508-2547 (copy_parameters_device: no config.device check, migrate_to CUDA + link_parameters), 2632-2642 (fallback to copy_parameters_device when config.device != CUDA), 2957-2965 (stubs: two silent no-ops and one OPENNN_CUDA_STUB_BODY), 1290-1296/1401 (CPU forward path taken when !is_gpu(), no device check on the views), examples/gpt2/main.cpp:85-86. All in-repo callers…

#### core-device-4 — plan_cache_directory uses the throwing temp_directory_path: a bad TMP disables the cuDNN frontend for the process

`opennn/core/cuda/cudnn_frontend_utilities.h:376-391` · medium · bug · lines +4 · effort S · risk low · confirmed

Every other filesystem call in the cache path uses the error_code overload, but plan_cache_directory() calls std::filesystem::temp_directory_path() (throwing) inside a static initializer. With TMP/TMPDIR pointing at a nonexistent or unreadable directory (common on locked-down servers and CI images) it throws filesystem_error; plan_cache_file -> load_cached_plan -> finalize -> GraphSlot::build propagate it into the conv/BN run_frontend body, which catches, sets cache.disabled = true permanently and prints 'cudnn-frontend path unavailable'; the layer then calls throw_frontend_unavailable with a message blaming a missing plan/workspace. This contradicts the cache's own contract ('a bad cache…

**Fix:** Use temp_directory_path(error_code) and return an empty path on failure; make load_cached_plan/store_cached_plan return early when plan_cache_directory().empty() (or fold that into plan_cache_enabled()). Print one 'plan cache disabled: <reason>' line so the slowdown is explained.

*Verifier:* cudnn_frontend_utilities.h:376-391: `std::filesystem::temp_directory_path()` (throwing overload) inside the static initializer at 383, while load_cached_plan/store_cached_plan (420-476) use error_code overloads everywhere else. run_frontend (113-129) catches any exception, sets cache.disabled = true permanently and prints 'cudnn-frontend path unavailable'; convolution_operator.cpp:626/758 then…

#### layers-a-14-extra-1 — YoloNetwork with BodyActivation::SiLU builds Identity convolutions on non-V8 backbones

`opennn/neural_network/standard_networks.cpp:494-580` · medium · bug · lines +4 · effort S · risk medium · verifier-added

The YOLO builder resolves act to "SiLU" when body_activation == BodyActivation::SiLU and passes it directly into Convolutional for the Vgg, DarknetTiny and Darknet53 paths (add_conv at 566, add_yolo_neck 576-580, 609, 619, 652, 678-689). Convolutional::set demotes SiLU to Identity with only a std::cerr warning (convolutional_layer.cpp:246-260), and none of those call sites append an Activation layer (only the residual block at 570 and the V8 add_cba block at 722-726 do). So a Darknet53+SiLU network silently loses the nonlinearity of every neck/lateral conv, trains, saves and exports as a mostly-linear network with no exception. Additionally…

**Fix:** Make add_conv in the YOLO builder split Conv(Identity)+Activation(act) whenever activation_needs_input(from_string(act)) (the add_cba lambda at 722-726 already does this for V8), then let Convolutional::set throw like set_activation_function does (layers-a-14). Add a YoloNetwork test that builds Darknet53 with BodyActivation::SiLU and asserts every conv block is followed by a SiLU Activation layer.

*Verifier:* found by verifier

#### training-optimizers-7 — set_display_period(0) / set_validation_period(0) / JSON DisplayPeriod 0 cause integer modulo by zero

`opennn/training_strategy/optimizer.h:52-74` · medium · UB · lines +4 · effort S · risk low · confirmed

Both setters store the value unchecked. should_display (optimizer.h:203) computes `epoch % display_period` and train() computes `epoch % validation_period` (optimizer.cpp:956). A zero reaches them from set_display_period(0), set_validation_period(0), or a saved file with "DisplayPeriod": 0 (read_common_json, optimizer.cpp:1278-1279), and the first epoch raises SIGFPE / integer division by zero, which is UB. Negative values silently disable display/validation. Every other numeric setter on the siblings that has a domain (set_beta_1/2) validates.

**Fix:** throw_if(new_display_period <= 0, "Optimizer::set_display_period: period must be positive.") and the same in set_validation_period (move both bodies to optimizer.cpp or keep inline, +2 lines each). read_common_json already routes through the setter so JSON input is covered.

*Verifier:* optimizer.h:52 and :74 store unchecked; should_display at optimizer.h:203 does `epoch % display_period`; optimizer.cpp:956 does `epoch % validation_period`; read_common_json (optimizer.cpp:1278-1279) routes JSON DisplayPeriod through set_display_period. No caller passes 0 today (grep examples/, docs/benchmarks/, tests/), but nothing prevents it. Fix respects the throw_if convention used elsewhere…

#### r2-set-vs-compile-device-ordering-5 — DReLU wiring is reset one-sided: reconfiguring the consumer after compile drops the producer's ReLU backward

`opennn/neural_network/layers/dense_layer.cpp:295-302` · medium · bug · lines +6 · effort S · risk low · unverified

wire_drelu_fusions (compile time, OPENNN_DRELU_FUSION=1) sets producer.combination.emit_relu_mask, producer.activation_operator.backward_fused_by_consumer and consumer.combination.drelu_source (dense_layer.cpp:262-265). Dense::configure_operators calls reset_drelu_fusion(), which clears only the layer's own fields. Scenario: CUDA network compiled with the env flag, then `consumer->set_activation_function("Tanh")` (or any consumer setter that does not change parameter counts, so the 'call compile() again' guard stays silent). The consumer now has drelu_source == nullptr and runs a plain linear_backward; the producer still emits the mask in its forward, so drelu_fused_by_layer[producer] == 1…

**Fix:** Keep the producer pointer as `Dense* drelu_producer` (instead of only the CombinationOperator*) and make reset_drelu_fusion clear the producer side too (`drelu_producer->combination.emit_relu_mask = false; drelu_producer->activation_operator.backward_fused_by_consumer = false;`). Alternatively derive both sides from one source of truth: the producer's forward should only set drelu_fused_by_layer when a consumer is registered, and the consumer's backward should clear it when it does not use the…

#### core-kernels-1 — GPU sampler silently ignores every logit beyond index 262,144 (no host check)

`opennn/core/cuda/kernel_attention.cu:677-801` · medium · bug · lines +8 · effort S · risk low · partial

logits_top_candidates_kernel<T, 8> is launched with LOGITS_SAMPLE_BLOCKS (128) blocks of SAMPLING_BLOCK_THREADS (256) threads, so the grid-stride is 32,768 and each thread stops collecting after SLOTS = 8 hits. Logits at positions >= 8 * 32,768 = 262,144 are never examined, and neither sample_logits_row_cuda nor chat.cpp's fast_gpu guard checks the vocabulary against that cap. Failure scenario: a model whose vocabulary exceeds 262,144 (e.g. a 300k-token tokenizer) decodes on the GPU fast path; whenever the true argmax or a top-k candidate lies in the dropped tail, the kernel returns a different token with no error. Today's shipped vocabularies (Qwen3 151,936) sit under the cap, which is why…

**Fix:** In sample_logits_row_cuda add checked_host_condition(n > LOGITS_SAMPLE_BLOCKS * SAMPLING_BLOCK_THREADS * 8, "sample_logits_row_cuda: vocabulary above 262144 is not supported.") and a static_assert(LOGITS_SAMPLE_BLOCKS * 32 <= SAMPLING_BLOCK_THREADS * 16) beside the two launches; expose the cap as a constexpr in kernel_attention.cuh so chat.cpp's fast_gpu condition can fall back to the host sampler instead of throwing.

*Verifier:* Cap is real: kernel_attention.cu:21 SAMPLING_BLOCK_THREADS=256, kernel_attention.cuh:74 LOGITS_SAMPLE_BLOCKS=128, and logits_top_candidates_kernel (lines 678-691) stops each thread at SLOTS=8 hits with stride gridDim*blockDim=32768, so indices >= 262144 are never read. sample_logits_row_cuda (790-801) and chat.cpp:521-524 fast_gpu guard check only temperature/top_k/repetition_penalty, never n;…

#### operators-b-2 — CPU causal/dropout attention ignores exported sequence lengths the CUDA path honours - CPU/GPU divergence after layer 1

`opennn/neural_network/operators/attention_operator.cpp:653-787` · medium · bug · lines +10 · effort S · risk low · partial

apply_unfused only consults explicit_lengths.host inside use_cpu_fast_path, which requires !use_causal_mask && !dropout.active(). Every causal decoder layer (and any layer with attention dropout) on CPU therefore falls into the generic branch, which masks padding exclusively by scanning source_input for all-zero rows (row_nonzero). As the file's own comment on the SDPA path explains, that scan is only valid for the first attention layer: after one normalization the padded rows are no longer zero. On CUDA the same configuration takes attention_length_masked_softmax_cuda with explicit_lengths.device, so padding is masked in every layer. Concrete failure: a 2+ layer causal Transformer with…

**Fix:** In the generic CPU branch, when explicit_lengths.host is present and sized batch_size, mask columns >= lengths[batch] (and, for zero_padded_queries, zero query rows >= lengths[batch]) instead of the zero-row scan; keep the scan only as the no-record fallback. This mirrors the CUDA kernel selection two lines above. Add a CPU-vs-CUDA parity test with causal mask + padded batch + 2 layers.

*Verifier:* True as read: apply_unfused 652-681 only consults explicit_lengths.host inside use_cpu_fast_path (requires !use_causal_mask && !dropout.active()); the generic CPU branch 776-793 masks via row_nonzero scan of source_input and the zero_padded_queries block 799-820 does the same; CUDA path 742-755 uses explicit_lengths.device in all configurations. The file's own SDPA comment (882-888) confirms the…

#### xcut-boilerplate-1 — 115 strictly trivial getters/setters still defined out-of-line in 25 .cpp files (~400 lines)

`opennn/dataset/dataset.cpp:227-235` · medium · boilerplate · lines -350 · effort M · risk low · partial

Pattern (a). An awk pass over every .cpp found 115 member functions named get_/set_/is_/has_ whose body is <= 2 non-blank lines with no lambda, loop or algorithm call (185 with a <= 3-line threshold). Per file (strict count): dataset.cpp 17, standard_networks.cpp 16, response_optimization.cpp 13, neural_network.cpp 11, tabular_dataset.cpp 5, tokenizer_layer.cpp 4, pooling_layer.cpp 4, multihead_attention_layer.cpp 4, clamping_layer.cpp 4, dense_layer.cpp 3, variable.cpp 3, training_strategy.cpp 2, training_result.cpp 2, loss.cpp 2, pooling_layer_3d.cpp 2, convolutional_layer.cpp 2, model_selection.cpp 2, and 1 each in registry.cpp, tokenizer_operator.cpp, c2psa_operator.cpp,…

**Fix:** One mechanical PR per folder: move each strict-trivial body into the class declaration as an inline one-liner (virtuals included; they may be defined in-class), following the existing precedent in dataset.h:62-124 and optimizer.h:46-52. Skip the few whose body needs a type the header only forward-declares (e.g. get_tokenizer_layer wrappers stay if tokenizer_layer.h is not already included). No behaviour change; verify with both build dirs.

*Verifier:* Pattern is real: dataset.cpp:227-235 (get_samples_number(SampleRole), get_used_samples_number) are one-liners while dataset.h:62-63 inlines the string_view overloads that forward to them; response_optimization.cpp:379-402 has five one-line setters; neural_network.cpp:1046-1058 has three one-line initialisers. My own awk pass (body <=2 non-blank lines, no loop/lambda/ranges) found 104 such…

#### xcut-build-tests-7 — 177 Configuration::set calls in tests are redundant with the per-test listener reset

`tests/test.cpp:20-27` · medium · boilerplate · lines -174 · effort S · risk low · partial

`CpuConfigurationListener::OnTestStart` already sets Device::CPU/Type::FP32, seeds, and resets the device error before every test. Yet 96 tests still open with `Configuration::instance().set(Device::CPU, Type::FP32);` as their first statement and 81 tests end with a trailing `Configuration::instance().set();` restore (counted with a script over tests/**/*.cpp). The prior audit removed 28 restores and counted 62 leading sets; the remaining ones are the same pattern (e.g. neural_network_test.cpp:668/672, memory_audit_test.cpp:107/113/119, quasi_newton_method_test.cpp:91). They are pure noise and invite the belief that tests must restore state themselves.

**Fix:** Delete every leading `Configuration::instance().set(Device::CPU, Type::FP32);` that is the first statement of a TEST/TEST_F body and every trailing `Configuration::instance().set();` (keep mid-test switches back to CPU after a CUDA section). Mechanical sed plus a build of both dirs.

*Verifier:* tests/test.cpp:20-27 listener confirmed. Counts: 131 total `set(Device::CPU, Type::FP32);` lines in tests, of which 93 (not 96) are the first statement after a TEST/TEST_F opening brace (awk over all files); 81 trailing `Configuration::instance().set();` confirmed. Corrected delta about -174. Mechanical removal is sound; mid-test resets after CUDA sections must be kept as the finding says.

#### nn-expression-13 — Four per-neuron exporters repeat one traversal; a LanguageSyntax-driven emitter removes ~180 lines

`opennn/neural_network/model_expression.cpp:589-2187` · medium · design · lines -120 · effort L · risk medium · partial

C (589-701, 113 lines), PHP (1674-1794, 121), JavaScript (1796-2028, 233) and Python (2030-2187, 158) all run: build_expression -> map output names -> split/rename lines -> has_softmax -> header with an `i) name` listing (four loops: 612, 1729, 1833, 2056) -> activation bodies (four loops) -> `name = inputs[i]` unpack (649, 1753, 1990, 2107) -> emit_body_lines -> fix_output_names + output collection (658-669, 1711, 2010-2016, 2126-2139) -> softmax (C/JS share emit_softmax_block; PHP 1764-1775 and Python 2141-2148 hand-roll it) -> driver. Roughly 330 of the 625 lines are this shared traversal; the rest (C main 25, PHP GET/response 45, JS HTML 120, Python class/batch/main 60) is genuinely…

**Fix:** After nn-expression-10/11/12 land: extend LanguageSyntax with header_item, activation body member, unpack_format, output_collect/return format, math-keyword prefix, statement terminator and a softmax variant, and write one `emit_model_function(ostringstream&, const ExportNames&, const vector<string>& lines, const LanguageSyntax&)` (~90 lines). Each get_expression_* then becomes prelude + emit_model_function + driver. Keep the generated Python/PHP softmax textually equivalent or accept the loop…

*Verifier:* The shared traversal is real: all four paths do build_expression -> name mapping -> lines -> has_softmax -> header listing (612, 1729, 1833, 2056) -> activations -> unpack (649, 1753, 1990, 2107) -> emit_body_lines -> fix_output_names -> softmax -> driver. Overstated on two points: (1) the Python softmax is a materially different list form that model_expression_test.cpp:368-371 asserts…

#### xcut-build-tests-13 — Numerical Hessian helper returns a zero matrix; its only consumer asserts nothing; input-deltas helper has no callers

`tests/numerical_derivatives.cpp:155-240` · medium · dead code · lines -95 · effort S · risk low · confirmed

`calculate_numerical_hessian` computes the numerical gradient and then returns `MatrixR::Zero(n, n)`; `calculate_inverse_hessian` detects the singular zero matrix and returns the inverse of `1e-4 * I`, i.e. `1e4 * I`. The single user, QuasiNewtonMethodTest.BFGS_Update (tests/training_strategy/quasi_newton_method_test.cpp:82-89), only checks `rows == cols` and `!isnan`, so the test is green by construction and verifies nothing about BFGS. `calculate_numerical_input_deltas` (155-201) is called from no test at all (repo-wide grep), and would write through a device pointer on GPU configs since it never uploads the batch. About 90 lines of misleading helper code.

**Fix:** Delete `calculate_numerical_input_deltas`, `calculate_numerical_hessian` and `calculate_inverse_hessian` (declarations in numerical_derivatives.h:23-25) and the two lines using them in BFGS_Update. If a Hessian reference is wanted, implement a real central-difference Hessian from the gradient helper in a follow-up and assert the BFGS update against it.

*Verifier:* tests/numerical_derivatives.cpp:203-208 calculate_numerical_hessian returns MatrixR::Zero after computing the gradient; 210-240 calculate_inverse_hessian damps the singular zero matrix with 1e-4*I and inverts it. Only caller: quasi_newton_method_test.cpp:82 (grep over tests/examples/docs), whose assertions at 84-87 are rows==cols and !isnan only. calculate_numerical_input_deltas (155-201) has no…

#### nn-builders-chat-6 — YoloNetwork ctor (731 lines) duplicates SPPF, FPNv8 prior-bias, v8 det head, class-activation, PAN/neck blocks

`opennn/neural_network/standard_networks.cpp:460-1190` · medium · duplication · lines -90 · effort L · risk medium · confirmed

Concrete twins inside the constructor: (a) SPPF block written twice — 795-808 (c8_) and 974-996 (sppf_), 14 + 22 lines, identical structure; (b) the FPNv8 `_cls_out` prior-bias block appears verbatim twice (870-881 and 1082-1093) and is a third variant of apply_yolo_prior_bias (519-534) differing only in the label predicate and fill range; (c) add_det_v8 (850-862) and add_det_head_v8 (1053-1069) differ only in add_cba vs add_conv(act, bn=true); (d) add_pan_block (1016-1021) is add_yolo_neck (574-582) with 3 convs instead of 5; (e) the Detection class-activation mapping is repeated at 556-559 and 1178-1181; (f) `compile(); set_parameters_random(); prior-bias` is hand-expanded twice (868-881,…

**Fix:** Introduce a file-local `struct YoloBuilder { NeuralNetwork& net; const char* act; Index classes; ... Index conv(...); Index cba(...); Index pool(...); Index upsample(...); Index concat(...); Index sppf(Index in, const string& prefix); Index c2f(...); Index neck(Index in, Index in_ch, Index small, Index large, Index convs, const string& pfx); Index det_head(...); void det_head_v8(Index feat, const string& name, bool separate_act); void prior_bias(string_view label_predicate, Index…

*Verifier:* Read 460-1190. SPPF twins at 791-806 (c8_) and 974-996 (sppf_); the _cls_out prior-bias block at 868-881 and 1080-1093 is byte-identical; add_det_v8 (850-862) vs add_det_head_v8 (1053-1069) differ only in add_cba vs add_conv(act,true); add_pan_block (1016-1021) is add_yolo_neck (574-582) truncated to 3 convs; Detection class-activation mapping at 552-555 and 1178-1181;…

#### nn-expression-8 — 124-line embedded scaler switch re-derives slope/offset that core/scaling.h scaling_affine() provides

`opennn/neural_network/model_expression.cpp:896-1019` · medium · duplication · lines -85 · effort M · risk low · confirmed

core/scaling.h exposes scaling_affine(scaler, descriptives, min_range, max_range) -> {scale, offset} with the same EPSILON guards, and scaling_layer.cpp:456 states that exporters must fold through it 'so the exported model cannot drift away from what the layer computes'. The embedded exporter instead hand-writes the forward formulas a third time (its `d.maximum - d.minimum < EPSILON` vs `standard_deviation > EPSILON` guards already differ in form from scaling_affine's `<= EPSILON`). Only the unscaling direction lacks a helper, and Unscaling::write_expression (unscaling_layer.cpp:127-143) hand-writes those formulas too, so an unscaling_affine sibling would serve two callers.

**Fix:** Add `unscaling_affine(...)` (~20 lines) beside scaling_affine in core/scaling.h mirroring unscale_column_cpu's degenerate rules (constant -> minimum / mean). In the embedded loop: `const auto [slope, offset] = is_unscaling ? unscaling_affine(...) : scaling_affine(...);` plus the two-line Logarithm flag. Use unscaling_affine in Unscaling::write_expression as well. The degenerate-scaling execution tests guard the behaviour.

*Verifier:* Lines 896-1019 hand-write the six scaler cases for both directions. For the scaling direction the results match core/scaling.h scaling_affine (lines 94-130) case by case: MinMax `range < EPSILON` identical (908 vs 108); MeanSD `> EPSILON else 0` is the complement of `<= EPSILON` (949 vs 115); SD same; ImageMinMax same; Logarithm/None {1,0} + flag. scaling_layer.cpp:456-458 documents that…

#### nn-core-9 — About 108 lines of one-expression wrappers defined out-of-line in neural_network.cpp

`opennn/neural_network/neural_network.cpp:513-2349` · medium · boilerplate · lines -85 · effort S · risk low · confirmed

The following definitions are single expressions or two-line delegations and belong inline in the header next to their siblings that already are (get_layers_number(), is_gpu(), get_parameter_specs()...): the two delegating constructors (513-527), has(string)/has(LayerType)/has_recurrent_layers (629-646), get_input_feature_names/get_output_feature_names (648-656), get_first(string) x2 and the non-const get_first(LayerType) (686-689, 699-707), get_inputs_number/get_outputs_number/get_input_shape/get_output_shape (855-879), set_parameters_random/glorot/pytorch (1046-1059), save_states_binary (2101-2105), get_layer_labels (2343-2349). They add ~108 lines and 22 cross-file lookups for readers…

**Fix:** Replace each declaration in neural_network.h with the inline one-liner (respecting the header's public/protected/private ordering) and delete the .cpp bodies. Leave the calculate_outputs(MatrixR/Tensor3/Tensor4) wrappers and copy_states_* where they are (they need the file-local single_input_view helpers and the CUDA #ifdef split).

*Verifier:* Read neural_network.cpp 513-527, 629-656, 686-707, 855-879, 1046-1059, 2101-2105, 2343-2349 and header declarations at 119-121, 146, 149, 159-162, 184-188, 196-198, 256, 299 (all out-of-line) next to inline siblings at 91-94, 139, 175. Header already includes layers/layer.h so Layer/Operator are complete. Counted 106 .cpp lines removed; the header grows by ~20 one-line bodies, net about -85,…

#### response-opt-2 — Forecasting path (fixed_history, time_roles, combine_input) is unreachable: no setter exists

`opennn/response_optimization/response_optimization.h:240-242` · medium · dead code · lines -75 · effort S · risk medium · confirmed

`time_roles` and `fixed_history` are private members with no setter, no JSON loader and no friend; grep over opennn/, tests/, examples/, docs/benchmarks finds no writer. Hence is_forecasting() is always false, is_history() always false, and the entire forecasting machinery is dead: combine_input (response_optimization.cpp:1070-1104, 35 lines incl. an Eigen device broadcast), the forecasting branch in calculate_outputs (1108-1116, whose error message tells the user to call a set_fixed_history() that does not exist), is_past/is_history/TimeType, and the four `!is_history(...)` filters in build_input_columns, get_variables_and_descriptives, combine_input and filter_feasible_points. Neural…

**Fix:** Either add the missing set_fixed_history(const Tensor3&, map<string,TimeType>) and a test, or (recommended) delete combine_input, the forecasting branch of calculate_outputs, is_forecasting/is_history/is_past, TimeType, time_roles, fixed_history and the is_history filters. Verify against Neural Designer that is_forecasting/is_history/is_past are not called before deleting the public predicates.

*Verifier:* time_roles and fixed_history are private (response_optimization.h:240-242) with no setter, no friend and no JSON path; grep over opennn/, tests/, examples/, docs/benchmarks for time_roles|fixed_history|set_fixed_history|is_forecasting|is_history|is_past|combine_input|TimeType finds nothing outside response_optimization.{h,cpp}. Inside the file: is_past/is_history (368-377), combine_input…

#### xcut-build-tests-6 — Qwen3/INT8 test helpers are still duplicated verbatim between two files

`tests/neural_network/int8_inference_test.cpp:22-110` · medium · duplication · lines -75 · effort S · risk low · confirmed

The prior audit listed the shared `tests/llm_test_helpers.{h,cpp}` as pending; it is still not done. `struct Dims`, `constexpr Dims TINY`, `fill_parameters`, `run`, `logits_row` (33 lines with the CUDA copy-back and bf16 unpack), `max_difference` and `round_parameters_to_bf16` appear identically in qwen3_network_test.cpp (19-81, 129-137, 238-246) and int8_inference_test.cpp (22-29, 38-110). Any fix to the bf16 unpack or the device copy must be made twice.

**Fix:** Create `tests/neural_network/llm_test_helpers.{h,cpp}` exporting Dims/TINY/WIDE, `make_qwen`, `fill_parameters`, `run`, `logits_row`, `max_difference`, `round_parameters_to_bf16`; include it from both tests and delete the local copies.

*Verifier:* int8_inference_test.cpp: Dims 22, TINY/WIDE 28-29, make_qwen 31, fill_parameters 38, run 48, logits_row 59, max_difference 94, round_parameters_to_bf16 104. qwen3_network_test.cpp: Dims 19, TINY 25, fill_parameters 27, run 37, logits_row 48, max_difference 129, round_parameters_to_bf16 238. diff of the two logits_row bodies: identical. ENGINEERING_AUDIT.md:162-164 marks the shared…

#### nn-builders-chat-5 — add_layer returns void: 71 'x = get_layers_number() - 1' lines in standard_networks.cpp (134 repo-wide)

`opennn/neural_network/neural_network.h:70-70` · medium · boilerplate · lines -71 · effort M · risk low · confirmed

Every graph-building site needs the index of the layer it just added, so the file repeats `add_layer(...); idx = get_layers_number() - 1;` 71 times (40 inside the YoloNetwork constructor, plus the ~8 lambdas add_conv/add_cba/add_residual_block/add_top_down/add_residual_and_norm/add_feed_forward/add_norm/add_linear whose bodies are mostly this idiom). NeuralNetwork::add_layer already computes `old_layers_number = get_layers_number() - 1` internally (neural_network.cpp:539) and returns nothing. Tests repeat the idiom 63 more times (yolo_overfit_test 22, yolo_loss_test 11, ...).

**Fix:** Make `Index NeuralNetwork::add_layer(unique_ptr<Layer>, const vector<Index>& = {})` return the index of the layer it appended (source-compatible: existing callers that ignore the result keep compiling, Neural Designer included). Then collapse each `add_layer(...); idx = get_layers_number() - 1;` pair to `const Index idx = add_layer(...);` and turn the builder lambdas into one-liners.

*Verifier:* neural_network.h:70 'void add_layer(unique_ptr<Layer>, const vector<Index>& = {})'; neural_network.cpp:535-560 computes old_layers_number and returns nothing. grep -c 'get_layers_number() - 1' gives 71 in standard_networks.cpp and 134 repo-wide (2+71+3+2+1+2+3+2+1+7+2+1+1+3+11+22). Returning Index is source-compatible for void-ignoring callers. -71 LOC within range (one line per site, lambdas a…

#### xcut-build-tests-23 — yolo example duplicates its FPN-head collection/decode block, GtBox, and box-to-pixel math inside one 1818-line main

`examples/yolo/main.cpp:1141-1240` · medium · duplication · lines -70 · effort M · risk low · confirmed

The block that walks the network layers, finds Detection/DetectionV8 layers, copies their slot views back from the GPU, builds `vector<YoloFpnHead>` and calls `decode_yolo_v8_fpn_detections`/`decode_yolo_fpn_detections` is written twice (visualisation loop 1141-1240 and mAP loop 1631-1670, ~45 lines each, differing only in variable names). `struct GtBox { int cls; float cx, cy, w, h; }` is declared twice (1262 and 1532). The detection-to-pixel-box conversion (`int(round(d.center_x - d.width*0.5f))` ... `-1`) appears four times (1386-1395, 1409-1414, 1467-1472). Together with a hand-rolled BMP reader/writer (47-140, a third copy of the tests' write_bmp_24) and a 270-line inline VOC mAP@0.5…

**Fix:** Extract file-local `vector<YoloDetection> run_fpn_detection(YoloNetwork&, const Tensor4& input, bool is_v8, Index reg_max, const Shape& input_shape, vector<vector<float>>& scratch)` used by both loops, one `GtBox`, and a `PixelBox to_pixel_box(const YoloDetection&)` helper; move the mAP computation into `float compute_voc_map(...)` so main reads top-to-bottom. Longer term the mAP belongs in TestingAnalysis (the library has no mean-average-precision routine today).

*Verifier:* examples/yolo/main.cpp (1818 lines): vector<YoloFpnHead> blocks at 1141 and 1631 ending in decode_yolo_v8_fpn_detections/decode_yolo_fpn_detections at 1223/1232 and 1660/1664; `struct GtBox { int cls; float cx, cy, w, h; }` at 1262 and 1532; `int(round(d.center_x - d.width * 0.5f))` at 1403, 1418, 1467 (three, not four, but the pattern holds); BMP writer/reader at 48/92; mAP block 1529-1795.…

#### nn-builders-chat-8 — Two parallel sampling implementations (sample_token_with_workspace vs DecoderSampler::sample_host) with diverging semantics

`opennn/neural_network/chat.cpp:145-231` · medium · duplication · lines -70 · effort M · risk medium · partial

sample_token_with_workspace (145-231, probability space, used by the classic sessions) and DecoderSampler::sample_host (613-688, logit space, used by the generic session) both implement temperature / top-k (nth_element) / top-p / repetition-penalty / final draw, ~85 and ~75 lines. They already disagree: penalty is `p /= penalty` vs sign-aware `logit*penalty / logit/penalty`; greedy ordering differs (finding -4); sample_host masks the padding token 0 (`adjusted[0] = NEG_INFINITY`, and the GPU kernel skips i == 0) while sample_token does not; the classic path draws from the global mutex-locked RNG (random_uniform) so the ChatSession seed is meaningless there, while DecoderSampler owns an…

**Fix:** Make the classic sessions use DecoderSampler too: read_classic_distribution already lands the row in host floats, so feed `log(p)` (with -inf for p <= 0 and a guard for an all-(-inf) row) into sample_host, and give ClassicGenerationState a DecoderSampler seeded from a ChatSession seed. Then delete SamplingWorkspace/sample_token_with_workspace and keep the public `sample_token` as a thin wrapper over the same routine (its 8 tests in tokenizer_layer_test.cpp are all satisfied in log space except…

*Verifier:* Duplication confirmed: sample_token_with_workspace 145-231 (probability space, global mutex-locked random_uniform, random_utilities.cpp:32-34) vs DecoderSampler::sample_host 613-688 (logit space, adjusted[0] = NEG_INFINITY, sign-aware penalty, per-session mt19937_64). ClassicGenerationState (753-778) holds only a SamplingWorkspace, no generator, so the seed is indeed unused on the classic path.…

#### xcut-boilerplate-3 — 73 two-line `if (has) read_json` guards exist only because read_json_* has no fallback parameter

`opennn/core/json.cpp:608-634` · medium · boilerplate · lines -65 · effort S · risk low · confirmed

Pattern (c). The write side is already table-driven (write_json(printer, {...}) in every layer, optimizer and selection class), so a per-class field descriptor would not save lines there. The read side, however, repeats one micro-pattern 73 times in 20 files: `if (root->has("X")) set_x(read_json_T(root, "X"));` (yolo_dataset.cpp 9, neural_network.cpp 5, multihead_attention_layer.cpp 5, language_dataset.cpp 5, stochastic_gradient_descent.cpp 4, dense_layer.cpp 4, optimizer.cpp 3, adaptive_moment_estimation.cpp 3, tokenizer_operator.cpp 3, batch_norm_operator.cpp 3, scaling_layer.cpp 3, normalization_layer_3d.cpp 3, embedding_layer.cpp 3, tabular_dataset.cpp 3, image_dataset.cpp 3,…

**Fix:** Add fallback overloads in json.h/json.cpp: `read_json_float(const Json*, string_view, float fallback)`, `read_json_index(..., long long fallback)`, `read_json_bool(..., bool fallback)`, `read_json_string(..., string_view fallback)` (about 12 lines), move read_json_float_alias/read_json_index_alias from selection_utilities to json.h, then collapse each guard to `set_x(read_json_bool(el, "X", use_bias));` and the yolo ternaries to one call each. In the same PR make ImageDataset::from_JSON read…

*Verifier:* json.cpp:608-634 read_json_float/index/bool/string return 0/0/false/"" for a missing field and json.h:144-150 offers no value-fallback overload (read_json_string_fallback takes alternative keys, not a default value). grep of `has("...")` immediately followed by a read_json call counts 68 guards across opennn/ (finding says 73; within tolerance). yolo_dataset.cpp:2234-2240 has the seven ternaries…

#### training-loss-6 — CE3d GPU forward still uses the serial one-thread-per-token argmax kernel the metrics path already replaced

`opennn/training_strategy/error_functions.cpp:383-421` · medium · overhead · lines -60 · effort M · risk medium · confirmed

cross_entropy_3d's GPU branch launches cross_entropy_3d_multiple_forward_kernel (kernel_losses.cu:159-209): one thread per token scans the whole vocabulary serially (uncoalesced across threads, ~150k iterations per thread for Qwen-size vocabularies), writes three tokens-sized arrays, then three cublasSasum passes reduce them. kernel_losses.cu:218-299 already provides cross_entropy_3d_metrics_cuda: warp-per-token coalesced argmax with block reduction directly into three floats, and its own comment says it 'replaces the per-token error/mask arrays and their three cublasSasum passes'. The old kernel remains the path for every validation epoch on GPU (optimizer.cpp:2156/2200/2258 call…

**Fix:** In cross_entropy_3d's GPU branch: device::set_zero_async(reduction_device, 3 floats), cross_entropy_3d_metrics_cuda<T>(...), copy 3 floats D2H, synchronize. Delete cross_entropy_3d_multiple_forward_kernel/_cuda (kernel_losses.cu:159-209), its .cuh declaration, its INSTANTIATE line and the now-dead stub at error_functions.cpp:119-121; make error_workspace_floats return 0 for CrossEntropy3d so the workspace is the 3-float reduction only (calculate_error_device_metrics and…

*Verifier:* error_functions.cpp:383-411: GPU branch launches cross_entropy_3d_multiple_forward_cuda (kernel_losses.cu:159-209, serial per-thread vocab scan into three token-sized arrays) then three cublasSasum. kernel_losses.cu:213-299 cross_entropy_3d_metrics_cuda (warp-per-token, block reduce into sums[0..2]) already exists and is used by calculate_error_device_metrics (loss.cpp:1627-1641). calculate_error…

#### nn-expression-7 — Embedded Recurrent/LSTM packing loops are element-wise identity copies of row-major views

`opennn/neural_network/model_expression.cpp:1234-1278` · medium · duplication · lines -55 · effort S · risk low · confirmed

RecurrentOperator::parameter_specs declares the views as {output_features}, {input_features, output_features}, {output_features, output_features}; MatrixR is Eigen::RowMajor (opennn_types.h:292) and TensorView::as_matrix maps rows=shape[0], so input_w_map(f, j) already lives at data[f*hidden + j] — exactly the index the loops write to. The Recurrent branch (1243-1278) copies three tables one element at a time into vectors laid out identically, and the LSTM branch (1346-1402) does the same for 12 views where the only real work is concatenating the 4 gates. The Dense branch two screens up already does the right thing with span<const float>(as<float>(), n).

**Fix:** Recurrent: emit_float_array(prefix + "_input_weights", span<const float>(parameter_views[1].as<float>(), size_t(features*hidden))) and likewise for biases/recurrent weights (keep the rank/host-fp32 check by calling as_matrix() once or require_host_fp32_data). LSTM: replace each gate's three nested loops with ranges::copy_n(parameter_views[g].as<float>(), n, dest.begin() + g*n). The ExpressionExecution and SaveCEmbeddedLstm/Recurrent tests cover the output.

*Verifier:* recurrent_layer.cpp:36-43 specs: {hidden}, {features, hidden}, {hidden, hidden}; LSTM specs (long_short_term_memory_layer.cpp:60-83) are 4x bias {hidden}, 4x {features, hidden}, 4x {hidden, hidden}, all FP32. TensorView::as_matrix (tensor_types.h:630-638) maps rows=shape[0], cols=size/rows onto MatrixMap = Map<MatrixR> with MatrixR RowMajor (opennn_types.h:292-305), so input_w_map(f,j) ==…

#### xcut-boilerplate-2 — set_default() bodies duplicate or contradict header default member initializers in 7 classes

`opennn/training_strategy/levenberg_marquardt_algorithm.cpp:31-49` · medium · design · lines -55 · effort S · risk medium · confirmed

Pattern (g). Every optimizer and selection class carries two sources of truth for its defaults. levenberg_marquardt_algorithm.h:80-88 initialises initial/minimum/maximum_damping_parameter and damping_parameter_factor to 0.0f, but the values actually used (1e-3, 1e-6, 1e6, 10) exist only in set_default() (lm.cpp:43-48) - the header values are dead and misleading (a factor of 0 would make the damping update a no-op). stochastic_gradient_descent.h:51-53 declares `float initial_learning_rate;` and `float initial_decay;` with no initializer at all; they are only set in set_default() (sgd.cpp:71-72). quasi_newton_method.h:57 `minimum_loss_decrease = 1.0e-6f` is repeated verbatim at qn.cpp:29.…

**Fix:** Move the real values into the header default member initializers (LM damping params, SGD learning-rate/decay, QN, Adam, GrowingNeurons, GrowingInputs, InputsSelection fields) and shrink each set_default() to the genuinely runtime-dependent part (GrowingInputs::maximum_inputs_number from the dataset, GrowingNeurons::maximum_neurons from the network, GeneticAlgorithm population sizing, the `name` string). Decide explicitly whether Adam should share the 1000/3600 epoch/time defaults of its…

*Verifier:* levenberg_marquardt_algorithm.h:80-88 initialises the four damping parameters and minimum_loss_decrease to 0.0f; levenberg_marquardt_algorithm.cpp:31-49 set_default() overwrites with 1e-3/10/1e-6/1e6 and also maximum_epochs=1000, maximum_time=3600. stochastic_gradient_descent.h:51-53 declares initial_learning_rate/initial_decay uninitialised; sgd.cpp:67-72 sets 0.001f. quasi_newton_method.h:57…

#### dataset-a-11 — infer_column_types duplicates infer_dataset_date_format and then calls it anyway

`opennn/dataset/tabular_dataset.cpp:2698-2757` · medium · duplication · lines -52 · effort S · risk low · confirmed

Lines 2698-2757 scan the 100 sampled rows for the first DateTime token with a detectable format and then, unless the hit was on row 0 or the file has <=100 rows, discard the result and call infer_dataset_date_format (1542-1583), which performs the identical scan over all rows and returns on the first hit. Case analysis: rows <= 100 -> the sample is the full file and both scans are the same; rows > 100 and hit on row 0 -> infer_dataset_date_format returns on row 0 with the same value (same line-then-column order); otherwise the full scan is performed anyway. So the sampled block is behaviour-identical to a single call and costs an extra 100-row pass plus 60 lines.

**Fix:** Replace lines 2698-2757 with `const DateFormat date_format = infer_dataset_date_format(variables, sample_lines, file_separator, has_sample_ids, missing_values_label, has_quotes);` (it already short-circuits when no variable is DateTime). The sampled_tokens vector is then only used by the type loop.

*Verifier:* infer_column_types (2598-2757): sampled rows are strided (row = i*total_rows/rows_to_check, 2617), so sampled row 0 is file row 0; block 2698-2757 scans the sample and then calls infer_dataset_date_format (1542-1583, same line-then-column order, returns on first hit) unless rows_to_check == total_rows (sample is the whole file, same result) or hit_row == 0 (full scan would also stop on row 0 with…

#### layers-b-4 — GQA project→qk_norm→rope→attend→o_proj pipeline is written four times (CPU/GPU × batch==1/batch>1)

`opennn/neural_network/layers/grouped_query_attention_layer.cpp:898-1250` · medium · duplication · lines -50 · effort M · risk medium · partial

The same nine-call sequence appears in forward_propagate batch==1 (lines 898-940), forward_propagate batch loop (952-975), forward_gpu non-fused batch==1 (1105-1110 + 1113-1121 + 1200-1210) and forward_gpu batch loop (1228-1250). The CPU copies differ from the GPU copies only in how the TensorViews are built (float* vs void*+Type+Device) and in not passing the optional scale views, which linear_forward_transposed defaults to {} anyway. The kv-cache handling (prepare_kv_cache, past offset, k_slot/v_slot views) is also duplicated between the two batch==1 blocks. Any fix to the pipeline (e.g. qk_norm in-place aliasing, position handling) must be applied in four places.

**Fix:** Add one private member `void attend_sequence(const TensorView& x_b, TensorView& q_v, TensorView& k_v, TensorView& v_target, TensorView& qr_v, TensorView& k_target, TensorView& attn_v, TensorView& o_b, const TensorView& cos_v, const TensorView& sin_v, Index position_offset, float* decode_partials = nullptr, const int* position_device = nullptr)` that performs the three projections (always passing q_scale/k_scale/v_scale/o_scale; they are empty on CPU), optional qk_norm, rope, apply_attention and…

*Verifier:* Four copies confirmed: CPU batch==1 (898-941: q/k/v projections, qk_norm, rope into qr_v/k_slot, apply_attention, o_proj), CPU batch loop (952-975), GPU batch==1 non-fused (1131-1146 projections+norm+rope, then 1207-1216 attention+o_proj) and GPU batch loop (1228-1250); the CPU copies omit the scale views that the GPU copies pass. However the GPU batch==1 block is not a straight-line copy:…

#### core-kernels-3 — Activation fwd/bwd carry three hand-written kernels each; one VecIO<T,vec16<T>> kernel covers both dtypes

`opennn/core/cuda/kernel_activation.cu:82-200` · medium · duplication · lines -45 · effort M · risk medium · confirmed

activation_forward has a generic strided kernel, a __nv_bfloat162 twin and a float4 twin plus a two-branch dispatcher; activation_backward repeats the same trio (six kernels, ~90 lines). kernel_cast.cu already shows the sanctioned shape: one cast_kernel over VecIO<Src,4> launched through launch_vec_on. The BF16 twin is also the weaker one: it moves 4 bytes per thread (bfloat162) where VecIO<bf16,8> moves 16, and it reinterpret_casts to __nv_bfloat162 without any alignment check while the FP32 path checks are_aligned<16>. Collapsing to activation_forward_kernel<T, VEC>/activation_backward_kernel<T, VEC> using VecIO<T, vec16<T>>::load_float/store_float (scalar tail inside, as cast_kernel…

**Fix:** Replace the six kernels with two templates `activation_forward_kernel<T, VEC>(n_vec, n, data, function)` and `activation_backward_kernel<T, VEC>(n_vec, n, outputs, delta, function)` built on VecIO<T, VEC> (VEC = vec16<T>), launched via launch_vec_on<vec16<T>>(stream, n, are_aligned<16>(...), ...); keep the grid-strided scalar tail inside the kernel exactly as cast_kernel does. Verify with the ResNet/transformer throughput benchmarks (bf16 activations should get faster, fp32 unchanged).

*Verifier:* kernel_activation.cu:82-200 read: activation_forward has activation_forward_kernel<T> (83), _bf162 (89), _f4 (99) and the two-branch dispatcher (121-132); backward repeats the trio (135-200). The bf162 path reinterpret_casts at 125 with only an (n & 1) check, while the f4 path uses are_aligned<16> at 128/194. kernel_cast.cu:17-38 is the existing single-kernel VecIO + launch_vec_on shape, and…

#### training-optimizers-5 — train_epoch and evaluate_epoch are the same epoch skeleton written twice (~80 lines removable with one driver)

`opennn/training_strategy/optimizer.cpp:1780-2277` · medium · duplication · lines -45 · effort M · risk medium · partial

Measured duplication between train_epoch (1780-2097) and evaluate_epoch (2099-2277): (a) the whole/tail split, lines 1796-1806 == 2111-2121 (11 lines, identical); (b) merge_tail, 1898-1917 == 2164-2179 (16 identical lines; train adds 3 metric write-backs); (c) the CPU inline loop, 1919-1977 vs 2181-2215, whose body is `fill; <exactly what context.step does>; accumulate` - i.e. the step lambda each function already defines for the GPU path is re-expanded by hand for CPU; (d) the GPU prologue/epilogue 1996-2016 + 2071-2084 vs 2217-2237 + 2266-2274 (use_device_metrics, DeviceEpochMetricSums reset, EpochLoopContext aggregate init with 12 positional fields, run_epoch_loop, device_metrics.read,…

**Fix:** One driver `Loss::EvaluationResult run_epoch(EpochLoopContext&)` that owns: the whole/tail split, `if (!on_gpu) { pop pooled batch; for each: fill(fill_mode); step(batch, result); push }` else `run_epoch_loop`, the device-metrics reset/read, average_epoch_metrics, and the weighted tail merge via a `function<Loss::EvaluationResult()> tail` member of EpochLoopContext. train_epoch then sets step/tail/finalize and keeps the graph dispatch; evaluate_epoch sets step/tail only. Introduce a small…

*Verifier:* Duplication is real: split 1798-1806 == 2113-2121 verbatim; merge_tail 1898-1917 vs 2164-2179 identical except the three metric write-backs; CPU inline loops 1919-1977 vs 2181-2215 re-expand what context.step does; GPU prologue/epilogue 1996-2016/2071-2084 vs 2217-2237/2266-2274 the same shape with 12-positional EpochLoopContext init. Five index vectors threaded positionally through warmup…

#### dataset-b-6 — TextGenerationDataset re-implements the tokenizer's vocabulary counting and rebuilds its lookup map

`opennn/dataset/text_generation_dataset.cpp:43-295` · medium · duplication · lines -35 · effort M · risk low · confirmed

create_vocabulary (43-72) is a line-for-line copy of the OpenMP token-count block in TokenizerOperator::build_vocabulary (tokenizer_operator.cpp:96-139) differing only in iterating a flat vector<string_view> instead of vector<vector<string>>; both end in make_vocabulary. encode_corpus (270-295) builds a fresh unordered_map<string_view, Index> from the vocabulary although the tokenizer already owns VocabularyMap (StringMap<Index>, heterogeneous find) exposed via token_to_id(string_view) and get_vocabulary_map(). ~55 lines of dataset code duplicating operator code, plus a second hash map of the whole vocabulary built per read_txt.

**Fix:** Add a `build_vocabulary(span<const string_view> tokens, Index, Index)` overload to TokenizerOperator (move the counting block there; let the documents overload flatten or share a count_tokens helper), delete create_vocabulary, and reduce encode_corpus to a parallel loop over `tokenizer->token_to_id(view)` (or inline it in read_txt). Keep the OpenMP loop so the parallel encode is preserved.

*Verifier:* text_generation_dataset.cpp:43-72 is the same OpenMP per-thread unordered_map<string_view,size_t> counting block as TokenizerOperator::build_vocabulary (tokenizer_operator.cpp:95-139), differing only in the flat vector<string_view> input, both ending in make_vocabulary; set_vocabulary (57-61) already calls rebuild_map, so the tokenizer owns the lookup map and exposes token_to_id(string_view)…

#### dataset-b-7 — Two uint8 bilinear resizers (bilinear_resize_uint8 vs blit_resized_into_canvas) with different sampling conventions

`opennn/dataset/yolo_dataset.cpp:558-601` · medium · duplication · lines -35 · effort S · risk medium · confirmed

bilinear_resize_uint8 (558-601) and blit_resized_into_canvas (1805-1838) both resample a uint8 HxWxC buffer bilinearly through bilinear_blend; the second only adds a destination offset. They use different pixel mappings, however: the first maps align-corners style ((src-1)/(dst-1)), the second half-pixel centres ((o+0.5)*src/dst-0.5), and resize_image (image_processing.cpp:640-685, used to build the cache) uses a third (x*scale, top-left aligned). A network resized at runtime (input_shape != cache_input_shape, the `resize_needed` path at 1914) therefore sees pixels sampled differently from the mosaic path and from the cached letterbox. No test exercises resize_needed (grep cache_input_shape…

**Fix:** Keep blit_resized_into_canvas (half-pixel centres, the convention the mosaic path already trains on) and implement bilinear_resize_uint8 as `blit_resized_into_canvas(src, src_h, src_w, dst, dst_w, 0, 0, dst_w, dst_h, channels)` after the memcpy fast path; add one test for the resize_needed path. This changes resampled pixel values on that path by at most sub-pixel shifts.

*Verifier:* bilinear_resize_uint8 (558-601) uses (src-1)/(dst-1) align-corners mapping; blit_resized_into_canvas (1805-1838) uses (o+0.5)*src/dst-0.5 half-pixel centres; resize_image (image_processing.cpp:637-685) uses x*scale top-left. Both uint8 resizers share bilinear_blend/round_to_byte and differ only in the destination offset. The resize_needed path (1914-1915, used at 1992) is reachable because…

#### layers-a-8 — Pooling and Convolutional mirror every geometry field of their operator

`opennn/neural_network/layers/pooling_layer.h:149-175` · medium · duplication · lines -35 · effort M · risk medium · confirmed

Pooling stores input_height/width/channels, pool_height/width, padding_height/width, row_stride/column_stride and pooling_method, then copies all nine into PoolOperator (which stores the identical nine, pooling_layer.h:41-52) in update_pool_operator; Pooling::get_output_height/width (pooling_layer.cpp:446-454) are byte-for-byte copies of PoolOperator::get_output_height/width (pooling_layer.cpp:221-229). Convolutional does the same with eight fields (convolutional_layer.h:110-120 vs convolution_operator.h:18-27) and a ten-argument convolution.set call (convolutional_layer.cpp:183-187). Two copies of the same state means two places that can disagree (apply_input_shape and read_JSON_body must…

**Fix:** Drop the layer-side copies and make the operator the single owner: Pooling's getters become { return pool.pool_height; } etc., get_output_height/width delegate to pool.get_output_height()/get_output_width(), set()/read_JSON_body assign into pool.* and call a one-line pool.refresh() (the cuDNN descriptor rebuild that PoolOperator::set does today). Same for Convolutional: getters read convolution.*, get_output_height uses convolution.kernel_height etc., update_convolution_operator only recomputes…

*Verifier:* PoolOperator (pooling_layer.h:37-52) and Pooling (pooling_layer.h:149-162) hold the same nine geometry fields; Pooling::update_pool_operator (pooling_layer.cpp:514-523) copies them across; Pooling::get_output_height/width (446-454) are textual duplicates of PoolOperator::get_output_height/width (221-229); Pooling::set (538-555) and read_JSON_body (600-608) each carry a parallel assignment block.…

#### nn-expression-10 — 44 call sites recompute the same input/output/fixed name vectors across the emitters

`opennn/neural_network/model_expression.cpp:589-2187` · medium · boilerplate · lines -32 · effort S · risk low · confirmed

get_flat_input_names() is called 14 times, get_output_feature_names() 12 times and fix_names() 18 times; each emit_* helper re-derives its own copies (emit_c_prelude, emit_c_calculate_outputs and emit_c_main each call get_flat_input_names during one C export; emit_js_inputs_html/emit_js_outputs_html/emit_js_runtime rebuild the same four vectors). The bool+vector parameter lists in model_expression.h lines 48-68 exist only to thread these through. A single struct computed once per export removes the declarations and the ambiguity of which variant (raw/fixed/flat) a helper is looking at.

**Fix:** Add a private `struct ExportNames { vector<string> inputs, outputs, fixed_inputs, fixed_outputs; Index inputs_number() const; ... }` and a `ExportNames collect_names() const` built once at the top of each get_expression_*; pass `const ExportNames&` to every emit_* (replacing the vector<string>& parameters) and delete the local recomputations.

*Verifier:* grep -c: get_flat_input_names() 16, get_output_feature_names() 13, fix_names( 18 occurrences in model_expression.cpp. emit_c_prelude (605), emit_c_calculate_outputs (637-640) and emit_c_main (679-680) each recompute during one C export; emit_js_inputs_html/emit_js_outputs_html/emit_js_runtime (1852-1853, 1899-1900, 1947-1950) likewise. An ExportNames struct computed once per get_expression_* is a…

#### dataset-a-12 — read_csv's post-parse type refinement re-implements infer_variable_types_from_data; both entry points are unused

`opennn/dataset/tabular_dataset.cpp:2287-2324` · medium · duplication · lines -30 · effort S · risk medium · confirmed

Lines 2287-2324 (Numeric -> Constant if constant, -> Binary if only 0/1; Binary/Categorical with one category -> Constant) are the same rules as infer_variable_types_from_data (441-470), which derives the facts from `data` via is_constant/is_binary while read_csv derives them from the streaming NumericColumnValues accumulator (1933-1939, 2011-2046) because binary storage has no `data`. infer_variable_types_from_data is only called by set_binary_variables (472), and neither set_binary_variables nor infer_variable_types_from_data has any caller in opennn/, tests/, examples/ or docs/benchmarks/ (grep verified).

**Fix:** Factor the rule into one file-local `refine_variable_type(Variable&, bool constant, bool zero_one)` used by read_csv; make infer_variable_types_from_data call it with is_constant/is_binary, or delete infer_variable_types_from_data and set_binary_variables outright after verifying against Neural Designer (they are not in the prior audit's ND alive list).

*Verifier:* read_csv 2287-2324 applies Numeric->Constant/Binary and single-category->Constant from NumericColumnValues (1933-1939, refine_numeric 2011-2046); infer_variable_types_from_data (441-470) applies the same rules from is_constant/is_binary on data. set_binary_variables (472) is its only caller and grep over opennn/, tests/, examples/, docs/benchmarks/ finds no caller of set_binary_variables or…

#### selection-testing-8 — GrowingNeurons re-declares InputsSelection's whole configuration block (7 knobs, 8 setters, save/load)

`opennn/model_selection/growing_neurons.h:46-92` · medium · duplication · lines -28 · effort M · risk medium · confirmed

growing_neurons.h:46-55 and :80-92 duplicate inputs_selection.h:46-55 and :76-88 line for line (trials_number, folds_number, display, validation_error_goal, maximum_epochs, maximum_validation_failures, maximum_time and their setters, including the identical `max<Index>(new_folds_number, Index(1))` clamp), plus get_training_strategy and the save/load pair whose bodies are identical (growing_neurons.cpp:76-84 vs inputs_selection.cpp:71-79). Any change to selection stopping configuration (e.g. the MaximumEpochs fix above, or adding a seed) has to be made twice, and the two already drifted: GrowingNeurons has set_training_strategy + set (one resets defaults, one does not) while InputsSelection…

**Fix:** Introduce `class SelectionAlgorithm` (selection_utilities.h or a new selection_algorithm.h, same folder) holding training_strategy, the seven knobs, their setters, get_training_strategy, display, and the save/load pair implemented against virtual to_JSON/from_JSON; derive InputsSelection and GrowingNeurons from it. Setter names and signatures stay identical so Neural Designer callers are unaffected. Keep StoppingCondition enums per class (they differ).

*Verifier:* growing_neurons.h:46-55 and :80-92 vs inputs_selection.h:46-55 and :76-88 read side by side: trials_number, folds_number, display, validation_error_goal, maximum_epochs, maximum_validation_failures, maximum_time, the same setters (incl. `max<Index>(new_folds_number, Index(1))`), get_training_strategy, and save/load bodies (growing_neurons.cpp:76-84 == inputs_selection.cpp:71-79). Drift confirmed:…

#### nn-expression-9 — get_expression_c_embedded: 960-line function with ~155 lines of constant C runtime text inlined

`opennn/neural_network/model_expression.cpp:713-1672` · medium · design · lines -25 · effort M · risk low · confirmed

The function is one block nested inside a vestigial `string result; { ... result = buffer.str(); } return result;` wrapper (718-720, 1669-1671), contains five per-layer branches (Scaling/Unscaling 210 lines, Dense 88, Clamping 45, Recurrent 108, LSTM 128), and inlines the generated C runtime (nn_activation_forward, nn_dense_forward, nn_affine_forward, nn_affine_flags_forward, nn_recurrent_forward, nn_lstm_forward, nn_softmax_inplace, nn_clamp_inplace; lines 1482-1637) as string literals whose only interpolated value is LEAKY_RELU_SLOPE. The static runtime text belongs next to c_header/php_subheader as file-scope constants, which also lets the flash/PROGMEM preamble be reviewed as C rather…

**Fix:** One PR, no behaviour change: (a) drop the wrapper block; (b) hoist the eight runtime snippets to `static constexpr const char* nn_dense_forward_source = R"C(...)C";` etc. and emit them via a small `{bool, const char*}` table loop (LEAKY_RELU_SLOPE via a single format); (c) introduce a file-local `EmbeddedState {ostringstream tables, body; bool uses_*; array<bool,2> buffer_used; Index max_width, max_hidden; string_view current; ...}` and move each layer branch into…

*Verifier:* get_expression_c_embedded spans 713-1672 (~960 lines); the `string result; { ... result = buffer.str(); } return result;` wrapper is at 717-720 and 1669-1671 and serves no purpose (HostParametersGuard is outside it). Runtime C text is inlined as string literals at 1482-1637 with LEAKY_RELU_SLOPE (1492) as the only interpolation. Hoisting to file-scope raw-string constants next to…

#### r2-duplicated-kernels-across-folders-1 — block_reduce_sum/_sum2 are twin bodies and kernel_losses/kernel_tensor/kernel_embedding re-roll them

`opennn/core/cuda/kernel_common.cuh:304-345` · medium · duplication · lines -20 · effort M · risk low · partial

kernel_common.cuh carries two 20-line block reductions that differ only in the number of accumulators (1 vs 2), and three kernels elsewhere write the same reduction by hand instead of calling them: kernel_losses.cu:263-285 folds three warp partials (loss/active/correct) through its own shared arrays and warp_reduce_sum, kernel_tensor.cu:118-120 re-writes warp_reduce_sum with __shfl_down_sync, and kernel_embedding.cu:92-93 re-writes it for int. A single template<int N> block_reduce_sum(float (&v)[N]) and a template<typename T> warp_reduce_sum(T) replace all five. Latent hazard worth fixing in the same change: both block reductions keep their `__shared__ float warp_a[32]` inside the inlined…

**Fix:** In kernel_common.cuh replace block_reduce_sum and block_reduce_sum2 with `template<int N> __device__ bool block_reduce_sum(float (&v)[N])` (one shared float[N][32], loop over N, trailing __syncthreads so repeated calls are safe) and keep the two old names as one-line wrappers if preferred; make warp_reduce_sum a template on T (float and int) and add a two-lane variant only if warp_reduce_sum2 users need it. Then: kernel_losses.cu:263-285 -> `float v[3] = {loss, active, correct}; if…

*Verifier:* Duplication confirmed: kernel_common.cuh:304-345 holds block_reduce_sum and block_reduce_sum2 as twin bodies with no trailing __syncthreads (shared warp_a[32] inside the inlined function, return right after warp 0 reads it), so the composability hazard is real though no caller hits it today (block_reduce_sum at kernel_attention.cu:416, block_reduce_sum2 at kernel_normalization.cu:44 and :978, one…

#### dataset-a-9 — fill_from_binary_cache issues one ReadFile/pread per row, serially, and duplicates read_int32_batch

`opennn/dataset/tabular_dataset.cpp:257-287` · medium · overhead · lines -20 · effort M · risk low · confirmed

Per batch, the binary-storage fill loops sample_indices and calls cache_reader.read_at once per row (io_utilities.cpp 291-314: one synchronous ReadFile with OVERLAPPED per call; pread on POSIX). A 1024-row batch is therefore 1024 syscalls on the training thread (~1-3 ms on Windows), plus a `vector<float> row_buffer` heap allocation per call in the non-contiguous case. core/io_utilities already has read_int32_batch (369-432): the same per-record read with identical index validation (compare 241-259 here with 380-395 there) but parallelised with OpenMP and with a per-thread buffer; it is specialised to int32 only because the language datasets needed it first.

**Fix:** Turn read_int32_batch into `template<typename Record> read_record_batch(...)` (float instantiation is a plain memcpy instead of the int->float cast) and call it from fill_from_binary_cache for the contiguous case, then apply the NaN replacement and cache transforms as now. Deletes the duplicated range validation here. Keep the row_buffer path for non-contiguous indices but hoist the buffer into a thread_local or the parallel region.

*Verifier:* fill_from_binary_cache (217-287) issues one cache_reader.read_at per sample serially (258-283) with a per-call row_buffer allocation (257); FileReader::read_at is one ReadFile+OVERLAPPED per call (io_utilities.cpp:291-314). read_int32_batch (io_utilities.cpp:369-432) performs the same per-record read parallelised with OpenMP and per-thread buffers, with equivalent range validation (380-395 vs…

#### xcut-boilerplate-5 — YoloNetwork constructor is 732 lines with a verbatim 13-line duplicate and PRIOR_BIAS defined three times

`opennn/neural_network/standard_networks.cpp:460-1192` · medium · design · lines -20 · effort M · risk medium · confirmed

Pattern (k), god-function not listed by the prior audit (which only named ForwardPropagation::set and run_graph_epoch). YoloNetwork::YoloNetwork runs from line 460 to 1192 as a single function whose top-level branches are `if (backbone == Backbone::Vgg)` (599), `else if (DarknetTinyV3)` (622), `else if (CSPDarknet53v11)` (696), `else if (Darknet53 || CSPDarknet53)` (887) and `else` (1097). The CSPDarknet53v11 and Darknet53 branches each end with an identical 13-line block that walks get_layers(), dynamic_casts to Convolutional, matches the `_cls_out` label and fills the bias with -4.5951f (868-881 and 1080-1093, character-for-character the same), and the same constant is declared a third…

**Fix:** One PR: (1) a file-static `apply_cls_prior_bias(NeuralNetwork&, string_view label_suffix)` with a single `constexpr float yolo_prior_bias = -4.5951f`, called from both branches and reused by the anchor-based lambda; (2) extract each backbone branch into a file-static builder (`build_vgg_yolo`, `build_tiny_v3`, `build_csp_v11`, `build_darknet53`, `build_default`) taking the constructor's parameters and the shared lambdas (add_conv/add_cba) as a small context struct, leaving the constructor as…

*Verifier:* Column-0 scan: YoloNetwork::YoloNetwork opens at standard_networks.cpp:460 and closes at 1190 (731 lines). Backbone branches at 599/622/696/887/1097 as stated. diff of lines 868-881 against 1080-1093 reports identical; PRIOR_BIAS = -4.5951f is declared at 520 (inside apply_yolo_prior_bias lambda), 871 and 1083. Fix and LOC are plausible.

#### layers-b-7 — LSTM fused path rebuilds Wcat/Ucat(/bcat) identically in forward and backward, per call

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:234-249` · medium · duplication · lines -18 · effort S · risk low · confirmed

apply (lines 234-253) and apply_delta (lines 615-624) both allocate MatrixR Wcat(F,4H) and Ucat(H,4H) and fill them from the same twelve parameter views with the same leftCols/middleCols/rightCols sequence (18 duplicated lines), and each call also allocates Zin / Dcat_all (BT×4H, e.g. 64*200*1024*4 = 52 MB for the memory-audit LSTM) plus gWcat/gUcat/Z_c/D_c. The concatenation runs twice per training step; the big temporaries are malloc'ed and page-faulted every step. The H>=96 path is benchmark-tuned (the comment above line 205), so the GEMM restructuring should be benchmark-gated, but the helper extraction is pure deduplication.

**Fix:** Add a file-local `MatrixR concat_gate_columns(const TensorView& f, const TensorView& i, const TensorView& g, const TensorView& o)` (and the VectorR twin for the biases) and call it from both paths; the eight MatrixMap locals in each function disappear with it. Follow-up, gated on lstm_fused_path_test DISABLED_BenchmarkBoundary: write the input GEMM directly into the four gate slots (four BT×H GEMMs) so Zin/Wcat are not needed at all, and keep Zin's recurrent update in place on the strided slice…

*Verifier:* apply (234-248) and apply_delta (615-624) build Wcat(F,4H) and Ucat(H,4H) with the identical leftCols/middleCols/rightCols sequence from the same eight MatrixMap locals (222-229 vs 592-600); apply additionally builds bcat (244-248). Both also allocate the BT x 4H temporaries (Zin at 252, Dcat_all at 629) per call. A file-local concat helper in the anonymous namespace at the top of the file…

#### selection-testing-9 — GrowingInputs and GeneticAlgorithm end with the same 15-line install-optimal-inputs tail

`opennn/model_selection/growing_inputs.cpp:287-305` · medium · duplication · lines -16 · effort S · risk low · confirmed

growing_inputs.cpp:287-305 and genetic_algorithm.cpp:570-588 both: set_variable_indices(optimal, targets); read get_features_number(Input); restore the single Time variable role; configure_neural_network_inputs; capture_input_scaling; apply_input_scaling; finalize_selected_model; print. The only differences are the order of capture_input_scaling vs configure_neural_network_inputs (GI captures after configuring, GA before) and GI's extra set_maximum_inputs_number call. Both also open identically (has_nan -> scrub_missing_values; optimizer set_display(false)).

**Fix:** Add a protected `InputsSelection::install_optimal_inputs(Dataset*, NeuralNetwork*, const vector<Index>& optimal_inputs, const vector<Index>& targets, const vector<Index>& time_indices, const VectorR& optimal_parameters, bool display)` next to configure_neural_network_inputs that performs the shared sequence (pick one capture/configure order after checking calculate_used_feature_scaling does not depend on the input shape), and call it from both perform_input_selection tails. Fold…

*Verifier:* growing_inputs.cpp:287-305 and genetic_algorithm.cpp:570-588 read: both do set_variable_indices(optimal, targets); get_features_number(Input); restore the single Time role; configure_neural_network_inputs; capture_input_scaling; apply_input_scaling; finalize_selected_model; print. Differences are exactly as stated (GI captures scaling after configure and calls set_maximum_inputs_number; GA…

#### operators-b-4 — C2PSA ships its own one-thread-per-row softmax fwd/bwd kernels; library cuDNN softmax and a shared softmax_backward would replace them and 3 hand loops

`opennn/neural_network/operators/kernel_c2psa.cu:28-110` · medium · duplication · lines -15 · effort M · risk medium · partial

c2psa_row_softmax_kernel and c2psa_softmax_bwd_kernel run one thread per attention row that serially loops over T (= h*w, e.g. 400) three times; adjacent threads touch addresses T elements apart, so the accesses are fully uncoalesced. The library already has softmax(TensorView&) (tensor_operations.cpp:1161, cudnnSoftmaxForward channel mode, which is exactly a row softmax on a (BT, T) view) and the attention operator calls cudnnSoftmaxBackward by hand (attention_operator.cpp:619-630). The CPU side of the same backward formula dY = y*(dY - rowwise<y,dY>) is hand-written three times: attention_operator.cpp:1117-1119 and 1145-1146, and c2psa_operator.cpp:377-381 (plus the CPU forward softmax…

**Fix:** Add `void softmax_backward(const TensorView& y, TensorView& dy)` to tensor_operations (CPU: the Eigen rowwise formula; CUDA: cudnnSoftmaxBackward, chunked like softmax_gpu). Use it in AttentionOperator (both CPU sites and the CUDA lambda) and in C2PSA (wrap the (B,T,T) slot / the d_A scratch region in a TensorView), call softmax() on the attention slot in C2PSA's forward, fold `scale` into the dQ/dK GEMM alphas, and delete c2psa_row_softmax_*/c2psa_softmax_bwd_* (kernels + launchers +…

*Verifier:* Verified: kernel_c2psa.cu:29-51 run one thread per row (uncoalesced, T loops x3) and the launchers at 98-108; forward calls c2psa_row_softmax_cuda on slot 4 (c2psa_operator.cpp:131) which is a real TensorView slot so softmax(TensorView&) (tensor_operations.cpp:1161, cudnnSoftmaxForward channel mode, chunked at 1595-1636) is a drop-in. CPU softmax-backward formula is indeed hand-written at…

#### response-opt-9 — Objectives encodes bool/Index/sense per objective as floats in three 2xN matrices plus three vectors

`opennn/response_optimization/response_optimization.h:84-103` · medium · design · lines -15 · effort M · risk low · confirmed

source_and_column(0,j) is a bool stored as 1.0/0.0, (1,j) an Index stored as float, utopian_and_sense(1,j) a sense stored as +/-1.0, scale_and_offset a pair; closeness_mask/closeness_target/closeness_scale are three more parallel containers. Every reader decodes by convention: `source_and_column(0, 0) > 0.5f ? optimal_set.first : optimal_set.second)(0, static_cast<Index>(source_and_column(1, 0)))` (1684-1686), `(sense > 0)` (1409), `static_cast<Index>(source_and_column(1, j))` (1369-1370), `closeness_mask[static_cast<size_t>(j)]` (1372, 1403). Column indices above 2^24 would silently round, and a future reader cannot tell which row means what without the constructor. This is exactly the…

**Fix:** `struct ObjectiveSpec { bool from_input; Index column; float utopian; float sense; float scale; float offset; bool closeness; float target; float closeness_scale; }; vector<ObjectiveSpec> specs;` with `Index size() const`. extract/normalize/update_utopian_from_points become loops over specs (normalize is a per-column scale/shift, no Eigen broadcast needed). Remove the decode casts at the six read sites.

*Verifier:* Objectives (h:84-103) stores bool/Index/sense as floats in 2xN matrices; constructor encodes at 595-627 (is_input ? 1:0, static_cast<float>(feature_pointer), +-1.0 sense). Decode sites verified: extract 1368-1370 (>0.5, static_cast<Index>), closeness_mask[size_t(j)] 1372 and 1403, sense>0 at 1409, single-objective 1684-1686, plus utopian_and_sense.row(1) at 1458 and .cols() at 2450 (two more read…

#### training-loss-2 — YOLO v8 GPU round-trip runs twice per batch: TAL assignment, D2H of targets and every head repeated

`opennn/training_strategy/loss.cpp:970-1099` · medium · overhead · lines -15 · effort M · risk medium · confirmed

Quantified as the prior audit asked. Loss::back_propagate (1353-1379) calls calculate_error -> calculate_yolo(nullptr) -> yolo_v8_error_multi and then calculate_output_deltas -> calculate_yolo(&bp) -> yolo_v8_gradient_multi. Both go through for_each_v8_head, so per training batch on GPU with H heads: 2 cudaStreamSynchronize, 2 D2H copies of the full target tensor, 2*H D2H copies of the head outputs (all synchronous, pageable), H H2D copies of the deltas, 2*H fresh `vector<float>` staging buffers, and tal_assign_head (the O(B*max_gt*cells) candidate scan + sort) executed twice on identical inputs. On the host, the DFL softmax over reg_max is evaluated three times per cell (tal_assign_head…

**Fix:** Make the v8 driver a single pass: for_each_v8_head takes both kernels, computes TAL once per head and returns the error alongside the gradient; have Loss::back_propagate for Error::Yolo call calculate_yolo(fp, target, &bp) once and take metrics.error from its EvaluationResult instead of calling calculate_error first. Verify with YoloLoss.V8* tests and the yolo example timing.

*Verifier:* loss.cpp:1362-1370 back_propagate calls calculate_error (-> calculate_yolo(nullptr) -> yolo_v8_error_multi, 1054-1074) then calculate_layers_error_gradient -> calculate_output_deltas -> calculate_yolo(&bp) -> yolo_v8_gradient_multi (1076-1096). Both go through for_each_v8_head (970-1040): per pass cudaStreamSynchronize + D2H of the target (986-988), per head D2H of the output (1014-1016),…

#### dataset-a-13 — Scaler methods round-trip enum -> string -> enum at four sites; unscale formulas hand-written despite scaling.h helpers

`opennn/dataset/tabular_dataset.cpp:989-1070` · medium · boilerplate · lines -12 · effort S · risk low · confirmed

get_feature_scalers (2855-2866) expands each Variable::scaler to a string per feature; calculate_used_feature_scaling (995-997), scale_features (1108), prepare_training_scaling (1170-1173) and scale_data (1080, which also parses scaler_method_to_string(...) back inside apply_scaler 1004) immediately convert those strings back with string_to_scaler_method. apply_scaler itself takes `const string& scaler` and parses it per feature column. The same functions also take `const string& variable_role` and go through string_to_variable_role again (prepare_training_scaling converts role -> string -> role at 1166/1171). Separately, apply_scaler writes the MinimumMaximum and MeanStandardDeviation…

**Fix:** Add `vector<ScalerMethod> get_feature_scaler_methods(VariableRole) const` (one loop, same shape as get_feature_scalers) and keep get_feature_scalers as a one-line string wrapper for Neural Designer. Change apply_scaler to take ScalerMethod and the internal callers to use the enum overloads of get_feature_indices/calculate_feature_descriptives. Replace the two hand-written inverses with the scaling.h helpers.

*Verifier:* get_feature_scalers (2855-2866) emits scaler_method_to_string per feature; calculate_used_feature_scaling (995-997), scale_features (1108 and 1124 string_to_scaler_method), prepare_training_scaling (1170-1173), unscale_features (1246, 1265) and scale_data (1080 scaler_method_to_string -> apply_scaler 1004 string_to_scaler_method) all round-trip. apply_scaler hand-writes the MinimumMaximum inverse…

#### training-loss-8 — Per-batch regularization penalty (full-parameter reduction + cuBLAS host sync) whose result the optimizer overwrites

`opennn/training_strategy/loss.cpp:1373-1376` · medium · overhead · lines -12 · effort S · risk low · partial

Loss::back_propagate calls add_regularization every batch, which reduces the entire parameter vector (l1/l2_regularization -> cublasSasum/cublasSdot with a host result pointer, i.e. a device synchronization per batch on GPU). The mini-batch optimizer never reads that value: finalize_epoch recomputes it once per epoch (optimizer.cpp:1831-1832 `metrics.regularization = loss->calculate_regularization(parameters); loss_value = result.error + regularization`), LM computes its own (levenberg_marquardt_algorithm.cpp:77), and the CUDA-graph path (back_propagate_device_metrics) does not compute it at all. The only consumer is quasi_newton_method.cpp:131-157/239, which reads metrics.loss_value after…

**Fix:** Remove lines 1373-1376 and Loss::add_regularization (1798-1811, plus its declaration); in QuasiNewtonMethod's train_step compute `loss_value = metrics.error + loss->calculate_regularization(parameters)` the way optimizer.cpp:1831 and LM already do. Keep add_regularization_gradient (the gradient term is needed every batch).

*Verifier:* Verified: loss.cpp:1373-1376 back_propagate calls add_regularization every batch; calculate_regularization -> l1/l2_regularization -> sum_abs_cuda/squared_norm_cuda (error_functions.cpp:45-56) use cublasSasum/cublasSdot with a HOST result pointer, i.e. a blocking sync per batch on GPU. optimizer.cpp:1825-1832 finalize_epoch recomputes regularization and loss_value once per epoch; LM computes its…

#### r2-arena-planner-and-propagation-structs-2 — Input-staging state is public only because NeuralNetwork and ChatSession hand-roll the staging

`opennn/neural_network/forward_propagation.h:127-140` · medium · design · lines -10 · effort M · risk low · unverified

Six public members exist solely so two outside functions can implement staging: `staged_input_storage`, `staged_inputs`, `host_bf16_input_scratch` (neural_network.cpp:1313-1398 only, ~85 lines of grow-or-resize + bf16 host cast + H2D copy), `host_bf16_output_scratch` (the two copy_device_to_host_float calls at neural_network.cpp:1153 and 2816), `position_pinned` (neural_network.cpp:2872 re-writes the pinned int by hand instead of reusing stage_position's host half) and `passthrough_overrides` (neural_network.cpp:1435 only). chat.cpp:1133-1139 `initialize_cuda_input` pre-sizes `staged_input_storage[0]` from outside so the steady-state allocation guard is not tripped during warmup. Every…

**Fix:** Move neural_network.cpp:1313-1398 verbatim into `const vector<TensorView>& ForwardPropagation::stage_inputs(span<const TensorView> inputs, bool cast_to_bf16(size_t), cudaStream_t)` (returns staged_inputs, reports whether anything was staged), add `reserve_staged_input(size_t index, Index bytes)` for chat's warmup pre-size, fold the 1435-1449 override loop into `bind_external_inputs(span<const TensorView>, Index first_layer)`, and expose `download_outputs(...)` over host_bf16_output_scratch.…

#### r2-set-vs-compile-device-ordering-8 — Convolutional constructor demotes GELU/SiLU to Identity with a warning while set_activation_function throws; a test now fails

`opennn/neural_network/layers/convolutional_layer.cpp:246-263` · medium · build/test · lines -10 · effort S · risk low · unverified

Convolutional::set (constructor path) turns an input-derivative activation into Identity and prints to cerr, whereas set_activation_function (281-293) throws for the same input. Besides the inconsistent policy (a JSON model or a constructor call silently loses its activation), the current tree fails ActivationsTest.ConvolutionalRejectsInputDerivativeActivations on a rebuilt CPU binary: `Expected: Convolutional(Shape{8, 8, 1}, Shape{3, 3, 1, 2}, "GELU") throws an exception of type exception. Actual: it throws nothing.` (three cases, activations_test.cpp:482-484). This is the only failure in the 55 Dense/Activation/Convolutional tests I ran on the rebuilt binary, and it cannot be seen on the…

**Fix:** Pick one policy and delete the other: make the constructor call set_activation_function (throwing, matching the test and the setter), or if the demotion is wanted for loading old YOLO models, move it into read_JSON_body only and update the test. Either way the cerr block and the duplicated check collapse into the existing throw_if.

#### core-types-6 — Shape(size_t rank, Index value) has zero callers and is the vector(n, v) trap next to Shape{a, b}

`opennn/core/tensor_types.h:192-199` · medium · API · lines -8 · effort S · risk medium · confirmed

grep over opennn/, tests/, examples/, docs/benchmarks finds no two-argument parenthesised Shape construction (the only paren two-arg use is the iterator-pair form in tensors_test.cpp:241). Yet the overload makes `Shape(2, 3)` mean rank-2 filled with 3 ([3, 3]) while `Shape{2, 3}` means [2, 3]; with Index-typed arguments both compile silently. Every real construction in the repo uses braces or push_back/append, so the constructor only exists to be misread.

**Fix:** Delete the (rank, value) constructor. Verify against Neural Designer first; if ND needs a filled shape, replace it with a named factory `static Shape filled(size_t rank, Index value)` so the intent is spelled out at the call site.

*Verifier:* tensor_types.h:192-199 matches the quote. grep for two-argument parenthesised Shape(...) over opennn/, tests/, examples/, docs/benchmarks finds only the iterator-pair template at tensor_types.h:212 and its test use at tensors_test.cpp:241 (`Shape(dimensions.begin(), dimensions.end())`). The (rank, value) overload has zero callers; the vector(n,v)-style trap is real. Deleting it is -8 lines; ND…

#### selection-testing-7 — GeneticAlgorithm keeps a full parameter vector for every individual of every generation; only the best is ever read

`opennn/model_selection/genetic_algorithm.cpp:185-238` · medium · overhead · lines -8 · effort S · risk low · confirmed

evaluate_population stores `individual_parameters(i) = neural_network->get_parameters_map()` for all 40 individuals (a VectorR copy each), and perform_input_selection reads only `individual_parameters(optimal_individual_index)` (line 525). Memory is individuals x parameters x 4 bytes held for the whole run (40 x 1M parameters = 160 MB; scales with population size), plus 40 heap copies per generation. The member also forces the three resize sites in set_default/set_individuals_number and the `= VectorR()` reset for the fold path.

**Fix:** Track the generation's best inside evaluate_population: keep `Index best_individual; VectorR best_parameters;` as members (or return them), and copy get_parameters_map() only when validation_error < the running best for this generation. Delete `Tensor<VectorR, 1> individual_parameters` and its three resize sites (lines 54, 87, 223); perform_input_selection uses best_parameters at line 525. minimal_index(validation_errors) can then go too.

*Verifier:* genetic_algorithm.h:61 `Tensor<VectorR, 1> individual_parameters;`; written at genetic_algorithm.cpp:213 for every individual (VectorR copy of the full buffer) and at :223 (fold path reset); the only read is :525 `individual_parameters(optimal_individual_index)` after `minimal_index(validation_errors)` at :497. Resize sites at :54 and :87. Keeping only the per-generation best (copy when…

#### layers-a-14 — Convolutional::set silently demotes unsupported activations and prints to std::cerr

`opennn/neural_network/layers/convolutional_layer.cpp:249-263` · medium · design · lines -8 · effort S · risk medium · confirmed

Convolutional::set replaces GELU/SiLU with Identity and writes a warning to std::cerr, while set_activation_function (lines 281-293) throws for the same input. A library that prints to stderr from a constructor is hidden I/O that callers (Neural Designer, tests) cannot capture or disable, and the two entry points give inconsistent results: Convolutional(..., "GELU", ...) builds an Identity layer, conv.set_activation_function("GELU") throws. A model built from a config that names SiLU ends up without its nonlinearity and no exception records it; the comment says the YOLO builder compensates, which is exactly the kind of coupling that drifts.

**Fix:** Make set() delegate to set_activation_function(new_activation_function) so both paths throw the same throw_if (set_activation_function already calls update_convolution_operator; set() then only needs the batch_norm.features assignment before it). If old saved models with inline SiLU must still load, handle that once in the YOLO builder / registry loader with an explicit Activation layer insertion instead of a silent demotion. Remove the <cstdio>/<iostream> dependency from the hot header path as…

*Verifier:* convolutional_layer.cpp:246-260 demotes GELU/GELUTanh/SiLU (activation_needs_input, tensor_operations.cpp:846-850) to Identity with a std::cerr warning, while set_activation_function (281-293) throws for the same input. Two further facts raise the severity: (1) tests/neural_network/layers/activations_test.cpp:480-487 (ConvolutionalRejectsInputDerivativeActivations) EXPECT_THROWs the constructor…

#### r2-duplicated-kernels-across-folders-3 — Softmax single-home: CPU row softmax and three CPU/cuDNN softmax-backward copies belong in tensor_operations

`opennn/neural_network/operators/attention_operator.cpp:100-108` · medium · duplication · lines -8 · effort M · risk low · partial

Inventory of the ten softmax implementations in scope and what can merge. INTERCHANGEABLE (same arithmetic, same epsilon-free normalisation): (1) tensor_operations.cpp:1145-1159 softmax_cpu and (2) attention_operator.cpp:100-108 softmax_rows_prefix - identical Eigen max/exp/sum except that the latter takes a prefix length; (3) the Eigen softmax backward `dot = (y*dY).rowwise().sum(); dY = y*(dY.colwise()-dot)` written at attention_operator.cpp:1117-1119 and again at :1143-1146, and (4) the cudnnSoftmaxBackward call injected as a lambda at :619-630, which is the only reason apply_delta_unfused is a template on SoftmaxBwd. MUST STAY as kernels (numerics or fusion differ on purpose):…

**Fix:** In tensor_operations: (a) add `void softmax_rows(float*, Index rows, Index cols, Index length)` (the OpenMP-at-65536 loop from softmax_cpu) and make softmax_cpu call it with length = cols; delete AttentionOperator::softmax_rows_prefix and call softmax_rows from the fast path. (b) Add `void softmax_backward(const TensorView& y, TensorView& dy)` next to softmax(): CPU body = the two Eigen lines, CUDA body = the cudnnSoftmaxBackward call now at attention_operator.cpp:619-630 (same descriptor…

*Verifier:* Substance confirmed: attention_operator.cpp:100-108 softmax_rows_prefix is the same max/exp/sum as tensor_operations.cpp:1145-1159 softmax_cpu minus the OpenMP clause and plus a prefix length; the Eigen softmax backward appears at :1117-1119 (leftCols(valid_length) block) and :1143-1146 (flat), and the cudnnSoftmaxBackward lambda at :619-630 is the only reason apply_delta_unfused is templated…

#### xcut-boilerplate-4 — Class layout depends on OPENNN_HAS_CUDA: data members wrapped in #ifdef in 5 headers

`opennn/neural_network/layers/convolutional_layer.h:103-108` · medium · API · lines -4 · effort S · risk medium · partial

Pattern (d), header side. Of the 185 OPENNN_HAS_CUDA blocks, 16 are in .h files and five of them wrap data members, so sizeof and member offsets of public classes differ between a CUDA and a CPU build: convolutional_layer.h:103-108 (`Buffer folded_parameters; bool folded_dirty`), pooling_layer.h:70-83 (`CudnnDescriptor<cudnnPoolingDescriptor_t> pooling_descriptor`, with the `private:` label itself inside the #ifdef), attention_operator.h:187-190 (`uint64_t sdpa_dropout_offset`), device_backend.h:340-343 (StreamCapture `cudaStream_t stream; bool finished`), cudnn_rnn.h:37-44 (the CudnnRnnConfig type). Neural Designer compiles against these headers; a consumer TU built without the define (or…

**Fix:** Make the data members (and the cudnn_rnn.h config struct) unconditional, relying on the stub typedefs; keep only method bodies under #ifdef in the .cpp, where the OPENNN_CUDA_STUB pattern already applies. Remove the two folded_dirty guards in convolutional_layer.cpp. Build both build dirs; the CPU build must link with the existing stub bodies (CudnnDescriptor's destructor must be a no-op for the stub type - check core/tensor_types.h before merging).

*Verifier:* Layout divergence is real: convolutional_layer.h:103-108, pooling_layer.h:70-83 (private: label inside the #ifdef), attention_operator.h:187-190, device_backend.h:340-343, cudnn_rnn.h:37-44 all wrap data members. Stub typedefs for cudaStream_t, cudnnPoolingDescriptor_t exist at opennn_types.h:101-130 and CudnnDescriptor (device_backend.h:20-51) already has a CPU deleter branch, so those four can…

#### operators-b-5 — C2PSA CPU back_propagate heap-allocates six Eigen matrices per batch element per call although a backward scratch slot is already planned

`opennn/neural_network/operators/c2psa_operator.cpp:358-384` · medium · overhead · lines -4 · effort S · risk low · confirmed

The CPU backward materializes d_concat (BT x C) once and then, inside the per-batch loop, allocates d_ao, dV, dA (tokens x tokens, 640 KB for 20x20 tokens), dQ and dK on every iteration: 1 + 5*B malloc/free pairs per backward step. The GPU path carves all of these out of bp.slots[layer][backward_scratch_slot], but C2PSA::get_backward_specs (c2psa_layer.cpp:80-82) plans that slot as Shape{} when get_compute_device() != CUDA, so the CPU path cannot use it. The CPU forward also keeps its per-batch loop serial while every other CPU attention loop in the library is OpenMP-parallel.

**Fix:** Plan the backward scratch on CPU too (drop the device condition in C2PSA::get_backward_specs, same layout as the GPU carve-up) and map d_cat/d_A/dQ/dK/dV/d_ao onto it with Eigen::Map in the CPU backward; then parallelize the per-batch loop with `#pragma omp parallel for` and accumulate dWq/dWk/dWv per thread or after the loop via the mapped dQ/dK/dV blocks (dW = xa^T * dQ over the flattened (BT, H) maps, one GEMM each, no per-batch accumulation). Validate with the C2PSA numerical gradient test.

*Verifier:* c2psa_operator.cpp:358-384: MatF d_concat once, then per batch element MatF d_ao (copy of a block), dV, dA (tokens x tokens), dQ, dK — five heap allocations per b per backward. GPU path (228-235) carves the same six regions from bp.slots[layer][backward_scratch_slot]; C2PSA::get_backward_specs (c2psa_layer.cpp:79-82) plans that slot as Shape{} off CUDA. CPU forward loop (163-192) is serial with…

#### training-loss-3 — MSVC compiles the v8 TAL gradient kernel at /Od, and that kernel is the GPU training path too

`opennn/training_strategy/loss.cpp:807-968` · medium · overhead · lines -4 · effort S · risk medium · partial

Commit 7781255f9 wrapped yolo_gradient_kernel in `#pragma optimize("", off)` to dodge a VS 2026 C1001 ICE, justified as 'cold CPU reference kernel, no practical perf cost'. The same pragma pair was later copied around yolo_v8_gradient_kernel_tal (807-809 / 966-968). That kernel is not cold: for_each_v8_head stages device outputs to the host and runs it on every GPU training batch (lines 1010-1032), so on MSVC the per-batch v8 gradient (nested B x G x G x C loops with per-cell softmaxes) executes unoptimized. The v1 kernel (339-438) is likewise the CPU training path, not a reference only.

**Fix:** Re-test the ICE on the installed VS 18 toolchain (AGENTS.md build dirs) and drop the pragmas if it is gone. If it persists, restructure instead of disabling optimization: move the per-cell body (box decode + DFL + class terms) into a separate static function so the nested loop is shallow, or limit the pragma to the one function that actually reproduces it.

*Verifier:* Pragmas verified at loss.cpp:807-809/966-968 (v8) and around yolo_gradient_kernel (~339/438). Hot-path claim verified: for_each_v8_head (970-1040) stages to host and runs yolo_v8_gradient_kernel_tal on every GPU training batch; yolo_gradient_kernel is the CPU training path via yolo_gradient_cpu (445-457, 1438-1441). Corrections: the v8 pragma arrived with the v8 head commit af5cacdd3, and the…

#### training-optimizers-3 — LM allocates a validation Jacobian (V*outputs x P) and a P x P Hessian that validation never reads

`opennn/training_strategy/levenberg_marquardt_algorithm.cpp:278-312` · medium · overhead · lines -3 · effort S · risk low · confirmed

validation_back_propagation_lm is a full BackPropagationLM: its set() zero-fills squared_errors_jacobian (validation_samples*outputs x parameters) and hessian (parameters x parameters) plus gradient, but hooks.validation_error only ever calls calculate_errors/calculate_squared_errors/calculate_error on it, i.e. it needs the errors vector and one float. For a 10k-parameter network with 10k validation samples and one output that is a 400 MB Jacobian plus a 400 MB Hessian of zeros allocated and touched once per train() for nothing; the training-side copies already exist. The same shape of waste exists in QuasiNewtonMethod (finding -4).

**Fix:** Replace the validation BackPropagationLM with a direct mean-squared-error over the validation outputs: `const VectorMap output = context.validation_forward_propagation->get_last_trainable_layer_outputs().as_vector(); const VectorMap target = context.validation_batch->get_targets().as_vector(); return (output - target).squaredNorm() / float(output.size());` (same arithmetic as calculate_errors + calculate_squared_errors + calculate_error). Delete validation_back_propagation_lm.

*Verifier:* levenberg_marquardt_algorithm.cpp:278 constructs validation_back_propagation_lm; BackPropagationLM::set (445-465) zero-fills squared_errors_jacobian (total_error_terms x parameters) and hessian (parameters x parameters). hooks.validation_error (304-311) only calls calculate_errors (87-98: errors = output - target), calculate_squared_errors, calculate_error (107-113: squared_errors.sum()/size).…

#### xcut-build-tests-10 — Global -Wno-unused-result silences every [[nodiscard]] in the library on the CI compilers

`CMakeLists.txt:75-80` · medium · build/test · lines -2 · effort S · risk low · confirmed

GCC and Clang report ignored `[[nodiscard]]` results under `-Wunused-result`; the root `add_compile_options(-Wno-unused-result)` applies to every target, including opennn. The library's six `[[nodiscard]]` annotations (configuration.h:41-42 `resolve()`/`resolve_for()`, response_constraints.h:150-164) are therefore inert exactly on the Linux CI build and on the clang-cl Windows build (the stale compile DB shows `-Wno-unused-result` on every library TU). A grep finds no unchecked fread/fwrite/system sites that would justify the suppression; it is a blanket flag that hides real problems. The same block's `-Wno-switch-enum` is then contradicted by `-Wswitch-enum` on the opennn target…

**Fix:** Remove `-Wno-unused-result` (and `-Wno-switch-enum`, which the opennn target overrides anyway); if a third-party dependency needs it, set it on that target only. Build both dirs and fix whatever surfaces (expected: a few ignored `resolve()` results, which is the point of the attribute).

*Verifier:* CMakeLists.txt:75-80 adds -Wno-switch-enum/-Wno-unused-parameter/-Wno-unused-result directory-wide for GNU|Clang; both compilers report ignored [[nodiscard]] under -Wunused-result. The six annotations are at configuration.h:41-42 and response_constraints.h:150-164 as cited. opennn/CMakeLists.txt:186-189 adds -Wswitch-enum on the target, contradicting the root -Wno-switch-enum. Removing the two…

#### operators-a-6 — CPU convolution allocates and zeroes a workers x weights gradient buffer plus im2col scratch on every call

`opennn/neural_network/operators/convolution_operator.cpp:526-533` · medium · overhead · lines -2 · effort M · risk low · partial

apply_cpu allocates `vector<float> scratch(workers * col_size)` per forward (line 483) and apply_delta_cpu allocates `MatrixR::Zero(workers, kernels_number * patch_size)`, `bias_gradient_partials` and a `2 * col_size` scratch per backward. The partials buffer scales with the weight count times the thread count: a 3x3x512x512 ResNet block has 2.36M weights (9.4 MB); with 16 workers that is 151 MB allocated, page-faulted and zeroed, then reduced with colwise().sum(), on every backward of every such layer, every step. For the CPU im2col path this dominates small-batch steps and defeats the cache locality the per-thread slicing was designed for. ConcurrentMixedShapesUseIndependentCpuScratch…

**Fix:** Replace the per-call vectors/matrices with a file-local `thread_local vector<float>` that grows monotonically (resize only when larger is needed) and slice it for im2col scratch and the per-worker gradient partials; zero only the partial region that is used. The caller thread owns the buffer, so concurrent callers on different threads keep independent scratch and the existing test still passes. Measure with the CPU conv benchmark before/after (the prior audit asked for benchmarks on hot-path…

*Verifier:* Code confirmed: convolution_operator.cpp:483 `vector<float> scratch(workers*col_size)` per forward; 526-533 `MatrixR::Zero(workers, kernels_number*patch_size)`, bias partials and `scratch(workers*scratch_stride)` per backward, reduced afterwards. Arithmetic for a 3x3x512x512 layer with 16 workers (~151 MB zeroed per backward) checks out. ConcurrentMixedShapesUseIndependentCpuScratch exists…

#### training-optimizers-6 — GPU validation loop host-syncs the compute stream after every batch; the Batch event machinery already covers the hazard

`opennn/training_strategy/optimizer.cpp:1762-1770` · medium · overhead · lines -1 · effort S · risk medium · confirmed

run_epoch_loop calls device::synchronize(compute) after every non-training batch. The hazard it guards is real: the pooled validation batch is pushed back to the empty queue right after, and a worker may refill its host buffer and issue a new H2D while the GPU still reads its device buffer. But Batch already solves this without a host sync: record_h2d_done(stream) records an event (batch.cpp:403-410), the worker calls wait_h2d_complete() before fill (optimizer.cpp:351, batch.cpp:412-419), and the training paths use exactly that (slot.record_h2d_done(compute) at 1648; the fixed-batch path at 1758). In the device-metrics validation path (MSE/CE losses) this synchronize is the only host sync…

**Fix:** Replace the synchronize with `current_batch->record_h2d_done(device::get_compute_stream());` (only when !use_fixed_device_batch, which is always the case for validation). The worker's wait_h2d_complete then blocks on compute completion for that batch before refilling, the next batch's H2D on the transfer stream keeps overlapping, and the host no longer drains the stream per batch. The final device_metrics.read() / calculate_error already synchronize where host values are needed. Validate with…

*Verifier:* optimizer.cpp:1762-1768: after sync_device, `if (on_gpu && context.fill_mode != FillMode::Training) device::synchronize(compute)` then push to empty_queue. Validation uses pooled batches (fixed_device_batch nullptr at 2227) uploaded via prefetch_batch (1308-1313) on the transfer stream; the worker calls wait_h2d_complete before fill (351-356), which blocks on h2d_done_event only if…

#### xcut-build-tests-9 — Tests link bare `gtest gtest_main`, which only exist in the FetchContent path, and gtest_main is redundant

`tests/CMakeLists.txt:12-26` · medium · build/test · lines -1 · effort S · risk low · confirmed

`find_package(GTest 1.11 CONFIG QUIET)` is tried first; an installed GTest config exports `GTest::gtest`/`GTest::gtest_main`, not the bare `gtest` target names, so when a system GTest is found the link line degrades to `-lgtest -lgtest_main` library-name lookups (works by accident on Debian where libgtest.a is in the default lib dir, fails elsewhere, and never propagates GTest's include dirs/usage requirements). Additionally tests/test.cpp defines its own `main` (to install the CPU listener), so linking `gtest_main` is dead weight and only works because the archive's main is never pulled.

**Fix:** `target_link_libraries(opennn_tests PRIVATE GTest::gtest opennn)` - the FetchContent googletest (>= 1.11) defines the `GTest::gtest` alias as well, so one spelling covers both paths. Drop gtest_main.

*Verifier:* tests/CMakeLists.txt:12 `find_package(GTest 1.11 CONFIG QUIET)`, 14-16 FetchContent fallback, 26 `target_link_libraries(opennn_tests gtest gtest_main opennn)`. GTestConfig exports GTest::gtest/GTest::gtest_main only; bare `gtest` becomes a -lgtest name lookup with no usage requirements. tests/test.cpp:31 defines main, so gtest_main is unused. googletest >= 1.11 defines the GTest:: aliases in the…

#### r2-batch-pipeline-and-device-gather-6 — BF16 host cast of the whole input batch runs on the GPU-feeding main thread inside upload_to_device_batch_async instead of in the prefetch worker

`opennn/dataset/batch.cpp:336-348` · medium · overhead · lines 0 · effort S · risk low · confirmed

For BF16 inputs without device residency, upload_to_device_batch_async converts `input_values_count` floats with float_2_bfloat16_host before issuing the copy. Every caller of this function is the main training thread (run_graph_epoch line 1641, run_epoch_loop line 1734, prefetch_batch), i.e. the thread whose only job is to keep the compute stream fed; BF16 batches are excluded from the staged/grouped graph path (`staged_h2d` requires input.type != BF16), so this cast sits serially between session->wait(iteration) and the graph launch on every step. For a ResNet-50 BF16 batch of 64 that is 9.6M conversions (~2 ms with OpenMP, and the `#pragma omp parallel for` region competes with the…

**Fix:** Move the cast into Batch::fill (worker side): after dataset->fill_batch(...), `if (input_host_bf16 && !device_gather) float_2_bfloat16_host(input.shape.size(), input.host.as<float>(), input_host_bf16.as<uint16_t>());`. upload_to_device_batch_async then only issues the copy from input_host_bf16. The tail batches (train_tail/evaluate_tail) call fill() too, so they are covered. Verify with the BF16 GPU comparison tests and the ResNet-50 BF16 benchmark.

*Verifier:* batch.cpp 336-348: float_2_bfloat16_host runs inside upload_to_device_batch_async. All callers are on the main training thread: prefetch_batch (1312, used by run_epoch_loop 1738 and the tails 1870/2143) and host_batch->upload_to_device_batch_async at 1641; staged_h2d excludes BF16 (1394-1395) so BF16 non-resident batches go through the single-slot else branch (1616-1660) where the cast sits…

#### xcut-boilerplate-7 — TabularDataset::read_csv is a single 713-line function with a 230-line OpenMP region inside

`opennn/dataset/tabular_dataset.cpp:1618-2330` · medium · design · lines 0 · effort L · risk medium · confirmed

Pattern (k). read_csv runs from 1618 to 2330 as one function with 62 top-level statements and no phase comments; its phases are: CSV parse + separator (1620-1700), sample-id / variable-type inference (1700-1800), binary-cache setup (1880), per-type token pass (1960-2040), a `#pragma omp parallel` fill region with three `#pragma omp critical` sections (2102-2190), and a missing-values / numeric post-pass (2200-2330). A bug in any phase must be reasoned about against every local variable of the whole function, and the OpenMP region cannot be unit-tested in isolation.

**Fix:** Extract along the six phases into private members that pass the parsed CsvReader::Result and the variable_token_indices vector explicitly (infer_variable_types, prepare_cache, fill_data_parallel, summarize_missing_values). Relocate verbatim first; the existing tabular_dataset tests cover csv round-trips and missing values.

*Verifier:* Column-0 scan of tabular_dataset.cpp: read_csv opens at 1618, closes at 2330, with #pragma omp parallel at 2102, omp for at 2114 and three omp critical sections at 2138/2170/2182; the next definition (missing_values_method_map) starts at 2332. A 713-line member with an embedded OpenMP region matches the description. Relocate-verbatim extraction is the right shape and LOC-neutral.

#### layers-a-10 — AdditionOperator forward spends a full copy pass before the first add

`opennn/neural_network/layers/addition_layer.cpp:18-26` · medium · overhead · lines 0 · effort S · risk low · confirmed

The forward does copy(inputs[0], output) and then add(output, inputs[i], output) for i >= 1. For the common two-input case (every transformer residual: standard_networks.cpp:1515,1528,1686,1698 add two Addition layers per block) that is a read+write copy pass followed by a two-read+one-write add pass: five passes over a batch x seq x embed tensor where three suffice. The tensor-op add already accepts three distinct views (tensor_operations.cpp:1020-1037), so the first add can consume inputs[0] and inputs[1] directly. Memory-bound, so the saving is about 40% of this layer's traffic per residual add on both CPU and CUDA.

**Fix:** add(inputs[0], inputs[1], output); for (size_t i = 2; i < inputs.size(); ++i) add(output, inputs[i], output); (Addition::set already enforces inputs_number >= 2, addition_layer.cpp:57). Verify with the existing addition/activations tests and a quick OPENNN_PROFILE run on the transformer example.

*Verifier:* addition_layer.cpp:18-26: copy(inputs[0], output) then add(output, inputs[i], output) for i>=1, i.e. for two inputs one read+write copy pass and one 2-read+1-write add pass. add() (tensor_operations.cpp:1020-1037) only requires same shape/device/type across the three views and on CPU uses Eigen noalias, while add_gpu (1541-1555) is either add_relu_cuda or cudnnOpTensor over two arbitrary inputs,…

#### layers-b-6 — LSTM CPU backward (H<96) allocates and zero-fills per-thread gradient scratch for omp_get_max_threads() every call

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:728-737` · medium · overhead · lines 0 · effort S · risk low · confirmed

Each apply_delta call allocates nthreads*(4H + 4FH + 4HH) floats, zero-fills them, and then serially reduces all nthreads copies into the gradients (lines 875-900) regardless of how many sequences the batch has. For the forecasting benchmark scenarios (H=64, F=8: 18,688 floats = 75 KB per thread) on a 32-thread machine that is 2.4 MB allocated, zeroed and reduced per mini-batch; with batch_size 4 (the test sizes) 28 of the 32 copies are reduced for nothing. grouped_attention_forward in this same scope already sizes its worker scratch by min(omp_get_max_threads(), work) and pins the team with num_threads(workers) (grouped_query_attention_layer.cpp:220-226).

**Fix:** const int workers = max(1, min(omp_get_max_threads(), to_int(batch_size))); size gradient_thread_scratch by workers; add num_threads(workers) to the parallel for; reduce over workers. Optionally keep the vector as a mutable operator member (grow-only) to drop the per-call allocation entirely.

*Verifier:* long_short_term_memory_layer.cpp:728-737 sizes gradient_thread_scratch by omp_get_max_threads() and value-initializes it on every apply_delta call; the `#pragma omp parallel for` at 738 carries no num_threads clause; the reduction at 875-905 loops tid over all nthreads regardless of batch_size. Per-thread size 4H+4FH+4HH matches the code (730-733). grouped_attention_forward in the same scope…

#### xcut-boilerplate-6 — ModelExpression::get_expression_c_embedded is a single 960-line function

`opennn/neural_network/model_expression.cpp:713-1672` · medium · design · lines 0 · effort L · risk low · confirmed

Pattern (k). The column-0 scan shows one function from 713 to 1672 (the next definition, get_expression_php, starts at 1674). The PHP/Python/JS exporters in the same file were already split into emit_php_*/emit_python_*/emit_js_* helpers (model_expression.h:48-68), but the C-embedded exporter kept everything inline: an activation-name switch inside a lambda (760-780), two buffer-management lambdas (822-833), a per-layer-type else-if chain (e.g. 1203 `else if(layer_type == LayerType::Recurrent)`) that emits scaling, dense, recurrent and LSTM code (1100-1420), then the shared runtime (1500-1660). Any change to one layer's C emission requires navigating the whole function, and the existing…

**Fix:** Split along the existing seams into private emit_c_embedded_scaling/dense/recurrent/lstm/runtime(ostringstream&, ...) members mirroring the emit_php_* set; hoist the activation-name lambda to a static table next to ActivationBodies. Pure text output, so verify by diffing the generated C for every example network against the pre-split output (the tests already exercise get_expression_c_embedded).

*Verifier:* Column-0 scan of model_expression.cpp shows get_expression_c_embedded at 713 with its closing brace at 1672 and get_expression_php at 1674 - one 960-line function. model_expression.h:48-68 declares emit_c_*/emit_php_*/emit_python_*/emit_js_* helpers and get_expression_c (model_expression.cpp:599-602) uses them; the embedded variant only reuses emit_c_main at 1667. Pure text output; split is low…

#### xcut-api-2 — calculate_outputs_resident defaults upload_parameters=true, which discards the CUDA graph every call

`opennn/neural_network/neural_network.h:226-228` · medium · API · lines 0 · effort S · risk medium · confirmed

The resident fast path takes bool upload_parameters = true. With the default, every call runs copy_parameters_device() (for BF16 networks that re-runs cast_parameters_to_bf16 over all parameters and link_parameters; see cpp:2510-2545), copy_states_device(), and forward_propagation.reset_cuda_graph(), which destroys inference_graph_exec and sets cuda_graph_warmup_calls = 0 (forward_propagation.cpp:1208-1214). The graph capture at cpp:2870 only happens after inference_graph_warmup_calls (=2) consecutive calls, so a caller that uses the default never reaches graph replay and pays a full parameter re-cast per inference. Every one of the 30 in-repo call sites (chat.cpp, gpu_comparison_test,…

**Fix:** Remove the default argument so the choice is made at the call site (compile-time loud; no in-repo caller relies on the default). Better still, split the upload out: a public upload_parameters_for_inference() (copy_parameters_device + copy_states_device + reset graph) called once, and calculate_outputs_resident(inputs, ForwardPropagation&) with no flag. Verify against Neural Designer before removing the default.

*Verifier:* h:226-228 default upload_parameters = true; cpp:2849-2855 calls copy_parameters_device() (cpp:2510-2545, including cast_parameters_to_bf16 for BF16) + copy_states_device() + reset_cuda_graph(); forward_propagation.cpp:1208-1214 resets inference_graph_exec and cuda_graph_warmup_calls; capture needs inference_graph_warmup_calls = 2 (cpp:2841, 2888). Every one of the 33 call lines outside…

#### xcut-build-tests-15 — MemoryAudit: seven assertion-free 'TEMPORARY DRIVER' tests run in every suite invocation

`tests/neural_network/memory_audit_test.cpp:4-149` · medium · build/test · lines 0 · effort S · risk low · confirmed

The file's own header says it is a measurement driver, not a regression test, to be run with OPENNN_MEMORY_DEBUG=1 and a gtest filter. But none of its seven TEST()s is DISABLED_, so every `opennn_tests` run (CI included) builds a 4-layer seq=256/batch=32 Transformer plus a T=200/hidden=256 LSTM, allocates their joint forward/backward arenas (CPU and, in the CUDA build, three more on the GPU) and dumps `memory_debug::print` to stdout, with zero EXPECT/ASSERT. It adds runtime, output noise, and GPU memory pressure to the suite while never being able to fail.

**Fix:** Prefix the seven tests with `DISABLED_` (they stay runnable via `--gtest_also_run_disabled_tests --gtest_filter=MemoryAudit.*`), or move the driver to docs/benchmarks/footprint where opennn_memory already lives. Alternatively turn them into real tests by asserting the arena byte sizes against recorded values.

*Verifier:* memory_audit_test.cpp header lines 4-9 say 'TEMPORARY DRIVER ... Run with OPENNN_MEMORY_DEBUG=1 and --gtest_filter=MemoryAudit.*'; seven TEST(MemoryAudit, ...) at 105/111/117/125/131/137/143, none DISABLED_; grep finds zero EXPECT/ASSERT; memory_debug::print(cout) at 53. DISABLED_ prefix or moving to docs/benchmarks/footprint is right.

#### r2-arena-planner-and-propagation-structs-3 — calculate_outputs tail tile builds a second ForwardPropagation with its own arena instead of sharing the tile's

`opennn/neural_network/neural_network.cpp:1202-1217` · medium · overhead · lines +1 · effort S · risk low · unverified

The tiled inference path constructs `tile_propagation` (tile_rows_max rows) and, for the remainder tile, a second `ForwardPropagation(rows, ...)` that plans and allocates its own arena (a second device cudaMalloc/cudaFree pair per calculate_outputs call whenever batch_size % tile_rows_max != 0, on top of the construction cost MEMORY already measured at 2.3x the inference itself). The tile arena is idle by then and always at least as large (same network, fewer rows), which is exactly the case ForwardPropagation::set's external_storage branch handles (`external_storage->byte_size() >= total_bytes` -> set_view); optimizer.cpp:845-849 already uses it for the validation propagation.

**Fix:** Replace the make_unique with `tail_propagation = make_unique<ForwardPropagation>(); tail_propagation->set(rows, this, &tile_propagation.arena, ForwardPropagationMode::Inference);` mirroring optimizer.cpp:845. The tail runs after the last full tile and its outputs are memcpy'd out before the next call, so aliasing is safe.

#### r2-batch-pipeline-and-device-gather-4 — Validation loop does a device-wide stream synchronize after every batch to protect pool device buffers; an event on the batch would do

`opennn/training_strategy/optimizer.cpp:1761-1770` · medium · overhead · lines +1 · effort S · risk medium · partial

run_epoch_loop in validation mode (`fill_mode != Training`) calls device::synchronize(compute) after every step. The reason is the buffer-reuse hazard analysed in the non-fixed path: pool batches own device buffers, prefetch_batch() self-uploads into them on the transfer stream, and the worker's wait_h2d_complete() only waits on the batch's event, which was recorded on the transfer stream at upload time — it does not cover the compute step that reads the buffer. The per-step synchronize closes that hole but serialises the whole validation epoch: the forward of batch i+1 cannot be enqueued until batch i has fully drained, so validation on small/medium networks becomes launch-latency bound,…

**Fix:** Replace the synchronize with `if (on_gpu && !use_fixed_device_batch) current_batch->record_h2d_done(device::get_compute_stream());` before pushing the batch back. The worker's wait_h2d_complete() then waits for the compute read, and prefetch_batch() of a re-published batch is ordered after it. Keep a single device::synchronize after the loop (evaluate_epoch already reads device_metrics after run_epoch_loop). The host-metrics path (calculate_error) synchronises internally when it reads the…

*Verifier:* The per-batch device::synchronize in validation mode is real (optimizer.cpp 1762-1768) and the hazard analysis is correct: validation batches are pool batches with device buffers (setup_batch_pools 296-300, or the shared training pool which is not prefetch_only when validation reuses it, 256-259), prefetch_batch self-uploads (1308-1313), and the worker only waits on the batch's event (351)…

#### dataset-a-7 — Correlation loops re-gather variable columns O(inputs x targets) and O(inputs^2)

`opennn/dataset/tabular_dataset.cpp:897-909` · medium · overhead · lines +2 · effort S · risk low · confirmed

In calculate_input_target_variable_correlations the target column is copied (get_variable_data with row indices -> fill_tensor_data) once per (i, j) pair instead of once per j; in calculate_input_variable_correlations input_j is re-gathered for every i (n^2/2 full-column copies). Each gather also recomputes the feature-offset prefix sum (transform_reduce over variables, lines 101-102 / dataset.cpp 744-745) and get_feature_indices twice (108-109). For 200 inputs x 1M rows that is ~20,000 extra 4 MB copies before a single correlation is computed, and under `omp parallel for schedule(dynamic)` the target copies are done concurrently by every thread.

**Fix:** Gather `vector<MatrixR> target_data` (and, in the input-input version, `vector<MatrixR> input_data`) once before the loops with the used sample indices, then index them inside. Memory is one extra copy of the used data, which these functions already create transiently.

*Verifier:* tabular_dataset.cpp:897-909: inside 'omp parallel for' over i, the target column is re-gathered per (i,j) via get_variable_data(target, used_sample_indices) (fill_tensor_data copy plus get_feature_indices computed twice at 108-109). calculate_input_variable_correlations (954-958) re-gathers input_j for every i. Pre-gathering once is a straightforward change, ~2 LOC.

#### training-loss-7 — YoloLambdas positional init shifted when `dfl` was inserted; tests now run with lambda_class = 0

`opennn/training_strategy/loss.h:216-224` · medium · build/test · lines +2 · effort S · risk low · confirmed

YoloLambdas is a six-float aggregate; commit 6d7bb211f inserted `dfl` as the second member. tests/training_strategy/yolo_loss_check_test.cpp:48 and :91 still initialise it positionally as `{5.0f, 0.5f, 2.0f, 0.0f}`, which the reader (and the original author) takes as {giou, noobj, cls, focal} but now means giou=5, dfl=0.5, noobj=2.0, cls=0.0. The expected value at line ~105 multiplies by `ev_lam.cls`, so both tests stay green while the class-loss term of yolo_error_kernel/yolo_gradient_kernel is no longer exercised at all and the no-object weight under test is 4x the default. loss.cpp:1397 builds the same struct positionally from six Loss members, the same hazard one insertion away.

**Fix:** Use designated initialisers at the two test sites ({.giou = 5.0f, .noobj = 0.5f, .cls = 2.0f}) and confirm the gradient check still passes with the class term active. In loss.h add a `YoloLambdas yolo_lambdas() const` that returns `{.giou = yolo_lambda_giou, .dfl = ..., ...}` and use it at loss.cpp:1397 so no positional six-float construction remains.

*Verifier:* loss.h:218-226: YoloLambdas = {giou, dfl, noobj, cls, focal_gamma, obj_focal_gamma}. tests/training_strategy/yolo_loss_check_test.cpp:48 and :91 use `{5.0f, 0.5f, 2.0f, 0.0f}` positionally -> giou=5, dfl=0.5, noobj=2.0, cls=0.0. Line 101 builds expA from `ev_lam.cls`, so the test is self-consistent and green with the class term disabled. git log -S'float dfl' -- opennn/loss.h shows 6d7bb211f…

#### layers-b-2 — LSTM allocates six unused B×T×H gate tensors on CUDA; cuDNN-only scratch allocated on CPU

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:1096-1127` · medium · overhead · lines +4 · effort S · risk low · confirmed

On CUDA the LSTM forward goes to apply_gpu, which touches only the Output slot (and HiddenState when !return_sequences); ForgetGate, InputGate, CandidateGate, OutputGate, CellState and CellActivation are never written or read (cuDNN keeps its own reserve space in layer_state_storage). They are still pooled: for memory_audit_test's LSTM (B=64, T=200, H=256) that is 6*64*200*256*4 = 78.6 MB of arena per layer, per ForwardPropagation. The mirror waste exists on CPU: get_backward_specs always allocates CudnnOutputDeltaScratch (B*T*H) and CudnnInputDeltaScratch (B*T*F), which only apply_delta_gpu reads, and on CUDA the six B*H step scratches used only by the CPU apply_delta.…

**Fix:** In LongShortTermMemory::get_forward_specs: const bool cuda = get_compute_device() == Device::CUDA; const Shape gate = cuda ? Shape{} : sequence_shape; use `gate` for the six gate/cell slots (keep HiddenState and Output). In get_backward_specs: use Shape{} for the six scratch_shape slots when cuda, and Shape{} for the two Cudnn*Scratch slots when !cuda. Same !cuda rule for Recurrent's CudnnInputDeltaScratchSlot. apply()/apply_delta() already receive these views by reference and only dereference…

*Verifier:* forward_propagate (152-172) routes CUDA to apply_gpu(input, output, hidden_state, ...) which touches only OutputSlot and, when !return_seq, HiddenStateSlot as sequence_output_scratch (998-1012); ForgetGate..CellActivation are never referenced on the GPU path, yet get_forward_specs (1096-1110) allocates all seven B*T*H slots with compute_dtype. back_propagate (480-518) uses the six B*H scratches…

#### core-device-5 — autotune_with_scratch sizes every scratch tensor as fp32: 2x the real transient on bf16 graphs

`opennn/core/cuda/cudnn_frontend_utilities.h:679-712` · medium · overhead · lines +6 · effort S · risk low · confirmed

The scratch duplicates each non-pass-by-value tensor with elements * sizeof(float) regardless of the tensor's data type. For bf16 conv/BN graphs (the ResNet-50 capacity and speed benchmarks, which call set_conv_autotune(true)) that is twice the footprint of the tensors being mirrored. The function's own comment says this is 'the largest transient allocation the conv path makes' and that failing it downgrades to the heuristic plan; doubling it unnecessarily lowers the batch at which autotune still fits and, at the capacity frontier, turns a tunable batch point into an 'autotune skipped' one. Tensor_attributes exposes get_data_type() (graph_properties.h:288).

**Fix:** Add a tiny element_bytes(DataType_t) helper (BFLOAT16/HALF -> 2, INT8/UINT8 -> 1, else 4) and size the scratch with tensor->get_data_type(); keep the fp32 fallback for NOT_SET. Verify on the resnet50-max-batch trial that the autotune-on max batch does not drop.

*Verifier:* cudnn_frontend_utilities.h:696-704: `buffer.resize_bytes(Index(elements * int64_t(sizeof(float))), ...)` for every non-pass-by-value tensor regardless of dtype. Tensors are typed via to_dtype(Type) (150+), so bf16 graphs carry BFLOAT16 attributes; graph_properties.h:288 exposes get_data_type(). Benchmarks that enable autotune: opennn_resnet50_maxbatch_trial.cpp:89,…

#### dataset-a-8 — Per-batch training scaling runs a per-element switch through scale_value in a strided loop

`opennn/dataset/tabular_dataset.cpp:398-421` · medium · overhead · lines +6 · effort M · risk low · confirmed

fill_features -> apply_training_scaling is on the per-batch path (Batch::fill -> Dataset::fill_batch_host -> fill_inputs/fill_targets). For every element it calls TrainingTransform::apply -> scale_value, a 6-way switch with branchy guards (`desc.maximum - desc.minimum < EPSILON`, `standard_deviation > EPSILON`) evaluated batch_size x features times, and the loop is feature-outer / sample-inner with stride features_number, i.e. column-strided access over a row-major buffer that fill_tensor_data just wrote row-contiguously. scaling.h already provides `scaling_affine(scaler, descriptives, min, max)` with guards documented to match scale_value exactly, so every scaler except Logarithm reduces…

**Fix:** When prepare_training_scaling installs a TrainingTransform, precompute `pair<float,float> affine = scaling_affine(...)` (already in scaling.h) and a `bool is_log`. In apply_training_scaling build a small per-call `vector<pair<float,float>>` for the requested feature_indices (or cache it keyed on the indices vector), then loop sample-outer / feature-inner: `v = v * a + b` (Logarithm keeps the scale_value path). Verify with the tabular benchmarks; results are bit-identical for the affine scalers.

*Verifier:* fill_features (382-392) calls apply_training_scaling per batch; 398-421 loop feature-outer/sample-inner with stride features_number and call TrainingTransform::apply (tabular_dataset.h:235-248) which is scale_value's 6-way switch per element (scaling.h:69-91). scaling_affine exists at scaling.h:94-131 with guards documented to match scale_value. enable_device_residency (1230-1240) does the same…

#### selection-testing-5 — k-fold finalisation re-trains all k folds just to recover an epoch count the selection loop already had

`opennn/model_selection/cross_validation.cpp:138-160` · medium · overhead · lines +6 · effort M · risk medium · partial

finalize_selected_model (selection_utilities.cpp:163-181) calls refit_final_model_on_development when folds_number > 1, and that function starts by running evaluate_folds on a fresh partition: k full trainings whose only use is `.epochs`. The selection loop has already evaluated exactly this candidate (same inputs/neurons, same seed-0 partition) via evaluate_candidate -> evaluate_folds, which returns a FoldEvaluation with `epochs`, but CandidateEvaluation (selection_utilities.h:22-26) drops that field. Cost: with folds_number = 5 the final model costs 6 trainings instead of 1, i.e. for GrowingInputs with V accepted inputs the total goes from 5V+1 to 5V+6, and for GeneticAlgorithm it is the…

**Fix:** Add `Index epochs = 0;` to CandidateEvaluation and copy fold_evaluation.epochs into it in evaluate_candidate; add `Index optimal_epochs` to InputsSelectionResult/NeuronsSelectionResult (set where optimum_validation_error is updated); change finalize_selected_model to take `Index refit_epochs` and refit_final_model_on_development to take the epoch count directly instead of folds_number/folds_seed (drop the evaluate_folds + build_fold_partition call). Check the folds test in…

*Verifier:* Substance confirmed: selection_utilities.cpp:163-181 finalize_selected_model calls refit_final_model_on_development(training_strategy, folds_number) when folds_number > 1 (note: two args, not three as quoted; folds_seed is defaulted in cross_validation.h:32), and cross_validation.cpp:138-160 starts with `evaluate_folds(training_strategy, build_fold_partition(...)).epochs` i.e. k extra trainings…

#### r2-arena-planner-and-propagation-structs-4 — Joint-plan handshake fails silently in both directions (inference-mode lifetimes, arena without offsets)

`opennn/neural_network/back_propagation.cpp:106-115` · medium · API · lines +6 · effort S · risk low · unverified

Two misuse paths produce a working-but-unshared layout with no error. (1) BackPropagation::set takes `external_arena` and `arena_offsets` as independent optionals; if offsets are empty (for example because the forward was built in Inference mode, or a caller forgot to pass forward.co_planned_offsets) it silently calls setup_arena and allocates a private arena, so the "deltas live inside the forward arena" contract and the steady-state allocation guard are quietly bypassed. TrainingContext checks the forward half (`forward.arena.owns_memory()` throw) but not the backward half. (2) ForwardPropagation::set accepts `co_planned_lifetimes` in Inference mode and silently drops them: line 149…

**Fix:** In BackPropagation::set add `throw_if((external_arena != nullptr) != !arena_offsets.empty(), "BackPropagation::set: an external arena and its offsets must be given together.")`. In ForwardPropagation::set add, next to the existing policy checks, `throw_if(new_mode != ForwardPropagationMode::Training && !co_planned_lifetimes.empty(), "...co-planned delta lifetimes are training-only.")`. Two throw_if statements, no behaviour change for correct callers.

#### nn-core-6 — ForwardPropagation::set rewrites a process-global conv workspace cap that keys the cuDNN plan cache

`opennn/neural_network/forward_propagation.cpp:832-836` · medium · design · lines +6 · effort S · risk medium · confirmed

Every ForwardPropagation::set ends with device::set_conv_workspace_auto_limit_bytes(bind_slots(...)), publishing *this propagation's* largest layer byte count as the process-wide auto cap. That cap is read at plan-build time (cudnn_frontend_utilities.h:570, deselect_workspace_greater_than) and is hashed into the on-disk plan-cache key (:413). The value in force is therefore whichever propagation was set last, not the one that is executing: optimizer.cpp:834-845 builds the training context and then the validation ForwardPropagation, so the training convolutions are planned under the (smaller) validation batch's cap; each GPU calculate_outputs() constructs a throwaway inference propagation…

**Fix:** Keep the value with the propagation: add `Index conv_workspace_limit_bytes = 0;` to ForwardPropagation, assign it from bind_slots in set(), and publish it at the start of NeuralNetwork::forward_propagate (5-arg) via device::set_conv_workspace_auto_limit_bytes(forward_propagation.conv_workspace_limit_bytes) so the cap in force always belongs to the propagation being run (a relaxed atomic store per forward - negligible). Benchmark ResNet-50 training before/after since plan keys for mixed-size…

*Verifier:* Read forward_propagation.cpp 832-836, device_backend.cpp 274-291 (auto cap is a process-global atomic, overwritten by each set(), clamped to 256 MiB, used only in auto mode -1), opennn/core/cuda/cudnn_frontend_utilities.h 412-413 (hashed into plan-cache key) and 570-582 (deselect_workspace_greater_than at plan build), optimizer.cpp 834-845 (training context then validation set()),…

#### operators-a-7 — BatchNormalizationOperator::apply_delta_gpu is a 218-line function with four obvious seams

`opennn/neural_network/operators/batch_norm_operator.cpp:626-843` · medium · design · lines +6 · effort M · risk low · partial

The GPU backward interleaves: (1) selecting and building the cuDNN backward graph with a rung-driven attempt list and a once-per-shape diagnostic (lines 652-734, inside the run_frontend lambda), (2) the own-kernel path (738-768), (3) the graph execution with optional FP32 staging (770-819), and (4) the legacy cudnnBatchNormalizationBackward fallback (822-843). All four are reached through a nested lambda that captures eleven locals, so the control flow (`return` inside the lambda vs. fall-through to the legacy path) is hard to follow and any change to the attempt policy risks the execution half. It is the only function in the operators scope above 150 lines.

**Fix:** Split verbatim along the seams: a file-local `choose_backward(Entry&, const BatchNormalizationOperator&, const TensorView& input, bool has_residual, bool fork_capable) -> const BackwardChoice&` for lines 652-734, and two private members `backward_own_kernel(...)` and `backward_graph(Entry&, ...)` for 738-768 and 770-819; apply_delta_gpu keeps the dispatch and the legacy fallback. No behaviour change; the same relocation shape the prior audit used for ForwardPropagation::set.

*Verifier:* apply_delta_gpu spans batch_norm_operator.cpp:625-842 (217 lines, the stated 626-843 is off by one) and the seams described (attempt selection inside the run_frontend lambda, own-kernel path, graph execution, legacy cudnnBatchNormalizationBackward fallback at 822-843) are as claimed. The statement 'the only function in the operators scope above 150 lines' is false:…

#### operators-b-6 — cuDNN RNN re-derives weight-region pointers on every forward and backward: 2 descriptor create/destroy + N cudnnGetRNNWeightParams per call

`opennn/neural_network/operators/cudnn_rnn.cpp:230-277` · medium · overhead · lines +6 · effort S · risk low · confirmed

cudnn_copy_weight_regions_ runs once per forward (via cudnn_pack_weights_) and once per backward (via cudnn_unpack_gradients_). Each call creates and destroys two cudnnTensorDescriptor_t and issues num_linear_layers cudnnGetRNNWeightParams calls (8 for LSTM, 2 for the plain RNN), whose only use is the returned pointers into the packed weight space. Those pointers are base + offset with offsets fixed by rnn_desc (they change only on topology_changed), so the whole block is recomputed per batch for a constant answer. On small RNNs (hidden 64-256) this host-side cuDNN chatter is a measurable share of the step.

**Fix:** Add to BackendState a small vector of {matrix_offset, vector_offset, rows} per linear layer, filled once when topology_changed (query cudnnGetRNNWeightParams against the current packed buffer and store ptr - base as byte offsets). cudnn_copy_weight_regions_ then builds the RnnCopySpec array from base + offset with no cuDNN calls. Keep the one-shot query in its own helper so the persistent-algo retry (which resets rnn_desc) refills it.

*Verifier:* cudnn_rnn.cpp:237-283: cudnn_copy_weight_regions_ creates/destroys two cudnnTensorDescriptor_t and issues num_linear_layers cudnnGetRNNWeightParams calls, using only the returned pointers. Called from cudnn_pack_weights_ (297, every forward: lstm 955 / recurrent 368) and cudnn_unpack_gradients_ (308, every backward: lstm 978 / recurrent 376). Offsets depend only on rnn_desc, rebuilt under…

#### response-opt-8 — perform_response_optimization is a 283-line function with four nested lambdas and two save/restore layers

`opennn/response_optimization/response_optimization.cpp:2323-2607` · medium · design · lines +10 · effort M · risk low · confirmed

The function contains: state snapshot/restore via ScopeExit (2325-2337), a local struct BranchAxis and axis collection (2343-2415), radix enumeration of all branch value combinations (2420-2443, allocating branches_number x axes vectors even in Budgeted mode), a second constraint snapshot for per-branch restore (2445-2447, 2485-2487), solve_branch/objective_of/merge_into_incumbent lambdas (2456-2516), the Exhaustive loop, and a successive-halving loop with reward ranking (2533-2600). Each block is independently testable but none can be tested today except through a full optimization run; the closures capture 10+ locals by reference, so the data flow is invisible from the signature.

**Fix:** Split along the existing seams into private members: `vector<BranchAxis> collect_branch_axes() const`, `static vector<vector<float>> enumerate_branch_values(const vector<BranchAxis>&)`, `MatrixR solve_branch(const vector<BranchAxis>&, const vector<float>&, Index cap)`, and `MatrixR successive_halving(...)`; keep one ConstraintSet snapshot type used by both restore layers. Move BranchAxis next to ConstraintSet in the header (class-scope types first, per AGENTS.md).

*Verifier:* perform_response_optimization spans 2323-2607 (285 lines): ScopeExit restore (2325-2337), local struct BranchAxis (2343-2350), axis collection (2352-2415), radix enumeration allocating branches_number x axes (2427-2443) before the Budgeted/Exhaustive split, second snapshot (2445-2447) restored in solve_branch (2485-2487), lambdas solve_branch/objective_of/merge_into_incumbent (2456-2516),…

#### xcut-boilerplate-8 — ResponseOptimization writes 36 unconditional cout messages and has no display flag

`opennn/response_optimization/response_optimization.cpp:1644-1694` · medium · API · lines +10 · effort S · risk low · partial

Pattern (j). Of the 142 cout sites in the library, Optimizer (23), the model-selection classes (26) and the datasets (~20) are gated on a `bool display` member (37 `if (display) cout` guards), but ResponseOptimization has 36 cout sites (response_optimization.cpp:793, 1294, 1566, 1644-1694, 1908-2044, 2141-2210, 2312, 2424, 2481, 2524-2603) and the word `display` does not occur anywhere in response_optimization.h or .cpp. Every perform_response_optimization / multi-objective / AllowedSet run prints per-iteration progress, warnings and budget messages to stdout with no way for a library consumer (Neural Designer, tests) to silence them. This is also the only class whose messages use ad-hoc…

**Fix:** Add `bool display = true;` + `set_display(bool)` to ResponseOptimization (matching Optimizer/InputsSelection/Dataset) and route all 36 sites through one private `void log(string_view) const { if (display) cout << message << '\n'; }` (or an `if (display)` guard where formatting is inline). Read/write it in to_JSON/from_JSON like the other classes. Consider the same helper for the 5 Optimizer stopping messages at optimizer.cpp:1120-1168, which duplicate the names in…

*Verifier:* grep counts exactly 36 cout sites in response_optimization.cpp (793, 1294, 1566, 1644, 1650, ...) and the word 'display' does not appear in response_optimization.h or .cpp; library-wide there are 141 cout lines with 23 `if (display) cout` guards. So the unconditional-output claim holds. Correction: ResponseOptimization has no to_JSON/from_JSON at all (grep for…

#### core-types-4 — CPU activation forward/backward run single-threaded while sibling ops in the file parallelise at 65536

`opennn/core/tensor_operations.cpp:1174-1246` · medium · overhead · lines +12 · effort S · risk low · confirmed

activation_forward_cpu and activation_backward_cpu evaluate one Eigen ArrayWrapper expression over the whole tensor on the calling thread. MKL is off by default (opennn/CMakeLists.txt:12 `option(OpenNN_ENABLE_MKL ... OFF)`), so try_activation_forward is the `return false` stub and even Tanh takes this path. In the same file softmax_cpu (line 1150), multiply_cpu (line 1050) and fill_tensor_data (tensor_types.cpp:143) all use `#pragma omp parallel for schedule(static) if(size >= 65536)`. A CPU training step on a 1024-wide hidden layer at batch 1024 therefore does 1M transcendental evaluations per dense layer per forward on one core, and every activation_backward (including the ReLU select,…

**Fix:** Wrap both switch bodies in a segment loop: split the flat vector into contiguous chunks (e.g. 16384 elements, or the rows of as_flat_matrix()), map each chunk as a VectorMap segment and apply the same Eigen expression per chunk under `#pragma omp parallel for schedule(static) if(size >= 65536)`. Elementwise, so results are bitwise identical; the existing activation tests cover it. Measure on the HIGGS CPU benchmark before merging, as the prior audit asks for hot-path changes.

*Verifier:* activation_forward_cpu (tensor_operations.cpp:1174-1209) and activation_backward_cpu (:1211-1246) evaluate one Eigen array expression, no omp pragma; try_activation_forward is the `return false` stub at :815 when MKL is off (opennn/CMakeLists.txt:12 OFF by default). softmax_cpu at :1145-1159 and multiply_cpu at :1048-1051 use `#pragma omp parallel for schedule(static) if(size >= 65536)`. ReLU…

#### dataset-a-10 — read_csv is 712 lines with six self-contained seams

`opennn/dataset/tabular_dataset.cpp:1618-2330` · medium · design · lines +12 · effort L · risk low · confirmed

The function mixes file reading, header validation, number-format detection, sample-id auto-detection, identifier-column exclusion, type inference, category-map construction, a 280-line OpenMP parse engine made of four nested lambdas sharing ~15 captured locals, post-parse type refinement, and binary-cache writing. The seams are member-independent and already pass their inputs explicitly: (a) sample-id detection 1681-1733 (inputs: lines, separator, has_quotes, missing label, number_format -> bool); (b) identifier-column exclusion 1777-1798; (c) expanded-size guard 1833-1870 (variables, samples -> throws); (d) category_maps build 1907-1931 (variables -> vector<unordered_map>); (e) the parse…

**Fix:** Extract (a)-(f) as file-local free functions / one struct in the anonymous namespace of tabular_dataset.cpp, one commit per seam, keeping read_csv as the ~120-line orchestrator. No header changes; the CMake glob needs nothing. Do (e) last and diff the missing-value counters on a fixture with NAs before/after.

*Verifier:* read_csv spans tabular_dataset.cpp:1618-2330 (712 lines). Verified the seams: sample-id detection 1681-1733 uses only lines/separator/has_quotes/missing label/number_format; identifier exclusion 1777-1798; expanded-size guard 1833-1870; category_maps 1907-1931; parse engine 1933-2221 is four nested lambdas (parse_row, refine_numeric, count_missing, parse_rows) over captured locals; post-parse…

#### nn-core-8 — ForwardPropagation::set is 724 lines with four member-independent seams

`opennn/neural_network/forward_propagation.cpp:124-847` · medium · design · lines +12 · effort M · risk low · confirmed

The prior audit proposed extracting 'phase 2' as plan_activation_offsets and deferred it. Reading the current function, the seams are sharper than that and none of them touches members except through the values it returns: (a) 211-290 shape policy (execution_start_layer, sequence/final-output capacity rewriting of forward_specs) depends only on forward_specs, layers, policy; (b) 614-719 the inference lifetime plan reads forward_specs, source_layers, retained_output_layers and get_last_trainable_layer_index() - a static function returning the pooled lifetimes; (c) 476-613 the training plan (co-planned deltas, recompute overlay, transient block); (d) 747-830 is 84 lines of pure…

**Fix:** Introduce a file-local `struct ActivationPlan { vector<vector<Index>> slot_offsets, transient_slot_offsets; Index pool_bytes, transient_bytes, lower_bound, fragmentation; vector<Index> co_planned_offsets; }` and two static functions `plan_training_activations(...)` / `plan_inference_activations(...)`, plus `apply_shape_policy(forward_specs, policy, ...)` and `record_activation_plan(const ActivationPlan&, ...)`. Move bodies verbatim one commit each; suites (ForwardPropagationMemoryTest,…

*Verifier:* Read forward_propagation.cpp 124-847 boundaries (set spans 124-847 = 724 lines), lambdas at 404 (place_transient_slots), 437 (collect_pooled_slots), 461 (apply_pool_plan), is_training branch 476 / else 614, memory_debug::record block 747-830, and 832-846. docs/ENGINEERING_AUDIT.md line 186-188 lists this as open item 4 (not DONE), so already_in_prior_audit=true is right and this refines it rather…

#### nn-builders-chat-7 — YoloNetwork silently ignores head_style/reg_max/use_sppf/model_size for unsupported combinations

`opennn/neural_network/standard_networks.cpp:478-492` · medium · API · lines +12 · effort S · risk low · confirmed

The up-front validation only covers FPN and PANet. HeadStyle::FPNv8 with Backbone::Vgg, DarknetTiny or DarknetTinyV3 is not rejected: none of those branches handles FPNv8, so control falls through to the anchor-based single head at 1169-1187 and the caller's reg_max is dropped without a word; Darknet53/CSPDarknet53 + FPNv8 ignores use_sppf (1071 feeds c5_index directly, while FPN/PANet honour it at 974); model_size only affects CSPDarknet53v11 (706-720); grid_size is used only in one throw_if (476). CSPDarknet53v11 with any non-FPNv8 head builds the entire ~40-layer backbone and only then throws (885). With 11 positional parameters (7 of them defaulted enums/bools/Index) a caller cannot see…

**Fix:** Add one validation table at the top of the constructor: a constexpr set of supported (Backbone, HeadStyle) pairs plus `throw_if(reg_max > 1 && head_style != HeadStyle::FPNv8, ...)`, `throw_if(use_sppf && !(darknet53-family && (FPN||PANet)), ...)`, `throw_if(model_size != ModelSize::l && backbone != CSPDarknet53v11, ...)`, and move the CSPDarknet53v11 head check there. Longer term (with nn-builders-chat-6) replace the 7 trailing positional parameters with a `YoloNetwork::Config` aggregate;…

*Verifier:* Validation at 478-492 covers only FPN and PANet. Vgg branch (599-621) and DarknetTinyV3 branch (622-694) have no FPNv8 case and no return, so FPNv8 falls through to the single anchor head at 1169-1187 where reg_max is never read (reg_max used only at 848/861/1051/1068). use_sppf is read only at 974 (FPN/PANet), FPNv8 Darknet53 feeds c5_index at 1071. model_size only at 707-717, grid_size only at…

#### training-optimizers-2 — Validation remainder batch allocates a Batch and a ForwardPropagation arena every validation epoch; the allocation guard is disabled to allow it

`opennn/training_strategy/optimizer.cpp:2126-2162` · medium · overhead · lines +12 · effort M · risk medium · confirmed

evaluate_tail constructs a fresh Batch (host + device buffers) and a fresh ForwardPropagation (arena.resize_bytes -> cudaMalloc, forward_propagation.cpp:737) on every validation epoch, while the training tail was deliberately given a persistent TrainingSession::TailContext to avoid exactly this. Because the allocation would trip CudaAllocationGrowthGuard, train() arms the steady-state guard only when there is no validation tail (lines 892-894, 906-907). With the usual 60/20/20 split and a power-of-two batch size the validation count is almost never a multiple of the batch size, so in practice the guard is off for most real runs and any other accidental steady-state allocation goes…

**Fix:** Add a validation tail slot to TrainingSession (unique_ptr<Batch> batch; unique_ptr<ForwardPropagation> forward; Index size), lazily built in evaluate_tail the same way train_tail builds TrainingSession::tail (rebuild only when size differs). In warmup_device_training include validation_batches->back() in validation_warmup_batch when its size differs from front(), so the slot exists before the guard arms. Then drop has_validation_tail and arm the guard on needs_cuda_warmup alone. The validation…

*Verifier:* optimizer.cpp:2126-2150: evaluate_tail builds `Batch batch(tail_size, ...)` and `ForwardPropagation tail_forward_propagation(tail_size, ...)` with no external storage, so forward_propagation.cpp:735-738 hits arena.resize_bytes (+ setZero at 742). optimizer.cpp:892-894 defines has_validation_tail and 906-907 arms CudaAllocationGrowthGuard(needs_cuda_warmup && !has_validation_tail). The training…

#### r2-batch-pipeline-and-device-gather-3 — Validation tail builds a Batch (cudaMalloc + cudaMallocHost + event) and a ForwardPropagation arena every epoch, and disables the allocation guard for the whole run

`opennn/training_strategy/optimizer.cpp:2134-2152` · medium · overhead · lines +12 · effort M · risk medium · confirmed

evaluate_epoch's evaluate_tail lambda constructs a local `Batch batch(tail_size, ...)` and a `ForwardPropagation tail_forward_propagation(...)` on every validation epoch. On GPU that is two or three cudaMalloc, two or three cudaMallocHost (pinned allocation is device-synchronizing and typically 0.1-1 ms each), a cudaEventCreate, and the forward arena allocation (memory notes: arena construction is pure cudaMalloc and costs ~2.3x an inference), followed by the matching cudaFree/cudaFreeHost (both synchronize the device) at scope exit. The training tail already solves this with TrainingSession::TailContext (built once on the warm-up pass, re-linked per epoch). Because the validation tail…

**Fix:** Add a `TailContext validation_tail` (Batch + unique_ptr<ForwardPropagation> + size) to TrainingSession next to `tail`, build it on the warm-up pass (warmup_device_training already receives validation_batches), reuse it in evaluate_tail exactly as train_tail reuses `training_session.tail`, and arm the guard with `needs_cuda_warmup` unconditionally. Pass TrainingSession& to evaluate_epoch (it already is a parameter).

*Verifier:* optimizer.cpp 2126-2160: evaluate_tail constructs a local Batch (2134) and a ForwardPropagation (2149-2154) every validation epoch with a tail; 893-907: has_validation_tail disables CudaAllocationGrowthGuard for the whole run. The training counterpart (1835-1850) already caches through TrainingSession::TailContext (optimizer.h 140-145) built on the warm-up pass. warmup_device_training receives…

#### xcut-api-1 — GPU 'buffer-reusing' calculate_outputs still allocates a ForwardPropagation arena per call

`opennn/neural_network/neural_network.h:219-226` · medium · overhead · lines +15 · effort S · risk low · partial

The header tells users to prefer calculate_outputs(inputs, MatrixR& outputs) when calling repeatedly, but on the GPU path the implementation (neural_network.cpp:1264-1269) constructs a fresh ForwardPropagation(batch_size, this, Inference) on every call, so only the host MatrixR is reused while the device arena (the expensive part: pure cudaMalloc, measured at ~2.3x the inference itself) is re-planned and re-allocated each time. The by-value overload (cpp:1163-1176) does the same. The reusable host-output path already exists as the private calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&, MatrixR&) (neural_network.h:344). TestingAnalysis::get_targets_and_outputs pays…

**Fix:** Promote the private calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&, MatrixR&) to a public overload named calculate_outputs(const vector<TensorView>&, ForwardPropagation&, MatrixR& outputs) (its body already handles FP32/BF16 and the CPU case through copy_device_to_host_float). Keep the convenience overloads as thin wrappers that build one ForwardPropagation and call it. In TestingAnalysis::get_targets_and_outputs build one ForwardPropagation(current_batch_size,…

*Verifier:* Substance confirmed: neural_network.cpp:1263-1269 (MatrixR& overload) and 1170-1176 (by-value) both build a fresh ForwardPropagation per call on GPU; calculate_outputs_device(const vector<TensorView>&, ForwardPropagation&, MatrixR&) at cpp:2799-2816 is private (h:344) and does the reuse; testing_analysis.cpp:151-159 builds a host Batch (host_config, Device::CPU) and calls…

#### r2-batch-pipeline-and-device-gather-5 — Default (non-graph) GPU training path has zero H2D/compute overlap: one fixed device slot, copy waits for the previous step

`opennn/training_strategy/optimizer.cpp:1723-1774` · medium · overhead · lines +15 · effort M · risk medium · partial

When CUDA graphs are off (the default, `use_cuda_graph = false`) train_epoch runs run_epoch_loop with `fixed_device_batch = training_session.fixed_batch()` (pipelines[0].slots[0], the only device batch; pool batches are prefetch_only with 0 device bytes). fetch_and_issue(i+1) is called after step i is enqueued and first makes the transfer stream wait on the fixed batch's event recorded on compute after step i (line 1731-1732), then issues the H2D; step i+1 then waits on that copy. So copy(i+1) strictly follows compute(i) and compute(i+1) strictly follows copy(i+1): the H2D is fully exposed on every step. For an MLP on a 1000x784 fp32 batch the ~3 MB copy (~250 us over PCIe) is comparable to…

**Fix:** Allocate pipelines[1].slots[0] for every GPU training session (not only when graphs are on) and let run_epoch_loop alternate between two fixed slots: issue the H2D of batch i+1 into slot (i+1)%2 before step i (each slot has its own h2d_done_event recorded on compute after its step, so the transfer wait targets the slot being overwritten). Cost: one extra batch of VRAM, which the memory-budget estimate (batch_copies) should include. Measure with OPENNN_PROFILE on the HIGGS and MNIST-sized…

*Verifier:* Claim verified at optimizer.cpp 1718-1778: with use_fixed_device_batch the H2D for batch i+1 is issued only after step i (1777), after the transfer stream waits on the event recorded on compute at 1758, and step i+1 waits on that copy (1751): no copy/compute overlap. Corrections: (a) the fix 'allocate pipelines[1].slots[0] for every GPU session' would flip has_graph_batches() (optimizer.h 123-126…

#### response-opt-1 — FD output-constraint repair runs a full network forward per row, per pass, per constraint

`opennn/response_optimization/response_constraints.cpp:1366-1391` · medium · overhead · lines +25 · effort M · risk medium · partial

When the analytic NetworkDifferential is unavailable (any GPU/BF16-resident network, batch-norm Dense, no-bias Dense, constant input feature — see response-opt-13), repair_output_constraints falls back to the finite-difference surrogate. gauss_newton_repair_row is row-serial: for every row (default 2000) and every pass (up to 64) it calls `forward(point)` (a 1-row NeuralNetwork::calculate_outputs, which constructs a fresh ForwardPropagation — neural_network.cpp:1193-1198) and then, for EVERY active constraint, `constraint_gradient` -> `vjp(point, cotangent)` which rebuilds and evaluates the identical 2n-row perturbed batch (lines 1937-1946) — the perturbed matrix does not depend on the…

**Fix:** Step 1 (contained): replace SurrogateVjp with a SurrogateJacobian `MatrixR(const VectorR& x)` evaluated once per (row, pass) in gauss_newton_repair_row; constraint_gradient then does `gradient + J.transpose() * cotangent`. NetworkDifferential can expose the same via K vjp calls on the tape, so both surrogates fit. Step 2: make repair_rows batch across rows — build one (R*(1+2n))-row matrix per pass, call batch_forward once, and do the per-row Gauss-Newton solve on the returned slices. Also add…

*Verifier:* Mechanics confirmed: repair_rows (response_constraints.cpp:1394-1417) is row-serial, gauss_newton_repair_row (1353-1391) calls forward(point) once per pass and constraint_gradient -> vjp once per active constraint (1379); the FD vjp (1937-1958) rebuilds the 2n-row perturbed matrix on every call although it does not depend on the cotangent; every batch_forward goes through…

#### xcut-build-tests-16 — 24 library .cpp files have no mirrored test; 9 of them are never even included from tests/

`tests/neural_network/operators:1-1` · medium · build/test · lines +400 · effort L · risk low · confirmed

Mapping every opennn/**/*.cpp to tests/<same path>/<name>_test.cpp (the AGENTS.md rule) and then grepping tests/ for the header: never included at all -> operators/attention_operator, c2psa_operator, combination_operator, convolution_operator, cudnn_rnn, embedding_lookup_operator, layer_normalization_operator, multihead_projection_operator, and training_strategy/training_context. No mirrored test but reached indirectly -> core/json, core/variable, model_selection/cross_validation, inputs_selection, selection_utilities, neural_network/back_propagation, forward_propagation, standard_networks, layers/detection_v8_layer, pooling_layer_3d, training_strategy/optimizer, error_functions, loss…

**Fix:** Add one test file per operator following tests/neural_network/operators/batch_norm_operator_test.cpp as the template (forward hand-computed values + numerical gradient via numerical_derivatives), starting with combination, layer_normalization, embedding_lookup and multihead_projection (CPU paths, no CUDA needed), plus a json round-trip test for core/json.cpp. GLOB picks them up with no CMake edits.

*Verifier:* Re-ran the mapping: attention_operator, c2psa_operator, combination_operator, convolution_operator, cudnn_rnn, embedding_lookup_operator, layer_normalization_operator, multihead_projection_operator, training_context -> 0 includes from tests/ and no *_test.cpp; core/json included by 5 test files but no json_test.cpp; core/variable 4 includes, no test. tests/neural_network/operators/ holds only…

#### response-opt-5 — GreaterThan/LessThan are strict in the linear filter path but inclusive-with-tolerance elsewhere

`opennn/response_optimization/response_constraints.cpp:1145-1172` · low · bug · lines -20 · effort S · risk low · confirmed

build_linear_constraint_set (used by filter_feasible_points only when ALL formula constraints are affine) sets lower = low - c + EPSILON for GreaterThan and gives no tolerance; constraint_is_satisfied (used when any constraint is nonlinear or a callback) treats GreaterThan exactly like GreaterEqualTo with +/- bound_tolerance; Domain::bound, interval_from_comparison, constraint_residual and promote_single_variable_constraints also treat them as inclusive. Scenario: constraint 'x1 + y > 3' with a point at exactly 3.0: feasible if the user also has an unrelated sqrt(...) constraint (nonlinear path), infeasible otherwise. The 28-line switch is also a hand-written copy of…

**Fix:** Replace the switch with: `float lo, up; if (interval_from_comparison(op, low_bound, up_bound, lo, up)) { lower(i) = lo - c - bound_tolerance(lo); upper(i) = up - c + bound_tolerance(up); }` (bound_tolerance(inf) is inf, so open sides stay open). Both filter paths then agree by construction.

*Verifier:* build_linear_constraint_set (1145-1172) gives GreaterThan/LessThan a strict EPSILON offset and no bound_tolerance, while constraint_is_satisfied (1189-1193), interval_from_comparison (h:85-101), constraint_residual (1228-1241) and Domain::bound (665-680) treat them as inclusive with tolerance. The two filter paths are selected by all_formula_constraints_are_linear (1205 vs 1230 in…

#### dataset-b-9 — calculate_cross_correlations lacks the lag guards its twin calculate_autocorrelations has

`opennn/dataset/time_series_dataset.cpp:510-524` · low · bug · lines -12 · effort S · risk low · confirmed

calculate_autocorrelations (478-481) guards the effective lag reduction with `lags_number > 2` / `lags_number > 1`; the copy in calculate_cross_correlations does not. Scenario: samples_number == lags_number == 1 (or 2) -> effective_lags_number = -1 (or 0) -> `Tensor3 cross_correlations(n, n, -1)` is a negative-dimension Eigen tensor (assertion in debug, allocation of a wrapped size in release). The two functions also duplicate the numeric-variable index gathering (466-476 vs 512-520).

**Fix:** Extract a file-local `pair<vector<Index>, Index> numeric_variables_and_effective_lags(const TimeSeriesDataset&, Index lags)` (or two small helpers) used by both functions, with the guarded formula; delete the two inline copies.

*Verifier:* calculate_autocorrelations (478-481) guards with lags_number > 2 / > 1; calculate_cross_correlations (510-512) has the unguarded ternary, so samples_number == lags_number == 1 gives effective_lags_number = -1 and Tensor3(n, n, -1) at 522-524 (the only check, 504-506, is lags > samples). The numeric-variable gathering at 466-472 and 514-520 is identical. Shared helper with the guarded formula is…

#### nn-core-7 — from_JSON lets layers through with no sources; invariants validated in add_layer are not re-established

`opennn/neural_network/neural_network.cpp:1683-1703` · low · UB · lines -11 · effort S · risk low · partial

add_layer validates source indices and arity for every layer (550-551). from_JSON validates only SourceLayer entries that are present and have non-empty Text: `if (text.empty()) continue;` and a missing entry both leave source_layers[i] empty, and validate_source_arity is never run for that layer. A layer with get_sources_number()==1 or 2 and an empty source list then reaches the operators, which index `forward_propagation.inputs[layer][1]` (batch_norm_operator.cpp:196, layer_normalization_operator.cpp:410, convolutional_layer.cpp:480) or compute `inputs[layer].size() - 1` (attention_operator.cpp:508, size_t underflow) on an empty vector - UB instead of a clean parse error for a…

**Fix:** After the SourceLayers block, validate every layer once: `for (Index i = 0; i < ssize(layers); ++i) { validate_source_indices(source_layers[i], i, ssize(layers)); validate_source_arity(*layers[i], source_layers[i], i); }` (resolving an empty list to the add_layer default `{i - 1}` if legacy files rely on it), drop the `continue` and the per-entry validation. Delete the unfalsifiable size checks at forward_propagation.cpp:199-209. Extend invalidate_trainable_layer_cache() to also reset…

*Verifier:* Read neural_network.cpp 536-560 (add_layer validates indices + arity), 574-607 (compile() validates neither), 805-814, 1655-1660, 1681-1701 (SourceLayer entries with empty Text or absent are skipped; source_layers[i] stays empty; no arity check), 1557-1565 (to_JSON always writes one entry per layer), header 182 (invalidate_trainable_layer_cache exists), forward_propagation.cpp 199-209,…

#### core-utils-6 — JSON \u escapes do not combine surrogate pairs and duplicate append_utf8

`opennn/core/json.cpp:357-389` · low · bug · lines -10 · effort S · risk low · confirmed

parse_string decodes each \uXXXX independently and emits a 3-byte sequence for any code >= 0x800, including surrogates 0xD800-0xDFFF. A pair such as 😀 (how Python's json.dumps with the default ensure_ascii=True writes every non-BMP character, e.g. emoji tokens in a vocab.json consumed by BytePairTokenizer::load, tokenizer_operator.cpp:687) becomes six bytes of invalid UTF-8 (CESU-8) instead of the 4-byte character, so the token never matches the bytes the tokenizer sees. The inline encoder also duplicates string_utilities.cpp:243-267 append_utf8, which already handles the 4-byte case.

**Fix:** Factor the 4-hex-digit read into a small lambda; if code is a high surrogate and the next six chars are \uDC00-\uDFFF, read the low half and combine (0x10000 + ((hi-0xD800)<<10) + (lo-0xDC00)); then call append_utf8(out, code) and delete the three hand-written branches.

*Verifier:* json.cpp:357-389 decodes each \uXXXX independently; no surrogate handling; emits a 3-byte sequence for 0xD800-0xDFFF (CESU-8). append_utf8 (string_utilities.cpp:243-267) already covers the 4-byte case and json.cpp already depends on string_utilities.h (uses parse_number at :132). BytePairTokenizer::load (tokenizer_operator.cpp:684-688) parses vocab.json with Json::parse, so an ensure_ascii vocab…

#### layers-a-6 — Pooling/Convolutional apply_input_shape skip the geometry validation their set() performs

`opennn/neural_network/layers/pooling_layer.cpp:558-567` · low · bug · lines -10 · effort S · risk low · confirmed

Pooling::set runs validate_pooling_configuration (pool <= padded input, stride <= padded input, ...) and Convolutional::set runs validate_convolution_configuration (kernel <= input for Valid, channels match, ...), but both apply_input_shape overrides (pooling_layer.cpp:558-567, convolutional_layer.cpp:270-279) only check the rank and assign. NeuralNetwork::set_input_shape therefore lets an input smaller than the window through: Pooling p({4,4,3},{2,2},{2,2}); p.set_input_shape({1,1,3}) yields get_output_shape() == {0,0,3} and a forward that silently produces an empty tensor; Convolutional c({8,8,1},{5,5,1,4},"ReLU",{2,2},"Valid"); c.set_input_shape({1,1,1}) makes get_output_height() return…

**Fix:** Replace both bodies with a call to set() using the current parameters: Pooling: set(new_input_shape, {pool_height,pool_width}, {row_stride,column_stride}, {padding_height,padding_width}, pooling_method_to_string(pooling_method), label); Convolutional: set(new_input_shape, {kernel_height,kernel_width,kernel_channels,kernels_number}, ActivationOperator::to_string(activation_operator.activation_function), {row_stride,column_stride}, use_padding ? "Same" : "Valid", batch_norm.active(), label). Both…

*Verifier:* Pooling::set (pooling_layer.cpp:525-556) runs validate_pooling_configuration (367-400) while Pooling::apply_input_shape (558-567) only checks rank and reassigns three fields before update_pool_operator; Convolutional::set (convolutional_layer.cpp:225-265) runs validate_convolution_configuration but apply_input_shape (270-279) does the same rank-only assignment. Shape's constructor throws on…

#### layers-a-3 — Scaling/Unscaling/Flatten apply_input_shape reset the label (and Scaling's statistics)

`opennn/neural_network/layers/scaling_layer.cpp:245-264` · low · bug · lines -8 · effort S · risk low · partial

Scaling::apply_input_shape calls set(), which unconditionally does set_label("scaling_layer") and resets descriptives, scalers and min/max range; Unscaling::apply_input_shape calls set(dim) whose second parameter defaults to "unscaling_layer" (unscaling_layer.cpp:36-39, 24-34); Flatten::apply_input_shape calls set(), which does set_label("flatten_layer") (flatten_layer.cpp:21-28, flatten_layer.h:32). NeuralNetwork::set_input_shape (neural_network.cpp:775-800) calls set_input_shape on every single-source layer and is invoked by InputsSelection (inputs_selection.cpp:25) and GrowingNeurons. Layers are looked up by label (NeuralNetwork::get_layer(const string&), neural_network.cpp:658-678;…

**Fix:** Move the default-label assignment out of set(): Scaling(const Shape&) ctor does set_label("scaling_layer") itself; Unscaling::apply_input_shape passes label (set(dim, label)); Flatten drops set() entirely (ctor: set_input_shape(new_input_shape); set_label("flatten_layer"); apply_input_shape uses the base default), also removing the redundant check_rank. For Scaling, keep the statistics reset only when the feature count actually changes (if (features != ssize(descriptives)) {...}) so a…

*Verifier:* Code facts confirmed: Scaling::set (scaling_layer.cpp:245-258) unconditionally set_label("scaling_layer"), reassigns descriptives/scalers/min_range/max_range, and Scaling::apply_input_shape (260-263) calls it; Unscaling::apply_input_shape (unscaling_layer.cpp:36-39) calls set(dim) whose label defaults to "unscaling_layer" (unscaling_layer.h:22); Flatten::apply_input_shape (flatten_layer.h:32)…

#### training-loss-14 — CPU cross-entropy masks NaN/inf to 10.0 and clamps outputs; GPU and the CPU gradient do neither

`opennn/training_strategy/error_functions.cpp:275-305` · low · bug · lines -4 · effort S · risk low · confirmed

binary_cross_entropy on CPU clamps outputs to [EPS, 1-EPS] and then replaces a NaN/inf result with 10.0; categorical_cross_entropy does the same replacement. The GPU kernels (kernel_losses.cu:39-51, 82-89) use log(out + eps) with no mask, and the CPU gradient (lines 343-345) uses the +eps form, not the clamp, so error and gradient are evaluated on different functions on CPU. A diverged network (NaN outputs) reports error 10.0 on CPU — which the optimizer treats as a valid batch and keeps stepping — but NaN on GPU, where optimizer.cpp:2051 skips the batch. Same inputs, different early-stopping behaviour by device.

**Fix:** Delete the two `if (isnan(error) || isinf(error)) error = 10.0f;` lines and write the CPU BCE with the same `log(out + EPSILON)` / `log(1 - out + EPSILON)` form as the gradient and the kernel, so CPU and GPU agree and NaN propagates to the optimizer's existing NaN handling. CrossEntropyErrorTest:104 only asserts !isnan on finite inputs.

*Verifier:* error_functions.cpp:277-283 CPU BCE clamps to [EPS,1-EPS] then `if (isnan(error) || isinf(error)) error = 10.0f;`; 300-305 categorical does the same mask. kernel_losses.cu:42-48 and 85-86 use log(out+eps)/log(1-out+eps) with no mask; CPU gradient 339-343 uses the +EPS form. optimizer.cpp:2051 `batch_ok = use_device_metrics || !isnan(error)` only skips NaN, which a CPU 10.0 never triggers.…

#### layers-b-11 — Detection uploads anchors with a raw, unchecked cudaMemcpyAsync from a pageable local vector

`opennn/neural_network/layers/detection_layer.cpp:61-76` · low · bug · lines -3 · effort S · risk low · confirmed

This is the only raw cudaMemcpyAsync in the layer code (everything else goes through device::copy_async or Buffer::migrate_to, e.g. GQA's prepare_rope_tables). Its return code is discarded: if the copy fails (bad stream after a prior error, device reset) device_anchors stays zero-filled and every decoded box gets width/height 0 with no diagnostic, and the layer never retries because byte_size() now matches. The pageable source makes the call host-synchronous so `flat` going out of scope is not itself a lifetime bug today, but the pattern relies on that CUDA detail and pulls <cuda_runtime.h> into a layer .cpp.

**Fix:** Build the flat anchors in a Buffer host(Device::CPU) (resize_bytes(anchor_bytes, Device::CPU), ranges::copy into host.as<float>()), then host.migrate_to(Device::CUDA, device::get_compute_stream()); device_anchors = std::move(host); — the pattern prepare_rope_tables uses. That routes through the checked device::copy_async, synchronizes the stream, and lets the <cuda_runtime.h> include go.

*Verifier:* detection_layer.cpp:20 includes <cuda_runtime.h> and 71-75 issue a raw cudaMemcpyAsync from the pageable local `flat` (67-69) with the return value discarded; grep over opennn/neural_network/layers/*.cpp shows this is the only raw cudaMemcpyAsync (detection_v8_layer.cpp:18 only includes the header). device_anchors is a Buffer (resize_bytes at 65), and prepare_rope_tables…

#### nn-expression-6 — replace_reserved_keywords duplicates names that start with '$'

`opennn/neural_network/model_expression.cpp:2230-2242` · low · bug · lines -3 · effort S · risk low · confirmed

The '$' branch seeds `out` with the whole input and then the loop appends every character again (the '$' itself is skipped because it is neither mapped nor alnum). "$price" becomes "$priceprice" and "$ amount" becomes "$ amount_amount", which contains a space and a '$' and is not a valid identifier in any of the target languages; the PHP exporter then prefixes another '$'. Currency-style column names are plausible dataset headers.

**Fix:** Delete the two-line '$' special case; the loop already drops '$' and produces a plain identifier (add '$' to char_replacements as "_usd_" if the information should be kept).

*Verifier:* Lines 2226-2240 read exactly as quoted: `if (input[0] == '$') out = input;` then the loop appends every mapped/alnum char of the same input, and '$' is neither in char_replacements nor alnum, so "$price" -> "$priceprice" containing '$'. Deleting the two-line special case is correct; nothing in tests relies on '$'-prefixed names (grep).

#### xcut-build-tests-11-extra-1 — LTO block references undefined LIBOMP_* variables and reports 'OpenMP not found' when IPO is unsupported

`CMakeLists.txt:34-44` · low · bug · lines -2 · effort S · risk low · verifier-added

The OpenNN_ENABLE_LTO block (default ON on non-MSVC) calls check_ipo_supported and, on success, does `include_directories(${LIBOMP_INCLUDE_DIR})` and `link_libraries(${LIBOMP_LIBRARY})`; those variables are only set inside the later APPLE block (lines 46-60), so on Linux/GCC they are empty (harmless no-ops that mislead the reader). On failure it prints `message(WARNING "OpenMP not found. Continuing without OpenMP support.")`, a copy-paste message unrelated to LTO that hides the real cause (IPO unsupported by the toolchain). Cosmetic on Linux, but a wrong diagnostic on any compiler without IPO support.

**Fix:** Reduce the block to `check_ipo_supported(RESULT OPENNN_LTO_OK)`; `if(OPENNN_LTO_OK) set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON) else() message(STATUS "IPO/LTO not supported by this toolchain; building without it.") endif()`; remove the two LIBOMP lines (the Apple block already handles libomp).

*Verifier:* found by verifier

#### core-utils-17 — FileWriter::open calls create_directories on an empty parent path; sibling in same file guards it

`opennn/core/io_utilities.cpp:446-455` · low · bug · lines -2 · effort S · risk low · confirmed

FileWriter::open runs filesystem::create_directories(tmp_path.parent_path()) unconditionally. For a bare filename the parent is empty and create_directories reports ENOENT (libstdc++ sets errc::no_such_file_or_directory for an empty path), so opening 'cache.bin.tmp' throws before any write. download_if_missing (line 34-37) in the same file guards with has_parent_path(). Reachable through TabularDataset::set_binary_cache_path("model.bin") (a relative override; tabular_dataset.cpp:1884-1888 repeats the same unguarded create_directories before calling open), which the prior audit lists as used by Neural Designer.

**Fix:** Guard with `if (tmp_path.has_parent_path())` in FileWriter::open and remove the now-redundant create_directories calls at the callers (image_dataset.cpp:518, tabular_dataset.cpp:1884) so the guard lives in one place.

*Verifier:* io_utilities.cpp:446-455 FileWriter::open calls filesystem::create_directories(tmp_path.parent_path()) unguarded; download_if_missing (:36-37) guards with has_parent_path(). Callers: tabular_dataset.cpp:1884-1888 (same unguarded create_directories on cache_file_path().parent_path(), which is cache_path_override verbatim per :143-150, so a bare 'model.bin' override yields an empty parent) and…

#### dataset-b-1 — impute_missing_values_unuse calls set_sample_role inside an OpenMP loop: data race + O(N^2) cache refresh

`opennn/dataset/time_series_dataset.cpp:265-275` · low · bug · lines -2 · effort S · risk low · partial

The loop is `#pragma omp parallel for` and calls Dataset::set_sample_role(i, "None") per sample. set_sample_role (dataset.cpp:248-254) invokes the virtual on_used_samples_changed() whenever a role flips to None; TabularDataset::on_used_samples_changed (tabular_dataset.cpp:219-225) calls refresh_cache_statistics(), which reassigns cache_feature_descriptives and cache_feature_replacement (shared members) and streams the whole cache. Scenario: a TimeSeriesDataset in StorageMode::BinaryFile with NaNs in many rows -> several threads run refresh_cache_statistics concurrently, writing the same vectors (data race, UB), and even single-threaded each flipped sample triggers a full pass over the cache…

**Fix:** Collect the indices to unuse into a vector<Index> (serial loop, or parallel with a per-thread list) and make a single call to the existing Dataset::set_sample_roles(const vector<Index>&, SampleRole::None) overload (dataset.cpp:269), which fires on_used_samples_changed once. Drop the OpenMP pragma (the scan is a trivial any_of over chars).

*Verifier:* The loop shape is as quoted (time_series_dataset.cpp:265-275) and TabularDataset::on_used_samples_changed (tabular_dataset.cpp:219-225) does refresh_cache_statistics in BinaryFile mode. But the race cannot occur on any reachable path: TabularDataset::scrub_missing_values (tabular_dataset.cpp:2557-2570) returns early when storage_mode == BinaryFile before ever reaching impute_missing_values_unuse,…

#### response-opt-7 — filter_feasible_points indexes outputs and the output Domain by variable index, not feature column

`opennn/response_optimization/response_optimization.cpp:1172-1200` · low · bug · lines -2 · effort S · risk low · confirmed

The loop uses column_index (position in get_output_variables()) as the column into `outputs`, and domain_index (+1 per variable) into output_domain frontiers, but both the output matrix and the Domain are laid out per FEATURE (Variable::get_feature_count; Domain::set uses get_feature_dimensions). Output variables can be categorical: Optimizer::set_names passes dataset->get_variables(VariableRole::Target) unchanged, and Objectives(...) in this very file walks targets with feature_dimensions_by_role. Scenario: output variables [class (Categorical, 3 categories), y (Numeric)] with set_constraint("y", LessEqualTo, 0.5): the filter reads outputs(row, 1) (category 1 of class) against domain…

**Fix:** Iterate the filtered "Target" variables from get_variables_and_descriptives with for_each_feature_block (already defined at line 157) and pass the feature index to both `outputs` and the Domain; for multi-feature variables filter each feature column.

*Verifier:* filter_feasible_points (1172-1200) iterates neural_network->get_output_variables() and passes column_index (variable index) into outputs(row, column) and domain_index into the frontiers, while Domain::set lays the frontiers out per feature (494-525, get_feature_dimensions) and the output matrix is per feature. Variable::get_feature_count returns categories_number for categoricals…

#### core-types-2 — TensorView::fill on host ignores dtype: BF16/INT8 host views overflow their storage

`opennn/core/tensor_types.cpp:78-88` · low · bug · lines -1 · effort S · risk low · partial

The host branch asserts FP32 and then writes `size()` floats. The assert is compiled out in Release, so a BF16 host view writes 2x its byte_size and an INT8 view 4x. Host BF16 views are a real shape in this codebase: tests/core/tensors_test.cpp:569 builds `TensorView bf16_view(storage.data(), shape, Type::BF16, Device::CPU)` and :608/:618 build more. Scenario: `TensorView v(storage.data(), {2, 2}, Type::BF16, Device::CPU); v.setZero();` writes 16 bytes into the 8-byte `storage` -> stack/heap corruption with no diagnostic. The same function has a CHECK-everything contract everywhere else (as_matrix/as_vector throw on non-FP32), so this is the one accessor that silently corrupts instead of…

**Fix:** Replace the assert with dtype-aware behaviour: `if (value == 0.0f) { memset(data, 0, byte_size()); return; } throw_if(!is_fp32(), "TensorView::fill: a non-zero fill requires FP32 storage.");` then the existing std::fill. byte_size() already accounts for type_bytes, so BF16/INT8 zeroing becomes correct and non-zero fills of narrow types fail loudly. Add a tensors_test case for setZero on a BF16 host view.

*Verifier:* Code verified: tensor_types.cpp:78-88 asserts FP32 then writes size() floats; assert vanishes in Release; byte_size() is type-aware. However no host BF16/INT8 TensorView that is ever filled exists in opennn/ (grep for `Type::BF16, Device::CPU` / `Type::INT8, Device::CPU` in opennn/ returns nothing); the only host BF16 views are in tests/core/tensors_test.cpp:569,608,618 and they only check that…

#### core-kernels-5 — Block-per-row norm kernels and rope_apply_kernel compute row offsets in int; overflow past 2^31 elements

`opennn/core/cuda/kernel_normalization.cu:18-22` · low · bug · lines 0 · effort S · risk low · partial

norm_forward_kernel (`X + idx * D`), norm_backward_kernel (lines 953-956), norm_weight_gradient_coalesced_kernel (`dY[n * D + d]`, lines 1030-1033) and rope_apply_kernel (`row * model_dim`, kernel_attention.cu:333) multiply two ints; the callers range-check rows and cols individually through to_int but not their product. The warp-per-row siblings already do `Index base = Index(row) * D`. Failure scenario: BF16 layer norm with D = 4096 (not a warp shape, so the block kernel runs) over 524,288 rows (batch 64 x sequence 8192): idx * D reaches 2^31 and wraps negative, reading/writing outside the tensor; that activation is 4 GiB, which fits an 80 GB GPU. rope over the same Q tensor wraps…

**Fix:** Use `const Index base = Index(idx) * D` in norm_forward_kernel and norm_backward_kernel, `Index(n) * D + d` in norm_weight_gradient_coalesced_kernel, and `const Index row_base = Index(row) * model_dim` in rope_apply_kernel (and Index for base_e/head_start).

*Verifier:* Overflow sites are real: norm_forward_kernel (normalization.cu:18-22) `X + idx * D` with int idx,D; norm_backward_kernel (950-956) same; norm_weight_gradient_coalesced_kernel (1028-1032) `dY[n * D + d]` int*int; rope_apply_kernel (attention.cu:333) `row * model_dim` int. Callers pass to_int(flat_rows())/to_int(flat_columns()) separately (layer_normalization_operator.cpp:273-274, gqa layer…

#### core-device-7 — allocate() turns the growth-guard diagnostic into a cache flush (device sync + cudaFree) before rethrowing

`opennn/core/device_backend.cpp:567-596` · low · bug · lines 0 · effort S · risk low · confirmed

allocate_cuda throws the 'CUDA alloc forbidden (warmup incomplete)' error as a runtime_error; allocate()'s catch (const runtime_error&) treats every runtime_error as an OOM, calls CudaBlockCache::flush() (cudaEventSynchronize on every pending event + cudaFree of every cached block) and then calls allocate_cuda again, which throws the same error. The guard is meant to be a pure diagnostic during warmup/capture; here it empties the whole block cache and forces a device-wide synchronization as a side effect, and if the forbidden allocation happens inside a StreamCapture (the guard's main use), the cudaFree/cudaEventSynchronize invalidate the capture so the user sees a capture error instead of…

**Fix:** Move the throw_if(cuda_allocation_growth_forbidden(), ...) from allocate_cuda into allocate() right after the take() (so recycled blocks stay allowed during capture but a real cudaMalloc is refused before the try/flush), leaving the catch to handle only cudaMalloc failures.

*Verifier:* allocate_cuda (121-134) throws the growth-guard message via throw_if at 124 (a runtime_error) before cudaMalloc; allocate() (567-596) catches `const runtime_error&` at 586 and calls CudaBlockCache::flush() (455-479: cudaEventSynchronize on every pending event + cudaFree of every block) before retrying allocate_cuda at 593, which throws the same message again. CudaAllocationGrowthGuard (322-346)…

#### core-types-5 — multiply_gpu computes strides and batch_count in int: signed overflow at 2^31 elements per matrix

`opennn/core/tensor_operations.cpp:1580-1582` · low · UB · lines 0 · effort S · risk low · confirmed

rows_a/cols_a/rows_b/cols_b are `int`. The products are formed in int and only then widened to `long long`. In the flattened CUDA path `rows_a = to_int(input_a.size() / cols_a)`, so rows_a * cols_a is the whole activation: a (B*S, D) = (524288, 4096) BF16 activation (4 GiB, feasible on an 80 GB card) is exactly 2^31 elements -> signed overflow (UB), and batch_count then divides by the wrapped value. The same file already widens correctly one line later for stride_output (Index multiply) and in get_descriptor (`Index(batch_count) * channels * ...`).

**Fix:** Widen before multiplying: `const long long stride_a = 1LL * rows_a * cols_a;` (same for stride_b) and `const int batch_count = to_int(input_a.size() / stride_a);`. Zero behaviour change below the limit.

*Verifier:* tensor_operations.cpp:1565-1568 declare rows_a/cols_a/rows_b/cols_b as int; :1570-1571 set rows_a = size/cols_a for the flattened path; :1580-1582 form `rows_a * cols_a` in int before widening to long long, and batch_count divides by it. stride_output (:1583) correctly multiplies Index. (524288, 4096) is exactly 2^31 -> signed overflow UB. Fix `1LL * rows_a * cols_a` and `to_int(input_a.size() /…

#### core-types-1 — TensorView::fill(0) / Buffer::setZero zero CUDA memory on the legacy stream, unordered with the lane streams

`opennn/core/tensor_types.cpp:53-59` · low · bug · lines 0 · effort S · risk low · partial

fill_cuda routes value==0 to device::set_zero, which is a plain cudaMemset on the legacy default stream (device_backend.cpp:630), while the non-zero branch uses cudnnSetTensor on the cuDNN handle bound to the active lane (device_backend.cpp:1190). Every lane stream is created with cudaStreamNonBlocking (device_backend.cpp:1154), and by CUDA semantics non-blocking streams perform no implicit synchronisation with the legacy stream. So the memset is unordered with respect to the kernels before and after it on the compute stream. Concrete scenario: back_propagation.cpp:624 `return destination.setZero();` in the per-batch delta merge on CUDA: a kernel on the lane stream that later reads…

**Fix:** In fill_cuda use the stream-ordered primitive that already exists: `device::set_zero_async(view.get_data(), view.byte_size(), device::get_compute_stream())` (TensorView::set_zero_async at tensor_types.h:714 already does exactly this). Do the same in Buffer::setZero for allocation_device == CUDA, and make Buffer::migrate_to default to `device::get_compute_stream()` instead of nullptr (its `if (stream) synchronize(stream)` then always runs, preserving the host-visibility guarantee). One PR, no…

*Verifier:* The race premise is wrong for the stream that runs almost everything. device_backend.cpp:1135 creates lane 0 with `create_stream_handle(cudaStreamDefault)`, i.e. a BLOCKING stream; only lanes >=1 (:1154) and transfer_stream (:1136) are non-blocking. By CUDA semantics a legacy-stream cudaMemset (set_zero, :630) is implicitly ordered before/after all work on blocking streams, so the memset in…

#### dataset-a-21 — get_features_number() accumulates Index into an int

`opennn/dataset/dataset.cpp:548-552` · low · bug · lines 0 · effort S · risk low · partial

std::accumulate deduces its accumulator type from the initial value; with `0` it is int, so every partial sum (lambda returns Index) is narrowed to int, and the result is widened back. The sibling get_features_number(VariableRole) three lines below correctly uses Index(0). Harmless at realistic sizes but a silent narrowing in the central feature-count query used for every buffer size, and a copy-paste slip between twins.

**Fix:** Use `Index(0)` as the initial value (or `get_variable_feature_count(variables)` from variable.h:176-180, which already computes exactly this sum).

*Verifier:* dataset.cpp:548-552 uses accumulate with initial value 0 (int) while the sibling at 554-558 uses Index(0): true, harmless narrowing. Correction: the suggested alternative get_variable_feature_count(variables) does not exist anywhere in the repo (grep -rn over opennn/ returns nothing), so the fix is just Index(0).

#### selection-testing-12 — GrowingInputs ranks candidate inputs by correlation with the first target only

`opennn/model_selection/growing_inputs.cpp:104-105` · low · bug · lines 0 · effort S · risk low · confirmed

The candidate order is derived from `calculate_input_target_correlation_values().col(0)`, i.e. only the first target column. For a multi-output regression (or a one-hot multi-class target whose correlation matrix has one column per class) inputs strongly related to targets 2..T are tried last and rejected early by validation failures. GeneticAlgorithm::initialize_population_correlations uses `rowwise().mean()` over all targets (genetic_algorithm.cpp:136-138), which is the intended semantics.

**Fix:** Use `dataset->calculate_input_target_correlation_values().array().abs().rowwise().mean()` (matches the GA). One-line change.

*Verifier:* growing_inputs.cpp:104-105 `calculate_input_target_correlation_values().col(0).array().abs()`; Dataset::calculate_input_target_correlation_values returns a MatrixR (dataset.h:114), one column per target. GA's initialize_population_correlations (genetic_algorithm.cpp:136-138) uses `.array().abs()` then `.rowwise().mean()`. For multi-target problems GI's ranking ignores targets 2..T. One-line fix,…

#### layers-a-7 — Upsampling launch size computed in int before the overflow check

`opennn/neural_network/layers/kernel_upsampling.cu:56-71` · low · bug · lines 0 · effort S · risk low · confirmed

upsampling_forward_cuda and upsampling_backward_cuda compute n = batch * (in_h*scale) * (in_w*scale) * channels entirely in int and only then pass it to launch_elementwise_strided, whose checked_int guard (kernel_common.cuh:63-69) cannot see an overflow that already wrapped. The sibling kernel_concat.cu:35 does it right (Index(batch) * H * W * slice_ch). A batch of 64 at 640x640x128 after 2x upsampling is 3.36e9 elements: n wraps negative, the kernel launches with a negative bound and processes nothing, leaving the output uninitialized without any error. Rare at current batch sizes but a silent failure mode on large images.

**Fix:** Compute as Index: const Index n = Index(batch) * (in_h * scale) * (in_w * scale) * channels; (same for backward) so checked_int throws on overflow instead of wrapping.

*Verifier:* kernel_upsampling.cu:58-59 and 66-67 compute n as int from int operands (batch, in_h, in_w, channels, scale are all const int parameters) and then pass it to launch_elementwise_strided(Index n, ...) (kernel_common.cuh:89-94), whose checked_int (63-69) only sees the already-wrapped value. kernel_concat.cu:35 uses Index(batch) * H * W * slice_ch, which is the correct pattern. Fix is a type change…

#### r2-set-vs-compile-device-ordering-7 — apply_input_shape on a configured cross-attention layer overwrites the source sequence length with the query length

`opennn/neural_network/layers/multihead_attention_layer.cpp:221-240` · low · bug · lines 0 · effort S · risk low · unverified

When heads_number > 0, MultiHeadAttention::apply_input_shape calls set(new_input_shape[0], new_input_shape[0], ...) - both query and source length from the query shape. For a cross-attention layer (constructed with distinct query/source shapes, or loaded with CrossAttention=true) any later set_input_shape({q, e}) silently makes source_sequence_length == q, so the key/value projections and every (B, H, S, D) slot are planned for the wrong source length and the encoder input no longer matches. Reachable from any neuron/shape-editing flow that calls Layer::set_input_shape on an already configured layer (from_JSON is safe only because heads_number is still 0 at that point). The guard branch at…

**Fix:** `set(new_input_shape[0], cross_attention ? source_sequence_length : new_input_shape[0], new_input_shape[1], heads_number, attention.use_causal_mask, label);` and the same in the heads_number <= 0 branch (line 229). A one-line test with the cross constructor followed by set_input_shape pins it.

#### nn-expression-16 — Generated HTML: CSS selectors use attribute `float` instead of `type`; PHP header uses `source` instead of `src`

`opennn/neural_network/model_expression.cpp:149-227` · low · bug · lines 0 · effort S · risk low · confirmed

javascript_subheader styles `.neural-cell input[float="range"]`, `input[float="number"]`, `input[float="text"]` and the slider thumb/track rules (8 selectors) but emit_js_inputs_html writes `<input type="range" ...>` / `type="number"`, so none of those rules ever match and the sliders and number boxes render unstyled. This is the signature of a global `type`->`float` rename. php_subheader loads jQuery/Bootstrap with `<script source=...>`; the attribute is `src`, so those scripts never load.

**Fix:** Replace `[float=` with `[type=` in the eight selectors and `source =` with `src=` in the two PHP script tags (or drop the unused jQuery/Bootstrap JS tags entirely since the page uses neither).

*Verifier:* javascript_subheader (lines 149-227) uses `input[float="range"]`, `input[float="number"]`, `input[float="text"]` and four slider-track/thumb selectors with [float="range"], while emit_js_inputs_html writes `type="range"` (1875) and `type="number"` (1876). php_subheader lines 60-61 use `<script source = ...>`; the HTML attribute is src. Both are plain text fixes with zero LOC delta.

#### xcut-api-5 — noexcept accessors call validators that throw (get_parameters_data const, flat_rows, get_inputs_number)

`opennn/neural_network/neural_network.h:126-142` · low · bug · lines 0 · effort S · risk low · confirmed

const float* get_parameters_data() const noexcept and get_states_data() const noexcept call Buffer::as<float>() const, which calls validate_state() (tensor_types.h:530-536) with three throw_if paths; the non-const siblings are correctly not noexcept. TensorView::flat_rows() noexcept (tensor_types.h:580-584) calls shape.size(), which throws on rank > MaxRank or Index multiplication overflow (checked_index_multiply). Layer::get_inputs_number() const noexcept (layer.h:132) calls get_input_shape().size() with the same throwing path. If any of those validators fires, the noexcept converts a catchable runtime_error into std::terminate; the const/non-const asymmetry also invites callers to assume…

**Fix:** Drop noexcept from get_parameters_data() const, get_states_data() const, TensorView::flat_rows() and Layer::get_inputs_number(); they are not on hot loops and the promise is false. (Alternatively make Buffer::as<T>() const a plain cast like TensorView::as<T>() and keep validation in the mutating entry points.)

*Verifier:* h:126 and h:142 are `const noexcept` and call Buffer::as<float>() const (tensor_types.h:389-393) which calls validate_state (tensor_types.h:530-535, two throw_if paths + validate_device); non-const siblings at h:125/141 are not noexcept. TensorView::flat_rows() noexcept (tensor_types.h:580-584) calls shape.size() (tensor_types.h:233-240) which goes through detail::checked_index_multiply (line…

#### operators-a-10 — LayerNorm variance computed as E[x^2] - mean^2 in float loses all precision for offset inputs

`opennn/neural_network/operators/layer_normalization_operator.cpp:51-58` · low · bug · lines 0 · effort S · risk low · confirmed

Both the CPU row loop and the CUDA norm_forward_kernel (kernel_normalization.cu:51) compute the variance in one pass as sum_sq/D - mean^2 in fp32, then clamp at 0. With a row whose mean is ~1e3 and standard deviation ~1, sum_sq/D and mean^2 are both ~1e6 and differ in the 7th digit, so the variance is rounding noise or exactly 0; the clamp then yields inv_std = 1/sqrt(eps) = 1000 and the normalized row explodes. Embedding + positional inputs and post-residual streams can carry offsets of this size in deep stacks. The two-pass form (mean first, then sum of squared deviations) costs one extra pass over a row that is already in L1 and has no such failure mode; PyTorch/cuDNN use it.

**Fix:** Compute `variance = (input_map - mean).square().sum() * inv_D` on the CPU (the second pass is over the same cached row), and make the same change in norm_forward_kernel so CPU and GPU stay bit-comparable; the RMS path is unaffected (no mean subtraction). Verify with the existing LN numerical-gradient tests plus one row with a 1e3 offset.

*Verifier:* layer_normalization_operator.cpp:51-56 computes `max(sum_sq*inv_D - mean*mean, 0)` in fp32; kernel_normalization.cu:47-51 does the same (`fmaxf(local_sum_sq*inv_D - mean*mean, 0.0f)`). Catastrophic cancellation for rows with |mean| >> std is a textbook failure of the one-pass formula. The second CPU pass is over the same L1-resident row (input_map is reused at line 62 anyway). No existing test…

#### operators-a-9 — Projection scratch is reshaped with input_features where the GEMM output width is output_features

`opennn/neural_network/operators/multihead_projection_operator.cpp:185-188` · low · bug · lines 0 · effort S · risk low · confirmed

linear_forward writes an (rows, output_features) matrix into scratch_2d, but scratch_2d and the backward's output_delta_2d (lines 219-221) are reshaped to {rows, input_features}. It only works because MultiHeadAttention::set enforces embedding_dimension % heads == 0 and head_dimension = embedding/heads, so heads*head_dim == input_features; MultiHeadProjectionOperator::set takes heads_number and head_dimension independently, so any caller (a GQA-style K/V projection, or Neural Designer building the operator directly) with output_features != input_features gets a GEMM whose output view has the wrong column count, silently misreading the scratch in the backward and, when output_features >…

**Fix:** Use `output_features` (== heads_number * head_dimension) in both reshape_prefix calls, and add `throw_if(output_features != heads_number * head_dimension, ...)` derived from the output view so a mismatch is reported instead of aliased.

*Verifier:* multihead_projection_operator.cpp:139-140 `scratch.reshape_prefix({rows, input_features})` feeds a GEMM that writes (rows, output_features); back_propagate 172-174 same. MultiHeadProjectionOperator::set (101-107) forwards heads*head_dim as output_features to CombinationOperator::set, and combination_operator.h:18-19 exposes both fields, so `output_features` is the correct width. The only caller…

#### operators-b-8 — next_utf8_codepoint returns the masked lead bits (not the lead byte) for a sequence truncated at end of text

`opennn/neural_network/operators/tokenizer_operator.cpp:317-332` · low · bug · lines 0 · effort S · risk low · partial

For a multi-byte lead whose continuation bytes are missing because the text ends, the function returns `codepoint`, which at that point holds only the masked payload bits of the lead (lead & 0x1F/0x0F/0x07), whereas the sibling branch for a bad continuation byte returns the raw `lead`. Concrete: encoding "caf\xC3" (a UTF-8 stream cut mid-character, common when chunking bytes) yields codepoint 0x03 for the last byte; the byte-level BPE then maps byte 0x03 through byte_encoder instead of 0xC3, so the token stream differs from the reference tokenizer and from the same text with the bad-continuation form; WordPiece's basic_tokenize drops it as a control character. Both invalid branches should…

**Fix:** Return `lead` in the truncated branch (`return length == 1 ? codepoint : lead;` or restructure so both invalid paths share one `++position; return lead;`). Add a tokenizer test with a trailing truncated multibyte sequence asserting the raw byte survives.

*Verifier:* next_utf8_codepoint (308-335): truncated-at-end branch returns `codepoint` = masked payload bits (e.g. 0xC3 -> 0x03), the bad-continuation branch returns `lead`. The inconsistency is real. However the consequence is mis-described: PreTokenizeRun::emit (864-870) re-encodes codepoints with append_utf8, so the bad-continuation branch turns raw byte 0xC3 into U+00C3 = bytes C3 83, which also does not…

#### selection-testing-13 — Cumulative gain accumulates 0.05f twenty times; the last bucket index exceeds the sample count for large testing sets, and positives are recounted from zero 20 times

`opennn/testing_analysis/testing_analysis.cpp:664-680` · low · UB · lines 0 · effort S · risk low · confirmed

`percentage += 0.05f` twenty times yields 1.0000001f, so `Index(percentage * float(n))` equals n+1 once n >= ~8.4M (verified: n = 10,000,000 gives 10,000,001) and the loop reads sorted_targets(n) past the end (Eigen release build: out-of-bounds read). Even below that size, the inner loop recounts positives from j = 0 for every one of the 20 buckets (20 n instead of n) and the 1.0 bucket is reported as 1.0000001.

**Fix:** Compute the bucket edge exactly: `const Index maximum_index = min((i + 1) * testing_samples_number / (points_number - 1), testing_samples_number);` and `cumulative_gain(i+1,0) = float(i + 1) / float(points_number - 1)`; keep a running `positives` and a `next_row` cursor so each sample is inspected once.

*Verifier:* testing_analysis.cpp:664-680 read. Reproduced numerically: twenty float32 additions of 0.05f give 1.0000001f; Index(1.0000001f * float32(10,000,000)) = 10,000,001 and float32(8,400,000) gives 8,400,001, so maximum_index == n+1 and sorted_targets(n) is read past the end (Eigen only asserts in debug). The positives recount from j=0 per bucket (20n) and the last bucket label being 1.0000001 are also…

#### selection-testing-20 — Unchecked preconditions: error-data descriptives dereference before check(); GoodnessOfFitAnalysis::save ignores a failed open

`opennn/testing_analysis/testing_analysis.cpp:229-248` · low · bug · lines 0 · effort S · risk low · partial

calculate_error_data_descriptives reads neural_network->get_outputs_number() and dataset->get_samples_number() before calling calculate_error_data(), which is where check() (the null-pointer guard) lives, so an unset TestingAnalysis segfaults instead of throwing the 'neural network is not set' message every sibling produces; calculate_error_data_histograms has the same shape. GoodnessOfFitAnalysis::save (lines 922-928) opens an ofstream and writes without testing is_open(), so saving to an unwritable path silently succeeds.

**Fix:** Call calculate_error_data() first (it runs check()) and derive outputs_number/testing_samples_number from error_data.dimension(2)/dimension(0), removing the two direct dereferences. In GoodnessOfFitAnalysis::save add `throw_if(!file.is_open(), "Cannot open file {}.", file_name.string());` after the ofstream line.

*Verifier:* calculate_error_data_descriptives (testing_analysis.cpp:229-248) dereferences neural_network and dataset at :232-234 before calculate_error_data() runs check() at :173 — confirmed. GoodnessOfFitAnalysis::save (:922-928) never tests is_open() — confirmed. Wrong part: calculate_error_data_histograms (:250-261) calls calculate_percentage_error_data(), whose first statement is check() (:211), and…

#### training-loss-4 — Seven raw, unchecked CUDA calls in the YOLO drivers (7 of the 8 in all opennn .cpp files)

`opennn/training_strategy/loss.cpp:982-1200` · low · bug · lines 0 · effort S · risk low · partial

cudaStreamSynchronize (986, 1198), cudaMemcpy (987, 1014, 1036, 1200), cudaMemcpyAsync (1145) and cudaMemsetAsync (1169) are called bare, with no CHECK_CUDA and no device:: helper, while the rest of the scope uses device::copy_async/synchronize/set_zero_async (error_functions.cpp:404-409, loss.cpp:1634). A grep over opennn/**/*.cpp shows loss.cpp holds 7 of the 8 bare calls in the library. A failed D2H at 987 leaves tgt_cpu zero-filled, so TAL assigns nothing and the batch trains on a silent all-background gradient; a failed H2D at 1036 leaves stale deltas on the device; a failed memset at 1169 accumulates the error onto garbage.

**Fix:** Route all seven through the existing helpers: device::synchronize(stream), device::copy_async(dst, src, bytes, CopyKind, stream) + synchronize for the D2H reads, device::set_zero_async for the memset. Same line count, every failure surfaces through CHECK_CUDA.

*Verifier:* The seven bare calls are real (grep: loss.cpp:986, 987, 1014, 1036, 1145, 1169, 1198, 1200 -- actually eight including 1198/1200 as two). The '7 of 8 in all opennn .cpp' count is wrong: device_backend.cpp holds ~17 bare calls (cudaGetLastError/cudaFree/cudaStreamDestroy in error-clearing and teardown, deliberate) and detection_layer.cpp:71 has a bare cudaMemcpyAsync; the count holds only if…

#### training-loss-5 — v8 delta upload relies on legacy-stream implicit ordering with the compute stream

`opennn/training_strategy/loss.cpp:1034-1039` · low · bug · lines 0 · effort S · risk low · confirmed

The host-computed delta is uploaded with synchronous cudaMemcpy from pageable memory on the legacy NULL stream, and the consumer (the head layer's backward, launched right after by back_propagate_layers) runs on device::get_compute_stream(). CUDA documents that pageable H2D cudaMemcpy returns once the data is staged, with the DMA possibly outstanding. Today this is safe only because lane 0 is created with cudaStreamDefault (device_backend.cpp:1135) so the legacy stream synchronizes with it; lanes >0 are cudaStreamNonBlocking (device_backend.cpp:1154) and the convolution backward already forks onto lane 1 (convolution_operator.cpp:672-723). The sibling path at line 1145 already does it right…

**Fix:** Use device::copy_async(device_delta, delta_cpu.data(), bytes, CopyKind::HostToDevice, device::get_compute_stream()) as line 1145 does; pageable sources are staged before the call returns, so delta_cpu may still be a loop-local vector. Fold into the PR for training-loss-4.

*Verifier:* loss.cpp:1034-1038 synchronous pageable cudaMemcpy H2D on the legacy stream; consumer is the head layer backward on get_compute_stream(). device_backend.cpp:1135 creates lane 0 with cudaStreamDefault (blocking), lanes >0 with cudaStreamNonBlocking (1154); convolution_operator.cpp:672-723 forks wgrad onto lane 1 and restores lane 0 via ScopeExit, so at loss time the active lane is 0 and the…

#### layers-a-4 — Clamping and Unscaling claim ranks 1-3 but size their state from dimension 0 only

`opennn/neural_network/layers/clamping_layer.cpp:127-139` · low · bug · lines +1 · effort S · risk low · confirmed

Clamping::accepts_input_rank returns true for 1,2,3 (clamping_layer.h:49) and Scaling's accepts_input_rank (1,2,3) is inherited by Unscaling, yet Clamping::set sizes lower/upper bounds from output_shape.dim_or_zero(0) and Unscaling::set collapses the shape to {dim0}. The forward path cycles bounds over the flattened columns (apply_clamping_cpu, clamping_layer.cpp:29-53, uses column_index % features with as_flat_matrix whose column count is the LAST dimension, tensor_types.h:579), so for a rank-2 shape {2,3} two bounds are cycled over three columns: column 2 silently gets feature 0's bounds, and write_expression (clamping_layer.cpp:242-256) iterates output_shape[0] rows while output_names…

**Fix:** Either make both honest about rank 1 (Clamping::accepts_input_rank -> is_one_of(rank, 1); add bool accepts_input_rank(Index rank) const override { return rank == 1; } to Unscaling) or make them feature-last like Scaling: Clamping uses output_shape.back() for the feature count and iterates get_outputs_number() in write_expression with i % features; Unscaling stores the full shape and sizes descriptives from back(). The first option is two lines and matches every current caller; the second is…

*Verifier:* clamping_layer.h:49 accepts ranks 1,2,3; Clamping::set (clamping_layer.cpp:127-138) sizes lower/upper_bounds from output_shape.dim_or_zero(0); apply_clamping_cpu (29-53) cycles feature_index = column_index % features over as_flat_matrix columns, whose column count is the last dimension (tensor_types.h:577-579 flat_columns), so for {2,3} two bounds cycle over three columns; write_expression…

#### layers-b-15 — rnn_copy_regions_cuda silently drops regions beyond RNN_COPY_MAX_REGIONS

`opennn/neural_network/layers/kernel_recurrent.cu:116-126` · low · bug · lines +1 · effort S · risk low · partial

The host wrapper clamps count to 16 and copies only the first 16 specs; any further weight/bias regions are neither packed nor unpacked and no error is raised. Today the largest caller (LSTM: 8 matrices + 4 biases = 12) fits, so the clamp is latent, but a future cell with more linear layers or per-gate recurrent biases would train with silently unpacked weights. The same file already uses checked_host_condition for a hard limit (line 191).

**Fix:** Replace the clamp with checked_host_condition(count > RNN_COPY_MAX_REGIONS, "rnn_copy_regions_cuda: too many regions.") and loop over count directly.

*Verifier:* kernel_recurrent.cu:116-123 clamps count to RNN_COPY_MAX_REGIONS (16, kernel_recurrent.cuh:8) and silently drops the rest; checked_host_condition is used for a hard limit at 191-192. Today's maximum is 12 (LSTM passes 8 matrices and 4 bias vectors, long_short_term_memory_layer.cpp:961-981; Recurrent passes 2). Mis-scoped: the only caller, CudnnRnnState::cudnn_copy_weight_regions_…

#### training-loss-18 — CE3d backward kernel declares outputs/output_deltas __restrict__ but the caller aliases them in place

`opennn/training_strategy/kernel_losses.cu:301-330` · low · UB · lines +1 · effort S · risk low · confirmed

When output_delta_overwrites_outputs() is true (CE3d + GPU + softmax output), Loss aliases the delta view onto the outputs (`back_propagation.get_output_delta() = input`, loss.cpp:1672-1673 and 1732-1733) so the gradient is written over the softmax outputs. cross_entropy_3d_multiple_backward_kernel then receives the same pointer as `const T* __restrict__ outputs` and `T* __restrict__ output_deltas`, which violates the restrict contract (formally UB). The access pattern is same-index read-then-write so current compilers emit correct code, but the qualifier promises the compiler something the caller deliberately breaks.

**Fix:** Drop `__restrict__` from `outputs` and `output_deltas` in that kernel (keep it on `targets`), with a one-line comment that the loss may alias them in place.

*Verifier:* kernel_losses.cu:301-330: `const T* __restrict__ outputs` and `T* __restrict__ output_deltas`; loss.cpp:1672-1673 and 1732-1733 alias `back_propagation.get_output_delta() = input` when output_delta_overwrites_outputs() (1717-1725: CE3d + GPU + softmax). Same-index read-then-write (line 327), so no miscompile today; dropping the qualifier on those two is the right minimal fix.

#### training-optimizers-9 — write_override_results emits the literal text "QUIET_NAN" into saved result files; get_training_error reads out of bounds on empty history

`opennn/training_strategy/training_result.cpp:143-145` · low · bug · lines +1 · effort S · risk low · confirmed

The placeholder for a missing validation history is the string "QUIET_NAN" - a search/replace of the NAN constant that also hit this string literal. TrainingResult::save writes it to the user's file (line 79). The size==0 branch of the same function writes "NA". In the same file, get_training_error() indexes training_error_history(size-1) with no size check (lines 39-42), so a default-constructed or fully-resized-to-zero result reads one element before the buffer (Eigen assert in debug, OOB read in release), while get_validation_error() guards and returns 0.0f.

**Fix:** Use "NA" (consistent with the empty branch) and give get_training_error the same guard as get_validation_error (`if (training_error_history.size() == 0) return QUIET_NAN;`). Two lines.

*Verifier:* training_result.cpp:143-145 has the string literal "QUIET_NAN" while the size==0 branch (132-136) writes "NA"; save (71-80) writes override_results verbatim. get_training_error (39-42) indexes size-1 unguarded; get_validation_error (44-53) walks back and returns 0.0f. TrainingResult(const Index = 0) default-constructs an empty history.

#### core-utils-18 — Json::parse rejects a UTF-8 BOM, so files re-saved by Windows tools fail to load

`opennn/core/json.cpp:469-481` · low · bug · lines +2 · effort S · risk low · confirmed

skip_ws (lines 290-298) only skips space/tab/CR/LF, so a file beginning with EF BB BF fails in parse_value with 'unexpected character' at position 0. AGENTS.md notes that `>`/Out-File in this environment write UTF-8 with BOM, and several repo headers carry one, so a model JSON touched by PowerShell or Notepad stops loading with a message that does not mention the cause. RFC 8259 allows parsers to ignore a leading BOM.

**Fix:** In Json::parse (or JsonDocument::load) strip a leading "\xEF\xBB\xBF" from the string_view before constructing the Parser: `if (text.starts_with("\xEF\xBB\xBF")) text.remove_prefix(3);`.

*Verifier:* read_text_file (io_utilities.cpp:69-88) returns the raw bytes; Parser::skip_ws (json.cpp:290-298) skips only space/tab/LF/CR; parse_value (:417-428) then fails on 0xEF with 'unexpected character'. Json::parse (:469-477) and JsonDocument::load (:478-481) are the entry points; BytePairTokenizer::load (tokenizer_operator.cpp:687) goes through the same path. Two-line strip of a leading EF BB BF in…

#### core-utils-11 — histogram()/histogram_centered() with bins_number <= 0 hit std::clamp UB and out-of-range writes

`opennn/core/statistics.cpp:367-492` · low · UB · lines +2 · effort S · risk low · confirmed

The free functions do not validate bins_number, unlike the sibling constructor Histogram(data, bins) which returns early for bins_number <= 0 (line 160). histogram(v, 0): unique_values gets one element, 1 <= 0 fails, the else branch computes length = inf, and fill_frequencies -> refined_bin -> clamped_bin calls std::clamp(x, 0, -1) (precondition violated: UB) and then writes frequencies(j) into an empty vector. histogram_centered(v, 0): bin_center = 0 and `minimums(bin_center-1)` indexes -1. bins_number is user-provided at the public API (histograms(matrix, bins), testing_analysis.cpp:259).

**Fix:** Add `throw_if(bins_number <= 0, "histogram: bins_number must be positive.");` at the top of histogram() and histogram_centered(), and make the constructor throw the same way instead of returning an empty object.

*Verifier:* histogram() (statistics.cpp:367-448) has no bins_number check: with bins 0 and non-empty data the unique loop breaks after the first push (ssize 1 > 0), `1 <= 0` fails, the else branch sets length = (max-min)/0, the lambda at :444 reads minimums(0) of an empty vector, and refined_bin (:68-78) calls clamped_bin -> clamp(x, 0, -1) (:63-66, precondition violated) then frequencies(j)++ on an empty…

#### layers-a-5 — Same-padding maths yields negative padding when stride exceeds kernel+1

`opennn/neural_network/layers/convolutional_layer.cpp:115-131` · low · bug · lines +2 · effort S · risk low · confirmed

get_padding_height computes total_padding = (out-1)*stride + k - in with out = ceil_div(in, stride) and returns (total_padding+1)/2 without clamping. When stride > kernel the total can be negative and C++ truncation makes the result -1: Convolutional({8,8,1}, {1,1,1,1}, "Identity", {4,4}, "Same") gives out = 2, total = 4 + 1 - 8 = -3, padding = (-3+1)/2 = -1. validate_convolution_configuration (lines 23-72) only requires odd kernels and stride <= input for "Same", so this configuration is accepted. The -1 reaches ConvolutionOperator::set (convolution_operator.cpp:280-281) and the CPU im2col uses input_row = output_row*stride + kernel_row - padding (convolution_operator.cpp:394) which then…

**Fix:** Clamp: return max(total_padding, Index(0)) before the (x+1)/2, in both get_padding_height and get_padding_width (the output size formula already matches cuDNN's floor((in+2p-k)/s)+1 with p = 0 in this regime, as checked for in=8,k=1,s=4). Add one unit test for a 1x1 stride-4 Same convolution comparing CPU output to the expected subsample.

*Verifier:* convolutional_layer.cpp:101-105 get_output_height = ceil_div(in, stride) for Same; 115-121 total_padding = (out-1)*stride + k - in, returned as (total+1)/2 with no clamp. For in=8,k=1,s=4: out=2, total=4+1-8=-3, (-3+1)/2 = -1. validate_convolution_configuration (convolutional_layer.cpp:23-71) only checks stride <= input (55-57) and odd kernel for Same (67-71), so the configuration is accepted.…

#### response-opt-17 — perform_response_optimization/solve_once dereference a null neural_network without a check

`opennn/response_optimization/response_optimization.cpp:2352-2353` · low · bug · lines +2 · effort S · risk low · confirmed

ResponseOptimization's constructor defaults to nullptr and set_formula_constraint explicitly guards `throw_if(!neural_network, ...)`, but perform_response_optimization (2352), get_descriptives (426), get_variables_and_descriptives (444) and get_advised_point (2070) dereference the pointer directly. `ResponseOptimization opt; opt.set_objective("y", Minimize); opt.perform_response_optimization();` segfaults instead of reporting the missing network like the sibling setter does.

**Fix:** One `throw_if(!neural_network, "ResponseOptimization: neural network not set")` at the top of perform_response_optimization and solve_once (get_advised_point too); the other paths are only reachable through them.

*Verifier:* Constructor defaults to nullptr (h:105, cpp:170-173); set_formula_constraint guards with throw_if(!neural_network) (252-253, 312-313) but perform_response_optimization dereferences at 2352-2353 (and earlier via expand_fixed_objectives -> get_variables_and_descriptives 444-452 -> get_descriptives 426), get_advised_point at 2070. solve_once -> Objectives ctor -> get_objectives_number ->…

#### training-optimizers-16 — TrainingStrategy::to_JSON/save dereference loss and optimizer without a check

`opennn/training_strategy/training_strategy.cpp:123-147` · low · bug · lines +2 · effort S · risk low · confirmed

set(nullptr, ...) and set_neural_network(nullptr) reset both unique_ptrs (lines 26-30, 44-48), and the default constructor goes through set(nullptr, nullptr). to_JSON then calls loss->get_name() and optimizer->get_name() on null pointers; `TrainingStrategy ts; ts.save(path);` segfaults instead of throwing. train() in the same class guards its preconditions with throw_if.

**Fix:** throw_if(!loss || !optimizer, "TrainingStrategy::to_JSON: loss or optimizer is not set.") at the top of to_JSON.

*Verifier:* training_strategy.h:22 `explicit TrainingStrategy(NeuralNetwork* = nullptr, Dataset* = nullptr)`; set() (cpp 22-33) resets optimizer and loss when neural_network is null; set_neural_network (42-54) likewise. to_JSON (125-147) calls loss->get_name() and optimizer->get_name() unguarded; save (186-189) calls save_json_file(*this). train() (111-120) uses throw_if.

#### selection-testing-11 — Inputs selection permanently switches the user's optimizer display off (and GrowingNeurons does not)

`opennn/model_selection/growing_inputs.cpp:92-95` · low · bug · lines +3 · effort S · risk low · confirmed

GrowingInputs::perform_input_selection (line 94) and GeneticAlgorithm::evaluate_population (line 192) call `training_strategy->get_optimization_algorithm()->set_display(false)` and never restore it. Scenario: user builds a TrainingStrategy with display on, runs inputs selection, then calls training_strategy.train() for the final model or a later experiment: no epoch output at all, with no indication why. GrowingNeurons does not touch the flag, so the three algorithms also behave differently. Optimizer::get_display exists (optimizer.h:44) and ScopeExit is already used for exactly this kind of restore in cross_validation.cpp:155.

**Fix:** In both sites: `Optimizer* optimizer = training_strategy->get_optimization_algorithm(); const bool saved_display = optimizer->get_display(); optimizer->set_display(false); ScopeExit restore_display([optimizer, saved_display] { optimizer->set_display(saved_display); });` (or do it once inside evaluate_candidate so all three algorithms share it).

*Verifier:* growing_inputs.cpp:94 and genetic_algorithm.cpp:192 call `training_strategy->get_optimization_algorithm()->set_display(false)` with no restore; grep of opennn/model_selection/*.cpp shows growing_neurons.cpp never touches it. Optimizer::get_display exists (optimizer.h:44) and ScopeExit is used for the same restore pattern at cross_validation.cpp:153. Note GA's call sits inside evaluate_population…

#### selection-testing-2-extra-1 — GrowingNeurons JSON round-trip silently drops maximum_epochs

`opennn/model_selection/growing_neurons.cpp:255-288` · low · bug · lines +3 · effort S · risk low · verifier-added

GrowingNeurons::to_JSON writes MinimumNeurons, MaximumNeurons, NeuronsIncrement, TrialsNumber, WarmStart, ValidationErrorGoal, MaximumValidationFailures, MaximumTime and FoldsNumber but not MaximumEpochs, and from_JSON never reads it. GrowingInputs serialises the same knob as "MaximumEpochsNumber" (growing_inputs.cpp:322/336). Scenario: set_maximum_epochs(3); save(); load() on a fresh object -> maximum_epochs is whatever set_default left (1000) or the header default (10); the saved configuration is not what runs. Also makes selection-testing-2 (MaximumEpochs never reported) invisible to any save/load-based test.

**Fix:** Add {"MaximumEpochsNumber", maximum_epochs} to to_JSON (same key as GrowingInputs) and `if (root_element->has("MaximumEpochsNumber")) set_maximum_epochs(read_json_index(root_element, "MaximumEpochsNumber"));` to from_JSON (guarded so existing ND JSON without the key still loads). Add a save/load round-trip assertion in growing_neurons_test.cpp.

*Verifier:* found by verifier

#### dataset-a-16 — set_sample_roles(vector<string>) / set_sample_roles(indices) / set_sample_role write unchecked; JSON path can overrun

`opennn/dataset/dataset.cpp:256-279` · low · bug · lines +4 · effort S · risk low · confirmed

The sibling setters for variables validate (set_variable_roles throws on size mismatch at 572-574; set_variable_role bounds-checks at 608-610) but the sample-role setters index sample_roles blindly. samples_from_JSON (1146-1149) resizes sample_roles to SamplesNumber and then calls set_sample_roles with every token of the "SampleRoles" string: a JSON whose SampleRoles list is longer than SamplesNumber (hand-edited, truncated or produced by an older writer) writes past the end of the vector. set_sample_roles(const vector<Index>&, SampleRole) and set_sample_role(Index, SampleRole) are public entry points with the same gap.

**Fix:** throw_if(ssize(new_roles) != ssize(sample_roles), ...) in the vector<string> overload (mirroring set_variable_roles), and a bounds throw_if in set_sample_role(Index) that the index-vector overload reuses.

*Verifier:* dataset.cpp:256-279 index sample_roles[i] without bounds checks in set_sample_roles(vector<string>), set_sample_roles(indices, role) and set_sample_role (248-254); samples_from_JSON (1143-1151) resizes to SamplesNumber then passes every token of 'SampleRoles' to set_sample_roles, so a longer list overruns. Variable siblings check (572-574, 608-610). Fix is the mirror of those checks, ~4 LOC.

#### dataset-a-18 — Preview rows are joined with "," on write and split on "," on read, so cells containing commas do not round-trip

`opennn/dataset/dataset.cpp:810-877` · low · bug · lines +4 · effort S · risk medium · confirmed

preview_data_to_JSON joins each preview row's cells with "," and preview_data_from_JSON splits the text with get_tokens(text, ","). A semicolon-separated file with decimal commas ("1,5;2,5" -> cells ["1,5","2,5"]) or a comma file with quoted cells that contain commas serialises to "1,5,2,5" and reloads as four cells. The preview is what the UI displays for the file, so the reloaded preview shows the wrong column count.

**Fix:** Write each row as a JSON string array ("Cells": [...]) and read it back as an array, keeping the legacy "Text" read path for old files. Coordinate with Neural Designer since it reads the same JSON.

*Verifier:* dataset.cpp:810-827 joins preview cells with ',' (convert_string_vector(data_file_preview, ",")); 866-877 splits 'Text' with get_tokens(text, ","). Preview rows are produced by tokenising with the file separator (921-946), so any cell containing a comma (decimal-comma semicolon files, quoted fields) does not round-trip. Risk medium is right because ND reads the same JSON.

#### nn-builders-chat-12 — TextClassificationNetwork and 3-arg AutoAssociationNetwork index Shape without rank checks (unchecked operator[])

`opennn/neural_network/standard_networks.cpp:1198-1202` · low · bug · lines +4 · effort S · risk low · confirmed

Shape::operator[] is `noexcept` with no bounds check (tensor_types.h:226) and dims past the rank are zero. TextClassificationNetwork reads input_shape[0], [1], [2] and complexity_dimensions[0]; with a rank-1 or rank-2 input_shape it silently builds Embedding(Shape{vocab, 0}, 0) and the failure surfaces later inside a layer with an unrelated message. AutoAssociationNetwork(const Shape&, const Shape&, const Shape&) reads complexity_dimensions[0] (line 191) with no emptiness check, while its 4-argument sibling validates both shapes (228-231, 237-238) and ImageClassificationNetwork/ResNet check the rank up front (275, 335).

**Fix:** Add `throw_if(input_shape.get_rank() != 3, "TextClassificationNetwork: input shape must be {vocabulary, sequence_length, embedding_dimension}.")` and `throw_if(complexity_dimensions.empty(), ...)` at the top of both constructors, matching the sibling builders.

*Verifier:* tensor_types.h:226 'Index operator[](size_t i) const noexcept { return dims[i]; }' with no bounds check (dim_or_zero at 233 is the checked variant). TextClassificationNetwork 1198-1201 reads input_shape[0..2] and complexity_dimensions[0] with no rank/emptiness check; 3-arg AutoAssociationNetwork reads complexity_dimensions[0] at 191 unchecked while the 4-arg sibling validates at 228-231 and…

#### response-opt-13 — NetworkDifferential disagrees with Scaling on constant features, so the analytic Jacobian is rejected for them

`opennn/response_optimization/network_differential.h:42-48` · low · bug · lines +4 · effort S · risk low · confirmed

Scaling::scale sets a MinimumMaximum column to zero when maximum - minimum < EPSILON (scaling_layer.cpp:41-45) and Unscaling inverts to the constant (79-84). NetworkDifferential::scale_forward instead computes (x - min)/guarded(max - min) = (x - min)/1e-12, producing values of order 1e12 for any probe off the minimum. tests/core/scaling_test.cpp:444-461 pins this mismatch as known. Consequence: any dataset with a constant (or near-constant) input column makes initialize_network_differential's forward probe fail validation, print the warning, and fall back to the finite-difference path of response-opt-1 — silently costing orders of magnitude more network evaluations for every…

**Fix:** Mirror the layer: in scale_forward/scale_derivative return 0 (value and derivative) when max - min < EPSILON, and in unscale_forward return minimum (derivative 0); same EPSILON guard for deviation-based methods. Update the pinned test to assert agreement instead of the mismatch.

*Verifier:* NetworkDifferential::guarded floors at 1e-12 (network_differential.h:42-48) and scale_forward divides by guarded(max-min) (57); scale_column_cpu zeroes a MinimumMaximum column when max-min < EPSILON (scaling_layer.cpp:41-45) and unscale_column_cpu returns the constant (79-84). tests/core/scaling_test.cpp:444-461 pins the mismatch explicitly. The validation in initialize_network_differential…

#### response-opt-6 — Multivariate AllowedSet constraints are silently treated as satisfied outside the branching driver

`opennn/response_optimization/response_constraints.cpp:1186-1193` · low · bug · lines +5 · effort S · risk low · confirmed

set_formula_constraint(expression, allowed_values) stores comparison_operator = AllowedSet. interval_from_comparison returns false for AllowedSet, so constraint_is_satisfied returns true, constraint_residual returns nullopt (never repaired), and build_linear_constraint_set leaves (-inf, +inf). The constraint is only honoured because perform_response_optimization rewrites it to EqualTo per branch (2460-2467). Calling the public solve_once() directly, or any future caller of filter_feasible_points/row_satisfies_formula_constraints, gets an unconstrained answer with no warning. allowed_values is a member of MultivariateConstraint but is read nowhere in the evaluation code.

**Fix:** In constraint_is_satisfied, before the interval check: `if (op == AllowedSet) return ranges::any_of(constraint.allowed_values, [&](float v){ return abs(value - v) <= bound_tolerance(v); });`. Optionally do the same in constraint_residual (snap to nearest allowed value) so repair can handle it.

*Verifier:* interval_from_comparison returns false for AllowedSet (h:99), so constraint_is_satisfied returns true (1189-1191), constraint_residual returns nullopt (1235-1236) and build_linear_constraint_set leaves +-inf (1166-1169). grep allowed_values in response_constraints.cpp returns no match: MultivariateConstraint::allowed_values is never read by the evaluation code. It is only honoured by…

#### dataset-a-17 — variables_missing_values_number changes meaning from per-variable to per-feature-column

`opennn/dataset/tabular_dataset.cpp:2590-2595` · low · bug · lines +6 · effort S · risk low · confirmed

read_csv sizes variables_missing_values_number by variables_number and counts one entry per variable (1904-1905, 2205-2212). calculate_missing_values_statistics (in Neural Designer's alive list) overwrites it with count_nans_per_variable(), which is `data.array().isNaN().cast<Index>().colwise().sum()` -- one entry per one-hot feature column. Scenario: a dataset with one 3-category categorical and one numeric variable: after read_csv the vector has 2 entries; after calculate_missing_values_statistics it has 4, with a missing categorical value counted three times. missing_values_to_JSON writes this vector under the same key either way, so a consumer mapping it onto variables reads misaligned…

**Fix:** Make count_nans_per_variable aggregate per variable: compute the per-column vector, then for each variable take the NaN count of its first feature column (all one-hot columns of a categorical are NaN together, see parse_categorical_token). Keep the per-column version under a per-feature name if anything needs it.

*Verifier:* read_csv sizes variables_missing_values_number by variables_number (1904-1905) and increments per variable (2205-2212); calculate_missing_values_statistics (2590-2595) overwrites it with count_nans_per_variable() = per-column NaN sums (125). Both are written under 'VariablesMissingValuesNumber' (2361) and read back at 2382-2399, where a per-column vector misaligns with variables. Note…

#### xcut-api-4 — HostParametersGuard/HostStatesGuard destructors call throwing uploads (implicit noexcept -> terminate)

`opennn/neural_network/neural_network.h:52-52` · low · bug · lines +6 · effort S · risk low · confirmed

Both guards put parameters/states back on the device from their destructors. copy_parameters_device() throws via throw_if (cpp:2520-2525 'parameters are a non-owning view', 2530-2532 INT8 requires host master) and through the CUDA CHECK macros inside Buffer::migrate_to/copy_async; copy_states_device() likewise migrates with CUDA calls. A destructor is implicitly noexcept, so any of these failures calls std::terminate. The comment on the guard says it exists precisely for the case where a read throws inside the scope; in that case the destructor runs during unwinding and a CUDA error (out of memory after a large expression export, a sticky error from an earlier kernel) turns a catchable…

**Fix:** Wrap each destructor body in try { ... } catch (const exception& e) { cerr << "HostParametersGuard: failed to restore parameters on device: " << e.what() << '\n'; } (same policy as ScopeExit). Alternatively add an explicit release() that callers invoke on the success path so the destructor only has to cope with the unwinding case.

*Verifier:* h:52 ~HostParametersGuard and h:358 ~HostStatesGuard call copy_parameters_device()/copy_states_device() from implicitly-noexcept destructors; copy_parameters_device throws via throw_if at cpp:2520-2525 and 2530-2532 and migrates through Buffer::migrate_to (tensor_types.h:487) which uses CUDA calls; convolution_operator.cpp:669-671 documents exactly this terminate hazard for ScopeExit. Five…

#### training-optimizers-14 — Async next-epoch shuffle makes seeded GPU runs with dropout non-reproducible; the comment claims draw order is unchanged

`opennn/training_strategy/optimizer.cpp:909-934` · low · bug · lines +6 · effort S · risk low · confirmed

The helper thread shuffles via shuffle_vector, which draws from the single mutex-guarded mt19937 in random_utilities.cpp:20-22. On the main thread, every dropout forward on GPU draws a per-call seed from the same generator (dropout_operator.cpp:82: random_integer(0, 1 << 30)), and image augmentation draws on the prefetch workers. The mutex prevents a data race but not interleaving: which thread gets which draw depends on scheduling, so after set_seed(n) two identical GPU runs of a network with dropout produce different batch orders from epoch 1 on, and different dropout masks. The comment asserts the opposite ("The shared RNG draw order is unchanged ... batch composition stays identical to…

**Fix:** Draw the permutation's seed on the main thread before launching the task (`const unsigned shuffle_seed = unsigned(random_integer(0, numeric_limits<int>::max()))`) and give get_batches/shuffle_vector an overload that shuffles with a local mt19937 seeded from it; then the global draw order is genuinely unchanged and the comment becomes true. Otherwise rewrite the comment to state the limitation.

*Verifier:* optimizer.cpp:909-913 comment and 926-931 async get_batches; shuffle_vector (random_utilities.cpp:112-116) and random_integer (38-43) share the single mutex-guarded mt19937 (20-21). dropout_operator.cpp:82 draws random_integer(0, 1<<30) per GPU forward; image_dataset.cpp:41 draws random_uniform on the fill path (prefetch workers). The mutex serialises but does not fix interleaving order, so after…

#### layers-b-5 — rotary_backward / rotary_backward_cpu / rope_backward_gpu are never called by the library (GQA is inference-only)

`opennn/neural_network/layers/grouped_query_attention_layer.cpp:110-160` · low · dead code · lines -90 · effort S · risk medium · partial

GroupedQueryAttentionOperator::back_propagate throws "inference-only" (line 801-804), and the header comment says the layer is the only caller of these helpers. grep across opennn/, examples/ and docs/benchmarks/ finds no caller of rotary_backward or rope_backward_gpu; the only call is tests/neural_network/layers/rope_test.cpp:99. The prior audit listed the "rope fwd/bwd sign twin" as a deferred dedup target; the new evidence is that the backward half is dead, so it should be deleted (rotary_backward_cpu 42 lines, rotary_backward 6, rope_backward_gpu 13, the forward declaration at line 48, the CUDA stub at 687, the header declaration at grouped_query_attention_layer.h:22-23, and the…

**Fix:** Delete rotary_backward, rotary_backward_cpu, rope_backward_gpu (both the CUDA definition and the OPENNN_CUDA_STUB), the static forward declaration, the header declaration, and the rope_test backward case; rope_backward_cuda in core/cuda/kernel_attention.cu(h) becomes deletable in the same PR. Verify against Neural Designer first (it links this library's public API; the symbol is declared in a public header).

*Verifier:* Dead in the library: grep over opennn/, tests/, examples/, docs/ finds rotary_backward called only from tests/neural_network/layers/rope_test.cpp:99; rope_backward_gpu/rotary_backward_cpu are called only by rotary_backward (grouped_query_attention_layer.cpp:158-162); rope_backward_cuda only from rope_backward_gpu (kernel_attention.cu:376, instantiation macro at 859, declaration…

#### dataset-a-15 — ~28 one-line forwarding members defined out-of-line across dataset.cpp / tabular_dataset.cpp

`opennn/dataset/dataset.cpp:222-1015` · low · boilerplate · lines -60 · effort M · risk low · partial

Each costs a 4-6 line out-of-line definition for a body that is a single expression or forwarding call, while dataset.h already defines dozens of equivalent members inline. dataset.cpp: set_data_constant (222), get_samples_number(SampleRole) (227), get_used_samples_number (232), split_samples_random (365), split_samples_sequential (372), set_default_variable_roles (379), set_default_variable_roles_forecasting (438), get_feature_names x2 (451, 456), get_variables_number(role) (523), get_used_variables_number (529), set_variable_role(string) (614), set_variable_type(string) (627), set_variables_number (645), save (911), load (916), has_validation (1012). tabular_dataset.cpp:…

**Fix:** Move these bodies into the class declarations in dataset.h / tabular_dataset.h (the headers already include what they need; statistics.h is already included by tabular_dataset.h). Pure relocation, no behaviour change, no header additions needed.

*Verifier:* Spot-checked the cited definitions (dataset.cpp:222-234, 365-377, 442-458, 495-499, 529-531, 627-630, 645-648, 911-919, 1012-1015; tabular_dataset.cpp:472, 526-529, 864-872, 914-922, 969-977, 1266-1274, 1460-1463): they are one-expression forwarders, and dataset.h already has ~41 inline bodies, so relocation is consistent. Overstated LOC:…

#### core-utils-7 — replace() and replace_all_appearances() are the same function except a hidden '_' rule

`opennn/core/string_utilities.cpp:303-364` · low · duplication · lines -35 · effort M · risk medium · confirmed

Three replace helpers coexist. replace (352-364) and replace_all_appearances (303-331) do the same thing, except the latter silently keeps the original text when the match is preceded by '_' (line 323) -- a rule that is not in its name, header, or the single test that pins it. It produces inconsistent output: process_body_line (model_expression.cpp:411-412) maps "["/"]" to "_", so a variable named x_ yields `x_[0]` -> `x_[0_` (the '[' survives, the ']' does not). The one caller that wants word semantics (model_expression.cpp:458, renaming multi-word variables) should be using replace_all_word_appearances, which checks both boundaries properly.

**Fix:** Point model_expression.cpp:411-412, 1707-1708, 1803-1804 at replace(); point model_expression.cpp:458 at replace_all_word_appearances(); delete replace_all_appearances and its ReplaceAllAppearances test. Verify against Neural Designer before removing the symbol (keep a one-line forwarding alias if it is used there).

*Verifier:* string_utilities.cpp:303-331 replace_all_appearances has the hidden rule at :323 (`buffer.back() == '_' ? to_replace : replace_with`); replace() at :352-364 (declared string_utilities.h:150) is the plain version and is already used by scaling_layer.cpp:522-523, unscaling_layer.cpp:156-157, model_expression.cpp:2047/2122. The x_[0] -> x_[0_ asymmetry at model_expression.cpp:411-412 follows…

#### training-loss-9 — Single-head YOLO CPU entry points duplicate the multi-head path, which only differs by an identity copy

`opennn/training_strategy/loss.cpp:1428-1442` · low · boilerplate · lines -35 · effort S · risk low · confirmed

calculate_yolo keeps four CPU entry points: yolo_error_cpu (322-335) / yolo_gradient_cpu (443-457) for one head and yolo_error_cpu_multi / yolo_gradient_cpu_multi (522-571) for several. The multi path already handles one head; the only reason for the split is that for_each_yolo_head always materialises the head target through assemble_head_target, which for a single head is a full copy of the batch target into a fresh vector. The GPU driver already skips the assembly when `per_sample_floats == head_floats` (line 1121). Applying the same guard on the CPU driver makes the single-head functions and the two `detection_indices.size() > 1` ternaries redundant, and removes the inconsistency that…

**Fix:** In for_each_yolo_head, when per_sample_floats == head_floats pass a TensorView over `tgt` directly (no vector) and otherwise assemble; delete yolo_error_cpu, yolo_gradient_cpu and the two branches in calculate_yolo. yolo_error_kernel/yolo_gradient_kernel stay public for the tests.

*Verifier:* loss.cpp:322-335 yolo_error_cpu, 445-457 yolo_gradient_cpu, 522-571 the _multi pair; for_each_yolo_head (498-520) always calls assemble_head_target (459-470), a full copy for a single head; the GPU driver (1121) skips assembly when per_sample_floats == head_floats. calculate_yolo (1431-1442) dispatches on detection_indices.size() > 1. For a single head, get_output_delta_layer_indices returns…

#### core-utils-8 — Free function tokenize(const string&) has no library caller; tokenize_views is the live variant

`opennn/core/string_utilities.cpp:47-77` · low · dead code · lines -33 · effort S · risk medium · confirmed

opennn::tokenize(const string&) (lowercasing, allocating a string per token) is called only from tests/core/string_utilities_test.cpp:109-126. Every library user goes through tokenize_views (text_generation_dataset.cpp:123, tokenizer_operator.cpp:290/298); the `tokenize(text)` at tokenizer_operator.cpp:164 resolves to the virtual member TokenizerOperator::tokenize. Nothing under docs/benchmarks uses it either. It is 31 lines of a second tokenizer that can drift from the first.

**Fix:** Delete tokenize(const string&) from string_utilities.{h,cpp} and its test, or reduce it to a 3-line wrapper over tokenize_views + ascii_lowercase if the lowercasing form is wanted. Verify against Neural Designer first.

*Verifier:* string_utilities.cpp:47-77 tokenize(const string&) (declared .h:80). Grep across opennn/, tests/, examples/, docs/benchmarks: only tests/core/string_utilities_test.cpp:109-126 call it; tokenizer_operator.cpp:164 `tokenize(text)` inside TokenizerOperator::encode resolves to the virtual member; chat_test.cpp:47 is an override. tokenize_views is the live function. Fix and LOC (-33) fine; Neural…

#### xcut-build-tests-11 — Arch/warning/optimisation flags are set twice (root add_compile_options and opennn target) and partly contradict

`opennn/CMakeLists.txt:241-267` · low · boilerplate · lines -30 · effort S · risk medium · confirmed

The same flags are applied at directory level in the root file and again on the target: `-march=native` (root 19-20 vs opennn 264-266), `/arch:AVX2` (root 26 vs opennn 243, the latter additionally conditioned on non-Debug), `-Wno-interference-size` (root 71-73 vs opennn 182-184), Apple `-Xpreprocessor -fopenmp` (root 59 vs opennn 170), `-Wno-switch-enum` (root 77) vs `-Wswitch-enum` (opennn 187). The opennn target also re-adds `-O3`/`-O0 -g` (251-263) that CMAKE_BUILD_TYPE already sets (the compile DB shows `-O3 ... -O3`), which forces -O3 onto RelWithDebInfo/MinSizeRel. CMake de-duplicates target compile options so the duplicates are harmless today, but every flag change must be made in…

**Fix:** Keep one owner: the opennn target (PUBLIC/BUILD_INTERFACE where consumers in-tree need the same ISA). Delete root lines 19-27, 71-73, 75-80 and the `-O0 -g`/`-O3` lines (251-263); keep `-mno-avx` next to `-march=native` on the target with a one-line comment saying why it exists (it is unexplained today).

*Verifier:* Root: -march=native 19-20, -mno-avx (WIN32) 22, /arch:AVX2 26, Apple -Xpreprocessor -fopenmp 59, -Wno-interference-size 71-73, -Wno-* 75-80. opennn target: -Xpreprocessor -fopenmp 170, -Wno-interference-size 182-184, -Wswitch-enum 187, /arch:AVX2 243, -O0 -g / -O3 251-263, -march=native 264-266. All duplicates verified. Risk medium is right because the root copies currently also reach…

#### dataset-a-19 — calculate_correlations_rank, get_used_variables_indices and both set_variable_type overloads have no callers in the repo

`opennn/dataset/tabular_dataset.cpp:979-987` · low · dead code · lines -30 · effort S · risk medium · confirmed

grep across opennn/, tests/, examples/ and docs/benchmarks/ finds no use of TabularDataset::calculate_correlations_rank (979-987), Dataset::get_used_variables_indices (495-499), Dataset::set_variable_type(Index, ...) (619-625) or set_variable_type(string, ...) (627-630) outside their own definitions/declarations. None of them appears in the prior audit's Neural Designer alive list, but the product was not cross-checked for these four.

**Fix:** Verify against Neural Designer; delete whichever are unused there too (declarations in dataset.h:91,202-203 and tabular_dataset.h:126, bodies in the two .cpp files).

*Verifier:* grep over opennn/, tests/, examples/, docs/benchmarks/ finds calculate_correlations_rank only at tabular_dataset.h:126 and its body (979-987); get_used_variables_indices only at dataset.h:91 and 495-499; set_variable_type only at dataset.h:202-203 and 619-630. None appears in the ENGINEERING_AUDIT alive/do-not-delete lists (lines 84-110), but those lists are not exhaustive, so the ND grep the…

#### dataset-b-5 — Target-cache writer duplicated in try_rebuild_target_from_boxes and build_cache, with a v8 divergence

`opennn/dataset/yolo_dataset.cpp:1390-1427` · low · duplication · lines -30 · effort M · risk low · partial

Lines 1390-1427 and 1637-1664 both: compute target_record_floats, create the .tmp path, open a FileWriter, fill a YoloTargetCacheHeader field by field, write header + anchors, loop over labels calling make_target into a per-sample buffer, finish_with_rename, reopen target_cache_reader, and store target_data_offset. ~60 duplicated lines. The copies already drifted: the rebuild path honours v8_mode (MAX_GT_BOXES*5 records + make_target_v8_gtlist) while build_cache always writes anchor-grid targets and ignores v8_mode, so the on-disk cache produced for the same configuration depends on which path ran.

**Fix:** Extract a private `uint64_t YoloDataset::write_target_cache(const vector<vector<Box>>& labels, const vector<array<float,2>>& cache_anchors)` that sets target_record_floats (one place decides v8 vs grid), writes header/anchors/records, opens target_cache_reader and returns targets_offset; call it from both sites. Pick one v8 behaviour for build_cache deliberately.

*Verifier:* Duplication confirmed: 1391-1427 (try_rebuild_target_from_boxes) and 1637-1666 (build_cache) are the same header-fill/anchors/per-sample make_target/finish_with_rename sequence, ~30 lines each. The v8 divergence is real (only the rebuild path checks v8_mode, build_cache always writes grid targets) but its practical effect is smaller than stated: fill_targets re-encodes from the boxes cache…

#### training-loss-17 — DFL box decode, DFL target computation and reg_max softmax written three, two and three times

`opennn/training_strategy/loss.cpp:741-906` · low · duplication · lines -30 · effort S · risk low · confirmed

The anchor-free decode (`if (reg_max > 1) dfl_decode_box(...) else { pred_cx = (col + out[..0]) * inv_g; ... }`) appears verbatim in tal_assign_head (669-678), yolo_v8_error_kernel_tal (741-750) and yolo_v8_gradient_kernel_tal (852-861). The four-line `d_tgts` clamp block is duplicated in the error (757-767) and gradient (869-879) kernels. The max-subtracted softmax over reg_max is written in dfl_decode (595-603), again in the error kernel (776-380) and again in the gradient kernel (896-906). These are distinct from the parameter preambles the prior audit listed.

**Fix:** Add two file-local helpers: `void decode_pred_box(const float* out, Index reg_max, Index col, Index row, Index G, float box[4])` (absorbing the reg_max==1 branch) and `void dfl_targets(const float* gr, Index col, Index row, Index G, Index reg_max, float d_tgts[4])`; give dfl_decode a variant that also exports the probabilities so the gradient kernel reuses it. ~55 lines become ~25.

*Verifier:* Decode block verbatim at loss.cpp:669-678 (tal_assign_head), 741-750 (error kernel), 852-861 (gradient kernel); d_tgts clamp block at 757-767 and 869-879; max-subtracted softmax at 596-601 (dfl_decode), 776-780 (error kernel; the finding's '776-380' is a typo) and 896-906 (gradient kernel). Helpers as proposed reduce ~55 lines to ~25. Overlaps the restructure suggested in training-loss-3.

#### core-kernels-10 — Launch-scaffolding helper quantified for this scope: ~25 lines over 9 sites, no dispatch_rows_cols needed

`opennn/core/cuda/kernel_attention.cu:313-321` · low · boilerplate · lines -25 · effort S · risk low · confirmed

The prior audit deferred a 'dispatch_rows_cols' helper (~120 lines library-wide). Measured here: of the ~30 explicit <<<>>> launches in these eight .cu files, only four compute a 1-D grid from a count that launch_elementwise already expresses - attention_sdpa_lengths_cuda (318-320), the w8a16 in-major launch (kernel_quantization.cu:135-137), single_output_gradient_finalize (kernel_tensor.cu:327-330) and norm_weight_gradient_finalize (kernel_normalization.cu:914-915); the last three need the count moved to the first kernel parameter. Four more are warp-per-row launches that repeat the same `blocks = ceil(rows / rows_per_block)` arithmetic (launch_masked_softmax_rows 165-167,…

**Fix:** Route the four 1-D launches through launch_elementwise (moving the count to the first kernel parameter where needed); add a 4-line `launch_warp_rows(stream, rows, rows_per_block, threads, kernel, args...)` next to launch_elementwise for the four warp-per-row sites; move one `threads_for_width(int)` ladder to kernel_common.cuh and use it from both files; delete the two local block_size constants. Do not introduce dispatch_rows_cols - the remaining grids are bespoke by design.

*Verifier:* Verified each cited site: attention_sdpa_lengths_cuda 1-D grid at kernel_attention.cu:318-320 (count already first param); w8a16 in-major launch kernel_quantization.cu:135-137 (m first, needs reorder); single_output_gradient_finalize kernel_tensor.cu:327-330 and norm_weight_gradient_finalize kernel_normalization.cu:914-915 (blocks first). Warp-per-row ceil arithmetic repeated at…

#### layers-b-16 — Recurrent and LongShortTermMemory layer wrappers repeat the same shape/return_sequences/JSON plumbing; Recurrent also mirrors Layer::input_shape

`opennn/neural_network/layers/recurrent_layer.cpp:733-790` · low · duplication · lines -25 · effort M · risk medium · partial

set_return_sequences, set_output_shape (rank {1,2}, last dim), apply_input_shape (check_rank {2} + configure), get_output_shape, the ReturnSequences JSON field and the write_expression output-name lambda (step_var / h_name) are line-for-line twins between recurrent_layer.cpp:733-790/792-815 and long_short_term_memory_layer.cpp:1159-1240. Recurrent additionally keeps its own time_steps/input_features members (recurrent_layer.h:187-189) that duplicate Layer::input_shape, while LSTM derives get_time_steps()/get_input_features() from input_shape; so the two siblings store the same state two different ways.

**Fix:** First step (no new class): make Recurrent use Layer::input_shape like LSTM (drop time_steps/input_features members; get_input_shape() falls back to the base). Second step: a small `SequenceLayer : Layer` holding output_features, return_sequences, get_output_shape, set_return_sequences, set_output_shape, apply_input_shape, the ReturnSequences JSON read/write, and the shared output-name lambda, with a pure virtual configure_operators(); both layers derive from it.

*Verifier:* Twins confirmed: set_output_shape (recurrent_layer.cpp:765-770 vs long_short_term_memory_layer.cpp:1219-1224), apply_input_shape (757-763 vs 1212-1217), ReturnSequences read/write (778-789 vs 1234-1244), get_output_shape (recurrent_layer.h:133-137 vs lstm header 166). Recurrent stores time_steps/input_features (recurrent_layer.h:187-189) and overrides get_input_shape (133), whereas LSTM derives…

#### layers-a-12 — Unscaling::write_expression is a hand-rolled twin of Scaling's affine export

`opennn/neural_network/layers/unscaling_layer.cpp:106-160` · low · duplication · lines -25 · effort M · risk low · confirmed

Scaling::write_expression (scaling_layer.cpp:470-526) folds every affine scaler through scaling_affine + affine_line + expression_literal (scaling_layer.cpp:442-466), which the file comment explains exists so the exported text means the same in C, Python and JavaScript. Unscaling::write_expression re-implements the same 6-way switch by hand with inline arithmetic strings (e.g. output=input*(range)+offset written from raw floats), a second copy of the constant-feature guards, and its own '+-'/'--' clean-up. The two bodies differ only in direction; scaling_affine has no inverse form, which is the only reason the twin exists.

**Fix:** Add an inverse flag (or unscaling_affine) next to scaling_affine in core/scaling.h returning {1/scale, -offset/scale} with the degenerate-feature constants ({0, minimum} for MinimumMaximum, {0, mean} for the std-based methods, matching unscale_column_cpu), move expression_literal/affine_line out of the anonymous namespace into a small shared helper that takes the target name, and let both write_expression bodies become a loop over affine_line plus the two non-affine cases…

*Verifier:* Unscaling::write_expression (unscaling_layer.cpp:106-160) re-implements the six-way switch with inline float arithmetic in strings, while Scaling::write_expression (scaling_layer.cpp:470-526) folds every affine case through affine_line/scaling_affine/expression_literal (442-466), with the file comment explaining the C-vs-Python literal hazard. scaling_affine (core/scaling.h:94-131) has no inverse…

#### xcut-build-tests-14 — Four numerical-derivative helpers repeat the same 14-line batch-building preamble

`tests/numerical_derivatives.cpp:43-122` · low · boilerplate · lines -25 · effort S · risk low · confirmed

`calculate_numerical_error`, `calculate_gradient`, `calculate_numerical_gradient` and `calculate_numerical_input_deltas` each re-fetch the network/dataset, null-check them with a per-function message, read the four index vectors, build the Batch, fill it and (in three of four) upload it. The blocks at 45-60, 73-88, 104-119 and 157-171 are identical apart from the message string.

**Fix:** Add a file-local `struct TrainingBatch { Batch batch; Index samples; }; TrainingBatch make_training_batch(Loss&, const char* caller)` that does the checks, fill and GPU upload once; call it from the remaining helpers.

*Verifier:* Preamble repeated at 45-60 (calculate_numerical_error), 73-88 (calculate_gradient), 104-119 (calculate_numerical_gradient), 157-171 (calculate_numerical_input_deltas; this one omits the Decoder indices and the upload). Batch at 58/86/117/170 and upload at 60/88/119 verified by grep. If finding 13 deletes input_deltas, three copies remain; a make_training_batch helper still nets about -25 lines.

#### training-loss-11 — Ten hand-written template CUDA stubs, one of them unreachable

`opennn/training_strategy/error_functions.cpp:87-125` · low · boilerplate · lines -24 · effort S · risk low · confirmed

The non-template stubs were collapsed to the OPENNN_CUDA_STUB X-macro in the prior audit, but ten template stubs (three lines each) remain hand-written because the macro cannot express a template signature. One of them, cross_entropy_3d_multiple_forward_cuda (119-121), is unreachable in a CUDA-less build: its only call (line 391) sits inside `#ifdef OPENNN_HAS_CUDA`. The remaining nine only need to satisfy overload resolution for a call that throws, so a single variadic template per name suffices.

**Fix:** Add `OPENNN_CUDA_TEMPLATE_STUB(name)` next to OPENNN_CUDA_STUB in opennn_types.h expanding to `template<typename... Ts, typename... As> static void name(As&&...) { throw runtime_error(#name " requires CUDA support."); }` (explicit template args bind to Ts, call args deduce As), list the nine names through it, and delete the dead forward stub.

*Verifier:* error_functions.cpp:87-121: ten 3-line template stubs. cross_entropy_3d_multiple_forward_cuda's only call is line 391, inside `#ifdef OPENNN_HAS_CUDA` (383), so the stub at 119-121 is unreachable. OPENNN_CUDA_STUB lives in opennn_types.h:133-134 and cannot take a template head. The proposed `template<typename... Ts, typename... As> static void name(As&&...)` is valid (explicit args bind to the…

#### core-utils-9 — Two Histogram constructors are unused; one re-implements histogram()

`opennn/core/statistics.cpp:151-178` · low · dead code · lines -22 · effort S · risk medium · partial

Histogram(const VectorR& centers, const VectorR& frequencies) and Histogram(const VectorR& data, Index bins) have no callers in opennn/, tests/, examples/ or docs/benchmarks (grep 'Histogram(' outside statistics.* returns nothing). The second is a reduced copy of the free function histogram() (lines 367-448) with different edge semantics (no unique-value path, no minimums/maximums), so it silently returns a different histogram for the same input.

**Fix:** Delete both constructors (statistics.h:66-68, statistics.cpp:151-178); keep Histogram(Index) which statistics.cpp itself uses. Verify against Neural Designer before removal.

*Verifier:* The claim that neither constructor has callers is wrong: tests/core/statistics_test.cpp:376 uses `Histogram histogram(vector, 10)` and checks its centers (:380-384). Histogram(const VectorR&, const VectorR&) (statistics.cpp:151-156) is indeed unused anywhere. The (data, bins) constructor (statistics.cpp:158-178) is a reduced re-implementation of histogram() with different edge semantics, as…

#### layers-a-9 — apply_input_shape overrides and set() bodies repeat the rank check the base already enforces

`opennn/neural_network/layers/activation_layer.cpp:35-40` · low · boilerplate · lines -22 · effort S · risk low · partial

Layer::set_input_shape (layer.h:71-78) validates via accepts_input_rank before calling apply_input_shape, and each layer's accepts_input_rank lists exactly the ranks its check_rank call repeats. Activation::apply_input_shape (activation_layer.cpp:35-40), Concatenation::apply_input_shape (concatenation_layer.cpp:162-166) and Tokenizer::apply_input_shape (tokenizer_layer.cpp:29-34) are therefore identical to the base default plus a dead check; Dense::apply_input_shape:358 and the set()/ctor bodies of Concatenation:151, Tokenizer:26, Flatten:23, Scaling:258 (which runs after the state was already mutated) and Upsampling:135-136 (wrapped in an if(!empty) that check_rank already performs)…

**Fix:** Delete the Activation, Concatenation and Tokenizer apply_input_shape overrides (declaration + definition) and let the base default assign; in their ctors/set() replace 'check_rank(x, {...}, "Name", "input"); input_shape = x;' with set_input_shape(x) (Activation::set already does this). Remove the redundant check_rank in Dense::apply_input_shape and the if(!empty) wrapper in Upsampling::set. Keep check_rank only where a function is reachable without set_input_shape (Dense::set output rank,…

*Verifier:* Duplication confirmed: Layer::set_input_shape (layer.h:71-78) validates via accepts_input_rank before apply_input_shape; Activation::apply_input_shape (activation_layer.cpp:35-40), Concatenation::apply_input_shape (concatenation_layer.cpp:162-166), Tokenizer::apply_input_shape (tokenizer_layer.cpp:29-34) and Dense::apply_input_shape (dense_layer.cpp:356-361) re-run check_rank with the same rank…

#### nn-expression-11 — Activation-body emission loop written four times

`opennn/neural_network/model_expression.cpp:623-631` · low · duplication · lines -22 · effort S · risk low · confirmed

emit_c_activations (623-631), emit_php_activations (1734-1742), emit_python_activations (2082-2090) and the inline loop in emit_js_runtime (1965-1970) are the same 7-line loop over activation_table() differing only in which ActivationBodies member is streamed and which names are emitted unconditionally (C/Python: Identity; JS: Identity and Tanh; PHP: none).

**Fix:** One static `emit_activations(ostringstream&, const string& expression, const char* ActivationBodies::*body, initializer_list<string_view> always)` and four one-line calls; remove the three member declarations from the header.

*Verifier:* Read all four loops: emit_c_activations 623-631 (Identity always), emit_php_activations 1734-1742 (none always), emit_python_activations 2082-2090 (Identity always), emit_js_runtime 1965-1970 (Identity and Tanh always). Identical structure differing only in the ActivationBodies member and the unconditional set. A pointer-to-member helper with an always-list collapses them; LOC estimate fine.

#### r2-batch-pipeline-and-device-gather-11 — Static update_parameters_cuda wrapper + CUDA stub duplicated in Adam and SGD; the Capturable branch already calls the kernel inline

`opennn/training_strategy/adaptive_moment_estimation.cpp:25-61` · low · boilerplate · lines -22 · effort S · risk low · partial

Both optimizers open with a 26-30 line file-static `update_parameters_cuda(BackPropagation&, OptimizerData&, ...)` that only unpacks the same four pointers (parameters, moments/velocity, gradient, bf16 mirror) and forwards to adam_update_cuda / sgd_update_cuda, plus an OPENNN_CUDA_STUB for the CPU build. Inside update_parameters the Capturable branch already does exactly that unpacking inline under `#ifdef OPENNN_HAS_CUDA` (adam 144-167, sgd 92-110), so the file carries two parallel ways to call into kernel_optimizers. The wrapper exists only so the Standard branch can be `#ifdef`-free, which the Capturable branch shows is not needed.

**Fix:** Delete both static wrappers and stubs; in each update_parameters replace `if (neural_network->is_gpu()) return update_parameters_cuda(...)` with the direct kernel call under `#ifdef OPENNN_HAS_CUDA` (with `#else throw runtime_error("... requires CUDA support.")`), reusing the local pointers the Capturable branch already computes (hoist `parameters`, `gradient`, `mirror` pointers above the mode switch so both branches share them).

*Verifier:* The duplication is real: adaptive_moment_estimation.cpp 25-61 and stochastic_gradient_descent.cpp 25-58 are file-static wrappers + OPENNN_CUDA_STUB, while the Capturable branches (adam 144-167, sgd 92-110) call the kernels inline under #ifdef. But the proposed direction (delete the stubs, inline #ifdef blocks) runs against the convention the project just applied — ENGINEERING_AUDIT.md line…

#### r2-batch-pipeline-and-device-gather-7 — Batch::set is only ever called by the constructor; its reset/shrink logic (~20 lines) never executes on a live object

`opennn/dataset/batch.cpp:35-62` · low · dead code · lines -20 · effort S · risk medium · confirmed

grep over opennn/, tests/ and examples/ finds no `batch.set(` / `batch->set(` call: every Batch is built with make_unique<Batch>(...) or a local `Batch batch(...)`, and Batch is non-copyable/non-movable. Consequently the re-entry logic in set() — wait_h2d_complete() at 42, the 14 lines of shape/contiguous/device_gather/view-cache clearing at 47-62, the `if (!on_gpu)` shrink block at 147-153 (resize_bytes(0) on freshly default-constructed buffers), the `else if (!host_bf16_input_cast && input_host_bf16) input_host_bf16.resize_bytes(0)` at 176-177, and the `host_bytes > slot.host.byte_size()` grow guard at 103 — always operate on default-initialised members. The `on_gpu ? type_bytes(type) :…

**Fix:** Fold set() into the constructor body (keep the public `set` declaration only if Neural Designer calls it — verify against Neural Designer; otherwise remove it from batch.h) and delete the reset lines 42, 47-62, the shrink block 147-153 (-> `if (!on_gpu) return;`), lines 176-177, and simplify 89-91 to `type_bytes(type)`. Use resize_bytes instead of grow_to for the pinned host slots since they are always empty at this point.

*Verifier:* Grep over opennn/, tests/, examples/ for '(batch|slot|Batch)...set(' finds only BatchNorm::set calls; Batch::set is invoked solely by the constructor (batch.cpp 27-33) and Batch is non-copyable/non-movable (batch.h 66-69). So wait_h2d_complete (42), the clears at 47-62, the grow guards (103, 170-177) and the !on_gpu shrink block (147-153) always act on default-initialised members; the on_gpu ?…

#### xcut-api-11 — The (batches, input, decoder, target) index tuple is threaded positionally through eight signatures

`opennn/training_strategy/optimizer.h:185-309` · low · design · lines -20 · effort M · risk low · confirmed

Five Optimizer members (warmup_device_training h:185-195, start_batch_prefetch h:220-227, run_graph_epoch h:285-293, train_epoch h:295-302, evaluate_epoch h:304-310), Batch::fill (batch.h:76-80), Dataset::fill_batch and fill_batch_host (dataset.h:292-308) all take three consecutive const vector<Index>& parameters (input, decoder, target feature indices), and TestingAnalysis::get_targets_and_outputs recomputes the same three (testing_analysis.cpp:114-117). Three adjacent same-typed positional parameters mean a swapped decoder/target compiles and only fails at run time on a dataset that has both; each call site spells the triple out on three lines (optimizer.cpp:304-307, 603-606, 1362-1365,…

**Fix:** Add struct FeatureIndices { vector<Index> input, decoder, target; } to dataset.h (next to FillMode), with a Dataset::get_feature_indices_by_role() that fills it once; pass const FeatureIndices& through Batch::fill, Dataset::fill_batch/fill_batch_host and the five Optimizer members. Mechanical; no behaviour change.

*Verifier:* batch.h:76-80 Batch::fill and dataset.h:292-308 fill_batch/fill_batch_host take four consecutive const vector<Index>& (samples + input/decoder/target); optimizer.h has 15 such parameter lines across the five members cited; testing_analysis.cpp:113-116 recomputes the same three indices. Mechanical struct-passing fix is sound and has no behaviour change; LOC -20 is plausible given each call site…

#### core-types-7 — Variable ctor/set take role and scaler as strings, with defaults that contradict the in-class initialisers

`opennn/core/variable.h:118-148` · low · design · lines -18 · effort S · risk low · partial

The in-class initialisers say `type = VariableType::None` and `scaler = ScalerMethod::None`, but they are never effective: the constructor always calls set(), whose string defaults make every Variable Numeric + MeanStandardDeviation after two linear EnumMap lookups and two temporary std::strings. A reader of the header is told the wrong defaults. Callers confirm the strings buy nothing: the only constructor use in the repo is `Variable()` (eight `variables.assign(n, Variable())` sites across datasets and neural_network.cpp) and the only set() uses are four identical `variable.set(variable.name, "None", VariableType::Constant)` calls in tabular_dataset.cpp (457, 466, 2302, 2319). Three…

**Fix:** Make the initialisers the truth (`type = VariableType::Numeric; scaler = ScalerMethod::MeanStandardDeviation;`), `Variable() = default;`, and replace the string-taking ctor/set with `void set(string name, VariableRole, VariableType, ScalerMethod = ScalerMethod::MeanStandardDeviation, vector<string> categories = {})`; update the four tabular_dataset.cpp calls to `set(variable.name, VariableRole::None, VariableType::Constant)`. Move set_type/get_type_string/get_categories_number inline into the…

*Verifier:* Verified: variable.h:118-130 string-taking ctor/set with defaults "None"/Numeric/"MeanStandardDeviation"; in-class initialisers :134 `type = VariableType::None`, :136 `scaler = ScalerMethod::None` are overwritten by set() in the ctor (variable.cpp:16-28), so they misdescribe the defaults. Only set() callers are tabular_dataset.cpp:457, 466, 2302, 2319, all `set(variable.name, "None",…

#### operators-b-11 — heads_input/heads_output are pure forwarding wrappers and both SDPA backward exits repeat the same cast-back epilogue

`opennn/neural_network/operators/attention_operator.cpp:276-292` · low · boilerplate · lines -18 · effort S · risk low · confirmed

heads_input and heads_output (12 lines) do nothing but call cudnn_frontend::bhsd_tensor / set_bhsd_output with the same arguments; the ten call sites could call the utilities directly. apply_sdpa_backward ends the flash branch (1263-1268) and the graph branch (1304-1309) with the identical three cast_bf16_to_fp32 calls, and apply_sdpa_forward repeats its single cast in both branches (973-974, 1009-1010). flash_attention_problem is also declared `static` inside an anonymous namespace.

**Fix:** Delete the two wrappers and call cudnn_frontend::bhsd_tensor/set_bhsd_output directly; in apply_sdpa_backward/forward, make the flash branch fall through to a single shared cast-back block (if/else on the problem instead of early return) so the epilogue exists once; drop the redundant `static`.

*Verifier:* attention_operator.cpp:280-291: heads_input/heads_output forward verbatim to cudnn_frontend::bhsd_tensor/set_bhsd_output; ten call sites (373-375, 397, 420-423, 429, 456-458). flash_attention_problem at 298-299 is `static` inside the anonymous namespace opened at 270. apply_sdpa_backward repeats the three cast_bf16_to_fp32 calls at 1263-1268 and 1304-1309; apply_sdpa_forward repeats its single…

#### r2-duplicated-kernels-across-folders-6 — C2PSA copy_right and scatter_dx kernels are strided column copies slice_channels/cudaMemcpy2DAsync already express

`opennn/neural_network/operators/kernel_c2psa.cu:13-67` · low · duplication · lines -18 · effort S · risk medium · partial

Beyond the softmax pair already flagged, kernel_c2psa.cu keeps two hand-written copy kernels. c2psa_copy_right_kernel copies columns [H, C) of a (BT, C) matrix into the same columns of another (BT, C) matrix: a pitched device-to-device copy (pitch C*sizeof(T), width (C-H)*sizeof(T), height BT) that cudaMemcpy2DAsync performs in one call. c2psa_scatter_dx_kernel builds din = [d_xa | d_cat[:, H:C]]: its left half is precisely slice_channels_cuda<T, true>(BT, 1, 1, H, C, 0, d_xa, din), the helper this very file already calls at line 94 for the forward, and its right half is the same pitched copy. Both kernels also do their index math in int (`row * C + H + col`), unlike the rest of the scope…

**Fix:** Delete both kernels. c2psa_split_cuda: gather_left as today plus `CHECK_CUDA(cudaMemcpy2DAsync(cat + H, C*sizeof(T), x + H, C*sizeof(T), (C - H)*sizeof(T), BT, cudaMemcpyDeviceToDevice, stream))`. c2psa_scatter_dx_cuda: `slice_channels_cuda<T, true>(BT, 1, 1, H, C, 0, d_xa, din)` plus the same pitched copy from d_cat. Put the pitched copy behind a small device::copy_2d_async helper so the raw call lives in device_backend.cpp. Verify with tests/neural_network/layers/c2psa_test.cpp and the YOLO…

*Verifier:* Both kernels verified at kernel_c2psa.cu:14-26 and :52-66, int index math confirmed, and :94 (c2psa_fill_cat_left_cuda) already uses slice_channels_cuda<T,true>(BT,1,1,H,C,0,...) for the mirror scatter; concat_slice_kernel (kernel_concat.cu:12-28) with H=W=1 is exactly the (BT,H)->(BT,C) column scatter, so the scatter_dx left half is correctly expressed by it. One correction that changes the fix:…

#### xcut-boilerplate-11 — Residual hand-written CUDA stubs left after the OPENNN_CUDA_STUB pass (3 sites)

`opennn/training_strategy/error_functions.cpp:87-125` · low · boilerplate · lines -18 · effort S · risk low · confirmed

Pattern (d), .cpp side. Today there are 185 OPENNN_HAS_CUDA blocks (147 in .cpp/.cu, 16 in .h, 22 in .cuh): 30 have an #else branch that throws, 23 have a non-throwing #else, 106 have no #else (mostly 1-2-line kernel-include guards or `if (x.is_cuda()) return *_cuda(...)` dispatch guards that cannot be removed without declaring the kernel in CPU builds). 27 of the 30 throwing branches already use OPENNN_CUDA_STUB/_BODY; three still hand-write the body: error_functions.cpp:87-125 (ten function templates, 3 lines each - OPENNN_CUDA_STUB_BODY applies to templates as well), grouped_query_attention_layer.cpp:677-682 (qk_rope_cache_append, 6 lines, with a different message "CUDA support not…

**Fix:** Replace the ten template bodies with `OPENNN_CUDA_STUB_BODY(name)` on the signature line (one line each), and convert the GQA and tensor_types stubs to OPENNN_CUDA_STUB so the message is the harmonised "requires CUDA support." Build the CPU dir only.

*Verifier:* error_functions.cpp:87-125 holds ten template stubs, each `template<...> static void f(...)` + `{ throw runtime_error("f requires CUDA support."); }` (3 lines with blank); grouped_query_attention_layer.cpp:677-682 hand-writes qk_rope_cache_append with the divergent message 'CUDA support not compiled in.'; tensor_types.cpp:71-74 fill_cuda likewise. docs/ENGINEERING_AUDIT.md:143-145 marks the stub…

#### training-optimizers-12 — SGD/QN/LM set_default() re-state defaults that member initializers express (or should)

`opennn/training_strategy/stochastic_gradient_descent.cpp:67-82` · low · boilerplate · lines -18 · effort S · risk medium · confirmed

SGD declares `float initial_learning_rate; float initial_decay;` uninitialised in the header and assigns them in set_default together with momentum/nesterov/batch_size that already have initializers. QN's set_default re-assigns minimum_loss_decrease (already 1.0e-6f in the header), training_loss_goal = 0, display = true and display_period = 10 (all base defaults). LM initialises all five damping/loss-decrease members to 0.0f in the header only to overwrite them in set_default. Only the base overrides (maximum_epochs, maximum_time, maximum_validation_failures, display_period) genuinely need a constructor statement. Adam already follows the lean pattern (3 lines).

**Fix:** Move the per-class constants to default member initializers in the three headers (initial_learning_rate = 0.001f, initial_decay = 0.001f, initial_damping_parameter = 1.0e-3f, damping_parameter_factor = 10.0f, minimum/maximum_damping_parameter = 1.0e-6f/1.0e6f) and keep in set_default only name plus the base overrides. Keep set_default() itself (public; verify against Neural Designer before removing it).

*Verifier:* stochastic_gradient_descent.h:51,53 declare initial_learning_rate/initial_decay without initializers; set_default (cpp 67-82) assigns them plus momentum/nesterov/batch_size already defaulted in the header (h:55,57; optimizer.h:325). quasi_newton_method.cpp:25-38 re-assigns minimum_loss_decrease (header :57 = 1.0e-6f), training_loss_goal = 0 (optimizer.h:314), display = true (optimizer.h:332),…

#### selection-testing-10 — evaluate_candidate skips on_trial on the fold path, so GrowingInputs and GrowingNeurons each carry a second record-optimum block

`opennn/model_selection/selection_utilities.cpp:21-59` · low · duplication · lines -16 · effort M · risk medium · confirmed

When folds_number > 1 evaluate_candidate returns before calling on_trial, so both callers duplicate the callback's bookkeeping in an `if (folds_number > 1)` block: growing_inputs.cpp:211-225 repeats lines 185-193 (optimal indices/names/errors, parameters replaced by VectorR()), growing_neurons.cpp:177-193 repeats lines 150-158 (history entries, optimal_neurons_number, errors). GeneticAlgorithm has the same pattern at 220-224. That is ~30 duplicated lines across three files whose only variation is `optimal_parameters = VectorR()` versus `get_parameters_map()`.

**Fix:** Have the fold path invoke `on_trial(0, training_error, validation_error, /*improved*/ true)` before returning, and extend the callback signature with a `bool parameters_valid` (false on the fold path, true otherwise); callers assign `optimal_parameters = parameters_valid ? VectorR(get_parameters_map()) : VectorR()` and skip the warm-start snapshot when !parameters_valid. Then delete the three `if (folds_number > 1)` record blocks (keep only the display line).

*Verifier:* selection_utilities.cpp:33-40: the fold path returns before on_trial. Duplicated record blocks verified: growing_inputs.cpp:211-225 vs 185-193, growing_neurons.cpp:177-193 vs 150-158, genetic_algorithm.cpp:220-224 vs 210-216; variation is only VectorR() vs get_parameters_map() and the warm-start snapshot capture. The proposed `parameters_valid` flag is workable; GrowingNeurons' callback must also…

#### xcut-build-tests-20 — examples/CMakeLists.txt: re-declared option, double gating, phantom subdirectory, duplicated example list

`examples/CMakeLists.txt:2-48` · low · boilerplate · lines -15 · effort S · risk low · confirmed

Lines 2-4 re-declare `OpenNN_BUILD_EXAMPLES`, which the root already declared and already used to decide whether to `add_subdirectory(examples)`; line 24 gates everything on it again. Lines 46-48 test for `beijing_pm25_forecasting/CMakeLists.txt`, a directory that does not exist. `examples_with_data` (29-32) repeats most of the foreach list (34-38) by hand, so adding an example with data means editing two lists; the presence of a `data/` directory is already the fact being encoded.

**Fix:** Drop lines 2-4, 24, 46-48 and `examples_with_data`; inside `opennn_example` do `if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/${name}/data")` to decide the copy step, and call `opennn_example(${example})` for every directory in one list.

*Verifier:* examples/CMakeLists.txt:2-4 re-declare OpenNN_BUILD_EXAMPLES (root line 166 already declared and gated add_subdirectory), 24 gates again, 29-32 examples_with_data duplicates the 34-38 list, 46-48 test for beijing_pm25_forecasting which does not exist (ls examples). `ls -d examples/*/data` returns exactly the ten directories in examples_with_data, so the IS_DIRECTORY rule reproduces today's…

#### core-utils-13 — descriptives(matrix, rows, cols) splits one per-column job into two OMP loops and five scratch vectors

`opennn/core/statistics.cpp:531-583` · low · boilerplate · lines -15 · effort S · risk low · confirmed

The per-column computation is fully independent, yet the function runs one parallel loop that scatters minimum/maximum/sum/squared_sum/count into five temporaries, computes means as a vector expression, then runs a second parallel loop to recombine them. It also disagrees with vector_descriptives on an all-NaN column: here mean = 0/0 = NaN (sums/count with count 0) while vector_descriptives (line 513-514) returns zeros.

**Fix:** Single `#pragma omp parallel for` over columns: compute masked_moments, then write descriptives_results[j] directly (min, max, mean = count ? sum/count : 0, sd via the shared sample_variance helper from core-utils-12). Delete the five temporaries and the second loop.

*Verifier:* statistics.cpp:531-583 as quoted: five scratch vectors, a first `#pragma omp parallel for` filling them from masked_moments, a vector-expression `mean = sums/count` (0/0 = NaN for an all-NaN column, whereas vector_descriptives :513-514 returns zeros), then a second `#pragma omp parallel for` recombining. Per-column work is independent so a single loop writing descriptives_results[j] is…

#### xcut-boilerplate-9 — OpenMP first-exception capture block copied four times

`opennn/dataset/image_dataset.cpp:618-627` · low · duplication · lines -15 · effort S · risk low · partial

Pattern (i). The same 9-line idiom - catch inside an `omp for`, `#pragma omp critical { if (omp_error.empty()) omp_error = e.what(); }`, then `throw_if(!omp_error.empty(), omp_error)` after the region - is copied at io_utilities.cpp:421-433, image_dataset.cpp:618-627, yolo_dataset.cpp:2005-2014 and yolo_dataset.cpp:2173-2182 (8 `#pragma omp critical` sites in total, 4 of them this idiom). Each copy also repeats the `sample_index < 0 || sample_index >= samples_number` range check (io_utilities.cpp:413, yolo_dataset.cpp:1943, 2045, 2079).

**Fix:** Add a small `struct OmpFirstError { string message; void capture(const exception&) /* omp critical inside */; void rethrow() const { throw_if(!message.empty(), message); } };` in core (next to the OpenMP helpers) and use it at the four sites; fold the range check into a `check_sample_index(Index, Index, string_view context)` used by all four loops.

*Verifier:* Idiom confirmed at image_dataset.cpp:618-627, core/io_utilities.cpp:423-433 (the file is opennn/core/io_utilities.cpp, not dataset/), yolo_dataset.cpp:2005-2014 and 2173-2182; the other four omp critical sites (tabular_dataset.cpp:2138/2170/2182/2817) are different. Correction on the range check: it appears at core/io_utilities.cpp:413 and yolo_dataset.cpp:1943/2045/2079, but not inside all four…

#### r2-arena-planner-and-propagation-structs-5 — ForwardPropagation builds the consumer-edge map a fourth and fifth time; hoist it to NeuralNetwork

`opennn/neural_network/forward_propagation.cpp:43-50` · low · duplication · lines -15 · effort S · risk low · unverified

find_early_output_release_steps (lines 43-50) and the inference branch's last_consumers/has_consumers loop (lines 621-648, through resolve_producer) each rebuild `for each consumer, for each input: edges[source].push_back({consumer, input})` from source_layers, the same map BackPropagation::make_consumer_edges builds (the existing finding at back_propagation.cpp:60 counts three builds there). The map is a pure function of the compiled graph, so one cached `NeuralNetwork::get_consumer_edges()` filled in compile() serves all five sites and removes the hand loops here and in BP. This complements, not duplicates, the BP finding: the fix location is NeuralNetwork and the extra copies are in this…

**Fix:** Add `const vector<vector<pair<size_t,size_t>>>& NeuralNetwork::get_consumer_edges() const` computed once in compile() beside source_layers; use it in find_early_output_release_steps, in the inference last_consumers loop (keeping the resolve_producer step), and in BackPropagation::make_consumer_edges (which becomes a one-line accessor). Net removal of two hand-written map builders.

#### response-opt-15 — Header exports nine repair_* entry points with colliding names; four are internal-only

`opennn/response_optimization/response_constraints.h:168-211` · low · API · lines -15 · effort S · risk medium · confirmed

repair_affine_inputs (batch LDLT projection with slacks) and repair_affine_inputs_with_fixed (per-row Gauss-Newton over affine AND nonlinear constraints) are different algorithms with names that differ by a suffix; repair_nonlinear_inputs is a 10-line wrapper that just forwards to repair_affine_inputs_with_fixed; repair_single_affine_input/_integer are two one-line wrappers around a file-local bool flag. Of the nine, repair_affine_inputs, repair_nonlinear_inputs, repair_single_affine_input and repair_affine_inputs_with_fixed are not referenced outside response_constraints.cpp except one test call each for the latter two (grep over opennn/, tests/, examples/, docs/benchmarks). compile_ast…

**Fix:** Keep the public surface to what response_optimization.cpp and the tests use (repair_inputs, repair_mixed_integer_inputs, repair_output_constraints x2, repair_single_affine_integer, repair_affine_inputs_with_fixed, snap_to_lattice); move the rest plus compile_ast/parse_to_ast into the anonymous namespace and rename repair_affine_inputs_with_fixed to repair_inputs_gauss_newton. Verify against Neural Designer before removing declarations.

*Verifier:* Header 168-233 declares nine repair_* entries. grep over opennn/, tests/, examples/, docs/benchmarks: repair_affine_inputs, repair_nonlinear_inputs and repair_single_affine_input are referenced only inside response_constraints.cpp (1740-1745); repair_affine_inputs_with_fixed only at tests/...:992, repair_single_affine_integer at tests/...:1046,1075; repair_inputs at response_optimization.cpp:1057…

#### response-opt-10 — Finite-difference Jacobian written twice; the validation copy issues 2n single-row forwards per probe

`opennn/response_optimization/response_optimization.cpp:2186-2201` · low · duplication · lines -15 · effort S · risk low · confirmed

initialize_network_differential hand-rolls central differences (2188-2197) with one `calculate_outputs(plus)` and one `calculate_outputs(minus)` per input feature (each constructing a ForwardPropagation), while repair_output_constraints already has the same computation batched into a single 2n-row call (response_constraints.cpp:1934-1955) with the same step rule `max(1e-4f, 1e-3f * range)` (2169 vs 1925). For the 31-input portfolio tests this is 4*31*2 = 248 single-row network calls at initialisation instead of 4.

**Fix:** Add one free function in response_constraints.{h,cpp}: `MatrixR finite_difference_jacobian(const SurrogateBatchForward&, const VectorR& x, const VectorR& step)` (returns n_out x n_in from one batched call) and `VectorR finite_difference_step(const VectorR& lower, const VectorR& upper)`. Use it in both places; the validation then does `J.transpose() * cotangent`. Combine with response-opt-1 step 1 if done together.

*Verifier:* initialize_network_differential (2186-2197) does one calculate_outputs(plus) and one calculate_outputs(minus) per input feature for vjp_probes=4 probes (2157, 2183), each a 1-row NeuralNetwork::calculate_outputs constructing a ForwardPropagation; repair_output_constraints' FD vjp (response_constraints.cpp:1937-1958) already batches the same central difference in one 2n-row call with the identical…

#### r2-batch-pipeline-and-device-gather-12 — adam_update_cuda/sgd_update_cuda declared with runs of unnamed positional floats; each has a near-identical capturable twin

`opennn/training_strategy/kernel_optimizers.cuh:9-18` · low · API · lines -15 · effort S · risk low · confirmed

kernel_optimizers.cuh declares `adam_update_cuda(const Index, float*, float*, float*, const float*, const float, const float, const float, const float, const float, const float, ...)` — six consecutive unnamed `const float` (beta_1, beta_2, learning_rate, epsilon, bias_correction_1, bias_correction_2): any transposition at the call site (adaptive_moment_estimation.cpp:40-52) compiles silently and only shows up as a slightly wrong optimizer. In kernel_optimizers.cu the non-capturable and capturable host wrappers are twins: adam_update_cuda (110-131) and adam_update_capturable_cuda (161-181) both compute `aligned` and call launch_vec_on<4>(…, adam_update_kernel, …) with the same argument…

**Fix:** Name every parameter in the header. Collapse each pair into one wrapper: `adam_update_cuda(n, parameters, m, v, gradients, beta_1, beta_2, lr_scalar, eps_scalar, const float* lr_device, const float* eps_device, mirror, stream)` where the Standard caller passes host-corrected scalars and null device pointers and the Capturable caller runs adam_prepare_kernel first (or keep the prepare launch in a tiny `adam_prepare_cuda`). Same for SGD with `lr_scalar / lr_device`.

*Verifier:* kernel_optimizers.cuh 9-16: adam_update_cuda has six consecutive unnamed const float and sgd_update_cuda two, while the capturable twins are fully named. kernel_optimizers.cu: adam_update_cuda (110-135) and adam_update_capturable_cuda (161-181) both compute aligned and call launch_vec_on<4>(.., adam_update_kernel, ..) differing only in scalar-vs-device lr/eps and the adam_prepare_kernel launch;…

#### xcut-build-tests-24 — Every example/test re-implements 'dynamic_cast the optimizer and hope': a typed accessor would remove 13 casts and 8 unchecked derefs

`opennn/training_strategy/training_strategy.h:33-34` · low · API · lines -15 · effort S · risk low · confirmed

`get_optimization_algorithm()` returns `Optimizer*`, so callers write `dynamic_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm())` (13 occurrences across examples, tests and benchmarks; 44 `get_optimization_algorithm()` uses overall). Eight examples dereference the result without a null check (airfoil 55-56, amazon_reviews 50-51, breast_cancer 43-44, emotion_analysis 48-49, melanoma_cancer 41-42, mnist 51-52, iris_plant 41-42, bert 79-80); translation, ecg5000 and blank_cuda add their own three-line `if(!adam) throw`. A mismatch (e.g. `set_optimization_algorithm("StochasticGradientDescent")` followed by an Adam cast) is a null deref in the first group and a…

**Fix:** Add to TrainingStrategy: `template<class T> T& get_optimizer() { T* p = dynamic_cast<T*>(optimizer.get()); throw_if(!p, "TrainingStrategy: optimizer is not a {}.", T::static_name()); return *p; }` (or pass the name string) and use `auto& adam = training_strategy.get_optimizer<AdaptiveMomentEstimation>();` in the examples and tests; keep `get_optimization_algorithm()` for Neural Designer.

*Verifier:* training_strategy.h:33-34 return Optimizer*. grep: 13 `dynamic_cast<...*>(...get_optimization_algorithm())` in examples/tests/docs, 49 get_optimization_algorithm() uses overall (audit said 44; immaterial), only 3 of the example casts are followed by a null check. airfoil:55-56 derefs unchecked as quoted. Optimizer exposes get_name() (optimizer.h:88), so the template accessor can use a name string…

#### xcut-api-14 — to_cudnn / to_cuda / type_bytes are three copies of the same switch over TypeInfo

`opennn/core/tensor_types.h:74-110` · low · boilerplate · lines -14 · effort S · risk low · confirmed

Three inline functions each write the identical four-case switch (FP32, BF16, INT8, Auto->break) followed by the same 'Type::Auto must be resolved' throw, differing only in which TypeInfo member they return. The file already has the generic visit_type dispatcher just above them; a sibling that hands the TypeInfo specialization to a callable makes each of the three a one-liner and any future Type value gets added in one place.

**Fix:** Add template<typename F> auto with_type_info(Type type, F f) { switch (type) { case FP32: return f(TypeInfo<Type::FP32>{}); case BF16: ...; case INT8: ...; case Auto: break; } throw runtime_error("Type::Auto must be resolved before tensor use."); } and define to_cudnn/to_cuda/type_bytes as return with_type_info(type, [](auto info){ return info.cudnn; }); etc. Header-only, no ABI concern; the CUDA-less build already has the stub typedefs TypeInfo uses.

*Verifier:* tensor_types.h:74-110: three 11-line functions with the identical switch and a per-name throw message; visit_type sits directly above (line 62). The proposed with_type_info dispatcher is header-only and the CUDA-less stubs already exist. LOC: ~33 lines replaced by a ~10-line helper plus three 1-2 line bodies (~9 lines), net about -14 rather than -20, within tolerance.

#### dataset-a-14 — Separator uses a hand-rolled tuple table with four linear searches; check_separators allocates two strings per CSV line

`opennn/dataset/dataset.cpp:659-767` · low · boilerplate · lines -14 · effort S · risk low · confirmed

separator_map is a vector<tuple<Separator,string,string>> searched linearly by four functions (get_separator_string, get_separator_name, set_separator_string, set_separator_name), each 6-8 lines, while the sibling enums in the same file use EnumMap. get_separator_string/get_separator_name return std::string by value, and Dataset::check_separators (947-951) calls both at the top of every line validation, so CsvReader::parse performs two heap-allocating string constructions per line of the file (millions for large CSVs) before the actual scan, plus the inner table loop per character (979-991).

**Fix:** Two `EnumMap<Separator>` instances (separator_string_map, separator_name_map) and four one-line bodies; return `const string&` from the getters (source-compatible for Neural Designer callers that take a string). Precompute sep_char/name once in read_csv and pass them into the validator lambda instead of recomputing per line.

*Verifier:* dataset.cpp:659-678: separator_map is a vector<tuple> with linear-search getters returning string by value; check_separators (947-951) calls get_separator_string() and get_separator_name() on every line, and read_csv installs it as the CsvReader line_validator (1625-1630) so it runs per line. EnumMap (core/enum_map.h:36-43) offers to_string returning const string& and from_string. Fix sound, -14…

#### nn-builders-chat-10 — Non-incremental GenerationParser re-decodes the whole channel per token (O(n^2) per response)

`opennn/neural_network/chat.cpp:446-459` · low · overhead · lines -14 · effort S · risk low · confirmed

For tokenizers whose supports_incremental_decode() is false — only the base TokenizerOperator, whose decode is 'join id_to_token(id) with spaces' (tokenizer_operator.cpp:218-233) — append_data_token pushes the id and emit_stable_delta calls tokenizer->decode(state.ids) over the entire channel, then complete_utf8_prefix_size and starts_with over the whole string, for every generated token. The classic sessions (Transformer / TextGenerationNetwork) always take this path. For a 512-token response that is ~130k id_to_token lookups and string appends plus ~130k bytes of repeated UTF-8 scanning, and the 'prefix-stable' throw_if at 451 is only there to guard this re-decode.

**Fix:** Give the base TokenizerOperator an incremental decode consistent with its decode(): decode_token(id) returns id_to_token(id) and the parser inserts the ' ' separator when the channel text is non-empty (or have the base class return `supports_incremental_decode() = true` with a `needs_separator` flag). Then delete the non-incremental branch, the ChannelState::ids member and the `incremental` flag from GenerationParser.

*Verifier:* chat.cpp:446-459 re-decodes tokenizer->decode(state.ids) over the whole channel per token when !incremental; incremental is tokenizer.supports_incremental_decode() (319), which is false for the base class (tokenizer_operator.h:68) and true only for BytePairTokenizer (161). WordLevel/WordPiece do not override decode, so classic sessions use the base decode (tokenizer_operator.cpp:218-233: skip id…

#### nn-builders-chat-9 — ResNet and YoloNetwork each define the same add_conv lambda

`opennn/neural_network/standard_networks.cpp:342-351` · low · duplication · lines -14 · effort S · risk low · confirmed

ResNet::ResNet (342-351) and YoloNetwork::YoloNetwork (504-513) both define a lambda that builds a Convolutional from get_layer(input)->get_output_shape() with "Same" padding, adds it with {input_index} and returns get_layers_number() - 1; the only difference is that ResNet hard-codes batch_norm = true. ImageClassificationNetwork (288-294) and the DarknetTinyV3/Vgg backbone loops (607-610, 650-653) write the same call out by hand a further 4 times.

**Fix:** Hoist a file-local `static Index add_conv(NeuralNetwork&, Index input_index, const Shape& kernel, const char* activation, const Shape& stride, bool batch_norm, const string& label)` next to add_dense_stack and use it from ResNet, YoloNetwork (or the YoloBuilder of -6), ImageClassificationNetwork and the two backbone loops.

*Verifier:* ResNet add_conv at 342-351 hard-codes batch_norm=true; YoloNetwork add_conv 504-513 takes bool batch_norm; otherwise identical (get_layer(input)->get_output_shape(), 'Same', {input_index}, return get_layers_number()-1). ImageClassificationNetwork 288-294 and the DarknetTinyV3 loop 650-654 write the same Convolutional construction by hand. A file-local static helper next to add_dense_stack fits…

#### nn-builders-chat-15 — Seven trivial constructors defined out-of-line (Transformer, TextGenerationNetwork, Qwen3, Bert, BertForSequenceClassification)

`opennn/neural_network/standard_networks.cpp:1269-1272` · low · boilerplate · lines -14 · effort S · risk low · confirmed

Transformer() (1269-1272), Transformer(const path&) (1760-1763), TextGenerationNetwork() (1446-1449), TextGenerationNetwork(const path&) (1797-1800), Qwen3() (1628-1631), Bert() (1609-1612) and BertForSequenceClassification() (1857-1860) are each a 4-line empty body whose only content is the NetworkTask passed to the base; they are scattered across the file (the path ctors sit 490 lines after their default siblings), so a reader has to grep to learn what task tag a family uses. The header already shows the signatures.

**Fix:** Define them inline in standard_networks.h (`Transformer() : NeuralNetwork(NetworkTask::LanguageModeling) {}` and `explicit Transformer(const filesystem::path& path) : NeuralNetwork(path, NetworkTask::LanguageModeling) {}`), which also keeps each family's task tag visible next to its declaration.

*Verifier:* Verified Transformer() 1269-1272, Transformer(path) 1760-1763, TextGenerationNetwork() 1446, TextGenerationNetwork(path) 1797, Bert() 1609, Qwen3() 1628, BertForSequenceClassification() 1857 are 4-line empty bodies forwarding a NetworkTask; header declarations at standard_networks.h:147,158,179,192,209,227,242. Inlining in the header removes 28 lines, adds 7: -21, within range of -14.

#### operators-a-11 — ConvolutionOperator::set takes ten positional Index arguments, has one caller, and only copies public fields

`opennn/neural_network/operators/convolution_operator.h:63-67` · low · API · lines -13 · effort S · risk medium · confirmed

The signature `set(Index, Index, Index, Index, Index, Index, Index, Index, Index, Index, Type)` is declared without parameter names, so the single caller (convolutional_layer.cpp:183-187) relies on argument order for height/width, kernel h/w/channels/number, strides and paddings; swapping any adjacent pair compiles and produces a wrong but plausible convolution. Every field it assigns is already a public data member (the header exposes them and the layer reads them directly elsewhere), so the function adds 24 lines and a hazard without adding an invariant.

**Fix:** Delete ConvolutionOperator::set and have Convolutional::update_convolution_operator assign the named public fields directly (`convolution.kernel_height = kernel_height;` etc.). If a single entry point is wanted, replace the positional list by a small aggregate `ConvolutionGeometry` struct with designated initializers. Verify Neural Designer does not call ConvolutionOperator::set directly before removing.

*Verifier:* convolution_operator.h:63-67 declares set with eleven unnamed positional arguments; convolution_operator.cpp:266-284 only copies them into public members (the struct has no access specifiers, and compute_dtype is a public Operator member at operator.h:66). Single caller convolutional_layer.cpp:183-187. grep of opennn/, tests/, examples/, docs/ finds no other caller, and a grep over C:\Artelnics…

#### xcut-build-tests-12 — GCC-only raw `gomp pthread` OpenMP linking forces an export special-case and leaves consumer pragmas serial

`opennn/CMakeLists.txt:169-180` · low · boilerplate · lines -12 · effort S · risk medium · confirmed

For GNU the target links `gomp pthread` by bare name and adds `-fopenmp` as a link option, while Clang/MSVC use `OpenMP::OpenMP_CXX`. This forces `_opennn_needs_openmp` (328-331) and the conditional `find_dependency(OpenMP)` in OpenNNConfig.cmake.in:9-11. Because the compile flag is PRIVATE on the GNU path, any in-tree consumer with `#pragma omp` (docs/benchmarks/capacity/higgs-max-batch/opennn_higgs_maxbatch_trial.cpp:297, the benchmark machine is Linux/GCC) is compiled without -fopenmp and runs that loop serially, whereas the same file built with Clang/MSVC gets the flag through the imported target.

**Fix:** `target_link_libraries(opennn PUBLIC OpenMP::OpenMP_CXX)` for every non-Apple compiler (it carries -fopenmp for both compile and link on GCC), delete the GNU branch, `_opennn_needs_openmp`, and make the Config file always `find_dependency(OpenMP)` when built with it. Verify the installed static-library link on Linux once.

*Verifier:* opennn/CMakeLists.txt:169-180: OpenMP_CXX_FLAGS PRIVATE; GNU branch links `gomp pthread` by name plus -fopenmp link option, others OpenMP::OpenMP_CXX. 328-331 `_opennn_needs_openmp` excludes GNU, consumed by OpenNNConfig.cmake.in:9-11. Because the compile flag is PRIVATE on GNU, in-tree consumers with `#pragma omp` get no -fopenmp on GCC. Using OpenMP::OpenMP_CXX PUBLIC everywhere removes the…

#### core-utils-14 — add_json_field is a three-layer forwarding chain; the templates add nothing

`opennn/core/json.h:106-130` · low · boilerplate · lines -12 · effort S · risk low · confirmed

JsonWriter::add_field(string_view, Json) is wrapped by a template add_field(Value&&) that only constructs a Json, which is wrapped again by free add_json_field (non-template) and a template add_json_field(Value&&). All Json constructors are implicit (bool, integral, double, float, const char*, string, string_view), so a plain `add_json_field(JsonWriter&, string_view, Json)` accepts every argument the templates accept. Repo-wide, `.add_field(` has zero callers outside json.cpp and add_json_field has 72, so two of the four layers are pure overhead in compile time and reading.

**Fix:** Delete both templates; keep JsonWriter::add_field(string_view, Json) and the non-template add_json_field (or make add_json_field an inline one-liner in the header and drop the .cpp definition).

*Verifier:* json.h:106-130: JsonWriter::add_field(string_view, Json) at :106, template add_field(Value&&) at :108-112, free add_json_field(JsonWriter&, string_view, Json) at :122 (defined json.cpp:594) and template add_json_field at :126-130. `.add_field(` has no callers outside json.cpp; add_json_field has 72. All 72 argument expressions (listed via grep) are bool, float, Index/size_t, const string&/string,…

#### core-utils-16 — Three fill loops hand-roll what fill_random already abstracts

`opennn/core/random_utilities.cpp:282-409` · low · duplication · lines -12 · effort S · risk low · partial

The comment on fill_random says 'the three fillers below differ in nothing else', but set_random_bernoulli (302-308), the gaussian fill inside set_random_orthogonal (322-328) and set_random_integer (403-409) still write the lock + distribution + element loop by hand. Each is a verbatim copy of fill_random's body with a different distribution.

**Fix:** Make fill_random accept any indexable container (span<uint8_t> via operator[] or a small overload) and rewrite the three as one-line calls: fill_random(tensor, uniform_int_distribution<Index>(min,max)); fill_random(gaussian, normal_distribution<float>(0,1)); fill_random(values, bernoulli_distribution(p)) with uint8_t cast.

*Verifier:* Line numbers are wrong: random_utilities.cpp is 186 lines; fill_random is at :52-61, set_random_bernoulli at :73-79, the gaussian fill in set_random_orthogonal at :94-99, set_random_integer at :174-180 (not 282-409). Substance confirmed: the comment at :53-54 says the fillers 'differ in nothing else', yet those three still hand-roll lock + distribution + loop. Note set_random_bernoulli works on…

#### core-types-8 — scale_value re-implements the guards and formulas that scaling_affine already encodes

`opennn/core/scaling.h:69-92` · low · duplication · lines -12 · effort S · risk medium · confirmed

scale_value and scaling_affine are two switch ladders over ScalerMethod with the same EPSILON guards and the same arithmetic; the header even carries a comment at line 99 ('Guards must match scale_value above') to keep them in lockstep by hand. Only Logarithm is non-affine. Callers of scale_value are tabular_dataset.cpp:307 and tabular_dataset.h:246 plus tests.

**Fix:** Define scale_value as `if (method == ScalerMethod::Logarithm) return log(max(value, EPSILON)); const auto [scale, offset] = scaling_affine(method, desc, min_range, max_range); return value * scale + offset;` and delete the ladder and the 'must match' comment. Note the MinimumMaximum result changes by floating-point rounding (x*scale+offset vs (x-min)/range*span+lo); tests/core/scaling_test.cpp uses EXPECT_NEAR 1e-6 and scaler_parity_test.cpp 1e-5, so run both before merging.

*Verifier:* scaling.h:69-92 (scale_value) and :95-130 (scaling_affine) carry the same EPSILON guards (MinMax `range < EPSILON`, std `> EPSILON` vs `<= EPSILON` -- equivalent) and the comment at :99 'Guards must match scale_value above'. Callers: tabular_dataset.cpp:307, tabular_dataset.h:246, tests/core/scaling_test.cpp:293-316; scaling_affine used by image_dataset.cpp:127 and scaling_layer.cpp:461. Only…

#### xcut-boilerplate-12 — Three residual hand-written enum<->string converters outside EnumMap

`opennn/dataset/dataset.cpp:658-680` · low · boilerplate · lines -12 · effort S · risk low · confirmed

Pattern (b). EnumMap now backs 14 enums (ActivationFunction, VariableType, ScalerMethod, VariableRole, SampleRole, StorageMode, Codification, MissingValuesMethod, ClampingMethod, ClassActivation, PoolingMethod, NetworkTask, LayerType, Loss::Error, Regularization). Three converters remain hand-written: Dataset::Separator uses a tuple table with two linear-search getters (dataset.cpp:658-680) plus two setters (753-768); TrainingResult::write_stopping_condition (training_result.cpp:25-37) indexes a positional `const char* names[]` by `size_t(*stopping_condition)` and returns "" on mismatch, so reordering StoppingCondition silently mislabels results; chat.cpp:101-110 `role_name` is a switch.…

**Fix:** Two EnumMap<Separator> instances (name and character) with to_string/from_string, an EnumMap<StoppingCondition> used by both TrainingResult::write_stopping_condition and the Optimizer messages, and an EnumMap<ChatRole>. Keeps the JSON strings identical.

*Verifier:* dataset.cpp:658-680 has the tuple separator_map with two linear-search getters and 753-768 two setters, while dataset.cpp:682 onward already uses EnumMap<Dataset::Codification>; training_result.cpp:25-37 indexes a positional names[] by size_t(*stopping_condition) and returns "" on mismatch (enum at training_result.h:17-21 has five members in that order, so any reorder mislabels); chat.cpp:101-110…

#### xcut-boilerplate-10 — InputsSelection common JSON fields repeated in GrowingInputs, GeneticAlgorithm and GrowingNeurons

`opennn/model_selection/growing_inputs.cpp:311-347` · low · duplication · lines -12 · effort S · risk low · partial

Pattern (c). Optimizer already centralises its shared fields in write_common_json/read_common_json (optimizer.cpp:1255-1280), but InputsSelection does not: TrialsNumber, WarmStart, ValidationErrorGoal (with the SelectionErrorGoal alias), MaximumValidationFailures (with the MaximumSelectionFailures alias), MaximumTime and FoldsNumber are written and read separately in growing_inputs.cpp:311-347, genetic_algorithm.cpp:597-629 and growing_neurons.cpp:252-288 (the last mirrors the same fields although it is not an InputsSelection). The alias handling is therefore repeated three times and any new shared field needs six edits.

**Fix:** Add InputsSelection::write_common_json(JsonWriter&) const / read_common_json(const Json*) for the six shared fields (mirroring Optimizer), call them from GrowingInputs and GeneticAlgorithm, and let GrowingNeurons reuse them through a free helper taking the six references. JSON field names stay identical, so saved files remain compatible.

*Verifier:* growing_inputs.cpp:311-347 and growing_neurons.cpp:258-288 do write/read the identical six fields (TrialsNumber, WarmStart, ValidationErrorGoal, MaximumValidationFailures, MaximumTime, FoldsNumber) with the same alias calls, and no write_common_json/read_common_json exists under model_selection/ (grep). Correction: genetic_algorithm.cpp:597-629 does NOT write or read TrialsNumber, WarmStart or…

#### nn-core-13 — The consumer-edge map of source_layers is built three times

`opennn/neural_network/back_propagation.cpp:60-77` · low · duplication · lines -12 · effort S · risk low · confirmed

BackPropagation::make_consumer_edges (back_propagation.cpp:60-77), find_early_output_release_steps (forward_propagation.cpp:45-52) and wire_drelu_fusions (neural_network.cpp:485-488) each invert source_layers into per-producer consumer lists with the same loop and the same `source >= 0` filter; the first two produce the identical vector<vector<pair<size_t,size_t>>>, the third only counts. Three copies of the graph inversion is three places to get the sign filter or the bound wrong.

**Fix:** Add `vector<vector<pair<size_t,size_t>>> NeuralNetwork::get_consumer_edges() const` (or a free function in neural_network.h taking source_layers) and use it in all three sites; wire_drelu_fusions uses `edges[s].size() != 1`.

*Verifier:* Read back_propagation.cpp 60-77, forward_propagation.cpp 45-52, neural_network.cpp 485-488. Same inversion loop with the same `source >= 0` filter; the first two build identical vector<vector<pair<size_t,size_t>>> (the BackPropagation copy adds a redundant `< layers_number` bound that validate_source_indices already guarantees), the third only counts. A NeuralNetwork::get_consumer_edges() used by…

#### layers-b-8 — Recurrent CPU forward/backward copy a B×H block per time step instead of using the strided maps directly

`opennn/neural_network/layers/recurrent_layer.cpp:206-320` · low · overhead · lines -12 · effort S · risk low · partial

apply copies the activated hidden slice into h_c every step (line 229) and materializes rec_acc = h_c * W_rec before adding it (lines 215-218); apply_delta copies the previous hidden slice into h_prev_c (line 311), computes d_c in a temporary and then copies it into the all_delta strided slice (lines 300-306). Eigen can multiply from a ConstStridedMap and accumulate with noalias() into a StridedMap, so h_c, rec_acc, h_prev_c and d_c (four per-call B×H heap allocations plus 3 copies per step) are avoidable. bias_grad.setZero() and w_in_grad.setZero() (lines 257-258) are redundant: both are assigned with noalias() = after the loop (lines 323-324).

**Fix:** Forward: `if (t > 0) h_t.noalias() += ConstStridedMap(hidden_data + (t-1)*H, B, H, stride) * w_rec_map;` and drop h_c/rec_acc. Backward: make d_t a StridedMap over all_delta's slice, compute into it directly, and use `ConstStridedMap h_prev(...)` in the two products; drop d_c/h_prev_c and the two redundant setZero() calls. Validate with RecurrentLayerTest.BackwardGradientMatchesNumerical and ReturnSequences.

*Verifier:* Code matches: apply allocates h_c/rec_acc (207-208), does rec_acc.noalias() = h_c * w_rec_map; h_t += rec_acc (215-218) and h_c = h_t (229); apply_delta allocates d_c/h_prev_c (285-286), stores d_c into the strided all_delta slice (304-305) and copies h_prev_c from a ConstStridedMap (309-312); bias_grad.setZero()/w_in_grad.setZero() at 257-258 are overwritten by noalias()= at 323-324, so those…

#### nn-core-12 — Legacy JSON 'Parameters/Values' read is asymmetric and silently truncates on size mismatch

`opennn/neural_network/neural_network.cpp:1749-1767` · low · API · lines -12 · effort S · risk medium · confirmed

to_JSON never writes a 'Parameters' element (parameters go to the .bin snapshot), but from_JSON still reads 'Parameters/Values'. When the count differs from the compiled buffer it prints a warning to cout and copies min(n, m) floats, leaving the rest zero - a silently wrong model - whereas set_parameters throws on the same mismatch (1008-1009) and the binary loader rejects layout mismatches with a fingerprint. Two code paths, two policies, and a std::copy into the host master that also bypasses the released-master guard discussed in nn-core-1 (it is protected only because HostParametersGuard throws in copy_parameters_host).

**Fix:** Replace the block with `if (!parameters_text.empty()) { VectorR v; string_to_vector(parameters_text, v); set_parameters(v); }` so the legacy path shares set_parameters' size check, residency handling and bf16 cast. Verify against Neural Designer that no shipped model files rely on the partial-load behaviour before changing the policy; if they do, keep the truncation but route through set_parameters after resizing.

*Verifier:* Read neural_network.cpp 1749-1767 and 1002-1017; grep for "Parameters" in the file shows only the read at 1749 - to_JSON never writes it (1545-1575 write Items/SourceLayers/TiedWeights only). The legacy path warns to cout and copies min(n,m) floats whereas set_parameters throws on mismatch (1008-1009) and the binary loader fingerprints the layout. The HostParametersGuard at 1764 protects the…

#### nn-core-15 — get_layers_number(const string&) and get_layers_number(LayerType) have no callers

`opennn/neural_network/neural_network.cpp:967-976` · low · dead code · lines -12 · effort S · risk medium · confirmed

Both overloads (header 176-177, bodies 967-976) are unused in opennn/, tests/, examples/ and docs/benchmarks/ (grep for `get_layers_number(` with any argument finds only the definitions). has(LayerType)/get_first(LayerType) cover the 'is there one' case and get_layers() the counting case.

**Fix:** Delete both overloads. Verify against Neural Designer first (the prior audit's ND alive-list does not mention them, but it is not exhaustive).

*Verifier:* grep -rn for `get_layers_number(` followed by a non-')' character across opennn/, tests/, examples/, docs/ finds only the header declarations (neural_network.h:176-177) and the definitions (neural_network.cpp:967-976). Neural Designer is not present on this machine (C:\Artelnics does not exist), so the ND caveat in the proposed fix cannot be checked here and the medium risk is appropriate. LOC…

#### nn-core-14 — HostParametersGuard and HostStatesGuard are the same RAII struct written twice

`opennn/neural_network/neural_network.h:44-59` · low · duplication · lines -12 · effort S · risk low · confirmed

The two guards differ only in which pair of copy_*_host/copy_*_device they call and which Buffer's device they test; each is ~18 lines with deleted copy operations. One template over the two member-function pointers (or a struct holding two `void (NeuralNetwork::*)()`) with two aliases keeps both names and removes the duplicate, and removes the opportunity for the predicate slip reported in nn-core-2.

**Fix:** `template<Buffer NeuralNetwork::*Storage, void (NeuralNetwork::*ToHost)(), void (NeuralNetwork::*ToDevice)()> struct HostGuard {...};` with `using HostParametersGuard = HostGuard<&NeuralNetwork::parameters, &NeuralNetwork::copy_parameters_host, &NeuralNetwork::copy_parameters_device>;` and the states alias. Keep HostParametersGuard public (model_expression.cpp uses it).

*Verifier:* Read neural_network.h 44-60 (HostParametersGuard, public section starting 38) and 347-364 (HostStatesGuard, private section starting 338); model_expression.cpp:330/715 use the public one. Both are the same RAII shape modulo the Buffer tested and the two copy_* members. A private HostGuard template with a public `using HostParametersGuard = ...` alias is legal C++ and keeps model_expression.cpp…

#### operators-b-10 — encode_sequence overloads and WordLevel tokenize/encode duplicate the same bodies

`opennn/neural_network/operators/tokenizer_operator.cpp:173-216` · low · duplication · lines -12 · effort S · risk low · confirmed

The two encode_sequence overloads are 20 lines each and identical except for the element-to-id projection (token_to_id(token) vs the id itself); WordLevelTokenizer::tokenize and ::encode both lowercase + tokenize_views and differ only in the final transform. Four bodies for two behaviours.

**Fix:** One private `frame_sequence(span<const Index> ids, Index sequence_length)` that adds start/end and truncates; the vector<string> overload maps tokens through token_to_id into a local vector and calls it, the string_view overload passes encode(text). WordLevelTokenizer::encode can call tokenize_views once and token_to_id per view, with tokenize() built from the same views (or the base encode()).

*Verifier:* encode_sequence overloads at 173-215 are 20 lines each and differ only in the projection (token_to_id(token) vs id) and the source (tokens vs encode(text)); WordLevelTokenizer::tokenize (287-292) and ::encode (294-304) both do ascii_lowercase + tokenize_views and differ only in the final transform. A shared frame_sequence(span<const Index>, Index) plus one views-based helper is straightforward;…

#### response-opt-19 — Small boilerplate: virtual Domain dtor, UnivariateConstraint ctor, append_rows vs stack_rows, std::function find

`opennn/response_optimization/response_optimization.h:57-82` · low · boilerplate · lines -12 · effort S · risk low · partial

(a) Domain declares `virtual ~Domain() = default` with no derived class anywhere, adding a vtable to a struct that is copied into `vector<Domain>` per Pareto point every iteration (1927, 2031). (b) UnivariateConstraint's three-argument constructor only repeats what default member initializers would express, and prevents aggregate init. (c) append_rows (25-42) is stack_rows({a, b}) (1709-1729) with an extra shape throw; append_columns is only used at two sites. (d) partition_input_constraints_by_variable wraps a non-recursive lambda in `function<Index(Index)> find` (response_constraints.cpp:1685), paying type erasure inside the union-find loop for nothing.

**Fix:** Drop the virtual dtor and the explicit default ctor; give UnivariateConstraint NSDMIs (`ComparisonOperator comparison = None; float low_bound = 0; float up_bound = 0;`) and use brace-init at the ~8 construction sites; implement append_rows as `stack_rows({a, b})`; make `find` an `auto` lambda.

*Verifier:* (a) confirmed: `virtual ~Domain() = default` at h:60 with no derived class; vector<Domain> built per Pareto point at 1927 and 2031. (b) confirmed: UnivariateConstraint ctor at response_constraints.h:120-121 only repeats defaults. (d) confirmed: `function<Index(Index)> find` at response_constraints.cpp:1685 wrapping a non-recursive lambda. (c) is overstated: append_rows (25-42) cannot simply…

#### training-loss-12 — cross_entropy_3d_gradient_device_count CPU fallback is unreachable

`opennn/training_strategy/error_functions.cpp:490-508` · low · dead code · lines -12 · effort S · risk medium · confirmed

The only caller is Loss::back_propagate_device_metrics (loss.cpp:1680), which is compiled under `#ifdef OPENNN_HAS_CUDA` and guarded by supports_device_epoch_metrics() (runs_on_gpu()), with `input` being the device-resident network output; the function's `if (input.is_cuda())` branch therefore always returns first. The serial host count + delegate below it (496-507), and the matching stub in the OPENNN_CUDA_STUBS list (line 82), never execute.

**Fix:** Replace the body with `throw_if(!input.is_cuda(), ...)` + the CUDA call, or move the function behind `#ifdef OPENNN_HAS_CUDA` in both header and source and drop the stub entry. Verify against Neural Designer that the free function is not called with host tensors.

*Verifier:* error_functions.cpp:490-508: CUDA branch returns first; the host count + delegate follows. Only caller is loss.cpp:1680 inside back_propagate_device_metrics, which sits in the `#ifdef OPENNN_HAS_CUDA` block (the `#else` stub starts at 1697) and is gated by runs_on_gpu, with `input` the device-resident output. Header is public (error_functions.h:40) so the ND check in the fix is warranted.

#### xcut-api-7 — Eleven accessors with zero callers in opennn/, tests/, examples/ and docs/benchmarks/

`opennn/training_strategy/optimizer.h:73-80` · low · dead code · lines -12 · effort S · risk medium · confirmed

Counting every call site across the four trees (definitions excluded): Optimizer::set_validation_period (0; it is also the only way to set validation_period - read_common_json does not read it - so the epoch%validation_period gate at optimizer.cpp:956 is always 1 unless Neural Designer calls it), Optimizer::get_maximum_validation_failures (0), get_gradient_clip_norm (0), get_display_period (0), Loss::get_yolo_lambda_noobj/class/giou/dfl and get_yolo_focal_gamma (0 each, 5 lines; note there is no get_yolo_obj_focal_gamma, so the set_ side is six and the get_ side five - asymmetry that shows they were added for symmetry, not use), Dataset::get_display (0; the one 'get_display()' hit in the…

**Fix:** Verify each against Neural Designer (the prior audit found ND uses many orphaned accessors). Delete the ones ND does not use; for set_validation_period decide whether validation_period is a feature (then expose it in JSON too) or dead (then delete the member and the gate). Drop the unused NeuralNetwork& parameter from get_batch_workers_number (or inline workers_number at its 3 callers and delete it).

*Verifier:* Grep over opennn/, tests/, examples/, docs/benchmarks/: set_validation_period, get_maximum_validation_failures, get_gradient_clip_norm, get_display_period (optimizer.h:73-80), get_yolo_lambda_noobj/class/giou/dfl, get_yolo_focal_gamma (loss.h:155-159) and Dataset::get_display (dataset.h:142; the only get_display() call at training_strategy.cpp:145 is Optimizer's) each appear only at their…

#### core-utils-15 — env_flag_enabled(name) duplicates env_flag_enabled(name, false)

`opennn/core/string_utilities.cpp:428-453` · low · duplication · lines -11 · effort S · risk low · confirmed

The one-argument overload re-implements the 'on' list with its own initializer_list and ranges::any_of; its result is identical to the two-argument overload with default_value=false for every input (unset, empty, on-words, off-words, garbage all map to false/true the same way). Two copies of the accepted-words list can drift.

**Fix:** Replace the body with `return env_flag_enabled(name, false);` or give the two-argument overload `bool default_value = false` and delete the one-argument declaration/definition.

*Verifier:* string_utilities.cpp:428-440 one-arg overload vs :442-452 two-arg overload: for unset -> false/default(false); empty string -> no on-word matches -> false vs `!*value` -> default false; on-words -> true both; off-words and garbage -> false both. Identical results for default_value=false; the accepted-words list is duplicated. Callers use both forms (cudnn_frontend_utilities.h:59/372/508,…

#### response-opt-14 — Scaling/Unscaling snapshot branch duplicated although Unscaling derives from Scaling

`opennn/response_optimization/network_differential.cpp:39-57` · low · boilerplate · lines -11 · effort S · risk low · confirmed

`class Unscaling final : public Scaling` (unscaling_layer.h:16) inherits get_descriptives/get_scalers/get_min_range/get_max_range, so the two 7-line branches selecting between `static_cast<const Scaling*>` and `static_cast<const Unscaling*>` read the same members through the same interface.

**Fix:** `const Scaling& scaling = static_cast<const Scaling&>(*layer);` once, then read descriptives/scalers/min_range/max_range; drop the two pointer locals and the unscaling_layer.h include if nothing else needs it. The per-feature copy loop can be a single ranges::transform per field or kept as is.

*Verifier:* network_differential.cpp:39-57 has two identical 7-line branches differing only in the static_cast target; `class Unscaling final : public Scaling` (unscaling_layer.h:16) so a single `static_cast<const Scaling&>(*layer)` reads the same get_descriptives/get_scalers/get_min_range/get_max_range. unscaling_layer.h is not otherwise needed in that file. LOC -11 accurate.

#### core-device-8 — finalize and finalize_attention duplicate the validate/load/build/workspace/store skeleton

`opennn/core/cuda/cudnn_frontend_utilities.h:512-622` · low · duplication · lines -10 · effort M · risk medium · partial

Both functions run the same sequence - validate, load_cached_plan early-return, build_operation_graph, create_execution_plans, build (top-K when autotuning, HEURISTICS_CHOICE otherwise), get_workspace_size, store_cached_plan - and differ only in three parameters: the heuristic mode list ({A} or {A,B,FALLBACK} vs heuristic_modes()), whether the workspace cap and engine-note restriction apply, and which flag gates autotune (sdpa_autotune_enabled() && allow_autotune vs conv_autotune_enabled()). About 30 lines are duplicated, and the cache-key fix in core-device-3 has to be applied consistently to both.

**Fix:** One finalize(graph, workspace_bytes, tag, const FinalizeOptions&) with {vector<HeurMode_t> modes; int64_t workspace_cap; const vector<NumericalNote_t>* notes; bool autotune;}; finalize_attention becomes a two-line wrapper building the options. Keep the restricted-notes fallback loop as is.

*Verifier:* Read finalize_attention (512-541) and finalize (553-622). The shared skeleton (validate, load_cached_plan early return, build_operation_graph, get_workspace_size, store_cached_plan) is real, ~10 lines, and the cache-key fix in core-device-3 does touch both. But the differing middle is larger than 'three parameters': finalize has the cap/notes prepare_candidates lambda, the restricted-notes retry…

#### core-device-11 — LtMatmulPlan hand-swaps nine members to move into the cache; preference handle leaks on a CHECK_CUBLAS throw

`opennn/core/device_backend.cpp:1289-1300` · low · boilerplate · lines -10 · effort S · risk low · confirmed

The move constructor exists only so get_lt_matmul_plan can build a plan locally and plans.emplace(key, std::move(plan)) it; it swaps nine fields by hand. In the same function the cublasLtMatmulPreference_t pref (1459-1474) is created raw and destroyed after cublasLtMatmulAlgoGetHeuristic, so the CHECK_CUBLAS throw in between leaks it (and any throw after the matrix layouts are created relies on the destructor calling the Destroy functions on nullptr handles, which returns an error status that is silently ignored).

**Fix:** Build the plan in place: auto [it, inserted] = plans.try_emplace(key); LtMatmulPlan& plan = it->second; with a ScopeExit that erases the entry if the function exits by exception; delete the move constructor. Wrap pref in unique_ptr<remove_pointer_t<cublasLtMatmulPreference_t>, decltype(&cublasLtMatmulPreferenceDestroy)> (the same shape as GraphExecHandle) so the throw path frees it.

*Verifier:* device_backend.cpp:1289-1300: hand-written nine-field swap move ctor, used only by `plans.emplace(key, std::move(plan))` at 1488. pref leak confirmed: cublasLtMatmulPreferenceCreate at 1460, CHECK_CUBLAS on SetAttribute (1461) and AlgoGetHeuristic (1467) can throw before cublasLtMatmulPreferenceDestroy at 1474. The destructor (1302-1308) does call the Destroy functions on nullptr handles when a…

#### dataset-a-20 — fit_softmax_correlation and fit_logistic_correlation repeat the dataset/loss/train/correlate sequence

`opennn/dataset/correlations.cpp:47-100` · low · duplication · lines -10 · effort S · risk low · partial

Both functions build a TabularDataset from the filtered columns, set all samples Training, construct a Loss with MeanSquaredError and no regularisation, train inside a try/catch that maps any exception to coefficient 0, then call output_target_correlation and set_confidence_interval. The only genuine differences are the network (ClassificationNetwork vs Scaling+Dense sigmoid), the optimizer choice and the sign/slope post-processing. ~35 lines are duplicated (61-99 vs 372-424), and the two early-return blocks in logistic_correlation / logistic_correlation_spearman (441-469) are identical as well.

**Fix:** One file-local `bool train_correlation_model(NeuralNetwork&, TabularDataset&, Optimizer&, Correlation&)` that owns the Loss setup, the try/catch and the output/confidence step; both fitters become ~12 lines each. A `sigmoid_nan_correlation()` helper for the two identical early returns.

*Verifier:* Duplication exists: correlations.cpp:78-104 and 386-424 both set Loss(MeanSquaredError, None regularization), wrap train() in try/catch mapping to coefficient 0, then output_target_correlation + set_confidence_interval. But the logistic fitter additionally selects between QuasiNewton and LevenbergMarquardt inside the try (393-407) and has an isfinite early return (414-418) before…

#### nn-expression-12 — get_expression_javascript re-implements split_expression_lines/prepare_body_lines with drifted rules

`opennn/neural_network/model_expression.cpp:1806-1815` · low · duplication · lines -10 · effort S · risk low · confirmed

The JS path hand-rolls the line split that split_expression_lines + rename_spaced_var_definitions (= prepare_body_lines) already implement, with three small divergences (`token.back()=='{'` instead of `find('{')`, `size()>1` guards, whitespace-only lines kept and later turned into blank output lines by blank_short_lines). The PHP path likewise repeats process_body_line's bracket replacement (1705-1709). Two copies of a parser rule are one more place for the next export bug.

**Fix:** Replace the block with `const vector<string> lines = prepare_body_lines(expression);` (the bracket replacement already ran on the whole expression two lines earlier) and drop blank_short_lines from LanguageSyntax if no other consumer needs it.

*Verifier:* get_expression_javascript 1806-1815 hand-rolls the split that split_expression_lines (284-300) + rename_spaced_var_definitions (= prepare_body_lines, 432-437) already implement; divergences are as described (`back()=='{'` vs find('{'), size()>1 guards, whitespace-only lines kept then blanked by blank_short_lines which only JS sets, line 555). PHP repeats the bracket replacement at 1705-1709.…

#### nn-expression-15 — activation_constant_for is a hand-written enum->string ladder; EnumMap is the sanctioned helper

`opennn/neural_network/model_expression.cpp:759-780` · low · boilerplate · lines -10 · effort S · risk low · confirmed

The lambda switches over ActivationFunction to produce NN_* constants; AGENTS.md names EnumMap<E> as the enum<->string facility and activation_function_map() already exists. The Softmax case is guarded by the caller (line 1126-1129) before the lambda is reached, so the throwing case inside is duplicate policy.

**Fix:** Replace with a file-local `static const EnumMap<ActivationFunction> embedded_activation_map{{Identity,"NN_IDENTITY"}, ...}` (Softmax omitted so to_string throws the map's error) and call `embedded_activation_map.to_string(activation)`; the enum list in the generated `typedef enum {...} nn_activation` at line 1485 can be generated from the same map so the two cannot drift.

*Verifier:* Lines 759-780 are the hand-written switch; Softmax is already intercepted by the Dense branch at 1126-1129 before the lambda. EnumMap exists (opennn/core/enum_map.h:19, to_string throws 'Unknown enum value' at line 39) and activation_function_map() is declared at tensor_operations.h:60. Minor caveat: a Recurrent/LSTM layer configured with Softmax would then get the generic EnumMap message instead…

#### core-types-9 — Dead umbrella aliases and duplicated using-directives in opennn_types.h / tensor_types.h

`opennn/core/opennn_types.h:312-321` · low · dead code · lines -9 · effort S · risk medium · confirmed

Tensor0, Tensor1, TensorR<Rank> and TensorMap2 have zero references in opennn/, tests/, examples/ and docs/benchmarks (Tensor2/3/4, TensorMap3/4 and TensorMapR are used). tensor_types.h:183 `CUDA_REDUCTION_DTYPE` has zero references anywhere (CUBLAS_COMPUTE_DTYPE beside it is used by device_backend.cpp:1370). The header also repeats `using namespace std;` (line 84 inside the CUDA block and line 136 unconditionally) and `using type = float;` (line 139 at global scope and line 170 inside namespace opennn), and carries both `#pragma once` and a classic include guard. Each is a line a reader has to rule out before trusting the public surface.

**Fix:** Delete Tensor0, Tensor1, TensorR, TensorMap2 and CUDA_REDUCTION_DTYPE after verifying against Neural Designer; drop the `using namespace std;` at line 84 (line 136 covers both branches) and the `using type = float;` inside namespace opennn at line 170 (the prior audit notes ND writes `type(0)`, which the global alias at line 139 serves); keep one include-guard mechanism.

*Verifier:* opennn_types.h:312-313 Tensor0/Tensor1, :318-319 TensorR, :321 TensorMap2: whole-word greps over opennn/, tests/, examples/, docs/benchmarks return zero uses. tensor_types.h:183 CUDA_REDUCTION_DTYPE: zero uses; CUBLAS_COMPUTE_DTYPE (:184) used at device_backend.cpp:1370. `using namespace std;` at opennn_types.h:84 (inside #ifdef OPENNN_HAS_CUDA) and :136 (unconditional); `using type = float;` at…

#### core-types-3 — uses_cuda_fill queries the driver per fill and ignores TensorView::device

`opennn/core/tensor_types.cpp:44-51` · low · overhead · lines -9 · effort S · risk medium · confirmed

In CUDA builds every fill/setZero (including host views: datasets, scaling columns, recurrent/LSTM bias init, embedding gradients) calls cudaPointerGetAttributes, a driver round-trip of a few microseconds, plus cudaGetLastError to clear the failure for pageable host pointers. copy/add/multiply/softmax/activation_* in tensor_operations.cpp all dispatch on `is_cuda()` (e.g. `if (source.is_cuda()) { copy_gpu(...); return; }`), and the CUDA-less stub of this very function is `return view.is_cuda();`. The driver query also silently repairs a view whose device tag is wrong, which every other op would reject, so a mis-tagged view works here and throws one call later.

**Fix:** Delete the CUDA-specific uses_cuda_fill and keep the single `view.is_cuda()` version outside the #ifdef (or inline it into TensorView::fill). If the suite then surfaces a view whose device tag is wrong, fix that tag at its constructor - that is a latent bug the driver query was masking.

*Verifier:* tensor_types.cpp:44-51 (CUDA) vs :68-71 (stub) read exactly as quoted; every CUDA-build fill/setZero of a host view (scaling_layer.cpp:43/51, recurrent_layer.cpp:59/71/257-259, long_short_term_memory_layer.cpp:32/105/119) pays a cudaPointerGetAttributes + cudaGetLastError. Sibling ops in tensor_operations.cpp dispatch on is_cuda() (e.g. :1141 multiply, :1166 softmax). The driver query is indeed…

#### layers-b-9 — Recurrent forward slot numbering is declared twice (operator enum and private layer enum) with no link between them

`opennn/neural_network/layers/recurrent_layer.h:21-27` · low · design · lines -9 · effort S · risk low · confirmed

RecurrentOperator::ForwardScratchSlot hard-codes StepInputForwardSlot = 3 .. StepDerivativesForwardSlot = 6, and Recurrent::Forward (lines 175-185) independently enumerates Input..Output with the scratch slots landing on 3..6. Nothing asserts that the two agree: reordering Recurrent::Forward (the one the spec list in get_forward_specs follows) silently makes the operator read the wrong scratch view on the GPU fused path. LongShortTermMemoryOperator keeps a single complete ForwardSlot enum in the operator and the layer uses it via `using enum`.

**Fix:** Make RecurrentOperator's enum the complete one (InputSlot = 0, HiddenStatesSlot, ActivationDerivativesSlot, StepInputSlot, ..., OutputSlot) and delete Recurrent::Forward; configure_operators and get_forward_slot_kind use the operator enum, as LongShortTermMemory::configure_operators already does.

*Verifier:* recurrent_layer.h:21-27 RecurrentOperator::ForwardScratchSlot starts at 3 with no reference to the layer's enum; recurrent_layer.h:175-185 Recurrent::Forward enumerates Input..Output independently; and there is a third hard-coded copy in get_forward_slot_kind (146-151: spec==1 TrainingOnly, 2..5 Transient). configure_operators (711-717) uses the private enum; the operator's forward_propagate…

#### core-device-9 — Compute capability and device presence are queried three different ways with two encodings

`opennn/core/device_backend.cpp:193-262` · low · duplication · lines -8 · effort S · risk low · partial

cuda_compute_capability() calls cudaGetDeviceProperties (uncached, ~ms on some drivers) and returns major*10+minor (86); cudnn_frontend::device_sm_version() (cudnn_frontend_utilities.h:34-45) uses cudaDeviceGetAttribute, caches, and returns major*100+minor*10 (860); gpu_info_string() calls cudaGetDeviceProperties again. has_cuda_device() and Backend::Backend() (1124-1133) both implement cudaGetDeviceCount with the same error-clearing dance. Two encodings of the same number in adjacent files is a latent off-by-10x comparison bug waiting for the next threshold (the 700/800 sm gates vs the 80 used by configuration.cpp).

**Fix:** Cache cuda_compute_capability() in a function-local static using cudaDeviceGetAttribute; define device_sm_version() as device::cuda_compute_capability() * 10; have Backend::Backend() call has_cuda_device() instead of repeating cudaGetDeviceCount; gpu_info_string can keep its single properties query.

*Verifier:* Confirmed: cuda_compute_capability (device_backend.cpp:214-227) calls cudaGetDeviceProperties uncached and returns major*10+minor; device_sm_version (cudnn_frontend_utilities.h:34-45) uses cudaDeviceGetAttribute, cached, returns major*100+minor*10; gpu_info_string (240-257) queries properties again; has_cuda_device (193-212) and Backend::Backend (1124-1133) both do the cudaGetDeviceCount dance…

#### dataset-b-13 — LanguageDataset::read_txt repeats record packing that write_binary_cache already implements

`opennn/dataset/language_dataset.cpp:153-196` · low · duplication · lines -8 · effort S · risk low · partial

The Matrix branch (153-188) packs input/target/decoder ids into `data` with the same min(ssize(x), maximum_*_sequence_length) truncation loops that write_binary_cache (566-599) performs into an int32 record; both branches also repeat the encode_streaming call and the two index vectors. The decoder columns differ only in being derived at fill time for the binary path (fill_sequences shift=1) versus stored explicitly in the matrix. ~25 duplicated lines and two places to keep truncation semantics in sync.

**Fix:** Hoist encode_streaming above the branch, and add a file-local `pack_record(const vector<Index>& in, const vector<Index>& tgt, Index in_len, Index tgt_len, auto write)` (or fill a float row via fill_sequences-style offsets) used by both the matrix packing and write_binary_cache.

*Verifier:* language_dataset.cpp:157-178 and write_binary_cache 577-589 repeat the encode_streaming call (both branches, 158 and 193) and the truncated input/target copy loops; the Matrix branch additionally writes the START_INDEX-shifted decoder columns (179-186), which the binary path derives at fill time. The shared portion is ~10 lines, not 25, so a pack_record helper nets about -8; hoisting…

#### dataset-b-12 — Batch seed derivation duplicated verbatim in fill_inputs and fill_targets (must stay identical)

`opennn/dataset/yolo_dataset.cpp:1902-1908` · low · duplication · lines -8 · effort S · risk low · confirmed

The epoch_seed lambda (FNV-style hash over sample_indices) appears identically at 1902-1908 and 2027-2033, together with the `augment`/`matrix_storage`/`cfg`/`mosaic` preamble. Image augmentation and box augmentation only line up because both copies compute the same value; any edit to one without the other silently desynchronises pixels and targets with no test catching it (augmentation tests use no_aug except one mosaic test).

**Fix:** Add a file-local `uint64_t batch_seed(const vector<Index>& sample_indices)` in the anonymous namespace next to splitmix64 and call it from both fills; one source of truth for the pairing.

*Verifier:* The epoch_seed lambda is byte-identical at yolo_dataset.cpp:1902-1907 and 2027-2032, together with the augment/matrix_storage/cfg/mosaic preamble; pixel and box augmentation pair only because both copies agree. One correction: splitmix64 (411) sits in namespace opennn, not in the anonymous namespace (which closes at 27), so the helper goes beside it as a file-local in namespace opennn. -8 LOC…

#### layers-b-13 — DetectionV8 JSON body re-derives input_shape from fields from_JSON already applied and hand-rolls an optional read

`opennn/neural_network/layers/detection_v8_layer.cpp:155-174` · low · boilerplate · lines -8 · effort S · risk low · confirmed

Layer::from_JSON calls set_input_shape(InputDimensions) before read_JSON_body (layer.cpp:192-198), so recomputing input_shape from ClassesNumber/GridSize/GridWidth is redundant and diverges if the two ever disagree; the `root ? root->find("RegMax") : nullptr` + read_json_index dance is what root->has(...) is for (MultiHeadAttention::read_JSON_body:270-278). write_JSON_body serializes the four integers as strings via to_string while the sibling layers in this scope use the write_json table with native values.

**Fix:** read_JSON_body: detection.reg_max = max(Index(1), root->has("RegMax") ? read_json_index(root, "RegMax") : Index(1)); configure_operator(); (drop the three derived reads). write_JSON_body: write_json(writer, {{"ClassesNumber", classes_number}, {"GridSize", grid_size}, {"GridWidth", grid_width}, {"RegMax", reg_max}}) — keep writing the derived fields for external readers.

*Verifier:* layer.cpp:192-198: from_JSON calls set_input_shape(InputDimensions) (unconditionally, so the field is required) before read_JSON_body; DetectionV8::apply_input_shape (144-147) already routes through set(shape, reg_max) -> configure_operator, so read_JSON_body (155-166) recomputing input_shape from ClassesNumber/GridSize/GridWidth is redundant and can diverge. The `root ? root->find("RegMax") :…

#### nn-expression-19 — Name mapping is O(names x lines x line_length): a whole-string replace per name per line

`opennn/neural_network/model_expression.cpp:1691-1702` · low · overhead · lines -8 · effort M · risk low · confirmed

process_body_line and the JS lambda run replace_all_word_appearances once per input name on every body line (plus once per activation name in Python), and get_expression_php runs it over the entire expression for every variable in php_vars, which includes every intermediate LHS. For Dense-only networks this is negligible (export is one-off), but for unrolled Recurrent/LSTM expressions it is 6*T*H variables against a multi-megabyte string: a 50-step, 64-unit LSTM exported to PHP performs ~19k full-string scans over ~40 MB. Not a hot training path, so this is a scalability note rather than a throughput issue.

**Fix:** Replace the per-name passes with one identifier tokenizer per line: scan [A-Za-z_][A-Za-z0-9_]* runs and look each up in an unordered_map<string,string> built once (input->fixed, output->fixed, var->$var). One ~15-line static `map_identifiers(string_view, const unordered_map<string,string>&)` serves process_body_line, the JS lambda and the PHP pass.

*Verifier:* get_expression_php 1691-1702 collects every LHS into php_vars and runs replace_all_word_appearances (string_utilities.cpp:278-301, a full linear scan per call) over the entire expression once per variable; process_body_line (410-430) and the JS lambda (1997-2005) do one scan per input name per line; Python additionally one per activation name (2116-2120). Not a training hot path; the scalability…

#### nn-core-11 — from_JSON: capture-free lambda, two identical Items walks, and a legacy block with its own error policy

`opennn/neural_network/neural_network.cpp:1602-1767` · low · boilerplate · lines -8 · effort S · risk low · confirmed

from_JSON (165 lines) has three seams: read_variables_array (1613-1645) captures nothing and is a 33-line static function trapped in a lambda, belonging beside define_variables_from_names; the 'Items' array is walked twice with the same skip predicate (`!item.is_object() || item.as_object().empty()`) and the same JsonDocument.set_root dance (1662-1679 to construct, 1734-1747 to load state) - a for_each_layer_item(items_array, fn) helper removes the duplicate and the manual layer_index counter; and the tied-weights and SourceLayers blocks are self-contained parse_* steps. Splitting makes the control flow (parse -> compile -> load state -> legacy parameters) visible.

**Fix:** Hoist read_variables_array to a static free function next to define_variables_from_names; add a static `for_each_layer_item(const Json* items, F&& fn)` used by both walks; optionally split parse_source_layers/parse_tied_weights out as statics taking (layers, source_layers, const Json*).

*Verifier:* Read neural_network.cpp 1602-1767. read_variables_array (1613-1645) has an empty capture list; the Items array is walked at 1665-1679 and 1737-1747 with the identical skip predicate and JsonDocument/set_root dance; tied-weights (1703-1725) and SourceLayers (1683-1701) are self-contained. 166 lines. Proposed helpers are mechanical; LOC -8 reasonable.

#### operators-b-12 — CudnnRnnConfig is a one-field struct wrapping cudnnRNNMode_t

`opennn/neural_network/operators/cudnn_rnn.h:37-44` · low · boilerplate · lines -8 · effort S · risk low · confirmed

After the prior audit's driver merge, the only thing CudnnRnnConfig carries is cell_mode; it is read exactly once (config.cell_mode at cudnn_rnn.cpp:65 and :98) and constructed as `{CUDNN_LSTM}` / recurrent_cudnn_config(activation). The struct plus its #ifdef wrapper and the const& parameters in two signatures are ceremony around an enum value.

**Fix:** Replace `const CudnnRnnConfig&` with `cudnnRNNMode_t cell_mode` in cudnn_setup_/cudnn_setup_attempt_ and have recurrent_cudnn_config() return cudnnRNNMode_t (LSTM passes CUDNN_LSTM); delete the struct. If a second config field ever appears, reintroduce the struct then.

*Verifier:* cudnn_rnn.h:37-44: CudnnRnnConfig has the single member cell_mode; read only at cudnn_rnn.cpp:68 (is_lstm) and :98 (cudnnSetRNNDescriptor_v8). Constructed as {CUDNN_LSTM} (long_short_term_memory_layer.cpp:931) and via recurrent_cudnn_config(activation) (recurrent_layer.cpp:350-359). Signatures cudnn_setup_/cudnn_setup_attempt_ (cudnn_rnn.h:85,89; cpp:26,55) take const CudnnRnnConfig&. Replacing…

#### r2-batch-pipeline-and-device-gather-8 — The H2D event handshake is re-implemented inline five times in optimizer.cpp instead of using Batch's helpers; one site bypasses record_h2d_done

`opennn/training_strategy/optimizer.cpp:1494-1495` · low · duplication · lines -8 · effort S · risk low · confirmed

Batch exposes wait_h2d_complete(), wait_h2d_on_compute_stream() and record_h2d_done(), yet optimizer.cpp pokes the public members directly: `if (x.h2d_done_recorded) device::synchronize_event(x.h2d_done_event.get())` at 1494-1495 and 1622-1623 (== wait_h2d_complete, minus the flag reset), `if (slot.h2d_done_recorded) device::stream_wait_event(transfer, slot.h2d_done_event.get())` at 1639-1640 and an unconditional variant at 1732 (== wait_h2d_on_compute_stream for a different stream), and `device::record_event(fixed_device_batch->h2d_done_event.get(), compute)` at 1758, which does not set h2d_done_recorded, so the fixed batch's destructor sync (finding -2) and wait_h2d_complete() silently…

**Fix:** Generalise `wait_h2d_on_compute_stream()` to `wait_h2d_on(cudaStream_t stream = device::get_compute_stream())` (guarded by h2d_done_recorded), replace the two synchronize_event sites with `x.wait_h2d_complete()`, the two stream_wait_event sites with `slot.wait_h2d_on(transfer)`, and line 1758 with `fixed_device_batch->record_h2d_done(compute)` (then `fixed_device_batch_in_use` is just `h2d_done_recorded` and can go). h2d_done_recorded / h2d_done_event can then move out of the public surface or…

*Verifier:* Verified all five sites: optimizer.cpp 1494-1495 and 1622-1623 (inline synchronize_event guarded by h2d_done_recorded), 1639-1640 (guarded stream_wait_event on transfer), 1731-1732 (unguarded stream_wait_event keyed on fixed_device_batch_in_use), 1758 (device::record_event without setting h2d_done_recorded, so wait_h2d_complete/~Batch skip that event). Batch already exposes record_h2d_done /…

#### training-optimizers-4 — QN builds a full validation BackPropagation (gradient + delta arena) just to hold three metric floats

`opennn/training_strategy/quasi_newton_method.cpp:192-252` · low · overhead · lines -8 · effort S · risk low · confirmed

validation_back_propagation is constructed with validation_samples_number, which makes BackPropagation::set plan and allocate the delta arena and a parameters-sized gradient buffer (back_propagation.cpp:79-217), yet the only use is to copy Loss::calculate_error's three fields into metrics and return metrics.error. calculate_error(const Batch&, const ForwardPropagation&) does not need a BackPropagation at all.

**Fix:** hooks.validation_error = [&]{ return loss->calculate_error(*context.validation_batch, *context.validation_forward_propagation).error; }; and delete validation_back_propagation.

*Verifier:* quasi_newton_method.cpp:192 `BackPropagation validation_back_propagation(context.validation_samples_number, *loss);` and hooks.validation_error 242-252 only copies Loss::calculate_error's fields into metrics and returns metrics.error. BackPropagation::set (back_propagation.cpp:79-120) plans the delta arena and allocates/zeroes a parameters-sized gradient (setup_gradient 89-118). No other use of…

#### selection-testing-17 — Constructor/initializer boilerplate: TestingAnalysis ctor out-of-line, NeuronsSelectionResult defaults overridden by its ctor, pch.h double include guard

`opennn/testing_analysis/testing_analysis.cpp:55-59` · low · boilerplate · lines -7 · effort S · risk low · partial

TestingAnalysis(NeuralNetwork*, Dataset*) is a 5-line out-of-line body that only assigns two pointers that already have default member initializers. NeuronsSelectionResult declares `optimum_training_error = 10.0f; optimum_validation_error = 10.0f;` (growing_neurons.h:119-121) and its constructor immediately overwrites both with MAX (growing_neurons.cpp:297-298), so the header lies about the default and differs from InputsSelectionResult (which initializes to MAX in-class). pch.h has both `#pragma once` and an `OPENNN_PCH_H_` guard (pch.h:9-10, 27); no other header in the repo uses a guard.

**Fix:** Header: `explicit TestingAnalysis(NeuralNetwork* nn = nullptr, Dataset* ds = nullptr) : neural_network(nn), dataset(ds) {}` and delete the .cpp body. NeuronsSelectionResult: set the in-class defaults to MAX and drop the two ctor assignments. pch.h: remove the #ifndef/#define/#endif triple.

*Verifier:* Two of three parts confirmed: testing_analysis.h:28 declares `explicit TestingAnalysis(NeuralNetwork* = nullptr, Dataset* = nullptr);` with the 5-line body at testing_analysis.cpp:55-59 only assigning two pointers; growing_neurons.h:119-121 initialise optimum_*_error to 10.0f while the ctor at growing_neurons.cpp:297-298 overwrites them with MAX (InputsSelectionResult uses MAX in-class). The…

#### r2-duplicated-kernels-across-folders-2 — Column-sum-over-rows reduction planned twice with opposite zeroing contracts; callers must pre-zero

`opennn/core/cuda/kernel_tensor.cu:55-70` · low · duplication · lines -6 · effort S · risk low · partial

bias_grad_sum_cuda (kernel_tensor.cu:55-70) and norm_backward_launch's weight-gradient launch (kernel_normalization.cu:1082-1097) are the same reduction: a grid of (column blocks x row chunks) with the row chunk chosen so the grid reaches ~200-256 blocks, floored at 64 rows, partials combined with float atomicAdd. The chunk planner is written twice with different magic numbers (256/64 vs 192/NUM_WARPS*8), and the two diverge on who zeroes the accumulator: normalization zeroes inside the launcher (with an unchecked raw cudaMemsetAsync, see finding 4) and only when grid_y > 1, storing directly otherwise; bias_grad_sum never zeroes and never stores directly, so every caller must remember…

**Fix:** Add one `static inline dim3 row_chunk_grid(int column_blocks, Index rows, int target_blocks, int min_chunk, int& chunk)` to kernel_common.cuh and call it from both launchers. Give bias_grad_sum_cuda a `bool accumulate` parameter: when false it calls device::set_zero_async itself (or, like normalization, stores directly when n_chunks == 1 and skips the memset), when true it keeps today's behaviour for recurrent_layer.cpp:643. The two tensor_operations.cpp blocks become one-line calls with…

*Verifier:* Planner duplication confirmed: kernel_tensor.cu:61-65 (256 target, floor 64) vs kernel_normalization.cu:1086-1089 (192 target, floor NUM_WARPS*8=64) are the same column-blocks x row-chunk grid. Zeroing contracts differ as stated: normalization zeroes only when grid_y > 1 (1090-1095, unchecked raw cudaMemsetAsync) and the kernel stores directly when gridDim.y == 1 (1055-1059); bias_grad_sum_kernel…

#### core-utils-10 — Descriptives member initializers are dead and contradict the constructor; `name` is unused

`opennn/core/statistics.h:16-32` · low · design · lines -6 · effort S · risk medium · confirmed

Descriptives declares defaults minimum=-1, maximum=1, mean=0, standard_deviation=1, but its only constructor defaults every argument to QUIET_NAN and always assigns all four, so a reader who trusts the header believes Descriptives() is (-1,1,0,1) when it is (NaN,NaN,NaN,NaN). vector_descriptives (statistics.cpp:508-509) returns Descriptives() for an empty input precisely because of the NaN behaviour, while the very next branch returns explicit zeros -- the contradiction is already confusing callers. The `string name = "Descriptives"` member is never read or written anywhere in opennn/, tests/, examples/ or docs/benchmarks, yet it is copied in every vector<Descriptives> and shared across the…

**Fix:** Set the member initializers to QUIET_NAN (matching the constructor) or drop the constructor's defaults; delete `name` (verify against Neural Designer). Descriptives::set can then be `*this = Descriptives(a,b,c,d);` or dropped.

*Verifier:* statistics.h:16-32: member initializers (-1,1,0,1) vs constructor defaults QUIET_NAN (statistics.cpp:112-121 assigns all four unconditionally), so Descriptives() is all-NaN. vector_descriptives (statistics.cpp:508-516) returns Descriptives() for empty input and explicit zeros for all-NaN, as described. `name` member: grep for Descriptives .name usage in opennn/ and tests/ returns nothing. Fix and…

#### r2-batch-pipeline-and-device-gather-14 — For GPU batches input_views_host_cache wraps a CUDA pointer in a TensorView labelled Device::CPU

`opennn/dataset/batch.cpp:119-139` · low · design · lines -6 · effort S · risk low · confirmed

Batch::set builds input_views_host_cache / target_view_host_cache unconditionally from `input.buffer.as<float>()` etc. For on_gpu batches those buffers live on Device::CUDA (and are BF16 for BF16 inputs), yet the views are stamped `Type::FP32, Device::CPU`. get_inputs() hides this by returning input_views_cache when uses_cuda(), so nothing reads the wrong view today, but any future consumer of the public `input_views_host_cache` member (or a batch whose uses_cuda() flips because is_cuda_build() is false at runtime) gets a host-labelled view over device memory — a hazard the type system is supposed to prevent. For prefetch-only GPU batches the host-side data actually lives in `input.host`,…

**Fix:** Guard the host-cache construction with `if (!on_gpu)` (the GPU branch builds input_views_cache a few lines later) and make the two caches a single `vector<TensorView> input_views` / `TensorView target_view` populated with the correct device, so get_inputs()/get_targets() become plain accessors without the uses_cuda() switch.

*Verifier:* batch.cpp 119-139 builds input_views_host_cache / target_view_host_cache from input.buffer.as<float>() with Type::FP32 / Device::CPU unconditionally, before the on_gpu branch that builds the CUDA views (199-221); for on_gpu non-prefetch batches the buffer is Device::CUDA (64-100) and possibly BF16. get_inputs()/get_targets() switch on uses_cuda() (batch.h 82-90) so nothing reads the mislabelled…

#### xcut-api-12 — 'int contiguous = -1' tri-state is converted from and back to optional<bool> at both ends

`opennn/dataset/dataset.h:272-288` · low · design · lines -6 · effort S · risk low · confirmed

BatchSlot stores the contiguity hint as optional<bool> (batch.h:37). Dataset::fill_batch_host converts it to an int with a local lambda (dataset.cpp:1122-1124: contiguous ? int(*contiguous) : -1) to call the virtual fill_inputs/fill_decoder/fill_targets, whose 'int contiguous = -1' parameter is then decoded back (tabular_dataset.cpp:237-239: contiguous_hint >= 0 ? static_cast<bool>(contiguous_hint) : is_contiguous(...)). The int form is repeated on the three virtuals, on their overrides in TabularDataset (h:242-258), ImageDataset, TimeSeriesDataset, LanguageDataset, TextGenerationDataset, on fill_features/fill_from_binary_cache, and on fill_tensor_data (tensor_types.h). One concept, two…

**Fix:** Change the parameter type to optional<bool> contiguous = nullopt on the virtuals, overrides, fill_features/fill_from_binary_cache and fill_tensor_data; delete the hint lambda and replace the decode with contiguous.value_or(is_contiguous(feature_indices)) (note value_or evaluates eagerly - keep the ternary if is_contiguous cost matters). Purely mechanical.

*Verifier:* batch.h:37 optional<bool> contiguous; dataset.cpp:1122-1126 hint lambda converting to int; tabular_dataset.cpp:237-239 decodes back; 'int contiguous' appears 27 times in opennn/ including fill_tensor_data at tensor_types.h:759. Changing to optional<bool> with the ternary kept (as the auditor notes about value_or eagerness) is mechanical.

#### nn-builders-chat-13 — chat.cpp resizes ForwardPropagation::staged_input_storage directly (only external user of the staging internals)

`opennn/neural_network/chat.cpp:1133-1140` · low · design · lines -6 · effort S · risk low · confirmed

Impl::initialize_cuda_input reaches into ForwardPropagation's public data members staged_input_storage/staged_inputs and sizes buffer [0] to sequence_capacity * sizeof(float); neural_network.cpp:1314-1359 is the only other code that knows this layout (it resizes the vector itself and stages per input). chat.cpp is therefore coupled to the staging representation and to the assumption that inputs are FP32 token floats; a change to how forward_propagate stages inputs (e.g. lazily sized, int32 tokens) silently breaks the pre-allocation that the 'NoCudaBufferGrowthFromFirstSend' test relies on. The same pattern is repeated for the draft's prefill and target_verify propagations (1263, 1281).

**Fix:** Add `void ForwardPropagation::reserve_staged_inputs(Index inputs_number)` (or do it inside ForwardPropagation::set when an InferenceShapePolicy with sequence_capacity is given) that allocates the staging buffers from the network's input shapes/dtype, and delete initialize_cuda_input from chat.cpp.

*Verifier:* chat.cpp:1133-1140 resizes ForwardPropagation::staged_input_storage/staged_inputs (public members, forward_propagation.h:125-126) and sizes [0] to sequence_capacity*sizeof(float); called at 1137 (impl), 1263 (draft prefill) and 1281 (target_verify). neural_network.cpp:1314-1365 is the only other code that manages that layout (resize to input count, grow_to per input). Coupling as described; -6…

#### layers-a-15 — Dense::set duplicates set_activation_function's Softmax-to-Sigmoid demotion and skips the label on empty shapes

`opennn/neural_network/layers/dense_layer.cpp:326-378` · low · boilerplate · lines -6 · effort S · risk low · confirmed

Lines 345-348 repeat the four lines of set_activation_function (369-378) verbatim, and the early return at 332-337 for both-shapes-empty skips set_label and the activation assignment, so Dense({}, {}, "ReLU", false, "hidden") ends up labelled "my_layer" with the operator's default activation. The registry's default construction goes through this path (registry.cpp:103) and is only rescued by from_JSON setting both afterwards; a user who configures shapes later via set_input_shape/set_output_shape keeps the wrong label and activation.

**Fix:** Reorder set(): assign shapes, batch_norm.features and set_label(new_label) unconditionally, then call set_activation_function(new_activation_function) (which performs the demotion and the single configure_operators). The early return then only needs to skip configure_operators when both shapes are empty, or not at all since configure_operators already tolerates zero features.

*Verifier:* dense_layer.cpp:332-337 returns early for both-empty shapes before set_label and the activation assignment; 345-348 repeats the Softmax->Sigmoid demotion found verbatim in set_activation_function (369-378), which also calls configure_operators. Dense's constructor defaults are empty shapes (dense_layer.h:25-29), so registry default construction (registry.cpp:103, construct_layer<Dense>) takes the…

#### operators-a-12 — Dropout CPU backward duplicates the forward's mask loop, serially, re-evaluating delta.size() per iteration

`opennn/neural_network/operators/dropout_operator.cpp:102-113` · low · duplication · lines -6 · effort S · risk low · confirmed

dropout_forward_cpu and dropout_backward apply the identical `x *= mask ? keep_scale : 0` loop; the forward version is OpenMP-parallel above 65536 elements, the backward copy is serial and calls delta.size() in the loop condition. Dropout sits on the Dense hot path, so the backward pass is the slower twin for no reason.

**Fix:** Factor a file-local `apply_dropout_mask_cpu(TensorView& values, const TensorView& mask, float keep_scale)` with the parallel loop from dropout_forward_cpu, and call it from both dropout_forward_cpu (after set_random_bernoulli) and dropout_backward.

*Verifier:* dropout_operator.cpp:46-52 (forward, omp parallel above 65536) and 69-73 (backward, serial, `delta.size()` evaluated per iteration) apply the identical `*= mask ? keep_scale : 0` loop. Factoring a file-local helper keeps the static-twin layout of the file. LOC -6 plausible.

#### nn-builders-chat-11 — get_tokenizer_layer uses try/catch around get_layer(label) as control flow; chat.cpp:911 calls get_layer_index for its throw

`opennn/neural_network/standard_networks.cpp:1735-1756` · low · design · lines -6 · effort S · risk low · confirmed

NeuralNetwork has no non-throwing lookup by label: get_layer(const string&) and get_layer_index both throw runtime_error when the label is absent (neural_network.cpp:658-684). get_tokenizer_layer therefore wraps get_layer in try/catch(exception) only to rethrow with a nicer message (8 lines of ceremony, and any other exception from get_layer is swallowed), and make_decoder_only_state (chat.cpp:911) calls `network.get_layer_index("embedding");` discarding the result purely so the generic 'Layer not found: embedding' fires — a reader cannot tell that line is a validation.

**Fix:** Add `Layer* NeuralNetwork::find_layer(const string&) const noexcept` (ranges::find_if on the label, nullptr when absent), implement get_layer(label)/get_layer_index on top of it, and use it in get_tokenizer_layer (drop the try/catch) and in make_decoder_only_state (`throw_if(!network.find_layer("embedding"), "ChatSession: decoder has no 'embedding' layer.")`).

*Verifier:* neural_network.cpp:658-684: get_layer(label) throws runtime_error('Layer not found in neural network'), get_layer_index throws after special-casing Dataset/decoder/input. No find_layer/has_layer exists (grep neural_network.h empty). get_tokenizer_layer (1735-1756) wraps get_layer in try/catch(const exception&){} purely to rethrow with a nicer message; chat.cpp:911…

#### selection-testing-15 — registry: construct_layer<T> duplicates construct<Layer,T>; two hand-rolled 'Component not found' throws bypass throw_if and mislabel vision-disabled layers

`opennn/registry.cpp:60-75` · low · boilerplate · lines -6 · effort S · risk low · confirmed

`construct<Base, Class>` (lines 60-64) and `construct_layer<Class>` (71-75) have identical bodies; the latter is just construct<Layer, Class>. Lines 188-192 and 233-236 both build `throw runtime_error(format("Component not found: {}", name))` by hand although the repo convention (prior audit, AGENTS) is throw_if; the create_layer variant also reports a layer that exists but whose factory is nullptr under OPENNN_NO_VISION as 'not found', which hides the actual cause from a Neural Designer user loading a vision model on a no-vision build.

**Fix:** Delete construct_layer and write `construct<Layer, X>` in the table (or alias `template<class T> constexpr auto construct_layer = construct<Layer, T>;`). Replace both manual throws with `throw_if(it == factories.end(), "Component not found: {}", name)` and, in create_layer, `throw_if(!registration, "Unknown layer type: {}", name); throw_if(!registration->factory, "Layer {} requires vision support (OPENNN_NO_VISION is defined).", name);`. registry_test.cpp:366 expects runtime_error; confirm…

*Verifier:* registry.cpp:56-60 `construct<Base, Class>` and :70-74 `construct_layer<Class>` have identical bodies; construct<Layer, X> yields the same unique_ptr<Layer>(*)() that LayerFactory (:62) needs, and construct<Optimizer,...>/construct<InputsSelection,...> are already used that way at :247-260 (24 construct_layer uses would be renamed). Manual throws at :190-191 and :234-235 (line numbers shift by ~2…

#### selection-testing-14 — ROC analysis makes 100 full passes over the testing set plus two redundant confusion passes

`opennn/testing_analysis/testing_analysis.cpp:497-560` · low · overhead · lines -6 · effort M · risk low · partial

calculate_roc_curve loops 99 thresholds, each scanning all samples (O(100 n), OpenMP-parallel), and first calls calculate_positives_negatives_rate (a full confusion pass). perform_roc_analysis computes the same positives_negatives_rate again before calling calculate_roc_curve, and calculate_area_under_curve_confidence_limit(targets, outputs) does the pair a third time. Sorting the outputs once (O(n log n)) and using upper_bound per threshold gives identical 101-row output in O(n log n + 100 log n); for a 1M-row testing set that is ~1e8 compares and 2 extra passes saved per analysis (Neural Designer runs this on every classification report).

**Fix:** In calculate_roc_curve: build two sorted VectorR of outputs for target-positive and target-negative samples (target >= 0.5f, which is what the 0/1 targets reduce to), then for each threshold TP = positives.size() - lower_bound(positives, threshold), FP likewise on negatives; drop the omp pragma. Make perform_roc_analysis pass the rate it already computed into calculate_roc_curve (or have calculate_roc_curve return counts), removing the duplicate confusion passes.

*Verifier:* calculate_roc_curve (testing_analysis.cpp:497-560) does 99 full passes under `#pragma omp parallel for schedule(dynamic)` and calls calculate_positives_negatives_rate (a confusion pass) at :499; perform_roc_analysis computes the same rate again at :484. That part is confirmed. Overstatement: calculate_area_under_curve_confidence_limit(targets, outputs) (:576-585) is a separate public overload…

#### training-loss-15 — Loss JSON serialises regularization twice and the error name three times, with asymmetric readers

`opennn/training_strategy/loss.cpp:1904-1971` · low · design · lines -6 · effort S · risk medium · confirmed

TrainingStrategy::to_JSON writes `Error` = name, then Loss::to_JSON opens an element named get_name() containing `Method` = name again plus `Regularization`/`RegularizationWeight`, then regularization_to_JSON writes a second `Regularization` element with `Type`/`RegularizationWeight`. On read, Loss::from_JSON requires the in-element `Regularization` field (read_json_string returns "" when absent and string_to_regularization("") throws 'Unknown enum string'), while regularization_from_JSON is optional and silently overrides whatever from_JSON set. A file that carries only the dedicated Regularization element (the more explicit format) fails to load; a file with conflicting values loads the…

**Fix:** Make from_JSON read `Regularization`/`RegularizationWeight` only if present (`root->find`, as the other optional fields do) and drop the redundant `Method` field from to_JSON; keep writing the dedicated Regularization element. Verify against Neural Designer which of the two regularization locations its reader uses before removing the in-element copy from the writer.

*Verifier:* loss.cpp:1922-1944 to_JSON writes Method/Regularization/RegularizationWeight inside the get_name() element; training_strategy.cpp:129-133 writes 'Error' then loss->to_JSON then loss->regularization_to_JSON (loss.cpp:1913-1921, second 'Regularization' element). from_JSON (1958-1959) unconditionally reads 'Regularization'; read_json_string returns "" when absent (json.cpp:629-634) and…

#### training-optimizers-17 — train_full_batch re-implements the per-epoch display that display_epoch_results already provides

`opennn/training_strategy/optimizer.cpp:1104-1112` · low · duplication · lines -6 · effort S · risk low · confirmed

Lines 1106-1112 print Training error / Validation error / extra / Elapsed time by hand, which is display_epoch_results (739-769) minus the perplexity lines plus a hook. The two copies already diverge: the mini-batch path prints "Validation error: ---" for non-validation epochs, the full-batch path would not. Both paths also duplicate the `time_t beginning_time; time(&beginning_time); float elapsed_time` bookkeeping (901-903, 1070-1072).

**Fix:** Give display_epoch_results a trailing `const function<void()>& extra = {}` parameter (printed before Elapsed time) and call it from train_full_batch with has_validation, validation_fresh = has_validation, is_token_cross_entropy = false. Delete the hand-written block.

*Verifier:* optimizer.cpp:1104-1112 hand-prints Training error / Validation error / display_extra / Elapsed time; display_epoch_results (740-769) prints the same set plus perplexity lines and the "---" stale marker. Time bookkeeping duplicated at 901-903 and 1070-1072. In train_full_batch validation runs every epoch (1090-1101), so validation_fresh = has_validation is correct.

#### training-optimizers-11 — Dead or duplicate Optimizer members: restore_best has no setter, print() has no override or caller, set_loss duplicates set, get_batch_workers_number ignores its argument

`opennn/training_strategy/optimizer.h:46-321` · low · dead code · lines -6 · effort S · risk medium · confirmed

`bool restore_best = true;` (line 321) is protected, has no setter and no reader other than the guard at optimizer.cpp:1236, so the guard is constant-true (grep across opennn/, tests/, examples/, docs/benchmarks/: only those two sites). `virtual void print() const {}` (line 90) has no override in the four optimizers and no caller (the only print() calls are TrainingResult::print). `set(Loss*)` (46) and `virtual set_loss(Loss*)` (48) do the same thing and nothing overrides set_loss. `get_batch_workers_number(const NeuralNetwork&)` (229) returns workers_number and ignores its parameter at all four call sites (optimizer.cpp:376, 390, 1679, 2093).

**Fix:** Delete restore_best and its guard term (or, if the option is wanted, add set_restore_best and keep it), delete print(), keep one of set/set_loss, and drop the unused parameter from get_batch_workers_number (four call sites). Verify against Neural Designer before removing set_loss/print.

*Verifier:* grep across opennn/, tests/, examples/, docs/benchmarks/: restore_best appears only at optimizer.h:321 and optimizer.cpp:1236. `virtual void print() const {}` at optimizer.h:90 has no override (`grep 'void print() const' opennn/training_strategy/` finds only that line) and the only print() calls are results.print() (TrainingResult, optimizer.cpp:1001,1136) and tests/training_result_test.cpp:17.…

#### training-optimizers-18 — display_epoch_results takes 9 positional scalars including 3 adjacent bools; train() keeps the same four floats as loose locals

`opennn/training_strategy/optimizer.h:205-208` · low · design · lines -6 · effort S · risk low · confirmed

display_epoch_results(Index, float, float, float, float, bool, bool, bool, float) cannot catch a swapped has_validation/validation_fresh/is_token_cross_entropy at the call site (optimizer.cpp:980-982), and train() mirrors the four error/accuracy values as separate locals (860-863) that are assigned field-by-field from two EvaluationResults (947-948, 970-971). check_stopping_condition has the same shape (7 positional params, optimizer.h:158-159).

**Fix:** Keep the two Loss::EvaluationResult values in train() (training_evaluation_result / validation_evaluation_result, already named) and pass them by const& together with a small `EpochDisplayFlags { bool has_validation, validation_fresh, token_cross_entropy; }` - or simply make display_epoch_results take `(Index epoch, const Loss::EvaluationResult& training, const Loss::EvaluationResult* validation /*null = stale*/, bool token_cross_entropy, float elapsed)`; that removes the four loose locals and…

*Verifier:* optimizer.h:205-208 signature with three adjacent bools; call site optimizer.cpp:980-982; loose locals 860-863 assigned from EvaluationResults at 947-948 and 970-971; check_stopping_condition optimizer.h:158-159 with 7 positional params. Design-level, low severity as stated.

#### training-optimizers-15 — OptimizerData is a union of every optimizer's private scalars, declared in training_result.h

`opennn/training_strategy/training_result.h:23-40` · low · design · lines -6 · effort M · risk medium · confirmed

Verified by grep: iteration is read only by Adam, current_learning_rate only by SGD, initial_learning_rate/training_slope/learning_rate/old_learning_rate/training_direction only by QN, damping_parameter only by LM, potential_parameters by QN and LM. Each optimizer therefore carries seven fields it never touches and a reader of SGD has to know that `learning_rate` is not its learning rate (that is `current_learning_rate`). The struct also lives in training_result.h although it has nothing to do with results and every consumer includes optimizer.h.

**Fix:** Keep OptimizerData as {data, views, set()} and move it to optimizer.h; move the scalars into the optimizer that owns them (Adam: step counter; SGD: current_learning_rate; QN: a private LineSearchState {initial, learning_rate, old_learning_rate, slope, direction, potential}; LM: damping_parameter + potential_parameters), reset in setup_optimizer_data / setup_state exactly where they are reset today. One optimizer trains at a time per object, so per-object state is equivalent to per-run state.

*Verifier:* training_result.h:23-39 defines OptimizerData with the listed scalars. grep of optimization_data./optimizer_data. field accesses: iteration only in adaptive_moment_estimation.cpp, current_learning_rate only in stochastic_gradient_descent.cpp, initial_learning_rate/training_slope/old_learning_rate/training_direction only in quasi_newton_method.cpp, damping_parameter only in…

#### operators-a-14 — link_states hand-rolls what the promoted link_views helper already does

`opennn/neural_network/operators/batch_norm_operator.cpp:115-121` · low · boilerplate · lines -5 · effort S · risk low · confirmed

The prior audit promoted link_views to operator.h and converted 23 link bodies, and BatchNormalizationOperator::link_parameters two functions above already uses it with the same invalidate-on-success shape. link_states still spells out the size check and the two assignments by hand, and CombinationOperator/ConvolutionOperator::link_parameter_scales repeat the same `views[use_bias && views.size() >= 2 ? 1 : 0]` selection (combination_operator.cpp:139-143, convolution_operator.cpp:314-318) even though Layer::redistribute_parameters_to_operators always passes exactly parameter_specs().size() views.

**Fix:** `if (link_views(views, {&running_mean, &running_variance})) invalidate_inference_cache();` for link_states; for the two link_parameter_scales bodies, `weight_scale = views.empty() ? TensorView{} : views.back();` since the weight view is always the last spec of both operators.

*Verifier:* batch_norm_operator.cpp:115-121 hand-rolls the size check and two assignments while link_parameters at 104-108 two functions above already uses `if (link_views(...)) invalidate_inference_cache();`. combination_operator.cpp:52-56 and convolution_operator.cpp:314-318 both select `use_bias && views.size()>=2 ? views[1] : views[0]`; Layer::redistribute_parameters_to_operators (layer.cpp:81-100)…

#### xcut-api-9 — set_batch_size duplicated in Adam and SGD while batch_size/get_batch_size live on the base; forces downcasts

`opennn/training_strategy/adaptive_moment_estimation.h:27-27` · low · duplication · lines -5 · effort S · risk low · confirmed

Optimizer owns the batch_size member (optimizer.h:79 get_batch_size, member at the bottom) but the setter is written twice, identically, in AdaptiveMomentEstimation (h:27) and StochasticGradientDescent (h:29). Because the setter is not on the base, every caller that sets a batch size must downcast the Optimizer* returned by TrainingStrategy::get_optimization_algorithm(): 13 dynamic_cast<AdaptiveMomentEstimation*>/<StochasticGradientDescent*> sites across examples and benchmarks, all dereferenced without a null check. The iris and mnist examples perform the downcast and then call only base-class setters (set_maximum_epochs, set_display_period), so the cast is pure ceremony there; the airfoil…

**Fix:** Move set_batch_size to Optimizer next to get_batch_size and delete the two copies. Then drop the downcasts in iris/mnist (call training_strategy.get_optimization_algorithm()->set_maximum_epochs(...) directly) and replace the C-style (Clamping*) cast in airfoil with dynamic_cast.

*Verifier:* adaptive_moment_estimation.h:27 and stochastic_gradient_descent.h:29 are identical one-liners writing the base member batch_size (getter on optimizer.h:79). 25 lines of dynamic_cast<AdaptiveMomentEstimation*>/<StochasticGradientDescent*> across opennn/tests/examples/benchmarks (the auditor's 13 sites undercounts but the point stands). iris_plant/main.cpp:41-42 casts and then only calls…

#### xcut-build-tests-19 — tests/pch.cpp and opennn/pch.cpp are vestigial one-line TUs; tests/CMakeLists exists partly to exclude one

`tests/CMakeLists.txt:18-22` · low · dead code · lines -5 · effort S · risk low · confirmed

`target_precompile_headers` needs no .cpp. tests/pch.cpp (`#include "tests/pch.h"`) is globbed and then removed with `list(REMOVE_ITEM ...)`; opennn/pch.cpp (`#include "opennn/pch.h"`) is globbed into the library and compiled as an empty TU on every build. `set(PCH_HEADER pch.h)` is an indirection used once.

**Fix:** Delete tests/pch.cpp and opennn/pch.cpp, drop line 22 and inline `pch.h` into the target_precompile_headers call.

*Verifier:* tests/pch.cpp is a single `#include "tests/pch.h"`, opennn/pch.cpp a single `#include "opennn/pch.h"` (cat). tests/CMakeLists.txt:18 `set(PCH_HEADER pch.h)`, 22 `list(REMOVE_ITEM ... pch.cpp)`, 28 `target_precompile_headers(opennn_tests PRIVATE ${PCH_HEADER})`; opennn/CMakeLists.txt:5 globs *.cpp so opennn/pch.cpp is compiled as an empty TU. Fix is sound.

#### core-types-10 — multiply accepts the flattened rank-3 x rank-2 product only on CUDA, so linear_forward_transposed keeps a CPU-only Eigen branch

`opennn/core/tensor_operations.cpp:1117-1124` · low · design · lines -4 · effort S · risk low · confirmed

The validation gate is `input_a.is_cuda() && rank_a > 2 && rank_b == 2`; on CPU the same shapes fall into the batched branch and throw 'batched operands and output must have matching ranks'. The one public caller that needs the flattened product, linear_forward_transposed (lines 1526-1528), therefore calls multiply on CUDA and hand-writes an as_flat_matrix GEMM on CPU - the same operation expressed twice, and a public op whose accepted shapes depend on the device. multiply_cpu (line 1046) also re-derives `input_a.size() / (shape[rank-2] * shape[rank-1])` instead of calling the matrix_count helper defined at line 942.

**Fix:** Drop `input_a.is_cuda()` from the gate, and in multiply_cpu handle `rank_b == 2 && rank_a > 2` with one `output.as_flat_matrix().noalias() = alpha * (input_a.as_flat_matrix() * B) + beta * ...` GEMM (transpose_b honoured, transpose_a already rejected). Then linear_forward_transposed ends with the single `multiply(input, false, embed_weight, true, output)` call for both devices, and multiply_cpu uses matrix_count(input_a). Covered by tests/core/linear_forward_transposed_test.cpp.

*Verifier:* tensor_operations.cpp:1117 `flattened_cuda_rhs = input_a.is_cuda() && rank_a > 2 && rank_b == 2`; CPU rank-3 x rank-2 falls into :1127 and throws 'batched operands and output must have matching ranks'. linear_forward_transposed :1526-1528 has the CUDA multiply call and a hand-written Eigen flat GEMM for CPU. multiply_cpu :1044-1046 re-derives batch_count instead of calling matrix_count…

#### r2-batch-pipeline-and-device-gather-9 — Device gather re-allocates DeviceGather::row_indices on every fill and memcpy's it into the pinned index buffer at upload time

`opennn/dataset/dataset.cpp:1083-1087` · low · overhead · lines -4 · effort S · risk low · confirmed

Dataset::start_device_gather does `batch.device_gather.emplace()` (destroys and re-constructs the optional, freeing the previous vector) then `row_indices.resize(n)` (malloc) on every batch fill in the worker, and upload_to_device_batch_async then memcpy's the vector into the already pre-sized pinned buffer `gather_indices_host` (batch.cpp:300-304) before the async copy. The grouped resident path additionally copy-assigns the optional into the slot (`slot.device_gather = host_batch->device_gather`, optimizer.cpp:1536/1553). The pinned buffer is sized `batch_size * sizeof(int)` in Batch::set for exactly this purpose, so the intermediate vector is a pure detour: one free + one malloc + one…

**Fix:** Drop `vector<int> row_indices` from DeviceGather (it becomes a trivially-copyable POD of Index fields); have start_device_gather write the ints straight into `batch.gather_indices_host.as<int>()` (already pinned and sized) and reset the optional only when disengaged (`if (!batch.device_gather) batch.device_gather.emplace();`). upload_to_device_batch_async loses the memcpy; the grouped path copies the POD plus a `memcpy(slot.gather_indices_host, host_batch->gather_indices_host, n*sizeof(int))`…

*Verifier:* dataset.cpp 1078-1092: device_gather.emplace() re-constructs the optional (freeing the previous vector) then row_indices.resize(); batch.cpp 300-304 memcpys it into gather_indices_host, which set() already sizes to batch_size*sizeof(int) when the dataset uses residency (188-196). Grouped path copy-assigns the optional (optimizer.cpp 1536, 1553). Writing the ints straight into gather_indices_host…

#### dataset-b-8 — Cold YOLO start hashes every image and label file three times and lists the directory five times

`opennn/dataset/yolo_dataset.cpp:1292-1308` · low · overhead · lines -4 · effort S · risk low · confirmed

hash_sources (191-248) lists the images directory and does file_size + last_write_time on every image and every label file. On a cache miss, open_or_build_cache runs it in try_open_cache (1485), again in try_rebuild_target_from_boxes (1328) and again in build_cache (1576); list_files of the images directory additionally runs in set() (1273) and build_cache (1555). For a 100k-image dataset that is ~600k extra stat calls and five directory scans before the first image is decoded, on a network share this is seconds to minutes.

**Fix:** Compute `const uint64_t sources_hash = hash_sources(images_directory, labels_directory);` once in open_or_build_cache (or set()) and pass it to the three functions; have build_cache reuse image_filenames instead of calling list_files again.

*Verifier:* hash_sources (191-248) calls list_files plus file_size and last_write_time per image and per label. On a miss it runs in try_open_cache (1485), try_rebuild_target_from_boxes (1328) and build_cache (1576); list_files of the images directory also runs in set() (1273, stored in image_filenames) and again in build_cache (1554). Computing the hash once in open_or_build_cache and reusing…

#### r2-arena-planner-and-propagation-structs-6 — ForwardPropagation ctor and set() disagree on arity/position of the same positional bool

`opennn/neural_network/forward_propagation.h:58-74` · low · API · lines -4 · effort S · risk low · unverified

The constructor is (batch, net, mode, policy, bool inputs_pre_scaled, lifetimes) while set() is (batch, net, Buffer* external_storage, mode, policy, bool inputs_pre_scaled, lifetimes): the same bool is the 5th argument in one and the 6th in the other, preceded by a nullable pointer. All library callers pass it as a bare positional `true` (training_context.cpp:34, optimizer.cpp:849, 1052), so a caller switching from ctor to set (as optimizer.cpp:844-849 does) has to count arguments; a misplaced `true` converts to a non-null Buffer* or an enum silently only by luck of the types, and `ForwardPropagationMode`/`InferenceShapePolicy` are both defaulted so partial argument lists compile either way.

**Fix:** Move `inputs_pre_scaled` into the policy struct (rename InferenceShapePolicy -> ForwardPolicy with `bool inputs_pre_scaled = false;`), so both signatures become (batch, net, [external_storage], mode, policy, lifetimes) with the pointer as the only difference, and callers write `ForwardPolicy{.inputs_pre_scaled = true}` instead of a bare `true`. Update the four call sites; the ctor keeps forwarding to set.

#### layers-a-13 — Embedding and Normalization3d re-parse dimension fields the base already applied; Embedding writes OutputDimensions twice

`opennn/neural_network/layers/embedding_layer.cpp:59-91` · low · boilerplate · lines -4 · effort S · risk low · confirmed

Layer::to_JSON already writes InputDimensions and OutputDimensions and Layer::from_JSON already calls set_input_shape/set_output_shape with them (layer.cpp:186-227). Embedding::write_JSON_body emits a second "OutputDimensions" key into the same object (JsonWriter::add_field just calls parent->set, json.cpp:581-588, so the key is overwritten/duplicated rather than rejected) and Embedding::read_JSON_body re-reads it to obtain sequence_length/embedding_dimension instead of overriding set_output_shape; Normalization3d::read_JSON_body (normalization_layer_3d.cpp:105-109) re-reads "InputDimensions" and calls set() again even though apply_input_shape (lines 99-103) has just done exactly that. The…

**Fix:** Embedding: override set_output_shape(const Shape& s) { set(vocabulary_size, s.dim_or_zero(0), s.dim_or_zero(1), label); } (pairs with the apply_input_shape override from layers-a-2), drop the OutputDimensions line from write_JSON_body and read only VocabularySize + flags in read_JSON_body. Normalization3d: delete the three InputDimensions lines from read_JSON_body. Keep reading the legacy key if old files omit the base field (they do not: Layer::to_JSON has always written it).

*Verifier:* Layer::to_JSON writes InputDimensions/OutputDimensions (layer.cpp:215-224) and Layer::from_JSON calls set_input_shape/set_output_shape before read_JSON_body (186-200). Embedding::write_JSON_body (embedding_layer.cpp:82-91) adds a second "OutputDimensions"; JsonWriter::add_field is parent->set (json.cpp:581-588), so the key is overwritten rather than duplicated - same value today, but two writers.…

#### nn-core-16 — Line-search forward_propagate(inputs, parameters, fp) does three full parameter transfers and three link_parameters walks per evaluation

`opennn/neural_network/neural_network.cpp:1464-1495` · low · overhead · lines -4 · effort S · risk medium · partial

quasi_newton_method.cpp:302 and levenberg_marquardt_algorithm.cpp:366 call this overload up to 20 times per step. Each call snapshots the master (D2H when on CUDA), then set_parameters(new) (H2D + bf16 cast + full link_parameters walk), forward, set_parameters(saved) (H2D + cast + walk), and - because quasi-Newton reads get_parameters_map() which requires host storage - a migrate_to(CPU) + third walk in the restore block. link_parameters re-derives every slot through for_each_parameter_slot with std::function visitors; nothing about the layout changes between the two uploads.

**Fix:** Write the candidate straight into the existing master (`memcpy`/`copy_async` into parameters.data()) followed by cast_parameters_to_bf16() when the mirror exists, and restore the same way; the views are unchanged so no link_parameters call is needed. Keep the size check from set_parameters. Validate with the quasi-Newton/LM suites and the numerical-derivative tests.

*Verifier:* Read neural_network.cpp 1464-1495, 978-1017, 2508-2547, 2763-2782, quasi_newton_method.cpp 294-302 (get_parameters_map requires host; 20-iteration loop), levenberg_marquardt_algorithm.cpp 364-368. The transfer/re-link waste is real but the count is off: on a CPU-configured network there are no transfers at all - upload_host_vector's CPU branch early-returns in resize_bytes (same size/device) and…

#### response-opt-12 — resolve_cardinality_columns keeps a process-wide static warning set and an unused parameter

`opennn/response_optimization/response_optimization.cpp:741-757` · low · design · lines -4 · effort S · risk low · confirmed

`static std::set<string> warned_zero_excluded` is hidden global state: it is shared across every ResponseOptimization instance and model in the process (Neural Designer is long-lived), so the 'cannot take the value 0' warning is printed once per variable NAME per process and never again for a different model; it is also an unsynchronised container written from a const method, so two optimizations on different threads race on it. The `fixed_mask` parameter is explicitly discarded (`(void)fixed_mask;`) yet still computed and passed by the single caller (1014, 1020).

**Fix:** Move the warned set into SamplingMemory (already reset per solve_once) or compute force_on/force_off once per solve and warn there; drop the fixed_mask parameter from resolve_cardinality_columns and its declaration.

*Verifier:* resolve_cardinality_columns (741-757) contains `(void)fixed_mask;` at 749 and `static std::set<string> warned_zero_excluded;` at 757, inserted at 792 from a const method; the only caller (1014, 1020) computes fixed_mask and passes it. SamplingMemory is reset in solve_once (2256), so moving the set there gives per-solve warnings. LOC -4 sound.

#### training-optimizers-10 — QN records the pre-update training error in the history but displays and stops on the post-update one

`opennn/training_strategy/quasi_newton_method.cpp:226-240` · low · design · lines -4 · effort S · risk medium · confirmed

QN's train_step runs back_propagate and then update_full_batch_parameters in the same hook and returns {pre-update error, post-update error, post-update loss}. train_full_batch stores step.training_error (pre-update) in training_error_history(epoch) (optimizer.cpp:1086) but prints step.displayed_error (post-update, 1108), evaluates validation on the post-update parameters (1092-1101) and uses step.loss (post-update) for MinimumLossDecrease. Consequence: the number printed as "Training error" at epoch e differs from results.get_training_error()/training_error_history(e), and training and validation histories are offset by one update step. LM passes the same value twice…

**Fix:** Return the post-update error as training_error (it is the error at the parameters that validation and the best-model snapshot refer to), delete FullBatchStep::displayed_error and the two-element initialisers in QN and LM. Check the QN/LM tests that compare get_training_error against a threshold; they should still pass because the post-update error is lower or equal.

*Verifier:* quasi_newton_method.cpp:226-240: training_error captured before update_full_batch_parameters; the line search (291-317) writes back_propagation.metrics.error = evaluation_result.error on success, so the returned displayed_error and loss are post-update. optimizer.cpp:1086 stores step.training_error, 1108 prints step.displayed_error, validation (1092-1101) runs on the updated parameters, 1114 uses…

#### xcut-build-tests-18 — tests/pch.h force-defines NDEBUG and re-defines Eigen macros the opennn target already exports

`tests/pch.h:5-8` · low · build/test · lines -4 · effort S · risk low · partial

`#define NDEBUG` unconditionally means test TUs never have asserts even in a Debug configuration, while the library's inline `TensorView::as()` (opennn/core/tensor_types.h:593-597, `assert(data)`) is compiled with asserts in the library TUs: the same inline function has two different definitions across TUs in a Debug build (ODR), and test-side asserts are silently no-ops. `EIGEN_MAX_ALIGN_BYTES`, `EIGEN_NO_DEBUG` and `EIGEN_USE_THREADS` are already PUBLIC compile definitions of the opennn target (opennn/CMakeLists.txt:269-276) and CMake's Release config defines NDEBUG, so all four lines are redundant in the normal build and only matter when they disagree.

**Fix:** Delete lines 5-8; the definitions flow from the opennn target's usage requirements. Build Release and a Debug configuration of opennn_tests once to confirm.

*Verifier:* tests/pch.h:5-8 defines NDEBUG, EIGEN_MAX_ALIGN_BYTES 64, EIGEN_NO_DEBUG, EIGEN_USE_THREADS unconditionally; opennn/CMakeLists.txt:269-276 exports EIGEN_NO_DEBUG only for non-Debug configs and the other two unconditionally; tensor_types.h:595 has assert(data) in an inline member. Substance holds. Correction: tests/pch.h also includes opennn/pch.h (line 18), and a Debug test build would then…

#### xcut-build-tests-21 — OneDrive '-DESKTOP-' conflict-copy filters are machine-specific hygiene baked into three GLOBs

`opennn/CMakeLists.txt:6-9` · low · boilerplate · lines -3 · effort S · risk low · confirmed

The regex `-DESKTOP-[^/\\]*\.cpp$` exists to drop OneDrive sync-conflict copies (`file-DESKTOP-XXXX.cpp`) from the build and appears in opennn/CMakeLists.txt:8-9 and tests/CMakeLists.txt:21 (and is missing from the CUDA glob at 193, so a conflict copy of a .cu would still be compiled). No such file exists in the tree today. This is a property of one developer's sync setup leaking into the project build definition; if a conflict copy ever lands, it is a stray file in git that should be caught, not silently skipped.

**Fix:** Remove the three filters and add `*-DESKTOP-*` (and `*-LAPTOP-*`) to .gitignore so conflict copies never reach the tree; if the filter must stay, centralise it in one `opennn_glob_sources(<out> <patterns...>)` function used by both CMakeLists and the .cu glob.

*Verifier:* Filters at opennn/CMakeLists.txt:8-9 and tests/CMakeLists.txt:21; the .cu glob at 193 has none. `find opennn tests examples -name '*-DESKTOP-*'` returns nothing; .gitignore has no such pattern. Fix is sound.

#### r2-duplicated-kernels-across-folders-8 — MaxPoolGeometry::decompose re-implements kernel_common's nhwc_decompose with a channel-group factor

`opennn/core/cuda/kernel_pooling.cuh:32-41` · low · duplication · lines -3 · effort S · risk low · partial

kernel_common.cuh:348-355 already provides nhwc_decompose(i, channels, width, height, n, h, w, c) - flat NHWC index to (n, h, w, c) - and kernel_concat.cu:23 and kernel_upsampling.cu:23/42 use it. MaxPoolGeometry::decompose (kernel_pooling.cuh:32-41) is the same four-way decomposition with the channel axis taken in groups of `vec`: it equals nhwc_decompose(gi, channels / vec, columns, rows, n, row, column, c0) followed by c0 *= vec. Two decompositions of one layout is one more thing for a reader to prove equal; since the existing finding at kernel_pooling.cuh:26 already asks to rewrite these lines for 32-bit div/mod, folding the body into nhwc_decompose (with an int-index overload) does…

**Fix:** Replace the body of MaxPoolGeometry::decompose with `nhwc_decompose(gi, channels / vec, columns, rows, n, row, column, c0); c0 *= vec;` (move nhwc_decompose above the struct or include order already allows it since kernel_pooling.cu includes kernel_common.cuh first; for the .cuh itself add the include). Combine with the int-index rewrite requested by the existing kernel_pooling.cuh:26 finding by giving nhwc_decompose an `int i` overload.

*Verifier:* Equivalence confirmed: MaxPoolGeometry::decompose (kernel_pooling.cuh:32-41) equals nhwc_decompose(gi, channels/vec, columns, rows, n, row, column, c0) followed by c0 *= vec (kernel_common.cuh:348-355; users kernel_concat.cu:23, kernel_upsampling.cu:23/42). The fix shape is wrong, though: kernel_pooling.cuh is included by the host-compiled pooling_layer.cpp (:11) and deliberately includes only…

#### core-device-14 — get_op_tensor_add_descriptor takes the lane mutex and touches the cuDNN handle for a ctor-created descriptor

`opennn/core/device_backend.cpp:35-41` · low · boilerplate · lines -3 · effort S · risk low · confirmed

Backend::get_op_tensor_add_descriptor calls backend.cudnn(0) before returning op_tensor_add_descriptor. The descriptor is created in the constructor and does not depend on the handle; cudnn(0) only locks lane_mutex and lazily creates the lane-0 cuDNN handle, which the caller (tensor_operations.cpp:1550 add_gpu, every non-fp32 tensor add) has already obtained through get_cudnn_handle() in the same expression. The extra lock per add is small but pure noise, and the method is the only static accessor in the class with a body.

**Fix:** return instance().op_tensor_add_descriptor; (the no-GPU test LibraryHandlesMatchBuild still sees nullptr because the constructor returns early before creating it).

*Verifier:* device_backend.cpp:36-41: get_op_tensor_add_descriptor calls backend.cudnn(0) (1180-1196: lane_mutex lock + lazy handle creation) and returns op_tensor_add_descriptor, which the constructor creates at 1139-1143 independently of any handle. Sole caller tensor_operations.cpp:1551 already evaluates device::get_cudnn_handle() in the same cudnnOpTensor call. tests/core/device_backend_test.cpp:345-361…

#### dataset-b-14 — set_multi_scale_heads and set_v8_mode update target_shape/target_record_floats but only one updates the target Variable

`opennn/dataset/yolo_dataset.cpp:1744-1779` · low · design · lines -3 · effort S · risk low · partial

set_v8_mode (1770-1779) refreshes variables[1].features after changing target_record_floats; set_multi_scale_heads (1744-1768) changes target_record_floats and target_shape but leaves variables[1].features at the single-scale value set in setup_metadata. After set_multi_scale_heads alone (examples/yolo/main.cpp:638,653; tests yolo_fpn_test.cpp:266), get_features_number(VariableRole::Target) and get_feature_indices(Target) disagree with get_target_shape().size(); consumers that size by features (loss.cpp:1319,1341 use get_features_number(Target) for error-type decisions) see the stale count. Sibling setters with different invariants invite the next mismatch.

**Fix:** Route both setters through one private `set_target_record_floats(Index)` that assigns target_record_floats, target_shape and variables[1].features together; call it from setup_metadata too.

*Verifier:* Confirmed that set_multi_scale_heads (1744-1768) updates target_record_floats and target_shape but not variables[1].features, while set_v8_mode (1770-1779) updates all three and setup_metadata (1722-1738) sets features = target_shape.size(). After set_multi_scale_heads alone (examples/yolo/main.cpp:638,653; yolo_fpn_test.cpp:266; yolo_dataset_test.cpp:380) get_features_number(Target) is stale.…

#### selection-testing-18 — GeneticAlgorithm loop leftovers: per-generation history growth on a pre-sized history and a duplicated role restore

`opennn/model_selection/genetic_algorithm.cpp:494-508` · low · boilerplate · lines -3 · effort S · risk low · confirmed

InputsSelectionResult input_selection_results(maximum_epochs) already allocates all four histories at maximum_epochs, yet every generation calls `resize_history(mean_training_error_history.size() + 1)`, growing them to 2*maximum_epochs with a NaN tail before the final `resize_history(epoch + 1)` trims them; the intermediate resizes are pure churn (4 conservativeResize copies per generation). Line 508 `dataset->set_variable_indices(original_input_indices, original_target_indices)` repeats what evaluate_population already does after every individual (line 235).

**Fix:** Delete line 494 and line 508. (The final resize at 557 still runs because the MaximumEpochs check fires on the last generation.)

*Verifier:* genetic_algorithm.cpp:473 `InputsSelectionResult input_selection_results(maximum_epochs)` and InputsSelectionResult::set (inputs_selection.cpp:41-47) allocate all four histories at maximum_epochs; :494 grows them by one every generation (four conservativeResize copies) so they reach 2*maximum_epochs before :557 trims to epoch+1. epoch < maximum_epochs always indexes within the initial allocation,…

#### layers-b-14 — Recurrent cuDNN path writes y into HiddenStates and then copies B×T×H into Output; LSTM writes y into Output directly

`opennn/neural_network/layers/recurrent_layer.cpp:405-411` · low · overhead · lines -3 · effort S · risk medium · confirmed

apply_gpu_cudnn_ always passes hidden_states as cuDNN's y and then, for return_sequences, issues copy(hidden_states, output) — a full B*T*H device-to-device copy per forward (and the fused fallback does the same at line 523). LongShortTermMemoryOperator::apply_gpu instead selects y_target = return_seq ? output : sequence_output_scratch (line 998) and its backward reads back forward_slots[return_sequences ? OutputSlot : HiddenStateSlot] (line 509). Recurrent's backward can do the same, since backward_uses_forward_output() is true by default and the Output slot is retained.

**Fix:** In apply_gpu_cudnn_: void* y = return_sequences ? output.get_data() : hidden_states.get_data(); pass y to cudnn_rnn_forward_, keep only the gather branch. In back_propagate pass forward_slots[return_sequences ? output_slots[0] : output_slots[1]] as the y view for the cuDNN backward (the fused fallback keeps using hidden_states for its t-1 gathers). Verify with the GpuComparison recurrent cases and a return_sequences GPU gradient check.

*Verifier:* recurrent_layer.cpp:393-411: apply_gpu_cudnn_ always passes hidden_states.get_data() as y, then copy(hidden_states, output) for return_sequences or gathers the last step; the fused fallback does the same copy at 522-523. LSTM writes y_target = return_seq ? output : sequence_output_scratch (998-999) and its backward reads forward_slots[return_sequences ? OutputSlot : HiddenStateSlot] (507).…

#### training-loss-13 — Minkowski pair detects GPU by two ad-hoc signals instead of input.is_cuda(), and p is unvalidated

`opennn/training_strategy/error_functions.cpp:350-375` · low · API · lines -3 · effort S · risk low · confirmed

minkowski_error rejects the GPU case by `throw_if(workspace_device, ...)` (a non-null scratch pointer), while minkowski_error_gradient takes an extra trailing `bool on_gpu = false` that the caller fills with `neural_network && neural_network->is_gpu()` (loss.cpp:1759-1760). Every sibling in the file decides on `input.is_cuda()`. The bool also makes the gradient signature differ from its error twin for no reason. Separately, `minkowski_parameter` has no setter or validation (only from_JSON at loss.cpp:1969-1970); a JSON value p < 1 makes `pow(x, p - 1)` return inf at any zero residual (line 374), producing inf/NaN deltas with no message.

**Fix:** Use `throw_if(input.is_cuda(), ...)` in both functions, drop the bool parameter from the header and the call site, and add `throw_if(minkowski_parameter < 1.0f, ...)` where the JSON value is read (and in a set_minkowski_parameter setter if one is added for API symmetry with the YOLO lambdas).

*Verifier:* error_functions.cpp:350-353 `throw_if(workspace_device, ...)` (workspace is non-null exactly when runs_on_gpu && error != Yolo, loss.cpp:1479-1485); 356-364 extra `bool on_gpu` filled at loss.cpp:1759-1760 with `neural_network && neural_network->is_gpu()`; header error_functions.h:34 carries the default. minkowski_parameter (loss.h:169, default 1.5) is only written by from_JSON…

#### r2-batch-pipeline-and-device-gather-13 — SGD CPU update forks an OpenMP region per step unconditionally; Adam guards it with if(parameters_size > 4096)

`opennn/training_strategy/stochastic_gradient_descent.cpp:131-148` · low · overhead · lines -3 · effort S · risk low · confirmed

AdaptiveMomentEstimation::update_parameters uses `#pragma omp parallel for if(parameters_size > 4096)` so tiny networks do not pay a fork/join per step; StochasticGradientDescent::update_parameters has two bare `#pragma omp parallel for` loops over the same parameter vector. For the small tabular networks typically trained with SGD on CPU (tens to hundreds of parameters) each step pays an OpenMP parallel region (several microseconds, plus thread wake-ups) to do a few hundred multiply-adds. Inconsistent sibling behaviour with a one-token fix.

**Fix:** Add `if(parameters_size > 4096)` to both SGD pragmas (match Adam); the momentum-free branch can simply be `parameters -= current_learning_rate * gradient;` (Eigen vectorised, no OpenMP needed), removing the loop.

*Verifier:* stochastic_gradient_descent.cpp 131-148: two bare '#pragma omp parallel for' over parameters_size; adaptive_moment_estimation.cpp 202: '#pragma omp parallel for if(parameters_size > 4096)'. The momentum-free loop is a plain axpy on Eigen VectorMaps (125-127), so 'parameters -= current_learning_rate * gradient;' is equivalent. -3 LOC correct.

#### xcut-build-tests-22 — CI dependency cache key ignores the files that pin the dependencies; stale '../datasets' comment

`.github/workflows/ci.yml:30-51` · low · build/test · lines -2 · effort S · risk low · confirmed

Both workflows key the `_deps` cache on `hashFiles('CMakeLists.txt')` only, but the dependency pins live elsewhere: zlib 1.3.2 and libjpeg-turbo 3.1.4 in opennn/CMakeLists.txt:60/103, googletest v1.18.0 in tests/CMakeLists.txt:7. Bumping any of them restores the old cache and re-downloads into it instead of producing a fresh key. The 'Run tests' step explains it runs from the repo root because tests load `../datasets/...`; a grep finds no test reading a repo-relative data path (all fixtures are generated under temp_directory_path), so the comment is stale and the ctest `add_test` needs no WORKING_DIRECTORY either.

**Fix:** `hashFiles('CMakeLists.txt', 'opennn/CMakeLists.txt', 'tests/CMakeLists.txt', 'cmake/flash_attention.cmake')` in both workflows; delete the stale comment (or run `ctest --test-dir build --output-on-failure`).

*Verifier:* ci.yml:34 and tinyml-parity.yml:60 key on hashFiles('CMakeLists.txt') only; pins are at opennn/CMakeLists.txt:60 (zlib 1.3.2), :103 (libjpeg-turbo 3.1.4), tests/CMakeLists.txt:7 (googletest v1.18.0). ci.yml:49-52 comment about '../datasets/...' - grep over tests/ for '../datasets' or '../data/' finds nothing.

#### core-kernels-4 — 64-bit div/mod per thread in index decomposition although the element count is already checked_int

`opennn/core/cuda/kernel_pooling.cuh:26-34` · low · overhead · lines -2 · effort S · risk low · confirmed

launch_elementwise/launch_elementwise_strided narrow n through checked_int, so every kernel they launch receives a count < 2^31, yet several kernels keep the loop counter as Index and decompose it with 64-bit division/modulo (a software routine on NVIDIA GPUs, several times the cost of the 32-bit one). Sites and counts per thread/element: MaxPoolGeometry::decompose (3 divmods per output/input group, used by both pooling kernels), swap_heads_kernel (4 per element, including the 16-byte vector path that is launched with n/vec16 elements), batchnorm_forward_apply_kernel and batchnorm_backward_apply_kernel (2 each: `gi % channel_groups`, `gi / channel_groups`), batchnorm_inference_kernel (`i %…

**Fix:** Make the kernels' count parameter and loop variable `int` (they already receive an int from the launch helpers) and decompose in 32-bit: `int gi`, `int rest`; in swap_heads_kernel compute d/q/p/b from `int i`; in the BN apply kernels `const int c0 = int(gi % channel_groups)` with int gi and only `const Index i = Index(gi / channel_groups) * channels + c0` in 64-bit; same for batchnorm_inference_kernel and embedding_forward_w8_kernel. Document on launch_elementwise* that kernels may take `int n`.

*Verifier:* launch_elementwise_strided (kernel_common.cuh:91-96) narrows through checked_int and passes an int; the kernels widen it back: max_pooling_forward/backward_kernel take `const Index groups` and loop `Index gi` (kernel_pooling.cu:18-27, 61-70) into MaxPoolGeometry::decompose (kernel_pooling.cuh:26-34, three 64-bit divmods); swap_heads_kernel (kernel_attention.cu:40-50) does four 64-bit div/mod on…

#### selection-testing-16 — InputsSelection::print() virtual no-op and the NeuronSelection alias have no users in opennn/, tests/, examples/ or docs/

`opennn/model_selection/inputs_selection.h:68-68` · low · dead code · lines -2 · effort S · risk medium · confirmed

`virtual void print() const {}` is never overridden by GrowingInputs/GeneticAlgorithm and never called (grep over opennn/, tests/, examples/, docs/ finds only the declaration). `using NeuronSelection = GrowingNeurons;` (growing_neurons.h:99) is likewise unreferenced; the only 'NeuronSelection' strings in the repo are the JSON element names in model_selection.cpp. Both are API surface that suggests behaviour that does not exist.

**Fix:** Delete both lines. Verify against Neural Designer first (grep its tree for `NeuronSelection` as a type and for `->print()` on an InputsSelection pointer); if ND uses the alias keep it with a comment naming ND as the consumer, per the prior audit's recommendation.

*Verifier:* inputs_selection.h:68 `virtual void print() const {}`: grep over opennn/model_selection, tests/model_selection and examples finds no `->print()` call on an InputsSelection and no override in growing_inputs.h / genetic_algorithm.h. `using NeuronSelection = GrowingNeurons;` (growing_neurons.h:99): the only other 'NeuronSelection' strings in opennn/, tests/, examples/, docs/ are the JSON element…

#### nn-builders-chat-14 — Per-token host-to-device 4-byte copies from stack locals force an extra stream sync each step

`opennn/neural_network/chat.cpp:1124-1131` · low · overhead · lines -2 · effort S · risk low · confirmed

stage_token and the host branch of DecoderSampler::sample_row (555-564) copy the sampled id to the device from a stack float and must synchronize immediately because the source dies at scope exit. In the speculative loop each catch-up token costs stage_token + run_draft_decode (1645-1654), i.e. an H2D copy, a full sync, then the graph launch; the non-speculative CPU-sampled path pays D2H + sync + H2D + sync per token. The D2H sync is unavoidable (the parser needs the id), the second one is not.

**Fix:** Keep a `device::PinnedBuffer pinned_token` next to token_device in Impl, SpeculativeDraft and DecoderSampler, write the float into it and issue copy_async from the pinned source without the trailing synchronize; stream ordering guarantees the following decode sees the value and the next write to the pinned slot only happens after the next D2H sync.

*Verifier:* stage_token (1124-1131) and the host branch of sample_row (555-564) copy from a stack float and synchronize immediately. Speculative loop 1645-1654 calls stage_token + run_draft_decode per catch-up token. DecoderSampler already owns a pinned_id PinnedBuffer (device_backend.h:248) for the D2H direction, so a pinned H2D slot is consistent with existing code. Stream ordering argument holds since the…

#### r2-duplicated-kernels-across-folders-5 — GQA value-tail zeroing issues one cudaMemsetAsync per batch-head instead of one pitched memset

`opennn/neural_network/layers/grouped_query_attention_layer.cpp:329-339` · low · overhead · lines -2 · effort S · risk low · confirmed

zero_grouped_attention_value_tail loops over batch_heads (= chunk * n_kv_heads) and enqueues one cudaMemsetAsync per head for the same tail_bytes at a fixed pitch of key_seq*head_dim*sizeof(T). That is exactly a 2D memset: cudaMemset2DAsync(values + valid_key_seq*head_dim, pitch, 0, tail_bytes, batch_heads, stream) does it in one call. It runs once per chunk on the materialized GEMM path (the fallback taken when the cuDNN SDPA graph declines at :617-623), so for batch 8 x 4 kv heads it is 32 launches where 1 suffices, on every causal prefill through that path.

**Fix:** Replace the loop with a single `CHECK_CUDA(cudaMemset2DAsync(values + Index(valid_key_seq) * head_dim, size_t(key_seq) * head_dim * sizeof(T), 0, tail_bytes, size_t(batch_heads), device::get_compute_stream()))`; optionally expose it as device::set_zero_2d_async beside set_zero_async so the raw call stays inside device_backend.cpp.

*Verifier:* grouped_query_attention_layer.cpp:328-339 loops batch_heads times issuing cudaMemsetAsync of the same tail_bytes at a fixed pitch of key_seq*head_dim*sizeof(T); that is a cudaMemset2DAsync with dst = values + valid_key_seq*head_dim, pitch = key_seq*head_dim*sizeof(T), width = tail_bytes, height = batch_heads. The sole caller is :403 inside the per-chunk loop of grouped_attention_gemm_gpu, which…

#### xcut-api-10 — Operator::compute_dtype is propagated by hand per layer while weights_dtype is propagated generically

`opennn/neural_network/layers/layer.h:169-181` · low · boilerplate · lines -2 · effort S · risk low · confirmed

Layer::set_compute_dtype pushes weights_dtype to every operator through the operators vector, but leaves Operator::compute_dtype (operator.h:66, a public field that parameter_specs()/state_specs() read in attention, c2psa, combination and convolution operators) to each layer's on_compute_dtype_changed override. Eight layers override it; three of them are the identical one-liner 'x.compute_dtype = get_compute_dtype();' (c2psa_layer.h:40, embedding_layer.h:65, grouped_query_attention_layer.h:143). A layer that forgets the override keeps its operators' specs at FP32 under a BF16 network with no diagnostic - the kind of dtype-path slip the base class could rule out, since it already iterates…

**Fix:** In the same loop add op->compute_dtype = compute_dtype; (before on_compute_dtype_changed() so layers that rebuild operators still win), then delete the three one-line overrides. The rebuild-style overrides (dense, convolutional, LSTM, recurrent, MHA) stay.

*Verifier:* layer.h:169-177 loops operators only for set_weights_dtype; Operator::compute_dtype is a public field (operator.h:66). The three identical one-line overrides are at c2psa_layer.h:40, embedding_layer.h:65, grouped_query_attention_layer.h:143; the other five (dense, convolutional, lstm, recurrent, MHA) rebuild. Only six operator sources read compute_dtype (attention, c2psa, combination,…

#### xcut-api-15 — Non-const get_parameter_views()/get_parameter_scales() hand out the layer's view vectors for resizing

`opennn/neural_network/layers/layer.h:187-191` · low · design · lines -2 · effort S · risk medium · partial

Layer returns vector<TensorView>& from the non-const accessors. The only legitimate structural mutation in the library is NeuralNetwork's slot rebuild (neural_network.cpp:2378-2379 emplace_back, 2463-2464 clear), which must keep the vectors in step with parameter_specs() and the operators' link_parameters. Every other library caller (standard_networks.cpp:527/876/1088, dense_layer.cpp:393, model_expression.cpp, selection_utilities.cpp, network_differential.cpp) only reads the views or writes through them, which the const overload already permits because TensorView::as<T>() const returns T*. Leaving the non-const version public means any consumer can push_back/erase/clear and desynchronise…

**Fix:** Remove the two non-const overloads; give NeuralNetwork what it needs through one protected/friend entry (e.g. friend class NeuralNetwork; or void reset_parameter_views() + push helpers used only by the slot rebuild). The ~12 test sites that bind vector<TensorView>& become const vector<TensorView>&. Verify against Neural Designer.

*Verifier:* layer.h:187-191 non-const accessors confirmed; neural_network.cpp:2378-2379 and 2463-2464 are the only structural mutations (emplace_back/clear); link_views at operator.h:23-30 returns false silently when the vector is short. Scope corrections: library callers binding non-const references that must become const auto& are six, not zero (model_expression.cpp:1083/1209/1318,…

#### nn-expression-18 — Recurrent::get_activation_function returns string, LSTM's returns the enum; exporter round-trips through strings

`opennn/neural_network/model_expression.cpp:1280-1283` · low · API · lines -2 · effort S · risk medium · confirmed

LongShortTermMemory::get_activation_function() returns const ActivationFunction& (lstm header 181) while Recurrent::get_activation_function() returns string (recurrent_layer.h:141), so the embedded exporter must parse the name back with ActivationOperator::from_string to reach the same enum its sibling hands over directly. Two layers with the same accessor name and different return types is the kind of inconsistency that invites a wrong call at the next export site; the only other caller (recurrent_layer.cpp:786 JSON writer) can call to_string itself.

**Fix:** In recurrent_layer.h return `const ActivationFunction&` like the LSTM (update the JSON writer to wrap it in to_string), then drop the from_string round trip here. Verify against Neural Designer for callers of the string form.

*Verifier:* recurrent_layer.h:141 `string get_activation_function() const { return ActivationOperator::to_string(recurrent_op.activation); }` vs long_short_term_memory_layer.h:181 `const ActivationFunction& get_activation_function() const noexcept`. model_expression.cpp:1280-1283 round-trips via ActivationOperator::from_string. In-repo callers of the Recurrent string form: only recurrent_layer.cpp:786 (JSON…

#### nn-core-10 — Pre-scaled input branch in forward_propagate duplicates passthrough_overrides and uses the wrong index

`opennn/neural_network/neural_network.cpp:1448-1451` · low · dead code · lines -2 · effort S · risk low · partial

When inputs are pre-scaled, ForwardPropagation::set clears the forward specs of the skipped Scaling layers (217-218), and bind_slots then resolves every consumer of a skipped layer through resolve_producer to the external input index and records it in passthrough_overrides (906-912). forward_propagate applies those overrides at 1435-1437 with the correct external index, and then lines 1450-1451 overwrite the same slot with pick_input(source_index) - the *position* of the source, not the external input it stands for. For every single-input network the two coincide (both 0), so the branch is dead; for a network whose layer reads only the second skipped Scaling layer it would substitute input…

**Fix:** Delete lines 1450-1451. PreScaledBoundaryLeavesTextInputPipelineActive and the optimizer suites cover the remaining path.

*Verifier:* Read neural_network.cpp 1435-1451, forward_propagation.cpp 213-218 (skipped Scaling layers have forward_specs cleared, execution_start_layer = count of leading skip_for_pre_scaled_input layers), 895-913 (bind_slots records passthrough_overrides with the external index from resolve_producer). Confirmed that every (layer, source) pair hit by 1450-1451 is also in passthrough_overrides, and the…

#### operators-b-13 — Byte-level decode does a hash lookup per byte through unordered_map byte_decoder although codepoints are < 324

`opennn/neural_network/operators/tokenizer_operator.cpp:1030-1044` · low · overhead · lines -2 · effort S · risk low · confirmed

byte_encoder maps bytes to codepoints 0..255 plus 256..323 (68 non-printable bytes), so the inverse is a dense table of 324 entries, but byte_decoder is an unordered_map<uint32_t, unsigned char> consulted once per UTF-8 codepoint of every decoded token. decode_token runs per generated token in chat; a flat array lookup removes the hashing and the map member/copies in clone().

**Fix:** Make byte_decoder an `array<int16_t, 324>` filled with -1 in the constructor (index = codepoint, value = byte); in decode_token check `cp < byte_decoder.size() && byte_decoder[cp] >= 0`.

*Verifier:* BytePairTokenizer ctor (tokenizer_operator.cpp:653-668): direct bytes keep codepoint b (<= 0xFF) and the 68 others get 256..323, so the maximum codepoint is 323 and a 324-entry dense table inverts it exactly. decode_token (1034-1044) does byte_decoder.find per codepoint of every token; header declares unordered_map<uint32_t, unsigned char> byte_decoder (tokenizer_operator.h:188) beside…

#### response-opt-16 — reselect_pareto_front computes the full n x n distance matrix twice

`opennn/response_optimization/response_optimization.cpp:1842-1871` · low · overhead · lines -2 · effort S · risk low · confirmed

reselect_pareto_front runs only when the Pareto front exceeds max_pareto_number (2000), i.e. n >= 2001. It computes `calculate_distances(objective_matrix)` (n^2 floats, 16 MB at n=2000, 100 MB at n=5000) and then calls local_outlier_factor(objective_matrix, 20), which recomputes the identical matrix at line 108 and adds an O(n^2) neighbour scan. The second matrix is live at the same time as the first.

**Fix:** Give local_outlier_factor an overload taking the precomputed distance matrix (the points are not otherwise used) and pass `distances` from reselect_pareto_front; keep the existing signature as a thin wrapper for the statistics test.

*Verifier:* reselect_pareto_front computes calculate_distances(objective_matrix) at 1842 and then calls local_outlier_factor(objective_matrix, 20) at 1871, which recomputes calculate_distances(points) at 108 while the first matrix is still live (used at 1873-1878). Only runs when points_number > maximum_number (1835). local_outlier_factor(points, k) is also used by tests/core/statistics_test.cpp:502, so…

#### training-loss-10 — TAL assignment and DFL gradient allocate heap vectors per (sample, ground-truth) pair and per positive cell

`opennn/training_strategy/loss.cpp:641-660` · low · overhead · lines -2 · effort S · risk low · confirmed

tal_assign_head allocates `assign_iou` per sample and `cands` (reserve(cells)) per ground-truth box per sample: B * max_gt allocations of cells*12 bytes per head per pass, and the pass runs twice per batch (training-loss-2). yolo_v8_gradient_kernel_tal allocates `vector<float> all_probs(4 * reg_max)` inside the positive-cell branch (line 893), i.e. once per assigned cell per batch. These are the innermost host loops of the v8 loss (and on MSVC they run at /Od, training-loss-3).

**Fix:** Hoist `cands` and `assign_iou` above the sample loop and clear()/assign() them per iteration; hoist `all_probs` above the cell loops (resize once per call). Pure lifetime moves, no semantic change.

*Verifier:* loss.cpp:636 `vector<float> assign_iou(size_t(cells), -1.0f)` per sample; 652-654 `vector<CellScore> cands; cands.reserve(size_t(cells))` per (sample, gt); 893 `vector<float> all_probs(size_t(4 * reg_max))` per positive cell inside yolo_v8_gradient_kernel_tal. Hoisting is a pure lifetime move.

#### xcut-api-8 — Optimizer::set(Loss*) and virtual set_loss(Loss*) are twins; the virtual one is bypassed and uncalled

`opennn/training_strategy/optimizer.h:46-48` · low · design · lines -2 · effort S · risk medium · confirmed

Optimizer exposes both set(Loss*) (non-virtual) and set_loss(Loss*) (virtual, forwards to set). TrainingStrategy - the only place that wires an optimizer to a loss - calls optimizer->set(loss.get()) (training_strategy.cpp:61, 68), and the constructor calls set(). set_loss has zero callers in the repo and zero overrides in the optimizer hierarchy, so today it is dead; but it is virtual, and the first derived optimizer that overrides it (the usual reason to make it virtual) will be silently bypassed by TrainingStrategy. Loss has the same shape (set(NeuralNetwork*, Dataset*) plus virtual set_dataset) but there the virtual is at least the one TrainingStrategy calls.

**Fix:** Keep one: delete set_loss (nothing calls or overrides it) or, if the virtual hook is wanted, delete set and make TrainingStrategy and the constructor call set_loss. Verify against Neural Designer which name it uses.

*Verifier:* optimizer.h:46 set(Loss*) non-virtual and h:48 virtual set_loss(Loss*) forwarding to set. Grep for set_loss across the four trees finds only TrainingStrategy::set_loss(const string&) and its callers; no call of Optimizer::set_loss(Loss*) and no override. TrainingStrategy wires via optimizer->set(loss.get()). Fix (delete one) is sound; ND check needed.

#### core-kernels-11 — Flash_bwd_params is default-initialised; set_parameters zeroes only the Flash_fwd_params base

`opennn/core/cuda/flash_attention.cu:93-98` · low · design · lines -1 · effort S · risk low · confirmed

backward() declares `FLASH_NAMESPACE::Flash_bwd_params parameters;` (indeterminate PODs) and set_parameters takes a `Flash_fwd_params&`, so `parameters = {}` value-initialises only the base subobject. Every bwd-only field FA2 currently declares (do/dq/dk/dv pointers and strides, accum pointers, dq_accum_split_stride, dsoftmax_sum, deterministic) is assigned afterwards at lines 249-273, which is why it works today - but nothing enforces that. Any FA2 upgrade that adds a bwd-only field (as dq_accum_split_stride and deterministic were added in earlier FA2 releases) would launch the kernels with that field uninitialised, with no compiler warning. Two characters fix it.

**Fix:** Declare `FLASH_NAMESPACE::Flash_fwd_params parameters{};` and `FLASH_NAMESPACE::Flash_bwd_params parameters{};` at the two call sites and delete `parameters = {};` from set_parameters (or keep it but document that callers must value-initialise).

*Verifier:* flash_attention.cu:93-97: set_parameters takes `Flash_fwd_params&` and does `parameters = {}`; backward() at 245 declares `FLASH_NAMESPACE::Flash_bwd_params parameters;` without an initialiser, so the bwd-derived members are indeterminate until the explicit assignments at 249-273 (do_*, dq_*, dk_*, dv_*, accum ptrs, dsoftmax_sum, dq_accum_split_stride, deterministic). forward() at 230 has the…

#### xcut-boilerplate-14 — `using type = float;` declared twice in opennn_types.h (global and inside namespace opennn)

`opennn/core/opennn_types.h:170-170` · low · dead code · lines -1 · effort S · risk low · confirmed

Line 139 declares `using type = float;` at global scope (the alias Neural Designer relies on per the prior audit's note `type(0)`), and line 170 declares it again inside `namespace opennn`, shadowing the global one. Both resolve to float so nothing breaks, but the inner alias is a dead duplicate that invites the two to drift.

**Fix:** Delete line 170 (keep the global alias for Neural Designer). Both build dirs must still compile.

*Verifier:* opennn_types.h:139 `using type = float;` at global scope (after `using namespace std; using Eigen::Index;`) and opennn_types.h:170 `using type = float;` inside `namespace opennn {` (opened at 141). Both float; inner one is redundant. Deleting line 170 leaves unqualified `type` inside namespace opennn resolving to the global alias.

#### r2-arena-planner-and-propagation-structs-10 — BackPropagation declares a virtual destructor but nothing derives from it

`opennn/neural_network/back_propagation.h:44-44` · low · boilerplate · lines -1 · effort S · risk medium · unverified

`virtual ~BackPropagation() = default;` is the struct's only virtual member; grep finds no `: BackPropagation` / `public BackPropagation` anywhere in opennn/, tests/, examples/ or docs/benchmarks/ (BackPropagationLM in levenberg_marquardt_algorithm.h is an unrelated struct with its own virtual dtor). ForwardPropagation, its sibling, is non-polymorphic. The virtual adds a vtable pointer to a value member of TrainingContext and, as a user-declared destructor, suppresses the implicit move operations for no reason.

**Fix:** Delete the line (or make it a non-virtual `~BackPropagation() = default;` only if a destructor declaration is wanted for documentation). Verify against Neural Designer that no product type derives from BackPropagation before removing the virtual.

#### operators-a-8 — Max pooling CPU forward heap-allocates a bool array per (sequence, step) via .eval()

`opennn/neural_network/operators/pool3d_operator.cpp:72-79` · low · overhead · lines -1 · effort S · risk low · confirmed

The inner loop materializes `(step_features > outputs.row(b).array()).eval()` into a dynamic Array<bool,1,Dynamic> for every batch element and every time step, then runs two select() passes over it. For a batch of 64 and sequence 512 that is 32,768 heap allocations and three passes over the features per step, inside an omp parallel region (allocator contention). A plain element loop keeping the running max and index needs no temporary and one pass.

**Fix:** Replace the three Eigen expressions with one loop over features: `for f: if (x > out[f]) { out[f] = x; if (is_training) idx[f] = step; }` (same arithmetic the CUDA kernel max_pooling_3d_forward_kernel uses), so the CPU and GPU paths are literally the same algorithm.

*Verifier:* pool3d_operator.cpp:72-79: `(step_features > outputs.row(b).array()).eval()` materializes a dynamic Array<bool> per (batch, step) inside the omp region, followed by two select() passes. A single fused loop keeping the running max and index is strictly simpler and allocation-free. Minor: the CUDA kernel name `max_pooling_3d_forward_kernel` did not match any symbol in opennn/core/cuda/*.cu, so that…

#### r2-set-vs-compile-device-ordering-9 — AGENTS.md's CPU check directory has no test binary; no CPU suite has run since the change that broke findings 1-2

`AGENTS.md:20-25` · low · build/test · lines 0 · effort S · risk low · unverified

AGENTS.md instructs that every library change be built and run in `build-consolidated` (CPU) and `build-resnet-capacity` (CUDA). build-consolidated/bin contains only a stray 17 KB Linux ELF file named `blank`; the only CPU test binaries on the machine are build-cpu-verification (2026-08-20 17:00) and build-consolidated.windows-stale-20260821 (2026-08-16), both older than the 2026-08-20 18:02 activation_operator.cpp edit that removed the is_cuda() guard, and the CUDA tree's binary is 2026-08-20 17:04 while 40 sources are newer. That is how findings 1, 2, 3 and 8 went unnoticed: the CPU-only regressions are invisible to the CUDA suite's DenseGeluTanh* tests on the stale binary, and the omp…

**Fix:** Point AGENTS.md at the tree that actually exists (build-cpu-verification) or re-create build-consolidated, remove the stray file, and add the CPU test build to CI (the prior audit added compile-only CI for benchmarks; the test target needs the same). No library code changes.

#### core-kernels-12 — norm_backward_launch calls raw, unchecked cudaMemsetAsync where device::set_zero_async is the sanctioned helper

`opennn/core/cuda/kernel_normalization.cu:1090-1095` · low · boilerplate · lines 0 · effort S · risk low · partial

The only two raw cudaMemsetAsync calls in the eight kernel files sit here, return values dropped; kernel_common.cuh declares opennn::device::set_zero_async(void*, Index, cudaStream_t) for precisely this and the sibling kernel_pool3d.cu:163 uses it. A failed memset (e.g. an invalid pointer after a workspace reshuffle) would go unreported until the next check_last_error, which then blames the following kernel launch.

**Fix:** Replace both with opennn::device::set_zero_async(dGamma, Index(D) * Index(sizeof(float)), stream) / same for dBeta. (If core-kernels-9's partials finalize is adopted, the same deterministic pattern could replace these atomics too, removing the memset entirely.)

*Verifier:* kernel_normalization.cu:1090-1095 confirmed: two raw cudaMemsetAsync with the return value dropped, and grep shows they are the only raw memsets in opennn/core/cuda/*.cu. The sanctioned helper is opennn::device::set_zero_async (declared kernel_common.cuh:18 and device_backend.h:204, implemented device_backend.cpp:640-650 with CHECK_CUDA); kernel_normalization.cu already includes device_backend.h…

#### core-types-11 — softmax_gpu and add_gpu build the same cuDNN descriptor two or three times per call

`opennn/core/tensor_operations.cpp:1600-1609` · low · overhead · lines 0 · effort S · risk low · confirmed

TensorView::get_descriptor (tensor_types.cpp:19-42) does cudnnCreateTensorDescriptor + cudnnSetTensor4dDescriptor + cudnnDestroyTensorDescriptor each time it is called. softmax_gpu passes `output.get_descriptor()` twice in one cudnnSoftmaxForward call (and again twice per chunk at 1630/1632); add_gpu's non-FP32 branch creates three. That is two or three heap-allocating driver-side objects per softmax per layer per batch for a descriptor that is identical within the call.

**Fix:** Hoist `const auto descriptor = output.get_descriptor();` (and `chunk_descriptor` in the chunk loop) and pass it for both x and y; same for add_gpu if any two operands share a shape/type. Pure call-count reduction, no numeric change.

*Verifier:* softmax_gpu (tensor_operations.cpp:1596-1638) calls output.get_descriptor() twice per cudnnSoftmaxForward and chunk.get_descriptor() twice per chunk; get_descriptor (tensor_types.cpp:19-42) creates, sets and (via RAII deleter) destroys a cudnn descriptor each call. add_gpu :1552-1554 builds three descriptors for input_1/input_2/output, which add() validation requires to share shape and type, so…

#### xcut-api-13 — Shape(rank, fill) constructor reads like Shape{a, b}: Shape(2, 3) is [3,3], Shape{2, 3} is [2,3]

`opennn/core/tensor_types.h:192-199` · low · API · lines 0 · effort S · risk medium · confirmed

Shape has a (size_t rank, Index value) fill constructor next to the initializer_list constructor. Parentheses never select initializer_list, so Shape(rows, cols) with two Index arguments silently narrows rows to size_t and builds a rank-rows shape filled with cols - the opposite of what the same tokens in braces mean, and the most natural thing to type for a 2-D shape. The only real users in the repo are the higgs CPU benchmark (lines 75 and 127), which has to write Shape(size_t(layers), hidden) - the explicit cast is there precisely to steer overload resolution - and one test exercising the negative-value throw. A named factory removes the trap at zero cost.

**Fix:** Replace the constructor with static Shape filled(size_t rank, Index value) (same body), update the two benchmark call sites and the test. Verify against Neural Designer for uses of the two-argument parenthesized form.

*Verifier:* tensor_types.h:192-199 Shape(size_t, Index) fill constructor next to the initializer_list constructor at 201-209. In-repo users are exactly docs/benchmarks/throughput/higgs/opennn_higgs_cpu.cpp:75 and :127 (both with the explicit size_t cast) and tests/core/tensors_test.cpp:238 (negative-value throw). A static factory is a sound replacement; ND check required as stated.

#### r2-batch-pipeline-and-device-gather-10 — gather_rows_cuda caps column threads at 32, so wide rows (images) launch only batch/8 blocks

`opennn/dataset/kernel_gather.cu:56-62` · low · overhead · lines 0 · effort S · risk low · confirmed

The block shape was tuned for HIGGS (28 columns): `col_threads` doubles only while `< 32`, so for any row wider than 32 columns the kernel uses 32 column threads and 8 rows per block, i.e. `blocks = ceil(rows / 8)`. ImageDataset::enable_device_residency stages the whole image matrix on the device (MNIST 784 cols, CIFAR 3072 cols), so a batch of 64 or 128 images is gathered by 8 or 16 blocks on a GPU with 40-130 SMs, each thread issuing one 4-byte load per iteration across a 784-3072-wide row — latency-bound at a small fraction of memory bandwidth. The comment above the kernel only considers the narrow-row case.

**Fix:** Let col_threads grow up to block_threads: `while (col_threads < cols && col_threads < block_threads) col_threads *= 2;` — narrow rows keep the current packing (28 cols -> 32 threads, 8 rows/block), wide rows get one 256-thread block per row and `blocks = rows`. Optionally split rows wider than ~4096 across gridDim.y. Measure with OPENNN_PROFILE on an image dataset with GPUPersistantData.

*Verifier:* kernel_gather.cu 55-62: col_threads stops at 32, rows_per_block = 8, blocks = ceil(rows/8); the comment (19-26) only discusses the narrow HIGGS case. For 784/3072-wide rows a 64-128 batch launches 8-16 blocks with each thread striding the row — the kernel shape claim holds from the code alone. The fix (grow col_threads up to block_threads) keeps the narrow-row packing unchanged. I did not verify…

#### xcut-boilerplate-13 — Index/size_t cast churn has doubled since the prior audit (1412 casts); 108 in one file

`opennn/dataset/tabular_dataset.cpp:316-368` · low · design · lines 0 · effort S · risk low · partial

Pattern (e), new measurement. Today: 597 `Index(` + 692 `size_t(` + 123 static_cast<Index/size_t> = 1412 casts (the prior audit counted ~670 and declined the Layer/Operator seam unification). Top files: tabular_dataset.cpp 108 size_t(, yolo_dataset.cpp 49 size_t( + 37 Index(, chat.cpp 48+25, tensor_operations.cpp 50 Index(, grouped_query_attention_layer.cpp 42 Index( + 17 size_t(, neural_network.cpp 29+38, optimizer.cpp 18+39, forward_propagation.cpp 36+23, loss.cpp 33+13, model_expression.cpp 29 size_t(. The tabular_dataset.cpp hits are not at the Layer/Operator seam at all: ~80 of the 108 are `vector<T>[size_t(Index)]` subscripts on the column accumulators and cache_feature_* tables…

**Fix:** Pilot in one file: switch the six column accumulators in refresh_cache_statistics (316-368) and the cache_feature_replacement/descriptives/transforms tables to Eigen vectors (VectorR/VectorXd/VectorI, which take Index directly) and measure the cast count drop (~80). Only then decide whether the pattern is worth applying to yolo_dataset.cpp and chat.cpp. Do not touch the Layer/Operator seam.

*Verifier:* Counts reproduce: 598 `Index(`, 692 `size_t(`, 123 static_cast<Index|size_t> across opennn/ (1413), and tabular_dataset.cpp has 108 `size_t(`; lines 316-330 show the vector<float>/vector<double> accumulators indexed via size_t(j). Correction: the 'doubled since the prior audit' framing is wrong - docs/ENGINEERING_AUDIT.md:200-201 counted ~670 casts at the Layer/Operator seam only, not…

#### selection-testing-19 — ModelSelection exposes no way to choose GeneticAlgorithm except through JSON

`opennn/model_selection/model_selection.h:43-52` · low · API · lines 0 · effort S · risk low · confirmed

set_inputs_selection(const string&) is private and there is no other setter or accessor for the InputsSelection (get_inputs_selection was removed in the prior audit), so a library user holding a ModelSelection can only ever run GrowingInputs unless they serialise a JSON document with InputsSelectionMethod="GeneticAlgorithm" and call from_JSON. The registry already supports the name-based factory and tests exercise create_inputs_selection("GeneticAlgorithm"), so the restriction looks accidental rather than deliberate.

**Fix:** Move `void set_inputs_selection(const string&);` to the public section (no body change). Add a one-line test in model_selection_test.cpp that set_inputs_selection("GeneticAlgorithm") makes get_inputs_selection_name() return it.

*Verifier:* model_selection.h:43-45: `set_inputs_selection(const string&)` is under `private:`; the only public path is from_JSON (model_selection.cpp:101-102), and set_default hardcodes "GrowingInputs" (:34). create_inputs_selection("GeneticAlgorithm") exists in the registry (registry.cpp:259) and is exercised by registry_test.cpp:278. Moving the declaration to public is a zero-risk, zero-LOC change. Cannot…

#### nn-builders-chat-16 — Public sample_token mutates the caller's probability vector in place without signalling it

`opennn/neural_network/chat.h:58-60` · low · API · lines 0 · effort S · risk low · confirmed

sample_token takes `VectorR&` and sample_token_with_workspace divides penalised entries (167), raises everything to 1/T (172) and zeroes every non-kept entry for top-k/top-p (190, 216), so after the call the caller's distribution is destroyed; the signature reads like an in/out by accident and the only non-test caller (ClassicDecodeLoop::sample_at, 1339) does not need the mutation — the function already copies the input into workspace.original for its own fallback. A caller that samples twice from the same distribution (e.g. best-of-n) gets a silently different second draw.

**Fix:** Take `const VectorR& probabilities`, do the work on the workspace copy (rename `original` to `scratch` and let the fallback argmax read the untouched input); moot if nn-builders-chat-8 unifies the samplers. Verify the signature change against Neural Designer.

*Verifier:* chat.h:58 takes VectorR&; sample_token_with_workspace copies to workspace.original (164) and then mutates the argument at 167, 172, 190 and 216; only non-test caller is sample_at (1336-1341) which does not need the mutation. Tests (tokenizer_layer_test.cpp:45-124) never reuse a vector, so nothing pins either behaviour. Taking const& and working on the scratch copy is zero net LOC; ND signature…

#### r2-arena-planner-and-propagation-structs-7 — memory_debug::record arguments are formatted eagerly on every ForwardPropagation construction

`opennn/neural_network/forward_propagation.cpp:746-850` · low · overhead · lines 0 · effort S · risk low · unverified

memory_debug::record takes `const string&` and checks enabled() only inside state().record, so every call site pays the std::format (and get_label copy) even with OPENNN_MEMORY_DEBUG off: 17 calls in forward_propagation.cpp (13 unconditional in set(), one per layer in bind_slots at 846-850, one per recomputable layer at 585-593) and 6 in back_propagation.cpp. record_pool_lifetimes does guard with `if (!enabled()) return;` but its `format("layers={},batch={}", ...)` argument is still built by the caller. ForwardPropagation is constructed per calculate_outputs call (neural_network.cpp:1121, 1173, 1196, 1202, 1266), so for small CPU networks, where the arena is a cheap malloc, a few dozen…

**Fix:** Make `memory_debug::record` a variadic template in memory_debug.h that takes a format string plus args and formats only after `if (!enabled()) return;` (one helper, the 23 call sites change only by dropping their `format(` wrapper), or wrap the per-layer and per-recompute sites in `if (memory_debug::enabled())`. No behaviour change; the string work disappears from the default path.

#### layers-b-10 — RecurrentOperator::set and LongShortTermMemoryOperator::set take the same three Index parameters in different orders

`opennn/neural_network/layers/long_short_term_memory_layer.cpp:47-57` · low · API · lines 0 · effort S · risk low · confirmed

RecurrentOperator::set(input_features, time_steps, output_features, activation, compute_dtype) vs LongShortTermMemoryOperator::set(input_features, output_features, time_steps, activation, recurrent_activation). Three positional Index arguments of identical type in swapped order between twin operators is the kind of slip that compiles and only shows up as a shape mismatch at run time. LSTM's set also has no dtype parameter, so lstm_op.compute_dtype is never set by the layer (parameter_specs hard-codes FP32; see layers-b-1).

**Fix:** Align LongShortTermMemoryOperator::set to (input_features, time_steps, output_features, activation, recurrent_activation, Type compute_dtype = FP32) and update the single caller LongShortTermMemory::configure_operators; grep tests for direct operator calls (none found in opennn/tests).

*Verifier:* LongShortTermMemoryOperator::set (long_short_term_memory_layer.cpp:47-58) takes (input_features, output_features, time_steps, activation, recurrent_activation) and is called at 1135-1139; RecurrentOperator::set is called as (input_features, time_steps, output_features, activation, compute_dtype) at recurrent_layer.cpp:712-713. LSTM's operator indeed never receives a dtype (ties to layers-b-1). No…

#### nn-expression-14 — ModelExpression takes const NeuralNetwork* but const_casts to migrate parameters and relink layers

`opennn/neural_network/model_expression.cpp:328-331` · low · API · lines 0 · effort S · risk medium · confirmed

HostParametersGuard calls copy_parameters_host() (Buffer::migrate_to CPU, clear_low_precision_parameter_storage, resets transposed_inference_active on every combination operator, link_parameters()) and copy_parameters_device() on exit. Doing that through const_cast on a pointer the public API declares const means (a) a NeuralNetwork object actually defined const is modified, which is undefined behaviour, and (b) a caller exporting a GPU-resident model gets two full parameter migrations plus a relink as a hidden side effect of a 'const' export. The constness is a promise the class cannot keep.

**Fix:** Change the constructor to `ModelExpression(NeuralNetwork*)` and store a non-const pointer (all in-repo callers pass non-const networks; verify against Neural Designer before merging) and document that exporting a CUDA-resident network round-trips the parameters. Alternatively make HostParametersGuard read-only by exporting from a host copy, but that costs a full copy.

*Verifier:* model_expression.cpp:328-331 and 715-716 do `*const_cast<NeuralNetwork*>(neural_network)` to build HostParametersGuard, whose ctor/dtor call copy_parameters_host()/copy_parameters_device() on a NeuralNetwork& (neural_network.h:44-59). Constructor is `ModelExpression(const NeuralNetwork*)` (model_expression.h:26). The const promise is indeed not kept; UB only if the object itself is const, and the…

#### response-opt-11 — evaluate_rpn heap-allocates its stack per call; nonlinear filter path copies each row twice

`opennn/response_optimization/response_constraints.cpp:797-803` · low · overhead · lines 0 · effort S · risk low · confirmed

evaluate_rpn allocates a vector<float> (reserve(16)) on every call. It is called per row per constraint in filter_feasible_points' nonlinear path (2000 rows x K constraints per sample), per input index per active constraint per pass in constraint_gradient, and per row in row_satisfies_input_affine. The nonlinear filter additionally materialises `const VectorR input_row = inputs.row(r).transpose()` and the same for outputs (response_optimization.cpp:1234-1235) only because evaluate takes `const VectorR&`; MatrixR is RowMajor so a row is contiguous and a Ref<const VectorR> would bind without copying.

**Fix:** Compute the maximum stack depth once in compile_ast (store `Index stack_depth` in CompiledFormula) and evaluate on a `thread_local vector<float>` or a fixed `array<float, 64>` guarded by a throw_if at compile time. Change evaluate_rpn/CompiledFormula::evaluate to take `const Ref<const VectorR>&` so callers pass `inputs.row(r).transpose()` directly (the public callback signature can stay VectorR).

*Verifier:* evaluate_rpn (797-803) allocates vector<float> with reserve(16) per call; it is called per row per constraint in the nonlinear filter path (response_optimization.cpp:1230-1240 via row_satisfies_formula_constraints), per input_gradient program per active constraint per pass in constraint_gradient (1334-1335, 1346-1347), and per row in row_satisfies_input_affine (1752-1759). MatrixR is RowMajor…

#### training-loss-16 — YOLO block still inline: 1210 of loss.cpp's 1977 lines (61%) under one #ifndef OPENNN_NO_VISION

`opennn/training_strategy/loss.cpp:29-1238` · low · design · lines 0 · effort S · risk low · confirmed

The prior audit's Level 3 #1 split has not happened. The block (CIoU forward/grad, TAL, DFL, CPU/GPU drivers) is still one anonymous-namespace region at 29-1238; the Loss class proper starts at 1240. New detail relevant to the move: loss.h already declares YoloLambdas, yolo_error_kernel and yolo_gradient_kernel publicly (loss.h:216-246) for the tests, so loss_yolo.cpp needs no header change beyond leaving Loss::calculate_yolo (1392-1443) as the one member that lives there; and the two `#pragma optimize` islands (training-loss-3) would then be confined to that file.

**Fix:** Move lines 29-1238 plus Loss::calculate_yolo verbatim to opennn/training_strategy/loss_yolo.cpp (GLOB picks it up); loss.cpp keeps the include of detection_head.h only if get_output_delta_layer_indices still needs DetectionHeadEndpoint. No behaviour change; do it after training-loss-2 so the file moves once.

*Verifier:* loss.cpp:29 `#ifndef OPENNN_NO_VISION` / `namespace {`, closing `}` at 1237 and `#endif` at 1238, `Loss::Loss` at 1240 of 1977 lines. ENGINEERING_AUDIT.md Level 3 #1 lists exactly this split and it is not marked DONE. loss.h:216-246 already exposes YoloLambdas and the two kernels for tests; Loss::calculate_yolo (1392-1443) is the only member in the block's orbit. Zero-behaviour-change move.

#### training-optimizers-13 — Validation batch index lists are rebuilt from the same unshuffled indices on every validation epoch

`opennn/training_strategy/optimizer.cpp:956-960` · low · overhead · lines 0 · effort S · risk low · confirmed

get_batches(validation_sample_indices, validation_batch_size, false, validation_batches) is deterministic (shuffle=false) and the inputs never change during train(), yet it runs on every validation epoch (every epoch by default) and once more in the warm-up (line 877). The function resizes/assigns batches_number vectors over the whole validation index range; the same file already treats index slicing of large datasets as expensive enough to move the training version onto a helper thread (comment at 909-913).

**Fix:** Build validation_batches once before the epoch loop (reuse the warm-up's list when needs_cuda_warmup, otherwise build it there) and delete the per-epoch call.

*Verifier:* optimizer.cpp:956-958: inside the epoch loop `if (val_fresh) { dataset->get_batches(validation_sample_indices, validation_batch_size, false, validation_batches); ...}`; also at 877 in the warm-up. get_batches (dataset.cpp:91-125) is deterministic with shuffle=false and inputs are const locals of train(). Note the non-warmup (CPU) path has no list before the loop, so the one-time build must be…

#### xcut-build-tests-17 — Test files that violate the one-for-one mirror rule by name or folder

`tests/neural_network/response_optimization_test.cpp:1-1` · low · design · lines 0 · effort S · risk low · partial

AGENTS.md says a test sits at the same relative path as what it exercises. Mismatches that make the test hard to find from the source file: tests/dataset/timeseries_dataset_test.cpp (source: time_series_dataset.cpp); tests/neural_network/layers/activations_test.cpp (activation_layer.cpp); tests/core/tensors_test.cpp (tensor_types.cpp + tensor_operations.cpp); layers/c2psa_test.cpp (c2psa_layer.cpp); layers/grouped_attention_test.cpp and layers/grouped_query_attention_test.cpp both for grouped_query_attention_layer.cpp; layers/concatenation_operator_test.cpp (an operator test under layers/); operators/swiglu_test.cpp and operators/tokenizer_test.cpp without the `_operator` suffix; and…

**Fix:** `git mv` to the mirrored names/folders (timeseries->time_series_dataset_test, activations->activation_layer_test, tensors->tensor_types_test, c2psa->c2psa_layer_test, concatenation_operator_test->operators/, swiglu->swiglu_operator_test, tokenizer->tokenizer_operator_test, create tests/response_optimization/). CMake globs recursively so no build edits. Add `opennn/response_optimization/` to the AGENTS.md folder table.

*Verifier:* Verified mismatches: tests/dataset/timeseries_dataset_test.cpp vs opennn/dataset/time_series_dataset.cpp; layers/activations_test.cpp vs activation_layer.cpp; core/tensors_test.cpp; layers/c2psa_test.cpp vs c2psa_layer.cpp; grouped_attention_test + grouped_query_attention_test both for grouped_query_attention_layer.cpp; operators/swiglu_test and tokenizer_test vs…

#### nn-core-5 — Unconditional copy_states_device() per GPU forward rebuilds every operator's state views each batch

`opennn/neural_network/neural_network.cpp:1311-1311` · low · overhead · lines +1 · effort S · risk low · partial

The GPU branch of forward_propagate calls self->copy_states_device() on every call. states.migrate_to() is a no-op once resident, but link_states(Device::CUDA) is not: Layer::link_views_to_operators (layer.cpp:113-140) does views.clear(), calls (op->*specs_fn)() - which returns a freshly allocated vector<TensorSpec> per operator - re-emplaces every view and calls op->link_states(span) again. That is one allocation per operator plus the re-linking on every training batch and every inference call, while the state device can only change through copy_states_host/set_states/compile, all of which re-link themselves.

**Fix:** Guard it like the parameters: `if (states.get_device() != Device::CUDA) self->copy_states_device();`. An empty `states` buffer migrates to an empty CUDA buffer on the first call and is skipped afterwards, so behaviour is unchanged. Verify with the existing GPU suites plus the memory/allocation guard tests.

*Verifier:* Read neural_network.cpp 1308-1311, 2783-2789 (copy_states_device: migrate only if !states.empty(), then link_states(CUDA) unconditionally), 2489-2506, layer.cpp 113-140 (views.clear(), fresh vector<TensorSpec> per operator, re-link) and 145-151, tensor_types.h 487-499 (migrate_to no-op when resident). The redundant per-forward re-link is real. Corrections: the note 'an empty states buffer…

#### nn-core-17 — CPU calculate_outputs computes get_forward_specs(1) on every call even when tiling cannot apply

`opennn/neural_network/neural_network.cpp:1178-1182` · low · overhead · lines +1 · effort S · risk low · confirmed

The tiling probe calls get_forward_specs(1) (a full spec walk allocating per layer, plus force_specs_to_fp32) on every CPU inference call to derive tile_rows_max, but tile_rows_max is clamped to >= 16, so `batch_size > tile_rows_max` is false for every batch of 16 rows or fewer - the common interactive case, where the call then also constructs a ForwardPropagation that walks the same specs again.

**Fix:** Short-circuit before the probe: `const bool tileable = batch_size > 16 && [&]{ ...row_bytes/tile_rows_max...; return batch_size > tile_rows_max && ranges::all_of(...); }();` or hoist `if (batch_size <= 16) { ...direct path... }` above it.

*Verifier:* Read neural_network.cpp 1165-1190 and header 101-106 (get_forward_specs walks every layer, allocates per layer and runs force_specs_to_fp32 on CPU). tile_rows_max is clamped to >= 16 at 1181-1182, so `batch_size > tile_rows_max` is false for any batch <= 16 and the probe is wasted; the subsequent ForwardPropagation construction repeats the spec walk. Short-circuiting on batch_size <= 16 is…

#### core-device-13 — lane_stream(int) has no range check: a lane >= MAX_LANES indexes the std::array out of bounds

`opennn/core/device_backend.cpp:1147-1160` · low · API · lines +2 · effort S · risk low · partial

device::lane_stream is a public header function (device_backend.h:361) and convolution_operator.cpp calls it with literal 0 and 1. Backend::stream(lane) indexes lane_streams[lane] (std::array<cudaStream_t, 4>) without validating against MAX_LANES or lanes_available(), while the sibling set_active_lane validates. A lane of 4 or a negative lane is UB (and lazily creates a stream in out-of-bounds memory under lane_mutex); a lane above lanes_available() but below MAX_LANES silently creates a stream the block cache never records events on (give() loops only to lanes_available()), so blocks used on it can be recycled while still in flight.

**Fix:** Add the same throw_if(lane < 0 || lane >= lanes_available(), ...) at the top of device::lane_stream (one line, mirrors set_active_lane) so every lane that can own a stream is also one the block cache fences.

*Verifier:* Backend::stream (device_backend.cpp:1147-1156) indexes lane_streams[lane] (std::array<cudaStream_t, MAX_LANES>, line 60) with no range check; device::lane_stream (1079-1082) forwards unchecked while set_active_lane (1072-1077) validates against lanes_available(). give() (440-441) fences only lanes < lanes_available(). However the only in-tree caller, convolution_operator.cpp:667-679, only calls…

#### layers-b-12 — Anchor-based Detection GPU path rejects non-square grids that the CPU path and DetectionV8 (CPU+GPU) accept

`opennn/neural_network/layers/kernel_detection.cu:19-32` · low · API · lines +2 · effort S · risk low · confirmed

DetectionOperator::forward_propagate throws "non-square grids not supported" on CUDA (detection_layer.cpp:58-59) only because detection_box_base/detection_forward_cuda take a single grid_size; the CPU loops and detection_v8_cell_base already carry grid_width. A network that trains fine on CPU fails on GPU for rectangular feature maps, an inconsistency between sibling heads.

**Fix:** Add grid_width to detection_forward_cuda/detection_backward_cuda (header + both launches use batch*grid_size*grid_width*boxes) and to detection_box_base (col = t % grid_width; t2 = t / grid_width; cell = ((b*grid_size+row)*grid_width+col)*channels), pass grid_width from DetectionOperator, and delete the throw.

*Verifier:* detection_layer.cpp:58-59 throws on grid_size != grid_width for CUDA only; the CPU loops use batch_size*grid_size*grid_width (cells_count in both forward and backward). kernel_detection.cu:19-32 detection_box_base decodes with a single grid_size, and detection_forward_cuda (84) / detection_backward_cuda (140) take grid_size only, while detection_v8_cell_base (36-45) and the v8 launches (192-201,…

#### nn-core-4 — Every forward call re-materialises all parameter specs just to compare one integer

`opennn/neural_network/neural_network.cpp:1291-1292` · low · overhead · lines +2 · effort S · risk low · partial

NeuralNetwork::forward_propagate (3-arg, the path every optimizer batch takes: optimizer.cpp:1080/1431/1874/2022/2153...) starts with `get_aligned_size(get_parameter_specs())`. get_parameter_specs() -> collect_layer_specs allocates one vector<vector<TensorSpec>>, one vector<TensorSpec> per layer, and Layer::get_parameter_specs (layer.cpp:39-50) allocates one more vector per operator and move-inserts it. For a 50-layer network with 2-3 operators per layer that is ~150-250 heap allocations plus the spec computation per batch, purely to detect a shape change after compile(). The check is also incomplete in the way it claims to protect: a shape change after ForwardPropagation::set would leave…

**Fix:** Move the check into ForwardPropagation::set, right after `forward_specs` is obtained (it already walks the layers): `throw_if(neural_network->get_parameters_buffer_size() != get_aligned_size(neural_network->get_parameter_specs()), "Network shapes changed since compile(); call compile() again.");` and delete lines 1291-1292. Every forward runs through a ForwardPropagation that was set after the last shape change (otherwise its slots are stale anyway), so coverage does not shrink.

*Verifier:* Read neural_network.cpp 1287-1296, header 91-94 (get_parameter_specs -> collect_layer_specs), layer.cpp 39-50 (per-operator vector alloc + move-insert), forward_propagation.cpp 124-210. The per-forward allocation walk is real. Two corrections: (1) severity: on the GPU training path (~1 ms+ per batch) ~200 small host allocations are noise; it only matters for small-batch CPU inference -> low. (2)…

#### xcut-api-3 — Residency has ~10 public entry points; most are implementation steps with 0-1 external callers

`opennn/neural_network/neural_network.h:278-297` · low · API · lines +2 · effort S · risk medium · partial

Ways to move a network's parameters/states between host, device, FP32, BF16 and INT8 that are public on NeuralNetwork: compile(), compile(Device), copy_parameters_device(), copy_parameters_host(), copy_states_device(), copy_states_host(), link_states(Device), cast_parameters_to_bf16(), upload_parameters_bf16_inference(), upload_parameters_int8_inference(), release_bf16_fp32_parameter_master_for_inference(), load_parameters_bf16_inference_binary(); plus Layer::set_compute_device/set_compute_dtype per layer, plus the process-global Configuration::instance().set(Device, Type) snapshot that compile() captures and that only warns (cerr, once) if changed afterwards (cpp:616-627). They do not…

**Fix:** Move link_states(Device) and cast_parameters_to_bf16() to the private section now (0 and 1 external callers; optimizer.cpp:671 can call copy_parameters_device() or a narrowly named refresh_bf16_mirror() friend hook). Add a one-paragraph comment above compile()/copy_parameters_device()/copy_parameters_host() naming them as the residency API. Leave upload_parameters_bf16_inference/int8 public until the 23 test call sites are migrated to copy_parameters_device() under a BF16/INT8 Configuration…

*Verifier:* Caller counts verified: link_states(Device) 0 external, cast_parameters_to_bf16 1 (optimizer.cpp:671 restore_pre_warmup_state), upload_parameters_bf16_inference 23 (all tests), Layer::set_compute_device/set_compute_dtype only layer.h + multihead_attention_layer_test.cpp. Corrections: upload_parameters_int8_inference has a non-test caller at examples/gpt2/main.cpp:86 (auditor said 0 in library -…

#### xcut-api-6 — get_layer(Index) is unchecked while its sibling get_layer(string) throws 'Layer not found'

`opennn/neural_network/neural_network.h:152-152` · low · API · lines +2 · effort S · risk low · confirmed

get_layer(const Index) returns layers[layer_index] with no bounds check: a stale or off-by-one index (e.g. after a layer is removed or on an empty network) reads past the vector and dereferences garbage as a unique_ptr - UB with no diagnostic. The string overload (cpp:658-667) and get_layer_index (cpp:684) both throw a clear runtime_error. Tests and benchmarks call the Index form freely (get_layer(0), get_layer(Index(0)), dynamic_cast<Scaling&>(*network.get_layer(0)) in the higgs benchmark). Dataset::get_variable_type(const Index) (dataset.h:97) has the same unchecked variables[index].

**Fix:** Add throw_if(layer_index < 0 || layer_index >= ssize(layers), "NeuralNetwork::get_layer: index {} out of range for {} layers.", layer_index, layers.size()) before the return (and the same one-liner in Dataset::get_variable_type). Not a hot path: get_layer is called for setup and export, the forward loop iterates the vector directly.

*Verifier:* h:152 returns layers[layer_index] unchecked; string overload at cpp:658-667 and get_layer_index at cpp:670-684 throw. dataset.h:97 get_variable_type is likewise unchecked. The forward loop does iterate the vector directly so this is not a hot path. The fix adds two throw_if lines, so the LOC delta is +2, not -4.

#### operators-a-13 — Inline #ifdef islands with a dangling else break the file-wide static _gpu twin + OPENNN_CUDA_STUB pattern

`opennn/neural_network/operators/embedding_lookup_operator.cpp:341-355` · low · boilerplate · lines +2 · effort S · risk low · confirmed

Every operator in this scope declares static `*_gpu` twins at the top and defines them either against the kernels or via OPENNN_CUDA_STUB, keeping forward/back_propagate free of preprocessor branches. Two spots deviate: EmbeddingLookupOperator::forward_propagate wraps an `if (indices.is_cuda()) {...} else` in `#ifdef OPENNN_HAS_CUDA` so the `else` binds to a statement after `#endif` (the CUDA-less build compiles a bare call), and layer_normalization_add_forward (layer_normalization_operator.cpp:134-148) inlines its CUDA dispatch while its three siblings use the twin pattern. The dangling-else form is fragile under any edit of the CPU call.

**Fix:** Add `static void token_valid_lengths_gpu(const TensorView&, ForwardPropagation&, size_t, Index)` and `static void layer_normalization_add_forward_gpu(...)` beside the other twins (OPENNN_CUDA_STUB in the #else block), and make the two call sites plain `if (x.is_cuda()) { ..._gpu(...); return; }` like every other operator in the folder.

*Verifier:* embedding_lookup_operator.cpp:341-355: `#ifdef OPENNN_HAS_CUDA if (...) {...} else #endif compute_token_valid_lengths(...)` - the else binds across the #endif exactly as described. layer_normalization_operator.cpp:134-148 inlines its CUDA dispatch while the file declares four static *_gpu twins at 20-23 for its siblings. Both fixes fit the pattern documented as DONE in…

#### core-device-12 — set_conv_workspace_cap(int64_t mode) encodes auto/unlimited/bytes in one undocumented integer

`opennn/core/device_backend.h:140-143` · low · API · lines +3 · effort S · risk low · confirmed

The header declares set_conv_workspace_cap(int64_t mode) with no comment; the meaning (-1 = auto budget from set_conv_workspace_auto_limit_bytes, 0 = unlimited, >0 = explicit byte cap) is only discoverable from conv_workspace_limit_bytes() in the .cpp. Every caller passes magic literals: gpu_comparison_test.cpp:515 set_conv_workspace_cap(-1), opennn_resnet50_speed.cpp:81-87 mixes -1, 0 and stoll(arg)*1024*1024, and throw_frontend_unavailable tells users to 'cap the convolution workspace with device::set_conv_workspace_cap()' without saying what to pass. Passing a cap in MiB instead of bytes (a natural mistake) silently selects a 16-byte budget and bars every plan.

**Fix:** Document the three values on the declaration (three comment lines) and add two named constants, conv_workspace_auto = -1 and conv_workspace_unlimited = 0, used by the tests and benchmarks; keep the signature so Neural Designer callers compile unchanged.

*Verifier:* device_backend.h:140-142 declares set_conv_workspace_cap(int64_t mode) with no comment; semantics live only in conv_workspace_limit_bytes at device_backend.cpp:276-281 (0 = unlimited, >0 = bytes, <0 = auto). Callers pass literals: gpu_comparison_test.cpp:515 (-1) and :535 (16 MiB in bytes), opennn_resnet50_speed.cpp:81-87 (-1, 0, stoll*1024*1024), opennn_resnet50_maxbatch_trial.cpp:89-91 (0).…

#### nn-expression-17 — LeakyReLU slope hard-coded as 0.1 in four activation bodies while embedded uses LEAKY_RELU_SLOPE

`opennn/neural_network/model_expression.cpp:497-502` · low · duplication · lines +3 · effort S · risk low · confirmed

The per-neuron exporters embed the literal 0.1 in the C, JS, Python and PHP LeakyReLU bodies; the embedded exporter interpolates c_float_literal(LEAKY_RELU_SLOPE) (line 1492) and the library kernels use LEAKY_RELU_SLOPE (tensor_operations.h:75). Changing the constant would desynchronize four exported languages from the network silently; the activation test only checks that a function named LeakyReLU exists.

**Fix:** Since activation_table() is a function-local static built at first use, store the four LeakyReLU bodies as std::string built with format("... {} ...", c_float_literal(LEAKY_RELU_SLOPE)) (change ActivationBodies members to string, or keep const char* and add a static storage string).

*Verifier:* activation_table LeakyReLU bodies at 497-502 hard-code 0.1f/0.1 in C, JS, Python, PHP; embedded export interpolates c_float_literal(LEAKY_RELU_SLOPE) at 1492; LEAKY_RELU_SLOPE = 0.1f is defined in opennn/core/configuration.h:21 (tensor_operations.h:75 is a use site, not the definition). The table is a function-local static (471) so building the four strings with format() at first use is…

#### response-opt-20 — set(NeuralNetwork*) keeps compiled formula constraints whose column indices belong to the old network

`opennn/response_optimization/response_optimization.cpp:177-182` · low · API · lines +3 · effort S · risk medium · confirmed

CompiledFormula stores resolved input/output column indices (compile_formula binds names to columns at set_formula_constraint time, which requires the network). set() replaces the network and clears the variables cache and Jacobian, but keeps constraint_set.multivariate/disjunctive. Scenario: set_formula_constraint("x1 + x2 <= 5") on network A (x1=col 0, x2=col 1), then set(&B) where B's inputs are [x2, x3, x1]: the constraint now silently applies to columns 0 and 1 of B, i.e. x2 + x3. Univariate and cardinality constraints are stored by name and survive correctly, which makes the inconsistency easy to miss.

**Fix:** Either keep the source expression and recompile multivariate/disjunctive constraints against the new network inside set() (expression is already stored on MultivariateConstraint; callback constraints can stay), or clear them and document it. Verify against Neural Designer which order it uses.

*Verifier:* set(NeuralNetwork*) (177-182) replaces the network and clears variables_descriptives and network_jacobian but leaves constraint_set.multivariate/disjunctive intact; set_formula_constraint binds names to column indices at call time via build_input_columns/build_output_columns + compile_formula (252-283, 304-330), and CompiledFormula stores input_indices/affine_input_terms by column…

#### core-kernels-8 — W8A16 GEMV keeps acc[] in local memory (runtime-bounded loop) and loads x as four scalar BF16 loads

`opennn/core/cuda/kernel_quantization.cu:29-65` · low · overhead · lines +4 · effort S · risk low · confirmed

w8a16_linear_out_major_kernel declares float acc[W8A16_MAX_M] and indexes it in `for (int r = 0; r < m; ++r)` loops with a runtime m; a dynamically indexed array cannot live in registers, so every FMA does a local-memory read-modify-write (L1-cached but an extra load/store per FMA). In the char4 path the four activations are read as xr[0..3], four 2-byte loads, because the compiler cannot prove 8-byte alignment of `x + r * in_features + kk`. This is the decode-time weight-streaming kernel (m <= 16); the weight stream is 1 byte per FMA, so the per-FMA overhead is what separates it from bandwidth. The in-major kernel has the same acc[] pattern (lines 106-113).

**Fix:** Bound the accumulator loops at compile time: `#pragma unroll for (int r = 0; r < W8A16_MAX_M; ++r) if (r < m) ...` so acc[] stays in registers (or template the kernel on M in {1,2,4,8,16} and dispatch from the host like the decode attention ladder). Load the four activations with VecIO<T,4>::load_float when `in_features % 4 == 0` and the x base is 8-byte aligned (the arena guarantees 16), keeping the scalar path as the fallback. Measure with the Qwen3 decode benchmark.

*Verifier:* kernel_quantization.cu:29-65 read: `float acc[W8A16_MAX_M]` (W8A16_MAX_M=16, kernel_quantization.cuh:8) is indexed in `for (int r = 0; r < m; ++r)` loops with runtime m (30, 45, 59, 64, 76, 82, 85) and the char4 path reads xr[0..3] as four scalar loads (47-51); the in-major kernel repeats the pattern at 106-113. Host checks 1 <= m <= 16 (128-129), so a compile-time-bounded `#pragma unroll` loop…

#### xcut-api-16 — get_compute_stream() resolves to the active lane; capture/sync points assume lane 0 without checking

`opennn/core/device_backend.h:363-365` · low · design · lines +4 · effort S · risk low · confirmed

get_compute_stream() returns Backend::instance().stream(active_lane_index) (device_backend.cpp:1084-1087), a thread-local that the convolution wgrad fork sets to 1 (convolution_operator.cpp:667-723, restored through ScopeExit). I checked the fork region: nothing inside it calls get_compute_stream() expecting lane 0 (workspaces are per-lane, the join event is recorded on lane_stream(1) explicitly), so there is no bug today. The hazard is structural: 100+ call sites across 30 files call get_compute_stream() as 'the' stream, and the two places whose correctness depends on it being lane 0 - StreamCapture in calculate_outputs_resident (neural_network.cpp:2866) and the synchronize/capture pairs…

**Fix:** Make the lane dependence loud at the points that require lane 0: in StreamCapture's constructor and CudaGraphWorkspaceScope's constructor add throw_if(device::active_lane() != 0, "... must start on lane 0"). Optionally document on get_compute_stream() that forked regions must use lane_stream(n) explicitly. No rename, no hot-path change.

*Verifier:* device_backend.h:363-365 and device_backend.cpp:1084-1087 confirmed: get_compute_stream() returns stream(active_lane_index); convolution_operator.cpp:667-680 forks to lane 1 with ScopeExit restore; device_backend.cpp:372-373 documents the lane dependence; neural_network.cpp:2864 takes device::get_compute_stream() as the capture stream without asserting lane 0. 136 call lines across 37 files.…

#### dataset-a-23 — BinaryFile mode: every single set_sample_role call triggers a full streaming pass over the cache file

`opennn/dataset/tabular_dataset.cpp:219-225` · low · design · lines +4 · effort S · risk medium · partial

on_used_samples_changed is fired by Dataset::set_sample_role whenever a sample flips between None and used (dataset.cpp 248-254), and TabularDataset's override immediately recomputes cache_feature_descriptives with compute_descriptives_streaming, which reads every used row of the binary file. Callers that mark samples one at a time -- unuse_samples_with_missing_targets (2419-2422), any UI loop over set_sample_role -- therefore pay O(used_rows) file I/O per sample, O(N^2) overall. The statistics are only needed at fill/descriptive time.

**Fix:** Replace the eager refresh with a `mutable bool cache_statistics_dirty` set in on_used_samples_changed and consumed at the top of fill_from_binary_cache / calculate_feature_descriptives / prepare_training_scaling (refresh once, under a small mutex because the ConcurrentBinaryFillsReadPreparedStatistics test fills from two threads). Keep the eager refresh in read_csv/from_JSON/set_binary_cache_path.

*Verifier:* Mechanism is real: Dataset::set_sample_role (dataset.cpp:248-254) fires on_used_samples_changed on each None flip and TabularDataset's override (219-225) calls refresh_cache_statistics -> compute_descriptives_streaming (311-370), a full pass over used rows. But the in-repo callers cited do not hit it in binary mode: scrub_missing_values returns early for BinaryFile (2557-2566) so…

#### r2-arena-planner-and-propagation-structs-8 — set() discards every owned buffer even when the new layout fits the existing ones

`opennn/neural_network/forward_propagation.cpp:163-176` · low · overhead · lines +4 · effort S · risk low · unverified

Answering the rebuild question directly: set() unconditionally clears staged_input_storage (the H2D staging buffers), layer_state_storage (cuDNN RNN state), device_valid_length_storage and the session state, then `arena.resize_bytes(total_bytes)` (line 737), which frees and re-mallocs whenever the byte count differs in either direction (tensor_types.h:422-447 only early-outs on an exact size match). Only the external_storage branch at 727 reuses memory that is already big enough. Inside the library nothing re-sets a live ForwardPropagation (training_context.cpp and optimizer.cpp:845 share via external_storage instead), so this costs nothing today; it matters for any consumer that keeps one…

**Fix:** In the owned branch, mirror the external one: `if (arena.owns_memory() && arena.get_device() == device && arena.byte_size() >= total_bytes) { /* keep */ } else arena.resize_bytes(total_bytes, device);` (setZero then zeroes only total_bytes via a view or keeps zeroing the whole buffer), and replace the `.clear()` of staged_input_storage / device_valid_length_storage with `.resize(n)` so their grow_to paths keep existing device memory. Verify against Neural Designer's usage of…

#### training-loss-19 — MSE/NSE/WSE/CE skip the shape and empty checks that MAE performs

`opennn/training_strategy/error_functions.cpp:129-147` · low · design · lines +4 · effort S · risk low · confirmed

mean_absolute_error validates `input.get_shape() != target.get_shape()` and returns 0 on empty input (154-161); mean_squared_error, normalized_squared_error, weighted_squared_error and the cross-entropy pair do neither. With an output count different from the target count (network declared with N outputs on a dataset with M targets) `input.as_vector() - target.as_vector()` is an Eigen size-mismatch: an assertion in debug, an out-of-bounds read in release. With batch_size 0 (calculate_error is public and called directly by tests, QN/LM and Neural Designer) MSE divides by zero. No shape check was found upstream in optimizer.cpp, batch.cpp or training_context.cpp.

**Fix:** Do the check once in Loss::calculate_error / calculate_output_deltas (`throw_if(input.get_shape() != target.get_shape(), ...)` after the two views are taken, skipping Yolo/CrossEntropy3d whose targets are shaped differently) and remove the per-function copy from MAE, so every error function gets the same guard from one place.

*Verifier:* error_functions.cpp:129-147 MSE/MSE-gradient, 201-217 NSE, and the CE pair have no shape or empty guard; MAE has both (154-161, 177-182). Loss::back_propagate guards `batch.is_empty()` (loss.cpp:1357) but calculate_error (1471) does not, and QuasiNewtonMethod::calculate_directional_point (quasi_newton_method.cpp:303) and tests call calculate_error directly. No test depends on the MAE-specific…

#### dataset-b-10 — ImageDataset::enable_device_residency materialises the whole dataset twice on the host

`opennn/dataset/image_dataset.cpp:85-111` · low · overhead · lines +5 · effort M · risk medium · partial

Inputs are filled into `inputs` (N x pixels), targets into `targets`, then both are copied into a third `staged` matrix just to get one contiguous [inputs|targets] block for upload_device_matrix. For an image dataset the inputs dominate, so peak host memory is ~2x the dataset size (a 50k x 224x224x3 set is 7.5 GB as float; this path needs ~15 GB transient) before the single device upload. TabularDataset::enable_device_residency (tabular_dataset.cpp:1230-1241) makes one copy.

**Fix:** Fill `staged` directly: fill_targets into a small (N x T) matrix, fill_inputs in row chunks straight into staged via `fill_inputs(chunk_indices, ..., &staged(row0, 0))` is not possible with the row stride, so instead give Dataset::upload_device_matrix an overload taking (inputs, targets) that issues two cudaMemcpy2DAsync copies with destination pitch (P+T)*4; drop `staged`. Alternatively upload `inputs` and `targets` as two device regions if the gather path can take a split layout.

*Verifier:* image_dataset.cpp:84-111 confirmed: inputs (N x P) and targets (N x T) are filled, then copied into a third staged (N x (P+T)) before upload_device_matrix(staged); transient host peak is about 2x the input matrix. TabularDataset::enable_device_residency (1230-1241) makes one extra copy of data. The proposed fix is self-contradicting as written (it first proposes chunked fill_inputs into staged,…

#### core-kernels-6 — Dropout forward initialises one Philox state per element and discards 3 of every 4 random numbers

`opennn/core/cuda/kernel_activation.cu:202-224` · low · overhead · lines +6 · effort S · risk low · confirmed

dropout_forward_kernel runs one thread per element, calls curand_init (Philox4x32-10: key/counter setup) and curand_uniform, which generates a 4-wide Philox block and returns one lane; the other three are thrown away. The kernel also reads/writes one 2- or 4-byte element per thread. For a transformer training step with dropout after every attention and MLP block this is the most ALU-heavy elementwise kernel in the family (10 Philox rounds per element) for 3 bytes of traffic. Processing 4 elements per thread with curand_uniform4 cuts the RNG work 4x and allows VecIO<T,4> loads/stores and a 4-byte mask store; the mask layout (one byte per element) is unchanged, so the backward is untouched.

**Fix:** One thread per 4 elements: curand_init(seed, idx, 0, &state); float4 u = curand_uniform4(&state); apply to elements 4*idx..4*idx+3 (scalar tail for n % 4) and store the four mask bytes as one uint32 when aligned. Launch through launch_vec_on<4> with are_aligned<16>(output) / are_aligned<4>(mask). Keep the seed/subsequence semantics so results stay reproducible for a given seed.

*Verifier:* kernel_activation.cu:202-224 read: one thread per element, curand_init(seed, idx, 0) + one curand_uniform per element (Philox generates 4 lanes, 3 discarded), scalar 2/4-byte load/store, launched via launch_elementwise (223). The 4-per-thread curand_uniform4 rewrite with VecIO<T,4>/launch_vec_on<4> (kernel_common.cuh:132-198) is consistent with the file's conventions. One correction of wording:…

#### core-types-12 — throw_if drops the source location whenever the message has format arguments

`opennn/core/opennn_types.h:179-192` · low · design · lines +6 · effort S · risk low · confirmed

Only the plain string_view overload of throw_formatted takes a defaulted source_location; the variadic format overload cannot (a defaulted parameter cannot follow a pack), so every throw_if with placeholders - the majority of the messages in this scope, e.g. 'Shape: rank {} exceeds MaxRank={}' - is thrown without '[at file:line]', while 'Shape::back() on empty' gets one. Diagnostics are therefore inconsistent in exactly the cases that carry the most information, and a maintainer who adds a placeholder to a message silently loses the location.

**Fix:** Use the standard wrapper: `template<typename... Args> struct FormatWithLocation { format_string<Args...> fmt; source_location loc; template<typename S> consteval FormatWithLocation(const S& s, source_location l = source_location::current()) : fmt(s), loc(l) {} };` and declare `template<typename... Args> [[noreturn]] void throw_formatted(FormatWithLocation<type_identity_t<Args>...> message, Args&&... args)`, which then also covers the zero-argument case so the string_view overload can go. The…

*Verifier:* opennn_types.h:180-191: the string_view overload takes a defaulted source_location and appends '[at file:line]'; the variadic format_string overload cannot and does not, so every throw_if with placeholders (e.g. tensor_types.h:194 'Shape: rank {} exceeds MaxRank={}') loses the location. The FormatWithLocation/type_identity_t wrapper is the standard remedy and subsumes the zero-arg overload. One…

#### layers-a-11 — Concatenation CPU loops re-resolve views and shapes per pixel and use two different channel sources

`opennn/neural_network/layers/concatenation_layer.cpp:52-66` · low · overhead · lines +6 · effort S · risk low · partial

Inside the omp pixel loop the forward calls inputs[i].get_shape()[3] and inputs[i].as<float>() per pixel per input, and the backward calls get_input_delta(back_propagation, layer, i) (two vector indexings into BackPropagation) plus in_delta.empty() per pixel per input (lines 103-119). Each memcpy moves only a channel slice (tens to a few hundred bytes), so the per-iteration bookkeeping is a visible fraction of the loop. Separately, the forward derives the slice width from the actual input shape while the backward uses the configured input_channels[i]; if they ever differ (Concatenation::set is reachable with a vector that does not match the sources) the forward silently writes a differently…

**Fix:** Before the parallel loop build a small vector of (const float* base, Index channels, Index offset) per input (forward) and (float* base or nullptr, Index channels, Index offset) per input delta (backward), throw_if the channel sum differs from total_channels, then loop over that array. Use input_channels[i] in both directions so forward and backward agree by construction.

*Verifier:* The consistency defect is real: the CPU forward derives each slice width from inputs[i].get_shape()[3] (concatenation_layer.cpp:60-61) while both the CUDA paths and the CPU backward use the configured input_channels[i] (41-45, 97-101, 111-114), and nothing checks that the sum of input_channels equals total_channels (only the count is checked at line 29-30). The overhead claim is not…

#### r2-set-vs-compile-device-ordering-6 — Recurrent plans cuDNN-only and fused-step-only slots on both devices

`opennn/neural_network/layers/recurrent_layer.cpp:728-765` · low · overhead · lines +6 · effort S · risk low · unverified

Recurrent::get_forward_specs always plans CudnnInputSequence (B x T x F) and CudnnOutputSequence (B x T x H) as Pooled slots and the four step scratches (B x F, 3 x B x H) as Transient; get_backward_specs always plans StepInputScratch, StepPrevH, DeltaScratch, NextCarry, StepInDelta and CudnnInputDeltaScratch (B x T x F). On CPU the operator touches none of the cuDNN slots or the step scratches (apply/apply_delta use the strided maps directly); on CUDA with a cuDNN-eligible layer (FP32, Tanh/ReLU) the step scratches are unused, and on the fused-kernel path (BF16 or Sigmoid/Identity) the cuDNN sequence buffers are unused. For a forecasting network this is 2 x B x T x F + B x T x H extra…

**Fix:** Gate the specs on `get_compute_device() == Device::CUDA` (cuDNN sequence slots and CudnnInputDeltaScratch) and on the fused-kernel eligibility (step scratches) with `Shape{}` otherwise, exactly as C2PSA does; the operator already null-checks nothing on these paths, so keep the same two-way split used by the dispatch (cudnn_rnn_eligible_).

#### core-kernels-7 — SwiGLU forward/backward are scalar per element (2-byte BF16 accesses across 3-5 streams)

`opennn/core/cuda/kernel_activation.cu:47-80` · low · overhead · lines +8 · effort S · risk low · confirmed

swiglu_forward_kernel reads gate and up and writes out one element per thread; swiglu_backward_kernel reads dout, gate, up and writes dgate and dup likewise. For BF16 LLM training (rows x intermediate, e.g. 11008 wide) these are pure bandwidth kernels running with 2-byte transactions per thread, the same pattern the activation kernels already vectorise for FP32. VecIO<T, vec16<T>> with launch_vec_on (as in cast_kernel) turns them into 16-byte transactions for both dtypes with a scalar tail, no behaviour change.

**Fix:** Rewrite both kernels as `<T, VEC>` templates over VecIO<T, VEC>::load_float/store_float (VEC = vec16<T>), launched with launch_vec_on<vec16<T>>(stream, n, are_aligned<16>(gate, up, out[, dout, dgate, dup]), ...). Can share the dispatcher introduced for core-kernels-3.

*Verifier:* kernel_activation.cu:47-80 read: swiglu_forward_kernel / swiglu_backward_kernel are one-element-per-thread scalar kernels launched through launch_elementwise (73, 79), while the same file already vectorises FP32 activations with float4 (99-118) and kernel_cast.cu:17-38 shows the VecIO + launch_vec_on pattern. Rewrite as <T,VEC> with are_aligned<16>(...) on all operands (dgate/dup may be null;…

#### dataset-a-22 — Unparseable numeric/binary tokens become NaN silently and are not counted as missing

`opennn/dataset/tabular_dataset.cpp:1489-1496` · low · design · lines +8 · effort S · risk low · confirmed

Column types are decided from at most 100 sampled rows (infer_column_types). During the full parse, parse_numeric_token and parse_binary_token map any token that fails parse_real to QUIET_NAN via parse_float_or_nan, while count_missing only counts tokens equal to the missing label or empty. Scenario: a numeric column whose sampled rows are clean but row 5,000 contains "n/a" (label is "NA") or "1.234,5": the value is NaN in `data`, missing_values_number and variables_missing_values_number report 0, the JSON says the file has no missing values, yet has_nan() is true and scrub_missing_values later imputes it without any diagnostic. The only hint a user gets is a mismatch they cannot see.

**Fix:** Have parse_row return the number of parse failures (or let parse_numeric_token/parse_binary_token take a counter), add them to thread_missing / thread_variables_missing, and print one warning per variable with the first offending token (the infer_column_types warning at 2688-2694 already has the wording).

*Verifier:* parse_float_or_nan (1480-1484) returns QUIET_NAN on parse failure; parse_numeric_token (1491-1498) and parse_binary_token (1528-1540) fall through to it; count_missing (2048-2080) only counts is_missing_token (empty or label). Types are inferred from at most 100 strided rows (2606-2624). So unparseable tokens outside the sample become silent NaNs not reflected in missing_values_number. Fix sound.

#### operators-b-9 — tokenize_into rescans the whole remaining text for every special token at every segment boundary

`opennn/neural_network/operators/tokenizer_operator.cpp:981-1005` · low · overhead · lines +8 · effort S · risk low · confirmed

Each loop iteration runs text.find(special, position) for all special_strings; a special that does not occur scans to the end of the text every time. Cost is O(#occurrences x #specials x |text|). For a Qwen3 chat transcript (~20 specials, <|im_start|>/<|im_end|> every few hundred bytes) a 100 KB prompt performs on the order of 10^8 byte comparisons before any BPE work. The results are monotone in `position`, so each special's next occurrence can be cached and only re-searched once position passes it.

**Fix:** Keep a vector<size_t> next_hit parallel to special_strings, initialised with text.find(special, 0); each iteration picks the minimum, and after advancing `position` only re-searches the specials whose cached hit < position (npos entries are never searched again). Same output, one pass per special over the text.

*Verifier:* tokenize_into (tokenizer_operator.cpp:981-1005): per loop iteration `text.find(special, position)` for every special in special_strings; a special absent from the text scans to the end on every segment. Cost O(occurrences x specials x |text|) as claimed. Cached next-hit vector (re-search only entries whose cached hit < position, never re-search npos) is output-identical. LOC +8 fine; severity low.

#### response-opt-18 — calculate_random_inputs is 209 lines: three sampling lambdas plus a discrete-repair tail

`opennn/response_optimization/response_optimization.cpp:858-1068` · low · design · lines +8 · effort S · risk low · confirmed

The function defines sample_scalar (876-894), sample_allowed_set (896-922) and sample_categorical (924-972) as closures over six captured locals, walks the variables (974-1012), and then (1014-1067) builds the lattice, resolves cardinality columns, derives free_lattice and discrete_is_coupled, and dispatches to one of two repair routines. The second half is independent of the first except for `random_inputs` and `discrete_explore`, and the lambdas could be file-static functions taking (random_inputs, rows) — which would also let the sampling be unit-tested without a network.

**Fix:** Extract `void repair_discrete_inputs(MatrixR&, const Domain&, const vector<Variable>&, const Lattice&, float discrete_explore) const` for lines 1014-1065, and promote the three lambdas to static free functions next to round_discrete_inputs. Keep behaviour identical; this is a pure relocation.

*Verifier:* calculate_random_inputs spans 858-1068 (211 lines): sample_scalar 876-894, sample_allowed_set 896-922, sample_categorical 924-972 capturing random_inputs, input_domain, original_domain, effective_evaluations, discrete_explore and sampling_memory; variable walk 974-1012; the discrete tail 1014-1067 (fixed_mask, lattice, cardinality columns, free_lattice, discrete_is_coupled, two repair dispatches)…

#### training-optimizers-8 — Optimizer::train() silently returns a NaN-filled result without loss/network/dataset, while QN/LM null-deref in the same state

`opennn/training_strategy/optimizer.cpp:771-777` · low · API · lines +8 · effort S · risk medium · partial

Base train() returns a TrainingResult with maximum_epochs+1 NaN entries and no stopping condition when loss, network or dataset is missing; the caller gets NaN from get_training_error() with no diagnostic. QuasiNewtonMethod::train (quasi_newton_method.cpp:181) and LevenbergMarquardtAlgorithm::train (levenberg_marquardt_algorithm.cpp:258) dereference loss->get_neural_network() unconditionally and crash. TrainingStrategy::train throws. Three sibling entry points, three behaviours. Scenario: `AdaptiveMomentEstimation adam; adam.train()` -> NaN result; `QuasiNewtonMethod qn; qn.train()` -> segfault.

**Fix:** Add a protected `NeuralNetwork& require_training_setup() const` that throw_ifs on !loss, !loss->get_neural_network(), !loss->get_dataset() (same messages as get_maximum_batch_size already uses) and call it at the top of all three train() bodies; replace the silent return. Verify against Neural Designer that nothing relies on the silent empty result.

*Verifier:* optimizer.cpp:773-777 silently returns `TrainingResult results(maximum_epochs + 1)` (NaN-filled per training_result.cpp:21-22). quasi_newton_method.cpp:181-182 and levenberg_marquardt_algorithm.cpp:258-259 dereference loss->get_neural_network() unguarded; training_strategy.cpp:111-120 throws. All true. LOC is off: a helper of ~8 lines plus three call sites minus the two removed lines is about +8,…

#### core-device-10 — execute_graph passes the shared_ptr-keyed VariantPack: the frontend rebuilds a uid map on every execute

`opennn/core/cuda/cudnn_frontend_utilities.h:82-104` · low · overhead · lines +10 · effort S · risk low · partial

VariantPack is unordered_map<shared_ptr<Tensor_attributes>, void*>. The frontend's matching Graph::execute overload (graph_interface.h:1365-1373) is a convenience wrapper that allocates a fresh unordered_map<int64_t, void*> and re-inserts every entry before delegating to the uid-keyed execute. That is one heap-allocated map build per conv/BN/attention graph execution on every uncaptured path: inference without a CUDA graph, evaluation, warmup and the non-graph training configurations. Tensor uids are fixed once validate() ran, so the translation is constant per GraphSlot.

**Fix:** Give GraphSlot a uid-keyed unordered_map<int64_t, void*> built once after finalize (from the VariantPack's tensors), have run_slot/execute_graph refresh only the pointer values each call and pass that map to execute. Same pattern for the attention callers that call execute_graph directly.

*Verifier:* Verified in build-resnet-capacity/_deps/cudnn_frontend-src/include/cudnn_frontend/graph_interface.h:1363-1372: the shared_ptr-keyed execute builds a fresh `unordered_map<int64_t, void*> uid_map` and delegates to the uid-keyed overload (1374-1380). execute_graph (cudnn_frontend_utilities.h:82-104) always passes the shared_ptr-keyed VariantPack (194), and callers run_slot:753,…

#### core-kernels-13 — Generic grouped_attention_kernel re-reads the query row from global memory for every key and keeps acc[256] in local memory

`opennn/core/cuda/kernel_attention.cu:450-491` · low · overhead · lines +12 · effort M · risk low · confirmed

The non-decode fallback (reached when the cuDNN SDPA and cuBLAS GEMM paths both decline, grouped_query_attention_layer.cpp:617-634) gives each thread one (batch, query, head) and loops over all keys. Inside the key loop it reloads q_vec[d] from global/L1 for every key (head_dim loads per key, so key_seq x head_dim redundant loads per thread), and the output accumulator `float acc[256]` indexed by a runtime head_dim lives in local memory (1 KB per thread, a load+store per FMA). The decode kernel directly below already shows the fix: a compile-time HEAD_DIM ladder (64/128/256) with q and acc as register fragments and warp-level dot products. As a fallback this is not the steady-state path,…

**Fix:** Template the kernel on HEAD_DIM (64/128/256, reusing the OPENNN_DECODE_ATTENTION_CASE-style ladder and load_head_fragment) so q and acc are register arrays, and assign one warp per (query, head) with lane-strided fragments plus warp_reduce_sum for the dot product, as grouped_attention_decode_kernel does. Alternatively make grouped_attention_gemm_gpu the unconditional fallback and delete this kernel (-40 lines) if the benchmark shows no case where it is needed.

*Verifier:* kernel_attention.cu:450-491 read: one thread per (b, i, hq), `float acc[256]` runtime-indexed by head_dim (470-471, 483, 488) and q_vec[d] re-read from global inside the key loop (477). Launch at 843-849 (block 128, grid ceil(total/128)) with a head_dim<=256 host check. Reached from grouped_query_attention_layer.cpp:617-634 when SDPA (also skipped whenever kv_length_device is set) and the GEMM…

#### core-kernels-9 — bias_grad_sum_cuda is the only reduction in the scope that uses float atomics (nondeterministic bias gradient)

`opennn/core/cuda/kernel_tensor.cu:41-70` · low · design · lines +12 · effort M · risk medium · partial

bias_grad_sum_kernel splits the batch into up to 256 row chunks (grid.y) and atomicAdds each chunk's partial into bias_grad, so the summation order, and therefore the FP32 result, varies run to run. Every other reduction in these files (linear_backward_single_output, norm_backward_warp, the BN reduce/finalize pairs) deliberately writes per-block partials and sums them in a fixed order - the single-output comment says 'so the result does not depend on how the rows were scheduled'. The atomics also force every caller to zero the output first (tensor_operations.cpp:2003-2005 and 2052-2054 call set_zero_async immediately before). Bit-exact parity tests on dense layers with bias can flake on…

**Fix:** Write chunk partials to ensure_workspace<float>(GraphWorkspaceKind::GradientPartials, n_chunks * features) and sum them in a tiny finalize kernel in chunk order (the existing single_output_gradient_finalize_kernel already does exactly this for weight partials; generalise it to `sum_partials_kernel(blocks, features, partials, out)`), with an `accumulate` flag for the recurrent caller that adds across time steps. Then drop the two set_zero_async calls in tensor_operations.cpp.

*Verifier:* bias_grad_sum_kernel (kernel_tensor.cu:42-53) does atomicAdd per (chunk, feature) with grid.y = n_chunks (55-69), and both callers zero the output first (tensor_operations.cpp:2003-2008 skinny path, 2052-2057 staged fallback), so the nondeterminism and the set_zero coupling are real. Two corrections: (1) it is NOT the only float-atomic reduction in scope - norm_weight_gradient_coalesced_kernel…

#### core-device-6 — 14 runtime knobs in this scope, 4 documented nowhere, 3 parsed by hand beside env_int_or

`opennn/core/device_backend.cpp:504-509` · low · design · lines +14 · effort S · risk low · partial

This scope reads OPENNN_DEVICE_CACHE, OPENNN_DEVICE_CACHE_MB, OPENNN_CUDNN_PLAN_CACHE, OPENNN_CUDNN_PLAN_CACHE_DIR, OPENNN_LANES, OPENNN_THREADS, OPENNN_OMP_DYNAMIC, OPENNN_GRAPH_NODES, OPENNN_GRAPH_TIMING, OPENNN_LT_AUTOTUNE_CANDIDATES, OPENNN_AUTOTUNE_CANDIDATES(+_FORWARD/_WGRAD/_DGRAD), OPENNN_CUDNN_HEURISTICS, OPENNN_CONV_ENGINE_NOTES, OPENNN_SDPA_AUTOTUNE and OPENNN_CUTLASS_NARROW_K. All are alive (each has a reader), but the first four appear in no README or docs file at all, and the rest are mentioned only inside individual benchmark plan notes (resnet50-large-batch-plan.md, transformer-training-batch-sweep.md). Three sites parse with raw getenv + atoi/strtoll (read_cap_bytes…

**Fix:** Replace the three hand parses with env_int_or (read_cap_bytes becomes a one-liner) and add one table to README.md (or docs/runtime_knobs.md) listing every OPENNN_* runtime variable with default, effect and the reading file; the grep in this audit is the starting list. Do not remove any knob: each has a measured rationale in its comment.

*Verifier:* Hand parses confirmed: read_cap_bytes 504-509 (strtoll), Backend ctor 1121-1122 (atoi), set_threads_number 1233-1234 (atoi); env_int_or exists at string_utilities.h:206. Doc gap confirmed: grep over docs/, README.md, AGENTS.md finds no mention of OPENNN_DEVICE_CACHE/_MB or OPENNN_CUDNN_PLAN_CACHE/_DIR; OPENNN_LANES/OPENNN_THREADS appear only in resnet50-large-batch-plan.md and…

#### dataset-b-11 — Image cache build decodes every image serially while every other image loop is OpenMP

`opennn/dataset/image_dataset.cpp:524-537` · low · overhead · lines +15 · effort M · risk low · confirmed

write_image_cache (and the Matrix branch at 443-450) call load_image one file at a time on one thread: full JPEG/PNG decode + resize + uint8 quantisation per image, with the file write being the only serial dependency. Building the cache for a 100k-JPEG dataset at ~5-10 ms per decode is 8-15 minutes single-threaded versus ~1-2 minutes on 8 cores; it happens on every first run and after any source change (signature includes the newest write time).

**Fix:** Process images in chunks of omp_get_max_threads()*4: decode/quantise the chunk with `#pragma omp parallel for` into a vector of per-slot uint8 buffers (collect the first exception as the batch fills already do), then write the chunk sequentially in order. Note resize_image is itself `omp parallel for`; pass a flag or keep the outer parallelism only (nested OMP is off by default so it is harmless).

*Verifier:* write_image_cache (image_dataset.cpp:527-540) and the Matrix branch (441-450) call load_image serially per image; only writer.write is order-dependent. resize_image (image_processing.cpp:662) is itself `#pragma omp parallel for` over pixels, so nested parallelism would be inert by default as the finding notes. Chunked decode into per-slot buffers followed by in-order writes, with the…

#### xcut-build-tests-26 — No .clang-tidy/.clang-format/.editorconfig; a first clang-tidy pass on 5 core files yields real narrowing and rounding hits plus known-noise categories to disable

`opennn/core/statistics.cpp:452-454` · low · build/test · lines +25 · effort S · risk low · partial

The repo has no static-analysis or formatting configuration. clang-tidy 22 (checks: bugprone-*, performance-*, clang-analyzer-core*/cplusplus*, readability-redundant-*, misc-unused-*) on tensor_types.cpp, memory_pool.cpp, statistics.cpp, string_utilities.cpp, json.cpp reported: bugprone-narrowing-conversions x16 (statistics.cpp:240, 254, 438-440, 596-597; tensor_types.cpp:124-125; string_utilities.cpp:389 - Index->float/double and size_t->Index without casts), bugprone-incorrect-roundings at statistics.cpp:454 (`Index(float(bins_number)/2.0f + 0.5f)` hand rounding; use `(bins_number + 1) / 2` integer math), readability-redundant-casting at opennn_types.h:276 (`float(-1e9f)`), and noise that…

**Fix:** Commit a minimal `.clang-tidy` (Checks: the set above minus performance-enum-size, bugprone-reserved-identifier, bugprone-easily-swappable-parameters; HeaderFilterRegex: 'opennn/.*'; no WarningsAsErrors) and a `.editorconfig` (utf-8, lf, 4-space indent, trim trailing whitespace, final newline) so every contributor and agent runs the same checks; fix the statistics.cpp:452-454 rounding with integer arithmetic and cast the 16 narrowing sites explicitly.

*Verifier:* No .clang-tidy/.clang-format/.editorconfig exists (ls -a). statistics.cpp:452-454 is as quoted and `(bins_number + 1) / 2` is exactly equivalent for positive bins_number; opennn_types.h:276 `float(-1e9f)` confirmed. I could not reproduce the clang-tidy run or the 16 narrowing sites, so those counts are unverified; the hand rounding is a style issue, not a bug (bins_number is a small positive…

#### r2-duplicated-kernels-across-folders-7 — grouped_attention_softmax_kernel reads each score row three times and computes every exp twice

`opennn/core/cuda/kernel_attention.cu:509-519` · low · overhead · lines +30 · effort M · risk low · partial

The GQA materialized-path softmax walks the row once for the max, once for the sum (computing __expf), and once more to write probabilities (recomputing the same __expf), so each fp32 score is fetched from global memory three times and exponentiated twice. masked_softmax_rows_kernel in the same file (lines 114-152) already solves this for the unfused attention path by holding the row in a MAX_ELEMS register array selected by a small template ladder (launch_masked_softmax_rows, lines 171-186): one read, one exp. key_seq on this path is bounded by the scratch budget at grouped_query_attention_layer.cpp:362-363 and the decode ladder, so the same ladder up to 2048 applies; rows above it can…

**Fix:** Template the kernel on MAX_ELEMS like masked_softmax_rows_kernel: load s_row[lane + e*32] (or -INFINITY past `valid`) into `float values[MAX_ELEMS]` once, reduce max, exponentiate in place while summing, write once; reuse the ceil_div(key_seq, 32) ladder from launch_masked_softmax_rows (factor it into a small `dispatch_row_elems` helper shared by both launchers) and keep today's body as the >2048 fallback. Measure on the GQA prefill benchmark before and after; expected gain is bandwidth-bound…

*Verifier:* Kernel body at kernel_attention.cu:509-519 confirmed: three strided passes over s_row and __expf evaluated twice per element, versus masked_softmax_rows_kernel (:114-152) which keeps values[MAX_ELEMS] in registers with the 4/8/16/32/64 ladder at :171-186. Correction on the bound: the 256 MiB budget at grouped_query_attention_layer.cpp:362-363 bounds chunk, not key_seq (per_batch_bytes scales with…

#### operators-b-7 — BytePairTokenizer::bpe is O(n^2) per piece and Qwen3's pre-tokenizer makes any non-space run (CJK, URLs, base64) a single piece

`opennn/neural_network/operators/tokenizer_operator.cpp:820-852` · low · overhead · lines +30 · effort M · risk low · partial

bpe() rescans every adjacent pair per merge (building and hashing a pair_key string each time) and erases from the middle of a vector<string>, so a piece of n codepoints costs up to n merges x n pair lookups = O(n^2) string hashes. Pieces are normally short, but is_letter() (line 455-460) treats every non-ASCII, non-whitespace codepoint as a letter and both pre_tokenize variants extend a letter run greedily (lines 910, 1123), so a CJK paragraph without spaces, a long URL or a base64 blob becomes one piece. Concrete: a 5,000-codepoint Chinese paragraph -> ~12.5M pair_key builds + StringMap lookups, i.e. seconds per paragraph, and the BPE cache (4096 entries, never evicted) does not help…

**Fix:** Replace the scan with the standard linked-list + min-heap merge (each symbol keeps prev/next indices; push (rank, position) for every adjacent pair once; pop the lowest rank, skip stale entries, merge, push the two new neighbour pairs): O(n log n) per piece, same output. Keep merge_ranks as the rank source. Add a test that tokenizes a long space-free string and compares with the current implementation's output.

*Verifier:* bpe() at tokenizer_operator.cpp:820-851 is as quoted: full rescan building pair_key per adjacent pair each merge plus vector::erase — O(n^2) hashes per piece. is_letter (455-460) treats every non-ASCII non-whitespace codepoint except x/÷ as a letter, and both pre_tokenize loops (910, 1123) extend letter runs greedily, so a space-free CJK/Thai paragraph is one piece; the 4096-entry cache (646,…

#### r2-arena-planner-and-propagation-structs-9 — find_memory_pool_overlay has no direct test; the only pin is an end-to-end overfit that the code comment says strategy can break

`opennn/core/memory_pool.cpp:135-165` · low · build/test · lines +35 · effort M · risk low · unverified

tests/core/memory_pool_test.cpp covers plan_memory_pool (4 tests, including BothStrategiesRespectRecordedLifetimes) but find_memory_pool_overlay is exercised only indirectly through two conv-only layouts (TrainingRecomputeScratchUsesFutureActivations, RecomputeOverlayUsesLifetimesAcrossLayerTypes) that assert which activation is overlaid, not that the overlay is safe. Meanwhile forward_propagation.cpp:497-506 documents that forcing Compact made YoloOverfit.CSPGradientFlowsAndLossDecreases stop learning. By construction the planner honours every recorded lifetime under both strategies, so a packing choice can only change results if some lifetime handed to it is understated (or the overfit…

**Fix:** Add a MemoryPoolTest that generates random lifetime sets (seeded), plans with both strategies, and for every (first, second) pair with a random size asserts that a returned overlay does not intersect any entry live at either step and fits under peak_bytes; plus a ForwardPropagationMemoryTest that runs the YOLO/CSP layout under a forced Compact plan (expose the strategy through a test hook or environment flag) and checks slot/overlay disjointness against the recorded lifetimes. If the…

#### xcut-build-tests-25 — Library .cpp files compile only through the forced-include PCH (<ranges> etc.); clang-tidy/IDEs without it fail

`opennn/core/string_utilities.cpp:115-115` · low · build/test · lines +40 · effort M · risk low · confirmed

Running clang-tidy with the project's own Windows clang compile database minus the `-include cmake_pch.hxx` flags, string_utilities.cpp fails to parse: `error: use of undeclared identifier 'views'` at line 115 (`text | views::split(separator)`), because neither the .cpp nor string_utilities.h includes `<ranges>`; only opennn/pch.h:19 does. A grep finds 37 library .cpp files that use `views::`/`ranges::` and do not include `<ranges>` themselves. check_headers.sh verifies headers in isolation but nothing verifies that the .cpp files are self-sufficient, so any tooling that does not replay the PCH (clang-tidy, clangd without the compile DB, a consumer compiling the sources,…

**Fix:** Add one CI configure with `-DCMAKE_DISABLE_PRECOMPILE_HEADERS=ON` (compile-only, opennn target) and fix the include-what-you-use fallout by adding `<ranges>` (and whatever else surfaces) to the .cpp files that need it. Keep the PCH for speed; the check is what prevents regressions.

*Verifier:* string_utilities.cpp includes only its header, <cctype>, <utility> (lines 9-12); the header includes no <ranges>; opennn_types.h and tensor_types.h include no <ranges>; line 115 uses `text | views::split(separator)`; opennn/pch.h:19 is the only <ranges>. 37 library .cpp files use views::/ranges:: without including <ranges> (counted). The MSVC SKIP_PRECOMPILE_HEADERS files…

#### xcut-build-tests-27 — Nine CUDA test files set Device::CUDA with no runtime device guard, unlike the four that GTEST_SKIP

`tests/neural_network/qwen3_network_test.cpp:372-380` · low · build/test · lines +40 · effort S · risk low · partial

backward_full_write_test, cutlass_narrow_gemm_test, device_backend_test and mean_squared_error_test guard GPU tests with `if (!device::has_cuda_device()) GTEST_SKIP()`. Nine other files only use the compile-time `#ifdef OPENNN_HAS_CUDA` and then call `Configuration::instance().set(Device::CUDA, ...)`: qwen3_network_test (6 sites), adaptive_moment_estimation_test (5), int8_inference_test (3), grouped_attention_test (2), neural_network_test (2), activations_test, batch_test, multihead_attention_layer_test, memory_audit_test. A CUDA-built opennn_tests.exe on a host without a GPU (a CUDA CI runner image, a laptop with the driver uninstalled, a container) fails those tests with 'Configuration:…

**Fix:** Add `#define OPENNN_SKIP_WITHOUT_CUDA() if (!device::has_cuda_device()) GTEST_SKIP() << "No CUDA device."` to tests/numerical_derivatives.h (already includes the library) or a tiny tests/cuda_guard.h, and use it as the first line of the ~23 GPU tests in those nine files; replace the four existing ad-hoc spellings with it.

*Verifier:* Guard mechanism confirmed (configuration.cpp:66-68 throws). qwen3_network_test.cpp:372-375 `Configuration::instance().set(Device::CUDA, Type::BF16)` under #ifdef only, no has_cuda_device/GTEST_SKIP in the file; same for adaptive_moment_estimation_test (9 sites), int8_inference_test (6), grouped_attention_test, neural_network_test, memory_audit. Corrections: (a) the biggest unguarded file is…

---

## Status — 2026-08-24

Everything below was checked against the tree at the time of writing rather than
against the notes, because the notes had drifted: several findings recorded as
open had been fixed weeks earlier, and one recorded as understood was still
live. The commits since `509857699` are the record of what changed; this section
is the record of what is *true now*.

### The fifteen high-severity bugs are closed

Fourteen were already fixed, each verified by reading the code rather than the
commit log. Most carry a comment at the site describing the original defect,
which is what made them quick to confirm:

| finding | how it reads now |
| --- | --- |
| `CudaBlockCache::give` throwing from a destructor | `deallocate` is `noexcept`; the comment names `~Buffer`/`~PinnedBuffer` as the callers |
| `set_threads_number` destroying a cached `ThreadPool` | `contraction_device()` rebuilds the handle per call; the comment records the use-after-free |
| Apple `from_chars` recursing forever | the integral branch calls `std::from_chars`; the comment marks `std::` as load-bearing |
| quoted-field tokenizer eating `,` and `;` | proper `in_quote` state machine keyed on the actual separator |
| `BinaryFile` analysis indexing an empty matrix | `require_in_memory_data(...)` guards the analysis entry points |
| float-only layers accepting BF16 | `Concatenation::on_compute_dtype_changed` refuses anything but FP32 |
| LSTM on CUDA with no FP32 guard | `cudnn_rnn.cpp` selects the descriptor from `config.data_type` and rejects the rest |
| `load_darknet_backbone_v11` targeting dead labels | targets `c8_*`, the labels the builder emits |
| `set_parameters` overflowing the compact bf16 mirror | `throw_if(fp32_master_released())` on all three entry points |
| Logarithm scaler exporting broken Python/JS | handled as the one non-affine method, via `log_pre`/`exp_post` |
| CPU valid-length record frozen after the first pass | re-inherited every pass; the comment records the stale-mask bug |
| `run_graph_epoch` null pipeline slot | the warm-up keeps a callable so it walks the branches the epoch will |
| Minkowski divided by `batch_size` | divided by the sample count, like its four siblings |
| NSE dropping its batch scaling | `result.error *= get_batch_scale(batch)`, as WSE does |

The fifteenth was still open and is fixed now.

**Dropout under CUDA graphs redrew one fixed mask.** The seed was chosen on the
host and passed as a kernel launch argument. A captured graph records the
arguments its kernels were launched with, so once a training step was captured
every replay reused the mask captured with it — one dropout pattern for the rest
of the run. Nothing fails when that happens: shapes, scaling and loss all stay
plausible, and the only symptom is that the regularisation quietly stops
varying, which is why it survived. The seed now lives in device memory with a
one-thread kernel advancing it before each draw, so the advance is inside
whatever capture is running.

`DropoutDeviceTest.GraphReplayDrawsANewMask` reproduces it: it reported the
replayed mask identical to the captured one before the change.
`ConsecutiveCallsDrawDifferentMasks` sits next to it so the first cannot pass
for the wrong reason.

A sample of the medium bugs was checked the same way — batch-norm's running
variance, the attention CPU padding inference, the GPU sampler's logit cap — and
all were fixed. The medium and low tiers were not verified exhaustively.

### Raised during this pass, not in the original audit

**Twenty gradient checks were running on networks of all zeros.** `compile()`
zeroes the parameters and only the `StandardNetworks` builders randomise them
afterwards, so a network assembled by hand from `add_layer()` reached its
gradient check with every weight still zero. With zero weights the delta
reaching every layer but the last is zero, so most of the gradient is
identically zero *on both sides of the comparison*: `BackPropagateConvolutional`
had 1 live component out of 432, `BackPropagateMultiheadAttention` 80 out of
25,920. A deliberate 1000x error on the attention gradient changed nothing any
of them measured.

`calculate_gradient()` now refuses an all-zero network so this cannot come back
quietly, and the twenty fixtures randomise after `compile()`.

The one test that then failed was not a library bug. Two of the three
convolution configurations use ReLU, and a central difference steps across its
corner: the error falls in proportion to `h` (7.8e-3 at 1e-3, 1.9e-3 at 1e-4)
where roundoff would grow and a smooth truncation error would fall as `h²`, and
the Identity configuration agrees to 2e-7. The bound was the problem — absolute,
across configurations whose gradients differ by 300x — and is now relative to
the largest component with the old value as its floor.

**The C2PSA suite could not fail.** Its fixture never randomised either, and
`CpuAndGpuForwardOutputsMatch` forward-propagated one network on both devices
though a network is compiled for one — so which device it measured depended on
what the previous test left in the global configuration. It builds two networks
now, and `CpuAndGpuGradientsMatch` compares them component by component relative
to each component, which reports 1.9e-2 against a real 2x error where it allows
5e-3.

### Deliberately not done

- **`xcut-build-tests-7` (remove 143 `Configuration::instance().set` calls from
  tests as redundant).** They are load-bearing, and the C2PSA bug above is the
  proof: a test that does not set its own device inherits whatever the previous
  one left behind. This finding should be treated as refuted.
- **`core-utils-8` (delete the free `tokenize`).** Neural Designer uses it. Same
  for several others that look dead from inside this repo — grep that tree
  first, as the audit header already says.
- **`training-optimizers-12` (move optimizer defaults to member initialisers).**
  The two class-owned cases named are already fixed. What is left are
  assignments to *inherited* members, which a derived class cannot express as a
  default member initialiser, so the proposed fix does not apply.
- **`neural_network.cpp` trivial-member inlining.** Its one-line definitions sit
  inside the non-CUDA branch of an `#ifdef` where the CUDA build has its own;
  hoisting them into the class defines them twice.

### Still open

Blocked on hardware: the self-hosted CUDA runner job in CI is written but inert
until `HAS_CUDA_RUNNER` is set, so nothing runs the GPU suite except this
machine.

The remaining duplication and boilerplate findings are individually small. The
larger ones named in the original list — the per-test image helpers, the numeric
Hessian stub, the rope backward, the GQA pipeline, the expression-emission
loops — were checked and are already done.
