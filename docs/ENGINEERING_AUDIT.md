# OpenNN engineering audit — simplification opportunities

Date: 2026-08-06. Scope: `opennn/` (219 files, ~54.5k effective lines),
`tests/`, `docs/benchmarks/`. Method: five parallel audit passes
(duplication, dead code, structure, consistency, tests/benchmarks), every
finding verified with repo-wide reference checks including tests and
benchmarks. Goal: leave everything as simple as possible, incrementally —
no grand redesign.

## What is already healthy (verified, not assumed)

- Error handling is uniform: `throw_if` dominates (582 calls, ~78%); the
  plain `throw` sites are legitimately unconditional (CUDA-absent stubs,
  switch defaults). CUDA error checking is one macro family.
- No commented-out code blocks, no `#if 0`, no TODO/FIXME debt markers, no
  dead enum values (all 88 enum classes checked).
- Layer JSON serialization is centralized in `Layer::to_JSON/from_JSON`;
  optimizers share `write_common_json`; datasets delegate `fill_*`.
- `cudnn_frontend` containment is clean (1 header, 4 consumers).
- `opennn/CMakeLists.txt` GLOBs sources: file splits cost zero build edits.

## Level 0 — correctness first (not simplification, but blocks everything)

The "pre-existing" test failures are root-caused. Four are ~9 lines total;
one is a real product bug; one needs a 10-minute triage.

| Failure | Cause | Fix | Effort |
| --- | --- | --- | --- |
| C2PSA.GpuGradientMatchesNumerical | `tests/numerical_derivatives.cpp:61,92,145` builds GPU batches but never calls `upload_to_device_batch_async` — both gradient passes read uninitialized device targets | lift the 4-line upload guard from `gpu_comparison_test.cpp:52-56` into the 3 sites; then delete the duplicate `compute_gradient` helper | 4 lines |
| Qwen3/Int8 `DirectLogicalBf16Weights*` (3 tests) | `write_logical_bf16_parameters` calls `filesystem::remove` on a file whose `ifstream` is still open — Windows sharing violation | `input.close()` before remove + `error_code` overload (pattern used 20+ times elsewhere) | 4 lines, 2 files |
| NormalizedSquaredErrorTest.BackPropagate | network shape drawn from the unseeded global RNG — order-dependent by construction | `set_seed` in the test; better: seed in `CpuConfigurationListener::OnTestStart` (`tests/test.cpp:19-24`) to de-flake the whole suite | 1 line |
| Normalization3dTest.FusedResidualAddAliasesSafeBranchDelta | **real product bug**: `back_propagation.cpp:269-270` aliases `backward_slots[i][2] = backward_slots[i][1]`; a consumer assigns instead of accumulating, clobbering the other's delta | fix accumulate-vs-assign around `back_propagation.cpp:262-348`; do NOT widen the tolerance (already 2x its peers) | real work |
| YoloLoss.V8DFLGradientMatchesNumerical | ambiguous: DFL backward bug, or finite differences across a TAL argmax assignment boundary (genuinely discontinuous) | triage: freeze the TAL assignment and re-run; if it passes, replace the numerical check with a fixed-assignment analytical check | 10-min triage |

A durably green suite removes the standing "compare against the known
failure list" overhead from every future verification cycle.

**Level 0 outcome (2026-08-06): both suites fully green** (CUDA 831 passed /
0 failed, CPU 773 / 0). Two corrections to the table above, established
empirically:

- Normalization3dTest was NOT a product bug. The fused-aliased gradient is
  bit-identical to an unfused reference (explicit Addition + plain LN), and
  the finite-difference estimate converges monotonically to the analytical
  value as h shrinks — the failure was FD truncation error in an
  ill-conditioned 3-feature LN. The test now compares against the exact
  unfused reference instead of finite differences.
- YoloLoss.V8DFLGradientMatchesNumerical WAS a real product bug, but not the
  suspected TAL discontinuity: `yolo_v8_gradient_kernel_tal` scaled the DFL
  cross-entropy gradient with `lam.giou` while the forward weighs that term
  with `lam.dfl` (a constant 5.0/1.5 mix mismatch, measured as a stable
  ~1.8x analytical/numerical ratio; the FD estimate was flat across h,
  proving the analytical side wrong). Fixed by scaling each term with its
  own lambda; the shared host kernel also serves the GPU path.

## Level 1 — deletions and trivial cleanups (revised after the Neural Designer cross-check)

**2026-08-06 cross-check against the Neural Designer product tree**
(`neuraldesigner/{neuralengine,neuraleditor,neurallabeler,neuralviewer,tests,tools}`,
excluding its vendored `opennn/` copy): the product uses almost every
symbol that is orphaned inside this repo. The real API surface of this
library is **opennn callers ∪ Neural Designer callers** — any future
deletion campaign must grep the ND tree first.

Confirmed deletable (dead in opennn AND unused by Neural Designer, ~55 lines):

| Item | Lines | Status |
| --- | --- | --- |
| `TestingAnalysis::get_batch_size` (the setter turned out to be USED by `tests/bert_dataset_test.cpp:185` — audit false positive, restored) | ~2 | DONE (getter only) |
| Never-called accessors with no ND callers: `YoloDataset::is_v8_mode`, `Layer::get_weights_dtype`, `ForwardPropagation::get_active_sequence_length`, `ForwardPropagation::get_cuda_graph`, `Dataset::get_separator`, `Dataset::get_codification`, `Dataset::set_variables`, `ModelSelection::get_inputs_selection` (ND comments confirm it already adapted to its removal) | ~8 | DONE |
| Dead env toggles: `OPENNN_CUDA_DEBUG_SYNC` (~11), `OPENNN_CUDA_SYNC_EACH_BATCH` (~7), `OPENNN_CONV_LEGACY` (~2) | ~20 | DONE |
| `OPENNN_BF16_HOST_INPUT_CAST` + its fp32-staging fallback | ~30 | postponed (MEDIUM — verify the fallback is truly unneeded first) |

Level 1 executed 2026-08-06 (suites stayed fully green). Also applied:
`get_regularization_method` returns by value (dangling hazard removed;
ND-compatible), `pooling_scratch_` converted to an immortal function-local
(static destruction ran after the CUDA context died), 34 shadowed-local
renames in `attention_operator.cpp` (C4458 source), 28 trailing
`Configuration` restores + the empty `DISABLED_YoloV8LossParity` stub and an
empty `TearDown` removed from tests. The C4061 switch warnings were left
as-is: MSVC emits them even with a `default:`, so the fix is a warning-flag
decision, not code.

ALIVE in Neural Designer — moved to the do-not-delete list (originally
flagged dead inside opennn): the entire Tukey/box-plot chain, the
lift-chart chain, `filter_data`, `get_advised_point`, the
descriptives-by-class cluster, the classification-errors chain and
`calculate_binary_classification_rates`+struct, both Spearman wrappers
(and transitively `correlation_spearman`/`logistic_correlation_spearman`),
Histogram `calculate_minimal_centers`/`calculate_maximal_centers`,
`steal_from`, `calculate_missing_values_statistics`,
`set_binary_cache_path`/`cache_path_override`,
`set_cache_directory`/`cache_directory`, `invalidate_trainable_layer_cache`,
`sample_role_to_string`, `get_data_path`, `get_header_line`,
`get_has_sample_ids`, `get_sample_ids`, `has_categorical_variables`,
`get_missing_values_method`, `get_missing_values_label`,
`get_images_directory`, `get_labels_directory`, `get_inputs_selection_name`,
Adam `get_learning_rate`, SGD `get_initial_learning_rate`, and the
`type` alias (`opennn_types.h:149` — ND writes `type(0)`).

Two follow-ups instead of deletions:

- `loss.h:89 get_regularization_method` is used by ND, so fix the hazard
  instead of deleting: return `string` by value, not `const string&`.
- Consider a comment in the affected headers noting Neural Designer as an
  external consumer, so the next audit does not re-flag these.

Do NOT delete (already known): `unuse_least_correlated_variables` /
`unuse_collinear_variables` (ND tasks), `OPENNN_DRELU_FUSION`,
`OPENNN_GRAPH_TIMING`, `OPENNN_MEMORY_DEBUG`, `OPENNN_THREADS`,
`OPENNN_PROFILE`.

Tests: delete 62 redundant `Configuration::instance().set()` calls (the
`CpuConfigurationListener` already resets before every test) and the empty
`DISABLED_YoloV8LossParity` stub (13 lines). Warnings: 4 switch sites
(prefer an explicit throwing `default:`), 5 shadowed locals in
`attention_operator.cpp` (rename to `q_len`/`src_len`), and move
`kernel_layers.cu:551 pooling_scratch_` — a namespace-scope global owning
device memory, destroyed after the CUDA context — into a `thread_local`
accessor like `device_backend.cpp:787`.

## Level 2 — targeted deduplication (~1,100 lines in the library, plus tests/benchmarks)

Most fixes promote helpers that already exist:

| Duplication | Lines | Fix (existing precedent) |
| --- | --- | --- |
| `link_parameters`/`link_gradients` mirrors, 23 functions / 10 operators | ~120 | promote `link_views` from `long_short_term_memory_operator.cpp:25` to `operator.h` |
| Hand-written "requires CUDA" stubs, 25+ bodies / 12 files | ~90 | apply `OPENNN_CUDA_STUB*` from `error_functions.h:13`; move macro next to `CHECK_CUDA` |
| SDPA graph construction written 3x | ~100 | route through `cudnn_frontend_utilities.h` `new_graph`/`finalize` (keep the HeurMode difference as a parameter); `convolution_operator.cpp` is the model |
| cuDNN RNN fwd/bwd drivers, Recurrent ≡ LSTM | ~110 | add `cudnn_rnn_forward_/backward_` to `CudnnRnnState` with a `has_cell_state` flag |
| Weight-init triples (random/glorot/pytorch), 16 functions / 7 operators | ~80 | `init_weight_bias(weights, bias, scheme, fan_in, fan_out)` next to `glorot_limit` |
| CUDA launch scaffolding (`rows/cols` + dispatch), ~22 sites | ~120 | `dispatch_rows_cols` helper + `TensorView::rows()/cols()` |
| YOLO loss preamble x8 + v8 GPU host round-trip x2 | ~120 | `YoloLossContext` struct + one host-staging driver |
| LayerNorm/RMS row loops + rope fwd/bwd sign twin | ~110 | `for_each_row` driver; verify no OMP/vectorization regression |
| `scale_gpu`/`unscale_gpu` twins | ~23 | `inverse` flag, mirroring the CPU side |
| Layer ctor + `set_input_shape` boilerplate (22+19 sites) | ~150 | opportunistic only — 3-4 lines per site |

Level 2 status (2026-08-06, suites stayed fully green):

- DONE `link_views` promoted to `operator.h` (bool-returning); 23
  link_parameters/link_gradients bodies across 10 operators converted.
- DONE `OPENNN_CUDA_STUB*` macros moved to `opennn_types.h` next to
  CHECK_CUDA; 23 hand-written stub bodies collapsed across 9 files
  (messages harmonized to "requires CUDA support.").
- DONE shared `finalize_attention` + `seq_len_scalar` in
  `cudnn_frontend_utilities.h`; both attention operators' three graph
  builders converted (local build_sdpa_graph_common/finalize copies and the
  GQA inline third copy deleted; unused handle parameters dropped).
- DONE `scale_gpu`/`unscale_gpu` merged with an `inverse` flag, mirroring
  the CPU side; X-macro list updated.
- DONE #4 shared cuDNN RNN drivers: `CudnnRnnState::cudnn_rnn_forward_` /
  `cudnn_rnn_backward_` with a `has_cell_state` flag and a reconfigure
  callback for the persistent-algorithm retry; both operators' duplicated
  cudnnRNNForward/BackwardData/BackwardWeights sequences deleted.
- Weight-init triples: downgraded to opportunistic — the variation axes
  (orthogonal recurrent init, tied_transposed guards, conv kernel fans,
  bias policies) make a single helper a forced abstraction.
- Launch-scaffolding, LN/rope row-loop and YOLO-context items: deferred
  until they can be verified with benchmarks, not only suites (hot paths).

Tests (~350 lines): shared `tests/llm_test_helpers.{h,cpp}` for the 9
helpers duplicated verbatim between `qwen3_network_test.cpp` and
`int8_inference_test.cpp`; shared vision fixtures (`write_bmp_24`,
`TempDir`, `write_label`) for the 4-5 YOLO test files. GLOB picks new
files up automatically. (Still pending.)

Benchmarks (~350 of 923 duplicated Python lines): `docs/benchmarks/tools/benchlib/`
with `gpu.py` (`nvidia_used_mib`, `PeakMonitor`, `cooldown`, `measure_idle`,
idle-delta trial wrapper) and `provenance.py` (`git_commit`, `sha256`,
`file_info`, `framework_versions`). The monitors define published numbers —
port one runner at a time and diff a JSON artifact before/after.

## Level 3 — structural splits (zero header/CMake churn first)

1. `loss.cpp` (2,014): lines 30-1294 are a self-contained
   `#ifndef OPENNN_NO_VISION` YOLO block — move verbatim to
   `loss_yolo.cpp`. The `Loss` class proper starts at line 1296.
2. `neural_network.cpp` (2,293): partitions cleanly into core (graph +
   forward), `neural_network_io.cpp` (~700 lines of JSON/binary I/O,
   including the 233-line bf16 loader), and `neural_network_device.cpp`
   (~500 lines of device/precision residency). No header edits.
3. `tensor_operations.cpp` (2,655): the 828-line `#ifdef OPENNN_HAS_CUDA`
   block (1696-2552) becomes `tensor_operations_gpu.cpp` + stubs file; the
   `OPENNN_GPU_OPS` X-macro is already the declared boundary.
4. `ForwardPropagation::set` (565 lines, 65% of its file): phase 2
   (offset planning, lines 257-533) is already member-independent —
   extract as a file-local `plan_activation_offsets(...) -> ActivationPlan`.
5. CUDA-graph inference state out of `ForwardPropagation` (7 members, 6
   methods; all external mutation is inside `neural_network.cpp:2144-2223`)
   into an `InferenceGraphState` member struct. Move verbatim.
6. `Optimizer` extractions, one commit each: batch staging (~190 lines →
   `batch.cpp`, where `BatchPools`/`BatchPrefetchSession` already live);
   dataset scaling (`set_scaling`/`set_unscaling`, 174 lines with
   `dynamic_cast<TabularDataset*>` → free functions, removes 3 heavy
   includes); the CUDA-graph epoch runner (`run_graph_epoch`, 350 lines +
   8 members → `TrainingGraphRunner`; most timing-sensitive code in the
   library — relocate verbatim, restructure later).

Sizeable but explicitly NOT recommended now: unifying `Index` vs `size_t`
at the Layer/Operator seam (~670 casts — mechanical but a huge diff),
`Device::PinnedHost` in `Buffer` (6 hand-rolled pinned pairs — audit ~30
`device_type == CPU` comparisons first), converting the remaining
`SDPACache`/GQA raw allocations and the 25 `mutable Buffer` caches (9
classes) to pool specs (cuDNN RNN reserve space needs persistent-lifetime
treatment), `get_` naming unification, file renames (6 class/file
mismatches), and the Eigen Tensor include split (compile-time only; 18
files). Each is worth doing only with a concrete trigger.

## Benchmarks hygiene (from the tests/benchmarks pass)

- Capacity READMEs never mention the idle-delta protocol the runners now
  implement — one paragraph each in the three capacity READMEs.
- Dead link: `energy/transformer-energy/...md:99` points at a nonexistent
  `rosenbrock-max-batch/run_energy.py`.
- `--result-json` semantics differ across the three capacity runners
  (opt-out / opt-in / always) — align to higgs's opt-out.
- 5 folders absent from `benchmark_manifest.json`; `throughput/string-processing`
  has a CMake target but no README/manifest entry. Extend
  `tools/validate_benchmarks.py` with an unlisted-folder check (~10 lines)
  so it cannot recur. `capacity/data-capacity` ships two overlapping
  PowerShell sweeps — keep one.

## Suggested execution order

1. Level 0 (suite to green; the aliased-slot product bug gets its own
   verified fix).
2. Level 1 deletions (one commit for dead code, one for tests, one for
   warnings/toggles).
3. Level 2 promotions that reuse existing helpers (link_views, CUDA stubs,
   SDPA graph utils, RNN driver) — the rest opportunistically.
4. Level 3 splits, one file per commit, suites green between each.

Estimated net effect of levels 0-3: roughly 2,000-2,500 lines removed or
deduplicated, ~2,500 lines relocated out of oversized files, a green test
suite, and no behavior changes outside the one real bug fix.
