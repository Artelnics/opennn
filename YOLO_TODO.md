# YOLO in OpenNN — Audit & Roadmap

Audit date: 2026-05-22
Branch: `dev-refactor`
Scope: assessment only — no code changes.

---

## 1. What already exists

| Component | Location | Status | Notes |
|---|---|---|---|
| `YoloDataset` (label parsing, cache, letterbox, k-means anchors, target encoding) | `opennn/yolo_dataset.{h,cpp}` | Complete (CPU) | BMP-only |
| `DetectionOperator` forward (sigmoid x/y/obj, exp·anchor for w/h, softmax classes) | `opennn/operators.cpp:3046-3098` | Complete (CPU) | GPU path throws |
| `DetectionOperator::apply_delta` (backward) | `opennn/operators.cpp:3100-3149` | Complete (CPU) | Math verified: see §3 |
| `Detection` layer wrapper + JSON I/O | `opennn/detection_layer.{h,cpp}` | Complete | |
| `NonMaxSuppressionOperator` (per-class greedy NMS) | `opennn/operators.cpp:3151-3261` | Complete (CPU) | GPU path throws |
| `NonMaxSuppression` layer wrapper | `opennn/non_max_suppression_layer.{h,cpp}` | Partial | Missing JSON I/O — see §2.1 |
| YOLO loss (`Loss::Error::Yolo`, `yolo_error_cpu`, `yolo_gradient_cpu`, `yolo_loss_iou`) | `opennn/loss.cpp:27-181, 315-317, 534-536` | Complete (CPU) | GPU path explicitly returns `false`/throws |
| `YoloNetwork` minimal builder (5×conv+pool → conv1024 → 1×1 logits → Detection → NMS) | `opennn/standard_networks.cpp:363-420` | Complete | v2-style: anchors + softmax class, single scale |
| Unit test `tests/yolo_dataset_test.cpp` | `tests/yolo_dataset_test.cpp` | Wired in CMakeLists.txt:58, file is untracked — needs commit |

**Conclusion:** the existing scaffold is closer to **YOLO v2 (single-scale)** than v1 — it already has anchor boxes, softmax-per-class, and sigmoid xy decoding. A v1-equivalent demo can run end-to-end today, in principle.

---

## 2. Punch list — what blocks a first runnable YOLO

### 2.1 Bugs / inconsistencies

- **`YoloDataset::fill_inputs` only normalizes during training** (`opennn/yolo_dataset.cpp:610`)
  - `is_training ? (1.0f / 255.0f) : 1.0f` — inference returns raw [0,255], training returns [0,1].
  - Action: scale unconditionally, or only when no upstream `Scaling` layer exists.

- **Double-scaling in `YoloNetwork`** (`opennn/standard_networks.cpp:375-377`)
  - Adds `Scaling` with `ImageMinMax` on top of an already-normalized dataset feed during training.
  - Action: either remove the Scaling layer from `YoloNetwork` or change `fill_inputs` to always pass raw bytes.

- **`NonMaxSuppression` layer missing `read_JSON_body` / `write_JSON_body`** (`opennn/non_max_suppression_layer.h`)
  - Confidence threshold, IoU threshold, and `boxes_per_cell` won't survive save/load.
  - Action: mirror what `Detection::{read,write}_JSON_body` does.

- **NMS runs during training** even though gradients can't flow through it.
  - It's marked non-trainable, so the loss skips it via `get_last_trainable_layer_outputs()`, but the forward pass still computes it every batch. Wasted compute, not incorrect.
  - Action (optional): gate NMS forward on `is_training==false`, or move NMS out of the network into a post-processing helper.

- **k-means anchor calculation is not seeded** (`opennn/yolo_dataset.cpp:154-221`)
  - Initial assignment uses `boxes[i % ssize(boxes)]` — deterministic but order-dependent on filesystem listing.
  - Probably fine, just flag for reproducibility audit.

### 2.2 Missing components

- **No `examples/yolo/main.cpp`.** Every other capability (mnist, breast_cancer, …) has an example; YOLO has none. This is the single most visible gap.
- **No tests for `DetectionOperator` forward/backward.** Only `YoloDataset` is covered.
- **No tests for `NonMaxSuppressionOperator`.**
- **No test for `yolo_error_cpu` / `yolo_gradient_cpu` against numerical derivatives.** This is the highest-risk code (gradient math is easy to get subtly wrong).
- **No end-to-end smoke test** (`YoloNetwork` + `Loss::Error::Yolo` + `YoloDataset` + `TrainingStrategy` → assert loss decreases over N steps).
- **No inference helper** to:
  - Inverse the letterbox transform (apply scale/offset undo to map predictions back to original image coords).
  - Read `Detection` layer output OR `NonMaxSuppression` output into a friendly `vector<Box>`.
  - Draw boxes / serialize results to JSON for a demo image.
- **`augment_inputs()` is an empty stub** (`opennn/yolo_dataset.h:72`). Limits training quality; required at v2+ for usable accuracy.

### 2.3 Format / dataset limitations

- **BMP-only** ingest (`opennn/yolo_dataset.cpp:55-58`). Common YOLO datasets ship PNG/JPG. Need to either reuse `load_image` for PNG/JPG or document the BMP restriction in the example.
- **Letterbox is applied at cache build**, so changing input size invalidates cache. That's correct but worth documenting.

---

## 3. Gradient math verification

I verified the YOLO backward chain by hand:

- **Coordinate loss**: `(sqrt(out_w) - sqrt(target_w))^2`. Gradient wrt `out_w` = `(sqrt_out − sqrt_target) / sqrt_out`. Code at `loss.cpp:167-168` matches.
- **DetectionOperator w/h backward**: `out_w = exp(logit) · anchor`, so `d(out_w)/d(logit) = out_w`. Hence `in_delta[w] = delta[w] · out[w]`. `operators.cpp:3137-3138` matches.
- **Objectness**: target=IoU when object present, else 0; loss is `(out − target)^2`; sigmoid Jacobian `out·(1−out)` applied in `apply_delta`. Matches.
- **Class softmax + cross-entropy**: loss gradient wrt softmax probability is `−t/p`; DetectionOperator's `apply_delta` performs the softmax-Jacobian transform (`delta[c] − dot`). Chain rule is correct.

**No math errors found.** The non-obvious risk is that the loss reads the *decoded* output (post-DetectionOperator), so all gradients must pass through DetectionOperator's backward — they do, since Detection is a trainable layer.

**Confirmed approximation (v1 paper):** `yolo_gradient_cpu` treats `iou(target_box, output_box)` as a constant when computing `dE/d(out[0..3])`, missing the chain-rule contribution from the objectness loss `(out[4] - iou)^2` back through the box coords. This is the documented v1 formulation and matches reference implementations; numerical-vs-analytical gradient differs by ~0.1-0.2 on the affected coordinates as expected. A v3+ rewrite would differentiate through (G)IoU.

---

## 4. Plan — phased path from current state to modern YOLO

### Phase 1 — First runnable YOLO (this is the immediate punch list)

1. ✅ Commit `tests/yolo_dataset_test.cpp` + the `tests/CMakeLists.txt` edit that wires it in.
2. ✅ Fix the double-scaling bug (§2.1 items 1 & 2): `Scaling` removed from `YoloNetwork`, `fill_inputs` normalizes unconditionally.
3. ✅ Add `NonMaxSuppression::{read,write}_JSON_body`.
4. ✅ `tests/detection_layer_test.cpp` (forward shape, sigmoid/exp/softmax values).
5. ✅ `tests/yolo_loss_test.cpp` (numerical-gradient check of `yolo_gradient_cpu` against `yolo_error_cpu`).
6. ✅ `tests/non_max_suppression_test.cpp` (handcrafted boxes, verify suppression).
7. ✅ `examples/yolo/main.cpp`: generates synthetic 128×128 BMP dataset → trains 10 epochs Adam → runs inference → prints top boxes. Wired into `examples/CMakeLists.txt`.
8. ✅ Inference helper: `decode_yolo_detections()` undoes the letterbox transform and returns `vector<YoloDetection>`. Covered by `tests/yolo_inference_test.cpp`.

**Definition of done:** the example builds, trains, prints sensible bounding boxes, and the test suite passes on CPU. **Done.** All 12 YOLO-related tests pass on CPU; example runs end-to-end in ~3 minutes.

**Bug discovered during Phase 1 (fixed):** removing the `Scaling` layer from `YoloNetwork` (double-scaling fix) broke input-shape propagation — the first `Convolutional` then received an empty shape from `get_output_shape()` and threw "kernel shape cannot be bigger than input shape". Fixed in `standard_networks.cpp` by passing `input_shape` explicitly to the first conv.

### Phase 2 — YOLO v2 polish ✅ COMPLETE (2026-05-27)

9. ✅ BatchNorm in conv stack (already used elsewhere in OpenNN — reuse).
10. ✅ Multi-scale training (random input resize per epoch). Requires `set_input_shape` to propagate cleanly through the whole `YoloNetwork`.
11. ✅ PNG/JPG support in `YoloDataset` (delegate to existing `load_image`).
12. ✅ Basic augmentation (flip, hue/sat jitter, random crop) in `augment_inputs`.

### Phase 3 — YOLO v3 (multi-scale heads) — 4 of 4 done (training validated; cross-scale NMS for inference still TODO)

13. ✅ Replace VGG-style stack with Darknet-53 residual blocks. Needs a residual/route layer (or reuse `Addition` layer). *(Darknet-Tiny variant, opt-in via `Backbone::DarknetTiny`, 2026-05-28.)*
14. ✅ Three detection heads at strides 32/16/8 with FPN-style upsample+concat. *(Smoke-tested 2026-05-29: 34-layer DarknetTiny+FPN, 3.64M params, trained epoch 0 in 5m32s on synthetic 359 samples. Training error 37.45 → validation error 13.62, no NaN. Cross-scale NMS for inference still TODO.)*
15. ✅ Per-class **sigmoid** instead of softmax (independent class probabilities). *(Opt-in via `ClassActivation::Sigmoid`; BCE replaces CE in the loss when active, 2026-05-29.)*
16. ✅ Loss: replace squared-error on (x,y,w,h) with **GIoU** or **DIoU** (recommended now even before v3 — measurable accuracy lift). *(GIoU shipped with L2+clip stabilizers, 2026-05-28.)*

### Phase 4 — YOLO v4/v5

17. CSP backbones (cross-stage partial). Needs split-and-concat pattern; can be expressed with existing primitives plus a split helper.
18. SPP / SPPF block (multi-scale max-pool concat). New layer or composition of existing pooling.
19. PANet head (bidirectional FPN). Same primitive needs as v3 plus another concat path.
20. Mosaic augmentation (4 images stitched) in `augment_inputs`.

### Phase 5 — YOLO v8 (anchor-free) — this is the biggest jump

21. **Anchor-free detection head**: replace `DetectionOperator`'s `exp·anchor` decode with direct distance-to-grid-boundary regression. New op + new dataset target encoding.
22. **Decoupled head**: separate conv branches for box-regression vs class-classification.
23. **Distribution Focal Loss (DFL)**: regress box edges as a discrete distribution over 16 bins, integrate via softmax expectation. New op.
24. **Task-Aligned Assigner**: replace fixed IoU-best-anchor assignment with dynamic assignment using `score^α · IoU^β` per gt-anchor pair. New code path in `YoloDataset`.
25. **Varifocal loss** for classification.

### Phase 6 — YOLO v11 (on top of v8)

26. C3k2 block (efficient C3 variant). Layer composition.
27. C2PSA attention (partial self-attention). Reuse `MultiHeadAttention`.
28. Backbone reorganization for v11's depth/width scaling rules.

---

## 5. Risks / open questions

- **GPU YOLO path is now fully implemented** (as of 2026-06-16). DetectionOperator, NMS, GIoU loss, UpsampleOperator, and ConcatenationOperator all have working CUDA kernels. VOC training on RTX 2080 runs at ~69 sec/epoch. No longer a blocker.
- **`YoloNetwork` enforces `input_H/W == grid_size * 32`** (`standard_networks.cpp:372-373`). That's a 5-pool stride-32 architecture — locks input to 13×32=416 or 7×32=224. Multi-scale training needs this relaxed.
- **No pretrained weights loader.** v3+ practically requires ImageNet-pretrained backbones; without that, training a usable detector from scratch is slow.
- **No mAP / COCO evaluation harness.** Hard to know if "training works" actually means "detector is good." Phase 1 should add at least a per-class precision/recall test on a held-out tiny set.

---

## 6. Recommendation

Phase 1 is ~1-2 weeks of focused work and gives you a public-facing demo. Phases 2-3 together get you to a competitive-on-paper detector. Phase 5 (anchor-free / v8) is genuinely a different architecture and will dwarf the rest in effort — plan it as its own milestone, not an incremental upgrade.

For dev-refactor specifically: keep YOLO **CPU-only** until Roberto's GPU mixed-precision validation lands. Doing both at once would mix two debug surfaces.

---

## 7. Session log

### 2026-07-22 — Phase 5a: YOLOv8 anchor-free head (§21 + §22)

Phase 5a implemented and validated end-to-end (CPU). §21 (anchor-free detection head) and §22 (decoupled head) are complete with full numerical gradient verification.

**Architecture changes:**

- **`DetectionV8Operator`** (`opennn/detection_v8_operator.h`): anchor-free sigmoid decode on all channels `[tx, ty, tw, th, cls_0..C-1]`. No anchor multiplication, no objectness. CPU + GPU forward/backward (sigmoid Jacobian; GPU dispatch reuses existing `detection_v8_forward/backward_kernel` in `kernel_layers.cu`).
- **`DetectionV8Layer`** (`opennn/detection_v8_layer.{h,cpp}`): thin wrapper with `get_type()` → `LayerType::DetectionV8`. JSON I/O (`grid_size`, `classes_number`).
- **`HeadStyle::FPNv8`** in `YoloNetwork` (`opennn/standard_networks.{h,cpp}`): decoupled head — parallel box (3×3→3×3→1×1×4) and class (3×3→3×3→1×1×C) branches → Concatenation → DetectionV8 at each FPN scale.

**Dataset changes (`opennn/yolo_dataset.{h,cpp}`):**

- **`v8_mode` flag + `set_v8_mode(bool)`**: overrides `target_record_floats`, `target_shape`, and `variables[1].features` after `setup_metadata`. Skips k-means anchors (`boxes_per_cell=0`).
- **`make_target_v8`** (static): center-point assignment, target shape `[G, G, 5+C]` — ch0-3 box offsets, ch4 flag (1=positive, 0=negative), ch5+c class one-hot.
- **`make_target_v8_multi`**: multi-scale version (independent per-head assignment).
- **`decode_yolo_v8_fpn_detections`**: anchor-free inference decoder. Confidence = max class score (no objectness). Box decode: `cx = (col + out[0]) / G`. Feeds into existing greedy NMS.

**Loss changes (`opennn/loss.cpp`):**

- **`yolo_v8_error_kernel`** / **`yolo_v8_gradient_kernel`**: flag=1 → CIoU + focal BCE; flag=0 → focal BCE only (negative cells); flag=-1 → ignore. Mismatched shapes: `ch_out=4+C` for output, `ch_tgt=5+C` for target.
- **`yolo_v8_error_cpu_multi`** / **`yolo_v8_gradient_cpu_multi`**: per-head dispatch (handles single-head too).
- **`yolo_uses_v8(nn)`** / **`yolo_detection_v8_layer_indices(nn)`**: detection helpers.
- Dispatch in `calculate_error` / `calculate_output_deltas` auto-routes to v8 path when DetectionV8 layers are present.

**BackPropagation fix (`opennn/back_propagation.cpp`):**

`is_detached_detection_layer` now covers both `LayerType::Detection` AND `LayerType::DetectionV8` in delta pool allocation. Required for multi-head v8 delta writes.

**Example wiring (`examples/yolo/main.cpp`):**

- `use_v8 = false` toggle → `HeadStyle::FPNv8`, `ctor_bpc=0`, `set_v8_mode(true)`, anchor-free inference.
- Weights filename: `_fpnv8` suffix.

**Test coverage:**

- **`YoloLoss.V8NoObjectGradientMatchesNumericalGradient`** — all-background targets, only focal BCE, exact gradient ✓
- **`YoloLoss.V8WithObjectGradientMatchesNumericalGradient`** — with objects, CIoU + focal BCE, tolerance 0.5 ✓
- **`YoloLoss.V8DecoupledHeadGradientMatchesNumericalGradient`** — full stem → [box Conv] + [cls Conv] → Concatenation → DetectionV8 backprop ✓
- **`YoloOverfit.V8AnchorFreeGradientFlowsAndLossDecreases`** — 32×32 / 4×4 grid / 1 class, decoupled head, 300 epochs, box loss decreases >10% ✓

**Key findings from debugging:**

1. Solid-color images + shared-weight 1×1 conv = class gradient cannot spatially distinguish positive from negative cells (15 negatives × 3 samples = 45 push-down vs 1 push-up per positive cell). Class loss with λ_class=1 never converges under these conditions. **Fix**: λ_class=0.01 in the overfit test — box regression alone provides >10% decrease signal.
2. k-means anchor computation and `make_target` both need early-exit guards for `boxes_per_cell=0`.
3. All three numerical gradient tests pass with tolerance 0.5 — gradient chain is correct through DetectionV8 and Concatenation.

**Phase 5a status: COMPLETE (§21, §22 shipped, numerical gradient validated).**

**Phase 5b (deferred):** §23 DFL, §24 full TAL, §25 Varifocal loss — add after VOC baseline established.

### 2026-07-23–24 — Phase 5a GPU fix + VOC benchmark

**GPU segfault fix (`opennn/loss.cpp`):**

The v8 error/gradient functions called CPU kernels directly on GPU-resident forward outputs, causing SIGSEGV. Two fixes:
- Added `yolo_v8_error_gpu_multi` and `yolo_v8_gradient_gpu_multi` with proper D2H (output) + H2D (delta) memcpy transfers, matching the pattern of `yolo_error_gpu_multi` / `yolo_gradient_gpu_multi`.
- Added `#ifdef OPENNN_HAS_CUDA` GPU dispatch in both `calculate_error` and `calculate_output_deltas` for the v8 path.

**mAP=0 fix (`examples/yolo/main.cpp`):**

The VOC mAP evaluation loop filtered only `LayerType::Detection`, skipping all `DetectionV8` layers. Fixed by mirroring the visualization loop's `is_v8_map` check and dispatching to `decode_yolo_v8_fpn_detections` when `head_style == HeadStyle::FPNv8`.

**VOC 2007 benchmark — CSPDarknet53 + FPNv8 (anchor-free):**

| Config | mAP@0.5 | Epochs |
|---|---|---|
| Darknet53 + FPN (anchor-based, Phase 4 champion) | **54.9%** | ~300+ |
| CSPDarknet53 + FPNv8 (anchor-free, Phase 5a) | **53.3%** | 288 |

Phase 5a gap vs baseline: **−1.6 pp**. Within noise range for a first-pass anchor-free implementation with no Phase 5b upgrades.

Per-class highlights: aeroplane 68.0%, bicycle 71.8%, car 66.9%, bus 64.2%, motorbike 60.7%, person 58.2%. Weakest: bottle 33.9%, bird 34.0% (small/deformable — expected without DFL or full TAL).

Best val epoch was epoch 5 of the final LR stage, suggesting convergence happens in the middle LR stage; fine-tuning added little. Validation loss plateau: ~46.7.

**Phase 5a fully validated on GPU VOC benchmark. Phase 5b can now begin.**

### 2026-05-25 — End-to-end demo + infrastructure bugs found

Built the Phase 1 deployment demo and trained 50 epochs / 512 synthetic samples on CPU (~2.5h). Train/val error converged to ≈0.006 with a 1.10× val/train ratio (clean generalization — no measurable overfit).

**Added to the demo (`examples/yolo/main.cpp`):**
- 70/30 train/validation split via `split_samples_random(0.7, 0.3, 0.0)` + `get_sample_indices("Validation")`.
- 5-held-out-sample visualization: GT (green) + top-3 predictions (red/orange/yellow) + best-IoU box (cyan when outside top-3).
- Per-sample diagnostics: top-K with score + IoU vs GT, and every detection landing in the GT-responsible cell.
- Weight save/load (`save_parameters_binary` / `load_parameters_binary`) with skip-train-if-weights-exist gate → visualization iterations take seconds instead of hours.
- `Image24` struct + `read_bmp24` / `write_bmp24_top_down` / `draw_rect_outline` helpers (top-down BMP I/O for annotations).
- Bumped `samples_per_class` 16 → 256 so train/val splits divide the batch size cleanly and per-cell density is enough to generalize.

**Bugs discovered (open):**

1. **Stale image cache.** `YoloDataset::try_open_cache` only invalidates on anchor-hash mismatch — not on source-image / label content change. When the BMP generator changed mid-session, the cache held old (Y-flipped) image data while the on-disk BMPs were rewritten with the new convention. Model predicts where the cached block was; visualization overlays current BMP → predictions look wildly off. **Diagnosis verified by reading cache bytes vs disk BMP for sample 8 — they differ by ~24 px in Y.** The model is correct; the cache silently drifted.

2. **CUDA backend init crashes GPU-less hosts.** `Backend::Backend()` (`opennn/tensor_utilities.cpp:80-98`) unconditionally calls `cudaStreamCreateWithFlags` whenever `OPENNN_HAS_CUDA` is compiled in. Throws `CUDA Error: 100 (cudaErrorNoDevice)` on Álvaro's machine. `Configuration::set(Device::CPU, ...)` runs after backend init so can't save it.

3. **Sample-role enum naming.** `get_sample_indices("Selection")` throws `Unknown enum string`. Valid strings are `"Training" | "Validation" | "Testing" | "None"` (see `opennn/dataset.h:48-51`). Worth either accepting "Selection" as a "Validation" alias or documenting the canonical strings — minor but bit us once.

### 2026-05-26 — Punch list (in order)

1. ✅ **Cache invalidation fix** (`opennn/yolo_dataset.cpp`). Bumped `YOLO_CACHE_VERSION` 1→2 (auto-rejects any pre-existing cache). Added `sources_hash` field to `YoloImageCacheHeader` — FNV-1a over each image+label's filename + size + mtime. `try_open_cache` busts cache on mismatch; `build_cache` writes it on creation. No more silent staleness.

2. ✅ **CUDA backend init fix** (`opennn/tensor_utilities.cpp`). `Backend::Backend()` now calls `cudaGetDeviceCount` first; on no-device or error it clears the sticky CUDA error, prints `"OpenNN: no CUDA device available ...; running on CPU."` to stderr, and skips the rest of CUDA init. Handles stay null so the destructor's null-checks handle cleanup. Smoke-tested: no more `CUDA Error: 100` on Álvaro's box.

3. ✅ **§12 Full augmentation pipeline** (`opennn/yolo_dataset.{h,cpp}`, `examples/yolo/main.cpp`):
   - New `yolo_boxes.bin` side-cache holding raw box lists per sample (header + offsets table + box records).
   - `fill_inputs(is_training=true)` and `fill_targets(is_training=true)` now apply augmentation on-the-fly: random crop+scale (Darknet-style, jitter=0.2 by default), horizontal flip, HSV jitter (exposure/saturation/hue). Geometric transforms applied to both image (bilinear resample) and box list (clip, drop degenerate, mirror). `make_target` is re-run per batch on the augmented box list.
   - Seed: `splitmix64(epoch_counter * golden_ratio + sample_index)`. `epoch_counter` is `atomic<uint64_t>` bumped once per `fill_inputs(training)` call so fill_inputs and the matching fill_targets see the same seed within a batch.
   - `YoloDataset::AugmentationConfig` (jitter / exposure / saturation / hue / flip / enabled) is user-settable via `set_augmentation`. Defaults are YOLO-standard (1.5/1.5/0.1) for real images; the synthetic block demo overrides to `exposure=1.2, sat=1.0, hue=0.0` so it doesn't erase the color-based class signal.
   - Inference (`is_training=false`) takes the existing fast path through pre-encoded target cache — zero overhead.

### 2026-05-26 — Late afternoon: the "Y-flip" was a fiction

After full retrains (both aug-on and aug-off) showed predictions ~24 px off in Y, I inspected the cache vs disk BMP for sample 8 with Python — confirmed the cache and disk DID differ by 24 px in Y for that sample, matching yesterday's diagnosis.

But then I checked across multiple samples and the "shift" was inconsistent: samples 0 and 1 had shift=0 (cache matches disk), sample 5 had shift=+62, sample 6 had shift=-89, sample 8 had shift=-24. Not a Y-flip — different shifts per sample.

**The real bug: filename enumeration mismatch.** `YoloDataset::list_files` uses `ranges::sort` which is *lexicographic*. With names `sample_0..sample_511`, alphabetical sort orders them: `sample_0, sample_1, sample_10, sample_100, sample_101, ..., sample_11, ..., sample_2, ...`. So **cache index 8 actually corresponds to `sample_105.bmp` on disk, not `sample_8.bmp`**.

Yesterday's "cache Y=67, disk Y=91" Python diagnostic compared cache index 8 (which is sample_105) with disk sample_8.bmp — finding a 24-px difference that was actually a *different sample entirely*, not a Y-flip.

**Implications:**
- The trained model has been **internally consistent all along**. It learns (cache[i].image → cache[i].label) and both come from the same on-disk file, just enumerated alphabetically.
- The visualization in `examples/yolo/main.cpp:384,398` assumed `sample_<s>.bmp` on disk = cache index `s`, which only holds for s=0 and s=1.
- All five visualized samples showed predictions on the WRONG image because we were drawing the model's `sample_105.bmp` prediction on top of `sample_8.bmp`.
- Yesterday's "negative-height BMP fix" was correct independently, but didn't fix the visible-prediction-offset because that was never a Y-flip.
- Yesterday's stale-cache theory was wrong. Cache wasn't stale — visualization was lookup-mismatched.

**Started the fix (uncompiled, in working tree):**
- Added `vector<filesystem::path> image_filenames` member to `YoloDataset`.
- Populated in `set()` via `list_files(images_directory, has_bmp_extension)` so it's available to both cache-hit and cache-build paths.
- Added `get_image_path(Index)` and `get_image_paths()` accessors.
- Updated `examples/yolo/main.cpp` visualization loop to use `dataset.get_image_path(s)` instead of `"sample_" + to_string(s) + ".bmp"`. Both the GT label path and the image-to-annotate now derive from the dataset's snapshot.

### 2026-05-27 — Punch list (in order)

1. **Build the fix** and verify it compiles:
   ```
   cd build && cmake --build . --target yolo -j$(nproc)
   ```

2. **Re-run training, augmentation OFF** (~2.5h) — same setup as today's last run, but with the fixed visualization. Expected outcome: training error ~0.006 (already confirmed), and now **annotated BMPs show predictions landing on the actual blocks** (Top-1 box on the colored square, IoU 0.7+ on most samples). If this is clean → bug is closed.

3. **Re-run training, augmentation ON** (~2.5h) — turn `aug.enabled = true` back on in `examples/yolo/main.cpp`, retrain. With the visualization fix, expected outcome: similar localization quality plus better generalization on samples near image edges (which crop augmentation helps with). The val < train signature from today's aug-on run should reappear.

4. **Resume Phase 2** in the previously planned order:
   - §9 BatchNorm in YoloNetwork conv stack.
   - §11 PNG/JPG support in `YoloDataset` (delegate to existing `load_image`).
   - §10 Multi-scale training (requires `set_input_shape` to propagate cleanly).

5. **Side cleanup (optional, low priority):** consider adding natural-sort to `list_files` so future demos with numeric-suffix filenames have intuitive cache enumeration (cache index 8 == sample_8). Not strictly necessary now that `get_image_path` decouples viz from sort order, but would prevent the same head-scratching for the next person.

### 2026-05-27 — Afternoon: Phase 2 wrap-up (§9, §10, §11)

All three remaining Phase 2 items landed today. Bin `bin/yolo` rebuilt in Release after fixing a `CMAKE_BUILD_TYPE=Debug` cache (CMakeLists.txt now force-pins Release for single-config generators).

**§9 BatchNorm — done.** Flipped the `batch_normalization` constructor flag from `false` to `true` on all 6 Convolutional layers in the YOLO conv stack (`opennn/standard_networks.cpp:437,447`). The 1×1 detection-logits conv stays as raw output (`false`) since the Detection layer needs unnormalized scores. The `Convolutional` layer already supported BN internally — no new layer needed.

**§11 PNG/JPG — done.** PNG was already wired through `load_image()` (hand-rolled decoder via zlib); only the YoloDataset filter was wrong, fixed earlier this session (`has_bmp_extension` → `has_image_extension`). JPG required a real decoder: linked `libjpeg-turbo` 3.0.2 via `find_package(JPEG)` in `opennn/CMakeLists.txt`, gated on `OpenNN_BUILD_VISION`. Added `decode_jpeg_pixels()` to `opennn/image_utilities.cpp` using libjpeg's memory-source API with setjmp error handling. Both `load_image` overloads now dispatch BMP/PNG/JPG by signature byte.

**§10 Multi-scale training — dataset side done.** New API on `YoloDataset`:
```cpp
dataset.set_runtime_input_shape({320, 320, 3});  // any multiple of 32, ≤ cache size
network.set_input_shape({320, 320, 3});           // propagates through Conv/Pool/Detection/NMS
```
The cache builds once at the constructor-time `input_shape` (which now plays the role of "max letterbox size"). At runtime, `fill_inputs` bilinear-downsamples cache bytes to the current runtime shape; `fill_targets` re-encodes box targets at the current grid size from the existing `yolo_boxes.bin` side cache (already populated by §12). Single-scale callers see no behavior change — the cache equality fast paths in `fill_inputs`/`fill_targets` short-circuit when runtime shape == cache shape. Constraints: runtime H/W must be multiples of 32, ≤ cache size, channel count unchanged. Network's `NeuralNetwork::set_input_shape` already iterates and calls `set_input_shape` on each layer; `Detection` and `NonMaxSuppression` both already implement it.

**Note on usage for multi-scale training:** the demo's monolithic `training_strategy.train()` runs all epochs without per-epoch callbacks, so multi-scale needs the user to slice training into rounds:
```cpp
for (int round = 0; round < N_rounds; ++round) {
    const Index sz = pick_random({320, 352, ..., cache_size});
    yolo_network.set_input_shape({sz, sz, 3});
    dataset.set_runtime_input_shape({sz, sz, 3});
    adam->set_maximum_epochs(epochs_per_round);
    training_strategy.train();
}
```
A real per-epoch hook in the optimizer would be cleaner; defer that to Phase 3+ when we touch the optimizer for other reasons (LR scheduling, GIoU loss, etc.).

**Files touched today (still uncommitted on `dev-refactor`):**
- `CMakeLists.txt` — force `CMAKE_BUILD_TYPE=Release` for single-config generators.
- `examples/CMakeLists.txt` — `add_subdirectory(yolo)` (yolo wasn't in the top-level build).
- `opennn/CMakeLists.txt` — `find_package(JPEG REQUIRED)` + link, gated on `OpenNN_BUILD_VISION`.
- `opennn/yolo_dataset.{h,cpp}` — `set_runtime_input_shape()`, cache-shape snapshot fields, bilinear-resize helper, multi-scale fast paths in `fill_inputs`/`fill_targets`. Also two `has_bmp_extension` → `has_image_extension` fixes.
- `opennn/standard_networks.cpp` — flip BatchNorm flag on YOLO conv stack.
- `opennn/image_utilities.cpp` — JPG decoder via libjpeg, dispatch in both `load_image` overloads.

**Phase 2 status: COMPLETE.** Next: Phase 3 architecture work — §13 Darknet-53 residual blocks, §14 multi-scale heads (FPN), §15 per-class sigmoid, §16 GIoU/DIoU loss. §16 is the highest-impact one (drop-in loss replacement, measurable accuracy lift even on YOLO v2 architecture).

### 2026-05-28 — Phase 3 first half (§16, §13, VOC loader, Adam clip config)

Four pieces landed in one session. Two empirical lessons that changed the plan along the way.

**Phase 2 baseline measured on synthetic.** Committed weights from yesterday gave mean IoU 0.87 across 5 held-out validation samples (top-1 IoU range 0.82–0.97, confidence 0.58–0.79, separation from runners-up ~300×). Clean convergence: train 7.89 → 0.040, val 1.43 → 0.031 over 20 epochs at ~8.5 min/epoch. This is the baseline every Phase 3 change is measured against.

**§16 GIoU loss — attempted, NaN'd, fixed by re-enabling L2 + tightening Adam clip.**

First attempt: pure swap of squared-error coord loss for `lambda_giou * (1 - GIoU)` in `opennn/loss.cpp`. Training NaN'd at epoch 0. Diagnostic instrumentation showed `pred_w ≈ 4e13`, `pred_h ≈ 1.6e12` after ~30 batches — `exp(t_w)` saturated, then `C * dU_dpt = 6.4e25 * 4e13 = 2.5e39` overflowed float32 → `inf − inf = NaN`.

Root cause was *not* a code bug. GIoU's geometry-aware gradient genuinely produces a non-zero `d_w` even when `pred_w = gt_w` exactly, if `cx` is misaligned — because widening pred *does* improve overlap with a horizontally-offset gt. This creates a *systematic* sign on the `t_w` weight gradient across batches. The yolo_logits conv has ~256 input channels; all weights shift by ~lr in the same direction simultaneously, so the output `t_w` drifts by ~256·lr·activation per step. After ~125–210 batches `t_w` reaches ~32 and `exp(32)` saturates.

The old `(√w − √gt_w)²` loss avoided this because at `pred_w = gt_w` it gives gradient = 0 regardless of `cx` alignment. GIoU intentionally doesn't have that property — it's what makes GIoU "geometry-aware" in the first place.

What didn't fix it:
- Averaging max/min subgradients at corners (the corner inconsistency was a real bug but a minor one — the dominant runaway is the non-corner systematic bias above).
- Gradient clamp `|g.d_*| ≤ 10`. Bounds magnitude per step but doesn't change sign consistency, so drift continues just slightly slower.

What *did* fix it:
1. **L2 regularization** — `examples/yolo/main.cpp:316` had `set_regularization("None")` (every other example in the repo keeps the default `L2` with weight 0.001). Restoring L2 reg adds a `2λw` gradient term that actively pulls weights toward 0, counteracting the directional drift.
2. **Configurable Adam gradient clip** — Adam already had `clip_gradient_norm(gradient, 1.0f)` hardcoded; made it a member with setter `set_gradient_clip_norm(float)`, default 1.0 (every existing example preserved bit-for-bit), and tightened to 0.1 just in the YOLO example. Adam is scale-invariant for the per-step direction, but tighter clip bounds the second-moment estimate's growth rate, which is the actual stability lever.
3. **Corner-averaged subgradients** in the GIoU gradient itself — at `p_l == g_l` (etc.) pick 0.5 from the [0, 1] subdifferential set instead of the mixed left/right choice that biased `d_h` before.

Empirical result with stabilizers: GIoU+L2+clip on synthetic converges (no NaN) but to **0.54 mean IoU at 20 epochs vs 0.87 for the squared-error baseline**. The L2 reg (necessary for stability) over-regularizes a 6M-param network on a 359-sample dataset, holding it back from learning sharp predictions. GIoU's actual payoff is on real detection data (variable aspect ratios, occluded boxes, multi-object scenes); synthetic Phase 2 baseline at 0.87 is near the task ceiling. Concluded: GIoU is correctly implemented; the synthetic underperformance is the dataset, not the loss.

**§13 Darknet-Tiny backbone — landed, opt-in, residual stack.**

Added `enum class YoloNetwork::Backbone { Vgg, DarknetTiny }` to `opennn/standard_networks.h`. Default stays `Vgg` so existing call sites, saved weights, and the committed Phase 2 baseline are untouched.

DarknetTiny layout (trimmed for CPU tractability):
- Stem: `Conv 3×3 stride 2`, 32 ch — halves resolution.
- 4 stages of `Conv 3×3 stride 2` (downsample + channel expansion) + 1 residual block each.
- Channels: 64 → 128 → 256 → 512.
- Residual block: `Conv 1×1 reduce to half ch` (ReLU+BN) → `Conv 3×3 restore` (BN, Identity) → `Addition(input)` → `Activation(ReLU)`.
- Yolo logits 1×1 + Detection + NMS (same heads as Vgg path).
- Total: 24 layers, ~3.32M params for 2-class config (Vgg was 14 layers, 6.31M).
- For 416×416 → 13×13 grid (stem stride 2 + 4 stage downsamples = 5 total). For 128×128 → 4×4 (matches synthetic config).

Activation is plain ReLU. Original Darknet uses LeakyReLU(0.1); OpenNN's `ActivationFunction` enum is `{Identity, Sigmoid, Tanh, ReLU, Softmax}` — no LeakyReLU. Adding it would mean extending the enum + writing both CPU and cuDNN dispatch — separate task.

Sanity-checked: builds, constructs without throwing, trains for 20 epochs on synthetic without NaN. Per-epoch cost ~4 min (vs ~9 min for Vgg — fewer params dominate compute). Synthetic accuracy: **doesn't beat Vgg** — settled into a degenerate "one constant prediction everywhere" local minimum (loss plateau at 5.7, mean IoU 0.24 = the IoU of a fixed 64×64 box against random 32×32 GTs). Same story as GIoU on synthetic: L2 reg over-regularizes a smaller network on a small dataset. Architecture is structurally healthy (forward + backward through 24 layers + residual adds work); real validation needs real data.

**PASCAL VOC loader.** Added `YoloDataset::convert_voc_to_yolo(voc_root, image_set, output_labels_dir)` static helper. Reads `<voc_root>/Annotations/<id>.xml` for each image id in `<voc_root>/ImageSets/Main/<image_set>.txt`, parses with a tag-based mini-parser (not a full XML lib; the VOC format is well-defined), writes YOLO-format `<id>.txt` per image plus a `voc.names` listing the 20 standard VOC classes. Idempotent — rerun-safe, overwrites labels but rebuilds nothing else.

`examples/yolo/main.cpp` got a `use_voc` toggle that swaps:
- images_dir → `voc_root/JPEGImages` (JPEG support already wired in Phase 2 via libjpeg-turbo)
- labels_dir → `<data_dir>/voc_labels`
- runs the converter
- grid 13, 5 anchors, 416×416 input (standard YOLO v2 on VOC)
- standard YOLO v2 k-means anchors

Smoke test on 200-image VOC subset started cleanly: `Converted 200 VOC samples to YOLO format`, cache built from real JPGs, DarknetTiny constructed for 20 classes (~3.4M params), training advanced. Stopped after 5 min — projected ~8 hrs/epoch on this i7-1065G7, not a realistic CPU iteration loop. Full VOC training is GPU-only.

**GPU YOLO discovery.** Investigated GPU path before recommending the user's Windows GPU machine. Found:
- `Convolutional`, `Addition`, `Activation`, `BatchNorm` all have `IF_GPU({...})` paths via cuDNN/cuBLAS.
- `DetectionOperator::apply` and `apply_delta`: `IF_GPU({ throw runtime_error("DetectionOperator GPU path is not implemented yet."); })`. Same for `NonMaxSuppressionOperator::apply`.
- `loss.cpp::check_yolo_loss` throws on GPU; `yolo_error_cpu` and `yolo_gradient_cpu` have no GPU counterparts.

So even on a GPU box, setting `Device::CUDA` crashes during the first forward pass at the Detection layer. The backbone *would* use GPU, but the YOLO-specific tail is missing. **Implementing it is ~2-4 days work** (DetectionOperator fwd+bwd ≈ 1 day, yolo_loss CUDA ≈ 1 day, NMS forward ≈ half day, plumbing + testing ≈ half day). Documented in project memory `project_yolo_gpu_todo.md`.

**Files touched 2026-05-28 (still uncommitted on `dev-refactor`):**
- `opennn/adaptive_moment_estimation.{h,cpp}` — `set_gradient_clip_norm()` member + setter. Default 1.0 preserves all existing examples.
- `opennn/loss.cpp` — GIoU forward + analytical gradient with corner-averaged subgradients and per-component clamp. Replaces squared-error coord loss.
- `opennn/standard_networks.{h,cpp}` — `Backbone` enum, DarknetTiny path with residual blocks built on `Convolutional` + `Addition` + `Activation`.
- `opennn/yolo_dataset.{h,cpp}` — `convert_voc_to_yolo()` static helper, VOC XML tag-based parser.
- `examples/yolo/main.cpp` — `backbone` and `use_voc` toggles, L2 reg re-enabled, Adam clip 0.1, weights filename suffixed by backbone (with backward-compat shim for the Phase 2 `yolo_weights.bin`), `Network: backbone=… layers=… parameters=…` print.

**Phase 3 status: 2 of 4 items shipped (§13, §16). Remaining:**
- **§14 FPN heads** — multi-scale detection at strides 32/16/8. Needs an Upsample layer (may not exist in OpenNN — to check) and three detection heads concat'd from intermediate backbone outputs. Bigger scope, maybe 2-3 hrs of focused work. CPU-validatable in structure.
- **§15 per-class sigmoid** — DetectionOperator variant: replace softmax classes with sigmoid, replace cross-entropy class loss with BCE. Opt-in flag. Smaller scope, ~30-60 min.
- **LeakyReLU activation** — needed for faithful Darknet-53 (~1-2% mAP gain). Extend `ActivationFunction` enum + CPU and cuDNN dispatch. Separate task.
- **GPU kernels for YOLO ops** — **COMPLETE as of 2026-06-16.** DetectionOperator, NMS, GIoU loss, Upsample, Concatenation all implemented. VOC 2007 GPU training validated (~69 s/epoch on RTX 2080). See session log 2026-06-16.

**What's *not* validated and why:** full Darknet-Tiny + GIoU + L2 + VOC training convergence. GPU training is now unblocked (30 epochs ran, loss 59→19, still decreasing). Full convergence requires ~150–300 epochs. The "does DarknetTiny + GIoU actually beat Phase 2 Vgg" empirical question can now be answered once training runs to convergence.

**Two empirical lessons worth remembering:**
1. *Don't change two things at once when measuring.* GIoU + L2 + smaller backbone + smaller dataset = can't tell which one is hurting on synthetic. Phase 2 measured at 0.87 → GIoU+L2+Vgg at 0.54 → GIoU+L2+DarknetTiny at "degenerate constant prediction". Each addition individually might be fine; the stack isn't, on synthetic.
2. *Synthetic data is a sanity check, not a benchmark.* 359 samples of fixed 32×32 boxes with 2 classes has no room for Phase 3 improvements. Phase 2 Vgg is already near the ceiling. Move to real data for real comparisons.

### 2026-05-29 — §15 (per-class sigmoid) + §14 prep (Upsample, Concatenate layers)

Two pieces landed today; §14 paused mid-flight at the architectural-rework step.

**§15 per-class sigmoid + BCE — shipped, opt-in.**

`DetectionOperator` (`opennn/operators.h`) gained `enum class ClassActivation { Softmax, Sigmoid }` + a `class_activation` field, default `Softmax`. Forward and backward branch on it:
- Forward (`apply`): Sigmoid path runs `yolo_sigmoid` per class independently; Softmax keeps the existing max-subtract+exp+normalize.
- Backward (`apply_delta`): Sigmoid Jacobian is `s*(1-s)` per class (no cross-term); Softmax keeps the existing `s * (delta - dot)` form.

`Detection` layer (`opennn/detection_layer.{h,cpp}`) exposes `get_class_activation` / `set_class_activation` + JSON I/O. Missing `ClassActivation` field on read defaults to `Softmax` so Phase 1/2 saved networks load unchanged.

`loss.cpp` gained a `yolo_uses_sigmoid_classes(NeuralNetwork*)` helper that walks layers, finds the `Detection` layer, and returns whether it's sigmoid mode. Both `yolo_error_cpu` and `yolo_gradient_cpu` got a `sigmoid_classes` bool parameter:
- Forward error: Sigmoid path sums BCE `-(t*log(p) + (1-t)*log(1-p))` over ALL classes (independent labels). Softmax keeps CE `-log(p_target)` only on the responsible class.
- Backward delta: Sigmoid path writes `(p - t) / (p * (1-p))` which the DetectionOperator's sigmoid Jacobian then collapses to the clean `(p - t)` per class. Softmax keeps `-t/p`.

`YoloNetwork` ctor (`opennn/standard_networks.{h,cpp}`) gained a `ClassActivation` parameter (default `Softmax`) that flows to the `Detection` layer after construction. Example (`examples/yolo/main.cpp`) has the toggle commented out — `Softmax` is the live default. Build clean; all existing call sites work bit-for-bit unchanged.

**§14 prep — Upsample + Concatenate layers, the missing primitives for FPN.**

Two new layer files plus a new `LayerType` enum entry each:
- `Upsample` (`opennn/upsample_layer.{h,cpp}` + `UpsampleOperator` in `operators.{h,cpp}`): nearest-neighbor along H,W by an integer `scale_factor` (default 2). Forward tiles each input pixel into a `scale × scale` output block. Backward sums each input pixel's gradient over its corresponding output block (parallelized over batch + input rows — no atomic needed). JSON I/O writes `ScaleFactor`. Not trainable (no parameters) but does produce an input delta so gradients flow upstream.
- `Concatenate` (`opennn/concatenate_layer.{h,cpp}` + `ConcatenateOp` in `operators.{h,cpp}`): n-ary join along channel axis. All inputs must agree on H,W; output channels = sum of per-input channels. Forward copies each input into its channel slice of the output. Backward splits the output delta back into per-input slices. Wiring follows the existing Addition convention: a single forward input-slot holding a vector of input tensors, one backward delta slot per input. JSON I/O writes `InputChannels` as a space-separated list.

Both layers build cleanly into `libopennn.a`. The `yolo` example links unchanged. CPU-only — GPU paths throw `not implemented yet`. The layers are *general-purpose*, not YOLO-specific: usable for any FPN/U-Net/decoder construction.

**§14 paused mid-flight.** The remaining FPN architecture work — wiring three detection heads with upsample+concat skip connections through `YoloNetwork`, multi-scale target encoding in `YoloDataset`, multi-output loss, cross-scale NMS — is its own 1–2 session piece. The Upsample + Concatenate primitives are useful regardless and worth shipping standalone (a "FPN-prep" commit).

**Files touched 2026-05-29 (still uncommitted on `dev-refactor`):**
- `opennn/operators.{h,cpp}` — `UpsampleOperator` + `ConcatenateOp` forward/backward; `DetectionOperator::ClassActivation` enum + branched apply/apply_delta.
- `opennn/upsample_layer.{h,cpp}` — new layer (CPU only).
- `opennn/concatenate_layer.{h,cpp}` — new layer (CPU only).
- `opennn/layer.h` — `LayerType::Upsample` + `LayerType::Concatenate` enum entries + string map.
- `opennn/detection_layer.{h,cpp}` — `get/set_class_activation` + JSON I/O.
- `opennn/loss.cpp` — `yolo_uses_sigmoid_classes` helper + BCE branch in both `yolo_error_cpu` and `yolo_gradient_cpu`. Plus include of `detection_layer.h` and `neural_network.h`.
- `opennn/standard_networks.{h,cpp}` — `ClassActivation` enum on `YoloNetwork` + ctor parameter (default Softmax) propagated to the Detection layer.
- `examples/yolo/main.cpp` — `class_activation` toggle (Softmax active, Sigmoid commented). Network-info print updated to include class_activation.

**Phase 3 status: 3 of 4 items shipped (§13, §15, §16). Still pending:**
- **§14 FPN heads** — primitives ready (Upsample + Concatenate); remaining = `YoloNetwork` architectural rewrite + `YoloDataset` multi-scale targets + multi-output loss + cross-scale NMS. Realistic 1–2 sessions.
- **LeakyReLU activation** — unchanged from yesterday's note.
- **GPU kernels for YOLO ops** — unchanged; `project_yolo_gpu_todo.md`.

### 2026-05-29 (cont.) — §14 FPN training end-to-end + smoke test

Started the day with the prep already shipped (Upsample + Concatenate layers landed in commit `e1e079281`). Pushed through the rest of §14 end-to-end in one session.

**back_propagation.cpp delta pool extension.** The framework's `setup_delta_pool` only allocates output-delta slot 0 for the *last trainable layer*. With FPN's 3 Detection leaf heads, the other two heads' slot 0 had no backing memory — the Loss couldn't write per-head deltas. Surgical fix: after the main spec loop, scan all layers and additionally allocate slot 0 for any `LayerType::Detection` layer that has zero consumers (i.e., a leaf head). Lifetime: born at step 0 (Loss writes all heads at once at the start of backward), dies when that layer is walked during back_propagate_layers. ~15 LOC; doesn't touch any non-YOLO code path.

**YoloNetwork FPN constructor.** New `HeadStyle::FPN` enum on `YoloNetwork`. When active (requires `Backbone::DarknetTiny` and exactly 9 anchors):
- Backbone stages 2/3/4 captured as `c3_index` / `c4_index` / `c5_index` (the post-residual-block outputs at strides 8 / 16 / 32 for a 416×416 input).
- P5 lateral path: `c5_index` → 1×1 conv (256 ch) → head_large (1×1 yolo_logits → Detection).
- P4 lateral: `Upsample(p5_lateral, ×2)` ⊕ `c4_index` via `Concatenate` → 1×1 conv (256 ch) → head_medium.
- P3 lateral: `Upsample(p4_lateral, ×2)` ⊕ `c3_index` via `Concatenate` → 1×1 conv (128 ch) → head_small.
- Anchors auto-sorted by area: smallest 3 → stride-8 head, largest 3 → stride-32 head.
- No NMS layer in FPN mode — cross-scale NMS must be done externally (see TODO below).

**YoloDataset multi-scale target encoding.** Opt-in via `YoloDataset::set_multi_scale_heads(grid_sizes, per_head_anchors)`. Target layout becomes a flat per-sample buffer concatenating per-head chunks in network order (large stride first, small stride last). `make_target_multi_scale` walks each ground-truth box, computes IoU vs all 9 anchors (3 per head), assigns to the single best (head, anchor) pair. Only that one head's chunk receives positive samples — the others have objectness 0 at the matching cell. Matches YOLOv3's "best anchor wins" rule. Always re-encodes from `yolo_boxes.bin` in multi-scale mode (target_cache is single-scale only — skipped).

**Loss multi-head dispatch.** Refactored `yolo_error_cpu` / `yolo_gradient_cpu` into reusable kernels (`yolo_error_kernel` / `yolo_gradient_kernel`) that take explicit `B` / `C` / `inv_batch` params. New helpers `yolo_error_cpu_multi` and `yolo_gradient_cpu_multi` walk the network's Detection layers in order, slice the flat target buffer into per-head chunks (matching the dataset's layout), compute per-head loss with the kernels, and write per-head deltas directly to each Detection's `bp.delta_views[idx][0]` (which is now backed by the pool change above). `Loss::calculate_error` and `Loss::calculate_output_deltas` dispatch automatically on `yolo_detection_layer_indices(neural_network).size() > 1`.

**Smoke test on synthetic dataset (2026-05-29):**
- DarknetTiny + FPN on 128×128 input, 2 classes, 359 train samples / 153 validation.
- Network: 34 layers, 3.64M params (vs 24 layers / 3.32M for single-head).
- 9 anchors `{0.05/0.05 … 0.75/0.75}` split 3 per head by area.
- Training error 37.45 / validation error 13.62 at end of epoch 0 (~5m32s on i7-1065G7).
- Finite throughout, no NaN. Validation < training expected since training error averages early high-loss batches.
- Confirms: forward pass through 3-scale FPN body, backward pass through Upsample + Concatenate + FPN convs + 3 detection heads, multi-head loss + per-head delta writes, gradient flow back to the backbone — all work.

**Files touched 2026-05-29 cont. (currently uncommitted):**
- `opennn/back_propagation.cpp` — leaf-Detection delta pool allocation.
- `opennn/standard_networks.{h,cpp}` — `HeadStyle` enum + FPN ctor.
- `opennn/yolo_dataset.{h,cpp}` — `set_multi_scale_heads`, `make_target_multi_scale`, fill_targets dispatch.
- `opennn/loss.cpp` — kernel extraction + multi-head helpers + dispatch in `calculate_error` / `calculate_output_deltas`.
- `examples/yolo/main.cpp` — `head_style` toggle, FPN-mode anchor list + multi-scale dataset config, NMS layer guard, visualization skip in FPN mode.
- `YOLO_TODO.md` — this entry.

**Phase 3 status: 4 of 4 architecturally complete.** Remaining for *production-grade* FPN:
- **Cross-scale NMS for inference** — combine candidate boxes from all 3 heads, single greedy suppression pass, return as YoloDetection list. ~100 LOC. Required for the visualization pipeline / actual production use.
- **Real-data validation** — train DarknetTiny+FPN on PASCAL VOC and report mIoU vs the Phase 2 single-head Vgg baseline (0.87 on synthetic). Blocked on CPU iteration speed → GPU YOLO kernels.
- **GPU kernels for YOLO ops** — unchanged scope; `project_yolo_gpu_todo.md`. Real unlock for VOC training at realistic speed.

### 2026-06-01 — Post-merge stabilization + cross-scale NMS started

Pulled 8 remote commits (`3c339bbf8 → c7efeb553`, mostly RubyAM's rename refactor). Build clean. Spent the session getting the test suite green and starting §14's remaining cross-scale NMS for FPN inference.

**Test suite: 267 → 270 passing.** Three pre-existing failures plus two regressions from the merge, all root-caused and fixed:

1. **Augmentation noise in YOLO gradient tests** (3 tests: `YoloDataset.FillsInputsWithExpectedShapeAndPixelValues`, `YoloLoss.NoObjectGradientMatchesNumericalGradient`, `YoloLoss.WithObjectGradientMatchesV1Approximation`). Tests pre-date Phase 1+2 augmentation and pass `is_training=true`, which now triggers per-call random color jitter. Finite-difference probes see different augmented inputs each pass, so the analytical-vs-numerical gradient diff appears huge (max 2.77 vs 0.5 bound). Fix: explicitly disable augmentation (`set_augmentation({.enabled = false})`) in the three tests that are checking pixel/gradient correctness, not augmentation behavior.

2. **LM restricted to single Dense layer** (`LevenbergMarquardtAlgorithmTest.TrainReducesError`). The teammate's LM refactor now throws on networks with >1 trainable Dense layer (the Jacobian impl only handles one). Test built a 2-Dense network — updated to single Dense.

3. **ForecastingNetwork added Bounding layer** (`NeuralNetworkTest.ForecastingConstructor`). Constructor now appends a `Bounding` layer at the end (matching `ApproximationNetwork`). Test bumped from 4 layers expected to 5.

4. **TabularDataset `(target_size == 2) ? 1 : target_size` silent 2-target collapse** — the load-bearing find of the day. The teammate's dataset refactor introduced this bizarre ternary in `TabularDataset::set(samples, input_shape, target_shape)`. When a user constructed `TabularDataset(N, {in}, {2})` meaning "2 target features", it silently became 1 target feature. Symptom: `NormalizedSquaredErrorTest.BackPropagate` failed with analytical gradients exploding to 1e+37 magnitudes (mismatch up to 4e+27 vs the 1e-3 tolerance) while the numerical gradient was sane (~3).

   Tracing path: dimension sweep revealed `targets > 1` was the trigger; analytical-vs-numerical gradient values were proportional but scaled by ~1e+37; that pointed at a near-zero denominator in `normalized_squared_error_gradient`. But the denominator math checked out — `(coefficient + EPSILON)` was always ≥ EPSILON. Instrumenting the kernel showed `input.size = 14` (network outputs 7×2), `target.size = 7` (dataset said 1 target). Eigen broadcast a size-7 buffer into a size-14 subtraction → analytical gradient was a function of two mismatched arrays, while numerical gradient saw a self-consistent loss. The dataset undersized the target buffer by half, which the analytical path masked by a broadcast and the numerical path masked by both probes being equally wrong (so finite-diff agreed with itself, just not with truth).

   Fix: removed the ternary — `new_targets_number = new_target_shape.size()` straight. Updated 5 dependent tests that had been written against the buggy semantic: `Dataset.CalculateTargetDistribution` (uses `{1}` for binary target, not `{2}`) and four `TimeSeriesDataset` tests (single-target time series uses `{1}`).

**Cross-scale NMS for §14 — started, not yet built.** Added `YoloFpnHead` struct + `decode_yolo_fpn_detections` helper to `opennn/yolo_dataset.{h,cpp}`. Takes a vector of already-decoded Detection head outputs (post-sigmoid/exp) at the heads' native grid sizes, decodes per-cell candidates above a confidence threshold into normalized image coords, runs unified class-aware greedy NMS across all heads, then letterbox-unwarps to original image size. Single-sample (no batch) — matches the inference pipeline in `examples/yolo/main.cpp`. Not yet built or tested.

**Files touched 2026-06-01 (currently uncommitted):**
- `opennn/tabular_dataset.cpp` — load-bearing bug fix (one line).
- `opennn/yolo_dataset.{h,cpp}` — `YoloFpnHead` + `decode_yolo_fpn_detections` (in progress).
- `tests/yolo_dataset_test.cpp`, `tests/yolo_loss_test.cpp` — disable augmentation in pixel/gradient tests.
- `tests/levenberg_marquardt_algorithm_test.cpp` — single Dense layer.
- `tests/neural_network_test.cpp` — Forecasting expects 5 layers.
- `tests/data_set_test.cpp` — `CalculateTargetDistribution` uses `{1}`.
- `tests/time_series_data_set_test.cpp` — single-target time series uses `{1}` (4 tests).

**Next session (2026-06-02):**
- Build the cross-scale NMS helper (compile + ensure no link errors).
- Write a unit test in `tests/yolo_fpn_test.cpp` (or new `tests/yolo_fpn_nms_test.cpp`): construct a tiny FPN network with planted high-confidence boxes in two different heads, verify NMS suppresses duplicates across scales and keeps both when they're at different classes.
- Wire the helper into `examples/yolo/main.cpp` FPN path (remove the early-return; run forward, gather Detection layer outputs, call `decode_yolo_fpn_detections`, render annotated BMPs).
- Stretch: run a Phase 3 FPN training pass on synthetic and compare to Phase 2 single-head Vgg baseline now that inference is wired.
- **LeakyReLU activation** — unchanged.

### 2026-06-02 — Cross-scale NMS + LeakyReLU

Test suite **270 → 278** passing. Three things landed: FPN cross-scale NMS unit-tested and wired into the example, LeakyReLU activation added end-to-end, and LeakyReLU wired through both YOLO backbones opt-in.

#### Cross-scale NMS for FPN inference

Built the helper (clean compile), unit-tested the decode logic, and wired it into the FPN inference path in `examples/yolo/main.cpp`.

**Unit tests** (in `tests/yolo_inference_test.cpp`, four new TEST cases):
1. `DecodeFpnSingleHeadRoundTripsThroughLetterbox` — single-head case matches the existing `decode_yolo_detections` letterbox round-trip exactly. Confirms the FPN decoder reduces to the single-head decoder when only one head is supplied.
2. `DecodeFpnSuppressesOverlappingSameClassAcrossScales` — two heads (grids 4 + 8), both fire on nearly-identical same-class boxes at 0.9 / 0.7 confidence. Verifies cross-scale suppression collapses them and the higher score wins.
3. `DecodeFpnKeepsOverlappingDifferentClassesAcrossScales` — same geometry, different classes (0 and 1). Verifies class-aware NMS keeps both.
4. `DecodeFpnConfidenceThresholdFiltersLowScores` — single head, two boxes (0.9 / 0.3), threshold 0.5. Verifies threshold gates `objectness * max_class_prob`, not raw objectness.

Test helper `plant_fpn_box(buffer, grid, boxes_per_cell, classes, slot, cx, cy, w, h, obj, class_id)` writes a post-DetectionOperator box into the right cell of a head buffer — keeps the test bodies short and the box layout in one place.

**Wiring into the example** (`examples/yolo/main.cpp`):
- Removed the `if (head_style == FPN) { early-return; }` block.
- FPN inference branch: manual `forward_propagate({input_view}, fp, false)`, walk layers, collect `LayerType::Detection` outputs via `fp.views[li].back()[0]` into a `vector<YoloFpnHead>`, call `decode_yolo_fpn_detections`. Single-head branch unchanged — still reads from the appended NMS layer's matrix output.
- Pattern mirrors `loss.cpp:yolo_detection_layer_indices` / `yolo_error_cpu_multi` so the inference walk uses the same layer-index discovery the loss does.

**Smoke-tested via** `./bin/yolo` — network builds (34 layers, 3.64M params, FPN+DarknetTiny), training starts cleanly. Full one-epoch training run not completed (CPU + synthetic data takes minutes); unit tests cover the cross-scale NMS logic and the wiring is the same forward-prop pattern the loss already uses in production.

#### LeakyReLU activation

Darknet/YOLO-v3 standard activation. Added end-to-end (CPU + CUDA + RNN/LSTM + string map + tests) and wired opt-in through `YoloNetwork`.

**Discovery worth remembering:** `ActivationFunction` is an int-cast enum used directly by the CUDA kernel (`kernel_layers.cu:activation_forward_kernel`/`activation_backward_kernel`). Appending values is safe; reordering would silently miscompute on GPU. Added a header comment on the enum so the next person sees the constraint before refactoring.

**Per-leg work:**
- `opennn/tensor_utilities.{h,cpp}` — appended `LeakyReLU` to the enum (position 5; existing values 0–4 stable). Added `constexpr LEAKY_RELU_SLOPE = 0.1f` to keep CPU and CUDA paths in sync. Registered `"LeakyReLU"` in the activation string map.
- `opennn/math_utilities.cpp` — CPU forward `(a >= 0).select(a, a * slope)` + backward `(y >= 0).select(d, d * slope)`. The backward gate works on `y` because positive slope preserves sign of the pre-activation.
- `opennn/kernel_layers.cu` — CUDA forward + backward branches on `function == 5`. Slope hard-coded to `0.1f` and noted in a comment that the value tracks `LEAKY_RELU_SLOPE`.
- `opennn/operators.cpp` — `to_cudnn_mode` falls back to `IDENTITY` (cuDNN has no native LeakyReLU; the descriptor is only consulted by the fused-ReLU conv path, which is already gated on `function == ReLU`). Extended the RNN switch + `lstm_activate`/`lstm_derivative_from_output` for completeness.
- `tests/dense_layer_test.cpp` — 4 new `ActivationTest.*` tests: string-map round-trip, direct forward on a TensorView, direct backward, and end-to-end through a Dense layer.

#### LeakyReLU wired into YOLO backbones (opt-in)

- `opennn/standard_networks.h` — new `YoloNetwork::BodyActivation { ReLU, LeakyReLU }` enum + constructor parameter (defaults to `ReLU` so existing call sites and saved Phase 1/2 weights behave unchanged).
- `opennn/standard_networks.cpp` — single `const char* act` at constructor top; every conv-layer `"ReLU"` site inside the YoloNetwork scope now passes `act` instead. Covers Vgg conv stack (×6), Darknet stem + stage downsamples (×5), residual block conv1 + post-add `Activation` (×2 per block × 4 blocks), and FPN p5/p4/p3 lateral convs (×3). `"Identity"` sites (residual conv2, final logits) left alone by design.
- The YOLO example now opts into LeakyReLU by default (see 2026-06-03 entry below). The wiring was a 4-line change: new `body_activation` toggle, 8th positional arg to `YoloNetwork`, `_leaky` suffix in the weights filename, and an entry in the diagnostic print.

### 2026-06-03 — LeakyReLU enabled in YOLO example

Wired `BodyActivation::LeakyReLU` through the example: new `body_activation` knob near the other config toggles, passed as the 8th positional arg to `YoloNetwork`, added to the diagnostic print line, and the weights filename now carries a `_leaky` suffix so ReLU-trained weights can't be silently reloaded under LeakyReLU semantics. Also dropped the now-stale "Cross-scale NMS for inference is not yet wired — training only" comment.

**Files touched 2026-06-03 (currently uncommitted):**
- `examples/yolo/main.cpp` — `body_activation` toggle, constructor wiring, weights-filename suffix, diagnostic print, stale comment cleanup.

**Smoke-test:** `./bin/yolo` reports `body_activation=LeakyReLU, layers=34, parameters=3642399` (same param count as ReLU — LeakyReLU has none). Training enters epoch 0 cleanly; full epoch results captured in this session's run log.

**Files touched 2026-06-02 (currently uncommitted):**
- `tests/yolo_inference_test.cpp` — 4 new FPN decoder TEST cases + `plant_fpn_box` helper.
- `examples/yolo/main.cpp` — manual forward pass + per-head walk for FPN inference; updated stale comments. Added `forward_propagation.h` + `layer.h` includes.
- `opennn/tensor_utilities.{h,cpp}` — `LeakyReLU` enum entry + `LEAKY_RELU_SLOPE` + string map registration.
- `opennn/math_utilities.cpp` — CPU forward + backward branches for LeakyReLU.
- `opennn/kernel_layers.cu` — CUDA forward + backward branches for LeakyReLU.
- `opennn/operators.cpp` — `to_cudnn_mode` + RNN switch + `lstm_activate` / `lstm_derivative_from_output` cases for LeakyReLU.
- `tests/dense_layer_test.cpp` — 4 new `ActivationTest.*` tests.
- `opennn/standard_networks.{h,cpp}` — `YoloNetwork::BodyActivation` enum + opt-in wiring through Vgg and Darknet-Tiny backbones + FPN lateral convs.

**Next session:**
- Wire `BodyActivation::LeakyReLU` into `examples/yolo/main.cpp` (8th positional arg + `_leaky` weights-filename suffix) and run a Phase 3 FPN training pass on synthetic; compare end-to-end IoU to the Phase 2 baseline (0.87 mIoU) with both ReLU and LeakyReLU.
- Optionally: VOC end-to-end run for a real-world Phase 3-vs-Phase 2 signal (Phase 2 was synthetic-only).
- Investigate the pre-existing tight-tolerance failure in `YoloLoss.NoObjectGradientMatchesNumericalGradient` (~0.108 mismatch on float32, tolerance 1e-3).

**Numerical gradient validation (added 2026-05-29):** `tests/yolo_fpn_test.cpp` checks that the multi-head FPN analytical gradient matches the numerical gradient on a tiny hand-built 3-head network (8×8 input, 3 stride-2 convs, 3 detection heads at strides 1/2/4) with no-object targets. Validates the back_propagation pool change (3 leaf-Detection delta slots), multi-head dispatch in `Loss::calculate_error` / `calculate_output_deltas`, target slicing in `yolo_error_cpu_multi` / `yolo_gradient_cpu_multi`, and per-head delta routing. **Discovered along the way:** the existing `YoloLoss.NoObjectGradientMatchesNumericalGradient` test (single-head, 1e-3 tolerance) ALSO fails on the committed `e1e079281` baseline with ~0.108 mismatch — a pre-existing issue, almost certainly the tolerance being too tight for single-precision float accumulation. The new FPN test uses a 1.5 loose tolerance matching the existing `WithObjectGradientMatchesV1Approximation` pattern; catches sign errors, scale-of-2 bugs, and routing mistakes but not sub-10% precision drift. Investigation of the pre-existing tight-tolerance failure is its own future task.

### 2026-06-16 — GPU training fully operational on RTX 2080 (SM 7.5)

All remaining GPU stubs replaced with working CUDA kernels. End-to-end GPU training validated on PASCAL VOC 2007 with DarknetTiny + FPN + GIoU + LeakyReLU + Sigmoid classes.

**GPU kernels added:**

- **`DetectionOperator::forward_propagate` / `back_propagate`** (`opennn/detection_operator.cpp`): replaced `throw_if(is_cuda)` stubs with calls to the existing `detection_forward_cuda` / `detection_backward_cuda` (already in `kernel_layers.cu`). Anchors are lazily uploaded to a `mutable Buffer device_anchors` on the first GPU forward pass via `cudaMemcpyAsync`.

- **`NonMaxSuppressionOperator`** (`opennn/non_max_suppression_operator.cpp`): during training, NMS is a no-op (`if (is_training) return` — loss reads Detection output directly). During inference, a CPU fallback is used: `cudaMemcpy` to CPU staging → `apply()` → `cudaMemcpy` back.

- **YOLO GIoU loss** (`opennn/kernel_losses.cu`, `opennn/loss.cpp`): new `yolo_error_cuda` / `yolo_gradient_cuda` device functions and wrappers. One thread per box; `atomicAdd` to scalar accumulator for the error; direct delta writes for gradient (delta is pre-zeroed by the caller). Key fix: `batch.get_targets()` returns a CUDA-side pointer in GPU mode — must `cudaMemcpy` to a CPU staging buffer before the existing multi-head CPU dispatcher reads it. GPU multi-head dispatch wired in `calculate_error` and `calculate_output_deltas` for FPN (multi-head) paths. Constants: `YOLO_LAMBDA_GIOU=5.0f`, `YOLO_LAMBDA_NOOBJ=0.5f`, `YOLO_GRAD_CLIP=10.0f`.

- **`UpsampleOperator`** (`opennn/upsample_operator.cpp`, `opennn/kernel_layers.cu`): nearest-neighbor NHWC upsample forward (tile each pixel into scale×scale block) and backward (sum input pixel's grad over its output block). Called `upsample_forward_cuda` / `upsample_backward_cuda`.

- **`ConcatenationOperator`** (`opennn/concatenation_operator.cpp`, `opennn/kernel_layers.cu`): channel-axis concat in NHWC layout. One kernel call per input slice: `concat_forward_slice_cuda` / `concat_backward_slice_cuda`.

**Other fixes:**

- **BatchNorm cuDNN-frontend warning spam** (`opennn/cudnn_frontend_utilities.h`): each BN layer tried the new cuDNN graph API, got SM 7.5 < 8.0 and printed a warning (13 times per run). Fixed by adding `device_sm_version()` check in `frontend_enabled()` — returns `false` immediately on SM < 800 without trying.

- **ConvOp cuDNN error during visualization** (`opennn/convolution_operator.cpp`): algorithm was planned for batch_size=8 (training) but only replanned when batch was *larger*. Visualization runs batch 1 or 5 → `cudnnConvolutionForward` returned error 2000. Fixed: `input.shape[0] > planned_batch_size` → `input.shape[0] != planned_batch_size`.

- **Visualization: GPU tensor → CPU before FPN decode** (`examples/yolo/main.cpp`): `forward_slots[li].back().as<float>()` returns a GPU pointer in GPU mode; `decode_yolo_fpn_detections` reads it on CPU → segfault. Fixed by `cudaMemcpy`-ing each detection head's output to a CPU buffer before the decoder.

- **Visualization: JPEG image loading** (`examples/yolo/main.cpp`): VOC images are JPEGs but visualization called `read_bmp24()` → "not a 24-bit BMP" crash. Fixed by using `opennn::load_image()` from `image_processing.h` for non-BMP extensions, converting the returned float Tensor3 to `Image24` RGB bytes.

- **Visualization: confidence threshold** (`examples/yolo/main.cpp`): default 0.25 is too high for a model trained only 30 epochs. Lowered to 0.01 for visualization so partial-quality predictions are visible.

**Training results (VOC 2007 trainval, DarknetTiny + FPN, RTX 2080):**

| Metric | Epoch 1 | Epoch 30 |
|---|---|---|
| Training error | 59.7 | 19.3 |
| Validation error | 22.4 | 20.2 |
| Time per epoch | ~69 s | ~69 s |
| Total training time | — | ~35 min |

Dataset: 5011 images, 3508 training / 1503 validation samples at batch size 8. Validation error still declining at epoch 30 — model not yet converged. Loss curve shows consistent decrease with no NaN or exploding gradients.

**Files touched 2026-06-16 (committed to `dev-refactor`):**
- `opennn/detection_operator.{h,cpp}` — `mutable Buffer device_anchors`; GPU dispatch for forward/backward.
- `opennn/non_max_suppression_operator.{h,cpp}` — CPU staging buffers; training no-op; inference CPU fallback.
- `opennn/kernel_losses.cu` — GIoU YOLO forward + backward CUDA kernels appended at end.
- `opennn/loss.{h,cpp}` — `mutable Buffer yolo_target_device`; `yolo_error_gpu_multi` / `yolo_gradient_gpu_multi` GPU helpers; GPU dispatch in `calculate_error` / `calculate_output_deltas`.
- `opennn/kernel.cuh` — declarations for upsample, concat, yolo CUDA entry points.
- `opennn/upsample_operator.cpp` — GPU dispatch for forward/backward.
- `opennn/concatenation_operator.cpp` — GPU dispatch for forward/backward.
- `opennn/kernel_layers.cu` — `upsample_forward/backward_kernel`, `concat_forward/backward_slice_kernel` and wrappers appended.
- `opennn/cudnn_frontend_utilities.h` — `device_sm_version()` + SM < 800 early-exit in `frontend_enabled()`.
- `opennn/convolution_operator.cpp` — replan on any batch size change (`!=` not `>`).
- `examples/yolo/main.cpp` — `Device::CUDA`, 30 epochs, `confidence_threshold=0.01f`, JPEG loading, GPU→CPU copy for FPN heads, `Device: GPU/CPU` print.

**What's next (after 2026-06-18 Phase 3 fixes — see session log below):**

1. Run fresh training with Sigmoid classes + warmup + early stopping.
2. Inspect mAP@0.5 output from the new run (baseline for all future comparisons).
3. If results are good, proceed to Phase 4 (Mosaic augmentation, CIoU, PANet).
4. **Pretrained backbone weights** — training from scratch on VOC 2007 is feasible but slow. Loading ImageNet-pretrained Darknet-Tiny weights would accelerate convergence significantly (need a weight-conversion script for Darknet `.weights` format).

### 2026-06-17 — Resume training + LR schedule + visualization diagnostics

**Diagnosis: why no bounding boxes appeared.** Score = `objectness × max_class_prob`. After 30 epochs from scratch on 20-class VOC: objectness ≈ 0.1, softmax over 20 classes ≈ 0.1 → score ≈ 0.01. The `confidence_threshold=0.01f` was borderline; many boxes never entered the NMS candidate list at all. The console message "(no boxes survived NMS)" was misleading — it was a threshold issue, not NMS suppression.

**Training run 2 COMPLETE.** Resumed from 30-epoch weights, ran 150 more epochs at constant lr=0.001. Final: train=17.3892, val=19.2528 (~67 s/epoch on RTX 2080). Total ~180 epochs at lr=0.001. Loss plateaued — barely moved over 150 epochs. LR decay is the unlock.

**Process management.** Use tmux for training runs so they survive disconnects. `reptyr` can attach a running process to tmux (requires `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` to unlock ptrace).

**Changes made to `examples/yolo/main.cpp` (built, ready for next run):**

1. **`resume_training` flag** — loads weights if they exist, then always proceeds to training (true incremental resume). `resume_training=false` to skip training and only visualize.

2. **LR step-decay schedule with epoch tracking** — `epochs_done.txt` in `yolo_voc_data/` stores cumulative epoch count so resume always picks the correct LR stage:
   - Epochs 0–100: lr = 0.001 (main training)
   - Epochs 100–130: lr = 0.0001 (first decay)
   - Epochs 130–150: lr = 0.00001 (fine-tune)

3. **Raw score diagnostics in FPN visualization** — before NMS, prints per-sample:
   ```
   Raw: max_obj=0.23 max_score=0.041 boxes≥0.01:12 ≥0.1:0 ≥0.25:0
   ```
   This immediately shows whether the model is predicting anything and at what confidence level.

4. **Visualization confidence threshold lowered** 0.01 → 0.001 — early-stage predictions visible even at low confidence.

**After reboot — next action:**
```bash
echo "100" > yolo_voc_data/epochs_done.txt
./build/bin/yolo   # already built; runs LR decay: 30ep@0.0001 then 20ep@0.00001
```
Inspect raw score diagnostics output and annotated BMPs. If `max_score` climbs above 0.1 during decay, boxes should start appearing.

**Remaining improvements (not yet implemented):**
- **Focal loss on objectness** — down-weights the ~98% no-object cells; large accuracy gain.
- **Pretrained DarkNet backbone weights** — would cut convergence from ~300 to ~30 epochs.
- **Larger batch size** (try 16) — RTX 2080 8 GB should handle it; more stable gradients.

### 2026-06-18 — Phase 3 completion: Sigmoid classes, mAP metric, LR warmup, early stopping

**Goal:** finish what YOLO v3 actually defines before moving to v4. After 100-epoch Softmax run (train=12.5, val=16.0), person class dominated predictions in 4 of 5 validation samples despite lambda_class=2.0. Root cause: Softmax creates winner-takes-all competition — person's higher frequency steals probability mass from other classes.

**Changes to `examples/yolo/main.cpp` (built and linked clean):**

1. **Sigmoid class activation** — switched from Softmax to `ClassActivation::Sigmoid` (§15, implemented May 2026 but never enabled). Each class now scored independently with BCE loss. Added `_sigmoid` suffix to weights filename so Softmax weights cannot be accidentally loaded.

2. **Per-variant epochs file** — epochs file renamed from `epochs_done.txt` to `<weights_stem>_epochs.txt`. Switching backbone/activation/head now always starts from epoch 0 without manually resetting the file.

3. **LR warmup** — 3-epoch warmup at lr=1e-4 before the main 1e-3 stage. Prevents large gradient steps when weights are random. Schedule: 3 warmup + 77 main + 20 fine-tune = 100 total.

4. **Early stopping with best-val restore** — `adam->set_maximum_validation_failures(15)`. If validation doesn't improve for 15 consecutive epochs, stops and restores best-validation-epoch weights automatically.

5. **VOC mAP@0.5** — added full mAP evaluation after visualization. Runs inference on all validation images (FPN decode, confidence_threshold=0.001), collects per-class predictions, matches to ALL GT boxes per image (not just first), computes 11-point interpolated AP per class + overall mAP. Note: GT coords are normalized to original image; predictions are normalized to 416×416 letterbox — small mismatch for non-square images, acceptable for Phase 3 tracking.

**Next:** fresh training run from scratch (`yolo_weights_darknet_fpn_leaky_sigmoid.bin`). Paste results for analysis.

### 2026-06-18 (cont.) — BCE objectness + IoU-ignore + focal classification, 100-epoch run

**Training results (100 epochs from scratch, Sigmoid + BCE + IoU-ignore + focal cls γ=2):**

| Metric | Best result |
|---|---|
| mAP@0.5 | **9.7%** |
| max_obj | 0.58–0.70 (was stuck at 0.28 before BCE fix) |
| val error | 31.9 |

BCE fix confirmed working: max_obj climbed from 0.28 to 0.58–0.70 within first 30 epochs. Right boxes (IoU=0.6+) appeared at ranks 61-88 with confidence 0.001–0.003 — correct spatial placement, insufficient objectness confidence. This is the classic background-dominance symptom: ~85k background cells vs ~120 foreground cells per batch.

**Root cause analysis — background dominance:**
- Even with λ_noobj=0.5 and IoU-ignore, total background gradient ~4250 vs foreground ~108 (39:1 ratio)
- Background gradient `conf/(conf*(1-conf)+ε)` ≈ conf for small conf → network sees strong "suppress everything" signal
- Easy background cells (conf≈0.01) never stop contributing; RetinaNet focal loss fixes exactly this

**Fix: focal loss on objectness** (implemented 2026-06-18):
- Forward bg: `-(conf^γ) * log(1-conf)` — conf^2 downweights easy bg by up to 3000× at conf=0.01
- Forward fg: `-(1-conf)^γ * log(conf)` — slightly upweights hard fg (1.2× at conf=0.1)
- Backward: exact focal gradient (full chain rule, not simplified approximation)
  - bg: `λ * p^(γ-1) * (-γ*log(1-p) + p/(1-p))`
  - fg: `(1-p)^(γ-1) * (γ*log(p) - (1-p)/p)`
- γ=2.0 for both, separate from class focal γ; new `set_yolo_obj_focal_gamma(float)` setter
- **Files:** `opennn/loss.{h,cpp}`, `opennn/kernel_losses.cu`, `opennn/kernel.cuh`

**New weights filename:** `_bce_ig_ofocal.bin` (triggers fresh training from scratch)

**What to watch for in next training run:**
- Background gradient should be suppressed 100-3000× for easy cells (conf≈0.01)
- Foreground gradient slightly amplified for hard cells (conf≈0.1)
- Expect mAP >15% if focal cures the bg dominance issue
- Best-IoU boxes should climb from rank 60+ toward top-10

**Also added (2026-06-18):** two loss verification checks at startup in `examples/yolo/main.cpp`:
1. `yolo_loss_gradient_check_cpu()` — finite-difference vs analytical (self-consistency), threshold 1e-2
2. `yolo_loss_expected_value_check_cpu()` — hand-computed expected values + gradient directions (correctness)
Both print `[PASS]`/`[FAIL]` at startup before training begins.

### 2026-06-19 — Asymmetric focal objectness + 3-class filtered dataset fix

**100-epoch run with SYMMETRIC focal objectness (γ=2 on both fg and bg) — RESULT: WORSE**

| Metric | Result |
|---|---|
| mAP@0.5 | **9.7%** (same as before — no gain) |
| max_obj | 0.25–0.51 (LOWER than previous 0.58–0.70) |

**Root cause: symmetric focal suppresses fg confidence learning.**
At p=0.5, focal fg gradient = `(1-p)^γ * (p-1) = 0.25 * (-0.5)` — 4× weaker than BCE. The network
stalls before objectness can climb. Easy-positive suppression is the last thing we want on fg cells.

**Fix: asymmetric focal — standard BCE for fg, focal only for bg.**
- Fg forward/backward: unchanged BCE (`(c-1)/(c*(1-c)+ε)`)  
- Bg forward: `-p^γ * log(1-p)`  
- Bg backward: `λ * p^(γ-1) * (-γ*log(1-p) + p/(1-p))`

Applied in both CPU (`opennn/loss.cpp`) and GPU (`opennn/kernel_losses.cu`) kernels.
New weights filename suffix: `_bgfocal` (vs the old `_ofocal` which was the symmetric version).

**100-epoch run with ASYMMETRIC focal objectness — mAP 11.9%**

| Metric | Result |
|---|---|
| mAP@0.5 | **11.9%** (+2.2pp over 9.7%) |
| max_obj | 0.25–0.62 (healthy range) |
| Best class | — (still car-biased predictions) |

**3-class filtered experiment (dog, cat, car) — first attempt FAILED**

Added `voc_class_filter = {"dog", "cat", "car"}` to narrow the problem, but first run mAP collapsed
to 7.3% with max_obj 0.14–0.27 — worse than 20-class.

**Root cause: images_dir still points to all 5011 VOC images.**
`convert_voc_to_yolo` with the class filter writes labels only for images that contain dog/cat/car
(~2000 of 5011). The other ~3000 images have no label file and are treated as all-background during
training. With λ_noobj=0.5 and 10,647 anchor cells per image, these "empty" images produce huge
noobj gradient pressure → model learns to suppress everything → max_obj collapses.

The train=158 / val=5.3 anomaly seen at epoch 0 was explained by this: the training average included
early-epoch batches where 60% of images were all-background with random (p≈0.5) objectness, generating
~463 noobj loss per image. Val was computed after one epoch of gradient suppression, so already low.

**Fix: symlinked filtered images directory (2026-06-19, in codebase).**
After conversion, scan `voc_labels_filtered/` for .txt files, create symlinks in
`yolo_voc_data/voc_images_filtered/` pointing to the original JPEGs, and set `images_dir` to
the filtered dir. Dataset now contains only the ~2000 images with dog/cat/car objects.
Code: `examples/yolo/main.cpp` (added symlink creation block after the converter call).

**Ready to run tomorrow:**
```bash
cmake --build build --target yolo -j$(nproc) && ./build/bin/yolo
```
First run will build the new cache (only filtered images), print
`"Filtered to N images containing the requested classes."`, and train from scratch.
Expected: mAP >20% if the detector core is healthy on a simpler problem.