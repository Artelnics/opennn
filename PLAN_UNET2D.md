# 2D U-Net plan

Target: a 2D binary/semantic segmentation U-Net following nnU-Net design
choices, built out of OpenNN layers and operators.

## Status (2026-10-05)

| Step | Status |
|------|--------|
| Pre-fixes (LayerRegistryTest, GPU bias throw, bak file) | ✓ done |
| 1: SegmentationDataset | ✓ done (9 tests pass) |
| 2: TransposedConvolution | ✓ done (11 CPU tests pass) |
| 3: InstanceNormalization | ✓ done (5 tests pass, 1188 total) |
| 4: Dice + CE loss | ✓ done (5 tests pass, 1193 total) |
| 5: UNet model | pending |
| 6: Example | pending |

## 1. Inventory: exists vs. missing

| nnU-Net 2D component | Status in OpenNN | Notes |
| --- | --- | --- |
| Conv 3x3, `"Same"` padding, arbitrary stride | Exists, `Convolutional` | Stride already supported, so strided-conv downsampling needs no new code |
| Strided-conv downsampling | Exists | `Convolutional(..., stride {2,2}, "Same")` |
| **Instance** norm with affine gamma/beta | Missing | Only `BatchNormalizationOperator`, which normalizes across the batch; nnU-Net normalizes per sample because it trains at batch size 2 |
| LeakyReLU | Exists | `ActivationFunction::LeakyReLU`, slope `0.1`; nnU-Net uses 0.01 |
| Transposed conv upsampling (`ConvTranspose2d(2,2)`) | Missing | `Upsampling` is nearest-neighbour only. Needs a new operator, layer and CUDA kernels |
| Skip-connection concatenation | Exists | `Concatenation`, channel concat, `add_layer(layer, {a, b})` |
| Dice + cross-entropy loss | Missing | `Loss::Error` has no Dice and no multi-term loss. Binary and categorical CE exist |
| Per-pixel sigmoid / softmax output | Exists | Last `Convolutional` with `"Sigmoid"`; `uses_binary_cross_entropy` auto-detects one output channel |
| Image + mask dataset | Missing | `ImageDataset` is one label per image with a rank-1 target. `load_image` and `resize_image` in `opennn/dataset/image_processing.h` are reusable |
| Dice metric | Missing from `Evaluation` | Computed in the example and reported by `judge.sh` |

The combined loss must be rank-aware. `binary_cross_entropy` divides the
elementwise sum by `input.get_shape()[0]`, which is the batch for a rank-4
segmentation tensor but the height for a rank-3 one, so the reported value and
the per-pixel gradient are both scaled by the pixel count.

## 2. Architecture

```
Scaling(input_shape, ScalerMethod::MeanStandardDeviation)
level 0 (H)   : [Conv3x3 s1 + Norm + LeakyReLU] x2
down to H/2   : Conv3x3 s2 + Norm + LeakyReLU
level 1 (H/2) : [Conv3x3 s1 + Norm + LeakyReLU] x2
down to H/4   : Conv3x3 s2 + Norm + LeakyReLU
level 2 (H/4) : [Conv3x3 s1 + Norm + LeakyReLU] x2
down to H/8   : Conv3x3 s2 + Norm + LeakyReLU          <- bottleneck
level 2       : TransposeConv2x2 -> Concat(skip 2) -> [Conv3x3 + Norm + LeakyReLU] x2
level 1       : TransposeConv2x2 -> Concat(skip 1) -> [Conv3x3 + Norm + LeakyReLU] x2
level 0       : TransposeConv2x2 -> Concat(skip 0) -> [Conv3x3 + Norm + LeakyReLU] x2
head          : Conv1x1 "Same" -> Sigmoid (binary) / Softmax (multiclass)
```

| Input | Depth | Channels per level | Parameters |
| --- | ---: | --- | ---: |
| 64x64 synthetic | 4 | 32, 64, 128, 256 | about 7.8 M |
| 128x128 Oxford Pets | 5 | 32, 64, 128, 256, 320 | about 31 M |

`ConvTranspose2d(2, stride 2, padding 0)` gives `out = 2 * H_in`, which matches
the skip resolution exactly.

## 3. Implementation order

| # | Component | Files | Tests |
| --- | --- | --- | --- |
| 1 | `SegmentationDataset` | `opennn/dataset/segmentation_dataset.{h,cpp}`, entry in the `OpenNN_BUILD_VISION` filter | `tests/dataset/segmentation_dataset_test.cpp`: shapes, `fill_inputs`/`fill_targets` against written images, resize, sample-role splits |
| 2 | `TransposedConvolution` | `opennn/network/operators/transposed_convolution_operator.{h,cpp}`, `opennn/network/layers/transposed_convolution_layer.{h,cpp}`, `kernel_transposed_convolution.{cu,cuh}`, `layer_registry` | `tests/network/layers/transposed_convolution_layer_test.cpp`: output shape, forward against hand-computed values, numerical gradients, CPU/CUDA agreement |
| 3 | `InstanceNormalization` | `opennn/network/operators/instance_norm_operator.{h,cpp}`, `kernel_instance_norm.{cu,cuh}`, wiring in `Convolutional` | `tests/network/operators/instance_norm_operator_test.cpp` plus a `Convolutional` case: per-sample statistics, forward values, numerical gradients |
| 4 | Dice + CE loss | `opennn/training/loss.{h,cpp}`, `opennn/training/error_functions.{h,cpp}`, `opennn/training/kernel_losses.{cu,cuh}` | `tests/training/dice_cross_entropy_test.cpp`: perfect, inverted and empty masks, rank-4 scaling, numerical gradients |
| 5 | `UNet` model | `opennn/models/models.h`, `opennn/models/vision_models.cpp` | `tests/network/unet_test.cpp`: output shape equals input spatial size, skip wiring, forward and numerical gradients end to end |
| 6 | Example and judge | `examples/unet_segmentation/main.cpp`, `examples/CMakeLists.txt`, `examples/README.md`, `judge.sh`, `experiments.log` | |

Step 3 is the only one that touches shared code, because `Convolutional` is
used by ResNet, Yolo, the detection heads and the ONNX exporter. The change is
strictly additive: the flag defaults off and throws if combined with batch
normalization. The full CPU and CUDA suites run after it.

## 4. Iteration

`judge.sh` runs in phases and exits with a distinct status: build (2), unit
tests (3), synthetic smoke train under two minutes (4), then prints validation
Dice and exits 0 when it reaches 0.95, otherwise 1.

Every attempt is logged in `experiments.log` with the attempt number, the change
made, the Dice score and the elapsed time. The best checkpoint is kept outside
git. Iteration stops after 3 attempts without improvement, or after 20
attempts. Phase 5 repeats the loop on Oxford-IIIT Pet with 500 training and 100
validation images at 128x128, targeting Dice >= 0.85.

## 5. Deliberate deviations from nnU-Net

- LeakyReLU slope 0.1 instead of 0.01, because `LEAKY_RELU_SLOPE` is shared
  with the language models and changing it would alter existing models.
- Dice and CE are both computed per sample and averaged over the batch. The CE
  term is a true per-pixel mean rather than OpenNN's batch-only mean, so the 1:1
  mix reproduces nnU-Net's relative weighting.
- No mirroring, spatial dropout or elastic augmentation in the first pass.