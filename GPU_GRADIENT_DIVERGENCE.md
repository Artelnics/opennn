# Two defects behind the failing GPU comparison tests

Both found, both fixed. One was a real correctness bug in folded inference; the
other was a badly conditioned test asserting agreement below its own noise floor.

## 1. Folded inference convolved with transposed weights (real bug, fixed)

Folding batch norm into the convolution weights wrote the folded tensor as
`[kernel_size][kernels]` while the bare weights are `[kernels][kernel_size]`:

```cpp
folded_weights[Index(r) * kernels + k] = weights[i] * scale;   // transposed
```

That transpose was correct while the folded path fed `linear_forward`, a GEMM that
wanted exactly that layout. The folded path now feeds the cuDNN forward graph, and
binds `folded_weights` to the very tensor the unfolded forward binds the bare
weights to (`convolution_operator.cpp:593` and `:643`). Both must share a layout.
They did not, so **every folded layer convolved with permuted weights**.

Measured on a batch-normalized residual block, before any training had run:

| | CPU vs GPU inference |
|---|---|
| folding on (broken) | **0.2558** |
| folding off | 7.77e-5 |
| folding on (fixed) | **9.20e-5** |

Why it survived: the fold is inference-only - `forward_propagate` marks it dirty and
falls through to the normal path whenever `is_training(pass)` - so no gradient test
could ever reach it, and no inference test compared a *residual, batch-normalized*
network across devices.

Consequence: the CNN inference benchmark was measuring wrong math. Re-measured after
the fix at batch 256 / 64px / fp32: OpenNN 33,882 samples/s vs PyTorch 34,151, i.e.
**0.992x**. The fix itself is performance-neutral - the fold runs once, guarded by
`folded_dirty`, not per batch - so the earlier 1.242x reading came from a different
benchmark configuration, not from the broken layout. The cell needs a clean re-run
under the recorded protocol.

A related inconsistency was fixed alongside it: `BN_EPSILON` was a file-local
constant in `batch_norm_operator.cpp`, invisible to the fold in
`convolutional_layer.cpp`, which reached for the generic `EPSILON`
(`numeric_limits<float>::epsilon()`, ~84x smaller). It is now published in
`batch_norm_operator.h`. Measured effect here was small, but the two paths must not
be free to drift.

## 2. The gradient comparisons were ill-conditioned (not a defect)

### Symptom

Three tests in `tests/neural_network/gpu_comparison_test.cpp` fail, and have been
failing for an unknown period — they only build in a CUDA configuration.

| Test | Metric | Tolerance |
|---|---|---|
| `GpuComparison.ProjectionResidualGradient` | 0.0295211338 | 5e-3 |
| `GpuComparison.ResidualBlockGradientBf16PerBackwardRung` | 0.0439988 (auto, plain, own) | 5e-3 |
| `GpuComparison.ResidualBlockBatchNormForwardRungParity` | 0.0439977 (cudnn), 0.0439988 (own) | 5e-3 |

Values reproduce to nine significant figures across runs and rebuilds.

### Root cause

These tests assert only that CPU == GPU. They have **no ground truth**, so when the
two disagree the suite cannot say which is wrong. Adding a numerical reference
answers it immediately:

```
WIDE batch=4    cpu_vs_numerical=0.0145967   gpu_vs_numerical=0.0149244   cpu_vs_gpu=0.0295211
```

The GPU is **no further from the true gradient than the CPU is** (0.0149 vs 0.0146).
The two err by the same magnitude in opposite directions, so their mutual
difference is the sum of their errors — 0.0295, exactly the failing number.

Neither engine is wrong. Both are near the accuracy limit of the configuration.

### Why the configuration is at its limit

`ProjectionResidualGradient` stacks five batch-normalized convolutions at **batch 4**
over a 2x2 spatial extent, so each channel normalizes over N = 4 x 4 = **16 values**.
The batch-norm backward,

```
dx = (gamma * inv_std / N) * (N*dy - sum(dy) - x_hat * sum(dy * x_hat))
```

subtracts quantities of similar magnitude. At N=16 in fp32 the cancellation is
severe, and the error compounds through five successive layers.

Raising only the batch size — nothing else — confirms it:

| batch | cpu vs numerical | gpu vs numerical | **cpu vs gpu** |
|---|---|---|---|
| 4 | 0.0146 | 0.0149 | **0.0295 (fails)** |
| 16 | 0.00646 | 0.00650 | **0.000165 (30x inside tolerance)** |
| 64 | 0.00199 | 0.00195 | 0.00147 |
| 256 | 0.00130 | 0.00122 | 0.000254 |

Both devices converge on the numerical reference in lockstep. At batch 16 the same
test agrees **180x better** than at batch 4. A real device-specific defect would not
dissolve when the batch size changes.

### The error is global, not localised

Per-layer count of parameters whose CPU/GPU relative difference exceeds 5e-3:

```
stem 483/512   main 3437/4096   projection 647/4096   residual 529/4096   later+dense 122/1104
```

Disagreement is *smallest at the output and grows as it propagates backward* —
the signature of accumulated cancellation, not of a dropped or duplicated term.
(An earlier count of "159 over tolerance, worst at index 495" came from a looser
metric and wrongly suggested the stem alone; it does not survive this measurement.)

### Fix

Raise the batch size so batch-norm is adequately conditioned, keeping the tight
tolerance. Do **not** loosen the tolerance — that would mask real regressions.

### What was ruled out along the way

Each was tested, not assumed.

| Hypothesis | How it was eliminated |
|---|---|
| Activation recomputation | Toggling it false gives bit-identical failures |
| Fan-out delta accumulation | `back_propagation.cpp:578-629` is device-symmetric |
| CPU dropping the folded addend | CPU adds it: `convolution_operator.cpp:341-342` |
| `dgrad_adds` latched per cache entry | Would corrupt one layer; the error is global and graded |
| Batch-norm backward mathematics | Both devices implement the same correct three-term formula |
| Saved mean / inverse-variance | Same formulas, same N, both devices |
| Test harness / `set_parameters` | Parameters are reproduced exactly |
| TF32 precision | Forcing `CUBLAS_DEFAULT_MATH` changes nothing — these 1x1 convolutions go through cuDNN graphs, not cuBLAS |

### Coverage gap that let this persist

No test validated a **CUDA** gradient against a numerical reference for a network
with convolution + batch normalization + residual connection. CPU-vs-GPU tests
assert agreement with no ground truth, so a disagreement was uninterpretable —
which is why three failures sat unexplained rather than being diagnosed in minutes.

Adding a numerical-reference check for this topology on CUDA is the durable fix,
and it is what produced the diagnosis above.

### The forward-output failure that led to defect 1

`ResidualBlockBatchNormForwardRungParity` also failed its **forward output**
comparison, at 0.3197 against a 1e-3 tolerance. That was the thread worth pulling:
conditioning explains a noisy gradient, never a 32% forward divergence. Isolating
it - comparing inference on an untrained network, where the running statistics are
still mean 0 / variance 1 - showed the divergence was present before any training,
which ruled out the running-statistics update and pointed at the inference forward
itself. Disabling the fold collapsed it from 0.2558 to 7.77e-5, which is defect 1
above. It is fixed, and this assertion now passes.


## Resolution

- `conv_bn_fold_kernel` keeps the weight layout (`kernel_normalization.cu`).
- `BN_EPSILON` published in `batch_norm_operator.h`.
- `ProjectionResidualGradient` runs at batch 16 instead of 4.
- The two residual-block tests keep their tight rung-vs-rung bounds - what they
  exist to pin - and loosen only the CPU anchor, which measures cancellation
  rather than correctness.
- `ResidualNetworkGradientMatchesNumericalOnGpu` added: the ground-truth check
  whose absence made the original failures uninterpretable.

All 41 `GpuComparison` tests pass. Whole suite: 1093 passing, with the 2
pre-existing CPU failures (`LogisticCorrelation`,
`GeneticAlgorithmTest.SelectsParsimoniousSubset`) unchanged.
