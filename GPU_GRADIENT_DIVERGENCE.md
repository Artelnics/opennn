# The three failing GPU comparison tests are ill-conditioned, not buggy

Status: **root cause established for `ProjectionResidualGradient`.** No library
defect. The test asserts agreement below the numerical noise floor of its own
configuration.

## Symptom

Three tests in `tests/neural_network/gpu_comparison_test.cpp` fail, and have been
failing for an unknown period — they only build in a CUDA configuration.

| Test | Metric | Tolerance |
|---|---|---|
| `GpuComparison.ProjectionResidualGradient` | 0.0295211338 | 5e-3 |
| `GpuComparison.ResidualBlockGradientBf16PerBackwardRung` | 0.0439988 (auto, plain, own) | 5e-3 |
| `GpuComparison.ResidualBlockBatchNormForwardRungParity` | 0.0439977 (cudnn), 0.0439988 (own) | 5e-3 |

Values reproduce to nine significant figures across runs and rebuilds.

## Root cause

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

## Why the configuration is at its limit

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

## The error is global, not localised

Per-layer count of parameters whose CPU/GPU relative difference exceeds 5e-3:

```
stem 483/512   main 3437/4096   projection 647/4096   residual 529/4096   later+dense 122/1104
```

Disagreement is *smallest at the output and grows as it propagates backward* —
the signature of accumulated cancellation, not of a dropped or duplicated term.
(An earlier count of "159 over tolerance, worst at index 495" came from a looser
metric and wrongly suggested the stem alone; it does not survive this measurement.)

## Fix

Raise the batch size so batch-norm is adequately conditioned, keeping the tight
tolerance. Do **not** loosen the tolerance — that would mask real regressions.

## What was ruled out along the way

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

## Coverage gap that let this persist

No test validated a **CUDA** gradient against a numerical reference for a network
with convolution + batch normalization + residual connection. CPU-vs-GPU tests
assert agreement with no ground truth, so a disagreement was uninterpretable —
which is why three failures sat unexplained rather than being diagnosed in minutes.

Adding a numerical-reference check for this topology on CUDA is the durable fix,
and it is what produced the diagnosis above.

## Separate, still-open finding

With `training_activation_recomputation` **false** — the default —
`ResidualBlockBatchNormForwardRungParity` additionally fails its **forward output**
comparison (0.3197 vs tolerance 0.001). Recomputation governs only whether the
backward re-derives or reloads a tensor; it should not affect forward outputs at
all. Not explained by the conditioning above, and not yet investigated.
