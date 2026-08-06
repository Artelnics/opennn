# Maximum batch to target: FP32 audit (RTX 3060 Laptop)

Run dates: 2026-08-05 to 2026-08-06. GPU: NVIDIA GeForce RTX 3060
Laptop, 6 GiB.
Frameworks: OpenNN (working tree at `2ad3ee7`), PyTorch 2.5.1+cu121,
TensorFlow 2.21.0. The capacity cap was 5,632 MiB for HIGGS and
Transformer and 5,888 MiB for ResNet-50.

## What was measured

Each engine first searched its largest successful training batch. That
capacity JSON was then hashed and consumed unchanged by the target runner.
For each of three seeds, the runner trained to a common training-loss gate and
integrated GPU board power sampled at 20 Hz only between the engine's
`TRAIN_START_UNIX` and `TRAIN_END_UNIX` markers.

The final gates were HIGGS BCE 0.65, ResNet-50 cross-entropy 2.0, and
Transformer non-PAD token cross-entropy 7.5. ResNet was repeated after the
original 2.5 gate was found to be above the uniform-random 10-class baseline
of `ln(10) = 2.303`. HIGGS and ResNet repeat one resident,
class-balanced maximum batch; Transformer makes full passes over a bounded
4,096-pair corpus. These are training-loss gates, not validation-quality
claims.

## Capacity result

| Workload | OpenNN | PyTorch | TensorFlow | OpenNN vs PT | OpenNN vs TF |
|---|---:|---:|---:|---:|---:|
| HIGGS | 348,416 | 345,600 | 290,304 | +0.8% | +20.0% |
| ResNet-50, capacity-first (16 MiB workspace) | 5,396 | 3,127 | 1,712 candidate / 1,562 stable | +72.6% | 3.45x vs stable |
| ResNet-50, faster OpenNN point (256 MiB workspace) | 5,163 | 3,127 | 1,562 stable | +65.1% | 3.31x |
| Transformer | 80 | 68 | 58 | +17.6% | +37.9% |

TensorFlow ResNet batch 1,712 passed one capacity attempt but failed when
repeated in target training. The energy runs therefore use the verified
1,562 batch and preserve both the original candidate and override in the
artifact.

OpenNN batch 5,396 also trains with a 256 MiB workspace, but peaks at
5,927 MiB and exceeds the common 5,888 MiB cap. A fresh bounded search found
5,163 (peak 5,887 MiB) as the valid maximum for that faster workspace.

Capacity artifacts:

- `docs/benchmarks/results/gpu-higgs-max-batch-combined-20260805.json`
- `docs/benchmarks/results/gpu-resnet50-max-batch-cifar10-combined-resident-20260805.json`
- `docs/benchmarks/results/gpu-resnet50-max-batch-cifar10-20260806T161118Z.json`
- `docs/benchmarks/results/gpu-transformer-max-batch-combined-20260805.json`

## Time and GPU energy to target

| Workload | Engine | Batch | Median steps/epochs | Median time | Median GPU energy |
|---|---|---:|---:|---:|---:|
| HIGGS | OpenNN | 348,416 | 7 steps | 2.626 s | 199.752 J |
|  | PyTorch | 345,600 | 13 steps | 3.702 s | 324.766 J |
|  | TensorFlow | 290,304 | 11 steps | 10.515 s | 606.467 J |
| ResNet-50 | OpenNN | 5,163 | 8 steps | 17.933 s | 871.828 J |
|  | PyTorch eager | 3,127 | 5 steps | 3.378 s | 276.953 J |
|  | TensorFlow graph | 1,562 | 7 steps | 27.471 s | 1,716.903 J |
| Transformer | OpenNN | 80 | 2 epochs | 33.999 s | 3,452.616 J |
|  | PyTorch | 68 | 2 epochs | 47.060 s | 3,913.834 J |
|  | TensorFlow | 58 | 2 epochs | 105.053 s | 7,158.515 J |

Final artifacts and their raw per-run logs/power CSVs:

- `docs/benchmarks/results/gpu-higgs-max-batch-to-target-final-higgs-20260805.json`
- `docs/benchmarks/results/gpu-resnet50-max-batch-to-target-corrected-ce2-final-20260806.json`
- `docs/benchmarks/results/gpu-transformer-max-batch-to-target-final-transformer-20260805.json`
- `docs/benchmarks/results/evidence/max-batch-to-target-final-*-20260805/`
- `docs/benchmarks/results/evidence/max-batch-to-target-corrected-resnet50-ce2-*-20260806/`

The older ResNet artifacts are retained as diagnostic evidence, but are
superseded by the corrected CE 2.0 artifact.

## Conclusions

HIGGS is the positive case: maximum batch and the execution path combine
well. Against PyTorch, OpenNN is 29.1% faster and uses 38.5% less GPU energy;
against TensorFlow it is 75.0% faster and uses 67.1% less energy.

The originally reported ResNet result was wrong. The energy runner failed to
propagate `TF_XLA=0` in its ResNet branch, so a result selected as TensorFlow
graph actually ran XLA and included its compilation. The OpenNN run also used
a 16 MiB cuDNN workspace whose convolution-backward plan dominated the step,
and the CE 2.5 gate sat above random chance. The corrected run uses graph mode,
CE 2.0, a 30 W idle baseline, and a bounded 256 MiB OpenNN workspace.

At the corrected maximum batches, OpenNN is 34.7% faster and uses 49.2% less
GPU energy than TensorFlow. It is still 5.31 times slower and uses 3.15 times
the energy of PyTorch. Most of the PyTorch gap is no longer a mysterious
24-fold engine penalty: OpenNN processes 41,304 samples before the gate versus
15,635 for PyTorch (2.64 times more), and consumes 0.0211 versus 0.0177 joules
per processed sample (19.2% more). PyTorch throughput is still 1.94 times
higher, which is the main remaining kernel/runtime target.

The 256 MiB workspace alone changed the CE 2.5 diagnostic from a median
72.774 s / 6,003.092 J to 29.538 s / 1,845.269 J at batch 5,396. Enforcing the
VRAM cap then selected batch 5,163 and avoided a slower convolution-plan
threshold, producing the final 17.933 s / 871.828 J result. This is direct
evidence that maximizing batch independently of plan selection is the wrong
objective for ResNet.

The Transformer execution result is useful for profiling but is not yet a
publishable framework-quality comparison. OpenNN's SDPA path infers padding
lengths from the current source activation
(`attention_operator.cpp:310-334`) and caches the result by pointer. A buffer
address stays constant while its batch contents change, so later batches can
reuse stale lengths. Deeper encoder/decoder activations can also make padded
rows non-zero, whereas the PyTorch and TensorFlow scripts derive masks from
token IDs. Capacity numbers remain meaningful; convergence-to-target can be
biased until mask propagation is fixed.

The apparent 171.0 s versus 25.7 s TensorFlow instrumentation anomaly was the
missing `TF_XLA=0`, not a 20 Hz observer penalty. With the command fixed,
instrumented TensorFlow graph runs have a 27.471 s median. Sampling overhead
should still be validated, but it does not explain the old result.

## Highest-value improvements

1. **Make the capacity search throughput-aware.** The current ResNet runner
   tries workspace modes from 16 MiB upward and stops at the first passing
   mode. It should retain every passing `(batch, workspace, recomputation)`
   point, benchmark several steady steps, and select joules/time to target
   under the cap. The 5,396-to-5,163 result shows this has first-order impact.
2. **Autotune within a bounded workspace.** In
   `cudnn_frontend_utilities.h`, autotuning is allowed only when the workspace
   cap equals zero. Add a mode that filters plans above a positive cap and
   times all remaining plans, then caches the winner by full convolution
   shape/type. This directly targets the convolution-backward bottleneck seen
   in the saved `OPENNN_PROFILE=1` log.
3. **Use a portable initialization and a validation gate.** A common integer
   seed does not create common weights across frameworks. Export one canonical
   weight/BN-state fixture, verify identical initial logits on a small batch,
   and gate on held-out loss or accuracy below chance. Keep framework-default
   initialization as a separate product benchmark.
4. **Fix Transformer mask ownership.** Compute encoder and decoder valid
   lengths once from token IDs per batch, store them separately in
   `ForwardPropagation`, and pass them to every self/cross-attention layer.
   Remove pointer-identity as a validity cache key. Add a regression test with
   consecutive batches having different padding patterns.
5. **Add selective recomputation.** Make checkpointing configurable by
   ResNet stage, and add Dense/FFN activation checkpointing for HIGGS and the
   Transformer. Gate every option on the joint energy search; ResNet shows why
   capacity-only enabling is unsafe.
6. **Measure the instrument.** Add `--power-sample-ms`, run paired 50/100/200
   ms controls, and reject a sampling rate when it changes uninstrumented wall
   time by more than 2%. Prefer direct NVML or a hardware power meter.
7. **Separate cold and steady-state products.** Keep the present cold
   compile-to-target metric, but also save a precompiled/steady-state variant.
   TensorFlow XLA took about 80 s even at ResNet batch 100, so it is unsuitable
   for the short cold run but may win for longer training.
8. **Require stable capacity.** Re-run the final passing batch at least three
   times and back off automatically on any OOM. Record both the single-shot
   candidate and stable maximum, as the ResNet TensorFlow case demonstrates.
