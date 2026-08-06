# Maximum batch to a fixed training-loss target

This benchmark joins the capacity and energy tracks. For HIGGS, ResNet-50, and
the encoder-decoder Transformer it:

1. consumes an immutable max-batch JSON produced by the workload's capacity
   runner;
2. selects each engine's own largest successful training batch;
3. trains with that batch until the same training-loss target;
4. integrates GPU board power over the exact training window; and
5. retains the complete engine log and raw 20 Hz power trace.

The primary metrics are total GPU joules and wall time to target. The artifact
also records batch, steps or epochs, final error, average power, SM clock,
framework versions, GPU/driver, git state, the capacity-artifact SHA-256, and
paths to all raw evidence.

## Important interpretation

HIGGS and ResNet-50 repeat a single maximum-size resident batch. This isolates
the combined capacity/execution effect without adding input-pipeline memory
that was absent from the capacity trial. The Transformer uses the prepared
token corpus and gates on epoch-mean non-PAD token cross-entropy.

The common gate is training loss, not held-out quality. A second validation
benchmark is required before making a generalization or accuracy claim.
For classification, choose a loss gate below the uniform-random baseline
(`ln(classes)` for categorical cross-entropy, `ln(2)` for balanced binary
cross-entropy). Otherwise the result is dominated by framework-default
initialization rather than training.

## Example

```bash
python run_max_batch_to_target.py \
  --workload resnet50 \
  --capacity-json ../../results/gpu-resnet50-max-batch-cifar10-RUN.json \
  --target 2.0 --max-steps 20 --precision fp32 --runs 3 \
  --workspace-mib 256 --tf-xla 0 --idle 30 \
  --bench-python ~/.venvs/opennn-combined/bin/python \
  --opennn-bin ~/build-opennn-combined/bin/opennn_resnet50_maxbatch_trial \
  --data-dir "$OPENNN_BENCH_DATA/cifar10"
```

Outputs:

- `docs/benchmarks/results/gpu-WORKLOAD-max-batch-to-target-RUN.json`
- `docs/benchmarks/results/evidence/max-batch-to-target-RUN/*.log`
- `docs/benchmarks/results/evidence/max-batch-to-target-RUN/*-power.csv`
