# Benchmark Result Artifacts

This directory is the evidence store for benchmark runs. Top-level benchmark
notes explain the narrative result; result JSON files keep the machine-readable
run data that supports those notes.

## Protocol

The suite follows an **MLPerf-inspired** protocol, but these are not official
MLPerf results unless they are submitted through MLCommons. The borrowed rules
are simple: fixed workload, explicit quality/correctness rule, documented timed
region, repeated runs where practical, machine metadata, raw logs, and immutable
JSON artifacts.

Use these benchmark classes in new artifacts:

- `training_time_to_quality`: preferred for training; wall time to a declared
  quality target.
- `training_throughput_with_quality`: fixed-epoch or fixed-step training with a
  reported quality metric; useful for engineering, not final headline training
  evidence until a quality target exists.
- `training_capacity`: largest batch that completes forward, backward, and the
  optimizer update under a stated physical VRAM or RAM cap. The quality rule
  should at least gate finite loss after the training step.
- `inference_offline`: batch throughput inference.
- `inference_single_stream`: latency-oriented inference.
- `footprint_or_packaging`: size, startup, dependencies, source LOC, or export.

References: MLCommons MLPerf Training
<https://mlcommons.org/benchmarks/training/>, MLPerf Inference Datacenter
<https://mlcommons.org/benchmarks/inference-datacenter/>, Training rules
<https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc>,
and Inference rules
<https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc>.

## Naming

Use this pattern for generated result files:

```text
<benchmark-id>-<dataset-or-config>-<YYYYMMDDTHHMMSSZ>.json
```

Example:

```text
resnet50-training-speed-cifar10-20260614T211500Z.json
```

## Required Fields

Each result JSON should contain:

```json
{
  "schema_version": 1,
  "benchmark_id": "gpu-resnet50-training",
  "run_id": "20260614T211500Z",
  "git_commit": "abcdef123456",
  "protocol": {
    "style": "mlperf_inspired",
    "official_mlperf": false,
    "benchmark_class": "training_throughput_with_quality",
    "division": "closed",
    "quality_rule": {
      "metric": "final_loss",
      "target": null,
      "status": "reported_not_gated"
    },
    "measurement_rule": {
      "warmup": "documented per benchmark",
      "runs": 5,
      "aggregation": "median"
    }
  },
  "dataset": "cifar10",
  "configuration": {
    "epochs": 5,
    "batch": 128,
    "precision": "fp32"
  },
  "machine": {
    "platform": "Linux-6.x-x86_64",
    "python": "3.12.x",
    "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU, 555.xx"
  },
  "metrics": {
    "samples_per_sec": {
      "opennn_cuda_graph": 8433,
      "pytorch_compile": 5268,
      "pytorch_eager": 3960
    },
    "speedup": {
      "opennn_vs_pytorch_compile": 1.6,
      "opennn_vs_pytorch_eager": 2.1
    }
  },
  "commands": {
    "opennn": "opennn_resnet50_speed (CUDA graph + GPU-resident data enabled in code)",
    "pytorch_compile": "python pt_compile_probe.py",
    "pytorch_eager": "python pytorch_resnet50_speed.py"
  },
  "raw_output": {
    "opennn": "...",
    "pytorch_compile": "...",
    "pytorch_eager": "..."
  }
}
```

## Rules

- Store the git commit, command line, framework versions, and raw output for
  every published headline number.
- Store the `protocol` object for every new result. Use
  `style=mlperf_inspired` and `official_mlperf=false` unless the run has gone
  through MLCommons submission/review.
- Keep result files immutable. If a benchmark is rerun, write a new file rather
  than editing an old one.
- Use the benchmark id from `../benchmark_manifest.json`.
- Prefer training time-to-quality over fixed-epoch training speed. Fixed-epoch
  speed artifacts are allowed as engineering evidence, but the note must say
  they are not final MLPerf-style headline results until a quality target is set.
- If a run is platform-specific, encode that in the result metadata and keep the
  benchmark folder README explicit about it.
