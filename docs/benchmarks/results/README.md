# Benchmark Result Artifacts

This directory is the evidence store for benchmark runs. Top-level benchmark
notes explain the narrative result; result JSON files keep the machine-readable
run data that supports those notes.

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
    "opennn": "OPENNN_CUDA_GRAPH=1 OPENNN_GPU_RESIDENT_DATA=1 opennn_resnet50_speed",
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
- Keep result files immutable. If a benchmark is rerun, write a new file rather
  than editing an old one.
- Use the benchmark id from `../benchmark_manifest.json`.
- If a run is platform-specific, encode that in the result metadata and keep the
  benchmark note explicit about it.
- Internal lab notes such as `CONTINUE_HERE.md` are not evidence artifacts until
  their commands, metadata, and raw output are captured here.
