# Dense Benchmarks: HIGGS Migration

Dense MLP benchmarks should now use the shared HIGGS dataset contract in
[`higgs/`](higgs/). The old Rosenbrock runs are useful engineering history, but
they should not be used as fresh headline evidence.

Full HIGGS files belong under `OPENNN_BENCH_DATA` outside the repository. See
[`DATA_POLICY.md`](DATA_POLICY.md) before downloading or preparing data.

## Canonical Workload

- Dataset: prepared HIGGS CSV, `feature_0,...,feature_27,label`
- Split: first 10,500,000 train rows and last 500,000 test rows
- Model: `28 -> hidden -> hidden -> 1`
- Default hidden width: `1024`
- Hidden activation: `ReLU`
- Objective: binary cross entropy
- Optimizer: Adam
- Quality metrics: test accuracy, test log loss, and ROC AUC on `higgs_test.csv`
- Required artifacts: raw logs, result JSON, framework versions, CUDA/cuDNN
  versions, `nvidia-smi`, commit hash, dirty status, and `higgs_metadata.json`

## Current Status

| Area | Status | Next action |
|------|--------|-------------|
| Shared data prep | Done | Run `higgs/prepare_higgs.py` on the full UCI file. |
| GPU dense training speed | JSON harness ready with quality metrics and CMake target | Build `opennn_speed`, then rerun OpenNN, PyTorch, and TensorFlow with `training-speed/run_higgs_dense.py`. |
| GPU dense inference speed | Historical Rosenbrock only | Migrate the CPU/GPU inference runners to load `higgs_test.csv` and use the HIGGS classifier. |
| GPU max batch | Historical Rosenbrock only | Rebuild the max-batch probes around the HIGGS classifier shape and BCE training step. |
| GPU energy | Historical Rosenbrock only | Point `run_energy.py` at the HIGGS throughput binaries/scripts before remeasuring. |
| CPU inference | Historical Rosenbrock only | Use `higgs_test.csv`, the same HIGGS classifier, and exported identical weights. |
| Accuracy / quality gate | Added to dense training-speed runners | Set threshold environment variables for publication runs if a hard pass/fail gate is desired. |
| Precision workflow | Historical Rosenbrock only | Reframe as optimizer capability on HIGGS only if the metric is classification quality, not MSE. |
| Data capacity | Historical Rosenbrock only | Decide whether capacity means full HIGGS load or repeated-HIGGS stress data, then document that choice. |

## Safe Wording Before Rerun

"The dense benchmark suite is being standardized on the HIGGS binary
classification dataset. Previous Rosenbrock results are retained as engineering
diagnostics, but new dense performance and quality claims will come from the
HIGGS rerun with archived raw artifacts."
