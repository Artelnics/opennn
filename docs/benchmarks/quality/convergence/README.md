# Convergence gate: wall-clock time to a fixed held-out quality (MLPerf-style)

The other speed benchmarks time a **fixed number of epochs** and report
throughput. That is gameable: an engine can look "fast" by doing less useful
work per step. This benchmark instead measures the metric that actually matters
— **how long until the model reaches a fixed quality target** — and gates on a
**held-out** metric so a model that overfits the training split cannot pass.

This directly answers the technical review question *"are you fast because you
don't actually learn?"*

## Method

All three engines train the **identical** model on the **identical** data and
stop at the **identical** held-out target:

- **Task:** Rosenbrock 10-input regression (`generate_rosenbrock.py`, the same
  deterministic, pre-normalized split the accuracy benchmark uses).
- **Model:** MLP 10 → 50 → 50 → 1, tanh, Glorot/Xavier init.
- **Optimizer / loss:** Adam (no weight decay, no gradient clipping), MSE,
  batch 64.
- **Gate:** train until the **held-out test MSE** ≤ target, evaluated after each
  epoch (OpenNN: after each 5-epoch chunk). The clock counts **training time
  only** — per-epoch evaluation is excluded.
- **Reported:** median ± stdev wall-clock time-to-target over N seeds, the
  held-out test MSE at the stopping point, and epochs taken. A run that fails to
  reach the target within `--max-epochs` is recorded `reached_goal=0` and
  excluded from the timing median.

Gating on the **held-out** metric (not the training loss) is the crucial design
choice: a training-loss gate lets an engine "pass" before it generalizes. Gating
on the test MSE removes that artifact and makes the time comparison fair.

## Usage

```bash
# build the OpenNN driver:
bash build.sh

python run_convergence.py --target 0.06 --max-epochs 3000 --runs 5
python run_convergence.py --target 0.06 --engines opennn,pytorch
```

Writes `../../results/convergence-gate-rosenbrock-<run_id>.json` with per-engine
median ± stdev time-to-target, held-out test MSE, epochs, convergence count,
versions, commit, and GPU.

## Extending to the classification benchmarks

The same gate applies to CNN (MNIST) and ResNet-50 (CIFAR): replace
"held-out test MSE ≤ target" with **"held-out top-1 accuracy ≥ target"**, using
`TestingAnalysis::calculate_multiple_classification_precision()` on the test
split in C++ and the equivalent eval in PyTorch/TF.
