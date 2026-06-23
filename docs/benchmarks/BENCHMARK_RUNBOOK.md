# Benchmark Runbook

This is the working map for `docs/benchmarks`. Use it to decide what to run,
what to cite, and what to ignore until it is refreshed.

## Source Of Truth

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | Public benchmark index and current headline table. |
| [`benchmark_manifest.json`](benchmark_manifest.json) | Machine-readable inventory: status, runner, metrics, and gaps. |
| [`PRESENTATION_CLAIMS.md`](PRESENTATION_CLAIMS.md) | Claim gate: what can be used now, after rerun, as support only, or not yet. |
| [`DATA_POLICY.md`](DATA_POLICY.md) | Where large datasets live and what must stay out of git. |
| [`results/README.md`](results/README.md) | Required result JSON schema. |
| [`DENSE_HIGGS_MIGRATION.md`](DENSE_HIGGS_MIGRATION.md) | Dense MLP migration status and remaining work. |
| Track READMEs | Current operator entrypoints inside active benchmark folders. |

Internal audit notes such as [`DUE_DILIGENCE_AUDIT_AND_PLAN.md`](DUE_DILIGENCE_AUDIT_AND_PLAN.md)
are useful background, but they are not a source of current commands or claims.

## Benchmark Status

| Track | Status | Canonical runner | Use as headline? |
|------|--------|------------------|------------------|
| Deployment footprint, startup, dependencies, LOC, code export | Current notes, mostly manual evidence | Top-level note files | Yes, with raw commands archived. |
| Transformer inference | Strong but needs fresh provenance-rich rerun | [`attention-speed/run_transformer.py`](attention-speed/run_transformer.py) | After rerun. |
| Dense MLP | Moving to HIGGS | [`training-speed/run_higgs_dense.py`](training-speed/run_higgs_dense.py) | Not until HIGGS result JSON exists. |
| CNN training | Needs optimized competitor paths or clearer scope | [`cnn-training-speed/run_cnn.py`](cnn-training-speed/run_cnn.py) | Hold back. |
| ResNet-50 training | Promising CIFAR result, ImageNet-geometry caveat pending | [`resnet50-training-speed/run_resnet50.py`](resnet50-training-speed/run_resnet50.py) | Hold back until rerun/quality gate. |
| Rosenbrock accuracy, precision, convergence, capacity | Historical controlled diagnostics | `accuracy/`, `precision/`, `convergence/`, `capacity/` | Support only; do not use as active dense evidence. |
| Rosenbrock dense max-batch/energy | Historical stress tests | [`rosenbrock-max-batch/`](rosenbrock-max-batch/) | Hold back; replace with HIGGS. |

## Run Order

1. **Dense HIGGS result JSON**: run
   [`training-speed/run_higgs_dense.py`](training-speed/run_higgs_dense.py) on
   the full train/test split with quality metrics, framework versions, raw
   stdout/stderr, commit, dirty status, and CUDA/cuDNN metadata.
2. **Transformer inference rerun**: use `run_transformer.py` with repeated runs and
   complete provenance.
3. **ResNet/CNN cleanup**: run optimized PyTorch and TensorFlow paths, then decide
   whether the claims survive.
4. **Structural evidence archive**: attach exact commands and versions for size,
   startup, dependencies, LOC, and export.

## File Lifecycle

| Kind | Location | Rule |
|------|----------|------|
| Public note | `*-opennn-vs-*.md` | Must cite current result artifacts or clearly say historical. |
| Track README | `*/README.md` | First file to read inside a benchmark folder; keep commands current here. |
| Runner | `run_*.py`, `run_*.sh`, `opennn_*.cpp` | Must print or write machine-readable metrics. |
| Result artifact | `results/*.json` | Immutable; never edit an old result after rerun. |
| Lab note | `CONTINUE_HERE.md`, `*_CONTINUE.md`, probe scripts | Useful for engineering history only; not evidence. |
| Generated data/build output | `$OPENNN_BENCH_DATA`, benchmark build dirs | Ignored; do not commit. |

## Dense HIGGS Quick Path

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/higgs/prepare_higgs.py \
  --raw /path/to/HIGGS.csv.gz

cmake -S . -B build-benchmarks \
  -DOpenNN_BUILD_BENCHMARKS=ON \
  -DOpenNN_BUILD_EXAMPLES=OFF
cmake --build build-benchmarks --config Release --target opennn_speed

cd docs/benchmarks/training-speed
python run_higgs_dense.py --epochs 5 --batch 7000 --runs 5 --precision bf16
```

For publication artifacts, run:

```bash
python run_higgs_dense.py \
  --epochs 5 --batch 7000 --hidden 1024 \
  --activation relu --hidden-layers 2 \
  --shuffle shuffle --precision bf16 --runs 5
```
