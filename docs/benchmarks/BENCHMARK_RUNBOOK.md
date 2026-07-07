# Benchmark Runbook

This is the working map for `docs/benchmarks`. Use it to decide what to run,
what to cite, and what to ignore until it is refreshed.

## Source Of Truth

| File | Purpose |
|------|---------|
| [`README.md`](README.md) | Public benchmark index and current headline table. |
| [`TRACKS.md`](TRACKS.md) | Human-readable consolidation by lifecycle: headline, supporting, internal, hold-back, historical. |
| [`benchmark_manifest.json`](benchmark_manifest.json) | Machine-readable inventory: status, runner, metrics, and gaps. |
| [`PRESENTATION_CLAIMS.md`](PRESENTATION_CLAIMS.md) | Claim gate: what can be used now, after rerun, as support only, or not yet. |
| [`DATA_POLICY.md`](DATA_POLICY.md) | Where large datasets live and what must stay out of git. |
| [`results/README.md`](results/README.md) | Required result JSON schema. |
| [`DENSE_HIGGS_MIGRATION.md`](DENSE_HIGGS_MIGRATION.md) | Dense MLP migration status and remaining work. |
| [`higgs-cpu-training-opennn-vs-pytorch-vs-tensorflow.md`](throughput/higgs/higgs-cpu-training-opennn-vs-pytorch-vs-tensorflow.md) | Internal CPU HIGGS dense training subset result. |
| [`higgs-cpu-inference-opennn-vs-pytorch-vs-tensorflow.md`](throughput/higgs/higgs-cpu-inference-opennn-vs-pytorch-vs-tensorflow.md) | Internal CPU HIGGS dense inference subset result. |
| [`tools/validate_benchmarks.py`](tools/validate_benchmarks.py) | Consistency check for manifest coverage, lifecycle labels, docs, and result references. |
| Track READMEs | Current operator entrypoints inside active benchmark folders. |

Internal audit notes such as [`archive/DUE_DILIGENCE_AUDIT_AND_PLAN.md`](archive/DUE_DILIGENCE_AUDIT_AND_PLAN.md)
are useful background, but they are not a source of current commands or claims.

## Lifecycle Labels

Every manifest entry has one lifecycle label:

| Lifecycle | Meaning |
|-----------|---------|
| `headline_candidate` | Can appear in the public headline index after raw commands/artifacts are archived. |
| `supporting_only` | Useful context, but not a primary performance headline. |
| `internal_only` | Useful for migration, development, or presentation drafts; not a public claim. |
| `hold_back` | Do not use externally until the listed rerun or methodology fix is complete. |
| `historical` | Retained as engineering history; superseded for new claims. |

## Benchmark Status

| Track | Status | Canonical runner | Use as headline? |
|------|--------|------------------|------------------|
| Deployment footprint, startup, dependencies, LOC, code export | Current notes, mostly manual evidence | Top-level note files | Yes, with raw commands archived. |
| Transformer inference | Strong but needs fresh provenance-rich rerun | [`attention-speed/run_transformer.py`](attention-speed/run_transformer.py) | After rerun. |
| Dense MLP | Moving to HIGGS; internal CPU subset exists | [`training-speed/run_higgs_dense.py`](training-speed/run_higgs_dense.py) | Hold back until full split rerun with committed result artifacts. |
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
| Legacy tracked artifact | Old binaries, ONNX files, generated CSVs | Historical only; do not add new ones. Replace with result JSON or reproducible commands. |

## Validation

Run this after adding, renaming, or retiring any benchmark:

```bash
cd docs/benchmarks
python tools/validate_benchmarks.py
```

Use `--strict-readmes` if a CI job should fail when a runner folder has no
local README.

## Dense HIGGS Quick Path

```bash
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/throughput/higgs/prepare_higgs.py \
  --raw /path/to/HIGGS.csv.gz

cmake -S . -B build-benchmarks \
  -DOpenNN_BUILD_EXAMPLES=OFF \
  -DOpenNN_BUILD_BENCHMARKS=ON
cmake --build build-benchmarks --config Release --target opennn_speed opennn_higgs_cpu

cd docs/benchmarks/throughput/training-speed
python run_higgs_dense.py --epochs 5 --batch 7000 --runs 5 --precision bf16
```

For publication artifacts, run:

```bash
python run_higgs_dense.py \
  --epochs 5 --batch 7000 --hidden 1024 \
  --activation relu --hidden-layers 2 \
  --shuffle shuffle --precision bf16 --runs 5
```
