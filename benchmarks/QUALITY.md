# Prediction quality

`run.py --family quality` trains OpenNN C++ and PyTorch Python, then scores their
held-out predictions in one neutral Python scorer. This is a separate experiment
from the published short throughput runs. It does not infer accuracy from shape
agreement, untrained inference, or equal training loss.

| Model | Data | Primary metric |
|---|---|---|
| Dense | HIGGS | test accuracy, percent; higher is better |
| LSTM | Beijing PM2.5 | test RMSE in original target units; lower is better |
| CNN | ImageNet subset, ResNet-50 v1.5 | test top-1 accuracy, percent; higher is better |
| Transformer | WMT14 English–German, base encoder-decoder | generated-translation SacreBLEU; higher is better |

The published Rosenbrock regression result is a different workload. Do not copy
its MSE into the HIGGS row or attribute these new runs to that historical report.

## Prepare

Use a separate Python environment with compatible torch/torchvision packages,
NumPy, pandas, Pillow and SacreBLEU. `manifests/quality-requirements.txt` lists the
versions/constraints. Install the CPU or CUDA wheel pair appropriate for the
machine. Do not add these packages to the application deployment environments.

Prepare the original datasets with the existing `prepare.py` commands, then:

```bash
python benchmarks/prepare.py quality --model all \
  --data-root "$OPENNN_BENCH_DATA" --out "$OPENNN_BENCH_DATA/quality"
```

The destination must be new and outside the checkout. Each model receives a
manifest and shared binary tensors. Input identity is checked before training.

- Dense retains HIGGS's prepared train/test files and training-fitted scaling.
- LSTM splits chronology 80/20 before forming windows, fits scaling on training
  rows only and scores in original target units. The prepared source retains the
  existing `prepare.py` missing-value interpolation; this is not a new raw-data
  cleaning study. The model uses a plain LSTM and head on prepared tensors.
- CNN splits each class 80/20 with a fixed seed, rejects identical image content
  across splits, and performs one shared RGB/bilinear resize. Both models divide
  the same uint8 inputs by 255. All classes remain present; no augmentation is used.
- Transformer deduplicates normalized pairs before a fixed 90/10 split. One
  vocabulary is fitted on training text and used by both engines. PAD/BOS/EOS
  handling, shifted decoder inputs and reference text are shared. Training uses
  causal decoder attention, dropout zero and cross-entropy averaged over non-PAD
  tokens. Scoring uses greedy autoregressive predictions through EOS or the
  fixed length, not teacher-forced token guesses. The SacreBLEU signature is saved.

These splits and preprocessing rules differ from speed-only datasets that were
used entirely for timing. Existing published speed figures are not relabeled as
measurements of this quality protocol.

## Build and run

Enable `OpenNN_BUILD_BENCHMARKS=ON` in an external Release build, then build the
`quality_opennn` target. It supports CPU FP32 and CUDA FP32/BF16. Both engines
receive the same prepared tensors, batch size, epoch budget, learning rate and
seed list. Choose the budget before inspecting test scores.

```bash
python benchmarks/run.py --family quality --model dense \
  --manifest "$OPENNN_BENCH_DATA/quality" --epochs 100 --batch 256 \
  --seeds 42,43,44,45,46 --device cuda --precision fp32 \
  --opennn-binary /path/to/release/bin/quality_opennn
```

Repeat with `--model lstm`, `cnn` and `transformer`, choosing an adequate budget
for each task. `--model all` applies one explicitly supplied budget to all four.
`--python` selects a different PyTorch environment; by default it is the runner's
interpreter. `--threads` applies to both engines. Long runs have a configurable
`--timeout` per process (default 24 hours).
Use `--opennn-library-path` and `--pytorch-library-path` when Linux/WSL needs
different native library search paths for the two drivers. Their effective
values are recorded, so a CUDA 12 OpenNN build need not load the CUDA 13
Python wheel's cuDNN library.

Adam uses beta1 .9, beta2 .999 and float32 epsilon in both engines. The default
learning rate is .001, or .0001 for Transformer. There is no regularization,
gradient clipping, early stopping, best-checkpoint restoration, training warm-up
or CUDA graph replay. The prepared order is fixed and both engines train every
tail sample. Use a batch that avoids singleton CNN training batches because
training batch normalization needs more than one value per channel.

Initial weights are independently seeded, **not identical tensors**. Dense
uses Glorot initialization, LSTM follows each implementation's PyTorch-style
initialization, and CNN/Transformer retain their engine initialization policies.
Backend rounding and mixed-precision policies can also differ. These are
framework training-outcome comparisons, not a claim of identical optimization
trajectories. Review those differences before explaining a score gap.

## Results and smoke checks

Every run writes under `benchmarks/results/scratch/`: raw stdout/stderr,
per-epoch training histories, binary predictions, driver metadata and hashes,
plus an aggregate JSON, CSV and a Markdown table. Failed runs are retained and
make the command return nonzero. The scorer reports means and sample standard
deviations; one seed has no estimated standard deviation. Metrics are compared
within a model, never averaged across different tasks.

The quality gate is deliberately **unassessed**. Similar averages, overlapping
spread or successful execution alone do not establish equivalence. Review
achieved task quality and declare acceptable differences before a parity claim.
Full training runs have not been substituted by synthetic checks.

For a quick end-to-end check of all model paths, including a test tail batch:

```bash
python benchmarks/prepare.py quality --smoke --out "$OPENNN_BENCH_DATA/quality-smoke"
python benchmarks/run.py --family quality --manifest "$OPENNN_BENCH_DATA/quality-smoke" \
  --epochs 1 --batch 2 --seeds 42,43 --device cpu \
  --opennn-binary /path/to/cpu/bin/quality_opennn
python -m unittest discover -s tests -p 'benchmark*_test.py'
```

Smoke fixtures use small topologies and synthetic data. Their numbers are
functional checks and must not appear as benchmark evidence in a presentation.
