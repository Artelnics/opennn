# OpenNN benchmarks

OpenNN against PyTorch, on four model families, measuring throughput, peak
memory and energy **from the same execution**.

There are no numbers in this file. Results live in [`results/`](results/), each
one naming the commit, machine and session it came from, and a claim is only
worth making with an artifact behind it. The measurement rules are in
[`PROTOCOL.md`](PROTOCOL.md); changing any of them means re-running everything.

## Running one

```bash
python prepare.py dense                       # once per family
python run.py --family dense --mode train --batch 8192
```

That prints throughput, peak memory and energy for each engine, and writes an
artifact. `--family` is `dense`, `cnn`, `transformer` or `lstm`; `--mode` is
`train` or `infer`.

`--batch` is the only sweep axis:

| | |
|---|---|
| `--batch 8192` | one rung: the speed cell |
| `--batch 1024,8192,65536` | several rungs: the throughput curve |
| `--batch 1024:OOM` | double until a launch fails: the capacity frontier |

## The families

| family | model | data | why this one |
|---|---|---|---|
| `dense` | 28 → 1024 × 2 → 1 classifier | HIGGS | the shape a tabular workload actually has |
| `cnn` | ResNet-50 v1.5 | ImageNet subset, 1000 classes × 50 | the citable convolution benchmark |
| `transformer` | d512 · h8 · ff2048 · 6L | WMT14 English-German | the *Attention Is All You Need* base model, on its own corpus |
| `lstm` | LSTM(14→128) → Linear | Beijing PM2.5, hourly | both engines reach the same cuDNN kernel here |

Each family is two files in [`families/`](families/) — one definition per
engine, four modes each. Adding a family means adding those two files and one
entry in `run.py`; there is no per-metric copy to keep in step, which is what
the previous suite spent 201 files doing.

## What every run checks before it reports

A throughput number means nothing if the two engines were not doing the same
work, so two gates run first and the artifact records both.

**The shape gate** compares what each engine reports about the work: sample
count, sequence length, vocabulary, and **parameter count**. It exists because
it kept finding real problems — OpenNN's tokeniser produced 158-position
sequences against PyTorch's 128, and `nn.Transformer` carried 2,048 parameters
of final `LayerNorm` that OpenNN's had no counterpart for. Neither is visible
in a samples-per-second figure.

**The quality gate** compares accuracy across engines at each batch. A speed
win bought by computing something different is not a speed win.

## Reading a result

Energy is reported only when the timed window was long enough to sample — a
short run says so rather than reporting `0.0000 Wh`, which is a claim and not a
measurement. Peak memory is whole-device, minus the idle reading;
`torch.cuda.max_memory_allocated()` never appears, because it excludes the CUDA
context and cached blocks and so flatters PyTorch by construction.

A dirty tree writes to `results/scratch/`, never to the evidence store. That is
enforced in code.

## Files

| | |
|---|---|
| [`run.py`](run.py) | the only entry point |
| [`prepare.py`](prepare.py) | every dataset, one subcommand per family |
| [`common.py`](common.py) | provenance, binaries, GPU sampling, metrics |
| [`families/`](families/) | one definition per engine per family |
| [`PROTOCOL.md`](PROTOCOL.md) | the contract |
| [`PLAN.md`](PLAN.md) | how the suite got this shape, and what is left |
| [`DATA_POLICY.md`](DATA_POLICY.md) | where datasets live; what stays out of git |
| [`gpu_clocks.sh`](gpu_clocks.sh) | lock the clock before measuring |
| [`footprint/`](footprint/) | not yet folded into a family — see PLAN.md |

Datasets never enter the repository. The one committed artefact is
`imagenet_subset.manifest`, which pins exactly which images the CNN family
measures on, so a subset can be verified rather than trusted.
