# Deployment: what the library costs to adopt

The twelve cells measure a run in progress. This document measures everything
around it — how large the library is, how much you write to use it, what has to
be installed, and what a trained model becomes — because those are the
questions asked before anyone gets to a throughput number, and because they had
no document until now.

That absence had a cost. A slide deck built from these claims carried a
first-prediction time of 36 ms against the 569 ms
[`footprint.md`](footprint.md) measures, a library size roughly half the real
one, and an application-line count no file in the repository supported. None of
the three was reproducible, so none was checkable, so none was caught. Every
figure below is emitted by `tools/deployment_facts.py` into an artifact under
`results/`, carries its counting method in that artifact, and — where this host
cannot measure it — is recorded as unmeasured rather than estimated.

Source facts taken 2026-09-07 at commit `48efcc03f`, plus the two files this
document introduces (`tools/deployment_facts.py` and the PyTorch counterpart
under `application/`), neither of which is counted by any figure above except
the 29. Footprint rows are session `2026-09-06-publish` at `e76425bd3`.

## Results

| question | OpenNN | PyTorch | source |
|---|---|---|---|
| library size | 64,523 lines of C++ | 834,319 lines of C/C++ | `deployment-facts`, and *Caveats* on the PyTorch figure |
| ready-made models | 14 | — | `deployment-facts` |
| layer types | 23 | — | `deployment-facts` |
| complete examples | 16 | — | `deployment-facts` |
| the same program | **13 statements** | **29 statements** | `deployment-facts`, both files in the tree |
| packages to install | 0 | **29**, fifteen of them `nvidia-*` | `deployment-facts`, resolved at torch 2.13.0 |
| weight before any work | 118.3 MiB | 449.4 MiB | [`footprint.md`](footprint.md) |
| time to first prediction | 0.569 s | 1.886 s | [`footprint.md`](footprint.md) |
| what a trained model exports to | 15,104 bytes of C, no runtime — **dense and recurrent networks only** | TorchScript, needs libtorch | [`footprint.md`](footprint.md), and *Caveats* |
| deployed size on disk | not measured | not measured | *What is not measured* |

## What is measured

Two groups, with different validity.

**Source facts** are read from the repository tree, so they are the same on
every machine and can be taken on a laptop. Library size counts every
`.cpp/.h/.cu/.cuh` under `opennn/`, excluding blank lines and whole-line
comments, with block comments tracked across lines. Models are the classes
deriving from `NeuralNetwork` in `models/models.h`; layer types are the headers
in `neural_network/layers/` less the abstract base; examples are the
directories under `examples/` with a `main.cpp`.

**Machine facts** are read from the host and are only valid for the machine
that will deploy: the packages installed in the environment that runs the
PyTorch side, the `NEEDED` list of a built OpenNN binary, and the disk cost of
that binary plus every object `ldd` resolves for it. On a checkout with no
build they are absent, and the artifact says so.

## Why

### The same program, 13 statements against 29

The pair is in the tree: [`../../examples/breast_cancer/main.cpp`](../../examples/breast_cancer/main.cpp)
and [`../application/breast_cancer_pytorch.py`](../application/breast_cancer_pytorch.py).
Both load the same CSV, build a classifier with one hidden layer of three
units, train it, and print the binary-classification report. The PyTorch file
was written to be as short as honesty allows — no argument parsing, no logging,
no configurability, nothing the OpenNN example does not also have — because
padding it would make the number worthless.

The whole difference is two objects. `TrainingStrategy` is the epoch loop and
`TestingAnalysis` is the report, so what is one statement each on the OpenNN
side becomes, on the PyTorch side: the train/test split, the feature scaling,
the loop with its `zero_grad`, `backward` and `step`, and four counted terms of
a confusion matrix. That is the claim, and it is about where the loop lives,
not about the languages.

**Quote `statements`, not lines.** `code_lines` reads 28 against 32, which
flatters neither side honestly: it counts lone braces, `#include` lines and the
`try`/`catch` frame as code, and those are an artefact of C++, not of the work.
The two languages are not comparable on that metric and the artifact records
both so the disagreement stays visible.

**The benchmark drivers are the wrong evidence for this question**, and were
briefly used as such. `families/dense.cpp` runs 604 lines against
`families/dense.py`'s 407, and the model definition inside them is 24 lines
against 15 — the opposite direction. That is because the drivers hand-write the
training loop on both engines so the two are measured doing identical work,
which means they deliberately never touch `TrainingStrategy`. They measure
fairness, not ergonomics.

### Library size

64,523 lines of code across 250 files, 84,407 including comments and blanks.
The number that was previously quoted, 34,926, predates a substantial amount of
the library: YOLO detection with its v8 head and non-max suppression, BERT,
GPT-2, Qwen3 with grouped-query attention, and the autoencoder models are all
in the current tree.

Against PyTorch's native core this is roughly thirteen times smaller, and the
honest reading of that ratio has two parts. OpenNN is smaller partly because it
delegates: MKL, oneDNN, cuBLAS and cuDNN do the arithmetic, and their lines are
in neither count. And it is smaller partly because PyTorch carries generality
OpenNN does not offer — device code for six SM architectures, cuFFT, cuSOLVER,
cuSPARSE and NCCL, and a dispatcher registering every operator it ships. As
`footprint.md` puts it, a reader who needs that breadth is buying the size on
purpose. What the ratio does support is the auditing claim: one engineer can
read all of OpenNN, and cannot read all of PyTorch.

### Coverage

14 ready-made networks — approximation, classification, forecasting, an LSTM
forecaster, autoencoders, image classification, ResNet, YOLO, text
classification, the transformer, text generation, Qwen3, BERT, and BERT for
sequence classification — over 23 layer types, with 16 complete examples.

Five of the fourteen export to standalone source: approximation,
classification, forecasting, LSTM forecasting and the autoencoder are built
from Scaling, Dense, Recurrent, LSTM and Unscaling layers, which is exactly
the set `ModelExpression` accepts. The other nine use convolution, embeddings
or attention and have no standalone-source path. A deck that puts "14
ready-made models" next to "exports to C" implies the export covers all of
them; it covers five, and saying so is what keeps the fourteen credible.

Object detection is native rather than an external repository: `YoloNetwork`
comes with `detection`, `detection_v8`, `non_max_suppression` and `c2psa`
layers and a `yolo_dataset`. Pre-trained weights are a first-class path:
`Qwen3::from_pretrained`, `BertForSequenceClassification::from_pretrained` and
`load_darknet_backbone` are in the library, and the GPT-2 example downloads its
weights from the project's own release.

## Asymmetries and caveats

- **The PyTorch line count is inherited, not measured here.** 834,319 comes
  from an earlier count whose tool is not recorded. Before the thirteen-times
  ratio is published anywhere, both trees need counting with one tool — the
  method this document uses for OpenNN and whatever produced that figure may
  not be counting the same thing.

- **"0 packages" means no package manager, not no dependencies.** OpenNN needs
  MKL, oneDNN and, in a CUDA build, the CUDA libraries present: twelve shared
  objects in the `NEEDED` list of `footprint_opennn`, five of them CUDA. The
  claim it supports is narrower and stronger than a bare zero — there is no
  dependency graph to resolve and no version to satisfy — and stating the
  twelve is what keeps it from being read as "OpenNN depends on nothing".

- **The package count is a resolve, not an inventory.** `pip install
  torch==2.13.0 --dry-run --ignore-installed` brings in 29 packages, fifteen of
  them `nvidia-*`, which corroborates the fifteen `footprint.md` records
  independently. Counting the environment instead would have been wrong and
  wrong in our favour: `benchenv` holds 89 packages because it carries both
  engines plus TensorFlow and the Intel oneAPI runtimes.

- **Export covers dense and recurrent networks, not the whole library.**
  `ModelExpression` accepts Scaling, Dense, Recurrent, LongShortTermMemory,
  Unscaling and Clamping layers and throws on anything else
  (`model_expression.cpp:374`), so a convolutional or attention model has no
  standalone-source path. The claim is real for the tabular and forecasting
  models it covers, and must be stated with that scope or it is false for the
  CNN and transformer cells in the same deck.

- **The first-prediction row means something narrower than it looks**, and
  `footprint.md` says so at length: it is `import torch` against "initialise
  CUDA and run on it", and the PyTorch process never touches `torch.cuda`, so
  it never pays for a context. The 3.3× is real and measured; it is not a
  claim that OpenNN starts 3.3× faster at equal work.

- **The footprint rows are single unrepeated launches**, outside the twelve
  cells and their geomeans. The session's second run agrees within one per
  cent, but they do not carry the same statistical weight and must not be
  pooled into the main table's means.

## What is not measured

Two figures a previous deck carried that this document does not, because
nothing in the suite produces them:

**Deployed size on disk.** Legitimate and easy to want; `tools/deployment_facts.py
--binary build-bench/bin/footprint_opennn` fills it, on the benchmark machine,
counting the binary plus every object `ldd` resolves, deduplicated by real path
so a CUDA install's symlink farm is not counted several times.

**Training-data capacity.** A `capacity` mode exists in the drivers, but no
artifact under `results/` contains a capacity run and no report has ever
interpreted one. It also measures the largest batch that fits before an
out-of-memory fault, which is not the same quantity as "samples trained" — and
`families/dense.cpp` records that the *old* capacity site seeded its network
differently from the other five, so it "had never measured the same initialised
network as the speed and quality ones". Any capacity figure predating that
consolidation should be treated as unmeasured.

## Reproducing

    cd benchmarks
    python3 tools/deployment_facts.py                        # source facts
    python3 tools/deployment_facts.py \
        --binary ../build-bench/bin/footprint_opennn \
        --pip /home/artelnics/benchenv/bin/pip               # and the host facts

The artifact lands in `results/` as `deployment-facts-<run_id>.json` and
carries the `method` string for every count in it.
