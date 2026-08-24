# Duplication ledger

Step 1 of [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md). Before deleting any
duplicate, record what is byte-identical, what silently diverges, and what is
genuinely distinct driver logic — so that when a number moves after the merge,
this says why.

Everything below was read out of the tree, not estimated. Line references are as
of the commit that adds this file.

Status: **complete** -- all four families, plus the two decisions they raised.

---

## What the four families say together

**There is no suite-wide convention to restore.** 42 is the majority seed, but
the exceptions do not form a pattern: dense seeds 0 at its capacity site,
transformer seeds 0 at three of four, and CNN seeds 42 everywhere including
capacity. "Capacity benchmarks use 0" looks true from two families and is
refuted by the third. Each family is separately inconsistent, so the merge picks
a value rather than discovers one.

**The patterns worth adopting already exist in the tree.** None of them needs
inventing:

| pattern | where it already works |
|---|---|
| a driver reusing definitions instead of copying them | `run_resnet50_energy.py` invokes the throughput drivers |
| configuration in one shared module | `xf_common.py` holds the scenarios for both Python engines |
| aggregating over several seeds | recurrent averages seeds 0-4 in all three engines |
| execution mode as one flag, not a second file | `PT_FAST` switches channels_last + compile + TF32 together |

The merge should propagate these four rather than design something new.

**Loss formulation is the recurring fairness axis.** It appears in three of the
four families and in a different form each time: dense site 5 has PyTorch on
logits while the other two take the sigmoid in the forward; transformer training
has OpenNN materialising a softmax the other two fuse into the loss; dense site
1 makes the logits choice deliberately and applies it to all three engines,
which is the only place it is handled consistently. This needs one decision
applied everywhere, not four local ones.

**Initialisation is unreconciled in every family, and differently each time.**
Dense: OpenNN and TensorFlow on Glorot, PyTorch on Kaiming. Transformer: bodies
agree on Xavier, embeddings and output projections do not. CNN: three different
initialisers, and OpenNN's is not fan-scaled at all. There is no single
statement that covers the suite -- each family has to be settled on its own
evidence.

**Two findings point outside the benchmarks.** The CNN initialisation
(`ResNet` and `YoloNetwork` bypassing the Glorot path) is a library-level
question about shipped builders. The recurrent scenario table, duplicated across
the C++/Python boundary because no module can span it, is the concrete argument
for section 7's `suite.json`.

---

## Dense (HIGGS)

18 definitions: 6 sites x 3 frameworks. Target 3.

| # | Site | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|---|
| 1 | `capacity/higgs-max-batch` | `opennn_higgs_maxbatch_trial.cpp` | `pytorch_higgs_maxbatch.py` | `tensorflow_higgs_maxbatch.py` |
| 2 | `quality/accuracy` | `opennn_accuracy.cpp` | `pytorch_accuracy.py` | `tensorflow_accuracy.py` |
| 3 | `quality/convergence` | `opennn_convergence.cpp` | `pytorch_convergence.py` | `tensorflow_convergence.py` |
| 4 | `throughput/higgs` (CPU) | `opennn_higgs_cpu.cpp` | `pytorch_higgs_cpu.py` | `tensorflow_higgs_cpu.py` |
| 5 | `throughput/higgs-gpu` (train) | `opennn_speed.cpp` | `pytorch_speed.py` | `tensorflow_speed.py` |
| 6 | `throughput/higgs-gpu` (infer) | `opennn_higgs_infer.cpp` | `pytorch_higgs_infer.py` | `tensorflow_higgs_infer.py` |

All eighteen claim the same model in their header comment: 28 -> 1024 -> 1024 ->
1, ReLU hidden, sigmoid output, binary cross-entropy, Adam. Six of them do
something else.

### What silently diverges

**1. The seed splits 0 / 42, and the split is not per-framework — it is per
site.** Site 1 seeds with 0 in all three engines; sites 2-6 seed with 42.

| | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|
| site 1 | `OPENNN_BENCH_SEED` else **0** | `--seed` default **0** | `--seed` default **0** |
| sites 2-6 | `set_seed(42)` hardcoded | `manual_seed(42)` | `set_seed(42)` |

Site 1 is internally consistent, so its three engines *are* comparable with each
other. What it is not comparable with is every other dense number in the suite:
the capacity benchmark has never measured the same initialised network as the
speed and quality ones. Site 1 is also the only site whose seed is reachable
without editing code.

**2. Site 5 is not running one model — it is running two.** In the headline GPU
training benchmark:

| | output layer | loss |
|---|---|---|
| `opennn_speed.cpp:143,150` | `Dense(..., "Sigmoid")` | `set_loss("CrossEntropy")` |
| `tensorflow_speed.py:104,109` | `Dense(1, activation="sigmoid")` | `BinaryCrossentropy()` (`from_logits` defaults false) |
| `pytorch_speed.py:105,109` | `Linear(current, 1)` — **no activation** | **`BCEWithLogitsLoss()`** |

PyTorch emits logits and fuses the sigmoid into the loss; the other two emit a
probability and take the sigmoid in the forward. The two formulations agree
mathematically on the loss value but are not the same graph: PyTorch runs one
fewer elementwise op in the forward and a different, more stable backward.

On FLOPs this is negligible — a sigmoid over `batch x 1` against two
`batch x 1024` GEMMs — so it is unlikely to move throughput much. It matters
because `tensorflow_speed.py` also scores accuracy / log-loss / ROC-AUC from
the same run (its `--min-accuracy` / `--max-log-loss` / `--min-auc` gates), and
those numbers come from a different formulation than PyTorch's.

Site 1 is mixed the other way. Its PyTorch and TensorFlow drivers both use
logits — the header calls that "the standard formulation" — while
`opennn_higgs_maxbatch_trial.cpp:162` builds a `"Sigmoid"` output like every
other OpenNN dense definition. So two of three on logits at site 1, one of three
at site 5, and three of three on sigmoid at sites 2, 3, 4 and 6:

| site | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|
| 1 capacity | sigmoid | **logits** | **logits** |
| 2 accuracy | sigmoid | sigmoid | sigmoid |
| 3 convergence | sigmoid | sigmoid | sigmoid |
| 4 CPU | sigmoid | sigmoid | sigmoid |
| 5 GPU train | sigmoid | **logits** | sigmoid |
| 6 infer | sigmoid | sigmoid | — |

Read down the OpenNN column and the pattern is not author whim: **OpenNN takes
the activation in the forward everywhere, in this family and in the transformer
one, because it has no from-logits loss to take.** `Loss::Error` offers
`CrossEntropy` and `CrossEntropy3d`, and both consume probabilities —
`error_functions.cpp:254` and `:414` read `log(output)` directly. PyTorch and
TensorFlow diverge from it wherever an author reached for the formulation their
framework makes natural.

The divergence therefore tracks a library capability gap rather than a
convention that was never agreed, which changes what fixing it means. See
"Decisions" below.

**3. Weight initialisation was never reconciled.** No PyTorch dense file sets an
initialiser, so each engine uses its own default:

| | initialiser |
|---|---|
| OpenNN | Glorot — `set_parameters_glorot()` at all six sites, directly at five and via `finalize_build` at site 4 |
| TensorFlow | Glorot — `keras.layers.Dense` defaults to `glorot_uniform` |
| PyTorch | **Kaiming uniform** (`a=sqrt(5)`) — the `nn.Linear` default |

Two engines start from Glorot and one does not. For throughput this is
irrelevant. For sites 2 and 3 — fixed-epoch accuracy and wall-clock time to
reach a quality target — the starting distribution is an input to the number
being published.

(The only explicit initialiser anywhere in the dense-family tree is
`pytorch_precision.py:45`, and that belongs to the Rosenbrock precision
benchmark, not this family.)

**4. Site 4 has an extra layer.** `opennn_higgs_cpu.cpp:74` builds through
`ClassificationNetwork`, which prepends a `Scaling` layer
(`standard_networks.cpp:165`). The other five OpenNN definitions add bare
`Dense` layers and have none.

The scaler is neutralised — `opennn_higgs_cpu.cpp:133` sets it to `"None"`,
matching `dataset.set_variable_scalers("None")` at line 62, because
`prepare_higgs.py` already normalises the CSV. So the arithmetic is right. But
the layer is still in the network, on the one benchmark that compares OpenNN's
CPU throughput against PyTorch and TensorFlow, neither of which has anything
equivalent. Whether an identity `Scaling` pass costs a copy or is elided is not
established here; it needs measuring before site 4 is merged, not assuming.

### What is byte-identical

Sites 2 and 3 are the same definition with different drivers. Their
model-construction blocks are character-for-character identical per framework —
layer stack, `BCELoss`, `Adam()` with default learning rate — and diverge only
where the protocol legitimately does: site 2 runs fixed epochs, site 3 stops on
a held-out target and reports epochs taken.

Whole-file diff is a poor measure of this (90 of ~140 lines differ for PyTorch,
84 of ~140 for TensorFlow) because the driver is inlined next to the definition
in the same file. Separating the two is the whole point of the merge.

### What is genuinely distinct

Not duplication; do not merge these away.

- **Site 1's execution mode.** One training step at an increasing batch until
  OOM, with fused Adam and `cudnn.benchmark`. Its `--seed` and `--target` flags
  exist because the driver sweeps.
- **Site 6 is inference-only.** No loss, no optimiser. `pytorch_higgs_infer.py`
  additionally times more than one dispatch path per batch rung and reports the
  faster in `pt_path` — a deliberate fairness measure, since PyTorch's best path
  for this network changes with batch size.
- **Site 4's CPU-specific choices.** Eager is PyTorch's fast path here and the
  file says why: `torch.compile` measured 29,449 samples/s against eager's
  41,523, inductor losing to eager on a three-GEMM MLP.
- **Per-engine tuning knobs** — `PT_COMPILE_MODE`, `PT_FUSED_ADAM`,
  `OPENNN_BENCH_SCALERS`, XLA `jit_compile`. These are how each engine is given
  its best shot and belong in the surviving definition as flags.

### What the three survivors must pin

Every field below currently varies across the eighteen. The merged definition
has to state each one explicitly rather than inherit a framework default:

seed; output-layer activation and the matching loss formulation; weight
initialiser; presence of a scaling layer; Adam learning rate (site 1 pins
`1e-3`, the rest take the framework default — which is also `1e-3`, so this one
is equivalent today and fragile tomorrow).

Two decisions have to be made before the merge, because the eighteen do not
agree and neither answer is free:

1. **Logits or sigmoid?** Site 1 already chose logits for all three engines.
   Adopting that everywhere makes site 5 like-for-like and matches what a
   practitioner would write, but it changes the quality numbers at sites 2-4,
   which are currently sigmoid + BCE in all three engines.
2. **Whose initialiser?** Glorot is the majority and is what two of the three
   frameworks default to, so pinning Glorot in PyTorch is the smaller change.
   It will move the published accuracy and convergence numbers.

Both are re-baseline material (plan step 6), not silent fixes.

---

## Transformer

15 definitions: 4 sites x 3 frameworks, plus 3 OpenNN-only variants. Target 3.

| # | Site | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|---|
| 1 | `throughput/attention-speed` (train) | `opennn_transformer_train.cpp` | `pytorch_transformer_train.py` | `tensorflow_transformer_train.py` |
| 2 | `throughput/attention-speed` (infer) | `opennn_transformer_infer.cpp` | `pytorch_transformer_infer.py` | `tensorflow_transformer_infer.py` |
| 3 | `capacity/transformer-max-batch` | `opennn_transformer_maxbatch_trial.cpp` | `pytorch_transformer_maxbatch.py` | `tensorflow_transformer_maxbatch.py` |
| 4 | `energy/transformer-energy` | `opennn_transformer_energy.cpp` | `pytorch_transformer_energy.py` | `tensorflow_transformer_energy.py` |

OpenNN-only, not duplicates — see "genuinely distinct" below:
`opennn_transformer_resident.cpp`, `opennn_qwen3_resident.cpp`,
`opennn_attention_validate.cpp`.

### What silently diverges

**1. The training benchmark is a different model from the other three sites.**
Not a different seed or a different loss — a different network.

| | d_model | heads | ff | layers | source |
|---|---|---|---|---|---|
| site 1 (train) | **256** | 8 | **1024** | **2** | `run_transformer_train.py:122-125` |
| site 2 (infer) | 512 | 8 | 2048 | 6 | `pytorch_transformer_infer.py:22-25`, `opennn_transformer_infer.cpp:33-36` |
| site 3 (capacity) | 512 | 8 | 2048 | 6 | `run_transformer_maxbatch.py:39` |
| site 4 (energy) | 512 | 8 | 2048 | 6 | `run_transformer_energy.py:49` |

Each encoder/decoder layer is about four times the parameters at d=512/ff=2048
than at d=256/ff=1024, and there are three times as many of them — so the
training benchmark exercises roughly an order of magnitude less model than the
inference, capacity and energy ones. All three engines agree with each other at
each site, so every individual number is internally fair. What is not true is
the sentence "the transformer benchmark": there are two transformers, and which
one a reader gets depends on which metric they are looking at.

**2. Three sites, three datasets** — as the plan says, now confirmed at the
file level: site 1-2 a synthetic LCG corpus (`make_synthetic_corpus.py`), site 3
WMT14 (`prepare_wmt14.py`), site 4 chat pairs (`prepare_chat.py`). Vocabulary
and sequence length follow the corpus, so they differ too: site 1 defaults to
vocab 256 / seq 256, site 2 to vocab 10,000 / seq 64.

**3. The seed split is inverted from the dense family.** There, one site used 0
and five used 42. Here three sites use 0 and the odd one out uses 42:

| | seed |
|---|---|
| sites 1-3 | `0`, hardcoded in all three engines |
| site 4 (energy) | `42` — `--seed` default in PyTorch/TF, `argv[10]` default in `opennn_transformer_energy.cpp:41` |

So there is no suite-wide convention to appeal to: dense and transformer
disagree about which value is the default, and both disagree internally.

**4. Training fuses the loss in two engines and not in the third.**

| | output projection | loss |
|---|---|---|
| OpenNN | `Dense(..., "Softmax")` (`standard_networks.cpp:1465-1466`) | `CrossEntropyError3d` |
| PyTorch | `nn.Linear(d_model, vocab)` — logits | `nn.CrossEntropyLoss()` |
| TensorFlow | `Dense(vocab)` — logits | `SparseCategoricalCrossentropy(from_logits=True)` |

The mathematics agree. The execution does not: OpenNN materialises a
`[batch, seq, vocab]` probability tensor in the forward and
`error_functions.cpp:414` then reads `log(p + EPSILON)` at the target index,
while PyTorch and TensorFlow hand logits to a fused loss that computes
log-softmax without ever writing the probability tensor.

The extra cost is memory traffic over the largest tensor in the model, and it
falls on OpenNN — this is the library handicapping itself in its own benchmark,
not the other way round. It is also the numerically weaker formulation:
`log(softmax(x))` guarded by an epsilon rather than `log_softmax(x)`.

Worth measuring before site 1 is merged: at WMT14 vocabulary this tensor is
large enough that the fusion difference may be a visible share of the step.

**5. Only site 4 pins `padding_idx`.** `pytorch_transformer_energy.py:87-88`
builds its embeddings with `padding_idx=0`; sites 1 and 3 do not
(`pytorch_transformer_train.py:82-83`, `pytorch_transformer_maxbatch.py:74-75`).
With it, token 0's embedding is held at zero and excluded from the gradient;
without it, it trains like any other token. On a padded corpus that changes both
the arithmetic and the number of live parameters.

**6. Initialisation agrees in the body and not at the edges** — the opposite of
the dense family, and worth stating because the dense finding does not carry
over. All three PyTorch sites build the body with `nn.Transformer`, whose
`_reset_parameters()` applies `xavier_uniform_` to every parameter of dim > 1,
so the encoder/decoder stack matches OpenNN's Glorot (`finalize_build` ->
`set_parameters_glorot`). The embeddings and the output projection do not:
`nn.Embedding` defaults to `N(0, 1)` and the output `nn.Linear` to Kaiming
uniform, against Glorot on the OpenNN side.

### What is verified equal

Checked because it is the obvious place to expect the divergence found at
training, and it is not there: **inference applies softmax in all three
engines** — `pytorch_transformer_infer.py:65` calls `torch.softmax` explicitly,
`tensorflow_transformer_infer.py:85` uses `Dense(vocab, activation="softmax")`,
and OpenNN's `Transformer` ends in a Softmax `Dense`. Site 2 is like-for-like on
this axis.

### What is genuinely distinct

- **`opennn_transformer_resident.cpp`** — the GPU-resident inference path, where
  tokens and parameters both stay on the device. It is the counterpart to
  PyTorch's inference loop rather than a second copy of site 2.
- **`opennn_qwen3_resident.cpp`** — the same resident path for a decoder-only
  model, deliberately mirroring the file above so the two are comparable.
- **`opennn_attention_validate.cpp`** — a correctness check, not a benchmark. Its
  header records why it exists: the library's MHA unit tests only cover
  construction, so nothing else exercises the forward/backward on GPU.

### What the three survivors must pin

Which of the two transformers is *the* transformer — d=256/ff=1024/2 layers or
d=512/ff=2048/6 layers — and then the same list as dense: seed; the loss
formulation and whether the softmax is in the forward or fused into the loss;
embedding and output-projection initialisers; `padding_idx`; and the dataset.

The dataset question is the one that cannot be deferred. Section 4 of the plan
already decides it — settle on WMT14, retire `chat_pairs` — but that leaves the
synthetic corpus carrying sites 1 and 2, which are the throughput numbers most
often quoted. Moving those to WMT14 changes vocabulary from 256 or 10,000 to
WMT14 scale, which changes the output projection, which is exactly where finding
4 says the engines already differ. These two decisions have to be made together.

## CNN (ResNet-50)

10 model definitions plus 2 diagnostic probes. Target 3.

| # | Site | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|---|
| 1 | `throughput/resnet50` (train) | `opennn_resnet50_speed.cpp` | `pytorch_resnet50_speed.py` | `tensorflow_resnet50_speed.py` |
| 2 | `throughput/resnet50` (infer) | `opennn_resnet50_infer.cpp` | `pytorch_resnet50_infer.py` | `tensorflow_resnet50_infer.py` |
| 3 | `capacity/resnet50-max-batch` | `opennn_resnet50_maxbatch_trial.cpp` | `pytorch_resnet50_maxbatch.py` | `tensorflow_resnet50_maxbatch.py` |
| 4 | `energy/resnet50-energy` | — reuses site 1 — | — reuses site 1 — | — reuses site 1 — |
| — | ImageNet variant | reuses site 1 | `pytorch_resnet50_lazy.py` | **missing** |

`cudnn_fusion_probe.cpp` and `pooling_probe.cpp` are diagnostics, not benchmarks.

**This is the family the other three should look like.** Recording what it gets
right matters as much as what it does not, because the merge should not
regress it.

### What is already right

**The energy site has no definitions of its own.** `run_resnet50_energy.py`
invokes `opennn_resnet50_speed`, `pytorch_resnet50_speed.py` and
`tensorflow_resnet50_speed.py` (lines 63-64, 108, 116) rather than carrying
copies. That is exactly the target architecture from section 8 of the plan —
one definition, several drivers — already working in the tree. Dense and
transformer both have a separate energy or capacity definition; this one does
not.

**The seed is 42 everywhere**, in all three engines and at every site, including
capacity:

| | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|
| sites 1-2 | `set_seed(42)` | `manual_seed(42)` | `tf.random.set_seed(42)` |
| site 3 | `OPENNN_BENCH_SEED` else **42** | `--seed` default **42** | `--seed` default **42** |

This kills a tempting generalisation. The dense capacity site seeds 0 and the
transformer capacity site seeds 0, which looks like "capacity benchmarks use 0"
— it is not a convention, because the ResNet capacity site seeds 42 like
everything around it. Dense and transformer are simply each wrong in their own
way, and 42 is the majority across the suite.

**All three engines build the same network and say so.** Each header names
ResNet-50 v1.5 with the stride on the 3x3 convolution, torchvision's
convention: OpenNN through `opennn::ResNet` with bottleneck counts [3,4,6,3],
PyTorch and TensorFlow written out explicitly rather than pulled from
`torchvision.models` or `keras.applications` so that the layer set is visible
and cannot drift with a library version.

**The geometry caveat is documented and mitigated.** ResNet-50 at CIFAR's 32x32
is not ResNet-50's intended shape — the stem reduces to 8x8 before the first
stage. The README knows: it lists "keep the CIFAR-vs-ImageNet geometry caveat
explicit" as a goal and ships `prepare_imagenet_like.py`, a 224x224 dataset
built from CIFAR content, to check whether a conclusion survives the real
geometry. Recorded here as handled, not as a defect.

### What silently diverges

**1. `pytorch_resnet50_lazy.py` is a second copy of the model.** Its
`Bottleneck` class is byte-identical to `pytorch_resnet50_speed.py`'s. Its
`ResNet` class differs in exactly one thing that matters — `classes=1000`
against `classes=10`, which is correct for ImageNet — and two that do not: the
stride expression is inlined into the `Bottleneck` call, and the final flatten
is inlined into the `return`.

The cosmetic drift is harmless in itself and is the point: the two copies have
already diverged in formatting without anyone intending it, which is how a real
change reaches one and not the other. This is the family's one genuine
duplicate.

**2. Every engine initialises this network differently, and OpenNN's is not
fan-scaled.** Neither hand-written model sets an initialiser, so each takes its
framework default, and OpenNN's builder makes a third choice:

| | initialiser |
|---|---|
| PyTorch | Kaiming uniform, fan-in scaled -- the `nn.Conv2d` default |
| TensorFlow | `glorot_uniform`, fan-avg scaled -- the `Conv2D` default |
| OpenNN | `set_parameters_random()` -> `set_random_uniform()` -> **uniform(-0.1, 0.1)**, independent of fan-in |

`ResNet::ResNet` (`standard_networks.cpp:477-478`) calls `compile()` and
`set_parameters_random()` directly instead of `finalize_build`, which is what
gives every other builder in that file its Glorot init. `ConvolutionOperator`
has a `set_parameters_glorot` that computes a proper fan-scaled limit
(`convolution_operator.cpp:314-321`); the ResNet builder does not reach it.

The gap is not small at depth. For a 3x3 convolution with 512 input channels
Glorot's limit is about 0.026, so a flat +-0.1 is roughly four times too wide,
and ResNet-50 stacks fifty of these. Throughput does not care. Any accuracy or
convergence claim does, and so does every library user who builds a `ResNet` --
this is the shipped builder, not a benchmark-local choice.

Worth confirming against Neural Designer and the library's own ResNet tests
whether this is deliberate. `YoloNetwork` makes the same call
(`standard_networks.cpp:589`), so if it is an oversight it is not confined to
one builder.

**3. The TensorFlow max-batch seeds more than the others do.**
`tensorflow_resnet50_maxbatch.py:85` calls `tf.keras.utils.set_random_seed`,
which seeds Python's `random`, NumPy and TensorFlow; sites 1 and 2 call
`tf.random.set_seed`, which seeds only TensorFlow. Where NumPy draws the data
order or the initial arrays, sites 1 and 2 are not reproducible from the seed
alone. Minor, but it is a difference in what "seed 42" means between sites.

### The ImageNet gap, stated precisely

The plan's section 2 heads this "CNN has no TensorFlow at all" and its step 4
asks to "write the missing TensorFlow ResNet-50". Both overstate the work.

`tensorflow_resnet50_speed.py` and `tensorflow_resnet50_infer.py` exist, build
the same v1.5 network as the other two engines, and two of the three runners
default to all three engines:

| runner | `--engines` default | dataset |
|---|---|---|
| `run_resnet50.py` | `opennn,pytorch,tensorflow` | cifar10 |
| `run_resnet50_infer.py` | `opennn,pytorch,tensorflow` | cifar10 |
| `run_imagenet_resnet50.py` | `opennn,pytorch_fast,pytorch_eager` | imagenet |

What TensorFlow is missing is not the model but a **streaming data path**.
`tensorflow_resnet50_speed.py:67-68` loads the whole set with `np.load` into
memory, which is fine for CIFAR and cannot hold decoded 224x224 ImageNet. Both
other engines solved this and neither did it in the model: PyTorch added
`pytorch_resnet50_lazy.py` with a `Dataset`/`DataLoader`, and OpenNN host-stages
instead of going resident (`opennn_resnet50_speed.cpp:12-13` — "the 224px
ImageNet path is too large and stays host-staged").

So the task is a `tf.data` input pipeline for the existing TensorFlow driver,
not a new definition. Worth correcting before step 4 estimates the work.

### What is genuinely distinct

- **The two probes.** `cudnn_fusion_probe.cpp` and `pooling_probe.cpp` answer
  "which fusion engines does this cuDNN offer for these shapes" and are
  diagnostics, not comparisons. The engineering audit's note applies to the
  first: its 0-of-9 result was measured on sm_86 and should be re-run before
  anything is concluded on other hardware.
- **`PT_FAST`.** One env flag switches channels_last + `torch.compile` + TF32
  together, so the eager and optimised paths are one file rather than two.
  This is the pattern the plan wants for execution mode.
- **`image_size` on the OpenNN driver.** Selects CIFAR geometry, ImageNet
  geometry, and whether the data goes device-resident — a flag, not a fork.

### What the three survivors must pin

Less than the other families, because most of it already agrees: the input
geometry and class count (32x32/10 against 224x224/1000, currently the
difference between the two PyTorch copies), and which seeding call is meant by
"seed 42".

The merge here is small and mostly mechanical: fold `pytorch_resnet50_lazy.py`
back into `pytorch_resnet50_speed.py` behind the existing `image_size` argument
so the model exists once, and give the TensorFlow driver the input pipeline
that lets it join the ImageNet runner.

## Recurrent (Beijing PM2.5 forecasting)

3 definitions, one site. Already flat — nothing to merge. Target 3, met.

| Site | OpenNN | PyTorch | TensorFlow |
|---|---|---|---|
| `quality/recurrent-lstm-forecasting` | `recurrent_lstm_forecasting_benchmark.cpp` | `pytorch_forecasting.py` | `tensorflow_forecasting.py` |

Shared: `xf_common.py` (scenario table, data loading, env controls), consumed by
both Python drivers and by `run_forecasting.py`.

### What is already right

**This family is the most rigorously measured in the suite, and by some
distance.** Two things nothing else does:

**It averages over five seeds.** Every other family in this ledger reports a
single run at a single seed. Here all three engines loop seeds 0-4 and
aggregate — `pytorch_forecasting.py:123` and `tensorflow_forecasting.py:52`
both compute `SEEDS = list(range(min(env_ints("OPENNN_FORECASTING_SEEDS",
[5])[0], 5)))`, and `recurrent_lstm_forecasting_benchmark.cpp:211` loops
`for (int seed = 0; seed < seed_count; ++seed)` with `seed_count` defaulting to
5 (line 89). Same seeds, same count, all three engines.

Given the audit's finding that the same configuration measured 6,994 and 8,682
samples/s an hour apart, a five-seed aggregate is the only place in the suite
where a quality number carries any statement about its own variance.

**Its configuration lives in one place for the two engines that can share it.**
`xf_common.py:22-27` holds the four scenarios and both Python drivers import
them. That is section 8's target architecture, working.

### What silently diverges

**1. The scenario table is duplicated across the language boundary.**
`recurrent_lstm_forecasting_benchmark.cpp:117-120` restates all four scenarios
in C++ syntax. They agree with `xf_common.py:23-26` exactly today — checked
field by field:

| id | past | future | hidden | lr | batch | max epochs | patience | multi |
|---|---|---|---|---|---|---|---|---|
| B1 | 24 | 1 | 32 | 0.003 | 128 | 120 | 20 | false |
| B2 | 48 | 1 | 48 | 0.003 | 128 | 100 | 20 | false |
| B3 | 72 | 24 | 64 | 0.002 | 128 | 80 | 20 | true |
| B4 | 168 | 24 | 64 | 0.001 | 128 | 60 | 15 | true |

Nothing here is wrong yet. It is worth recording because it is the one
duplication in the suite that cannot be fixed by importing a module: C++ cannot
read `xf_common.py`. This family is therefore the concrete argument for section
7's `suite.json` — a data file both sides read is the only way a definition is
shared across the boundary rather than transcribed and hoped over.

**2. `SEEDS` is copied where `SCENARIOS` is shared.** The identical `SEEDS`
expression appears in both Python drivers rather than in `xf_common.py` next to
the scenarios it belongs with. Harmless today, and the same class of thing as
the scenario table, one import away from being fixed.

**3. The runner does not run the comparison by default.**
`run_forecasting.py:273` sets `--frameworks` to `"opennn"`. A default invocation
produces OpenNN numbers only — so the three-way comparison this directory is
built for happens only when someone remembers to ask for it. The two Python
drivers, the shared scenario module and the aggregation are all in place; the
default just does not reach them.

This is the family's one actual defect, and it is a one-word fix
(`default="opennn,pytorch,tensorflow"`). Worth doing before step 6 re-baselines,
because a five-seed three-way forecasting comparison is the strongest quality
evidence the suite is capable of producing and it is currently not being run.

### What is genuinely distinct

- **The CPU/GPU second phase.** The OpenNN driver trains every scenario on GPU
  and then reruns them on CPU to report a speedup per scenario and network
  (`recurrent_lstm_forecasting_benchmark.cpp:9-12`). That is a
  device-comparison protocol, not a duplicate definition.
- **Two networks per scenario.** Each engine trains both a plain `Recurrent`
  and an `LSTM` for every scenario, which doubles the definitions in a way that
  is the point of the benchmark rather than redundancy.

### What the three survivors must pin

Nothing to merge. What this family needs is the opposite of the others: keep it
as it is, fix the `--frameworks` default, and move its scenario table into
whatever `suite.json` becomes so the C++ side stops transcribing it.

It should also be the model for the rest. If the merged dense, transformer and
CNN definitions end up with a shared scenario table, five-seed aggregation and
one driver per protocol, they will look like this directory does now.

---

## Decisions

Settled 2026-08-24 from the evidence above, for plan steps 4-6. Both were listed
in the family sections as "must be made before the merge".

### D1 — Output formulation: logits, once OpenNN can express them

**Decided: all three engines emit logits into a fused loss for training, and
keep the activation for inference. Blocked on a library change; the interim rule
is below.**

The reading changed the question. This looked like a convention nobody agreed —
some sites on logits, some on sigmoid. It is not. OpenNN takes the activation in
the forward at *every* dense and transformer site, because `Loss::Error` has no
from-logits member: `CrossEntropy` and `CrossEntropy3d` both consume
probabilities (`error_functions.cpp:254`, `:414`). PyTorch and TensorFlow depart
from it wherever an author wrote what their framework makes natural.

So "pick a convention" is not available. The two real options:

| | consequence |
|---|---|
| **(a) everyone takes the activation in the forward** | matches OpenNN today, no library work — but strips PyTorch and TensorFlow of a fused path they genuinely have, and makes all three slower in the same way. A benchmark that removes a competitor's real optimisation to match its own limitation flatters itself. |
| **(b) everyone emits logits into a fused loss** | the formulation practitioners write, the numerically stable one, and the faster one. Needs a from-logits loss in OpenNN. |

**(b).** (a) would be measuring OpenNN's ceiling and calling it everyone's.

The cost is not only fairness. At the transformer, OpenNN materialises a
`[batch, seq, vocab]` probability tensor that PyTorch and TensorFlow never
write. That is real throughput OpenNN is leaving on the table, in the library
rather than in the benchmark, so the fix helps users and not just this suite.

**Library task, prerequisite to step 4:** add a from-logits cross-entropy —
binary and 3d — that consumes pre-activation outputs and fuses the log-softmax
(or log-sigmoid) into the loss and its gradient. Then the merged definitions
drop the output activation for training and keep it for inference.

**Interim rule, so step 4 is not blocked on the library:** the merged definition
declares its formulation per engine in `suite.json` rather than claiming the
three are identical, and the one *unintended* divergence gets fixed now —
`pytorch_speed.py` moves to sigmoid + `BCELoss`, matching the other two engines
at site 5 and its own five sibling sites. Site 1 stays as it is: its two logits
drivers are deliberate and documented, and will be correct once (b) lands.

### D2 — Initialisation: Glorot, pinned explicitly in all three

**Decided: every merged definition sets its initialiser explicitly. Glorot
(Xavier uniform) where the three must agree. No definition inherits a framework
default.**

Nothing agrees today, and differently in each family: dense has OpenNN and
TensorFlow on Glorot against PyTorch's Kaiming; transformer bodies agree on
Xavier while embeddings and output projections do not; CNN has three different
initialisers and OpenNN's is not fan-scaled at all.

Glorot because it is already the majority — OpenNN's `finalize_build` and both
Keras `Dense`/`Conv2D` defaults — so it is the smallest change, and because all
three can express it explicitly: `nn.init.xavier_uniform_`,
`kernel_initializer="glorot_uniform"`, `set_parameters_glorot()`.

Explicitly, not by default, because "each framework as a practitioner would use
it" is the wrong instinct for the quality track. A convergence benchmark that
lets three engines start from three distributions is partly measuring
initialisation, and reports it as engine quality.

Two consequences worth stating:

- **It moves the published accuracy and convergence numbers.** Re-baseline
  material (step 6), not a silent fix.
- **It does not matter for speed, capacity or energy.** Those tracks may keep
  framework defaults; pinning them costs nothing and is simpler to state, so
  pin them anyway.

**OpenNN already ships the alternative.** `NeuralNetwork::set_parameters_pytorch()`
exists and is overridden where PyTorch's scheme genuinely differs — `Recurrent`,
`LongShortTermMemory`, `Combination` — defaulting to Glorot elsewhere. If a
future decision prefers matching PyTorch instead, the mechanism is there and
unused; nothing in the benchmarks calls it.

**Library task, separate from the merge:** `ResNet::ResNet` and `YoloNetwork`
call `set_parameters_random()` — flat `uniform(-0.1, 0.1)`, no fan scaling —
instead of going through `finalize_build`'s Glorot, and
`ConvolutionOperator::set_parameters_glorot` exists but is never reached from
those builders. Confirm whether that is deliberate before the CNN quality
numbers are re-baselined; against Glorot's ~0.026 limit for a 3x3 convolution
with 512 input channels, a flat +-0.1 is about four times too wide, fifty layers
deep.
