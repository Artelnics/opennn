# Duplication ledger

Step 1 of [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md). Before deleting any
duplicate, record what is byte-identical, what silently diverges, and what is
genuinely distinct driver logic — so that when a number moves after the merge,
this says why.

Everything below was read out of the tree, not estimated. Line references are as
of the commit that adds this file.

Status: **dense done**, transformer / CNN / recurrent pending.

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

Site 1 makes the same choice deliberately and says so — its header calls logits
plus `BCEWithLogitsLoss` / `from_logits=True` "the standard formulation" — and
there it is applied to **all three** engines, so it stays like-for-like. Site 5
applies it to one of three.

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

Pending. 15 definitions across `throughput/attention-speed`,
`capacity/transformer-max-batch`, `energy/transformer-energy`.
Known before starting: the three sites use three different datasets — a
synthetic LCG corpus, WMT14, and `chat_pairs`.

## CNN (ResNet-50)

Pending. 12 definitions across `throughput/resnet50`,
`capacity/resnet50-max-batch`, `energy/resnet50-energy`.

**Correction to the plan.** Section 2 of REORGANIZATION_PLAN.md heads this
"CNN has no TensorFlow at all". That is too strong — the evidence cell under it
is accurate but the headline is not. `tensorflow_resnet50_speed.py` and
`tensorflow_resnet50_infer.py` both exist, and two of the three runners default
to all three engines:

| runner | `--engines` default |
|---|---|
| `run_resnet50.py` | `opennn,pytorch,tensorflow` |
| `run_resnet50_infer.py` | `opennn,pytorch,tensorflow` |
| `run_imagenet_resnet50.py` | `opennn,pytorch_fast,pytorch_eager` |

So the gap is narrower and more specific: the **ImageNet** ResNet-50 runner is
two-way, while the CIFAR/imagenet-like ones are three-way. What is missing is a
TensorFlow arm for the ImageNet runner, not a TensorFlow ResNet-50.

## Recurrent

Pending. 3 definitions in `quality/recurrent-lstm-forecasting`; already flat, so
nothing to merge. The defect here is different: `run_forecasting.py` defaults to
`--frameworks opennn`, so it does not currently run as a three-way comparison at
all.
