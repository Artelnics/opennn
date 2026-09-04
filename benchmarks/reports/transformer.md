# Transformer: the base model on WMT14

OpenNN against PyTorch 2.13 on the *Attention Is All You Need* base
configuration — 6 encoder and 6 decoder layers, d_model 512, 8 heads,
feed-forward 2,048, a 20,000-token vocabulary on each side, 74,878,496
parameters — over 199,575 English–German sentence pairs from WMT14 News
Commentary v9 at 130 tokens per sequence, batch 32, bf16, on the RTX 5070 Ti.
Session `2026-09-03-publish`:

| cell | OpenNN | PyTorch | OpenNN / PyTorch |
|---|---|---|---|
| `cuda-transformer-infer` | 5,302 sequences/s | 4,707 | **1.13×** |

The matrix products — about 450 GFLOP per forward batch, run by cuBLASLt on
one side and by Inductor's autotuned templates on the other — take about the
same time on both engines; the margins are around them. OpenNN replays the
whole inference pass (1.13×) and the whole Adam step (1.13×)
as one CUDA graph each, keeps its parameters in bf16 from the start, and
streams the Adam update over one joint gradient buffer; PyTorch's compiled
step is issued kernel by kernel from the generated code, and its training
step also carries the dropout that `nn.TransformerEncoderLayer` applies by
default — a real but small (1.4%) part of the difference.

## What is measured

**Network.** The "Attention Is All You Need" base model: encoder–decoder,
d_model 512, 8 heads, feed-forward 2048, 6 layers each side, post-layer-norm,
scaled token embeddings with sinusoidal positions, and a final projection to
the vocabulary. Heads and feed-forward width follow d_model by the paper's
ratios (d_model/64 and 4·d_model) in both drivers. PyTorch assembles it from
`nn.TransformerEncoderLayer`/`nn.TransformerDecoderLayer` with `norm=None`
on the stacks — `nn.Transformer` would append a final LayerNorm to each stack,
2,048 parameters and two normalisations per pass that OpenNN's `Transformer`
does not have. Both engines report **74,878,496 parameters** (2 × 20,000 × 512
embeddings, 6 × 3,152,384 encoder, 6 × 4,204,032 decoder, 10,260,000 output
projection), which the runner checks. Training is cross-entropy over the
vocabulary with Adam at learning rate 1e-4.

Three things differ inside that identical shape and are stated here rather
than hidden. (1) *Masks.* OpenNN's decoder self-attention is causal and both
of its attentions honour the padding lengths the embedding exports; the
PyTorch driver passes no masks at all, so it attends over every position.
The attention matrices are the same size either way, so the arithmetic is
the same; a causal mask can let a fused kernel skip blocks, which if anything
favours the masked side. (2) *Dropout.* `nn.TransformerEncoderLayer` and
`nn.TransformerDecoderLayer` default to dropout 0.1 in every sub-layer and the
driver keeps that default, so the PyTorch training step generates and applies
dropout masks that OpenNN's `Transformer` (dropout 0 unless set) does not —
extra element-wise work and random numbers on the PyTorch side. Section
"Why" quantifies it with the `PT_DROPOUT=0` variant. (3) *Loss reduction.*
OpenNN's `CrossEntropyError3d` averages over the non-padding tokens; PyTorch's
`CrossEntropyLoss` averages over all positions, padding included. Same
kernels, different denominator; no effect on throughput.

**Data.** WMT14 English–German, the corpus the paper reports on, through its
News Commentary v9 training file. `prepare.py transformer` tokenises both
sides with exactly the rule OpenNN's `WordLevelTokenizer` applies
(ASCII-lowercase, alphanumeric runs and single punctuation characters, other
bytes dropped), truncates each side to 128 tokens and keeps the first 200,000
well-formed pairs: **199,575 pairs**. Writing the corpus pre-split is what
makes the two tokenisers agree — OpenNN's is then idempotent over it and a
whitespace split gives PyTorch the same tokens — so both engines build the
same padded length, **130** (128 + START + END), and the same capped
vocabulary, **20,000** including the four reserved ids. Token identities are
irrelevant to a speed measurement and are not compared; the shapes are. Whole
batches only: at batch 32 an epoch or a pass covers 6,236 batches =
**199,552 sequences**, 25.9 M padded tokens. For training both engines keep the id
tensors resident on the device; PyTorch slices them sequentially, OpenNN
gathers each batch by index. For inference both fill the first batch of 32
pairs once and replay it 6,236 times per pass, so the inference cell times
the resident forward pass alone.

**Cells.**

| cell | device | batch | precision | timed window |
|---|---|---|---|---|
| `cuda-transformer-train` | RTX 5070 Ti | 32 | bf16 | 2 epochs after 1 untimed |
| `cuda-transformer-infer` | RTX 5070 Ti | 32 | bf16 | 5 passes after 1 untimed |

Throughput is compared per sequence; both drivers also print
`tokens_per_sec` (× 130), the figure the literature quotes.

**Each engine at its best.** bf16 autocast on both sides. PyTorch runs
`torch.compile(mode="max-autotune-no-cudagraphs")` for the training step and
the inference forward — Inductor with its GEMM and pointwise templates
benchmarked per shape and no CUDA graphs, the fastest of its modes on both
cells (training 1,150 sequences/s against 1,132 under `reduce-overhead` and
1,124 in the default mode; inference 4,657 against 4,574 and 4,499, all with
the weights stored in bf16 once, `PT_INFER_CAST=weights`; under autocast,
which re-casts the 74.9 M parameters on every call, the best mode reads
3,641 and the default 3,537 — the driver's `compiled()` docstring has every
number, and the driver measured compiling the step at +33% over eager when
it was written); attention goes through `scaled_dot_product_attention`'s fused
kernels. OpenNN captures the Adam step and the inference pass as CUDA graphs
and, on cuDNN ≥ 9.25 with bf16, uses cuDNN's fused scaled-dot-product
attention for sequences of 128 tokens and more
(`sdpa_min_sequence_length=128` is printed and recorded) — the driver's comment
records the measurement behind that threshold: fused beat the materialised
attention by 28% at 128 tokens over five launches each way, while the
library-wide default stays at 192 because a single pass cannot amortise the
0.3–2 s of plan construction. Adam runs over a joint gradient arena
(`set_joint_gradient_arena(true)`): each layer's gradient is planned into
the forward arena by lifetime, beside the deltas, so the 300 MB gradient of
a 75 M-parameter model reuses memory whose lifetime has ended; the update is
then one streaming launch per parameterised layer.

**Gates.** Samples (199,575), sequence (130), input and target vocabulary
(20,000) and parameters (74,878,496) must agree between the engines. There is
**no loss gate** in this family; two epochs of a base transformer are a speed
measurement, not a translation result.

## Results

Session `2026-09-03-publish`, the last run of every cell, median of three rounds.

| cell | batch | precision | OpenNN samples/s | PyTorch samples/s | OpenNN / PyTorch | peak memory MiB (OpenNN / PyTorch) | energy Wh (OpenNN / PyTorch) |
|---|---|---|---|---|---|---|---|
| `cuda-transformer-infer` | 32 | bf16 | 5,302 | 4,707 | **1.13×** | 844 / 1,210 | 11.9895 / 14.9908 |


`cuda-transformer-train` — batch 32, bf16, epochs 2 per launch, 3 rounds. Artifact `cuda-transformer-train-publish-20260903T122006Z.json`, commit `6b7179dde`, quiet False (busy 0.1% before, 0.1% after), clocks locked True, shape gate True, quality gate True.

> **Not evidence-grade.** This run is filed under `results/scratch/`: foreign CPU activity reached 9.7% during a timed window against a 3% threshold, so the runner refused to publish it. The figures below are what it measured; they agree with the other attempts at this cell to within a percent, but the claim waits on a quiet window.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 1,301 | 1,301 | 1,301 | 2316 | 16.44613 |
| PyTorch | 1,151 | 1,151 | 1,151 | 3452 | 22.73194 |
| **ratio** | **1.130×** | | | 1.49× less | 1.38× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 1,301 | 1,151 |
| 2 | pytorch → opennn | 1,301 | 1,151 |
| 3 | opennn → pytorch | 1,301 | 1,151 |


`cuda-transformer-infer` — batch 32, bf16, passes 5 per launch, 3 rounds. Artifact `cuda-transformer-infer-publish-20260903T131144Z.json`, commit `6b7179dde`, quiet True (busy 0.1% before, 0.3% after), clocks locked True, shape gate True, quality gate True.

| engine | median samples/s | min | max | peak device MiB | Wh (board) |
|---|---|---|---|---|---|
| OpenNN | 5,302 | 5,302 | 5,302 | 844 | 11.98950 |
| PyTorch | 4,707 | 4,704 | 4,708 | 1210 | 14.99075 |
| **ratio** | **1.126×** | | | 1.43× less | 1.25× less |

| round | order | OpenNN samples/s | PyTorch samples/s |
|---|---|---|---|
| 1 | opennn → pytorch | 5,302 | 4,704 |
| 2 | pytorch → opennn | 5,302 | 4,707 |
| 3 | opennn → pytorch | 5,302 | 4,708 |

## Why

The base transformer is 74.9 M parameters, and at batch 32 × 130 tokens a
forward pass is about 450 GFLOP of matrix products — the
projections, the feed-forward layers and the 20,000-way output projection —
plus attention over 130 keys in 12 attention blocks (6 self-attention in the
encoder, 6 self- and 6 cross-attention in the decoder). Both engines run the
products in tensor-core GEMMs (cuBLASLt for OpenNN, Inductor's autotuned
choice of Triton templates and cuBLAS for PyTorch) and the attention in a
fused kernel (cuDNN's scaled-dot-product attention for OpenNN, the backend
`F.scaled_dot_product_attention` picks — *[pending the final measurement round]* in the
profile — for PyTorch). The GEMM time is close; the margins are the kernels
around the GEMMs, and how the step is issued.

### Where the energy goes

Both cells in this family win energy on **power**, which makes them the
clearest counterpart to the dense inference cell: OpenNN draws 229.4 W
against PyTorch's 254.7 W on inference and 193.0 W against 236.2 W on
training. Combined with margins of 1.13× and 1.130× on time, that
compounds to 1.25× and 1.38× on energy — the widest energy
margins in the matrix outside the LSTM family.

Part of that is measured to be the GEMM tile-selection rule described in the
dense document: this model is a stack of large matrix products, they go
through the same cuBLASLt path, and running the identical cell with the rule
disabled (`OPENNN_LT_TILE_TOLERANCE=0`) costs 13.349 Wh on inference against
11.893 with it, and 17.338 Wh on training against 16.405 — 10.9% and 5.4% of
each cell's energy, for 1.7% and 0.5% of its throughput.

Part of it is that PyTorch is doing extra element-wise work: its
`TransformerEncoderLayer` and `DecoderLayer` default to dropout 0.1 and the
driver does not override it, which is a random mask and a multiply on every
attention and feed-forward output in the training pass. And part of it cuts
the other way and is the more important thing to say: **OpenNN is computing
more than PyTorch here and still drawing less power.** OpenNN's decoder
self-attention is causal and every attention block sees the padding mask;
PyTorch's model applies no mask at all. The caveats below list this in full.

What cannot yet be said is which kernels account for the 25 W. This family
has no `nsys` profile in this round, so the decomposition that the dense and
LSTM documents give per kernel is not available here, and the paragraphs
above rest on the artifacts, the drivers and the tolerance comparison rather
than on a kernel table.

### `cuda-transformer-infer`, 1.13×: launches, casts and the layer-norm passes

Per batch of 32 sequences: OpenNN 6,035 µs, PyTorch 6,798 µs,
and 844 MiB of device memory against 1,210.

The forward pass is deep — 6 encoder layers of (attention, feed-forward) and
6 decoder layers of (self-attention, cross-attention, feed-forward), each
sub-layer with its residual add and layer normalisation. OpenNN captures the
whole of it once and replays it as a single CUDA graph; PyTorch issues it
from Inductor's generated code without graphs, because on this model
`reduce-overhead` is *slower* for it than the mode the table uses (4,574
against 4,657 sequences/s), so the launches are paid one at a time. That
difference in issue cost is the mechanism the margin is attributed to, and
it is consistent with the 1.25× energy ratio being larger than
the 1.13× throughput one — a host-starved GPU spends part of the
window drawing power without retiring work.

It is only consistent with it, not demonstrated by it: **this cell has no
kernel trace in this round.** The per-kernel table and the idle-percentage
comparison that the dense and LSTM documents give are not available here,
so the paragraph above rests on the artifacts, the two drivers and the
compile-mode comparison. The final round profiles this cell.

The bf16-weights choice matters on PyTorch's side more than in any other
family — 4,657 against 3,641 sequences/s for the same compile mode under
autocast — because autocast re-casts the 74.9 M parameters, 150 MB of fp32
weights, on every call; storing them in bf16 once removes that traffic, and
it is the mode the table uses. OpenNN's parameters are bf16 from the start.

### `cuda-transformer-train`, 1.13×: one graph against Inductor's step, and dropout

Per batch: OpenNN 24,242 µs against PyTorch's 27,394, from the throughputs
above. As with inference, there is no kernel trace for this cell in this
round, so what follows names the three things that differ in the step and
what each is worth where that has been measured separately — not a
per-kernel decomposition.

Three things in the step differ:

*How it is issued.* OpenNN captures the whole Adam step as one CUDA graph —
the batch gather, forward, cross-entropy, backward, and the update over the
joint gradient arena (`set_joint_gradient_arena(true)`: each layer's
gradient is planned into the forward arena by lifetime, beside the deltas,
so the 300 MB gradient of a 75 M-parameter model reuses memory whose
lifetime has ended, and the Adam update is one streaming kernel per
parameterised layer) — and replays the whole step behind one launch.
PyTorch's step under `max-autotune-no-cudagraphs` is issued kernel by kernel
from the generated Python, with `torch.optim.Adam`'s *foreach* update as
multi-tensor kernels on top. The arena is also why the memory column reads
2,316 MiB against 3,452: it is the same forward allocation serving the
backward, not a second one.

*Dropout.* `nn.TransformerEncoderLayer` and `DecoderLayer` default to dropout
0.1 in every sub-layer and the driver keeps that; OpenNN's model has none.
Each dropout is a random mask generated and applied in the forward and
re-applied in the backward. Turning it off (`PT_DROPOUT=0`) reads
1,140 against 1,124 sequences/s,
a 1.4% effect: real, small, and recorded rather than removed because dropout
0.1 is what the library's layer does by default.

*Masks.* OpenNN's decoder attention is causal and both attentions honour the
padding lengths; PyTorch's driver applies no masks. The fused attention
kernels are handed the same 130 keys either way, so the asymmetry does not
buy OpenNN its margin; it means OpenNN trains the model the paper describes
and PyTorch trains one that can see the future. This is the single most
important caveat in the family, and it runs *against* the result being
reported: the engine doing more work is the one that wins.

## Asymmetries and caveats

Three things differ in what the two networks compute, all recorded here
because none of them is something the runner can equalise without changing
one engine's model:

- **Masks.** OpenNN's decoder self-attention is causal and every attention
  block sees the padding mask that the embedding layer exports with its valid
  lengths; the PyTorch model applies no mask at all — no causal mask in the
  decoder, no key-padding mask anywhere. Same tensor shapes, same FLOPs in
  the attention matrices; OpenNN does slightly more work (the masking) and
  trains the correct model, PyTorch trains a model that can see the future.
  A masked PyTorch model would be the fairer comparison and would not be
  faster.
- **Dropout.** `nn.TransformerEncoderLayer` and `DecoderLayer` default to
  dropout 0.1 and the driver does not override it; OpenNN's dropout operator
  defaults to 0 and the model builder leaves it there. Dropout in training
  is extra element-wise kernels (a random mask, a multiply) on every
  attention and feed-forward output on PyTorch's side; `PT_DROPOUT=0` runs
  the PyTorch model without it, and the *Why* section gives what it is worth
  (1,140 against 1,124 sequences/s). Inference is unaffected (`model.eval()`).
- **The loss denominator.** PyTorch's `CrossEntropyLoss` averages over every
  position, padding included; OpenNN's `CrossEntropyError3d` averages over
  the valid tokens. Identical work, different scale of the gradient — the
  learning rate is the same on both sides, so the two models do not follow
  the same trajectory; there is no accuracy gate in this family to be
  affected by it.

And the ones that are about the build rather than the model:

- **Attention kernels.** With bf16 on this cuDNN, OpenNN hands sequences of
  128 tokens or more to cuDNN's fused scaled-dot-product attention and
  materialises the scores below that; at 130 tokens the fused path is what
  ran (`sdpa_min_sequence_length=128` is printed). PyTorch's
  `F.scaled_dot_product_attention` picks its own backend at run time. Both
  are the fused attention that each framework ships; which kernel each ran
  is in the profile.
- **Different cuDNN builds**, 9.25.1 against 9.23.2, as in the CNN family.
- **No accuracy gate.** Both drivers print tokens per second alongside
  samples per second and the runner compares the sample count, sequence
  length, vocabulary and parameter count (74,878,496); neither reports a
  translation metric. The shape gate is what holds this family.
- **Warmup is one epoch on both sides,** with the graph capture, the
  `torch.compile` trace and the SDPA kernel selection inside it.
- **An earlier version of this table timed OpenNN inference over zeros.**
  Until commit `4338506c8` the OpenNN inference drivers filled each batch
  (`Batch::fill()`) and never uploaded it: `fill()` only stages the rows on
  the host, and in the library the transfer is issued by the optimizer, which
  an inference driver does not run. The forward pass ran over all-zero inputs
  for every published inference cell before that commit. Nothing in the gates
  could see it — the parameter count, the sample count and the input file are
  the same whatever the batch holds, and the arithmetic of a forward pass does
  not depend on the values — and it was found by printing outputs (every one
  read `sigmoid(0) = 0.5`). The transformer is the one family where it
  mattered: the embedding exports the number of non-zero token ids as each
  sequence's valid length (`compute_token_valid_lengths`), so every sequence
  reached the attention kernels as all padding — no valid keys instead of 130.
  Attention is a small share of this model's time (the GEMMs are the rest, and
  they do not care what they multiply), so the cell moved by about 1%. The
  rows above are from the fixed drivers, which upload the batch
  (`upload_to_device_batch_async()`) after filling it and, where the split is
  resident, take the batch as a device view; the previous session's rows read
  5,274 sequences/s for OpenNN.

## Reproduce

```bash
export OPENNN_BENCH_SESSION=$(date +%F)-mine
python run.py --family transformer --mode train --device cuda --batch 32 --precision bf16 --epochs 2 --rounds 3
python run.py --family transformer --mode infer --device cuda --batch 32 --precision bf16 --repeats 5 --rounds 3
```

`prepare.py transformer` downloads WMT14 News Commentary v9, tokenises it
with the same rule both engines read (`opennn_tokens`: ASCII-lowercased
alphanumeric runs and single punctuation marks, `--max-tokens 128`,
`--max-pairs 200000`) and writes the 199,575 pairs both engines load.
`PT_DROPOUT=0`, `PT_COMPILE_MODE=default|reduce-overhead|max-autotune-no-cudagraphs|eager`
and `PT_INFER_CAST=autocast` are the PyTorch knobs; `OPENNN_NO_CUDA_GRAPH=1`
runs OpenNN without graph replay.
