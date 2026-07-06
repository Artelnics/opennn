# Transformer training energy to target on the GPU: OpenNN vs PyTorch vs TensorFlow

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-07-03. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.71.05.*

Most training benchmarks measure **throughput** — samples per second at a fixed
epoch count. This one measures what a training run actually costs: **the GPU
energy consumed to train the same model, on the same data, to the same
quality**. Each engine runs in its fastest configuration and stops the moment it
reaches a fixed loss target; a power logger integrates the GPU draw over exactly
that window. Faster epochs only win if they translate into fewer joules for the
same trained model.

The workload is the ChatGPT example that ships with OpenNN (`blank_cuda`,
block 6): the encoder-decoder Transformer from *Attention Is All You Need*
(paper base **d512 / h8 / ff2048 / 6L**, ~84.8 M parameters) trained
sequence-to-sequence on 47,487 Stanford-Alpaca chat pairs
(`prompt <TAB> response`, vocab 19,443 in / 30,000 out, sequences 64 / 127).

## The result

**OpenNN reaches the same trained model with 28 % less GPU energy than PyTorch
and 40 % less than TensorFlow** (medians over converged runs, 4 seeds per
engine, target epoch-mean token CE ≤ 3.5):

| | OpenNN | PyTorch | TensorFlow |
|---|---:|---:|---:|
| **GPU energy to target** | **24.1 Wh** | 33.2 Wh (**+38 %**) | 39.8 Wh (**+66 %**) |
| Active energy (idle subtracted) | **21.4 Wh** | 29.3 Wh | 34.9 Wh |
| Time to target | **299 s** | 432 s | 546 s |
| Epochs to target | 8 | 8.5 | 9 |
| Average power while training | 290 W | 277 W | 263 W |
| Steady-state epoch time | **33.2 s** | 45.7 s | 47.2 s (+79 s XLA compile) |
| Runs converged | 2/4 | 4/4 | 3/4 |

The mechanism is the same one the transformer training-speed note found: OpenNN
draws the **highest instantaneous power** (290 W — it keeps the GPU busiest)
but finishes the same work in ~30 % less wall time, so it consumes the least
total energy. TensorFlow shows the opposite profile — lowest power, longest
time (its window also pays ~79 s of XLA compilation) — and ends up using the
most energy. Being fast and being frugal are the same thing here.

## Method: same curve, different engines

Energy-to-target is only meaningful if every engine walks the same loss curve —
otherwise the comparison is a hyperparameter lottery, not an engine benchmark.
Three things make the curves match here:

1. **Identical data, token for token.** PyTorch and TensorFlow read OpenNN's
   binary token cache (`tokens.bin`) directly: per sample
   `[input_seq | target_seq]` int32, PAD = 0, decoder input = START + target
   shifted right. No re-tokenization, no re-splitting.
2. **Identical model.** Same architecture and attention semantics — PAD keys
   masked in every attention block and a causal mask in decoder self-attention,
   which OpenNN applies natively — and the same Glorot-uniform initialization
   (biases zero, PAD embedding row zero). One subtle but decisive detail:
   PyTorch fuses Q/K/V into a single (3d, d) matrix whose joint Xavier fan
   halves the init limit; re-initializing it as three separate d×d Glorot draws
   (what OpenNN and Keras do) aligns its convergence with the others.
3. **Identical convergence hyperparameters and gate.** Batch 128, plain Adam
   lr 1e-4 (no decay/clipping/dropout/schedule), shuffled epochs, and the same
   stop rule: epoch-mean token cross-entropy over non-PAD targets ≤ 3.5,
   checked at every epoch end. That is exactly what OpenNN's
   `CrossEntropyError3d` reports and `set_loss_goal` stops on; the PyTorch gate
   is `CrossEntropyLoss(ignore_index=0)`, the TensorFlow gate a PAD-masked
   sparse categorical cross-entropy.

With those three fixed, the calibration runs put all three engines on the same
trajectory (epoch-mean CE over the first three epochs):

| engine | epoch 0 | epoch 1 | epoch 2 |
|---|---:|---:|---:|
| OpenNN (bf16) | 7.07 | 6.20 | 5.59 |
| OpenNN (fp32) | 7.07 | 6.22 | 5.67 |
| PyTorch (bf16) | 7.07 | 6.34 | 5.80 |
| TensorFlow (bf16 + XLA) | 7.05 | 5.96 | 5.24 |

Two calibration findings worth recording. First, the learning rate the ChatGPT
example uses at batch 64 (5e-4) parks **all three engines** on the unigram
plateau (~CE 6.77) at batch 128 — 1e-4 descends steadily, so that is the
benchmark setting. Second, OpenNN's fp32 and bf16 curves are indistinguishable,
so bf16 is pure speed (33.6 vs 68 s/epoch), not a quality trade.

What is *not* equalized is the execution stack — that is the thing being
measured. Each engine runs its fastest honest configuration:

| Engine | Precision | Execution |
|--------|-----------|-----------|
| OpenNN | bf16 tensor-core path | CUDA graph on, fused cuDNN attention |
| PyTorch | `autocast(bfloat16)` | eager, SDPA/flash attention, fused Adam, TF32, cudnn.benchmark |
| TensorFlow | `mixed_bfloat16` | `@tf.function(jit_compile=True)` — full XLA |

Unlike the [max-batch benchmark](transformer-max-batch/README.md) (which keeps
XLA off to compare op-by-op execution), here TensorFlow gets XLA because the
question is "minimum energy to the target", and its compile time and energy are
charged to its training window.

## Energy measurement

Same protocol as the [dense energy benchmark](rosenbrock-max-batch/run_energy.py):
`nvidia-smi power.draw` sampled at 20 Hz, trapezoid-integrated — but restricted
to the **training window** each engine marks with `TRAIN_START_UNIX` /
`TRAIN_END_UNIX` around its training loop. One-time corpus tokenization and
interpreter/library start-up are excluded; warmup, cuDNN plan selection,
CUDA-graph capture and XLA compilation are **included**, because they are real
electricity the training pays. The idle baseline is measured fresh on the quiet
GPU before the first run; both total and active (idle-subtracted) energy are
reported. Four runs per engine (one seed each, `seed_base + r`), medians over
converged runs, GPU clock/temperature/throttle state snapshotted before and
after every run. GPU energy only (board
sensor, sampled power — not a hardware joule counter).

Conditions for the published run: idle baseline 32.3 W, desktop-only GPU load
verified with `nvidia-smi pmon` before starting (no browser/compute processes),
no thermal throttling in the before/after GPU-state snapshots. Software:
OpenNN at commit `b78d3bab550b` (g++ 13, CUDA 13.0, cuDNN frontend), PyTorch
2.6.0+cu124, TensorFlow 2.21.0, CPython 3.12.3, driver 595.71.05.

## Reproducing

Code and per-engine drivers live in [`docs/benchmarks/energy/`](energy/):

```bash
# 1. Build the OpenNN driver (registered in docs/benchmarks/CMakeLists.txt).
cmake --build build --target opennn_transformer_energy -j

# 2. Check the machine is quiet, then run the full comparison.
nvidia-smi pmon -c 5
python docs/benchmarks/energy/run_transformer_energy.py \
    --target 3.5 --batch 128 --lr 1e-4 --runs 4
```

The orchestrator derives the model shape from the OpenNN corpus probe, runs the
three engines under the power logger, and writes an immutable JSON artifact
(per-run energies, loss histories, versions, commit, GPU states) to
[`results/`](results/).

The published run is
[`results/gpu-transformer-energy-to-target-20260703T103849Z.json`](results/gpu-transformer-energy-to-target-20260703T103849Z.json).

## Caveats

- **GPU energy only.** CPU/DRAM/PSU overhead is not included; OpenNN's lower
  host overhead (native C++, no interpreter) would only widen a whole-system
  gap.
- **Sampled power.** 20 Hz board-sensor sampling, not a hardware joule counter;
  the dense energy benchmark cross-checked this methodology against a wall
  meter.
- **One machine, one workload.** RTX 4080 (16 GB), this chat corpus and model
  shape. Repeat on the reference machine before public headline use.
- **Convergence dispersion.** The engines' curves match within ±0.5 CE over the
  calibration epochs but are not bit-identical (independent RNG streams,
  framework-internal numerics); epochs-to-target spread 7–10 across seeds. The
  per-epoch loss histories ship in the JSON so this is auditable.
- **Occasional optimization collapse.** At lr 1e-4 / batch 128 this task sits
  near an escape boundary: a run occasionally fails to leave the unigram
  plateau (~CE 5.5–6.2 after 20 epochs) and never reaches the target. Failed
  runs are recorded (`reached_goal=0`, loss history included) and excluded from
  the medians, as in the convergence-gate benchmark. Rates in the published
  run: OpenNN 2/4, TensorFlow 3/4, PyTorch 4/4 converged. PyTorch is
  bit-deterministic run-to-run; OpenNN is not — in a preliminary pass one of
  three *identical-seed* OpenNN runs collapsed while the other two matched, so
  for OpenNN the collapse is per-run stochastic (GPU/OMP reduction ordering),
  not seed-determined. Tightening OpenNN's convergence robustness here (or a
  short LR warmup in the benchmark recipe for all engines) is the natural
  follow-up.
