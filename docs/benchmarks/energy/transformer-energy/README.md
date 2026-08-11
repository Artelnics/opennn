# Transformer fixed-work GPU energy — OpenNN vs PyTorch vs TensorFlow

Fixed-work energy benchmark: every engine trains the same model on the same
data for the **same number of epochs**, and we integrate GPU power over exactly
that training window. The question it answers: *how much electricity does each
framework spend on identical work?* Same semantics as
[`../higgs-dense-energy/`](../higgs-dense-energy/) and
[`../resnet50-energy/`](../resnet50-energy/), so the three families compare
like-for-like. (The runner's earlier energy-to-target mode — train until an
epoch-mean CE gate — was retired in 2026-08; a run that misses the gate wastes
its full budget and the discarded runs made the aggregate unstable.)

## Workload

The ChatGPT example from `blank_cuda` (block 6): the encoder-decoder Transformer
from *Attention Is All You Need* (paper base **d512 / h8 / ff2048 / 6L**, ~84.8 M
parameters) trained sequence-to-sequence on the chat corpus (`prompt <TAB>
response`, Stanford Alpaca 47,487 pairs, vocab 19,443 in / 30,000 out, sequences
64 / 127).

## Fairness rules

**Identical everything that shapes the loss curve; per-engine fastest execution.**

- **Identical data, token for token**: PyTorch and TensorFlow read OpenNN's
  `tokens.bin` cache directly (per sample `[input_seq | target_seq]` int32,
  PAD=0; decoder input = START(2) + target shifted right).
- **Identical model**: same architecture, same parameter count (84,843,312 in
  OpenNN and TF; 84,845,360 in PyTorch, +0.002 %), same attention semantics
  (PAD keys masked in every attention, causal decoder self-attention — OpenNN
  applies both, so the counterparts do too), same Glorot-uniform initialization
  (biases zero, PAD embedding row zero; PyTorch's fused QKV is re-initialized
  as three separate d×d Glorot draws to match OpenNN/TF's per-projection fans).
- **Identical convergence hyperparameters**: batch 128, plain Adam lr 1e-4
  (no weight decay, no clipping, no dropout, no LR schedule), shuffled epochs,
  all samples in the training split, partial last batch kept.
- **Identical work**: every engine runs exactly `--epochs` epochs (the wrapper
  passes an unreachable CE target so max-epochs is the only stopping
  condition); the per-run check verifies the executed epoch count. The
  epoch-mean token CE over non-PAD targets is still recorded per epoch, so
  convergence equivalence stays auditable.
- **Per-engine fastest execution** (this is what the benchmark compares):
  OpenNN bf16 tensor-core path + CUDA graph; PyTorch autocast(bf16) + fused
  Adam + SDPA; TensorFlow `mixed_bfloat16` + `@tf.function(jit_compile=True)`
  (XLA).

Calibration note: lr 5e-4 (the ChatGPT example default at batch 64) parks
**all three** engines on the unigram plateau; lr 1e-4 descends steadily and every
engine follows the same trajectory. OpenNN's fp32 and bf16 loss curves are
indistinguishable, so bf16 is pure speed, not a quality trade.

## Energy measurement

Sampled GPU-power methodology:

- `nvidia-smi power.draw` sampled at 20 Hz for the whole process,
- trapezoidal integration restricted to the **training window** — each engine
  prints `TRAIN_START_UNIX` / `TRAIN_END_UNIX` around its training loop, so
  one-time corpus tokenization and Python imports are excluded, while warmup,
  cuDNN plan selection, CUDA-graph capture and XLA compilation are **included**
  (they are real energy the training pays),
- idle baseline measured fresh at startup on a quiet GPU; both total and
  active (idle-subtracted) energy are reported,
- N runs per engine (default 3), median ± stdev, GPU clock/temperature/throttle
  state snapshotted before and after every run.

GPU energy only (board sensor; sampled power, not a hardware joule counter).

## Files

| File | Purpose |
|------|---------|
| `opennn_transformer_energy.cpp` | OpenNN driver (bf16 + CUDA graph); also `probe` mode to derive the shared shapes |
| `pytorch_transformer_energy.py` | PyTorch counterpart (bf16 autocast, fused Adam, SDPA, matched masks/init/gate) |
| `tensorflow_transformer_energy.py` | TensorFlow counterpart (`mixed_bfloat16`, XLA, matched masks/init/gate) |
| `run_transformer_energy.py` | Orchestrator: idle baseline, 20 Hz power logging, windowed integration, immutable JSON to `../../results/` |

## How to run

```bash
# 1. Build the OpenNN driver (registered in docs/benchmarks/CMakeLists.txt).
cmake --build build --target opennn_transformer_energy -j

# 2. Make sure the machine is quiet (no other GPU/CPU-heavy processes):
nvidia-smi pmon -c 5

# 3. Full comparison (torch + TF live in the ml venv; TF gets its bundled CUDA
#    libs on LD_LIBRARY_PATH automatically).
python docs/benchmarks/energy/transformer-energy/run_transformer_energy.py \
    --epochs 10 --batch 128 --lr 1e-4 --runs 3
```

Writes `../../results/gpu-transformer-energy-fixed-work-<run_id>.json` with
per-run and aggregate energy (Wh, total and active), µJ per nominal
epoch-sample, training-window wall time, per-epoch loss histories, versions,
commit and GPU state.
