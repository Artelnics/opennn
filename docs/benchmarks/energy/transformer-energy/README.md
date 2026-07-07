# Energy to target: chat Transformer trained to a fixed quality (GPU) — OpenNN vs PyTorch vs TensorFlow

MLPerf-style `training_time_to_quality` benchmark with **energy** as the headline
metric: every engine trains the same model on the same data until it reaches the
same quality target, and we integrate GPU power over exactly that training window.
The question it answers: *how much electricity does each framework need to produce
the same trained model?*

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
- **Identical gate**: epoch-mean token cross-entropy over non-PAD targets
  ≤ target, checked at every epoch end. This is the same quantity OpenNN's
  `CrossEntropyError3d` reports and its `set_loss_goal` stops on; PyTorch uses
  `CrossEntropyLoss(ignore_index=0)`, TensorFlow masks PAD tokens explicitly.
- **Per-engine fastest execution** (this is what the benchmark compares):
  OpenNN bf16 tensor-core path + CUDA graph; PyTorch autocast(bf16) + fused
  Adam + SDPA; TensorFlow `mixed_bfloat16` + `@tf.function(jit_compile=True)`
  (XLA).

Calibration notes (RTX 4080, batch 128): lr 5e-4 (the ChatGPT example default at
batch 64) parks **all three** engines on the unigram plateau (~6.77); lr 1e-4
descends steadily and every engine follows the same trajectory within ±0.5 CE.
OpenNN's fp32 and bf16 loss curves are indistinguishable (7.07/6.22/5.67 vs
7.07/6.20/5.59 over three epochs), so bf16 is pure speed, not a quality trade.

## Energy measurement

Same methodology as the dense energy benchmark (`rosenbrock-max-batch/run_energy.py`):

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
| `opennn_transformer_energy.cpp` | OpenNN driver (bf16 + CUDA graph, `set_loss_goal` gate); also `probe` mode to derive the shared shapes |
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
    --target 3.5 --batch 128 --lr 1e-4 --runs 4
```

Writes `../../results/gpu-transformer-energy-to-target-<run_id>.json` with per-run
and aggregate energy (Wh, total and active), training-window wall time, epochs
to target, per-epoch loss histories, versions, commit and GPU state.

Results write-up:
[`../transformer-energy-to-target-gpu-opennn-vs-pytorch-vs-tensorflow.md`](transformer-energy-to-target-gpu-opennn-vs-pytorch-vs-tensorflow.md).
