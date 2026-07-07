# Transformer max-batch + speed (GPU) — OpenNN vs PyTorch vs TensorFlow

Capacity **and** throughput benchmark for the encoder-decoder Transformer from
*Attention Is All You Need* (paper base **d512 / h8 / ff2048 / 6L**, ~85 M
parameters), sequence-to-sequence. Per framework, per precision (fp32, bf16),
per mode:

- **max batch** — largest batch that fits the VRAM budget
- **speed** — samples/sec

Two modes (`--modes train,infer`):

- **train** — forward + backward + Adam step (the original benchmark)
- **infer** — forward only: no gradients, no optimizer state (OpenNN
  device-resident path `calculate_outputs_resident` with the logits left on
  the GPU, PyTorch `torch.inference_mode()`, TF `training=False`)

Two supported corpora (`--corpus`, format `source <TAB> target`):

- **Alpaca chat pairs** (`chat_pairs.txt`, 47 k pairs — same data the ChatGPT
  example in `blank_cuda` trains on): the original configuration, kept for
  regression continuity with the measured numbers below.
- **WMT14 En-De** (prepare with [`prepare_wmt14.py`](prepare_wmt14.py)): the
  standard dataset the paper's base model was trained and reported on — use
  this configuration for publishable runs. Capacity depends on the corpus
  only through the derived vocab and sequence lengths (WMT sentences are much
  shorter than chat pairs, so every engine's max batch rises; WMT14 numbers
  are **not** comparable to the Alpaca table below).

**Like-for-like execution.** All three engines run op-by-op kernels with no
whole-graph compiler: CUDA graph is OFF for OpenNN (it adds <1 % for the
transformer — big GEMMs, so the launch overhead a graph amortizes is already
negligible), PyTorch runs eager (no `torch.compile`), and **TensorFlow runs
standard graph mode without XLA** (`@tf.function`, `jit_compile` off). XLA is a
whole-graph fusing/rematerializing compiler with no OpenNN or eager-PyTorch
equivalent; with it, TF previously measured max batch 196 fp32 / 258 bf16 and
669/963 samples/s — those numbers are kept below for reference.

## Fairness — identical model, identical shapes

For OpenNN-only inference speed investigations, the driver can also enable CUDA
Graph replay with `--opennn-infer-cuda-graph` (or
`OPENNN_TRANSFORMER_INFER_CUDA_GRAPH=1` for the C++ trial). Keep it off for the
archived eager comparison tables unless the PyTorch counterpart is also run with
an equivalent CUDA Graph path.

The vocabulary and sequence lengths are derived **once** from the OpenNN corpus
(`input_vocab`, `output_vocab` capped at 30 000, `input_seq`, `decoder_seq`) and
passed to every engine, so all three build the **same 84 843 312-parameter
network**. Max batch runs a fresh process per candidate (OOM-safe) and takes the
largest batch that completes warmup + one real training step inside the VRAM
cap. Speed is samples/sec at a fixed batch after a warmup that every engine
excludes from its timing (OpenNN via a warmup `train()` that caches the cuDNN
plans, PyTorch/TF via warmup steps).

### Engine settings

| Engine | fp32 | bf16 | execution |
|--------|------|------|-----------|
| OpenNN | fused cuDNN attention | bf16 tensor-core path | op-by-op (CUDA graph off, on purpose) |
| PyTorch | TF32 matmul/cudnn, cudnn.benchmark | `autocast(bfloat16)` | eager; SDPA/flash attention (`nn.Transformer`), fused Adam |
| TensorFlow | TF32 (default) | `mixed_bfloat16` policy | graph mode (`@tf.function`), **XLA off** |

## Results (RTX 4080, 16 GB) — Alpaca corpus, train mode

The measured table below is the original configuration: **Alpaca chat corpus,
training mode**. The inference mode and the WMT14 corpus have a ready harness
but no archived run yet — re-measure before quoting either.

Speed is steady-state training samples/sec at batch 64, sustained over 160
steps. For OpenNN it is measured by two-point differencing (40- vs 160-step
runs): OpenNN's `wall_s` covers the whole `train()` call — including a fixed
in-window warmup and teardown (parameter D2H) that PyTorch/TF exclude by timing
only their bare step loop — and differencing cancels that fixed cost exactly.
An earlier single-run 40-step measurement charged that overhead to OpenNN and
understated it (fp32 621, bf16 1041); nsys confirms the steady-state numbers
(OpenNN GPU-busy ≈ 89 ms/step fp32 at ~99 % GPU utilization — the gap to the
others is real kernel work, not host overhead).

| Framework | max batch fp32 | max batch bf16 | speed fp32 (samples/s) | speed bf16 (samples/s) |
|-----------|---------------:|---------------:|-----------------------:|-----------------------:|
| OpenNN     | 214 | **423** | **719** | **1310** |
| PyTorch    | 128 | 178 | 613 | 1067 |
| TensorFlow | **224** | 256 | 408 | 482 |

OpenNN capacity includes the July 2026 memory work (warmup OptimizerData reuse,
host-side warmup snapshots, the in-place softmax+CE output delta that removed
the largest delta_pool entry, and the shared transient-scratch block that
collapses the 18 attention head-staging tensors into one): fp32 161 → 214,
bf16 317 → 423 at unchanged speed. The TensorFlow fp32 max batch was
re-measured alongside (its allocator self-caps at ~14.4 GB, so it is largely
insensitive to desktop VRAM; 215 vs 224 across runs is its own variance).

For reference, TensorFlow **with XLA** (whole-graph compiler, measured earlier):
max batch 196 fp32 / 258 bf16, speed 669 fp32 / 963 bf16 samples/s.

All three engines run the same plain-Adam configuration (no weight decay, no
gradient clipping — OpenNN's former L2 + clip-norm defaults cost ~3 % fp32 /
~6 % bf16 and were changed to match the PyTorch/TF convention).

**Reading it.** **OpenNN has the highest throughput in both precisions**
(fp32: +18 % vs PyTorch, +77 % vs TF; bf16: +23 % vs PyTorch, +172 % vs TF) —
and that is while still computing train-accuracy (argmax over the 30 k vocab
inside the loss) every step, which the others skip. Without XLA, TensorFlow's
speed collapses (408/482 — XLA was worth +64 % fp32 / +100 % bf16 to it; its
unfused `mixed_bfloat16` graph is barely faster than its fp32 one), while its
**fp32 max batch actually rises to 215** — so TF's fp32 capacity lead comes
from its runtime's buffer reuse, not from XLA (XLA costs it ~10 % capacity).
**OpenNN wins bf16 max batch (423 — 1.65× TF, 2.38× PyTorch)** because its
bf16 activations are truly 16-bit and it fills VRAM to the cap (TF's allocator
tops out ~14.4 GB, leaving ~1.4 GB unused). PyTorch has the smallest capacity
in both. OpenNN's bf16/fp32 batch ratio (1.97×) is the best. Improvement
target for OpenNN: fp32 activation memory — plain graph-mode TF fits 34 % more
fp32 samples in 9 % less VRAM, so there is real headroom in fp32 buffer reuse;
speed is already ahead everywhere.

## Files

| File | Purpose |
|------|---------|
| `opennn_transformer_maxbatch_trial.cpp` | OpenNN trial: one (mode, batch, precision) attempt, CUDA graph off |
| `pytorch_transformer_maxbatch.py` | PyTorch counterpart (TF32 / autocast bf16 / SDPA / fused Adam / optional `torch.compile`) |
| `tensorflow_transformer_maxbatch.py` | TensorFlow/Keras counterpart (`mixed_bfloat16`, graph mode, XLA off) |
| `run_transformer_maxbatch.py` | Driver: derives the shared model shape, binary-searches max batch per mode, measures speed, prints the summary, optional `--result-json` |
| `prepare_wmt14.py` | Downloads WMT14 En-De (Stanford NMT preprocessed) and writes the `source<TAB>target` pair file under `OPENNN_BENCH_DATA` |

## How to run

```bash
# 1. Build the OpenNN trial (registered in docs/benchmarks/CMakeLists.txt).
cmake --build build --target opennn_transformer_maxbatch_trial -j

# 2a. Alpaca corpus (regression continuity): the ChatGPT example already
#     writes /home/.../datasets/chat/chat_pairs.txt from Stanford Alpaca.
# 2b. WMT14 En-De corpus (standard, publishable):
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/transformer-max-batch/prepare_wmt14.py

# 3. Run the full comparison (torch + TF live in the ml venv).
#    Alpaca (train-mode regression vs the table above + new infer axis):
python docs/benchmarks/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 \
    --modes train,infer --speed-batch 64
#    OpenNN inference speed mode (not eager like-for-like):
python docs/benchmarks/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn --precisions fp32,bf16 --modes infer \
    --speed-batch 64 --opennn-infer-cuda-graph
#    WMT14 (standard corpus; archive the artifact):
python docs/benchmarks/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 \
    --modes train,infer --speed-batch 64 \
    --corpus "$OPENNN_BENCH_DATA/wmt14/wmt14_en_de_pairs.txt" \
    --result-json docs/benchmarks/results/gpu-transformer-max-batch-wmt14.json
```

Environment: PyTorch and TensorFlow are installed in `~/.venvs/ml`. TensorFlow
needs the venv's bundled NVIDIA libs on `LD_LIBRARY_PATH` to see the GPU; the
driver sets this automatically for the TF subprocess only (PyTorch and OpenNN
use their own CUDA runtimes).
