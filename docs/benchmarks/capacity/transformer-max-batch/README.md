# Transformer max-batch + speed (GPU) — OpenNN vs PyTorch vs TensorFlow

Capacity **and** throughput benchmark for the encoder-decoder Transformer from
*Attention Is All You Need* (paper base **d512 / h8 / ff2048 / 6L**, ~85 M
parameters), sequence-to-sequence. Per framework, per precision (fp32, bf16),
per mode:

- **max batch** — largest batch that fits the VRAM budget
- **speed** — samples/sec

Two modes (`--modes train,infer`):

- **train** — forward + backward + Adam step
- **infer** — forward only: no gradients, no optimizer state (OpenNN
  device-resident path `calculate_outputs_resident` with the logits left on
  the GPU, PyTorch `torch.inference_mode()`, TF `training=False`)

Two supported corpora (`--corpus`, format `source <TAB> target`):

- **Alpaca chat pairs** (`chat_pairs.txt`, 47 k pairs — same data the ChatGPT
  example in `blank_cuda` trains on): the original configuration.
- **WMT14 En-De** (prepare with [`prepare_wmt14.py`](prepare_wmt14.py)): the
  standard dataset the paper's base model was trained and reported on — use
  this configuration for publishable runs. Capacity depends on the corpus
  only through the derived vocab and sequence lengths (WMT sentences are much
  shorter than chat pairs, so every engine's max batch rises; the two corpora
  are **not** comparable to each other).

**Like-for-like execution.** All three engines run op-by-op kernels with no
whole-graph compiler: CUDA graph is OFF for OpenNN (it adds <1 % for the
transformer — big GEMMs, so the launch overhead a graph amortizes is already
negligible), PyTorch runs eager (no `torch.compile`), and **TensorFlow runs
standard graph mode without XLA** (`@tf.function`, `jit_compile` off). XLA is a
whole-graph fusing/rematerializing compiler with no OpenNN or eager-PyTorch
equivalent, so it is kept off for the like-for-like comparison; run it separately
if you want to measure the XLA path.

## Fairness — identical model, identical shapes

For OpenNN-only inference speed investigations, the driver can also enable CUDA
Graph replay with `--opennn-infer-cuda-graph` (or
`OPENNN_TRANSFORMER_INFER_CUDA_GRAPH=1` for the C++ trial). Keep it off for the
eager comparison unless the PyTorch counterpart is also run with an equivalent
CUDA Graph path.

The vocabulary and sequence lengths are derived **once** from the OpenNN corpus
(`input_vocab`, `output_vocab` capped at 30 000, `input_seq`, `decoder_seq`) and
passed to every engine, so all three build the **same 84 843 312-parameter
network**. Max batch runs a fresh process per candidate (OOM-safe) and takes the
largest batch that completes warmup + one real training step inside the VRAM
cap. Speed is samples/sec at a fixed batch after a warmup that every engine
excludes from its timing (OpenNN via a warmup `train()` that caches the cuDNN
plans, PyTorch/TF via warmup steps). All engines run the same plain-Adam
configuration (no weight decay, no gradient clipping).

### Engine settings

| Engine | fp32 | bf16 | execution |
|--------|------|------|-----------|
| OpenNN | fused cuDNN attention | bf16 tensor-core path | op-by-op (CUDA graph off, on purpose) |
| PyTorch | TF32 matmul/cudnn, cudnn.benchmark | `autocast(bfloat16)` | eager; SDPA/flash attention (`nn.Transformer`), fused Adam |
| TensorFlow | TF32 (default) | `mixed_bfloat16` policy | graph mode (`@tf.function`), **XLA off** |

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

# 2a. Alpaca corpus: the ChatGPT example already writes
#     /home/.../datasets/chat/chat_pairs.txt from Stanford Alpaca.
# 2b. WMT14 En-De corpus (standard, publishable):
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/capacity/transformer-max-batch/prepare_wmt14.py

# 3. Run the full comparison (torch + TF live in the ml venv).
#    Alpaca (train + infer):
python docs/benchmarks/capacity/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 \
    --modes train,infer --speed-batch 64
#    OpenNN inference speed mode (not eager like-for-like):
python docs/benchmarks/capacity/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn --precisions fp32,bf16 --modes infer \
    --speed-batch 64 --opennn-infer-cuda-graph
#    WMT14 (standard corpus; archive the artifact):
python docs/benchmarks/capacity/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 \
    --modes train,infer --speed-batch 64 \
    --corpus "$OPENNN_BENCH_DATA/wmt14/wmt14_en_de_pairs.txt" \
    --result-json docs/benchmarks/results/gpu-transformer-max-batch-wmt14.json
```

Environment: PyTorch and TensorFlow are installed in `~/.venvs/ml`. TensorFlow
needs the venv's bundled NVIDIA libs on `LD_LIBRARY_PATH` to see the GPU; the
driver sets this automatically for the TF subprocess only (PyTorch and OpenNN
use their own CUDA runtimes).
