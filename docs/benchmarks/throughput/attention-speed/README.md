# Transformer / attention benchmarks (OpenNN vs PyTorch)

GPU benchmarks for the encoder-decoder Transformer from *Attention Is All You
Need* — token embeddings + positional encoding + N encoder/decoder layers
(multi-head attention + feed-forward + layer norm) + vocab projection. Both
**inference** and **training**, in bf16 and fp32. Articles:
[inference](transformer-inference-gpu-opennn-vs-pytorch.md) ·
[training](transformer-training-gpu-opennn-vs-pytorch.md).

## Results (RTX 3060 Laptop, 6 GB)

**Inference, bf16** (paper base shape d512/h8/ff2048/6L), tokens/sec — OpenNN
device-resident (fused flash-attention) vs PyTorch `torch.autocast(bf16)`:

| Config | OpenNN | PyTorch | ratio |
|--------|-------:|--------:|-------|
| seq 128, b 32 | 124,664 | 103,041 | **1.21×** |
| seq 256, b 32 | 141,010 | 102,718 | **1.37×** |
| seq 512, b 32 | 144,080 | 108,440 | **1.33×** |

**Inference, fp32** (via the fp32-via-bf16 fused path), configs that fit in 6 GB:

| Config | OpenNN | PyTorch | ratio |
|--------|-------:|--------:|-------|
| seq 256, b 16 | 76,964 | 74,433 | **1.03×** |
| seq 512, b 16 | 76,498 | 60,734 | **1.26×** |
| seq 512, b 8  | 74,941 | 57,942 | **1.29×** |

**Training, fp32** (fwd + bwd + Adam), samples/sec at d256/h8/ff1024/2L, b16:

| seq | OpenNN | PyTorch | ratio |
|-----|-------:|--------:|-------|
| 256 | 512 | 347 | **1.48×** |
| 384 | 400 | 236 | **1.69×** |

OpenNN wins inference (both precisions) and training, and the training lead grows
with sequence length. Energy per sample is also lower (0.169 vs 0.220 J at seq
256). The wins rest on the fused cuDNN flash-attention path (forward and
backward), the device-resident inference path, and Glorot init for the
Transformer — see the articles.

## Files

| File | Purpose |
|------|---------|
| `opennn_transformer_resident.cpp` | OpenNN device-resident inference (the fair number) |
| `opennn_transformer_infer.cpp` | OpenNN convenience-API inference (shows per-call overhead) |
| `opennn_transformer_train.cpp` | OpenNN training throughput (fwd + bwd + Adam) |
| `opennn_attention_validate.cpp` | Forward correctness: CPU vs CUDA agreement (also the layer-norm-NaN regression probe) |
| `pytorch_transformer_infer.py` | PyTorch `nn.Transformer` inference counterpart |
| `pytorch_transformer_train.py` | PyTorch training counterpart (reads the same corpus) |
| `make_synthetic_corpus.py` | Synthetic tab-separated corpus generator for the training benchmark |
| `build.sh` | Hand-link recipe for all benchmarks (paths machine-specific; edit for your tree) |

## How to run

```bash
# Build (all, or name specific basenames). Paths inside build.sh are machine-specific.
./build.sh                          # all four benchmarks
./build.sh opennn_transformer_train # just one

# Inference (args: seq d_model heads ff layers vocab batch iters)
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_transformer_resident 128 512 8 2048 6 10000 32 50
python pytorch_transformer_infer.py 128 512 8 2048 6 10000 32 50
#   OPENNN_BF16=1 / PT_BF16=1 -> bf16

# Forward correctness (RESULT=MATCH, no NaN)
./opennn_attention_validate 128 512 8 2048 6 10000 8

# Training (args: corpus d_model heads ff layers batch epochs)
python make_synthetic_corpus.py corpus.txt 256 256 1024 1234
./opennn_transformer_train corpus.txt 256 8 1024 2 16 20
python pytorch_transformer_train.py corpus.txt 256 8 1024 2 16 20
#   OPENNN_LR overrides the LR on both sides; OPENNN_BF16=1 / PT_BF16=1 -> bf16
```

## Notes / gotchas

- The OpenNN `Transformer` ctor order is `(input_seq, decoder_seq, input_vocab,
  output_vocab, embedding_dim, heads, ff_dim, layers)`; inputs are Tensor3 token
  ids `(batch, seq, 1)`. The resident inference path reuses the general
  `calculate_outputs_resident` (2-input — encoder + decoder tokens).
- cuDNN flash-attention is bf16-only; the fp32 path casts Q/K/V to bf16, runs the
  fused graph, and casts back (forward and backward), so fp32 also uses fused
  attention.
- Training uses `LanguageDataset` + `TrainingStrategy` (Adam + token cross-entropy),
  the same path as `examples/translation`.
- Always run GPU benchmarks with `timeout N + CUDA_LAUNCH_BLOCKING=1` (a CUDA error
  in OpenNN can otherwise spin the host for hours).
