# Transformer / attention benchmarks (OpenNN vs PyTorch vs TensorFlow)

GPU benchmarks for the encoder-decoder Transformer from *Attention Is All You
Need* — token embeddings + positional encoding + N encoder/decoder layers
(multi-head attention + feed-forward + layer norm) + vocab projection. Both
**inference** and **training**, in bf16 and fp32.

## Files

| File | Purpose |
|------|---------|
| `opennn_transformer_resident.cpp` | OpenNN device-resident inference (the fair number) |
| `opennn_transformer_infer.cpp` | OpenNN convenience-API inference (shows per-call overhead) |
| `opennn_transformer_train.cpp` | OpenNN training throughput (fwd + bwd + Adam) |
| `opennn_attention_validate.cpp` | Forward correctness: CPU vs CUDA agreement (also the layer-norm-NaN regression probe) |
| `pytorch_transformer_infer.py` | PyTorch `nn.Transformer` inference counterpart |
| `pytorch_transformer_train.py` | PyTorch training counterpart (reads the same corpus) |
| `tensorflow_transformer_infer.py` | TensorFlow inference counterpart |
| `tensorflow_transformer_train.py` | TensorFlow training counterpart |
| `make_synthetic_corpus.py` | Synthetic tab-separated corpus generator for the training benchmark |
| `run_transformer.py` | 3-way orchestrator for the inference comparison (`gpu-transformer-inference-speed`) |
| `run_transformer_train.py` | 3-way orchestrator for the training comparison (`gpu-transformer-training-speed`) |
| `build.sh` | Portable wrapper: builds the four Transformer CMake targets and symlinks them here |

## How to run

```bash
# Build (all, or name specific basenames). Paths inside build.sh are machine-specific.
./build.sh                          # all four benchmarks
./build.sh opennn_transformer_train # just one

# Inference (args: seq d_model heads ff layers vocab batch iters)
LD_LIBRARY_PATH=/usr/lib/wsl/lib ./opennn_transformer_resident 128 512 8 2048 6 10000 32 50
python pytorch_transformer_infer.py 128 512 8 2048 6 10000 32 50
#   OPENNN_BF16=1 / PT_BF16=1 -> bf16

# Or run the full inference comparison harness:
python run_transformer.py --seqs 128,256,512 --batch 32 --runs 5 --precision bf16

# Forward correctness (RESULT=MATCH, no NaN)
./opennn_attention_validate 128 512 8 2048 6 10000 8

# Training (args: corpus d_model heads ff layers batch epochs)
python make_synthetic_corpus.py corpus.txt 256 256 1024 1234
./opennn_transformer_train corpus.txt 256 8 1024 2 16 20
python pytorch_transformer_train.py corpus.txt 256 8 1024 2 16 20
python tensorflow_transformer_train.py corpus.txt 256 8 1024 2 16 20
#   OPENNN_LR overrides the LR; OPENNN_BF16=1 / PT_BF16=1 / TF_BF16=1 -> bf16

# Or run the full 3-way training comparison harness:
python run_transformer_train.py --runs 5 --precision both
```

## Notes / gotchas

- The OpenNN `Transformer` ctor order is `(input_seq, decoder_seq, input_vocab,
  output_vocab, embedding_dim, heads, ff_dim, layers)`; inputs are Tensor3 token
  ids `(batch, seq, 1)`. The resident inference path reuses the general
  `calculate_outputs_resident` (2-input — encoder + decoder tokens).
- cuDNN flash-attention is bf16-only; the fp32 path casts Q/K/V to bf16, runs the
  fused graph, and casts back (forward and backward), so fp32 also uses fused
  attention.
- Training uses `TextDataset` + `TrainingStrategy` (Adam + token cross-entropy),
  the same path as `examples/translation`.
- Always run GPU benchmarks with `timeout N + CUDA_LAUNCH_BLOCKING=1` (a CUDA error
  in OpenNN can otherwise spin the host for hours).
