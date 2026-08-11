# GPU Transformer max batch: OpenNN vs PyTorch vs TensorFlow

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-08-11. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifacts: [`results/gpu-transformer-max-batch-chat-20260811T123217Z.json`](../../results/) (OpenNN cells, current checkout) and [`results/gpu-transformer-max-batch-chat-20260810T080429Z.json`](../../results/) (PyTorch/TensorFlow cells and the batch-64 speed table — their binaries are unchanged).*

OpenNN trains the 84.8M-parameter *Attention Is All You Need* base Transformer
at batch **497** in bf16 where PyTorch fits **178** and TensorFlow **272** —
**2.79× PyTorch's training batch** — and runs inference at batch **2,015**
versus PyTorch's 951 and TensorFlow's 563 (**2.12×** and 3.58×). Every OpenNN
and PyTorch ceiling is a genuine VRAM limit; TensorFlow's inference ceiling is
an internal INT32 limit it hits long before memory runs out.

## The result

Largest batch that completes warmup plus one real step inside the VRAM budget
(16,376 MiB total − 512 MiB reserve = 15,864 MiB), fresh process per candidate,
exponential growth + binary search to step 1:

| Mode | Precision | OpenNN | PyTorch | TensorFlow | OpenNN / PyTorch | OpenNN / TF |
|---|---|---:|---:|---:|---|---|
| train | fp32 | **252** | 128 | 220 | **1.97×** | 1.15× |
| train | bf16 | **497** | 178 | 272 | **2.79×** | 1.83× |
| infer | fp32 | **1,003** | 435 | 563 † | **2.31×** | 1.78× |
| infer | bf16 | **2,015** | 951 | 563 † | **2.12×** | 3.58× |

OpenNN's cells were re-searched on the 2026-08-11 checkout (the arena-layout
rework moved every ceiling up 1–2% over the 2026-08-10 sweep: 248→252,
490→497, 985→1,003, 1,987→2,015). The frontier carries a few-samples
run-to-run jitter from ambient desktop VRAM, so reproduce via the runner's
search rather than single trials at the exact boundary.

† TensorFlow aborts at batch 564 with an internal error while using only
14.4 GiB of the 15.9 GiB budget: 564 × 127 × 30,000 logits exceed the 2³¹−1
element limit of a 32-bit tensor descriptor. OpenNN hit the **same wall at the
same batch** until 2026-08-10, when the vocabulary softmax was taught to chunk
tensors above INT32_MAX elements ([`tensor_operations.cpp`](../../../../opennn/core/tensor_operations.cpp),
`softmax_gpu`); its inference ceilings above are true out-of-memory boundaries
(peak 15,873 MiB at the max, next batch fails). PyTorch indexes with 64-bit
sizes and never hits the wall.

Throughput at a common batch of 64, from the 2026-08-10 three-way sweep
(samples/s):

| Mode | Precision | OpenNN | PyTorch | TensorFlow |
|---|---|---:|---:|---:|
| train | fp32 | **761** | 655 | 415 |
| train | bf16 | **1,431** | 1,120 | 489 |
| infer | fp32 | 2,347 | **2,856** | 1,809 |
| infer | bf16 | **5,041** | 4,319 | 2,076 |

Unlike the [attention-speed note](../../throughput/attention-speed/transformer-inference-gpu-opennn-vs-pytorch.md),
these speed cells run **eager, like-for-like**: OpenNN with CUDA graph off,
PyTorch eager (no `torch.compile`), TensorFlow graph mode without XLA. PyTorch's
fp32 inference edge here comes from TF32 matmuls; in bf16 — the deployment
precision — OpenNN leads while fitting twice the batch.

## Why capacity differs

All three engines build the identical 84,843,312-parameter network from the
same corpus-derived shape (input vocab 19,443, output vocab 30,000, input seq
64, decoder seq 127 — Stanford Alpaca chat pairs). What differs is activation
memory: OpenNN plans a single arena with lifetime-aware placement (largest-first
for inference), so the same VRAM holds more concurrent activations than
PyTorch's caching allocator or TensorFlow's BFC allocator manage for this graph.

## Setup

| | Value |
|---|---|
| Model | encoder-decoder Transformer d512 / h8 / ff2048 / 6+6L, 84,843,312 parameters |
| Corpus | Alpaca chat pairs (47k), `source<TAB>target`; vocab and sequence lengths derived once by OpenNN and passed to every engine |
| Budget | total VRAM − 512 MiB reserve; peak sampled via `nvidia-smi` at 20 Hz |
| Search | fresh process per candidate; exponential then binary to step 1; max batch = largest batch completing warmup + 1 real step (train: fwd+bwd+Adam; infer: forward only) |
| Execution | like-for-like eager: OpenNN CUDA graph off, PyTorch eager + SDPA + fused Adam, TensorFlow graph mode XLA off |

Hardware/software: RTX 4080 (16 GB, driver 595.84), i9-12900K, Linux x86_64;
OpenNN commit `c63275648` (OpenNN cells; PyTorch/TensorFlow cells from the
2026-08-10 sweep at `52e21e15d`), g++ 13.3, CUDA 13.3,
cuDNN 9.23.1; PyTorch 2.13.0+cu130, TensorFlow 2.21.0, CPython 3.12.3.

## Caveats

* Capacity depends on the corpus only through the derived vocab and sequence
  lengths. WMT14 sentences are much shorter than chat pairs, so every engine's
  max batch rises there; numbers from different corpora are not comparable.
* The speed cells are eager like-for-like; each engine's *fastest* path
  (CUDA graph / torch.compile / XLA) is measured in the throughput notes, not
  here.
* Single consumer desktop GPU; ceilings scale with VRAM, ratios are the claim.

## Reproducing

```bash
cmake --build build --target opennn_transformer_maxbatch_trial -j
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/energy/transformer-energy/prepare_chat.py   # corpus, once
python docs/benchmarks/capacity/transformer-max-batch/run_transformer_maxbatch.py \
    --engines opennn,pytorch,tensorflow --precisions fp32,bf16 \
    --modes train,infer --speed-batch 64
```
