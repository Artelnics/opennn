# Transformer fixed-work GPU energy: OpenNN vs PyTorch vs TensorFlow

*Last updated 2026-08-10. Linux x86_64, NVIDIA GeForce RTX 4080 (16 GB), driver 595.84, CUDA 13.3, cuDNN 9.23.1. Artifact: [`results/gpu-transformer-energy-fixed-work-20260810T131926Z.json`](../../results/).*

OpenNN spends **25.6 Wh** to train the 84.8M-parameter chat Transformer for 10
epochs where PyTorch spends 34.8 Wh and TensorFlow 39.7 Wh — **1.36× and 1.55×
less electricity for identical work**, finishing the same workload 1.45× and
1.74× sooner.

## The result

Every engine trains the identical encoder-decoder Transformer (d512 / h8 /
ff2048 / 6+6L, vocab 19,443 in / 30,000 out, Stanford Alpaca 47,487 pairs,
token-identical data) for exactly **10 epochs** at batch 128, plain Adam
lr 1e-4, bf16, each on its fastest execution path (OpenNN CUDA graph, PyTorch
autocast + fused Adam + SDPA, TensorFlow mixed_bfloat16 + XLA). GPU board
power is sampled at 20 Hz and trapezoid-integrated over each engine's
`TRAIN_START`/`TRAIN_END` window; the idle baseline is measured fresh and
subtracted for the active figures. Median of 3 runs (seeds 42-44):

| Engine | Energy total | Energy active | Train window | Avg power | µJ / epoch-sample |
|---|---:|---:|---:|---:|---:|
| **OpenNN** | **25.6 Wh** | **23.1 Wh** | **309 s** | 298 W | **194,250** |
| PyTorch | 34.8 Wh | 31.2 Wh | 449 s | 279 W | 263,797 |
| TensorFlow | 39.7 Wh | 35.4 Wh | 537 s | 266 W | 300,851 |

| Comparison | Energy ratio | Time ratio |
|---|---:|---:|
| PyTorch vs OpenNN | 1.36× | 1.45× |
| TensorFlow vs OpenNN | 1.55× | 1.74× |

OpenNN draws the *highest* average power (298 W — it keeps the GPU busiest)
but finishes the same work so much sooner that it spends the least energy.
Run-to-run energy dispersion is under 1% for every engine.

## Why fixed work

An earlier version of this benchmark trained to a quality target (epoch-mean
token CE ≤ 3.5). Runs that missed the gate burned their whole epoch budget and
were discarded, which made the aggregate unstable — at this configuration the
loss trajectory is seed-sensitive in **all three engines** (some seeds park on
the unigram plateau; today's per-run final CE spans 3.2–6.4 for OpenNN, PyTorch
and TensorFlow alike). Fixed work sidesteps that lottery: identical epochs,
identical data and hyperparameters, energy for the work actually done. The
per-epoch loss histories stay in the artifact so convergence equivalence
remains auditable, and the same semantics are used by
[`../higgs-dense-energy/`](../higgs-dense-energy/) and
[`../resnet50-energy/`](../resnet50-energy/).

## Caveats

* GPU board energy only (`nvidia-smi power.draw`, sampled, not a hardware
  joule counter); CPU/DRAM/PSU losses are excluded for every engine equally.
* One-time tokenization, imports and cuDNN plan selection are outside the
  window; warmup, CUDA-graph capture and XLA compilation are inside (they are
  real energy the training pays).
* bf16 headline; the harness measures fp32 with `--precision fp32` if needed.

## Reproducing

```bash
cmake --build build --target opennn_transformer_energy -j
export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
python docs/benchmarks/energy/transformer-energy/prepare_chat.py   # corpus, once
python docs/benchmarks/energy/transformer-energy/run_transformer_energy.py \
    --epochs 10 --batch 128 --lr 1e-4 --runs 3
```
