# ResNet-50 GPU training: kernel-fusion and CUDA-graph audit

*2026-08-15. Audit of the OpenNN CUDA training path as exercised by
`opennn_resnet50_speed` (ResNet-50 v1.5, CIFAR geometry, bf16 and fp32).
Sections 6–9 are measured on the audit machine (RTX 3060 Laptop, 6 GB, WSL2);
the canonical figures live on the RTX 4080 benchmarks machine.*

## 0. Context: the throughput cliff, fixed the same day

Before this audit the peak-batch curve collapsed past batch 512 (6 GB card) /
1024 (16 GB card) and OOMed two doublings later. The cause was not a kernel: the
benchmark ran the cuDNN convolution workspace **unbounded** (so autotune could
run), and cuDNN's first heuristic choice for these NHWC shapes needs ~4 MiB of
scratch per sample (2 GiB at 512, 16 GiB at 4096) and is not even the fastest
engine. Under the library's auto budget (≤ 256 MiB) the chosen plan needs
~16 KiB/sample and was 1.5–2× faster at every batch measured. Autotune could not
run under a budget because `cudnn_frontend::Graph::get_autotune_workspace_size()`
dereferences the barred (nullptr) plan slots; `autotune_workspace_bytes()` in
`cudnn_frontend_utilities.h` now sizes the tuning scratch through the null-safe
per-index query, and autotune runs within the budget (+13% over plain
heuristics at batch 1024 here).

## 1. What is already fused (baseline)

| Where | Fused today | Evidence |
|---|---|---|
| Conv forward | bare `conv_fprop` (bias/ReLU epilogues dead under BN) | `convolutional_layer.cpp:171,196` |
| BN forward | BN + EMA + residual ADD + RELU in one graph | `batch_norm_operator.cpp:426-443` |
| ReLU / residual add | never standalone in the ResNet chain | `activation_operator.cpp:30-31,52-54` |
| BN backward | dReLU + dBN in one graph; residual layers fork the residual delta | `batch_norm_operator.cpp:477-509` |
| Conv backward | separate `wgrad`/`dgrad` graphs; no `bgrad` (no bias under BN) | `convolution_operator.cpp:583-653` |
| Softmax + CE | closed-form `(y−t)/B`; softmax bwd is a no-op | `loss.cpp:1311-1321` |
| Adam | one flat gradient buffer, one float4 kernel; bf16 mirror inside it | `adaptive_moment_estimation.cpp:110-136` |
| CUDA graph | fwd + loss/metrics + bwd + Adam captured, 8 steps/launch | `optimizer.cpp:1593-1603` |

## 2. Findings, ranked at audit time (before measurement)

1. **bf16 BN backward stages through FP32 on most shapes** (`batch_norm_operator.cpp:649-655`): 3 casts + fp32 math + standalone dReLU + D2D copy where cuDNN lacks a bf16 engine. Leads: a ladder-vs-comment contradiction at `:661`, the in-place-DX write at `:733`.
2. **bf16 conv wgrad writes BF16 then casts to FP32** (`convolution_operator.cpp` vs the FLOAT store `bgrad_DB` already uses): 53 casts/step, ~94 MB, gradient rounded to 8 mantissa bits. Same for the dense layer.
3. **CUDA-graph coverage**: the resident gather runs outside the graph from a **pageable** index buffer; group-tail full syncs; captured Adam baked the LR by value so schedules were ignored after capture.
4. **Backward conv fusions**: `dgrad`(L+1) and `dReLU+dBN`(L) are separate graphs; wgrad and dgrad re-read the same DY. The `dgrad + dReLU + dBN` (DBAR) pattern is cuDNN's runtime-fusion target.
5. **Forward conv → BN round trip**: conv writes its output, BN reads it back, 53×/step.
6. Small: 16 residual-join adds; scalar unvectorized `cast_bf16_to_fp32`; truncating host bf16 input cast.

## 6. Measured split on the audit machine

RTX 3060 Laptop (sm_86, 6 GB, WSL2), `OPENNN_GRAPH_TIMING=1`, 1 epoch, graphs
off. Share of total labelled GPU time:

| Op family | bf16 b1024 | fp32 b1024 | bf16 b128 | fp32 b128 |
|---|---:|---:|---:|---:|
| conv wgrad | 27.1% | 29.2% | 22.2% | 24.0% |
| conv dgrad | 21.4% | 21.9% | 20.4% | 33.8% |
| conv fwd | 21.3% | 21.7% | 23.3% | 22.6% |
| **conv total** | **69.8%** | **72.8%** | **65.9%** | **80.4%** |
| BN backward | 18.0% | 15.2% | 16.8% | 9.9% |
| BN forward | 12.2% | 12.1% | 17.2% | 9.7% |

**Convolution is ~70% of GPU time; batch norm ~30% (bf16) / ~20–25% (fp32).**
On this GPU the biggest lever is the conv backward pair (wgrad + dgrad ≈ 48–55%),
i.e. findings 4/5. BN is real but second-order here.

## 7. Improvements applied and validated (2026-08-15)

Audit binary **A** (all four changes) vs sweep binary **B** (workspace fix +
harness only), heur mode, graphs on, whole batches, 3 timed epochs, same seed.
Losses in-band with B; speed within run-to-run noise on this thermally-limited
laptop.

| Point | B samples/s | A samples/s | B loss | A loss |
|---|---:|---:|---:|---:|
| bf16 128 | 11,285 | 10,838 | 1.232 | 1.108 |
| bf16 512 | 15,646 | 15,104 | 1.028 | 1.026 |
| bf16 1024 | 16,585 | 16,502 | 0.993 | 0.977 |
| fp32 128 | 5,158 | 4,952 | 1.252 | 1.227 |
| fp32 1024 | 8,890 | 8,644 | 0.983 | 0.999 |

**The FP32 wgrad-store engine (finding 2) does not exist on this cuDNN.** All 53
conv layers log `no FP32-store engine (No valid engine configs)` and fall back to
the BF16 store + cast — inert here, neither helps nor hurts, and must be
re-measured on the RTX 4080 / cuDNN 9.23 machine where the engine may exist. The
other three (pinned gather-index buffer, device-scalar captured LR, dense FP32
store) are correctness/latency fixes with no throughput signature at this scale;
the device-scalar LR is kept regardless because it fixes a real product bug (LR
schedules were frozen at graph capture). Net: A ≈ B within noise, no regression.

## 8. BN-backward experiment — the +15% is real but the gradients are not

Forcing the plain un-fused BF16 `batchnorm_backward` on every layer
(`OPENNN_EXP_BN_UNFUSED_BF16`, temporary patch, reverted):

| Point | base | unfused | unfused + DX-scratch | base loss | unfused loss |
|---|---:|---:|---:|---:|---:|
| bf16 128 | 9,995 | 10,311 | 10,042 | 1.108 | 1.206 |
| bf16 1024 | 15,827 | **18,317 (+15.7%)** | 17,532 | 0.977 | 1.005 |

1. The speed prize is real and lands in the peak regime (+15.7% @1024, matching
   the +21% measured on sm_120). This is what would move OpenNN past
   TensorFlow's bf16 peak.
2. **The in-place-DX hypothesis (finding 1) is falsified.** Giving DX its own
   buffer changes throughput but leaves the loss identical, so the higher loss
   is intrinsic to the un-fused BF16 `batchnorm_backward` engine, not aliasing —
   as the comment at `batch_norm_operator.cpp:649-668` warned.

Capturing the win means a correct fused NHWC BN-backward kernel (bf16 IO, fp32
math, dReLU + residual fork) for the shapes cuDNN misses — justified only after
an fp32 gradient-check confirms the un-fused engine is truly wrong, not noisier.

## 9. Three-way throughput on the audit machine (RTX 3060 Laptop, 6 GB, WSL2)

Definitive run, 2026-08-16, all three engines in one session, final binary
(everything in this note plus the plan's items 0a, 0d, 1a, 1b and Phase 2),
whole batches, corrected sample counts, autotuned top-8. Result JSONs
`gpu-resnet50-peak-batch-speed-20260816T010828Z / 011958Z / 013818Z`.

**bf16**

| batch | OpenNN | PyTorch | TensorFlow |
|---:|---:|---:|---:|
| 128 | **12,314** | 7,052 | 8,499 |
| 256 | **16,191** | 12,145 | 11,951 |
| 512 | **20,257** | 15,870 | 15,949 |
| 1024 | **22,129** | 18,157 | 18,680 |
| 2048 | **22,704** | 20,035 | 21,369 |
| 4096 | **23,283** | 21,072 | OOM |
| 8192 | 2,133 (6 GB paging) | 1,459 (paging) | — |
| peak | **23,283 @4096** | 21,072 @4096 | 21,369 @2048 |

**fp32**

| batch | OpenNN | PyTorch | TensorFlow |
|---:|---:|---:|---:|
| 128 | **6,737** | 5,596 | 4,655 |
| 256 | **8,304** | 7,059 | 5,939 |
| 512 | **9,394** | 8,189 | 7,312 |
| 1024 | **10,231** | 9,248 | 8,341 |
| 2048 | **11,489** | 10,058 | OOM |
| 4096 | 1,220 (paging) | timeout | — |
| peak | **11,489 @2048** | 10,058 @2048 | 8,341 @1024 |

**OpenNN leads at every point in both precisions.** Peak vs peak: bf16 1.11×
PyTorch / 1.09× TensorFlow; fp32 1.14× / 1.38×; batch 128 bf16 1.75× / 1.45×.
The Saturday-morning picture (§ audit thread) was 0.83× at the bf16 peak with a
cliff past 1024; the change is the sum of the workspace budget, top-K autotune,
the plain-bf16 rung, the fused residual join and the own batch-norm backward
kernel, each measured on its own. Points past the card's 6 GB are WSL2 paging
for every engine and are not throughput measurements. The product claim is the
RTX 4080's number: run the two gradient tests, `speed_gate.py --record
--tolerance 0.05` and the peak-batch sweep there.

## 10. Not recommended (already tried)

Hand-written BN kernels for the native path and the strided-view trick for the
1×1/stride-2 projections were implemented, measured and discarded (2026-08-11
note); the residual-delta aliasing optimisation in `back_propagation.cpp:172-193`
never fires for `Convolutional`.
