# ImageNet-geometry ResNet-50 benchmark — resume notes (paused 2026-06-13)

Goal: run the ResNet-50 training-speed benchmark at **full ImageNet geometry**
(224×224×3 inputs, 1000-class head) to see whether OpenNN's CIFAR lead (1.6×
torch.compile, 2.1× eager — all at 32×32) survives at production resolution.
Expectation going in: it does NOT — at 224² the workload is conv-FLOP-bound,
both engines run the same cuDNN kernels, and the launch-overhead mega-graph
that won CIFAR has no idle time to recover. Likely parity or PyTorch ahead.

## Setup (DONE, committed)

- `prepare_imagenet_like.py [dir] [classes] [size]` — upsamples 50,000 CIFAR-10
  images to 224×224 (bilinear) and lays them across `classes` populated folders
  (label = i % classes) so ResNet-50 gets a real 1000-way head + softmax. No
  ImageNet download/license needed; content is irrelevant to a speed benchmark,
  only the 224×224×3 / 1000-class shape and per-batch disk-decode cost matter.
  Built at `~/opennn-precision/docs/benchmarks/resnet50-training-speed/imagenet_like/`
  (WSL): 50k BMPs ≈ 7 GB on disk, 1000 folders, ~50 imgs/class.
- `pytorch_resnet50_lazy.py [epochs] [batch] [dir] [workers]` — PyTorch
  counterpart with a plain-PIL `BmpFolder` Dataset + DataLoader (num_workers,
  pinned, prefetch) reading the SAME BMP folders. No torchvision (suite
  convention). Now also prints `peak_vram_mib` / `peak_reserved_mib` via
  `torch.cuda.max_memory_allocated/reserved`.
- OpenNN side needs NO code change: `ImageDataset` defaults to lazy `BinaryFile`
  storage (builds a uint8 `images.bin` disk cache once — 32 s here — then reads
  each batch from disk in the background prefetch worker). Run WITHOUT
  `OPENNN_GPU_RESIDENT_DATA` (30 GB of fp32 images can't fit the 6 GB card).

## Findings so far (RTX 3060 Laptop, 6 GB, WSL, fp32)

- Model shape correct: **25,557,040 params** (1000-class ResNet-50; CIFAR-100
  was 23.71M, the +1.8M is the 2048→1000 classifier). Lazy cache builds in 32 s
  and is NOT the bottleneck.
- **Batch 128 @ 224²: OpenNN OOMs** (cudaErrorMemoryAllocation, real, first
  uncontended run). Activations — not data — exceed 6 GB; feature maps are ~49×
  CIFAR's at 32×32.
- **Batch 64 @ 224²: FITS on OpenNN** — clean uncontended run trained at **100%
  GPU compute utilization, ~5.2–5.3 GB peak (85% of card)**. NO throughput
  number captured yet (killed mid-epoch at shutdown). 100% util = pure
  compute-bound regime, the opposite of CIFAR's launch-bound regime.
- An earlier b64 "OOM at device_backend.cpp:280" was a FALSE OOM from running two
  OpenNN processes on the GPU at once — contention, not a real limit. LESSON:
  run GPU memory tests strictly sequentially, verify `nvidia-smi` shows 0 MiB
  between runs; never overlap OpenNN + PyTorch on this 6 GB card.

## NEXT STEPS (in order)

1. **Clean OpenNN sweep**, one batch at a time, GPU idle between each (verify
   0 MiB): `for b in 64 32 16; do OPENNN_CUDA_GRAPH=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib
   ./opennn_resnet50_speed imagenet_like/train 1 $b fp32; done` — record
   samples_per_sec + which fit. (b64 fits; b128 OOMs; 32/16 untested.)
2. **PyTorch ALONE** at the same batches (GPU idle first):
   `~/benchenv/bin/python pytorch_resnet50_lazy.py 1 $b imagenet_like 6` —
   record samples_per_sec AND peak_vram_mib. Key question the user raised: does
   PyTorch fit a LARGER batch than OpenNN? Expected yes — OpenNN's CUDA-graph
   capture PINS every buffer at a fixed address for the whole step (can't free
   activations mid-graph), so its peak footprint is higher than PyTorch eager,
   which frees activations as backward consumes them. That memory cost is the
   real, fixable OpenNN gap at this geometry (see below).
3. Report the clean head-to-head: throughput (expected ≈ parity, compute-bound)
   AND peak VRAM (expected OpenNN > PyTorch). Decide documentation — likely a
   short "at ImageNet geometry the launch-overhead lead fades to parity and the
   graph-memory cost limits batch size" caveat/section, NOT a headline win.

## ROOM FOR IMPROVEMENT (user asked) — what's real vs not

- **Memory, not speed, is the OpenNN gap here.** The b128 OOM is a
  memory-management deficiency, not physics. Two fixable costs: (a) CUDA-graph
  capture pins all buffers (no mid-step free/reuse); (b) no aggressive
  activation liveness/reuse analysis. Fixing these lets OpenNN fit bigger
  batches / run this regime — but at 100% util it lands at PARITY, not ahead.
  Do NOT expect the CIFAR-style win here; there's no idle time to recover.
- **The realistic next lever is bf16 at 224²** (`./opennn_resnet50_speed ...
  bf16`): halves activation memory (likely fits b128) AND ~2× tensor-core
  throughput. This is how ResNet-50 is actually trained at this resolution and
  is a fairer comparison than fp32. Untested at 224² — try after step 1-3.
- Gradient checkpointing (not implemented in OpenNN) is the other memory lever.

## Files (all committed on dev-refactor)
prepare_imagenet_like.py, pytorch_resnet50_lazy.py — this dir.
Dataset lives only in WSL (imagenet_like/, gitignored — 7 GB).
