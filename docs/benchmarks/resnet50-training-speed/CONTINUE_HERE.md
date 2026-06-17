# ResNet-50 speed work — resume notes (updated 2026-06-13)

Goal (MET): beat `torch.compile`. Now 1.6x AHEAD of it.

## Current Full ImageNet Path (2026-06-15)

Use the real ImageNet class-folder tree for the reference Linux run. The
`imagenet_like/` geometry proxy is historical only.

Expected layout:

```text
imagenet/
  train/
    n01440764/
      *.JPEG
    ...
```

Build and run through the JSON harness:

```bash
cmake -S . -B build-gpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOpenNN_BUILD_EXAMPLES=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build-gpu --target opennn_resnet50_speed -j"$(nproc)"

cd docs/benchmarks/resnet50-training-speed
python prepare_imagenet_smoke.py /path/to/imagenet /tmp/imagenet_smoke \
  --classes 4 --images-per-class 4
python run_imagenet_resnet50.py --data /tmp/imagenet_smoke \
  --batches 4 --epochs 1 --runs 1 --engines opennn,pytorch_eager \
  --gpu-index 0
python run_imagenet_resnet50.py --data /path/to/imagenet \
  --batches 64,32,16 --epochs 1 --runs 1 --precision fp32 \
  --engines opennn,pytorch_fast,pytorch_eager \
  --gpu-index 0 --require-gpu-idle \
  --opennn-cache-dir /local_nvme/opennn-image-cache
```

OpenNN builds a uint8 cache on first run. Full ImageNet at 224x224x3 needs
roughly 190 GB, so use local NVMe for `--opennn-cache-dir` when possible.

## CURRENT STANDING: 8,433 median (8,770/8,433/8,233) — BEATS torch.compile 1.6x

Same-session interleaved (fp32 b128 RTX 3060 WSL, 2026-06-13):
OpenNN 8,302/8,228 vs torch.compile 5,268 vs PyTorch eager 3,960.
OpenNN is **1.6x torch.compile, 2.1x eager**. Numerics correct (train error
2.3 → ~1.0 over 3 epochs). Tests at the 87 pre-existing failures throughout.

### THE WIN: resident CUDA-graph mega-launch (5,235 → 8,433, +61%)
The whole last-10% project was a misframe. Per-kernel CUDA-event timing
(OPENNN_GRAPH_TIMING=1, kept in cudnn_frontend_utilities.h) showed OpenNN's
conv KERNELS already cost 13.6 ms/step vs PyTorch's entire 53-conv budget of
17.1 ms — compute was never the gap. The gap was LAUNCH OVERHEAD (~150
launches/step on WSL's slow CUDA API).

The CUDA-graph path mega-batched 8 steps/launch for the staged (host-loaded)
path but only 1 step/launch for the GPU-resident path — which is what the
benchmark uses. Extended the M=8 mega-graph to the resident gather path in
`optimizer.cpp` (run_graph_epoch, new `resident_gather` branch): issue the 8
device gathers on the TRANSFER stream OUTSIDE the graph, capture only the 8
compute steps. One launch per group. That single change: 5,235 → 8,433.
- GOTCHA that cost a rebuild: the gather's internal `record_h2d_done` records
  a CUDA event; if captured INSIDE the graph, the host's later
  cudaEventSynchronize on it throws CUDA error 1. Keep gather + its event
  OUTSIDE the capture window (matches the working single-step pattern).

### RULED OUT this session (measured, reverted)
- Custom hand-written NHWC batch-norm CUDA kernels: correct but 2-5% SLOWER
  than the cudnn-frontend BN graphs (cudnn BN already near roofline at these
  small spatial sizes). Removed.
- Subsampled-view trick for 1x1 stride-2 projection convs (stride-1 conv over
  a strided input view): regressed the strided dgrads 2-4x (defeats cuDNN's
  vectorized NHWC kernels). Reverted.

### Earlier ladder (Phase 1, kernels): medians of 3, fp32 b128 RTX 3060 WSL
2,912 base → 3,842 conv autotune (+30%) → 4,373 residual fwd fusion + fork
bwd (+14%) → 4,824 single-pass skip-join accumulate (+10%) → 4,863 BN
autotune (+1%) → 5,048 biasless convs under BN (+4%) → 5,235 single-step
CUDA graph (+4%) → **8,433 resident mega-graph (+61%)**.

Since the last entry:
- **Fork bwd fusion**: BN bwd graph = BN_infer(X)(+ADD S) → RELU_BWD whose
  output is BOTH real (written straight to the skip's delta slot) AND the
  DBN input ("DADD" fork, sample `BN_inference DRelu DBN` / first backward
  TEST_CASE in samples/cpp/norm/batchnorm.cpp). Per-entry try/catch fallback
  (failed fork build clears the stale tensor attrs, rebuilds plain, manual
  dReLU+copy) so the BN frontend can never be disabled by it.
- **accumulate_output_deltas**: gathers valid sources, then one
  add(s0,s1,dest) (+ chained adds for >2) instead of setZero+add-per-edge.
- **BN graph autotune** via `cudnn_fe::autotune_with_scratch` (times plans on
  throwaway buffers — REQUIRED because BN bwd is in-place and BN fwd updates
  running stats; never autotune those on live data).
- **Biasless convs**: `ConvolutionOp::use_bias=false` when BN follows (BN
  beta absorbs it; matches torchvision bias=False). Kills the per-conv bgrad
  reduction graph (53 graph executions/step) + the fwd bias add. Param count
  drops by exactly the 26,560 biases.
- **OPENNN_CUDA_GRAPH=1 now pays** (+4%) — it was neutral before the step
  became launch-overhead-sensitive; re-test config levers after big changes.

## Measured budgets (the map for the last 10%)

PyTorch bare-conv budget on the exact 53 shapes (pt_conv_budget.py, CUDA
events, channels_last+TF32+benchmark): **fwd 4.35 + bwd 13.2 = 17.5 ms/step**.
Compiled total 22.2 ms/step. Our step at 5,235/s = 24.4 ms.

Our split (profile_split.log scopes, pre-bias-removal, profiled epoch):
op:conv_bwd 10.5 ms (FASTER than torch's 13.2!), op:conv_fwd 6.45 (torch
4.35), **op:bn_fwd 4.3 + op:bn_bwd 4.9 vs ~2.5 roofline** — BN graphs are
the remaining excess; adam 2.0 (fp32 roofline), accumulate ~1.2.

## RULED OUT

- conv+genstats / ConvBNfprop restructure (stats free during conv): the
  runtime-fusion genstats engines are **HALF-I/O ONLY** (see SBRCS sample,
  samples/cpp/convolution/fprop.cpp — io=HALF). Dead for the fp32 protocol.
- NVIDIA_TF32_OVERRIDE (either value), Adam (roofline), CUDA-graph pre-fusion.

## Next leads for the last ~10%

1. **nsys --cuda-graph-trace=node on native Windows** (admin terminal; see
   ../training-speed memory notes) — exact kernel-time map of the captured
   step; identifies whether bn kernels or specific conv shapes hold the rest.
2. **Custom fused BN kernels** (hand CUDA): cudnn's BN engines are ~2x off
   roofline at our small spatial sizes; a hand-written two-pass BN
   (vectorized, fused add+relu, one block per channel) could halve bn time
   (~+8-9%). Write bn_fwd/bn_bwd kernels in kernel_layers.cu and use them
   instead of the frontend graphs (keep graphs as fallback).
3. conv_fwd excess vs torch (6.45 vs 4.35 pre-bias; re-profile post-bias) —
   per-shape comparison via an extended pt_conv_budget + per-shape scopes.

## DONE: block-end residual fusion — 4,312/4,026/3,959, median ~4,026 (+5% on
## top of autotune; spreads don't overlap, so real)

Session total: 2,912 → ~4,026 median (+38%, peak run +48%). Now ~1.9x faster
than PyTorch eager, ~1.43x behind torch.compile. Verified: 58 layers (32
removed), parameter count bit-identical (23,555,088), training trajectory
matches the unfused net. Windows full build green, tests at the 87
pre-existing failures. NOTE the run-to-run GPU variance (~±5%) — always
median of 3.

Implementation:
- `Convolutional::set_residual(true)` + two sources {main, skip}: the
  block-end conv consumes the skip tensor directly; `Addition` and block-end
  `Activation` layers are GONE from `opennn::ResNet`.
- `BatchNormOp::fuse_add`: fwd graph = BN → ADD(residual) → RELU (legacy GPU /
  inference / CPU paths apply add+relu manually). Backward: ActivationOp runs
  its in-place dReLU (backward_fused=false for residual — the skip branch
  needs the materialized post-activation delta), then BatchNormOp copies that
  delta to `residual_delta_slot` (=2) BEFORE its in-place transform; framework
  delta routing (consumer_edges / restore / accumulate) handles the rest.
- `get_backward_specs` for residual conv returns a second output-shaped spec
  (slot 2) → routed to the skip source; projection-skip convs get it aliased
  for free (single consumer), identity skips still pay accumulate.
- Validation in `validate_source_arity` (residual conv must have 2 sources);
  JSON field "Residual".

Remaining un-recovered ceiling (probe said 5,339 with everything skipped):
the per-block dReLU kernel + residual-delta copy, and the identity-skip
accumulate (setZero+2 adds). Next candidate: backward fork fusion —
BN_infer→DRelu(fork: real output to skip slot)→DBN single graph (kills the
separate dReLU AND the copy); needs per-entry build-failure fallback so an
unsupported fork pattern doesn't disable the whole BN frontend.

## DONE 2026-06-12 night: conv engine autotuning — +30%, OpenNN at ~3,850

**3,903/3,842/3,705 samples/s (12.8 s/epoch)** vs ~2,950 before. Now 1.9x
faster than PyTorch eager; remaining gap to torch.compile is 1.48x.

Implementation (`cudnn_frontend_utilities.h` + `convolution_operator.cpp`):
`finalize(..., request_autotune)` builds ALL candidate plans
(`BuildPlanPolicy_t::ALL`); on each conv graph's first execution
`cudnn_fe::autotune_now()` times every plan with a THROWAWAY max-size
workspace, keeps the fastest (frontend `Graph::autotune`), then allocates the
persistent workspace for the winner only. This is the `cudnn.benchmark=True`
equivalent. Conv graphs only (pure → safe to execute repeatedly); BN graphs
deliberately excluded (in-place backward DY==DX would corrupt step-1
gradients if executed ~100x). `OPENNN_CONV_AUTOTUNE=0` disables. GOTCHA: do
NOT pre-allocate `get_autotune_workspace_size()` per graph at build time —
max-over-all-plans for ~90 graphs OOMs the 6 GB card (that bug cost one run).

Probes that settled the diagnosis (all on WSL, fp32 b128):
- `NVIDIA_TF32_OVERRIDE=0` → 745/s (4x slower) → our convs were ALREADY on
  TF32 tensor cores; precision parity with PyTorch, not the gap.
- `NVIDIA_TF32_OVERRIDE=1` → 1,266/s — breaks engine selection, never use.
- `OPENNN_CUDA_GRAPH=1` → 2,680/s pre-autotune, 3,654/s post-autotune —
  capture works (sane training) but never helps; loop is kernel-bound.

## DONE 2026-06-12: BN+ReLU fusion implemented — wall-clock NEUTRAL

The BN fwd+ReLU / BN bwd+dReLU fusion from the original sketch is implemented,
working, and verified engaged — and it bought ~0%: 2,919/2,971/2,906 samples/s
vs 2,912 baseline (17.1 vs 17.2 s/epoch). Keeping it: correct, removes ~41k
kernel launches/epoch (matters more on launch-bound setups, e.g. Windows), and
is the prerequisite for any deeper cross-layer fusion.

**The 19% activation lead in the host-scope profile was launch/wait latency,
not GPU time** — the MLP nsys lesson repeated. After fusion the freed "time"
just moved into the conv scopes. Same artifact: `optim:adam_update_cuda`
dropped 20 ms → 1.96 ms/step *with no optimizer change* (it was absorbing GPU
queue drain; Adam is already a single fused kernel over the flat 23.5M-param
vector — dead lead). Post-fusion profile (fp32 b128, profiling run ~14.2 s/ep):

- fwd:Convolutional 38% + bwd:Convolutional 36% → **~74% conv-kernel-bound**
- bwd:accumulate_output_deltas 5.9%, adam 5.4%, standalone activations ~6.6%
  (27,300 scope entries remain — 53 of 70/step are fused early-returns)

### Implementation (committed to both trees)

- `batch_norm_operator.{h,cpp}`: `BatchNormOp::fuse_relu`; fwd graph appends
  RELU_FWD pointwise (BN output virtual); bwd graph uses the
  **BN_infer→DRelu→DBN pattern** — recomputes the pre-ReLU BN output
  *virtually in-graph* from X/mean/inv_var/gamma/beta, then RELU_BWD(DY, bn_y)
  → batchnorm_backward. GOTCHA: feeding the materialized post-ReLU Y as a
  plain graph input to RELU_BWD is NOT supported ("No valid engine configs",
  `!dact_op->getXDesc()->getIsVirtual()`) and the failure disables the whole
  BN frontend → 1,942 samples/s legacy regression. The virtual-recompute
  pattern is the one in cudnn-frontend `samples/cpp/norm/batchnorm.cpp`
  ("BN_inference DRelu DBN", cuDNN ≥ 8.9.4).
- Legacy/inference/bf16 fallbacks apply ReLU/dReLU manually
  (`activation_forward/backward`) so every path stays correct.
- `activation_operator.{h,cpp}`: `backward_fused` flag, GPU early-return in
  `back_propagate` (mirrors `forward_fused`).
- `convolutional_layer.cpp::update_convolution_operator`: BN+ReLU → BN fusion;
  ReLU w/o BN → conv epilogue fusion (as before). Dense + standalone
  Activation layers untouched.

## Ruled out

- Adam optimizer (single fused kernel, 1.96 ms/step ≈ roofline).
- `NVIDIA_TF32_OVERRIDE=1` env hack: **3x SLOWER** (1,266 samples/s) — breaks
  engine selection; don't retry.
- BN+ReLU and any other pointwise-launch elimination: GPU is conv-bound.

## THE NEXT PROJECT — block-end fusion, ceiling MEASURED at +39%

Post-autotune profile (profile_autotuned.log): conv 67% (bwd 42 + fwd 24.6),
then a pointwise tail: accumulate_output_deltas 7.7% (34,320 calls),
adam 7.1% (2.01 ms/step = fp32 roofline, NOT winnable), standalone Activation
8.5%, Addition 7.2%.

**Skip-probe verdict (2026-06-12 night): disabling AddOp fwd/bwd + standalone
Activation fwd/bwd + accumulate_output_deltas → 5,339 samples/s vs 3,850
baseline (+39%) — within 8% of torch.compile's 5,772.** Unlike the four
previous host-scope mirages, this tail is real cost. (Probe = env-guarded
early returns, garbage numerics, speed-only; reverted, not committed.)

Why it's so expensive — per bottleneck block today: Addition fwd add (1 pass),
block-end Activation bwd copy+dReLU (2 passes), Addition bwd 2 copies,
accumulate setZero+2 adds (3 passes) ≈ 8 full-tensor passes that fusion/
aliasing can mostly remove. Plan:

1. Forward: fuse skip-add + ReLU into conv3's BN forward graph
   (BN→ADD→RELU epilogue, supported cudnn pattern) — kills Addition fwd and
   Activation fwd. Cross-layer: Convolutional needs the skip tensor as an
   extra input slot; Addition/Activation layers become pass-through (slot
   aliasing), like forward_fused but across layers.
2. Backward: dReLU at the block end fuses into the same BN bwd prologue
   (mask from post-ReLU block output); replace AddOp bwd copies and
   accumulate's setZero+add chain with delta-view aliasing where the
   consumers permit (watch in-place BN bwd which transforms delta).
3. Optional after 1-2: autotune BN graphs safely (scratch delta during
   timing) and re-check bwd conv engines with nsys on native Windows.

Expected landing zone if 70-90% of the ceiling is recovered: ~4,900-5,200
samples/s vs torch.compile 5,772.

## Where things run

- WSL repo copy: `~/opennn-precision` (NO .git — synced tree = Windows
  working tree as of 2026-06-12 evening, INCLUDING the operator split and the
  fusion). Build that matters: `build-gpu129` (CUDA 12.9):
  `make -C ~/opennn-precision/build-gpu129 opennn -j12`.
- Benchmark binary is hand-linked (no cmake target):
  `cd ~/opennn-precision/docs/benchmarks/resnet50-training-speed` then
  compile `opennn_resnet50_speed.cpp` with the flags/libs copied from
  `build-gpu129/tests/CMakeFiles/run_tests.dir/{flags.make,link.txt}`
  (c++ -O3 -std=c++20 -flto=auto -march=native -fopenmp, defines
  EIGEN_NO_DEBUG/HAVE_CUDNN_FRONTEND/OPENNN_HAS_CUDA, includes: opennn,
  benchenv cudnn, opennn-wsl eigen + cudnn_frontend; link libopennn.a +
  benchenv libcudnn.so.9 + cuda-12.9 cudart/cublas/cublasLt/nvrtc + libjpeg.a
  + tbb + gomp + stubs/libcuda.so).
- Run: `OPENNN_GPU_RESIDENT_DATA=1 LD_LIBRARY_PATH=/usr/lib/wsl/lib
  ./opennn_resnet50_speed cifar10/train 3 128 fp32`
- The benchmark builds the net via `opennn::ResNet` (standard_networks,
  bottleneck [3,4,6,3]) since 2026-06-12 — verified identical parameter count
  (23,555,088; 90 layers incl. the no-op global avg pool) and throughput
  (2,980 samples/s) vs the old hand-built network.
- Reference logs in that dir: bn.log (2,912 baseline), profile_2026-06-12.log
  (pre-fusion), profile_fused.log (post-fusion), fused1.err (the unsupported
  RELU_BWD-on-input failure).

## State of the Windows working tree (dev-refactor, NOT committed)

- MLP benchmark fixes + results: see ../training-speed/CONTINUE_HERE.md
  (OpenNN 1.5–2.1x faster than PyTorch eager on Windows).
- operators.cpp/h split into one file per operator + BN+ReLU fusion (this
  note). Windows `build-ninja` opennn_speed target green; WSL build green +
  resnet bench re-measured.
- Before committing: full-tree build + test run (user's standing rule).
