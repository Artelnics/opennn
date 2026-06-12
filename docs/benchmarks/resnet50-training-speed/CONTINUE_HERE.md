# ResNet-50 speed work — resume notes (updated 2026-06-12, night)

Goal: close the gap to `torch.compile` (compiled PyTorch 5,772 samples/s;
eager PyTorch 2,080).

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
