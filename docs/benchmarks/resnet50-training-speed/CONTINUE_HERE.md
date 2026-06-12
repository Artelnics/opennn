# ResNet-50 speed work — resume notes (updated 2026-06-12, evening)

Goal: close the 2x gap to `torch.compile` (OpenNN ~2,930 samples/s vs compiled
PyTorch 5,772; eager PyTorch 2,080 already beaten).

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

## Next leads (in order)

1. **nsys kernel-level profile, native Windows** (nsys broken under WSL):
   need the resnet bench built on Windows (no cmake target — hand-link like
   WSL, see below) + cifar10 copied from WSL. Question to answer: are our
   fp32 conv engines TF32 tensor-core kernels (names like
   `sm80_xmma_*tf32f32*`) like PyTorch's, or strict-fp32 ones? PyTorch
   defaults `cudnn.allow_tf32=True` for convs; our frontend graphs request
   FLOAT compute (TF32 eligibility is engine-internal; this frontend version
   has no TF32 datatype/knob — only numeric-note filtering).
2. **Engine autotuning**: PyTorch sets `cudnn.benchmark=True` (times engines,
   picks best); our `finalize()` takes heuristic-A first choice
   (`HEURISTICS_CHOICE`). Try `HeurMode_t::B` and/or frontend autotune API.
3. **accumulate_output_deltas** (5.9%, 33,930 calls/epoch) — skip-connection
   delta accumulation; could fold the add into conv dgrad epilogue or the
   fused BN bwd graph (cross-layer, invasive).

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
