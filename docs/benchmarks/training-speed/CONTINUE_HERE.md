# Training-speed work — resume notes (paused 2026-06-09)

Goal: make OpenNN GPU training **faster than PyTorch** on the Neural Designer
training-speed benchmark (MLP 1000→1000→1, tanh, Adam, MSE, batch 1000,
Rosenbrock). Branch: `dev-refactor`. Last commit at pause: `d48831348`.

## Where we are

All optimizations committed and pushed (opt-in, default path untouched):
- GPU-resident gather (`OPENNN_GPU_RESIDENT_DATA=1`) — kills per-step host fill.
- CUDA graph capture/replay (`OPENNN_CUDA_GRAPH=1`) — capturable Adam, worker
  pipeline bypass.
- Per-operator profiling scopes (under `OPENNN_PROFILE=1`).
- `pytorch_speed.py` gates `torch.compile` behind `PT_COMPILE` (=0 → eager).

### Current standing (RTX 3060, 6 GB, bf16, samples/sec)

| Platform | OpenNN (bf16+graph) | PyTorch (bf16) | Notes |
|---|---|---|---|
| **WSL2 Linux** | ~815k | ~982k (compile) / ? eager | PyTorch wins via torch.compile |
| **Windows native** | ~775k (500k) / ~902k (200k) | ~770k median (500k, eager) / ~919k (200k) | **TIED**; compile unavailable on Windows |

**Key finding:** the WSL ~1.2× PyTorch lead is *entirely* `torch.compile`'s
kernel fusion. On **Windows, torch.compile does not work** (no Triton
toolchain), so PyTorch runs eager and OpenNN's CUDA graph pulls it to a tie.
Latest 500k Windows bf16 (3 runs each): OpenNN 759/775/792k (steady);
PyTorch eager 800/933/575k (noisy, median ~800k). Essentially even.

**Profiling conclusion (WSL):** OpenNN's individual kernels are competitive or
FASTER than PyTorch (Adam, activation; GEMM at parity). The remaining gap is
distributed per-step overhead, not one hotspot. The per-batch **gather costs
~7%** (measured: 805k→861k with gather skipped) — removing it is the clearest
lever to push OpenNN clearly ahead (~960k → past PyTorch).

## NEXT STEP (the plan when resuming)

**Push OpenNN clearly past PyTorch on Windows.** Highest-leverage:
1. **Eliminate / cheapen the per-batch gather** (~7%). It copies ~4 MB/batch
   the GPU already has. Options: zero-copy contiguous-batch path (point layer
   input at `data_device + offset`, no copy) — needs layers to accept an
   external device pointer + handling shuffle. Or on-device shuffle once/epoch.
2. Re-measure Windows bf16-vs-bf16 (3+ runs, median) to confirm OpenNN > PyTorch.
3. Note: PyTorch tf32 eager (~1060k @200k) still beats OpenNN bf16 — but that's
   cross-precision. Fair fights are bf16-vs-bf16 and tf32-vs-tf32 (OpenNN fp32 =
   TF32 internally, but its fp32 runs are noisy ~555–725k — investigate).

## How to run on Windows (the toolchain that finally works)

The VS-2026 CMake generator is **locked to CUDA 13.2**, but the driver
(555.85) only supports **CUDA 12.5** → 13.2 binaries report "no GPU detected".
The fix is the **Ninja generator + CUDA 12.5 + vcvars** (NOT the VS generator).

Configure + build batch files live at `%TEMP%`; recreate if gone:

```bat
:: configure  (run from a plain cmd)
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA125=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5"
set "PATH=%CUDA125%\bin;%PATH%"
cd /d C:\Users\Roberto\Documents\opennn
cmake -S . -B build-ninja -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="C:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe" ^
  -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_EXAMPLES=ON ^
  -DCMAKE_CUDA_COMPILER="%CUDA125%/bin/nvcc.exe" -DCUDAToolkit_ROOT="%CUDA125%" ^
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler ^
  -DCUDNN_INCLUDE_DIR="C:/Program Files/NVIDIA/CUDNN/v9.20/include/12.9" ^
  -DCUDNN_LIBRARY="C:/Program Files/NVIDIA/CUDNN/v9.20/lib/12.9/x64/cudnn.lib"
:: build
cmake --build build-ninja --target opennn_speed
```

Binary: `build-ninja\bin\opennn_speed.exe`. At runtime put the 12.5 + cuDNN
DLLs on PATH:
`set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin;C:\Program Files\NVIDIA\CUDNN\v9.20\bin\12.9;%PATH%"`

### Run OpenNN (Windows)
```
set OPENNN_GPU_RESIDENT_DATA=1
set OPENNN_CUDA_GRAPH=1
opennn_speed.exe <csv> 1000 30 1000 bf16
```
Datasets: `%TEMP%\opennndata\rb_200k_1000.csv`, `rb_500k_1000.csv` (regen with
`docs\benchmarks\capacity\generate_rosenbrock.exe 1000 <samples> <out> 1234`).

### Run PyTorch (Windows)
venv at `%TEMP%\winbench` (torch 2.6.0+cu124, sees the GPU). Eager mode:
```
set PT_COMPILE=0
%TEMP%\winbench\Scripts\python.exe docs\benchmarks\training-speed\pytorch_speed.py 500000 1000 30 1000 bf16
```

## WSL (still fully works, for Linux numbers + torch.compile)
- OpenNN build: `~/opennn-wsl/build-gpu/bin/opennn_speed`, run with
  `LD_LIBRARY_PATH=/usr/lib/wsl/lib` (WSL CUDA driver shim must win).
- venv `~/benchenv` (torch 2.6.0+cu124, compile works after python3-dev).
- TF needs `LD_LIBRARY_PATH` to its bundled nvidia libs (see run_speed.sh).
- nsys does NOT work under WSL (GPU-PV blocks CUPTI). Works native-Windows.

## Gotchas
- Windows driver 555.85 = CUDA 12.5 max. Build with 12.5, not 13.2.
- VS generator → 13.2 (broken); Ninja+vcvars → 12.5 (works).
- torch.compile: works WSL (with python3-dev), NOT Windows.
- WSL needs `.wslconfig` memory=12GB (set) to avoid OOM crashes at 1M samples.
