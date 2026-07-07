# Training-speed work — resume notes (updated 2026-06-12)

Goal: make OpenNN GPU training **faster than PyTorch** on the Neural Designer
training-speed benchmark (MLP 1000→1000→1, tanh, Adam, MSE, batch 1000,
Rosenbrock). Branch: `dev-refactor`.

## RESOLVED 2026-06-12: OpenNN wins ~1.5–2.1× on Windows

Timeline analysis of the existing nsys capture (post-processed to sqlite, no
new capture needed — see `analyze_gaps2.py` / `analyze_gap_apis.py`) showed the
GPU step loop was already **95% busy (~28 us idle/step)**; roughly **half the
benchmark wall time was host-side dataset scale/unscale inside the timer**
(~3.3 s per timed train() at 200k: `set_scaling()` descriptives + serial
per-feature scale pass, the end-of-train unscale, and the residency re-upload
the scaling forces). PyTorch's script trains on raw data and uploads before
its timer. Two fixes (working tree, 2026-06-12):

1. `tabular_dataset.cpp`: `#pragma omp parallel for` on the per-feature loops
   in `scale_features` / `unscale_features` / `scale_data` (903k → 952k with
   scaling kept).
2. `opennn_speed.cpp`: `dataset.set_variable_scalers("None")` — protocol
   parity with the PyTorch/TF scripts (raw data). Override with
   `OPENNN_BENCH_SCALERS=1` to restore scaling for A/B runs. Training error
   prints ~1.1e9 — that is MSE on raw Rosenbrock targets, expected.

### Current standing (RTX 3060, 6 GB, bf16, samples/sec, Windows native, 2026-06-12)

| Dataset | OpenNN (bf16+graph) | PyTorch eager bf16 | Ratio |
|---|---|---|---|
| 200k | **~1.55M** (1.552/1.546/1.542M) | 731k | **2.1×** |
| 500k | **~1.49M** (1.467/1.488/1.556M) | 962k | **1.5×** |

Steady epochs ~102 ms @200k; remaining per-step cost is `step:wait_fill`
(prefetch worker latency, ~90% of host loop — GPU still 95% fed). The old
gather/Adam/activation kernel leads are all dead ends (see memory notes);
the engine loop needs nothing for this benchmark.

Beware one-off slow runs right after a long build (GPU busy/power state gave
two anomalous 456k readings); always take 3 runs.

WSL/torch.compile standing unchanged: PyTorch compile ~982k @200k — now also
beaten by the fixed benchmark (untested on WSL since the fix; re-measure when
needed).

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
opennn_speed.exe <csv> 1000 30 1000 bf16
```
(GPU-resident data and CUDA graphs are now built into `opennn_speed.cpp` via
`StorageMode::GPUPersistantData` and `set_cuda_graph(true)`; no env flags.)
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
