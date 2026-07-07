"""PyTorch baseline-memory benchmark.

Construct the minimum PyTorch objects used by a training application
(empty TensorDataset, tiny nn.Module, loss, optimizer), then report:
  - baseline_ram_mb: current resident set size for the process
  - gpu_ready_vram_mb: nvidia-smi-reported GPU memory for this process after
    one tiny CUDA matrix multiply, or NA when unavailable
"""

from __future__ import annotations

import os
import platform
import subprocess
try:
    import resource
except ImportError:
    resource = None


def current_rss_mb() -> float:
    if platform.system() == "Linux":
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0

    if platform.system() == "Windows":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        ctypes.windll.psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)

    # ru_maxrss is kilobytes on Linux, bytes on macOS. The benchmark protocol
    # is Linux, but keep the fallback readable for ad-hoc local runs.
    if resource is None:
        return 0.0

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024.0 if platform.system() == "Linux" else 1024.0 * 1024.0)


def current_process_vram_mb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    total = 0.0
    found = False
    pid = os.getpid()

    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total += used_mb
            found = True

    return total if found else None


def gpu_used_memory_mb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for line in output.splitlines():
        try:
            return float(line.strip())
        except ValueError:
            continue
    return None


vram_before_mb = gpu_used_memory_mb()

import torch
import torch.nn as nn


torch.manual_seed(42)

inputs = torch.empty((0, 1), dtype=torch.float32)
targets = torch.empty((0, 1), dtype=torch.float32)
dataset = torch.utils.data.TensorDataset(inputs, targets)
model = nn.Sequential(nn.Linear(1, 1))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

baseline_ram_mb = current_rss_mb()

if torch.cuda.is_available():
    torch.cuda.init()
    with torch.no_grad():
        a = torch.ones((32, 32), device="cuda", dtype=torch.float32)
        b = torch.ones((32, 32), device="cuda", dtype=torch.float32)
        c = a @ b
    torch.cuda.synchronize()

print(f"baseline_ram_mb {baseline_ram_mb:.1f}")

vram_mb = current_process_vram_mb()
if vram_mb is None:
    vram_after_mb = gpu_used_memory_mb()
    if vram_before_mb is not None and vram_after_mb is not None:
        vram_mb = max(0.0, vram_after_mb - vram_before_mb)
print(f"gpu_ready_vram_mb {vram_mb:.1f}" if vram_mb is not None else "gpu_ready_vram_mb NA")
