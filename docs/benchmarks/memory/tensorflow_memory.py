"""TensorFlow baseline-memory benchmark.

Construct the minimum TensorFlow/Keras objects used by a training application
(empty Dataset, tiny Keras model, loss, optimizer), then report:
  - baseline_ram_mb: current resident set size for the process
  - gpu_ready_vram_mb: nvidia-smi-reported GPU memory for this process after
    one tiny CUDA matrix multiply, or NA when unavailable
"""

from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

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

import tensorflow as tf


tf.random.set_seed(42)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

with tf.device("/CPU:0"):
    inputs = tf.zeros((0, 1), dtype=tf.float32)
    targets = tf.zeros((0, 1), dtype=tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,)),
            tf.keras.layers.Dense(1),
        ]
    )
    loss_fn = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()

baseline_ram_mb = current_rss_mb()

if gpus:
    with tf.device("/GPU:0"):
        a = tf.ones((32, 32), dtype=tf.float32)
        b = tf.ones((32, 32), dtype=tf.float32)
        _ = tf.linalg.matmul(a, b).numpy()

print(f"baseline_ram_mb {baseline_ram_mb:.1f}")

vram_mb = current_process_vram_mb()
if vram_mb is None:
    vram_after_mb = gpu_used_memory_mb()
    if vram_before_mb is not None and vram_after_mb is not None:
        vram_mb = max(0.0, vram_after_mb - vram_before_mb)
print(f"gpu_ready_vram_mb {vram_mb:.1f}" if vram_mb is not None else "gpu_ready_vram_mb NA")
