# TensorFlow data-capacity benchmark (the standard pandas load path).
#
# Mirrors pytorch_capacity.py / opennn_capacity.cpp: read a headerless HIGGS
# CSV with pandas (float64 by default, the out-of-the-box path), move it into
# float32 tensors, build a small 28 -> hidden -> 1 tanh MLP, and run a short
# Adam training so the batch buffers are allocated. Prints the process peak
# working set (resident memory).
#
# The CSV is the prepared HIGGS training file (28 features + 1 label per row)
# tiled up to the sweep's target sample count by tile_higgs.exe. See
# ../../throughput/higgs/README.md for the dataset contract.
#
# TensorFlow users load CSVs with pandas exactly as PyTorch users do, so the load
# footprint -- the thing that runs out of RAM on a large file -- is the same
# pandas DataFrame allocation.
#
#   usage:  python tensorflow_capacity.py <csv_path> [read_dtype]

import ctypes
import ctypes.wintypes as wt
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU memory benchmark


def _mem_counters():
    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wt.DWORD),
            ("PageFaultCount", wt.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    get_mem = ctypes.windll.psapi.GetProcessMemoryInfo
    get_mem.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    get_mem.restype = wt.BOOL
    if not get_mem(handle, ctypes.byref(pmc), pmc.cb):
        return (-1.0, -1.0)
    mb = 1024.0 * 1024.0
    return (pmc.PeakWorkingSetSize / mb, pmc.WorkingSetSize / mb)


def peak_working_set_mb():
    return _mem_counters()[0]


def current_working_set_mb():
    return _mem_counters()[1]


def main():
    if len(sys.argv) < 2:
        sys.stderr.write("usage: python tensorflow_capacity.py <csv_path> [read_dtype]\n")
        return 2

    csv_path = sys.argv[1]
    read_dtype = sys.argv[2] if len(sys.argv) > 2 else "float64"

    try:
        import numpy as np
        import pandas as pd
        import tensorflow as tf

        tf.random.set_seed(42)
        np_read = np.float32 if read_dtype == "float32" else np.float64

        # The standard load path: pandas parses the entire CSV into a DataFrame.
        frame = pd.read_csv(csv_path, header=None, dtype=np_read)
        print(f"read_dtype={read_dtype}")
        print(f"loaded_samples={len(frame)}")

        # HIGGS layout: 28 feature columns then the label column last.
        input_variables = frame.shape[1] - 1

        values = tf.convert_to_tensor(frame.to_numpy(dtype=np.float32))
        del frame
        import gc
        gc.collect()
        print(f"sustained_after_load_mb={current_working_set_mb():.3f}")
        print(f"after_load_peak_mb={peak_working_set_mb():.3f}")
        inputs = values[:, :input_variables]
        targets = values[:, input_variables:]

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_variables, activation="tanh",
                                  input_shape=(input_variables,)),
            tf.keras.layers.Dense(1),
        ])
        opt = tf.keras.optimizers.Adam()
        mse = tf.keras.losses.MeanSquaredError()

        @tf.function
        def step(xb, yb):
            with tf.GradientTape() as tape:
                loss = mse(yb, model(xb, training=True))
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))

        batch_size = 1000
        n = inputs.shape[0]
        for start in range(0, n, batch_size):
            step(inputs[start:start + batch_size], targets[start:start + batch_size])

        print("trained=1")
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=OK")
        return 0

    except MemoryError:
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=OOM")
        return 1
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
