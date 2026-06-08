# PyTorch data-capacity benchmark (the standard pandas load path).
#
# Mirrors opennn_capacity.cpp: read a headerless Rosenbrock CSV with pandas,
# move it into float32 tensors, build the same N -> N -> 1 tanh MLP, and run a
# short Adam training so the batch buffers are allocated. Prints the process
# peak working set (resident memory).
#
# This is the way the overwhelming majority of PyTorch tutorials and pipelines
# load a CSV. pandas parses the whole file into a DataFrame (float64 by
# default) before anything is converted to tensors, which is where a too-large
# dataset exhausts RAM.
#
#   usage:  python pytorch_capacity.py <csv_path> <input_variables>

import ctypes
import ctypes.wintypes as wt
import sys


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
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.GetCurrentProcess()
    # K32GetProcessMemoryInfo is exported by kernel32 on modern Windows and
    # avoids the psapi.dll vs kernel32 forwarding confusion.
    get_mem = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if get_mem is None:
        get_mem = ctypes.WinDLL("psapi").GetProcessMemoryInfo
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
    if len(sys.argv) < 3:
        sys.stderr.write("usage: python pytorch_capacity.py <csv_path> <input_variables>\n")
        return 2

    csv_path = sys.argv[1]
    input_variables = int(sys.argv[2])
    # Optional 3rd arg: pandas read dtype. Default float64 is what pd.read_csv
    # does with no dtype= argument — the realistic out-of-the-box path. Pass
    # "float32" to measure the leaner best case.
    read_dtype = sys.argv[3] if len(sys.argv) > 3 else "float64"

    try:
        import numpy as np
        import pandas as pd
        import torch

        torch.manual_seed(42)

        np_read = np.float32 if read_dtype == "float32" else np.float64

        # The standard load path: pandas parses the entire CSV into a DataFrame.
        # This is the allocation that runs out of memory on large files.
        frame = pd.read_csv(csv_path, header=None, dtype=np_read)
        print(f"read_dtype={read_dtype}")
        print(f"loaded_samples={len(frame)}")

        # Training in float32 regardless, matching OpenNN; the read dtype above
        # is what determines the pandas-side load footprint.
        values = torch.from_numpy(frame.to_numpy(dtype=np.float32))
        del frame
        import gc
        gc.collect()
        print(f"sustained_after_load_mb={current_working_set_mb():.3f}")
        print(f"after_load_peak_mb={peak_working_set_mb():.3f}")
        inputs = values[:, :input_variables]
        targets = values[:, input_variables:]

        model = torch.nn.Sequential(
            torch.nn.Linear(input_variables, input_variables),
            torch.nn.Tanh(),
            torch.nn.Linear(input_variables, 1),
        )
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters())

        # One epoch over batches of 1000 — enough to allocate the training buffers.
        batch_size = 1000
        n = inputs.shape[0]
        model.train()
        for start in range(0, n, batch_size):
            xb = inputs[start:start + batch_size]
            yb = targets[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        print("trained=1")
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=OK")
        return 0

    except MemoryError:
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=OOM")
        return 1
    except Exception as exc:  # noqa: BLE001 - report and continue the sweep
        sys.stderr.write(f"{type(exc).__name__}: {exc}\n")
        print(f"peak_mb={peak_working_set_mb():.3f}")
        print("RESULT=ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
