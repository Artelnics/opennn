#!/usr/bin/env python3
"""What PyTorch costs before it does any work.

The counterpart of footprint.cpp.

  footprint.py memory    resident set after importing and declaring intent
  footprint.py startup   time to first prediction
  footprint.py export    write a TorchScript model requiring its runtime

Each mode is its own process. A cost paid at startup is already paid by
anything sharing a process with it, so measuring two in one run measures
neither.

`memory` deliberately counts the import. For OpenNN the equivalent cost is
paid by the dynamic loader before `main()`, so neither engine's figure is
"the library's own size" -- both are what a process weighs once the framework
is available and an empty model exists. That is the comparable quantity.
"""

from __future__ import annotations

import os
import ctypes
import sys
import time
from pathlib import Path

ENTERED = time.perf_counter()

def resident_mb() -> float | None:
    """Current Windows working set or Linux resident set, in MiB."""
    try:
        if os.name == "nt":
            class Counters(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                    (name, ctypes.c_size_t) for name in (
                        "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                        "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")]
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            kernel.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                                       ctypes.POINTER(Counters), ctypes.c_ulong]
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if kernel.K32GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                return counters.working_set / (1024.0 * 1024.0)
            return None
        with open("/proc/self/statm") as handle:
            resident = int(handle.read().split()[1])
        return resident * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except (OSError, ValueError, IndexError, AttributeError):
        return None

def measure_memory() -> int:
    import torch                                        # noqa: F401  the cost being measured

    model = torch.nn.Sequential()
    optimizer = torch.optim.Adam(list(model.parameters()) or [torch.zeros(1, requires_grad=True)])
    _ = (model, optimizer)

    memory = resident_mb()
    print(f"baseline_ram_mb={memory:.3f}" if memory is not None else "baseline_ram_mb=null")
    print("baseline_ram_metric=process_resident_set_mib")
    print("baseline_ram_note=available" if memory is not None
          else "baseline_ram_note=resident_memory_query_unavailable")
    print("device=cpu")
    return 0

def measure_startup() -> int:
    import torch

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(10, 64), torch.nn.Tanh(),
                                torch.nn.Linear(64, 1))

    with torch.no_grad():
        output = model(torch.ones(1, 10))

    seconds = time.perf_counter() - ENTERED
    print(f"prediction={output.item():.6g}")
    print(f"first_prediction_s={seconds:.6g}")
    print("first_prediction_scope=script_timer_to_completed_prediction_includes_torch_import_excludes_interpreter_start")
    print("device=cpu")
    return 0

def measure_export() -> int:
    """PyTorch has no dependency-free source export, and that is the finding.

    OpenNN writes a `.c` and a `.py` that run with no runtime at all.
    TorchScript and ONNX both produce artifacts that still need a runtime to
    execute -- libtorch or onnxruntime -- so the sizes below are not
    comparable to OpenNN's and are reported as what they are. Calling either
    "standalone source" would be the wrong claim, which is why the
    `standalone_source` flag is here rather than left implicit.
    """
    import torch

    torch.manual_seed(42)
    model = torch.nn.Sequential(torch.nn.Linear(3, 64), torch.nn.Tanh(),
                                torch.nn.Linear(64, 1)).eval()

    # A temporary directory: these are run outputs, and a benchmark that
    # litters the tree it runs from will eventually have one committed.
    import tempfile

    target = Path(tempfile.mkdtemp(prefix="pytorch_footprint")) / "model.pt"
    scripted = torch.jit.script(model)
    scripted.save(str(target))
    size = target.stat().st_size

    print(f"export_torchscript_bytes={size}")
    print("standalone_source=0")
    print("export_note=TorchScript needs libtorch to run; not dependency-free source")
    return 0

MODES = {"memory": measure_memory, "startup": measure_startup, "export": measure_export}

def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""

    if mode not in MODES:
        print("usage: footprint.py memory | startup | export", file=sys.stderr)
        return 2

    print(f"engine=pytorch\nmode={mode}")
    status = MODES[mode]()
    print("RESULT=OK")
    return status

if __name__ == "__main__":
    raise SystemExit(main())
