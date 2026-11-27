#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Artificial Intelligence Techniques, S.L. and Roberto López González

"""ONNX Runtime dense benchmark matching dense.cpp's infer mode.

The model is the one dense.cpp measures, with the same seeded weights, written
by `dense_opennn export` -- the path a network trained in OpenNN takes to this
runtime. ONNX Runtime is an inference engine, so infer is the only mode. FP32
permits TF32 on CUDA, as the other engines do; strict disables it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from common import find_binary, report_anonymous_baseline  # noqa: E402

def resident_mib() -> float:
    try:
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def load_inputs(path: str) -> np.ndarray:
    """Float32 inputs of the same CSV OpenNN reads; the last column is the target."""
    import pandas as pd

    print(f"dataset_test={Path(path).resolve()}", flush=True)
    frame = pd.read_csv(path, header=None, dtype=np.float32)
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float32)[:, :-1])

def export_model(directory: str, features: int, hidden: str, layers: str,
                 activation: str) -> tuple[str, str]:
    """Have OpenNN write the network; on the CPU, so no GPU memory is spent here."""
    binary, found = find_binary("dense_opennn")
    if not found:
        raise SystemExit(f"dense_opennn not built (looked at {binary})")

    path = str(Path(directory) / "dense.onnx")
    completed = subprocess.run([binary, "export", path, str(features), hidden, layers,
                                activation, "cpu", "fp32"],
                               capture_output=True, text=True, check=True)
    fields = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    return path, fields.get("parameters", "unknown")

def session_for(path: str, device: str, precision: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # The runner pins CPU work to performance cores and may set a matched thread
    # count for every engine; honour it, or keep the runtime's default.
    threads = os.environ.get("OMP_NUM_THREADS")
    if threads:
        options.intra_op_num_threads = int(threads)

    if device == "cpu":
        return ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])

    ort.preload_dlls()
    cuda = {"device_id": "0",
            "use_tf32": "0" if precision == "strict" else "1",
            "do_copy_in_default_stream": "1"}
    session = ort.InferenceSession(path, options,
                                   providers=[("CUDAExecutionProvider", cuda),
                                              "CPUExecutionProvider"])
    if session.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit("cuda requested but ONNX Runtime fell back to "
                         + session.get_providers()[0])
    return session

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = [int(b) for b in (argv[4] if len(argv) > 4 else "1024").split(",") if b]

    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    hidden, layers, activation = at(5, "1024"), at(6, "2"), at(7, "relu").lower()
    device, precision = at(8, "cuda"), at(9, "fp32")

    if precision == "bf16":
        raise SystemExit("onnxruntime: the pilot runs fp32 only (OpenNN exports fp32 ONNX)")

    print(f"baseline_rss_mib={resident_mib():.1f}")
    report_anonymous_baseline()
    x = load_inputs(argv[2])
    print(f"engine=onnxruntime\nmode=infer\ndevice={device}")
    print(f"blas={'mlas' if device == 'cpu' else 'cublas'}")

    samples, features = x.shape
    row_bytes = features * x.itemsize

    with tempfile.TemporaryDirectory() as directory:
        path, parameters = export_model(directory, features, hidden, layers, activation)
        session = session_for(path, device, precision)

    print(f"parameters={parameters}", flush=True)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    location = "cpu" if device == "cpu" else "cuda"

    # The split stays where the engine reads it: on the device for CUDA, as the
    # other engines keep theirs, and every batch is a view into it. Outputs go
    # to one preallocated buffer, so a run allocates nothing.
    resident = x if device == "cpu" else ort.OrtValue.ortvalue_from_numpy(x, location, 0)
    base = x.ctypes.data if device == "cpu" else resident.data_ptr()

    # CUDA runs are queued without a host synchronisation each, and the pass
    # waits once at its end, as the other engines do.
    queued = ort.RunOptions()
    queued.add_run_config_entry("disable_synchronize_execution_providers", "1")
    waited = ort.RunOptions()

    for batch in batches:
        processed = (samples // batch) * batch
        output = ort.OrtValue.ortvalue_from_shape_and_type([batch, 1], np.float32, location, 0)
        binding = session.io_binding()
        binding.bind_ortvalue_output(output_name, output)
        starts = list(range(0, processed, batch))

        def run_pass():
            for start in starts:
                binding.bind_input(input_name, location, 0, np.float32, [batch, features],
                                   base + start * row_bytes)
                session.run_with_iobinding(binding, waited if start == starts[-1] else queued)

        run_pass()
        run_pass()

        # The same marks the other engines print: energy is integrated between
        # them, so start-up, export and session creation stay outside the window.
        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        print(f"batch_{batch}_pass_times={','.join(f'{t:.6g}' for t in times)}")
        times.sort()
        median = times[len(times) // 2]
        print(f"batch_{batch}_samples_per_sec={int(processed / median)}"
              f" median_pass_s={median:.6g}", flush=True)
        print(f"batch_{batch}_mode=ort:{session.get_providers()[0]}:io_binding", flush=True)

    print("RESULT=OK")
    return 0

def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "infer":
        return infer(sys.argv)
    print(f"onnxruntime runs inference only; mode {mode!r} is not available", file=sys.stderr)
    print("RESULT=ERROR")
    return 2

if __name__ == "__main__":
    sys.exit(main())
