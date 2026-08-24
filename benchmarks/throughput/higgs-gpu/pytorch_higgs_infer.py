#!/usr/bin/env python3
"""PyTorch GPU HIGGS dense inference benchmark, written the way a PyTorch user
writes it: nn.Sequential in eval mode, the forward under no_grad, the test
split GPU-resident. The measurement protocol - rotation, soaking, medians over
rounds - lives in run_higgs_infer.py and run_higgs_infer_sweep.py, not here.

    python pytorch_higgs_infer.py <test_csv> [batch[,batch...]] [runs] [fp32|bf16] [hidden] [hidden_layers] [activation]

Precision: fp32 (TF32 matmuls, as in every engine of this benchmark) or bf16
(torch.autocast), matching opennn_higgs_infer's Configuration::set precision;
"strict" turns TF32 off.

PyTorch's best configuration for this network is not the same at every batch
size, so the driver times its candidate paths at each rung and reports the
faster, naming it in `pt_path`, the way the TensorFlow driver reports the
faster of its two dispatch paths. Measured at bf16, medians of three rotated
rounds, samples/s:

    batch    eager + CUDA graph     compile max-autotune
      256            12,491,364                4,350,403
    1,024            19,210,673               17,544,645
    8,192            29,370,503               37,018,545
   65,536            25,480,267               36,726,327

The crossover is between 1,024 and 8,192, and it is large on both sides: 2.9x
for the graph at 256, 1.44x for inductor at 65,536. A driver pinned to
reduce-overhead measured PyTorch 2.7x under itself at 256 and 29% under itself
at 8,192.

Each rung and path gets a freshly compiled model (torch._dynamo.reset), so
PyTorch measures a statically compiled model per shape rather than a dynamic
recompilation, which is what a deployment at that batch size would have. Each
batch is staged into a fixed buffer with one device-to-device copy before the
forward, on every path, so the engines are compared on equal terms. The batch
sizes run inside one process, so they share one load and one thermal window.

Environment:
    PT_PATHS=graph,max-autotune  comma-separated candidate paths. `graph` is
                                 eager modules under a hand-captured CUDA
                                 graph; anything else is a torch.compile mode
                                 (default, reduce-overhead, max-autotune),
                                 which brings its own CUDA graph.
    PT_BF16_WEIGHTS=1            hold the weights in bf16 instead of casting
                                 them inside autocast on every call (what a
                                 deployment does); the runners set it for the
                                 bf16 cell
    PT_NOGRAPH=1                 no CUDA graph at all on the `graph` path
"""

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def load_csv(path):
    # np.loadtxt on the 500k-row split is minutes of wall clock, none of it
    # measured, so the parsed array is cached next to the CSV.
    path = Path(path)
    cache = path.with_suffix(path.suffix + ".npy")
    if cache.exists() and cache.stat().st_mtime >= path.stat().st_mtime:
        data = np.load(cache, mmap_mode="r")
    else:
        data = np.loadtxt(path, delimiter=",", dtype=np.float32)
        try:
            np.save(cache, data)
        except OSError:              # read-only data directory: parse next time
            pass
    return np.ascontiguousarray(data[:, :-1])


def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch_text = sys.argv[2] if len(sys.argv) > 2 else "8192"
    batch_list = [int(item) for item in batch_text.split(",") if item.strip()] or [8192]
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    precision = sys.argv[4] if len(sys.argv) > 4 else "fp32"
    hidden = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    hidden_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    activation = (sys.argv[7] if len(sys.argv) > 7 else "relu").lower()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    torch.manual_seed(42)

    use_autocast = precision == "bf16"
    paths = [p.strip() for p in os.environ.get("PT_PATHS", "graph,max-autotune").split(",") if p.strip()]
    bf16_weights = os.environ.get("PT_BF16_WEIGHTS") is not None and use_autocast
    # fp32 runs with TF32 tensor cores in every engine of this benchmark
    # (OpenNN's fp32 GEMMs are CUBLAS_COMPUTE_32F_FAST_TF32); "strict" disables it.
    allow_tf32 = precision != "strict"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True

    x_np = load_csv(test_csv)
    features = x_np.shape[1]
    samples = x_np.shape[0]

    print("engine=pytorch")
    print("mode=infer")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"runs={runs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision}")
    print(f"paths={','.join(paths)}{' +bf16_weights' if bf16_weights else ''}")

    parameters = features * hidden + hidden
    parameters += (hidden_layers - 1) * (hidden * hidden + hidden)
    parameters += hidden + 1
    print(f"parameters={parameters}")

    x = torch.from_numpy(np.ascontiguousarray(x_np)).to(device).contiguous()
    if bf16_weights:
        x = x.to(torch.bfloat16)

    act_layer = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
    single = len(batch_list) == 1

    for batch in batch_list:
        processed = (samples // batch) * batch
        if processed <= 0:
            print(f"batch_{batch}_error=batch larger than the test split")
            continue

        n_batches = processed // batch
        medians = {}

        for path in paths:
            # A fresh compile per rung and per path: otherwise the second
            # distinct shape recompiles and the third pushes dynamo to dynamic
            # shapes, slower than what a deployment at a fixed batch size runs.
            torch._dynamo.reset()

            layers = []
            current = features
            for _ in range(hidden_layers):
                layers += [torch.nn.Linear(current, hidden), act_layer()]
                current = hidden
            layers += [torch.nn.Linear(current, 1), torch.nn.Sigmoid()]
            model = torch.nn.Sequential(*layers).to(device).eval()

            # bf16 weights make the autocast casts unnecessary, so the timed
            # pass moves only activations; the arithmetic is bf16 either way.
            if bf16_weights:
                model = model.to(torch.bfloat16)

            compile_mode = None if path == "graph" else path
            if compile_mode:
                model = torch.compile(model, mode=None if compile_mode == "default" else compile_mode,
                                      dynamic=False)

            ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                   if use_autocast and not bf16_weights else contextlib.nullcontext())

            # torch.compile's reduce-overhead / max-autotune wrap the step in
            # their own CUDA graph; capturing a compiled model by hand fights
            # that, so the manual graph is only for the eager path.
            use_graph = os.environ.get("PT_NOGRAPH") is None and compile_mode is None

            static_x = x[:batch].clone()
            graph = None

            if use_graph:
                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream), torch.no_grad(), ctx:
                    for _ in range(3):
                        model(static_x)
                torch.cuda.current_stream().wait_stream(side_stream)
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph), torch.no_grad(), ctx:
                    model(static_x)

                def run_pass():
                    for s in range(0, processed, batch):
                        static_x.copy_(x[s:s + batch], non_blocking=True)
                        graph.replay()
            else:
                def run_pass():
                    with torch.no_grad(), ctx:
                        for s in range(0, processed, batch):
                            static_x.copy_(x[s:s + batch], non_blocking=True)
                            model(static_x)

            run_pass()
            run_pass()
            torch.cuda.synchronize()

            times = []
            for _ in range(runs):
                t0 = time.perf_counter()
                run_pass()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)

            # This path's model and graph pool are released before the next
            # path builds, so the paths do not share the device.
            del run_pass, model, static_x, graph

            # Temporal order, before the sort: a median hides a drifting machine.
            print(f"batch_{batch}_{path}_pass_times=" + ",".join(f"{t:.9g}" for t in times))

            times.sort()
            medians[path] = times[len(times) // 2]
            print(f"batch_{batch}_samples_per_sec_{path}={processed / medians[path]:.0f}")
            print(f"batch_{batch}_ms_per_batch_{path}={medians[path] * 1000.0 / n_batches:.6f}")

        pt_path = min(medians, key=medians.get)
        median_pass_s = medians[pt_path]
        samples_per_sec = processed / median_pass_s
        ms_per_batch = median_pass_s * 1000.0 / n_batches

        print(f"batch_{batch}_pt_path={pt_path}")

        if single:
            print(f"samples={processed}")
            print(f"batch={batch}")
            print(f"pt_path={pt_path}")
            print(f"median_pass_s={median_pass_s:.9g}")
            print(f"samples_per_sec={samples_per_sec:.0f}")
            print(f"ms_per_batch={ms_per_batch:.6f}")
        else:
            print(f"batch_{batch}_samples={processed}")
            print(f"batch_{batch}_samples_per_sec={samples_per_sec:.0f}"
                  f" median_pass_s={median_pass_s:.9g} ms_per_batch={ms_per_batch:.6f}")
        sys.stdout.flush()

    print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}")
    print("RESULT=OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
