# PyTorch GPU HIGGS dense inference-speed benchmark, the counterpart to
# opennn_higgs_infer.
#
# Mirrors the canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1,
# ReLU hidden, sigmoid output -- see docs/benchmarks/throughput/higgs/README.md).
# Forward-only: the model is in .eval() and the timed region runs under
# torch.no_grad(). The whole (batch-aligned) test slice is made GPU-resident once
# and only the forward is timed, after a warmup. Reports samples/sec and
# ms/batch.
#
# Precision: fp32 (strict, tensor cores off) or bf16 (torch.autocast, TF32
# matmuls on) -- matching opennn_higgs_infer's Configuration::set precision.
#
#   usage:  python pytorch_higgs_infer.py <test_csv> [batch[,batch...]] [runs]
#                                         [fp32|bf16] [hidden] [hidden_layers] [activation]
#   env:    PT_PATHS=graph,max-autotune -> comma-separated candidate paths. Each
#                      is timed at every rung and the faster is reported, the way
#                      the TensorFlow driver reports the faster of its two
#                      dispatch paths. `graph` is eager modules under a
#                      hand-captured CUDA graph; anything else is a torch.compile
#                      mode (default, reduce-overhead, max-autotune), which
#                      brings its own CUDA graph.
#           PT_BF16_WEIGHTS=1 -> hold the weights in bf16 instead of casting them
#                      inside autocast on every call (what a deployment does)
#           PT_NOGRAPH=1 -> no CUDA graph at all on the `graph` path
#
# Why both paths rather than one: PyTorch's best configuration for this network
# is NOT the same at every batch size, and picking one hands it a handicap
# somewhere. Measured here at bf16, medians of three rotated rounds, samples/s:
#
#     batch    eager + CUDA graph     compile max-autotune
#       256            12,491,364                4,350,403
#     1,024            19,210,673               17,544,645
#     8,192            29,370,503               37,018,545
#    65,536            25,480,267               36,726,327
#
# The crossover is between 1,024 and 8,192, and it is large on both sides: 2.9x
# for the graph at 256, 1.44x for inductor at 65,536. A driver pinned to
# reduce-overhead - which is what this one used to do - measured PyTorch 2.7x
# under itself at 256 and 29% under itself at 8,192.
#
# A comma-separated batch list is swept inside one process, matching the OpenNN
# driver: the batch sizes then share one load and one thermal window. Each rung
# and path get a freshly compiled model (torch._dynamo.reset) so PyTorch
# measures a statically compiled model per shape rather than a dynamic
# recompilation, which is what a deployment at that batch size would have.

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

def load_csv(path):
    # np.loadtxt on the full split is minutes of wall clock per engine per run,
    # none of it measured, so the parsed array is cached next to the CSV and
    # re-read with np.load.
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

def parse_batches(text):
    values = [int(item) for item in str(text).split(",") if item.strip()]
    return values or [8192]

def build_model(features, hidden, hidden_layers, activation, device, bf16_weights):
    act_layer = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
    layers = []
    current = features
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(current, hidden))
        layers.append(act_layer())
        current = hidden
    layers.append(torch.nn.Linear(current, 1))
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers).to(device).eval()

    # bf16 weights make the autocast casts unnecessary, so the timed pass moves
    # only activations; the comparison keeps its bf16 arithmetic either way.
    if bf16_weights:
        model = model.to(torch.bfloat16)
    return model

def time_path(path, batch, processed, x, runs, bf16_weights, use_autocast,
              features, hidden, hidden_layers, activation, device):
    """One candidate path at one rung: build it, warm it, and return its times."""
    # A fresh compile per rung and per path: otherwise the second distinct shape
    # recompiles and the third pushes dynamo to dynamic shapes, which is slower
    # than what a deployment at a fixed batch size would run.
    torch._dynamo.reset()
    model = build_model(features, hidden, hidden_layers, activation, device, bf16_weights)

    compile_mode = None if path == "graph" else path
    if compile_mode:
        model = torch.compile(model, mode=None if compile_mode == "default" else compile_mode,
                              dynamic=False)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast and not bf16_weights else contextlib.nullcontext())

    # torch.compile's reduce-overhead / max-autotune wrap the step in their own
    # CUDA graph; capturing a compiled model by hand fights that, so the manual
    # graph is only for the eager path.
    use_graph = os.environ.get("PT_NOGRAPH") is None and compile_mode is None

    static_x = x[:batch].clone()

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
                    # The same fixed-buffer staging the graph paths do, so the
                    # engines are compared on equal terms.
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

    del model
    return times

def measure(batch, x, samples, runs, paths, bf16_weights, use_autocast,
            features, hidden, hidden_layers, activation, device, single):
    processed = (samples // batch) * batch
    if processed <= 0:
        print(f"batch_{batch}_error=batch larger than the test split")
        return

    n_batches = processed // batch

    medians = {}
    for path in paths:
        times = time_path(path, batch, processed, x, runs, bf16_weights, use_autocast,
                          features, hidden, hidden_layers, activation, device)

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

def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch_list = parse_batches(sys.argv[2] if len(sys.argv) > 2 else "8192")
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

    single = len(batch_list) == 1
    for batch in batch_list:
        measure(batch, x, samples, runs, paths, bf16_weights, use_autocast,
                features, hidden, hidden_layers, activation, device, single)

    print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}")
    print("RESULT=OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
