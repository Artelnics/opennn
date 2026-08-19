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
#   usage:  python pytorch_higgs_infer.py <test_csv> [batch] [runs] [fp32|bf16]
#                                         [hidden] [hidden_layers] [activation]
#   env:    PT_COMPILE_MODE=default|reduce-overhead|max-autotune -> torch.compile
#                      the model. reduce-overhead and max-autotune bring their own
#                      CUDA graphs, so the hand-built one is dropped for them.
#           PT_BF16_WEIGHTS=1 -> hold the weights in bf16 instead of casting them
#                      inside autocast on every call (what a deployment does)
#           PT_NOGRAPH=1 -> no CUDA graph at all

import contextlib
import os
import sys
import time

import numpy as np
import torch

def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)

    x = np.ascontiguousarray(data[:, :-1])
    return x

def main():
    test_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_test.csv"
    batch = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    runs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    precision = sys.argv[4] if len(sys.argv) > 4 else "fp32"
    hidden = int(sys.argv[5]) if len(sys.argv) > 5 else 1024
    hidden_layers = int(sys.argv[6]) if len(sys.argv) > 6 else 2
    activation = (sys.argv[7] if len(sys.argv) > 7 else "relu").lower()

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    torch.manual_seed(42)

    use_autocast = precision == "bf16"
    compile_mode = os.environ.get("PT_COMPILE_MODE")        # None -> eager modules
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
    processed = (samples // batch) * batch

    print("engine=pytorch")
    print("mode=infer")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"samples={processed}")
    print(f"batch={batch}")
    print(f"runs={runs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision}")
    print(f"mode={'compile:' + compile_mode if compile_mode else 'eager'}"
          f"{' +bf16_weights' if bf16_weights else ''}")

    if processed <= 0:
        print("RESULT=ERROR")
        raise SystemExit("batch larger than the test split")

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
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    # bf16 weights make the autocast casts unnecessary, so the timed pass moves
    # only activations; the comparison keeps its bf16 arithmetic either way.
    if bf16_weights:
        model = model.to(torch.bfloat16)

    if compile_mode:
        model = torch.compile(model, mode=None if compile_mode == "default" else compile_mode)

    x = torch.from_numpy(x_np[:processed]).to(device).contiguous()
    n_batches = processed // batch

    if bf16_weights:
        x = x.to(torch.bfloat16)

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast and not bf16_weights else contextlib.nullcontext())

    # torch.compile's reduce-overhead / max-autotune wrap the step in their own
    # CUDA graph; capturing a compiled model by hand fights that, so the manual
    # graph is only for the eager path.
    manual_graph_modes = {None, "default"}
    use_graph = os.environ.get("PT_NOGRAPH") is None and compile_mode in manual_graph_modes

    if use_graph:
        static_x = x[:batch].clone()
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
        print("cuda_graph=on")

        def run_pass():
            for s in range(0, processed, batch):
                static_x.copy_(x[s:s + batch], non_blocking=True)
                graph.replay()
    else:
        print("cuda_graph=" + ("compiled" if compile_mode else "off"))
        static_x = x[:batch].clone()

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

    times.sort()
    median_pass_s = times[len(times) // 2]
    samples_per_sec = processed / median_pass_s
    ms_per_batch = median_pass_s * 1000.0 / n_batches

    print(f"median_pass_s={median_pass_s:.9g}")
    print(f"samples_per_sec={samples_per_sec:.0f}")
    print(f"ms_per_batch={ms_per_batch:.6f}")
    print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}")
    print("RESULT=OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
