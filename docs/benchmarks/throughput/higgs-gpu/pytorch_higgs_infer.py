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

import contextlib
import sys
import time

import numpy as np
import torch


def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    # x = features (all but the last column); y (last column) is ignored for speed.
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
    allow_tf32 = precision in ("bf16", "tf32")
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

    # Whole batch-aligned test slice resident on the GPU once (no host<->device
    # copy per step): the fast inference protocol.
    x = torch.from_numpy(x_np[:processed]).to(device).contiguous()
    n_batches = processed // batch

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast else contextlib.nullcontext())

    def run_pass():
        with torch.no_grad(), ctx:
            for s in range(0, processed, batch):
                model(x[s:s + batch])

    # Warmup: cuDNN autotuning, workspace allocation.
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
