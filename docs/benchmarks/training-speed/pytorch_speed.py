# PyTorch GPU training-speed benchmark, tuned for maximum throughput.
#
# Mirrors the Neural Designer training-speed benchmark: a 2-layer MLP
# (F -> F -> 1, tanh then linear) trained with Adam + MSE on the Rosenbrock
# dataset, batch 1000. Reports median seconds/epoch and samples/second.
#
# "Highest performance" path:
#   * whole dataset resident on the GPU (no host<->device copies per step),
#   * TF32 matmuls enabled,
#   * autocast (bf16) mixed precision,
#   * torch.compile for kernel fusion (uses CUDA graphs under the hood),
#   * channels-last / contiguous tensors, no Python-side per-sample work.
#
#   usage:  python pytorch_speed.py <samples> <features> [epochs] [batch]

import sys
import time

import numpy as np
import torch


def rosenbrock_dataset(n, f, seed=1234, device="cuda"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.rand(n, f, generator=g) * 2.0 - 1.0)
    # y = sum_i (1-x_i)^2 + 100 (x_{i+1}-x_i^2)^2
    a = (1.0 - x[:, :-1])
    b = (x[:, 1:] - x[:, :-1] ** 2)
    y = (a * a + 100.0 * b * b).sum(dim=1, keepdim=True)
    return x.to(device).contiguous(), y.to(device).contiguous()


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
    # Precision: bf16 (autocast mixed precision, default), tf32 (fp32 math on
    # tensor cores), or fp32 (strict IEEE, tensor cores off).
    precision = sys.argv[5] if len(sys.argv) > 5 else "bf16"

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    torch.manual_seed(42)

    # Tensor-core (TF32) math is allowed for bf16 and tf32 modes, off for strict fp32.
    allow_tf32 = precision in ("bf16", "tf32")
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True
    use_autocast = precision == "bf16"
    print(f"precision={precision} autocast={use_autocast} tf32={allow_tf32}")

    x, y = rosenbrock_dataset(samples, features, device=device)
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"samples={samples} features={features} batch={batch} epochs={epochs}")

    model = torch.nn.Sequential(
        torch.nn.Linear(features, features),
        torch.nn.Tanh(),
        torch.nn.Linear(features, 1),
    ).to(device)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Fuse the step for throughput; bf16 autocast for mixed precision.
    # Default compile mode (not max-autotune) to keep the warmup memory spike
    # modest on a 6 GB GPU where the dataset already occupies most of VRAM.
    import contextlib

    @torch.compile
    def train_step(xb, yb):
        optimizer.zero_grad(set_to_none=True)
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_autocast else contextlib.nullcontext())
        with ctx:
            pred = model(xb)
            loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        return loss

    n = x.shape[0]
    starts = list(range(0, n - batch + 1, batch))

    def run_epoch():
        model.train()
        for s in starts:
            train_step(x[s:s + batch], y[s:s + batch])
        torch.cuda.synchronize()

    # Warmup (includes torch.compile autotuning, which is slow on first call).
    print("warmup (compiling)...")
    run_epoch()
    run_epoch()

    times = []
    for e in range(epochs):
        t0 = time.perf_counter()
        run_epoch()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    samples_per_sec = samples / median
    print(f"median_epoch_s={median:.4f}")
    print(f"samples_per_sec={samples_per_sec:.0f}")
    print(f"peak_vram_mb={torch.cuda.max_memory_allocated()/1e6:.0f}")
    print("RESULT=OK")


if __name__ == "__main__":
    main()
