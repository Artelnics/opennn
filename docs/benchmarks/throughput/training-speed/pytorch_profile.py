# Per-kernel GPU profile of the PyTorch training step, to compare against
# OpenNN's per-phase timing and locate the largest divergence.
#
# Same step as the speed benchmark: F->F->1 MLP, tanh, Adam, MSE, batch B,
# data resident on the GPU. Uses torch.profiler to get self-time per CUDA
# kernel, then groups kernels into forward / backward / optimizer buckets.
#
#   usage:  python pytorch_profile.py [features] [batch] [precision]

import sys
import torch
from torch.profiler import profile, ProfilerActivity

F = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
precision = sys.argv[3] if len(sys.argv) > 3 else "bf16"

assert torch.cuda.is_available()
dev = "cuda"
torch.manual_seed(42)
allow_tf32 = precision in ("bf16", "tf32")
torch.backends.cuda.matmul.allow_tf32 = allow_tf32
torch.backends.cudnn.allow_tf32 = allow_tf32
use_autocast = precision == "bf16"

x = torch.randn(B, F, device=dev)
y = torch.randn(B, 1, device=dev)

model = torch.nn.Sequential(
    torch.nn.Linear(F, F), torch.nn.Tanh(), torch.nn.Linear(F, 1)
).to(dev)
loss_fn = torch.nn.MSELoss()
opt = torch.optim.Adam(model.parameters())

import contextlib
def step():
    opt.zero_grad(set_to_none=True)
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast else contextlib.nullcontext())
    with ctx:
        loss = loss_fn(model(x), y)
    loss.backward()
    opt.step()

for _ in range(50):
    step()
torch.cuda.synchronize()

ITERS = 200
with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
    for _ in range(ITERS):
        step()
    torch.cuda.synchronize()

# Aggregate CUDA self-time per kernel.
events = [e for e in prof.key_averages() if e.self_device_time_total > 0]
events.sort(key=lambda e: e.self_device_time_total, reverse=True)
total_us = sum(e.self_device_time_total for e in events)

print(f"precision={precision} F={F} B={B} iters={ITERS}")
print(f"total GPU us/iter = {total_us/ITERS:.1f}")
print(f"{'kernel':<55}{'us/iter':>10}{'%':>7}")
for e in events[:20]:
    name = e.key[:54]
    print(f"{name:<55}{e.self_device_time_total/ITERS:>10.2f}{100*e.self_device_time_total/total_us:>6.1f}%")
