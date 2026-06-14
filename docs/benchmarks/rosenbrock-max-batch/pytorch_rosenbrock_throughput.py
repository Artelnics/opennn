# Equal-batch steady-state speed for the shallow Rosenbrock MLP in PyTorch,
# the counterpart to opennn_rosenbrock_throughput. Same net (inputs->hidden->1,
# tanh, MSE, Adam, fp32), GPU-resident synthetic data, warmup excluded.
#
#   usage: python pytorch_rosenbrock_throughput.py <train|inference> [batch] [iters] [inputs] [hidden]

import sys
import time

import torch
import torch.nn as nn

mode   = sys.argv[1] if len(sys.argv) > 1 else "train"
batch  = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
iters  = int(sys.argv[3]) if len(sys.argv) > 3 else 100
inputs = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
hidden = int(sys.argv[5]) if len(sys.argv) > 5 else 1000

assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device("cuda")
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(dev)
x = torch.randn(batch, inputs, device=dev)
y = torch.randn(batch, 1, device=dev)

if mode == "inference":
    net.eval()
    with torch.no_grad():
        net(x)  # warmup
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / iters
else:
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    def step():
        opt.zero_grad(set_to_none=True)
        loss_fn(net(x), y).backward()
        opt.step()

    step()  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / iters

print(f"mode={mode} batch={batch}")
print(f"step_s={per:.6f}")
print(f"samples_per_sec={int(batch / per)}")
