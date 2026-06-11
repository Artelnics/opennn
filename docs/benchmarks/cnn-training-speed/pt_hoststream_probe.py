# Probe (not part of the benchmark): PyTorch with host-resident data and a
# per-step pinned H2D copy, approximating OpenNN's input pipeline, to see how
# much of PyTorch's edge comes from the GPU-resident dataset.

import sys
import time

import numpy as np
import torch

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

x = torch.from_numpy(np.load("mnist_images.npy")).permute(0, 3, 1, 2).div(255.0).contiguous().pin_memory()
y = torch.from_numpy(np.load("mnist_labels.npy")).pin_memory()
n = x.shape[0]

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, kernel_size=3, padding="same"),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(14 * 14 * 16, 10),
).cuda()

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
starts = list(range(0, n - batch + 1, batch))


def run_epoch():
    model.train()
    perm = torch.randperm(n)
    for s in starts:
        idx = perm[s:s + batch]
        xb = x[idx].cuda(non_blocking=True)
        yb = y[idx].cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()


run_epoch()
run_epoch()
times = []
for _ in range(epochs):
    t0 = time.perf_counter()
    run_epoch()
    times.append(time.perf_counter() - t0)
times.sort()
median = times[len(times) // 2]
print(f"epoch_s={median:.4f}")
print(f"samples_per_sec={n / median:.0f}")
