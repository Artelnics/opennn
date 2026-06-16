# Probe (not part of the benchmark): the benchmark model in NHWC
# (channels_last) fp32, to measure the cuDNN layout penalty that OpenNN's
# NHWC fp32 convolution path pays.

import sys
import time

import numpy as np
import torch

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

x = (torch.from_numpy(np.load("mnist_images.npy")).permute(0, 3, 1, 2).div(255.0)
     .cuda().to(memory_format=torch.channels_last))
y = torch.from_numpy(np.load("mnist_labels.npy")).cuda()
n = x.shape[0]

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, kernel_size=3, padding="same"),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(14 * 14 * 16, 10),
).cuda().to(memory_format=torch.channels_last)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
starts = list(range(0, n - batch + 1, batch))


def run_epoch():
    model.train()
    perm = torch.randperm(n, device="cuda")
    for s in starts:
        idx = perm[s:s + batch]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[idx]), y[idx])
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
print(f"epoch_s={times[len(times) // 2]:.4f}")
print(f"samples_per_sec={n / times[len(times) // 2]:.0f}")
