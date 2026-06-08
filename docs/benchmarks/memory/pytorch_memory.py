# PyTorch peak-memory benchmark: load sum.csv (1000 x 101, regression), build
# the same MLP, train, and report resident-set-size (RSS) at two points:
# baseline (framework loaded + model built, before training) and peak.

import resource
import csv

import torch
import torch.nn as nn


def peak_rss_mb():
    # ru_maxrss is kilobytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


# sum.csv: 100 inputs + 1 target, ';'-separated, no header.
rows = []
with open("sum.csv", newline="") as f:
    for r in csv.reader(f, delimiter=";"):
        rows.append([float(x) for x in r])

data = torch.tensor(rows, dtype=torch.float32)
inputs = data[:, :-1]
targets = data[:, -1:]

torch.manual_seed(42)
net = nn.Sequential(
    nn.Linear(100, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

print(f"baseline_rss_mb {peak_rss_mb():.1f}")

optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()
batch_size = 32
n = inputs.shape[0]

for epoch in range(50):
    for i in range(0, n, batch_size):
        xb = inputs[i:i + batch_size]
        yb = targets[i:i + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(net(xb), yb)
        loss.backward()
        optimizer.step()

print(f"peak_rss_mb {peak_rss_mb():.1f}")
