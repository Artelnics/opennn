# PyTorch precision benchmark on the Rosenbrock dataset (10 inputs), after the
# Neural Designer blog protocol: 10 -> 10 (tanh) -> 1 (linear), U(-1, 1) init,
# Adam (lr 0.001), MSE, batch 1000, 10,000 epochs, trained on all samples.
# Prints the training wall time and writes full-dataset predictions.
#
# usage: python pytorch_precision.py <seed>

import sys
import csv
import time

import torch
import torch.nn as nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([float(x) for x in r])
    data = torch.tensor(rows, dtype=torch.float32)
    return data[:, :-1], data[:, -1:]


x, y = load("rosenbrock.csv")

net = nn.Sequential(
    nn.Linear(10, 10), nn.Tanh(),
    nn.Linear(10, 1),
)
for m in net:
    if isinstance(m, nn.Linear):
        nn.init.uniform_(m.weight, -1.0, 1.0)
        nn.init.uniform_(m.bias, -1.0, 1.0)

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
batch_size = 1000
epochs = 10000
n = x.shape[0]

start = time.perf_counter()
for epoch in range(epochs):
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(net(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
print(f"train_time={time.perf_counter() - start}")

with torch.no_grad():
    preds = net(x)

with open("pred_pytorch.txt", "w") as f:
    for v in preds[:, 0].tolist():
        f.write(f"{v:.10f}\n")
