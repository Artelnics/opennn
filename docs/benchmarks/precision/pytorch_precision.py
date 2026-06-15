# PyTorch precision benchmark on the Rosenbrock dataset (10 inputs), after the
# Neural Designer blog protocol: 10 -> 10 (tanh) -> 1 (linear), U(-1, 1) init,
# MSE, trained on all samples. Prints the training wall time and writes
# full-dataset predictions.
#
# Two optimizers, so the comparison reports the error PyTorch reaches with the
# BEST optimizer it actually ships:
#   Adam    -> first-order, lr 0.001, batch 1000, 10,000 epochs (the default path).
#   LBFGS   -> torch.optim.LBFGS, PyTorch's only built-in second-order/quasi-Newton
#              optimizer. Full-batch, closure-based (the API forces a closure that
#              re-evaluates the loss -- not a drop-in replacement for Adam's loop).
#              This is the like-for-like opponent to OpenNN's QuasiNewton/LM.
# TensorFlow's core keras.optimizers has no second-order option at all.
#
# usage: python pytorch_precision.py <seed> [Adam|LBFGS] [epochs]

import sys
import csv
import time

import torch
import torch.nn as nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
which = sys.argv[2] if len(sys.argv) > 2 else "Adam"
epochs = int(sys.argv[3]) if len(sys.argv) > 3 else (10000 if which == "Adam" else 1000)
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

loss_fn = nn.MSELoss()
n = x.shape[0]

start = time.perf_counter()
if which == "LBFGS":
    # PyTorch's built-in second-order optimizer. Full-batch, closure-based.
    optimizer = torch.optim.LBFGS(net.parameters(), lr=1.0, max_iter=20,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        return loss

    for epoch in range(epochs):
        optimizer.step(closure)
else:
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    batch_size = 1000
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(net(x[idx]), y[idx])
            loss.backward()
            optimizer.step()
print(f"optimizer={which}")
print(f"train_time={time.perf_counter() - start}")

with torch.no_grad():
    preds = net(x)

with open("pred_pytorch.txt", "w") as f:
    for v in preds[:, 0].tolist():
        f.write(f"{v:.10f}\n")
