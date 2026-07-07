# PyTorch accuracy-parity benchmark on the Rosenbrock dataset (10 inputs).
# Same architecture / loss / optimizer / epochs as the OpenNN and TensorFlow
# programs; consumes the shared normalized split and writes test predictions.

import sys
import csv

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


train_x, train_y = load("rosenbrock_train.csv")
test_x, _ = load("rosenbrock_test.csv")

net = nn.Sequential(
    nn.Linear(10, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 1),
)
# Match OpenNN's Glorot/Xavier initialization for fairness.
for m in net:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

optimizer = torch.optim.Adam(net.parameters())
loss_fn = nn.MSELoss()
batch_size = 64
n = train_x.shape[0]

for epoch in range(200):
    perm = torch.randperm(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        optimizer.zero_grad()
        loss = loss_fn(net(train_x[idx]), train_y[idx])
        loss.backward()
        optimizer.step()

with torch.no_grad():
    preds = net(test_x)

with open("pred_pytorch.txt", "w") as f:
    for v in preds[:, 0].tolist():
        f.write(f"{v:.10f}\n")
