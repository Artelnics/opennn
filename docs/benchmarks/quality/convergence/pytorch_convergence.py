# PyTorch convergence-gate benchmark on the Rosenbrock dataset (10 inputs).
#
# MLPerf-style metric: wall-clock time to reach a fixed training-MSE target,
# plus the held-out TEST MSE at the stopping point. Same architecture / data /
# optimizer / target as opennn_convergence and tensorflow_convergence, so the
# only thing being compared is how fast each engine reaches the same quality.
#
#   usage: python pytorch_convergence.py [seed] [target_mse] [max_epochs] [lr]

import sys
import csv
import time

import torch
import torch.nn as nn

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
target_mse = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05
max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
lr = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-3
torch.manual_seed(seed)


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            rows.append([float(x) for x in r])
    data = torch.tensor(rows, dtype=torch.float32)
    return data[:, :-1], data[:, -1:]


train_x, train_y = load("rosenbrock_train.csv")
test_x, test_y = load("rosenbrock_test.csv")

net = nn.Sequential(
    nn.Linear(10, 50), nn.Tanh(),
    nn.Linear(50, 50), nn.Tanh(),
    nn.Linear(50, 1),
)
for m in net:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_fn = nn.MSELoss()
batch_size = 64
n = train_x.shape[0]

# The convergence gate is the HELD-OUT test MSE, not the training loss (matches
# the OpenNN and TF drivers): evaluate the test set after each epoch and stop
# when it reaches the target. Evaluation time is excluded from the clock.
reached = False
epochs_taken = max_epochs
final_train_mse = float("nan")
test_mse = float("nan")
train_s = 0.0
for epoch in range(1, max_epochs + 1):
    perm = torch.randperm(n)
    sq_sum, count = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        optimizer.zero_grad()
        out = net(train_x[idx])
        loss = loss_fn(out, train_y[idx])
        loss.backward()
        optimizer.step()
        sq_sum += loss.item() * idx.numel()
        count += idx.numel()
    train_s += time.perf_counter() - t0
    final_train_mse = sq_sum / count
    with torch.no_grad():
        test_mse = loss_fn(net(test_x), test_y).item()
    if test_mse <= target_mse:
        reached = True
        epochs_taken = epoch
        break
time_to_target = train_s

print(f"target_mse={target_mse}")
print(f"reached_goal={1 if reached else 0}")
print(f"epochs_to_target={epochs_taken}")
print(f"final_train_mse={final_train_mse:.10f}")
print(f"test_mse={test_mse:.10f}")
print(f"time_to_target_s={time_to_target:.6f}")
print(f"RESULT={'OK' if reached else 'DID_NOT_CONVERGE'}")
