# PyTorch startup-latency benchmark: import torch, construct the same small MLP,
# run one forward pass, print the result, exit. Measures time-to-first-prediction.

import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(10, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

with torch.no_grad():
    output = net(torch.ones(1, 10))

print("prediction", float(output[0, 0]))
