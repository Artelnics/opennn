"""The PyTorch counterpart of `examples/breast_cancer/main.cpp`.

This file exists as evidence, not as a benchmark.  The claim it supports is
"how much do you write to use each library", and that claim was previously made
with a number no file in this repository backed.  Here both programs are in the
tree, so anyone can disagree with the count by reading them.

The rule it was written under: **as short as honesty allows.**  Padding the
PyTorch side would make the comparison worthless, so there is no argument
parsing, no logging, no configurability, no type annotations beyond what the
code needs -- nothing the OpenNN example does not also have.  Where PyTorch
offers a terser idiom it is used.  What is left is the work `Training`
and `Evaluation` do on the other side and that the caller must do here:
splitting the data, scaling it, the epoch loop, the gradient step, and the
binary-classification report.

Deliberately NOT counted against PyTorch, because the OpenNN example does not
do them either: early stopping, validation-based model selection, and any
device placement beyond the CPU default.

Run with: python3 breast_cancer_pytorch.py ../../examples/data/breast_cancer/breast_cancer.csv
"""

import sys

import pandas as pd
import torch

frame = pd.read_csv(sys.argv[1], sep=";")
inputs = torch.tensor(frame.iloc[:, :-1].to_numpy(), dtype=torch.float32)
targets = torch.tensor(frame.iloc[:, -1:].to_numpy(), dtype=torch.float32)

split = int(0.8 * len(inputs))
permutation = torch.randperm(len(inputs))
train, test = permutation[:split], permutation[split:]

mean, deviation = inputs[train].mean(0), inputs[train].std(0).clamp(min=1e-8)
inputs = (inputs - mean) / deviation

network = torch.nn.Sequential(torch.nn.Linear(inputs.shape[1], 3),
                              torch.nn.Tanh(),
                              torch.nn.Linear(3, 1))

loss_function = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(network.parameters())

for _ in range(1000):
    optimizer.zero_grad()
    loss = loss_function(network(inputs[train]), targets[train])
    loss.backward()
    optimizer.step()

with torch.no_grad():
    predictions = (network(inputs[test]) > 0).float()

actual = targets[test]
true_positive = int(((predictions == 1) & (actual == 1)).sum())
true_negative = int(((predictions == 0) & (actual == 0)).sum())
false_positive = int(((predictions == 1) & (actual == 0)).sum())
false_negative = int(((predictions == 0) & (actual == 1)).sum())

print(f"confusion matrix: {true_positive} {false_positive} {false_negative} {true_negative}")
print(f"accuracy:    {(true_positive + true_negative) / len(actual):.4f}")
print(f"sensitivity: {true_positive / max(true_positive + false_negative, 1):.4f}")
print(f"specificity: {true_negative / max(true_negative + false_positive, 1):.4f}")
