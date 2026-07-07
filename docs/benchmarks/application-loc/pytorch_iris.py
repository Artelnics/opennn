import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1) Load CSV
df = pd.read_csv("irisflowers.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float32)
y = pd.factorize(df["class"])[0].astype(np.int64)

# 2) Split + scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = torch.tensor(X_train)
X_test = torch.tensor(X_test)
y_train = torch.tensor(y_train)
y_test = torch.tensor(y_test)

# 3) Model
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.Tanh(),
    nn.Linear(16, 3)
)

opt = torch.optim.LBFGS(model.parameters(), lr=1.0)
loss_fn = nn.CrossEntropyLoss()

# 4) Training
def closure():
    opt.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    return loss

print("Starting PyTorch training with L-BFGS...")

for epoch in range(15):
    loss = opt.step(closure)
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

# 5) Confusion matrix
with torch.no_grad():
    preds = model(X_test).softmax(dim=1).argmax(dim=1)

cm = torch.zeros(3, 3, dtype=torch.int64)

for t, p in zip(y_test, preds):
    cm[t, p] += 1

print("Confusion matrix:\n", cm)

# 6) Deployment
x = torch.tensor(
    scaler.transform([[5.1, 3.5, 1.4, 0.2]]),
    dtype=torch.float32
)

with torch.no_grad():
    y = model(x).argmax(1)

print("Predicted class:", y.item())

# 7) Export
example = torch.randn(1, 4)
torch.jit.trace(model, example).save("iris_model.pt")

torch.onnx.export(
    model, example, "iris_model.onnx",
    input_names=["input"], output_names=["logits"],
    opset_version=17
)
