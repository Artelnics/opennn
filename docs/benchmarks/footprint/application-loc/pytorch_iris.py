import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("irisflowers.csv")
X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.astype(np.float32)
y = pd.factorize(df["class"])[0].astype(np.int64)

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

model = nn.Sequential(
    nn.Linear(4, 16),
    nn.Tanh(),
    nn.Linear(16, 3)
)

opt = torch.optim.LBFGS(model.parameters(), lr=1.0)
loss_fn = nn.CrossEntropyLoss()

def closure():
    opt.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    loss.backward()
    return loss

for epoch in range(15):
    opt.step(closure)

with torch.no_grad():
    preds = model(X_test).softmax(dim=1).argmax(dim=1)

cm = torch.zeros(3, 3, dtype=torch.int64)

for t, p in zip(y_test, preds):
    cm[t, p] += 1

print("Confusion matrix:\n", cm)

x = torch.tensor(
    scaler.transform([[5.1, 3.5, 1.4, 0.2]]),
    dtype=torch.float32
)

with torch.no_grad():
    y = model(x).argmax(1)

print("Predicted class:", y.item())

torch.jit.trace(model, torch.randn(1, 4)).save("iris_model.pt")
