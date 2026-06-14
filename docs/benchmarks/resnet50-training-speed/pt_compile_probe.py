# PyTorch torch.compile counterpart for the ResNet-50 benchmark.
#
# This is the optimizing PyTorch path used by the headline benchmark, so the
# shell runner includes it alongside OpenNN and PyTorch eager.

import sys
import time

import numpy as np
import torch
import torch.nn as nn

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
data_dir = sys.argv[3] if len(sys.argv) > 3 else "cifar10"

assert torch.cuda.is_available(), "CUDA GPU required"
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid, stride=1):
        super().__init__()
        out = mid * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)
        self.conv3 = nn.Conv2d(mid, out, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out))

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.relu(self.bn2(self.conv2(y)))
        y = self.bn3(self.conv3(y))
        return self.relu(y + identity)


class ResNet50(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        layers = []
        in_channels = 64
        for stage, (mid, blocks) in enumerate(zip([64, 128, 256, 512], [3, 4, 6, 3])):
            for block in range(blocks):
                layers.append(Bottleneck(in_channels, mid, 2 if (block == 0 and stage > 0) else 1))
                in_channels = mid * Bottleneck.expansion
        self.stages = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stages(x)
        return self.fc(torch.flatten(self.avgpool(x), 1))


x = (torch.from_numpy(np.load(f"{data_dir}/cifar_images.npy"))
     .permute(0, 3, 1, 2).div(255.0).contiguous().cuda())
y = torch.from_numpy(np.load(f"{data_dir}/cifar_labels.npy")).cuda()
n = x.shape[0]
classes = int(y.max().item()) + 1
print(f"device={torch.cuda.get_device_name(0)}")
print(f"samples={n} batch={batch} epochs={epochs} classes={classes}")

model = ResNet50(classes).cuda()
print(f"parameters={sum(p.numel() for p in model.parameters())}")
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
starts = list(range(0, n - batch + 1, batch))


@torch.compile
def train_step(xb, yb):
    optimizer.zero_grad(set_to_none=True)
    loss = loss_fn(model(xb), yb)
    loss.backward()
    optimizer.step()


def run_epoch():
    model.train()
    perm = torch.randperm(n, device="cuda")
    for s in starts:
        idx = perm[s:s + batch]
        train_step(x[idx], y[idx])
    torch.cuda.synchronize()


print("compiling...")
run_epoch()
run_epoch()

times = []
for _ in range(epochs):
    t0 = time.perf_counter()
    run_epoch()
    times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(f"epoch_s={median:.4f}")
print(f"samples_per_sec={n / median:.0f}")
print("RESULT=OK")
