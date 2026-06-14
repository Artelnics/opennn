# PyTorch GPU ResNet-50 training-speed benchmark on CIFAR-10.
#
# The model is written out explicitly (identical to torchvision's resnet50,
# v1.5 bottleneck with the stride on the 3x3) so the architecture matches the
# OpenNN program exactly without a torchvision dependency. CIFAR-10 is held
# GPU-resident; cross-entropy + Adam; median epoch time after a 2-epoch warmup.
#
# Two paths, so the comparison against OpenNN is fair:
#   default     -> plain eager fp32, NCHW (the framework default).
#   PT_FAST=1   -> channels_last (NHWC, the layout OpenNN's cuDNN convs use)
#                  + torch.compile + TF32. PyTorch's optimized fast path, the
#                  fair opponent for OpenNN's CUDA-graph path.
#
#   usage:  python pytorch_resnet50_speed.py [epochs] [batch] [data_dir]
#   env:    PT_FAST=1  -> channels_last + torch.compile + TF32

import sys
import time
import os

import numpy as np
import torch
import torch.nn as nn

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
data_dir = sys.argv[3] if len(sys.argv) > 3 else "cifar10"
fast = os.environ.get("PT_FAST") is not None

assert torch.cuda.is_available(), "CUDA GPU required"
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
if fast:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


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
                stride = 2 if (block == 0 and stage > 0) else 1
                layers.append(Bottleneck(in_channels, mid, stride))
                in_channels = mid * Bottleneck.expansion
        self.stages = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.stages(x)
        x = torch.flatten(self.avgpool(x), 1)
        return self.fc(x)


x = (torch.from_numpy(np.load(f"{data_dir}/cifar_images.npy"))
     .permute(0, 3, 1, 2).div(255.0).contiguous().cuda())
y = torch.from_numpy(np.load(f"{data_dir}/cifar_labels.npy")).cuda()
if fast:
    # NHWC / channels_last: the tensor-core-friendly layout OpenNN's convs use.
    x = x.to(memory_format=torch.channels_last)
n = x.shape[0]
classes = int(y.max().item()) + 1
print(f"device={torch.cuda.get_device_name(0)}")
print(f"path={'fast(channels_last+compile)' if fast else 'eager(NCHW)'}")
print(f"samples={n} batch={batch} epochs={epochs} classes={classes}")

model = ResNet50(classes).cuda()
print(f"parameters={sum(p.numel() for p in model.parameters())}")
if fast:
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
starts = list(range(0, n - batch + 1, batch))


def run_epoch():
    model.train()
    perm = torch.randperm(n, device="cuda")
    for s in starts:
        idx = perm[s:s + batch]
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()


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
