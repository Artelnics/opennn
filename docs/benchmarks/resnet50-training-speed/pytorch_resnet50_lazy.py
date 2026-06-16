# PyTorch GPU ResNet-50 training-speed benchmark, ImageNet geometry, LAZY data.
#
# At 224x224x3 the dataset (30 GB fp32) cannot be GPU-resident, so this reads
# the BMP class folders from disk per batch via a DataLoader with worker
# processes -- the fair counterpart to OpenNN's ImageDataset BinaryFile cache.
# Same explicit ResNet-50 v1.5 as pytorch_resnet50_speed.py; eager fp32;
# cross-entropy + Adam; median epoch time after a 1-epoch warmup.
#
#   usage:  python pytorch_resnet50_lazy.py [epochs] [batch] [data_dir] [workers]

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
data_dir = sys.argv[3] if len(sys.argv) > 3 else "imagenet_like"
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8

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
    def __init__(self, classes=1000):
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


# Read the same BMP class folders OpenNN reads; decode per item in worker
# processes so the DataLoader overlaps disk+decode with GPU compute. Plain PIL
# Dataset (no torchvision) to match the rest of this benchmark suite.
class BmpFolder(Dataset):
    def __init__(self, root):
        root = Path(root)
        self.classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        index = {name: i for i, name in enumerate(self.classes)}
        self.items = []
        for name in self.classes:
            for bmp in sorted((root / name).glob("*.bmp")):
                self.items.append((bmp, index[name]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        pixels = np.asarray(Image.open(path), dtype=np.float32) / 255.0  # HWC
        return torch.from_numpy(pixels).permute(2, 0, 1).contiguous(), label


dataset = BmpFolder(f"{data_dir}/train")
classes = len(dataset.classes)
n = len(dataset)
loader = DataLoader(dataset, batch_size=batch, shuffle=True,
                    num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=True, prefetch_factor=4)

print(f"device={torch.cuda.get_device_name(0)}")
print(f"samples={n} batch={batch} epochs={epochs} classes={classes} workers={workers}")

model = ResNet50(classes).cuda()
print(f"parameters={sum(p.numel() for p in model.parameters())}")

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def run_epoch():
    model.train()
    for xb, yb in loader:
        xb = xb.cuda(non_blocking=True)
        yb = yb.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()


run_epoch()  # warmup (also primes the OS page cache for the BMPs)

times = []
for _ in range(epochs):
    t0 = time.perf_counter()
    run_epoch()
    times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(f"epoch_s={median:.4f}")
print(f"samples_per_sec={n / median:.0f}")
print(f"peak_vram_mib={torch.cuda.max_memory_allocated() / 1024**2:.0f}")
print(f"peak_reserved_mib={torch.cuda.max_memory_reserved() / 1024**2:.0f}")
print("RESULT=OK")
