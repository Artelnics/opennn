#!/usr/bin/env python3
"""PyTorch ResNet-50/CIFAR-10 max training batch trial.

One process tests exactly one batch size. The driver runs this repeatedly in
fresh processes so CUDA OOMs do not corrupt later attempts.
"""

import argparse
import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn


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
                nn.BatchNorm2d(out),
            )

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


def make_batch(data_dir, batch):
    images = np.load(os.path.join(data_dir, "cifar_images.npy"), mmap_mode="r")
    labels = np.load(os.path.join(data_dir, "cifar_labels.npy"), mmap_mode="r")
    idx = np.arange(batch, dtype=np.int64) % images.shape[0]
    xb = np.asarray(images[idx], dtype=np.float32) / 255.0
    yb = np.asarray(labels[idx], dtype=np.int64)
    return xb, yb, int(labels.max()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../throughput/resnet50-training-speed/cifar10")
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--path", choices=["compile", "eager"], default="compile")
    ap.add_argument("--memory-fraction", type=float, default=None)
    # 1 = speed config (cuDNN autotune picks fastest algo, more scratch);
    # 0 = memory config (heuristic, avoids autotune scratch dominating capacity).
    ap.add_argument("--cudnn-benchmark", type=int, default=0)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    xb_np, yb_np, classes = make_batch(args.data, args.batch)
    xb = torch.from_numpy(xb_np).permute(0, 3, 1, 2).contiguous().cuda()
    yb = torch.from_numpy(yb_np).cuda()
    xb = xb.to(memory_format=torch.channels_last)
    del xb_np, yb_np
    gc.collect()

    model = ResNet50(classes).cuda().to(memory_format=torch.channels_last)
    if args.path == "compile":
        model = torch.compile(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb)
        if not torch.isfinite(loss):
            raise RuntimeError("loss is not finite")
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        return float(loss.detach().cpu())

    loss0 = train_step()
    loss1 = train_step()

    print(f"engine=pytorch_{args.path}")
    print(f"path={args.path}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"samples={args.batch} batch={args.batch} precision=fp32 classes={classes}")
    print(f"parameters={sum(p.numel() for p in model.parameters())}")
    print(f"loss_warmup={loss0:.6g}")
    print(f"loss_final={loss1:.6g}")
    print("RESULT=OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAIL : {exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise SystemExit(1)
