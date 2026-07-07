#!/usr/bin/env python3
"""PyTorch ImageClassificationNetwork/CIFAR-10 max training batch trial."""

import argparse
import gc
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageClassificationNet(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        channels = [3, 16, 32, 64, 128]
        blocks = []
        for i in range(4):
            blocks.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            ])
        self.features = nn.Sequential(*blocks)
        self.fc1 = nn.Linear(2 * 2 * 128, 128)
        self.fc2 = nn.Linear(128, classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        return F.softmax(self.fc2(x), dim=1)


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
    args = ap.parse_args()

    assert torch.cuda.is_available(), "CUDA GPU required"
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    xb_np, yb_np, classes = make_batch(args.data, args.batch)
    xb = torch.from_numpy(xb_np).permute(0, 3, 1, 2).contiguous().cuda()
    yb = torch.from_numpy(yb_np).cuda()
    xb = xb.to(memory_format=torch.channels_last)
    del xb_np, yb_np
    gc.collect()

    model = ImageClassificationNet(classes).cuda().to(memory_format=torch.channels_last)
    if args.path == "compile":
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    def train_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        probs = model(xb)
        loss = F.nll_loss(torch.log(probs.clamp_min(1.0e-7)), yb)
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
    print("model=ImageClassificationNetwork-CIFAR")
    print("complexity=16,32,64,128")
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
