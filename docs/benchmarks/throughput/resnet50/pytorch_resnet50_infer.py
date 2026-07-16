# PyTorch GPU ResNet-50 inference-speed benchmark on CIFAR-10 (FORWARD ONLY).
#
# Inference twin of pytorch_resnet50_speed.py. Same explicit v1.5 bottleneck
# ResNet-50 (identical to torchvision's resnet50, stride on the 3x3), same
# GPU-resident CIFAR-10 batch. The difference is that the loop is forward-only:
# model.eval() + torch.no_grad(), one batch held on the GPU, warmup then N timed
# forward passes -- no loss, no backward, no optimizer step. This is the fair
# counterpart to OpenNN's device-resident inference path.
#
# Two paths, so the comparison against OpenNN is fair:
#   default     -> plain eager fp32, NCHW (the framework default).
#   PT_FAST=1   -> channels_last (NHWC, the layout OpenNN's cuDNN convs use)
#                  + torch.compile + TF32. PyTorch's optimized fast path, the
#                  fair opponent for OpenNN's resident path.
#
#   usage:  python pytorch_resnet50_infer.py [batch] [runs] [data_dir]
#   env:    PT_FAST=1  -> channels_last + torch.compile + TF32
#           PT_BF16=1  -> torch.autocast(bfloat16) (matches OpenNN's bf16 column)

import sys
import time
import os

import numpy as np
import torch
import torch.nn as nn

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 128
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
data_dir = sys.argv[3] if len(sys.argv) > 3 else "cifar10"
fast = os.environ.get("PT_FAST") is not None
bf16 = os.environ.get("PT_BF16") is not None

assert torch.cuda.is_available(), "CUDA GPU required"
torch.manual_seed(42)
# PT_NOBENCH=1 -> cuDNN heuristic (memory config); default -> autotune (speed config).
torch.backends.cudnn.benchmark = os.environ.get("PT_NOBENCH") is None
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
batch = min(batch, n)
classes = int(y.max().item()) + 1
print(f"device={torch.cuda.get_device_name(0)}")
print(f"path={'fast(channels_last+compile)' if fast else 'eager(NCHW)'}{' +bf16' if bf16 else ''}")
print(f"samples={n} batch={batch} runs={runs} classes={classes}")

model = ResNet50(classes).cuda()
print(f"parameters={sum(p.numel() for p in model.parameters())}")
if fast:
    model = model.to(memory_format=torch.channels_last)
    # reduce-overhead = torch.compile + CUDA graphs, PyTorch's fastest
    # inference execution path and the same-condition counterpart of OpenNN's
    # captured resident forward.
    model = torch.compile(model, mode="reduce-overhead")
model.eval()

# One GPU-resident batch, held constant so the timed loop is pure forward compute.
xb = x[:batch].clone()

# CUDA graph capture/replay (PT_NOGRAPH=1 disables): the same-condition
# counterpart of OpenNN's captured resident forward. Warmup on a side stream,
# capture the eval forward once, replay it in the timed loop.
use_graph = os.environ.get("PT_NOGRAPH") is None and not fast

if use_graph:
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream), torch.no_grad():
        for _ in range(3):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                model(xb)
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            static_out = model(xb)
    print("cuda_graph=on")

    def run_forward():
        graph.replay()
        torch.cuda.synchronize()
        return static_out

else:
    @torch.no_grad()
    def run_forward():
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(xb)
        torch.cuda.synchronize()
        return out


run_forward()  # warmup (traces + compiles + selects cuDNN plans)
run_forward()

times = []
for _ in range(runs):
    t0 = time.perf_counter()
    run_forward()
    times.append(time.perf_counter() - t0)

times.sort()
median = times[len(times) // 2]
print(f"ms_per_batch={median * 1000:.4f}")
print(f"samples_per_sec={batch / median:.0f}")
print("RESULT=OK")
