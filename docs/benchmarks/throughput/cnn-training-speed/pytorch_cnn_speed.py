# PyTorch GPU CNN training-speed benchmark on MNIST.
#
# Same model as the OpenNN and TensorFlow programs: 28x28x1 -> Conv 16@3x3
# (Same, ReLU) -> MaxPool 2x2 -> Flatten -> Dense 10, cross-entropy, Adam,
# batch 128. The dataset is GPU-resident and reshuffled every epoch. Reports
# the median epoch time after a 2-epoch warmup.
#
# Two paths, so the comparison against OpenNN is fair:
#   default     -> plain eager fp32, NCHW (the framework default).
#   PT_FAST=1   -> channels_last (NHWC, the layout OpenNN's cuDNN convs use)
#                  + torch.compile (kernel fusion / CUDA graphs). PyTorch's
#                  optimized fast path.
#
#   usage:  python pytorch_cnn_speed.py [epochs] [batch]
#   env:    PT_FAST=1  -> channels_last + torch.compile + TF32
#           PT_BF16=1  -> torch.autocast(bfloat16) around the step (matches
#                         OpenNN's bf16 training_type for a precision-fair column)

import sys
import time
import os

import numpy as np
import torch

epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 128
fast = os.environ.get("PT_FAST") is not None
bf16 = os.environ.get("PT_BF16") is not None

assert torch.cuda.is_available(), "CUDA GPU required"
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
if fast:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

x = torch.from_numpy(np.load("mnist_images.npy")).permute(0, 3, 1, 2).div(255.0).contiguous().cuda()
y = torch.from_numpy(np.load("mnist_labels.npy")).cuda()
if fast:
    # NHWC / channels_last: the tensor-core-friendly layout OpenNN's convs use.
    x = x.to(memory_format=torch.channels_last)
n = x.shape[0]
print(f"device={torch.cuda.get_device_name(0)}")
print(f"path={'fast(channels_last+compile)' if fast else 'eager(NCHW)'}{' +bf16' if bf16 else ''}")
print(f"samples={n} batch={batch} epochs={epochs}")

model = torch.nn.Sequential(
    torch.nn.Conv2d(1, 16, kernel_size=3, padding="same"),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Flatten(),
    torch.nn.Linear(14 * 14 * 16, 10),
).cuda()
if fast:
    model = model.to(memory_format=torch.channels_last)
    model = torch.compile(model)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

starts = list(range(0, n - batch + 1, batch))


def run_epoch():
    model.train()
    perm = torch.randperm(n, device="cuda")
    for s in starts:
        idx = perm[s:s + batch]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
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
