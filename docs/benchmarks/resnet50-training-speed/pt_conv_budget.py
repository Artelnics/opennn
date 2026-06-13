# Probe: time PyTorch's convolution kernels alone (fwd + bwd) for the exact
# ResNet-50-on-CIFAR conv stack, CUDA events, channels_last + TF32 +
# cudnn.benchmark — an upper bound on how fast the conv work can go on this
# GPU with PyTorch's engine selection. Compares against OpenNN's conv share.

import torch
import torch.nn as nn
import time

torch.backends.cudnn.benchmark = True
batch = 128

# (in_c, out_c, kernel, stride, spatial_in) for every conv in the network,
# replicated per block, matching opennn::ResNet on 32x32x3.
convs = []

def conv(in_c, out_c, k, s, hw):
    convs.append((in_c, out_c, k, s, hw))

conv(3, 64, 7, 2, 32)                       # stem
hw = 8                                      # after maxpool 16->8
stages = [(64, 256, 3), (128, 512, 4), (256, 1024, 6), (512, 2048, 3)]
in_c = 64
for stage_index, (mid, out, blocks) in enumerate(stages):
    for b in range(blocks):
        s = 2 if (b == 0 and stage_index > 0) else 1
        conv(in_c, mid, 1, 1, hw)           # conv1
        conv(mid, mid, 3, s, hw)            # conv2 (stride here)
        hw_out = hw // s
        conv(mid, out, 1, 1, hw_out)        # conv3
        if s != 1 or in_c != out:
            conv(in_c, out, 1, s, hw)       # projection skip
        in_c = out
        hw = hw_out

total_fwd = 0.0
total_bwd = 0.0
per_shape = {}
start = torch.cuda.Event(enable_timing=True)
stop = torch.cuda.Event(enable_timing=True)

for (in_c, out_c, k, s, hw) in convs:
    layer = nn.Conv2d(in_c, out_c, k, stride=s, padding=k // 2, bias=False) \
                .cuda().to(memory_format=torch.channels_last)
    x = torch.randn(batch, in_c, hw, hw, device="cuda") \
             .to(memory_format=torch.channels_last).requires_grad_(True)

    for _ in range(10):                     # warmup + benchmark-mode search
        y = layer(x)
        y.backward(torch.ones_like(y))
        x.grad = None
        layer.weight.grad = None

    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        y = layer(x)
    stop.record()
    torch.cuda.synchronize()
    fwd_ms = start.elapsed_time(stop) / 50

    g = torch.ones_like(y)
    torch.cuda.synchronize()
    start.record()
    for _ in range(50):
        y = layer(x)
        y.backward(g)
        x.grad = None
        layer.weight.grad = None
    stop.record()
    torch.cuda.synchronize()
    bwd_ms = start.elapsed_time(stop) / 50 - fwd_ms

    total_fwd += fwd_ms
    total_bwd += bwd_ms

    key = f"{hw}x{hw}x{in_c} k{k}x{k}x{out_c} s{s}"
    aggregate = per_shape.setdefault(key, [0.0, 0.0, 0])
    aggregate[0] += fwd_ms
    aggregate[1] += bwd_ms
    aggregate[2] += 1

for key, (fwd_ms, bwd_ms, count) in sorted(per_shape.items(), key=lambda e: -(e[1][0] + e[1][1])):
    print(f"pt_shape {key:<24} n={count} fwd_ms={fwd_ms:.3f} bwd_ms={bwd_ms:.3f}")

print(f"convs={len(convs)}")
print(f"pt_conv_fwd_ms_per_step={total_fwd:.3f}")
print(f"pt_conv_bwd_ms_per_step={total_bwd:.3f}")
print(f"pt_conv_total_ms_per_step={total_fwd + total_bwd:.3f}")
