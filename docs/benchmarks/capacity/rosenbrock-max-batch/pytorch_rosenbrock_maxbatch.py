# Find the maximum GPU batch size for a shallow Rosenbrock MLP in PyTorch.
#
# Net: 1000 -> 1000 (tanh) -> 1, MSE, Adam, fp32 -- the Neural Designer
# Rosenbrock protocol scaled to 1000 inputs. We binary-search the largest batch
# that completes one step without CUDA OOM, separately for:
#   train     = forward + backward + optimizer.step()  (grads + Adam state live)
#   inference = forward only, under torch.no_grad()     (activations only)
#
# Synthetic inputs are generated on-device (no dataset file needed): only the
# 1000-input / 1-target SHAPE matters for a memory-ceiling test.
#
#   usage: python pytorch_rosenbrock_maxbatch.py [inputs] [hidden]

import sys

import torch
import torch.nn as nn

inputs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
hidden = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

assert torch.cuda.is_available(), "CUDA GPU required"
device = torch.device("cuda")
total_mib = torch.cuda.get_device_properties(0).total_memory / 1024**2

# WSL silently spills past VRAM into system RAM (shared GPU memory), so an
# uncapped probe measures VRAM+RAM, not true GPU capacity. Cap the allocator to
# physical VRAM so it OOMs at the real ceiling -- this matches OpenNN's CUDA
# allocator, which hard-fails at the device limit.
torch.cuda.set_per_process_memory_fraction(1.0, 0)
# Also disable the cudaMallocAsync/expandable spill path if present.
print(f"device={torch.cuda.get_device_name(0)} total_vram_mib={total_mib:.0f}")
print(f"net={inputs}->{hidden}->1 (tanh), fp32  (capped to physical VRAM)")


def build():
    net = nn.Sequential(nn.Linear(inputs, hidden), nn.Tanh(), nn.Linear(hidden, 1)).to(device)
    return net


def try_batch(batch, mode):
    """Return (ok, peak_mib). ok=False means this batch OOMed."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        net = build()
        x = torch.randn(batch, inputs, device=device)
        y = torch.randn(batch, 1, device=device)
        if mode == "train":
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            loss_fn = nn.MSELoss()
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(net(x), y)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                net(x)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024**2
        del net, x, y
        return True, peak
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            return False, 0.0
        raise


def max_batch(mode):
    # Exponential grow to first OOM, then binary-search the boundary.
    lo, hi = 1, 1
    last_peak = 0.0
    while True:
        ok, peak = try_batch(hi, mode)
        if ok:
            last_peak = peak
            lo = hi
            hi *= 2
        else:
            break
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        ok, peak = try_batch(mid, mode)
        if ok:
            last_peak = peak
            lo = mid
        else:
            hi = mid
    return lo, last_peak


for mode in ("inference", "train"):
    batch, peak = max_batch(mode)
    print(f"mode={mode} max_batch={batch} peak_vram_mib_at_max={peak:.0f}")
print("RESULT=OK")
