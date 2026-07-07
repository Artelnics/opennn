# PyTorch HIGGS dense max-batch counterpart to opennn_higgs_maxbatch_trial.
# Same canonical model (28 -> hidden -> hidden -> 1, ReLU hidden, binary
# cross-entropy, Adam -- see docs/benchmarks/higgs/README.md). The output
# layer produces logits and the loss is BCEWithLogitsLoss, the standard
# most-optimized PyTorch formulation of the same objective (OpenNN computes
# the sigmoid explicitly and uses binary cross-entropy).
#
# Most-optimized PyTorch config: TF32 matmul/cudnn (fp32 path), autocast(bf16)
# (bf16 path), fused Adam, cudnn.benchmark, optional torch.compile
# (PT_COMPILE=1).
#
# mode "train": one training step per --steps (forward + backward + Adam).
# mode "infer": forward-only under torch.inference_mode() (no autograd graph,
# no optimizer state), matching OpenNN's device-resident inference path.
#
# Data is synthetic with the HIGGS contract shapes: capacity depends on the
# shapes and the step, not the feature values (randn matches the standardized
# prepared HIGGS features; labels are random {0,1}).
#
#   usage: python pytorch_higgs_maxbatch.py --mode train|infer --batch B
#              [--hidden 1024] [--layers 2] [--steps N] [--warmup W]
#              [--device cuda|cpu]
#   env:   PT_BF16=1 -> autocast(bf16, CUDA only);  PT_COMPILE=1 -> torch.compile

import argparse, os, time
import torch
import torch.nn as nn

INPUTS = 28   # HIGGS contract: 28 features, 1 target

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["train", "infer"], default="train")
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--hidden", type=int, default=1024)
ap.add_argument("--layers", type=int, default=2)
ap.add_argument("--steps", type=int, default=1)
ap.add_argument("--warmup", type=int, default=1)
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
args = ap.parse_args()

use_cuda = args.device == "cuda"
if use_cuda:
    assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device(args.device)
torch.manual_seed(0)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def sync():
    if use_cuda:
        torch.cuda.synchronize()


use_bf16 = use_cuda and os.environ.get("PT_BF16") is not None
use_compile = os.environ.get("PT_COMPILE") is not None
print(f"precision={'bf16' if use_bf16 else 'fp32'} mode={args.mode} "
      f"device={args.device} "
      f"inputs={INPUTS} hidden={args.hidden} hidden_layers={args.layers} "
      f"batch={args.batch} steps={args.steps} compile={use_compile}")

layers = []
width = INPUTS
for _ in range(args.layers):
    layers += [nn.Linear(width, args.hidden), nn.ReLU()]
    width = args.hidden
layers.append(nn.Linear(width, 1))   # logits; sigmoid fused into the loss
model = nn.Sequential(*layers).to(dev)
model.train(args.mode == "train")
print(f"parameters={sum(p.numel() for p in model.parameters())}")

step_fn = model
if use_compile:
    step_fn = torch.compile(model)

ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 \
    else torch.autocast("cuda", enabled=False)

# Real HIGGS rows (float32 bin, rows x 29: features then label) when
# HIGGS_BIN is set; rows repeat modulo beyond the file (np.resize), the same
# convention as the ResNet-50 capacity runner. Synthetic otherwise.
higgs_bin = os.environ.get("HIGGS_BIN")
if higgs_bin:
    import numpy as np
    raw = np.fromfile(higgs_bin, dtype=np.float32).reshape(-1, INPUTS + 1)
    print(f"data=higgs_bin rows={raw.shape[0]}")
    x = torch.from_numpy(np.resize(np.ascontiguousarray(raw[:, :INPUTS]),
                                   (args.batch, INPUTS))).to(dev)
    y_host = np.resize(np.ascontiguousarray(raw[:, INPUTS:]), (args.batch, 1))
else:
    print("data=synthetic")
    x = torch.randn(args.batch, INPUTS, device=dev)
    y_host = None
sync()

if args.mode == "train":
    y = torch.from_numpy(y_host).to(dev) if y_host is not None \
        else torch.randint(0, 2, (args.batch, 1), device=dev, dtype=torch.float32)

    try:
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, fused=True)
    except (RuntimeError, ValueError):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        with ctx:
            loss = loss_fn(step_fn(x), y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(args.warmup):
        step()
    sync()

    t0 = time.perf_counter()
    last = None
    for _ in range(args.steps):
        last = step()
    sync()
    wall_s = time.perf_counter() - t0

    assert torch.isfinite(last), "non-finite loss"
    print(f"final_loss={float(last):.5f}")
else:
    with torch.inference_mode():
        out = None
        for _ in range(args.warmup):
            with ctx:
                out = step_fn(x)
        sync()

        t0 = time.perf_counter()
        for _ in range(args.steps):
            with ctx:
                out = step_fn(x)
        sync()
        wall_s = time.perf_counter() - t0

        assert torch.isfinite(out.flatten()[:8].float()).all(), "non-finite outputs"

samples_per_s = args.steps * args.batch / wall_s
try:   # peak memory for the CPU-capped runs (POSIX only)
    import resource
    print(f"peak_rss_mib={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024}")
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmPeak:"):
                print(f"vm_peak_mib={int(line.split()[1]) // 1024}")
                break
except Exception:
    pass
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print("RESULT=OK")
