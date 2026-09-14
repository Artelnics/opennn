"""Small native-tensor applications through the standard PyTorch Python API."""

import time

entered_ns = time.monotonic_ns()
import sys
import os
import json
import math
import torch
from torch import nn

family, device_name, precision = sys.argv[1:4]
device = torch.device(device_name)
dtype = torch.bfloat16 if precision == "bf16" else torch.float32
torch.manual_seed(42)
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
torch.set_num_interop_threads(1)
models = []


def prepare(model):
    model.eval().to(device=device, dtype=dtype)
    models.append(model)
    return model


with torch.inference_mode():
    if family == "dense":
        model = prepare(
            nn.Sequential(
                nn.Linear(28, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
        )
        output = model(torch.full((2, 28), 0.1, device=device, dtype=dtype))
    elif family == "lstm":
        recurrent = nn.LSTM(15, 128, batch_first=True)
        recurrent.bias_hh_l0.zero_().requires_grad_(False)
        recurrent = prepare(recurrent)
        head = prepare(nn.Linear(128, 1))
        values, _ = recurrent(torch.full((2, 8, 15), 0.1, device=device, dtype=dtype))
        output = head(values[:, -1, :])
    elif family == "cnn":
        model = prepare(
            nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(8 * 8 * 32, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.Softmax(dim=1),
            )
        )
        inputs = torch.full((2, 3, 32, 32), 0.1, device=device, dtype=dtype)
        output = model(inputs / 255.0)
    elif family == "transformer":
        source_embedding = prepare(nn.Embedding(128, 32))
        target_embedding = prepare(nn.Embedding(128, 32))
        encoder = prepare(
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    32,
                    4,
                    dim_feedforward=64,
                    dropout=0.0,
                    layer_norm_eps=1e-6,
                    batch_first=True,
                ),
                1,
            )
        )
        decoder = prepare(
            nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    32,
                    4,
                    dim_feedforward=64,
                    dropout=0.0,
                    layer_norm_eps=1e-6,
                    batch_first=True,
                ),
                1,
            )
        )
        head = prepare(nn.Linear(32, 128))
        tokens = torch.ones((2, 8), device=device, dtype=torch.long)
        positions = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, 32, 2, dtype=torch.float32) * (-math.log(10000.0) / 32)
        )
        angles = positions * frequencies
        pe = (
            torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
            .reshape(1, 8, 32)
            .to(device=device, dtype=dtype)
        )
        source = source_embedding(tokens) * math.sqrt(32.0) + pe
        target = target_embedding(tokens) * math.sqrt(32.0) + pe
        causal = torch.full((8, 8), float("-inf"), device=device, dtype=dtype).triu(1)
        output = head(decoder(target, encoder(source), tgt_mask=causal))
    else:
        raise ValueError("Unknown architecture: " + family)
    actual_device = output.device.type
    output = output.to(device="cpu", dtype=torch.float32, non_blocking=False)
    ready_ns = time.monotonic_ns()

assert actual_device == device_name, "Unexpected output device"
assert all(
    p.device.type == device_name for model in models for p in model.parameters()
), "Unexpected parameter device"
assert bool(torch.isfinite(output).all()), "Nonfinite output"
parameters = sum(
    p.numel() for model in models for p in model.parameters() if p.requires_grad
)
stored = sum(p.numel() for model in models for p in model.parameters())
record = {
    "main_ns": entered_ns,
    "ready_ns": ready_ns,
    "engine": "pytorch",
    "interface": "python",
    "device": actual_device,
    "precision": precision,
    "parameters": parameters,
    "stored_parameters": stored,
    "output_values": output.numel(),
    "first": float(output.flatten()[0]),
    "torch_version": torch.__version__,
    "python_version": sys.version.split()[0],
    "torch_file": torch.__file__,
    "python_prefix": sys.prefix,
    "cuda_version": torch.version.cuda,
    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    "tf32_cudnn": torch.backends.cudnn.allow_tf32,
}
print("STARTUP_READY " + json.dumps(record), flush=True)

if len(sys.argv) > 4:
    from pathlib import Path

    audit = Path(sys.argv[4])
    audit.mkdir(parents=True, exist_ok=True)
    modules = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and Path(file).is_file():
            modules.append(
                {
                    "name": name,
                    "file": str(Path(file).resolve()),
                    "cached": getattr(module, "__cached__", None),
                }
            )
    (audit / "modules.json").write_text(json.dumps(modules, indent=2))
    (audit / "maps.txt").write_text(Path("/proc/self/maps").read_text())
    import sysconfig

    (audit / "python-runtime.json").write_text(
        json.dumps(
            {
                "executable": sys.executable,
                "base_prefix": sys.base_prefix,
                "prefix": sys.prefix,
                "paths": sysconfig.get_paths(),
                "sys_path": sys.path,
            },
            indent=2,
        )
    )
    # Inventory packages after capturing the application imports and mappings.
    import importlib.metadata

    packages = sorted(
        [
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions()
        ],
        key=lambda d: d["name"].lower(),
    )
    (audit / "packages.json").write_text(json.dumps(packages, indent=2))
