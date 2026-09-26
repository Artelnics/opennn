"""Generate and load identical named FP32 tensors for both quality drivers.

The artifact is independent of either engine's random-number generator. Its
NumPy PCG64 stream initializes Linear/attention weights with Xavier uniform,
convolutions with fan-out Kaiming normal, embeddings with N(0, 1), and LSTM
parameters with U(-1/sqrt(hidden), 1/sqrt(hidden)). The ResNet classifier uses
U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for both weight and bias, matching the
topology's reference initialization scale. Other biases are zero except the
single trainable LSTM bias, which uses that LSTM distribution. Normalization
starts from identity, including shared running mean/variance. The unused
PyTorch LSTM recurrent bias is exactly zero and remains frozen.

Each file holds one C-order, little-endian float32 tensor. Names and logical
shapes follow quality.py's state_dict; the C++ importer translates layouts.
This is a controlled initialization policy, not either engine's defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def _digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _build(manifest):
    import torch

    driver = Path(__file__).resolve().parents[1] / "families" / "quality.py"
    spec = importlib.util.spec_from_file_location("quality_initialization_model", driver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Constructing the shape template must not alter the caller's RNG stream.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return module.build(manifest)


def _initial_arrays(model, seed, model_kind):
    from torch import nn

    rng = np.random.Generator(np.random.PCG64(seed))
    modules = dict(model.named_modules())
    for name, tensor in sorted(model.state_dict().items()):
        shape = tuple(tensor.shape)
        owner, _, member = name.rpartition(".")
        layer = modules[owner]
        if name == "position":
            sequence, width = shape
            positions = np.arange(sequence, dtype=np.float64)[:, None]
            frequencies = np.exp(
                np.arange(0, width, 2, dtype=np.float64)
                * (-math.log(10000.0) / width)
            )
            values = np.empty(shape, dtype=np.float64)
            values[:, 0::2] = np.sin(positions * frequencies)
            values[:, 1::2] = np.cos(positions * frequencies)
        elif isinstance(layer, (nn.LayerNorm, nn.modules.batchnorm._BatchNorm)):
            if member in ("weight", "running_var"):
                values = np.ones(shape)
            elif member in ("bias", "running_mean", "num_batches_tracked"):
                values = np.zeros(shape)
            else:
                raise ValueError(f"Unsupported normalization tensor: {name}")
        elif isinstance(layer, nn.LSTM):
            if member == "bias_hh_l0":
                values = np.zeros(shape)
            elif member in ("weight_ih_l0", "weight_hh_l0", "bias_ih_l0"):
                bound = 1 / math.sqrt(layer.hidden_size)
                values = rng.uniform(-bound, bound, shape)
            else:
                raise ValueError(f"Unsupported LSTM tensor: {name}")
        elif isinstance(layer, nn.Embedding) and member == "weight":
            values = rng.standard_normal(shape)
        elif isinstance(layer, nn.Conv2d) and member == "weight":
            fan_out = shape[0] * math.prod(shape[2:])
            values = rng.standard_normal(shape) * math.sqrt(2 / fan_out)
        elif model_kind == "cnn" and owner == "fc" and isinstance(layer, nn.Linear):
            bound = 1 / math.sqrt(layer.in_features)
            values = rng.uniform(-bound, bound, shape)
        elif isinstance(layer, (nn.Linear, nn.Conv2d, nn.MultiheadAttention)):
            if member in ("bias", "in_proj_bias"):
                values = np.zeros(shape)
            elif member in ("weight", "in_proj_weight") and len(shape) == 2:
                bound = math.sqrt(6 / (shape[0] + shape[1]))
                values = rng.uniform(-bound, bound, shape)
            else:
                raise ValueError(f"Unsupported weight tensor: {name}")
        else:
            raise ValueError(f"No shared initialization policy for {name}")
        yield name, np.asarray(values, dtype="<f4")


def create_initialization(manifest, output, seed, model=None):
    """Write a fresh artifact and return its manifest path; never overwrite one."""
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Initialization seed must be a nonnegative integer")
    model = _build(manifest) if model is None else model
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Initialization directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema": 1,
        "model": manifest["model"],
        "profile": manifest["profile"],
        "seed": seed,
        "initializer": "numpy-pcg64-v1",
        "numpy_version": np.__version__,
        "tensors": [],
    }
    for index, (name, values) in enumerate(
        _initial_arrays(model, seed, manifest["model"])
    ):
        path = output / f"{index:04d}.bin"
        values.tofile(path)
        record["tensors"].append(
            {
                "name": name,
                "shape": list(values.shape),
                "file": path.name,
                "dtype": "float32",
                "sha256": _digest(path),
            }
        )
    path = output / "manifest.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def validate_initialization(manifest, artifact, model=None, expected_seed=None):
    """Validate every tensor before returning arrays suitable for a strict load."""
    model = _build(manifest) if model is None else model
    path = Path(artifact).resolve()
    if path.is_dir():
        path /= "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != 1
        or record.get("model") != manifest["model"]
        or record.get("profile") != manifest["profile"]
        or record.get("initializer") != "numpy-pcg64-v1"
    ):
        raise ValueError("Shared initialization schema, architecture or policy mismatch")
    seed = record.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Invalid shared initialization seed")
    if expected_seed is not None and seed != expected_seed:
        raise ValueError("Shared initialization seed differs from the trial seed")
    expected = model.state_dict()
    arrays = {}
    files = set()
    for spec in record["tensors"]:
        name = spec["name"]
        if name in arrays or name not in expected:
            raise ValueError(f"Duplicate or unexpected initialization tensor: {name}")
        shape = spec["shape"]
        if (
            not isinstance(shape, list)
            or any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in shape)
            or shape != list(expected[name].shape)
            or spec.get("dtype") != "float32"
        ):
            raise ValueError(f"Initialization shape or dtype mismatch: {name}")
        item = (path.parent / spec["file"]).resolve()
        if not item.is_relative_to(path.parent) or item in files:
            raise ValueError(f"Invalid or repeated initialization file: {name}")
        files.add(item)
        if item.stat().st_size != math.prod(shape) * 4 or _digest(item) != spec["sha256"]:
            raise ValueError(f"Initialization tensor identity mismatch: {name}")
        values = np.fromfile(item, dtype="<f4").reshape(shape)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite initialization tensor: {name}")
        if name.endswith("num_batches_tracked") and np.any(values != 0):
            raise ValueError("Shared BatchNorm batch counters must start at zero")
        if name == "lstm.bias_hh_l0" and np.any(values != 0):
            raise ValueError("Shared LSTM recurrent bias must be zero")
        if not expected[name].is_floating_point() and np.any(values != np.floor(values)):
            raise ValueError(f"Nonintegral initialization buffer: {name}")
        arrays[name] = values
    if arrays.keys() != expected.keys():
        missing = sorted(expected.keys() - arrays.keys())
        raise ValueError(f"Missing initialization tensors: {missing}")
    provenance = {
        "mode": "shared",
        "manifest": str(path),
        "manifest_sha256": _digest(path),
        "seed": seed,
        "tensor_count": len(arrays),
    }
    return arrays, provenance


def load_initialization(model, manifest, artifact, expected_seed=None):
    """Install validated shared parameters and buffers into a CPU or CUDA model."""
    import torch

    arrays, provenance = validate_initialization(
        manifest, artifact, model=model, expected_seed=expected_seed
    )
    state = model.state_dict()
    model.load_state_dict(
        {
            name: torch.from_numpy(values).to(dtype=state[name].dtype)
            for name, values in arrays.items()
        },
        strict=True,
    )
    return provenance


def export_model_tensors(model, manifest, output, seed, gradients=False):
    """Write canonical tensors for cross-engine fixed-batch diagnostics."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = (
        {name: parameter.grad for name, parameter in model.named_parameters()
         if parameter.requires_grad}
        if gradients else model.state_dict()
    )
    record = {
        "schema": 1,
        "model": manifest["model"],
        "profile": manifest["profile"],
        "seed": seed,
        "kind": "gradients" if gradients else "parameters",
        "tensors": [],
    }
    for index, (name, tensor) in enumerate(sorted(source.items())):
        if tensor is None:
            raise ValueError(f"Missing diagnostic gradient: {name}")
        values = tensor.detach().float().cpu().numpy().astype("<f4")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite diagnostic tensor: {name}")
        path = output / f"{index:04d}.bin"
        values.tofile(path)
        record["tensors"].append(
            {
                "name": name,
                "shape": list(values.shape),
                "file": path.name,
                "dtype": "float32",
                "sha256": _digest(path),
            }
        )
    (output / "manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Generate a shared initialization")
    create.add_argument("--manifest", type=Path, required=True)
    create.add_argument("--out", type=Path, required=True)
    create.add_argument("--seed", type=int, required=True)
    verify = commands.add_parser("verify", help="Validate an existing initialization")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--initialization", type=Path, required=True)
    verify.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.command == "create":
        print(create_initialization(manifest, args.out, args.seed))
    else:
        _, provenance = validate_initialization(
            manifest, args.initialization, expected_seed=args.seed
        )
        print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
