#!/usr/bin/env python3
"""Train each reference twice, check repeatability and Python export, record hashes."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error("Generated models must remain outside the checkout")
    if output.exists() and any(output.iterdir()):
        parser.error("Output must be empty; previous model records will not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    data_report = json.loads((args.data_root / "reproduction.json").read_text())
    report = {
        "schema_version": 1,
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generator_source_sha256": sha(Path(__file__).with_name("main.cpp")),
        "executable_sha256": sha(args.executable),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "models": {},
    }
    for name, relative in (("iris", "iris_plant/iris_plant_original.csv"), ("concrete", "concrete/concrete_uci.csv")):
        data = args.data_root / relative
        provenance = data_report["datasets"][name]
        if sha(data) != provenance["files"][data.name]:
            raise ValueError(f"Reconstructed training data changed: {data}")
        runs = [output / f"{name}-{i}" for i in (1, 2)]
        for directory in runs:
            subprocess.run([str(args.executable.resolve()), name, str(data.resolve()), str(directory)], check=True)
        checked = ("model.json", "model.bin", "training.json", "split.csv", "reference.csv")
        for filename in checked:
            if sha(runs[0] / filename) != sha(runs[1] / filename):
                raise ValueError(f"Repeated generation differs for {name}/{filename}")
        metadata = json.loads((runs[0] / "generation.json").read_text())
        spec = importlib.util.spec_from_file_location(f"reproduced_{name}", runs[0] / "model.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.NeuralNetwork()
        reference = np.loadtxt(runs[0] / "reference.csv", delimiter=",")
        columns = metadata["input_columns"]
        predicted = np.asarray([model.calculate_outputs(row[:columns].tolist()) for row in reference])
        expected = reference[:, columns:]
        np.testing.assert_allclose(predicted, expected, rtol=1e-4, atol=1e-5)
        report["models"][name] = {
            **metadata,
            "data_sha256": sha(data),
            "source_url": provenance["source_url"],
            "source_sha256": provenance["source_sha256"],
            "repeat_identical": True,
            "python_max_absolute_error": float(np.max(np.abs(predicted - expected))),
            "files": {path.name: sha(path) for path in sorted(runs[0].iterdir()) if path.is_file()},
        }
        print(f"{name}: repeated model bytes identical; Python export agrees with native predictions", flush=True)
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
