#!/usr/bin/env python3
"""Parity check for the Python export of ModelExpression.

Loads the generated iris_model.py without requiring numpy/pandas (a small
math-based shim is injected) and compares its outputs against the reference
CSV written by the iris example.

Usage: check_python_export.py <iris_model.py> <reference.csv> <n_inputs> <n_outputs> [tolerance]
"""

import importlib.util
import math
import sys
import types


def install_shims() -> None:
    numpy_shim = types.ModuleType("numpy")
    numpy_shim.exp = math.exp
    numpy_shim.tanh = math.tanh
    numpy_shim.maximum = lambda a, b: max(a, b)
    numpy_shim.max = lambda values: max(values)
    numpy_shim.sum = lambda values: sum(values)
    numpy_shim.where = lambda condition, a, b: a if condition else b
    sys.modules.setdefault("numpy", numpy_shim)
    sys.modules.setdefault("pandas", types.ModuleType("pandas"))


def main() -> None:
    if len(sys.argv) < 5:
        sys.exit(__doc__)

    model_path, csv_path = sys.argv[1], sys.argv[2]
    n_inputs, n_outputs = int(sys.argv[3]), int(sys.argv[4])
    tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-4

    install_shims()

    spec = importlib.util.spec_from_file_location("exported_model", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    network = module.NeuralNetwork()

    max_diff = 0.0
    agree = 0
    rows = 0

    with open(csv_path) as csv_file:
        for line in csv_file:
            line = line.strip()
            if not line:
                continue
            values = [float(x) for x in line.split(";")]
            inputs, reference = values[:n_inputs], values[n_inputs:n_inputs + n_outputs]
            outputs = [float(x) for x in network.calculate_outputs(inputs)]
            rows += 1
            max_diff = max(max_diff, max(abs(o - r) for o, r in zip(outputs, reference)))
            agree += outputs.index(max(outputs)) == reference.index(max(reference))

    print(f"OpenNN reference vs Python export: max abs diff = {max_diff:.3e} ({rows} vectors)")
    print(f"Predicted class agreement: {agree}/{rows}")
    print(f"Tolerance: {tolerance:g}")

    if max_diff > tolerance or agree != rows:
        sys.exit("FAIL: Python export does not match the reference")
    print("PARITY OK")


if __name__ == "__main__":
    main()
