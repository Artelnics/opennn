"""Compare shared-initialization predictions, gradients and fixed-batch updates.

This is a numerical diagnostic, not a prediction-quality equivalence test.
All differences are retained without selecting tolerances after observing them.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from common import BENCHMARKS, cpu_state, gpu_state
from experiment import git_metadata, result_directory
from quality_data import digest, read_array, validate_manifest
from quality_runner import verify_dataset_files, verify_implementation_files


def compare_arrays(opennn, pytorch):
    """Compare finite arrays in bounded chunks; relative L2 is symmetric."""
    left, right = np.asarray(opennn), np.asarray(pytorch)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("Compared tensors must have identical, nonempty shapes")
    maximum = squared = left_squared = right_squared = 0.0
    for start in range(0, left.size, 1 << 20):
        a = left.reshape(-1)[start:start + (1 << 20)].astype(np.float64)
        b = right.reshape(-1)[start:start + (1 << 20)].astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("Nonfinite compared tensor")
        delta = a - b
        maximum = max(maximum, float(np.max(np.abs(delta))))
        squared += float(np.sum(delta * delta))
        left_squared += float(np.sum(a * a))
        right_squared += float(np.sum(b * b))
    denominator = max(left_squared, right_squared)
    return {
        "shape": list(left.shape), "elements": int(left.size),
        "max_abs": maximum, "rms": math.sqrt(squared / left.size),
        "relative_l2": math.sqrt(squared / denominator) if denominator else 0.0,
        "opennn_l2": math.sqrt(left_squared), "pytorch_l2": math.sqrt(right_squared),
    }


def _load_array(folder, spec, filename=None):
    shape = spec.get("shape")
    if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Invalid tensor shape")
    if spec.get("dtype") != "float32":
        raise ValueError("Diagnostic tensors must be little-endian float32")
    folder = Path(folder).resolve()
    path = (folder / (filename or spec["file"])).resolve()
    if not path.is_relative_to(folder) or path == folder:
        raise ValueError("Tensor path escapes its artifact directory")
    if path.stat().st_size != math.prod(shape) * 4:
        raise ValueError(f"Tensor size mismatch: {path}")
    identity = digest(path)
    if spec.get("sha256", identity) != identity:
        raise ValueError(f"Tensor hash mismatch: {path}")
    values = np.memmap(path, dtype="<f4", mode="r", shape=tuple(shape))
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite tensor: {path}")
    return values, {"path": str(path), "sha256": identity, "bytes": path.stat().st_size}


def read_tensor_manifest(path):
    """Load a complete named tensor artifact, checking identity and structure."""
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema") != 1 or not isinstance(record.get("tensors"), list):
        raise ValueError("Unsupported diagnostic tensor manifest")
    values, files = {}, {}
    used_paths = set()
    for spec in record["tensors"]:
        name = spec["name"]
        if not isinstance(name, str) or not name or name in values:
            raise ValueError("Duplicate or invalid diagnostic tensor name")
        values[name], files[name] = _load_array(path.parent, spec)
        if files[name]["path"] in used_paths:
            raise ValueError("Diagnostic tensors must use distinct files")
        used_paths.add(files[name]["path"])
    if not values:
        raise ValueError("Empty diagnostic tensor manifest")
    return values, {"path": str(path), "sha256": digest(path), "files": files}


def _frozen(name, model):
    return (model == "lstm" and name == "lstm.bias_hh_l0") or (
        model == "cnn" and name.endswith(".num_batches_tracked")
    )


def _buffer(name, model):
    return _frozen(name, model) or (model == "transformer" and name == "position") or (
        model == "cnn" and name.endswith((".running_mean", ".running_var"))
    )


def compare_tensor_sets(opennn, pytorch, initial, *, model, kind, step=None):
    """Require all expected names; document only known native state omissions."""
    if kind not in ("initialization", "gradients", "parameters"):
        raise ValueError("Unknown diagnostic tensor group")
    if kind != "initialization" and (type(step) is not int or step < 0):
        raise ValueError("Training tensor groups require a nonnegative step")
    expected = {
        name for name in initial if kind != "gradients" or not _buffer(name, model)
    }
    permitted = {name for name in expected if _frozen(name, model)}
    if set(pytorch) != expected:
        raise ValueError(f"PyTorch {kind} names differ: missing={sorted(expected - set(pytorch))}, "
                         f"unexpected={sorted(set(pytorch) - expected)}")
    if set(opennn) - expected or expected - set(opennn) - permitted:
        raise ValueError(f"OpenNN {kind} names differ: missing={sorted(expected - set(opennn) - permitted)}, "
                         f"unexpected={sorted(set(opennn) - expected)}")
    omissions = []
    for name in sorted(expected):
        if pytorch[name].shape != initial[name].shape or (
            name in opennn and opennn[name].shape != initial[name].shape
        ):
            raise ValueError(f"Tensor shape differs from shared initialization: {name}")
        if name not in opennn:
            expected_value = (step + 1 if kind == "parameters" else 0) if (
                name.endswith(".num_batches_tracked")
            ) else 0
            if np.any(initial[name] != 0) or np.any(pytorch[name] != expected_value):
                raise ValueError(f"Invalid omitted native state: {name}")
            omissions.append({"name": name, "reference_value": expected_value,
                              "reason": "counter_not_stored" if name.endswith(".num_batches_tracked")
                              else "frozen_zero_recurrent_bias", "verified": True})
    rows = [{"name": name, **compare_arrays(opennn[name], pytorch[name])}
            for name in sorted(opennn)]
    if not rows:
        raise ValueError("No comparable tensors")
    result = {
        "tensors": rows, "tensor_count": len(rows), "native_omissions": omissions,
        "max_abs": max(row["max_abs"] for row in rows),
        "max_relative_l2": max(row["relative_l2"] for row in rows),
        "largest_absolute_difference_tensor": max(rows, key=lambda row: row["max_abs"])["name"],
    }
    if kind == "initialization":
        result["source_roundtrip"] = {
            engine: [{"name": name, **compare_arrays(values[name], initial[name])}
                     for name in sorted(values)]
            for engine, values in (("opennn", opennn), ("pytorch", pytorch))
        }
        result["source_roundtrip_exact"] = {
            engine: all(row["max_abs"] == 0 for row in rows)
            for engine, rows in result["source_roundtrip"].items()
        }
    return result


def compare_outputs(opennn, pytorch, *, model, targets):
    result = {"all_positions": compare_arrays(opennn, pytorch)}
    if model == "transformer":
        active = np.asarray(targets) != 0
        if active.shape != opennn.shape[:-1] or not active.any():
            raise ValueError("Transformer target mask has wrong shape or no active tokens")
        result["active_positions"] = compare_arrays(opennn[active], pytorch[active])
        result["active_tokens"] = int(active.sum())
        result["padded_tokens"] = int((~active).sum())
        result["padded_positions"] = (
            compare_arrays(opennn[~active], pytorch[~active]) if (~active).any() else None
        )
    return result


def _read_outputs(folder, prefix, *, model, mode, targets, vocabulary_size):
    path = Path(folder) / f"{prefix}.json"
    record = json.loads(path.read_text())
    output = ("teacher_forced_logits" if mode == "evaluation" else "logits") if (
        model == "transformer"
    ) else "regression" if model == "lstm" else "probabilities"
    shape = list(targets.shape) + [vocabulary_size] if model == "transformer" else list(targets.shape)
    if any(record.get(key) != value for key, value in {
        "mode": mode, "output": output, "samples": len(targets), "shape": shape,
    }.items()):
        raise ValueError(f"Output metadata mismatch: {path}")
    values, identity = _load_array(path.parent, record, f"{prefix}.bin")
    return values, {"metadata_sha256": digest(path), **identity}


def compare_run_artifacts(directory, manifest_path, initialization_path, *, steps, batch):
    """Compare every available stage; keep other stages when one is invalid."""
    directory, manifest_path = Path(directory), Path(manifest_path)
    manifest = validate_manifest(manifest_path)
    model = manifest["model"]
    initial, identity = read_tensor_manifest(initialization_path)
    result = {"initialization_artifact": identity, "comparisons": {}, "errors": []}

    def capture(label, operation):
        try:
            result["comparisons"][label] = operation()
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["errors"].append({"stage": label, "error": str(error)})

    def tensors(kind, step=None):
        relative = Path("initialization") if step is None else Path("diagnostic") / f"step_{step}" / kind
        groups, identities = {}, {}
        for engine in ("opennn", "pytorch"):
            groups[engine], identities[engine] = read_tensor_manifest(
                directory / engine / relative / "manifest.json"
            )
        comparison = compare_tensor_sets(groups["opennn"], groups["pytorch"], initial,
                                         model=model, kind=kind, step=step)
        return {**comparison, "artifacts": identities}

    def outputs(step=None):
        split = "test" if step is None else "train"
        limit = 8 if step is None else batch
        targets = np.asarray(read_array(manifest_path.parent, manifest[split]["y"])[:limit])
        relative = Path() if step is None else Path("diagnostic") / f"step_{step}"
        prefix = "initial_predictions" if step is None else "outputs"
        arrays, identities = {}, {}
        for engine in ("opennn", "pytorch"):
            arrays[engine], identities[engine] = _read_outputs(
                directory / engine / relative, prefix, model=model,
                mode="evaluation" if step is None else "training", targets=targets,
                vocabulary_size=len(manifest.get("vocabulary", [])),
            )
        return {**compare_outputs(arrays["opennn"], arrays["pytorch"], model=model,
                                  targets=targets), "artifacts": identities}

    def losses():
        values, identities = {}, {}
        count = min(batch, manifest["train"]["x"]["shape"][0])
        for engine in ("opennn", "pytorch"):
            path = directory / engine / "diagnostic.json"
            document = json.loads(path.read_text())
            records = document["steps"]
            if document.get("mode") != "fixed_batch_training" or len(records) != steps:
                raise ValueError("Diagnostic step count or training mode mismatch")
            for index, record in enumerate(records):
                value = record["loss"]
                if record["step"] != index or record["samples"] != count or (
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    raise ValueError("Invalid diagnostic step, sample count or loss")
            values[engine] = records
            identities[engine] = {"path": str(path), "sha256": digest(path)}
        return {"steps": [{"step": index,
                            "opennn": values["opennn"][index]["loss"],
                            "pytorch": values["pytorch"][index]["loss"],
                            **compare_arrays(np.asarray(values["opennn"][index]["loss"]),
                                             np.asarray(values["pytorch"][index]["loss"]))}
                           for index in range(steps)], "artifacts": identities}

    capture("initialization", lambda: tensors("initialization"))
    capture("initial_predictions", outputs)
    capture("losses", losses)
    for step in range(steps):
        capture(f"step_{step}_outputs", lambda step=step: outputs(step))
        for kind in ("gradients", "parameters"):
            capture(f"step_{step}_{kind}", lambda kind=kind, step=step: tensors(kind, step))
    result["complete"] = not result["errors"]
    return result


def _run(command, folder, environment, timeout):
    folder.mkdir(parents=True, exist_ok=True)
    result = {"command": command, "directory": str(folder), "status": "error"}
    started = time.monotonic()
    try:
        with (folder / "stdout.txt").open("w") as stdout, (folder / "stderr.txt").open("w") as stderr:
            completed = subprocess.run(command, env=environment, timeout=timeout, stdout=stdout, stderr=stderr)
        result["returncode"] = completed.returncode
        if completed.returncode:
            raise RuntimeError(f"Driver exited {completed.returncode}; see {folder / 'stderr.txt'}")
        result["status"] = "ok"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result["error"] = str(error)
    result["process_seconds"] = time.monotonic() - started
    result["logs"] = {name: digest(folder / name) for name in ("stdout.txt", "stderr.txt")
                      if (folder / name).is_file()}
    return result


def _write_results(out, result):
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Numerical diagnostics", "", "Shared initial tensors and a fixed training batch.",
             f"Execution status: {result['execution_status']}.",
             "Differences below are descriptive; no numerical tolerance or quality parity is inferred.", "",
             "| Stage | Maximum absolute difference | Relative L2 difference |",
             "|---|---:|---:|"]
    for name, item in result.get("comparison", {}).get("comparisons", {}).items():
        if "all_positions" in item:
            for region in ("all_positions", "active_positions", "padded_positions"):
                if item.get(region):
                    value = item[region]
                    lines.append(f"| {name}: {region} | {value['max_abs']:.6g} | {value['relative_l2']:.6g} |")
        elif "tensor_count" in item:
            lines.append(f"| {name} (maxima across tensors) | {item['max_abs']:.6g} | {item['max_relative_l2']:.6g} |")
        elif name == "losses":
            for step in item["steps"]:
                lines.append(f"| step_{step['step']}_loss | {step['max_abs']:.6g} | {step['relative_l2']:.6g} |")
    lines.extend(["", "Relative L2 divides by the larger engine tensor norm; tensor-group maxima may refer to different tensors.",
                  "Transformer active and padded query positions are reported separately using target token != 0.",
                  "Frozen recurrent biases and BatchNorm counters omitted by native exports are explicitly validated.",
                  "See results.json for every tensor, initialization round trips, commands, hashes, and failures."])
    for error in result.get("errors", []):
        lines.append(f"\nError: {error}")
    for run in result.get("runs", []):
        if run.get("error"):
            lines.append(f"\nError ({run['engine']}): {run['error']}")
    for error in result.get("comparison", {}).get("errors", []):
        lines.append(f"\nError ({error['stage']}): {error['error']}")
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--opennn-binary", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--opennn-library-path")
    parser.add_argument("--pytorch-library-path")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--adam-epsilon", type=float, default=float(np.finfo(np.float32).eps))
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    args.adam_epsilon = float(np.float32(args.adam_epsilon))
    if min(args.batch, args.steps, args.threads) < 1 or args.seed < 0 or (
        not math.isfinite(args.timeout) or args.timeout <= 0
    ) or not math.isfinite(args.adam_epsilon) or args.adam_epsilon <= 0 or (args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    )):
        parser.error("Batch, steps, threads, timeout and learning rate must be positive; seed nonnegative")
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("CPU diagnostics use FP32")
    manifest_path, binary = args.manifest.resolve(), args.opennn_binary.resolve()
    manifest = validate_manifest(manifest_path)
    if not binary.is_file():
        parser.error(f"Missing OpenNN binary: {binary}")
    out = result_directory("quality-numerics", args.out)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("OPENNN_QUALITY_")}
    environment.update({name: str(args.threads) for name in (
        "OMP_NUM_THREADS", "TORCH_NUM_THREADS", "OPENNN_THREADS", "MKL_NUM_THREADS"
    )})
    environment.update({"OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE", "OPENNN_NO_CUDA_GRAPH": "1",
                        "OPENNN_ADAM_BF16_MOMENT": "0", "NVIDIA_TF32_OVERRIDE": "0",
                        "OPENNN_QUALITY_ADAM_EPSILON": str(args.adam_epsilon)})
    result = {
        "schema": "quality-numerics-v1", "status": "diagnostic", "execution_status": "pending",
        "quality_gate": {"assessed": False, "reason": "Numerical differences are not prediction-quality equivalence evidence"},
        "provenance": {
            "git": git_metadata(), "platform": platform.platform(), "cpu": cpu_state(),
            "gpu": gpu_state() if args.device == "cuda" else None,
            "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            "model": manifest["model"], "smoke": manifest["smoke"],
            "manifest_sha256": digest(manifest_path), "opennn_binary_sha256": digest(binary),
            "pytorch_driver_sha256": digest(BENCHMARKS / "families/quality.py"),
            "quality_data_sha256": digest(BENCHMARKS / "tools/quality_data.py"),
            "initialization_generator_sha256": digest(BENCHMARKS / "tools/quality_initialization.py"),
            "controls": {key: value for key, value in environment.items()
                         if key.startswith(("OPENNN_", "OMP_", "MKL_", "NVIDIA_", "TORCH_"))},
        },
        "runs": [], "errors": [],
    }
    (out / "plan.json").write_text(json.dumps(result["provenance"], indent=2), encoding="utf-8")
    _write_results(out, result)
    initialization = out / "initialization" / "manifest.json"
    preparation_env = environment.copy()
    if args.pytorch_library_path is not None:
        preparation_env["LD_LIBRARY_PATH"] = args.pytorch_library_path
    try:
        verify_implementation_files(binary, result["provenance"])
        verify_dataset_files(manifest_path, result["provenance"]["manifest_sha256"])
        result["preparation"] = _run([
            args.python, str(BENCHMARKS / "tools/quality_initialization.py"), "create",
            "--manifest", str(manifest_path), "--out", str(initialization.parent), "--seed", str(args.seed),
        ], out / "preparation", preparation_env, args.timeout)
        verify_implementation_files(binary, result["provenance"])
        verify_dataset_files(manifest_path, result["provenance"]["manifest_sha256"])
        if result["preparation"]["status"] != "ok":
            raise RuntimeError("Shared initialization preparation failed")
        _, source_identity = read_tensor_manifest(initialization)
        result["provenance"]["initialization"] = source_identity
        environment.update({"OPENNN_QUALITY_INITIALIZATION": str(initialization),
                            "OPENNN_QUALITY_INITIAL_PREDICTIONS": "1",
                            "OPENNN_QUALITY_EXPORT_INITIALIZATION": "1",
                            "OPENNN_QUALITY_DIAGNOSTIC_STEPS": str(args.steps)})
        rate = args.learning_rate or (0.0001 if manifest["model"] == "transformer" else 0.001)
        for engine in ("opennn", "pytorch"):
            folder = out / engine
            command = [str(binary)] if engine == "opennn" else [args.python, str(BENCHMARKS / "families/quality.py")]
            command += [str(manifest_path), str(folder), "1", str(args.batch), str(args.seed),
                        args.device, args.precision, str(rate)]
            child_env = environment.copy()
            library_path = args.opennn_library_path if engine == "opennn" else args.pytorch_library_path
            if library_path is not None:
                child_env["LD_LIBRARY_PATH"] = library_path
            print(engine, "numerical diagnostics ->", folder, flush=True)
            run = {"engine": engine, "command": command, "directory": str(folder),
                   "status": "error", "process_seconds": None,
                   "library_path": child_env.get("LD_LIBRARY_PATH", "")}
            try:
                verify_implementation_files(binary, result["provenance"])
                verify_dataset_files(manifest_path, result["provenance"]["manifest_sha256"])
                run.update(_run(command, folder, child_env, args.timeout))
                verify_implementation_files(binary, result["provenance"])
                verify_dataset_files(manifest_path, result["provenance"]["manifest_sha256"])
                if run["status"] == "ok":
                    driver_path = folder / "driver.json"
                    driver = json.loads(driver_path.read_text())
                    if not isinstance(driver, dict):
                        raise ValueError("Diagnostic driver metadata must be a JSON object")
                    expected = {"engine": engine, "model": manifest["model"], "seed": args.seed,
                                "device": args.device, "precision": args.precision, "epochs": 0,
                                "diagnostic_only": True, "diagnostic_steps": args.steps,
                                "adam_epsilon": args.adam_epsilon, "status": "ok"}
                    if any(driver.get(key) != value for key, value in expected.items()) or (
                        engine == "pytorch" and driver.get("interface") != "python"
                    ):
                        raise ValueError("Diagnostic driver identity or work mismatch")
                    reported = driver.get("initialization", {})
                    if reported.get("mode") != "shared" or reported.get("seed") != args.seed or (
                        Path(reported.get("manifest", "")).resolve() != initialization.resolve()
                    ) or (reported.get("manifest_sha256", source_identity["sha256"]) != source_identity["sha256"]) or (
                        reported.get("tensor_count") != len(source_identity["files"])
                    ):
                        raise ValueError("Diagnostic shared initialization identity mismatch")
                    _, current_identity = read_tensor_manifest(initialization)
                    if current_identity != source_identity:
                        raise ValueError("Shared initialization changed during execution")
                    run.update(driver=driver, driver_sha256=digest(driver_path))
            except (OSError, ValueError, KeyError, TypeError) as error:
                run.update(status="error", error=str(error))
            result["runs"].append(run)
            _write_results(out, result)
        result["comparison"] = compare_run_artifacts(out, manifest_path, initialization,
                                                       steps=args.steps, batch=args.batch)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        result["errors"].append(str(error))
    success = not result["errors"] and len(result["runs"]) == 2 and all(
        run["status"] == "ok" for run in result["runs"]
    ) and result.get("comparison", {}).get("complete", False)
    result["execution_status"] = "ok" if success else "error"
    _write_results(out, result)
    print("Results:", out, flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
