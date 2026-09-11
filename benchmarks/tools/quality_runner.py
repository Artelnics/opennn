"""Train paired models on shared tensors and score held-out predictions.

Runs use a fixed epoch budget, no training warm-up, all tail samples, fixed
sample order, Adam (beta1=.9, beta2=.999, epsilon=float32 epsilon), no clipping,
no regularization, and dropout zero for translation. Initial weights use each
engine's recorded initialization policy; they are not identical tensors.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

import numpy as np

from common import BENCHMARKS, cpu_state, find_binary, gpu_state
from experiment import git_metadata, result_directory
from quality_data import MODELS, digest, read_array, validate_manifest

METRICS = {
    "dense": ("accuracy_percent", "higher"),
    "lstm": ("rmse", "lower"),
    "cnn": ("accuracy_percent", "higher"),
    "transformer": ("bleu", "higher"),
}


def score_predictions(manifest_path, prediction_path):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    target = np.asarray(
        read_array(manifest_path.parent, manifest["test"]["y"]), dtype=np.float64
    )
    values = np.fromfile(prediction_path, dtype="<f4")
    if values.size != target.size or not np.isfinite(values).all():
        raise ValueError("Prediction count mismatch or nonfinite predictions")
    values = values.reshape(target.shape).astype(np.float64)
    kind = manifest["model"]
    if kind in ("dense", "cnn"):
        if np.any(values < -1e-5) or np.any(values > 1 + 1e-5):
            raise ValueError("Classification predictions must be probabilities")
        if kind == "dense":
            if not np.isin(target, [0, 1]).all():
                raise ValueError("Invalid binary targets")
            correct = (values >= 0.5) == target
        else:
            if not np.allclose(values.sum(1), 1, atol=0.01):
                raise ValueError("Class probabilities do not sum to one")
            correct = values.argmax(1) == target.argmax(1)
        return {
            "accuracy_percent": float(correct.mean() * 100),
            "test_samples": len(target),
        }
    if kind == "lstm":
        scale = manifest.get("target_scale", 1.0)
        residual = (values - target) * scale
        mse = float(np.square(residual).mean())
        variance = float(np.square((target - target.mean()) * scale).sum())
        return {
            "rmse": math.sqrt(mse),
            "mae": float(np.abs(residual).mean()),
            "mse": mse,
            "r2": 1 - float(np.square(residual).sum()) / variance
            if variance > 0
            else None,
            "test_samples": len(target),
        }
    from sacrebleu.metrics import BLEU

    vocabulary = manifest["vocabulary"]
    if (
        (values != np.floor(values)).any()
        or (values < 0).any()
        or (values >= len(vocabulary)).any()
    ):
        raise ValueError("Invalid generated token IDs")
    hypotheses = []
    for row in values.astype(int):
        words = []
        for token in row:
            if token == 3:
                break
            if token not in (0, 2):
                words.append(vocabulary[token])
        hypotheses.append(" ".join(words))
    if len(manifest["references"]) != len(hypotheses):
        raise ValueError("Reference count mismatch")
    scorer = BLEU(tokenize="13a", effective_order=True)
    result = scorer.corpus_score(hypotheses, [manifest["references"]])
    return {
        "bleu": float(result.score),
        "bleu_signature": str(scorer.get_signature()),
        "test_samples": len(target),
        "generated_hypotheses": hypotheses,
    }


def summarize(records):
    rows = []
    for model in MODELS:
        metric, direction = METRICS[model]
        for engine in ("opennn", "pytorch"):
            values = [
                r["metrics"][metric]
                for r in records
                if r["model"] == model and r["engine"] == engine and r["status"] == "ok"
            ]
            if values:
                rows.append(
                    {
                        "model": model,
                        "engine": engine,
                        "metric": metric,
                        "direction": direction,
                        "runs": len(values),
                        "mean": statistics.mean(values),
                        "stddev": statistics.stdev(values) if len(values) > 1 else None,
                        "minimum": min(values),
                        "maximum": max(values),
                    }
                )
    return rows


def write_results(out, provenance, records):
    rows = summarize(records)
    result = {
        "schema": "quality-v1",
        "status": "diagnostic",
        "provenance": provenance,
        "quality_gate": {
            "assessed": False,
            "reason": "Descriptive results; equivalence margins and adequate achieved quality require review before a parity claim",
        },
        "runs": records,
        "summary": rows,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if rows:
        with (out / "comparisons.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    lines = [
        "# Prediction quality",
        "",
        "Diagnostic results. These do not automatically establish parity.",
        "",
        "| Model | Metric | OpenNN | PyTorch |",
        "|---|---|---:|---:|",
    ]
    for model in MODELS:
        pair = {r["engine"]: r for r in rows if r["model"] == model}
        if not pair:
            continue

        def cell(engine):
            row = pair.get(engine)
            if not row:
                return "Not measured"
            spread = (
                f" ± {row['stddev']:.4g}" if row["stddev"] is not None else " (one run)"
            )
            return f"{row['mean']:.4g}{spread}"

        lines.append(
            f"| {model} | {METRICS[model][0]} | {cell('opennn')} | {cell('pytorch')} |"
        )
    lines.extend(
        [
            "",
            "Values are test-set means and sample standard deviations over the recorded seeds.",
            "Synthetic smoke fixtures are functional checks, not benchmark evidence.",
            "See results.json, histories and prediction files for provenance and failed runs.",
        ]
    )
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["quality"], default="quality")
    parser.add_argument("--model", choices=(*MODELS, "all"), default="all")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="One manifest or directory containing model/manifest.json",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Fixed training budget; choose before inspecting test results",
    )
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--learning-rate", type=float, help="Default .001, or .0001 for Transformer"
    )
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--python", default=sys.executable, help="PyTorch Python environment"
    )
    parser.add_argument("--opennn-binary", type=Path)
    parser.add_argument(
        "--opennn-library-path",
        help="Optional Linux LD_LIBRARY_PATH for the native driver",
    )
    parser.add_argument(
        "--pytorch-library-path",
        help="Optional Linux LD_LIBRARY_PATH for the Python driver",
    )
    parser.add_argument("--timeout", type=float, default=86400)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    if min(args.epochs, args.batch, args.threads, args.timeout) <= 0 or (
        args.learning_rate is not None and args.learning_rate <= 0
    ):
        parser.error(
            "Training budget, threads, timeout and learning rate must be positive"
        )
    if args.device == "cpu" and args.precision != "fp32":
        parser.error("CPU quality uses FP32")
    models = MODELS if args.model == "all" else (args.model,)
    manifests = {
        m: args.manifest / m / "manifest.json"
        if args.manifest.is_dir()
        else args.manifest
        for m in models
    }
    parsed = {m: validate_manifest(p) for m, p in manifests.items()}
    for model, manifest in parsed.items():
        if manifest["model"] != model:
            parser.error("Manifest model does not match --model")
    binary = args.opennn_binary
    if binary is None:
        name, found = find_binary("quality_opennn")
        if not found:
            parser.error("Build quality_opennn or pass --opennn-binary")
        binary = Path(name)
    binary = binary.resolve()
    if not binary.is_file():
        parser.error(f"Missing native driver: {binary}")
    if "transformer" in models:
        pass  # fail before training if the neutral scorer is missing
    out = result_directory("quality", args.out)
    env = os.environ.copy()
    env.update(
        {
            key: str(args.threads)
            for key in (
                "OMP_NUM_THREADS",
                "TORCH_NUM_THREADS",
                "OPENNN_THREADS",
                "MKL_NUM_THREADS",
            )
        }
    )
    env.update(
        {
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
            "OPENNN_NO_CUDA_GRAPH": "1",
            "OPENNN_ADAM_BF16_MOMENT": "0",
            "NVIDIA_TF32_OVERRIDE": "0",
        }
    )
    provenance = {
        "git": git_metadata(),
        "platform": platform.platform(),
        "cpu": cpu_state(),
        "gpu": gpu_state() if args.device == "cuda" else None,
        "python": sys.version,
        "arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "opennn_binary_sha256": digest(binary),
        "pytorch_driver_sha256": digest(BENCHMARKS / "families/quality.py"),
        "manifests": {
            m: {"path": str(p.resolve()), "sha256": digest(p), "data": parsed[m]}
            for m, p in manifests.items()
        },
        "controls": {
            k: env[k]
            for k in env
            if k.startswith(("OPENNN_", "OMP_", "MKL_", "NVIDIA_", "TORCH_"))
        },
        "protocol": __doc__,
    }
    records = []
    for model in models:
        for round_index, seed in enumerate(seeds):
            pair = []
            for engine in (
                ("opennn", "pytorch") if round_index % 2 == 0 else ("pytorch", "opennn")
            ):
                folder = out / f"{model}-{engine}-seed{seed}"
                folder.mkdir()
                rate = args.learning_rate or (
                    0.0001 if model == "transformer" else 0.001
                )
                command = (
                    [str(binary)]
                    if engine == "opennn"
                    else [args.python, str(BENCHMARKS / "families/quality.py")]
                )
                command += [
                    str(manifests[model].resolve()),
                    str(folder),
                    str(args.epochs),
                    str(args.batch),
                    str(seed),
                    args.device,
                    args.precision,
                    str(rate),
                ]
                record = {
                    "model": model,
                    "engine": engine,
                    "seed": seed,
                    "command": command,
                    "smoke": parsed[model]["smoke"],
                    "status": "error",
                }
                print(model, engine, "seed", seed, "->", folder, flush=True)
                mark = time.monotonic()
                child_env = env.copy()
                library_path = (
                    args.opennn_library_path
                    if engine == "opennn"
                    else args.pytorch_library_path
                )
                if library_path is not None:
                    child_env["LD_LIBRARY_PATH"] = library_path
                record["library_path"] = child_env.get("LD_LIBRARY_PATH", "")
                try:
                    with (
                        (folder / "stdout.txt").open("w") as stdout,
                        (folder / "stderr.txt").open("w") as stderr,
                    ):
                        completed = subprocess.run(
                            command,
                            stdout=stdout,
                            stderr=stderr,
                            env=child_env,
                            timeout=args.timeout,
                        )
                    record["returncode"] = completed.returncode
                    if completed.returncode:
                        raise RuntimeError(
                            f"Driver exited {completed.returncode}; see {folder}"
                        )
                    driver = json.loads((folder / "driver.json").read_text())
                    if any(
                        driver[k] != v
                        for k, v in {
                            "model": model,
                            "engine": engine,
                            "seed": seed,
                            "epochs": args.epochs,
                            "device": args.device,
                            "precision": args.precision,
                            "train_samples": parsed[model]["train"]["x"]["shape"][0],
                            "test_samples": parsed[model]["test"]["x"]["shape"][0],
                        }.items()
                    ):
                        raise ValueError("Driver work gate failed")
                    if engine == "pytorch" and driver.get("interface") != "python":
                        raise ValueError("Wrong PyTorch interface")
                    record.update(
                        driver=driver,
                        metrics=score_predictions(
                            manifests[model], folder / "predictions.bin"
                        ),
                        predictions_sha256=digest(folder / "predictions.bin"),
                        status="ok",
                    )
                except (
                    OSError,
                    ValueError,
                    RuntimeError,
                    subprocess.TimeoutExpired,
                ) as error:
                    record["error"] = str(error)
                    print(record["error"], flush=True)
                record["process_seconds"] = time.monotonic() - mark
                pair.append(record)
                records.append(record)
                write_results(out, provenance, records)
            if (
                all(r["status"] == "ok" for r in pair)
                and pair[0]["driver"]["parameters"] != pair[1]["driver"]["parameters"]
            ):
                for record in pair:
                    record.update(
                        status="error", error="Cross-engine parameter-count gate failed"
                    )
                write_results(out, provenance, records)
    print("Results:", out, flush=True)
    return 1 if any(r["status"] != "ok" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
