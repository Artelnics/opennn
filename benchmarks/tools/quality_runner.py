"""Train paired models or evaluate saved paired checkpoints on held-out tensors.

Runs use a fixed epoch budget, no training warm-up, all tail samples, fixed
sample order, Adam (beta1=.9, beta2=.999, configurable epsilon), no clipping,
no regularization, and dropout zero for translation. Initial weights use each
engine's recorded initialization policy by default. --initialization shared
loads identical logical tensors generated once per seed into both engines.
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
from quality_statistics import analyze_pairs

METRICS = {
    "dense": ("accuracy_percent", "higher"),
    "lstm": ("rmse", "lower"),
    "cnn": ("accuracy_percent", "higher"),
    "transformer": ("bleu", "higher"),
}


def verify_dataset_files(manifest_path, expected_sha256):
    """Recheck the frozen manifest and all referenced data outside driver timing."""
    if digest(manifest_path) != expected_sha256:
        raise ValueError("Quality data manifest changed after the run plan was recorded")
    validate_manifest(manifest_path)
    if digest(manifest_path) != expected_sha256:
        raise ValueError("Quality data manifest changed during identity verification")


def verify_implementation_files(binary, provenance):
    expected = {
        Path(binary): provenance["opennn_binary_sha256"],
        BENCHMARKS / "families/quality.py": provenance["pytorch_driver_sha256"],
        BENCHMARKS / "tools/quality_data.py": provenance["quality_data_sha256"],
    }
    if provenance.get("initialization_generator_sha256"):
        expected[BENCHMARKS / "tools/quality_initialization.py"] = provenance["initialization_generator_sha256"]
    for path, identity in expected.items():
        if digest(path) != identity:
            raise ValueError(f"Quality implementation changed after the run plan was recorded: {path.name}")


def verify_history(path, epochs, evaluation_only):
    """Require one finite, ordered loss row for every training epoch."""
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["epoch", "training_loss"]:
            raise ValueError("Training history header is invalid")
        rows = list(reader)
    expected = 0 if evaluation_only else epochs
    if len(rows) != expected:
        raise ValueError(f"Training history has {len(rows)} rows; expected {expected}")
    for index, row in enumerate(rows, 1):
        if int(row["epoch"]) != index or not math.isfinite(float(row["training_loss"])):
            raise ValueError("Training history is unordered or nonfinite")


def verify_initialization_files(initialization):
    """Catch artifact changes before or during either engine's launch."""
    path = Path(initialization["manifest"])
    if digest(path) != initialization["manifest_sha256"]:
        raise ValueError("Shared initialization manifest changed")
    artifact = json.loads(path.read_text())
    for tensor in artifact["tensors"]:
        item = (path.parent / tensor["file"]).resolve()
        if not item.is_relative_to(path.parent.resolve()) or digest(item) != tensor["sha256"]:
            raise ValueError(f"Shared initialization tensor changed: {tensor['name']}")


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
    tokenizer = manifest.get("tokenizer")
    processor = None
    if tokenizer and tokenizer["kind"] == "sentencepiece":
        import sentencepiece as spm

        processor = spm.SentencePieceProcessor(
            model_file=str(manifest_path.parent / tokenizer["model_file"])
        )
        if processor.vocab_size() != len(vocabulary) or any(
            processor.id_to_piece(i) != piece
            for i, piece in enumerate(vocabulary)
        ):
            raise ValueError("Tokenizer pieces do not match the manifest vocabulary")
    hypotheses = []
    for row in values.astype(int):
        tokens = []
        for token in row:
            if token == 3:
                break
            if token not in (0, 2):
                tokens.append(token)
        hypotheses.append(
            processor.decode(tokens) if processor else
            " ".join(vocabulary[token] for token in tokens)
        )
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
    comparisons = analyze_pairs(
        records,
        planned_seeds=[int(value) for value in provenance["arguments"]["seeds"].split(",")],
        models=list(provenance["manifests"]),
    )
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
        "paired_analysis": comparisons,
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
        "| Model | Metric | OpenNN | PyTorch | Complete seed pairs | Pair status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for comparison in comparisons["rows"]:
        def cell(engine):
            values = [pair[engine] for pair in comparison["pairs"]]
            if not values:
                return "Not measured"
            spread = f" ± {statistics.stdev(values):.4g}" if len(values) > 1 else " (one pair)"
            return f"{statistics.mean(values):.4g}{spread}"

        complete = not comparison["pair_issues"] and not comparison["unplanned_seeds"]
        lines.append(
            f"| {comparison['model']} | {comparison['metric']} | {cell('opennn')} | {cell('pytorch')} "
            f"| {comparison['complete_pairs']}/{len(comparisons['planned_seeds'])} "
            f"| {'Complete' if complete else 'Incomplete or invalid'} |"
        )
    lines.extend(
        [
            "",
            "Both displayed means and sample standard deviations use the same complete seed pairs.",
            "The JSON summary and comparisons.csv retain individual-engine summaries, including unpaired successes.",
            "Synthetic smoke fixtures are functional checks, not benchmark evidence.",
            "See results.json, histories and prediction files for provenance and failed runs.",
        ]
    )
    for comparison in comparisons["rows"]:
        for issue in comparison["pair_issues"]:
            reasons = "; ".join(
                f"{item.get('engine', 'pair')}: {item['reason']}" for item in issue["issues"]
            )
            lines.append(f"\n- {comparison['model']} seed {issue['seed']}: {reasons}.")
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
    parser.add_argument(
        "--initialization", choices=("native", "shared"), default="native",
        help="Use engine-native initialization or one verified tensor artifact for each paired seed",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Require deterministic kernels in both engines for repeatable training",
    )
    parser.add_argument(
        "--learning-rate", type=float, help="Default .001, or .0001 for Transformer"
    )
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument(
        "--adam-epsilon", type=float, default=float(np.finfo(np.float32).eps),
        help="Adam denominator epsilon used identically by both engines",
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument(
        "--weight-decay", type=float, default=0.0,
        help="L2 gradient coefficient used identically by both engines",
    )
    parser.add_argument(
        "--learning-rate-milestones", default="",
        help="Comma-separated completed epochs after which SGD learning rate is multiplied by --learning-rate-gamma",
    )
    parser.add_argument(
        "--learning-rate-gamma", type=float, default=0.1,
        help="Multiplicative SGD learning-rate reduction at each milestone",
    )
    parser.add_argument(
        "--recalibrate-batchnorm", action="store_true",
        help="Recompute CNN running statistics over the training split before evaluation",
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
    parser.add_argument(
        "--checkpoint-run-dir",
        type=Path,
        help="Evaluate checkpoints from a completed paired run without retraining",
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    args.adam_epsilon = float(np.float32(args.adam_epsilon))
    args.learning_rate_gamma = float(np.float32(args.learning_rate_gamma))
    args.weight_decay = float(np.float32(args.weight_decay))
    try:
        args.learning_rate_milestones = (
            [int(value) for value in args.learning_rate_milestones.split(",")]
            if args.learning_rate_milestones else []
        )
    except ValueError:
        parser.error("Learning-rate milestones must be comma-separated integers")
    seeds = [int(value) for value in args.seeds.split(",")]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        parser.error("Seeds must be distinct nonnegative integers")
    if min(args.epochs, args.batch, args.threads, args.timeout) <= 0 or (
        args.learning_rate is not None and args.learning_rate <= 0
    ) or not 0 <= args.momentum < 1 or not math.isfinite(args.adam_epsilon) \
            or args.adam_epsilon <= 0 or not math.isfinite(args.learning_rate_gamma) \
            or not 0 < args.learning_rate_gamma < 1 \
            or not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error(
            "Training budget, threads, timeout and learning rate must be positive"
        )
    if (len(set(args.learning_rate_milestones)) != len(args.learning_rate_milestones)
            or args.learning_rate_milestones != sorted(args.learning_rate_milestones)
            or any(not 0 < epoch < args.epochs
                   for epoch in args.learning_rate_milestones)):
        parser.error("Learning-rate milestones must be distinct, increasing, and inside the training budget")
    if args.learning_rate_milestones and args.optimizer != "sgd":
        parser.error("Learning-rate milestones currently require --optimizer sgd")
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
    checkpoint_source = None
    if args.checkpoint_run_dir is not None:
        if args.initialization != "native":
            parser.error("Checkpoint evaluation cannot also initialize a new shared model")
        if models != ("transformer",):
            parser.error("Checkpoint evaluation currently supports --model transformer")
        source_dir = args.checkpoint_run_dir.resolve()
        source_path = source_dir / "results.json"
        if not source_path.is_file():
            parser.error(f"Missing paired source results: {source_path}")
        checkpoint_source = json.loads(source_path.read_text())
        source_manifest = checkpoint_source["provenance"]["manifests"]["transformer"]["data"]
        final_manifest = parsed["transformer"]
        if any(source_manifest[key] != final_manifest[key]
               for key in ("train", "vocabulary", "profile")):
            parser.error("Checkpoint train tensors, vocabulary or topology differ from evaluation manifest")
        source_subset = source_manifest.get("evaluation_subset", {})
        final_subset = final_manifest.get("evaluation_subset", {})
        if source_manifest["test"] == final_manifest["test"] and (
            source_manifest.get("references") != final_manifest.get("references")
        ):
            parser.error("References differ for identical checkpoint test tensors")
        if source_manifest["test"] != final_manifest["test"] and (
            not source_subset.get("indices_sha256")
            or final_subset.get("excluded_indices_sha256") != source_subset["indices_sha256"]
            or final_subset.get("excluded_size") != source_subset.get("size")
        ):
            parser.error("Evaluation subset is not locked against the source pilot subset")
        source_args = checkpoint_source["provenance"]["arguments"]
        if any(source_args[key] != value for key, value in {
            "epochs": args.epochs,
            "batch": args.batch,
            "device": args.device,
            "precision": args.precision,
            "threads": args.threads,
        }.items()):
            parser.error("Checkpoint training configuration differs")
        source_runs = {
            (run["engine"], run["seed"]): run
            for run in checkpoint_source["runs"]
            if run["model"] == "transformer" and run["status"] == "ok"
        }
        if any((engine, seed) not in source_runs
               for engine in ("opennn", "pytorch") for seed in seeds):
            parser.error("Source paired run lacks a successful checkpoint seed")
        for engine in ("opennn", "pytorch"):
            names = (("parameters.bin", "states.bin") if engine == "opennn"
                     else ("parameters.pt",))
            for seed in seeds:
                folder = source_dir / f"transformer-{engine}-seed{seed}"
                if any(not (folder / name).is_file() for name in names):
                    parser.error(f"Missing saved checkpoint: {folder}")
        checkpoint_source = {
            "directory": source_dir,
            "results_sha256": digest(source_path),
            "opennn_binary_sha256": checkpoint_source["provenance"]["opennn_binary_sha256"],
            "pytorch_driver_sha256": checkpoint_source["provenance"]["pytorch_driver_sha256"],
            "runs": source_runs,
        }
    binary = args.opennn_binary
    if binary is None:
        name, found = find_binary("quality_opennn")
        if not found:
            parser.error("Build quality_opennn or pass --opennn-binary")
        binary = Path(name)
    binary = binary.resolve()
    if not binary.is_file():
        parser.error(f"Missing native driver: {binary}")
    if checkpoint_source and (
        digest(binary) != checkpoint_source["opennn_binary_sha256"]
        or digest(BENCHMARKS / "families/quality.py")
        != checkpoint_source["pytorch_driver_sha256"]
    ):
        parser.error("Checkpoint evaluation drivers differ from the training drivers")
    if "transformer" in models:
        pass  # fail before training if the neutral scorer is missing
    out = result_directory("quality", args.out)
    env = os.environ.copy()
    env.pop("OPENNN_QUALITY_LOAD_CHECKPOINT", None)
    env.pop("OPENNN_QUALITY_INITIALIZATION", None)
    env.pop("OPENNN_QUALITY_INITIAL_PREDICTIONS", None)
    env.pop("OPENNN_QUALITY_DIAGNOSTIC_STEPS", None)
    env.pop("OPENNN_QUALITY_EXPORT_INITIALIZATION", None)
    env.pop("OPENNN_QUALITY_EXPORT_FINAL", None)
    env.pop("OPENNN_QUALITY_OPTIMIZER", None)
    env.pop("OPENNN_QUALITY_ADAM_EPSILON", None)
    env.pop("OPENNN_QUALITY_MOMENTUM", None)
    env.pop("OPENNN_QUALITY_WEIGHT_DECAY", None)
    env.pop("OPENNN_QUALITY_LR_MILESTONES", None)
    env.pop("OPENNN_QUALITY_LR_GAMMA", None)
    env.pop("OPENNN_QUALITY_RECALIBRATE_BATCHNORM", None)
    for key in (
        "OPENNN_DETERMINISTIC", "OPENNN_LT_DETERMINISTIC",
        "OPENNN_QUALITY_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG",
    ):
        env.pop(key, None)
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
    if args.deterministic:
        env.update(
            {
                "OPENNN_DETERMINISTIC": "1",
                "OPENNN_LT_DETERMINISTIC": "1",
                "OPENNN_QUALITY_DETERMINISTIC": "1",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
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
        "quality_data_sha256": digest(BENCHMARKS / "tools/quality_data.py"),
        "initialization_generator_sha256": (
            digest(BENCHMARKS / "tools/quality_initialization.py")
            if args.initialization == "shared" else None
        ),
        "manifests": {
            m: {"path": str(p.resolve()), "sha256": digest(p), "data": parsed[m]}
            for m, p in manifests.items()
        },
        "controls": {
            k: env[k]
            for k in env
            if k.startswith(("OPENNN_", "OMP_", "MKL_", "NVIDIA_", "TORCH_", "CUBLAS_"))
        },
        "protocol": __doc__,
        "checkpoint_source": (
            {"directory": str(checkpoint_source["directory"]),
             "results_sha256": checkpoint_source["results_sha256"]}
            if checkpoint_source else None
        ),
    }
    records = []
    # Freeze the intended trials before observing any scores. This preserves
    # planning provenance; it does not turn a diagnostic run into confirmation.
    (out / "plan.json").write_text(json.dumps(provenance, indent=2, allow_nan=False))
    write_results(out, provenance, records)
    for model in models:
        for round_index, seed in enumerate(seeds):
            pair = []
            initialization = None
            if args.initialization == "shared":
                initialization_dir = out / "initialization" / f"{model}-seed{seed}"
                initialization_dir.parent.mkdir(parents=True, exist_ok=True)
                preparation = [
                    args.python, str(BENCHMARKS / "tools/quality_initialization.py"),
                    "create", "--manifest", str(manifests[model].resolve()),
                    "--out", str(initialization_dir), "--seed", str(seed),
                ]
                preparation_env = env.copy()
                if args.pytorch_library_path is not None:
                    preparation_env["LD_LIBRARY_PATH"] = args.pytorch_library_path
                try:
                    verify_implementation_files(binary, provenance)
                    verify_dataset_files(manifests[model], provenance["manifests"][model]["sha256"])
                    completed = subprocess.run(
                        preparation, env=preparation_env, timeout=args.timeout,
                        capture_output=True, text=True,
                    )
                    log_prefix = initialization_dir.parent / f"{model}-seed{seed}"
                    log_prefix.with_suffix(".stdout.txt").write_text(completed.stdout)
                    log_prefix.with_suffix(".stderr.txt").write_text(completed.stderr)
                    verify_implementation_files(binary, provenance)
                    verify_dataset_files(manifests[model], provenance["manifests"][model]["sha256"])
                    if completed.returncode:
                        raise RuntimeError(f"Shared initialization exited {completed.returncode}; see {log_prefix}.stderr.txt")
                    initialization_path = initialization_dir / "manifest.json"
                    artifact = json.loads(initialization_path.read_text())
                    initialization = {
                        "mode": "shared", "manifest": str(initialization_path),
                        "manifest_sha256": digest(initialization_path), "seed": seed,
                        "tensor_count": len(artifact["tensors"]),
                    }
                except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.TimeoutExpired) as error:
                    for engine in ("opennn", "pytorch"):
                        records.append({
                            "model": model, "engine": engine, "seed": seed,
                            "smoke": parsed[model]["smoke"], "status": "error",
                            "error": str(error), "preparation_command": preparation,
                        })
                    write_results(out, provenance, records)
                    print(str(error), flush=True)
                    continue
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
                record["process_seconds"] = None
                child_env = env.copy()
                child_env["OPENNN_QUALITY_OPTIMIZER"] = args.optimizer
                child_env["OPENNN_QUALITY_ADAM_EPSILON"] = str(args.adam_epsilon)
                child_env["OPENNN_QUALITY_MOMENTUM"] = str(args.momentum)
                child_env["OPENNN_QUALITY_WEIGHT_DECAY"] = str(args.weight_decay)
                child_env["OPENNN_QUALITY_LR_MILESTONES"] = ",".join(
                    str(epoch) for epoch in args.learning_rate_milestones
                )
                child_env["OPENNN_QUALITY_LR_GAMMA"] = str(args.learning_rate_gamma)
                if args.recalibrate_batchnorm:
                    child_env["OPENNN_QUALITY_RECALIBRATE_BATCHNORM"] = "1"
                if initialization:
                    child_env["OPENNN_QUALITY_INITIALIZATION"] = initialization["manifest"]
                    child_env["OPENNN_QUALITY_INITIAL_PREDICTIONS"] = "1"
                    record["initialization"] = initialization
                if checkpoint_source:
                    source_folder = checkpoint_source["directory"] / f"{model}-{engine}-seed{seed}"
                    checkpoint_files = (
                        ("parameters.bin", "states.bin") if engine == "opennn"
                        else ("parameters.pt",)
                    )
                    child_env["OPENNN_QUALITY_LOAD_CHECKPOINT"] = str(source_folder)
                    record["checkpoint"] = {
                        "source_folder": str(source_folder),
                        "files_sha256": {
                            name: digest(source_folder / name)
                            for name in checkpoint_files
                        },
                    }
                library_path = (
                    args.opennn_library_path
                    if engine == "opennn"
                    else args.pytorch_library_path
                )
                if library_path is not None:
                    child_env["LD_LIBRARY_PATH"] = library_path
                record["library_path"] = child_env.get("LD_LIBRARY_PATH", "")
                try:
                    verify_implementation_files(binary, provenance)
                    verify_dataset_files(manifests[model], provenance["manifests"][model]["sha256"])
                    if initialization:
                        verify_initialization_files(initialization)
                    with (
                        (folder / "stdout.txt").open("w") as stdout,
                        (folder / "stderr.txt").open("w") as stderr,
                    ):
                        mark = time.monotonic()
                        try:
                            completed = subprocess.run(
                                command,
                                stdout=stdout,
                                stderr=stderr,
                                env=child_env,
                                timeout=args.timeout,
                            )
                        finally:
                            record["process_seconds"] = time.monotonic() - mark
                    record["returncode"] = completed.returncode
                    verify_implementation_files(binary, provenance)
                    verify_dataset_files(manifests[model], provenance["manifests"][model]["sha256"])
                    if completed.returncode:
                        raise RuntimeError(
                            f"Driver exited {completed.returncode}; see {folder}"
                        )
                    driver = json.loads((folder / "driver.json").read_text())
                    if not isinstance(driver, dict):
                        raise ValueError("Driver metadata must be a JSON object")
                    if initialization:
                        reported = driver.get("initialization", {})
                        if not isinstance(reported, dict):
                            raise ValueError("Shared initialization metadata must be a JSON object")
                        verify_initialization_files(initialization)
                        if (reported.get("mode") != "shared"
                                or reported.get("seed") != seed
                                or Path(reported.get("manifest", "")).resolve() != Path(initialization["manifest"]).resolve()
                                or reported.get("tensor_count") != initialization["tensor_count"]
                                or (engine == "pytorch" and reported.get("manifest_sha256") != initialization["manifest_sha256"])):
                            raise ValueError("Shared initialization identity gate failed")
                    if driver.get("diagnostic_only", False):
                        raise ValueError("Numerical diagnostics cannot stand in for a full quality run")
                    if any(
                        driver.get(k) != v
                        for k, v in {
                            "model": model,
                            "engine": engine,
                            "seed": seed,
                            "epochs": args.epochs,
                            "device": args.device,
                            "precision": args.precision,
                            "deterministic": args.deterministic,
                            "optimizer": args.optimizer,
                            "adam_epsilon": args.adam_epsilon,
                            "weight_decay": args.weight_decay,
                            "learning_rate_milestones": args.learning_rate_milestones,
                            "learning_rate_gamma": args.learning_rate_gamma,
                            "batchnorm_recalibrated": args.recalibrate_batchnorm,
                            "train_samples": parsed[model]["train"]["x"]["shape"][0],
                            "test_samples": parsed[model]["test"]["x"]["shape"][0],
                            "status": "ok",
                        }.items()
                    ):
                        raise ValueError("Driver work gate failed")
                    if type(driver.get("parameters")) is not int or driver["parameters"] < 1:
                        raise ValueError("Driver parameter count is missing or invalid")
                    if engine == "pytorch" and driver.get("interface") != "python":
                        raise ValueError("Wrong PyTorch interface")
                    if driver.get("evaluation_only", False) != bool(checkpoint_source):
                        raise ValueError("Driver checkpoint-evaluation mode mismatch")
                    verify_history(folder / "history.csv", args.epochs, bool(checkpoint_source))
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
                    KeyError,
                    TypeError,
                    RuntimeError,
                    subprocess.TimeoutExpired,
                ) as error:
                    record["error"] = str(error)
                    print(record["error"], flush=True)
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
