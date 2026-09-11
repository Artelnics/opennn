"""Build a publication review from pinned evidence, without changing raw results.

The selection manifest identifies historical observations to review. A filename
containing 'publish' and an old passing flag never establish publication readiness.
This tool inventories duplicates, recalculates tables and retains missing checks.
It does not publish results or run new measurements.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import statistics
from collections import defaultdict

BENCHMARKS = Path(__file__).resolve().parents[1]
MODELS = ("dense", "lstm", "cnn", "transformer")
NAMES = {"dense": "Dense", "lstm": "LSTM", "cnn": "CNN", "transformer": "Transformer"}
UNITS = {
    "dense": "samples/s",
    "lstm": "windows/s",
    "cnn": "images/s",
    "transformer": "sequences/s",
}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def stats(values):
    if not values or not all(finite(v) and v > 0 for v in values):
        raise ValueError("Measurements must be finite and positive")
    mean = statistics.mean(values)
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "cv_percent": 100 * statistics.stdev(values) / mean
        if len(values) > 1
        else None,
    }


def flatten(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from flatten(v)
    elif isinstance(value, list):
        for v in value:
            yield from flatten(v)
    else:
        yield value


def performance(data, selection):
    """Recalculate from launches; distinguish missing evidence from failed checks."""
    config = data["configuration"]
    launches = data["launches"]
    checks, reasons, engines = {}, list(selection.get("issues", [])), {}
    checks["clean_source"] = (
        bool(data.get("git", {}).get("commit"))
        and data.get("git", {}).get("dirty") is False
    )
    checks["shape"] = data.get("shape_gate", {}).get("agrees") is True
    checks["quiet_machine"] = data.get("machine_quiet", {}).get("quiet") is True
    checks["gpu_clock_lock_recorded"] = (
        data.get("clocks_locked") is True if config["device"] == "cuda" else None
    )
    datasets = data.get("datasets", {})
    checks["input_hashes_recorded"] = bool(datasets) and all(
        isinstance(v, dict) and bool(v.get("sha256") or v.get("content_sha256"))
        for v in datasets.values()
    )
    quality = list(flatten(data.get("quality_gate", {}).get("accuracies", {})))
    checks["finite_quality_values_recorded"] = bool(quality) and all(
        finite(v) for v in quality
    )
    checks["quality_flag_passed"] = data.get("quality_gate", {}).get("agrees") is True
    for engine in ("opennn", "pytorch"):
        runs = [x for x in launches if x.get("engine") == engine]
        if len(runs) != config["rounds"] or any(x.get("returncode") != 0 for x in runs):
            raise ValueError(
                f"Incomplete or failed selected launches: {selection['path']}"
            )
        throughput = stats([x["samples_per_sec"] for x in runs])
        memory = stats([x["instruments"]["peak_mib"] for x in runs])
        energy = stats([x["instruments"]["energy_wh"] for x in runs])
        engines[engine] = {
            "throughput": throughput,
            "memory": memory,
            "energy": energy,
            "memory_metric": sorted(
                {x["instruments"].get("memory_metric", "unknown") for x in runs}
            ),
            "energy_domain": sorted(
                {x["instruments"].get("energy_domain", "unknown") for x in runs}
            ),
        }
        checks[f"{engine}_three_rounds"] = len(runs) >= 3
        for metric, observations in (
            ("throughput", throughput),
            ("memory", memory),
            ("energy", energy),
        ):
            cv = observations["cv_percent"]
            checks[f"{engine}_{metric}_variation"] = cv is not None and cv <= 3
            if cv is not None and cv > 3:
                reasons.append(
                    f"{engine.title()} {metric} varies by {cv:.2f}% (limit: 3%)."
                )
        checks[f"{engine}_energy_samples"] = all(
            x["instruments"].get("energy_measurable") is True
            and x["instruments"].get("window_samples", 0) >= 50
            for x in runs
        )
        summary = data["summary"][engine]
        expected = {
            "median_samples_per_sec": throughput["median"],
            "peak_mib": memory["max"],
            "energy_wh": energy["median"],
        }
        checks[f"{engine}_saved_summary_matches"] = all(
            math.isclose(summary[k], v, rel_tol=0.0001, abs_tol=0.000001)
            for k, v in expected.items()
        )
    if not checks["input_hashes_recorded"]:
        reasons.append("The saved input records contain no content hashes.")
    if not checks["finite_quality_values_recorded"]:
        reasons.append(
            "The old quality flag contains no complete set of measured quality values."
        )
    for key, value in checks.items():
        if value is False and not (
            key.endswith("_variation")
            or key in ("input_hashes_recorded", "finite_quality_values_recorded")
        ):
            reasons.append("Check not satisfied: " + key.replace("_", " ") + ".")
    reasons.append(
        "Review this historical workload against the current protocol before using it in the new release."
    )
    return {
        "id": data["benchmark_id"],
        "model": config["family"],
        "mode": config["mode"],
        "device": config["device"],
        "batch": int(config["batch"]),
        "precision": config["precision"],
        "session": data["session_id"],
        "commit": data["git"]["commit"],
        "source": selection["path"],
        "sha256": selection["sha256"],
        "engines": engines,
        "checks": checks,
        "status": "Needs repeat",
        "reasons": list(dict.fromkeys(reasons)),
    }


def startup(data, source):
    from application_startup import summarize, comparisons

    raw = [read(p) for p in sorted((source.parent / "raw").glob("*.json"))]
    measured = [x for x in raw if x.get("label", "").startswith("r")]
    rows = summarize(measured, data["cells"])
    for row, old in zip(rows, data["cells"]):
        if row["count"] != old["count"] or not math.isclose(
            row["median_ms"], old["median_ms"], abs_tol=1e-9
        ):
            raise ValueError("Startup summary differs from the preserved raw launches")
    return {
        "cells": rows,
        "comparisons": comparisons(rows),
        "timed_launches": len(measured),
        "failures": sum(x.get("status") != "ok" for x in measured),
        "unstable_cells": sum(x.get("variation_over_3_percent", False) for x in rows),
        "status": "Needs repeat",
        "reason": "Timing varies too much and machine controls were not enforced.",
    }


def deployment(data):
    cases, rows = data["cases"], []
    for case in cases:
        paths = [x["path"] for x in case["deployment_files"]]
        if (
            case["status"] != "ok"
            or case.get("unresolved_dependencies")
            or len(paths) != len(set(paths))
        ):
            raise ValueError(
                "Deployment inventory is incomplete or contains duplicate paths"
            )
        if (
            sum(x["bytes"] for x in case["deployment_files"])
            != case["deployment_bytes"]
        ):
            raise ValueError("Deployment size differs from the file inventory")
    for case in cases:
        if case["engine"] != "opennn" or case["precision"] != "fp32":
            continue
        peer = next(
            x
            for x in cases
            if x["engine"] == "pytorch"
            and all(x[k] == case[k] for k in ("device", "family", "precision"))
        )
        # Merge precision rows only when the actual file inventories are equal.
        if case["device"] == "cuda":
            for engine_case in (case, peer):
                bf16 = next(
                    x
                    for x in cases
                    if x["engine"] == engine_case["engine"]
                    and x["backend"] == engine_case["backend"]
                    and x["family"] == case["family"]
                    and x["precision"] == "bf16"
                )
                if engine_case["deployment_files"] != bf16["deployment_files"]:
                    raise ValueError(
                        "FP32 and BF16 use different deployment files; keep separate rows"
                    )
        rows.append(
            {
                "model": case["family"],
                "backend": case["backend"],
                "device": case["device"],
                "opennn_MB": case["deployment_bytes"] / 1e6,
                "pytorch_MB": peer["deployment_bytes"] / 1e6,
                "percent_of_pytorch": 100
                * case["deployment_bytes"]
                / peer["deployment_bytes"],
                "status": "Needs packaging check",
            }
        )
    return rows


def describe_artifact(path):
    description = {
        "kind": path.suffix.lstrip(".") or "file",
        "recorded_status": "",
        "session": "",
    }
    if path.suffix != ".json":
        return description
    try:
        data = read(path)
    except (ValueError, UnicodeError):
        return {**description, "kind": "unreadable_json"}
    if not isinstance(data, dict):
        return description
    description["recorded_status"] = str(
        data.get("status", data.get("label", "")) or ""
    )
    description["session"] = str(data.get("session_id", ""))
    if "benchmark_id" in data:
        description["kind"] = str(data["benchmark_id"])
    elif "protocol" in data and isinstance(data["protocol"], str):
        description["kind"] = data["protocol"][:100]
    elif data.get("schema") == "quality-v1":
        runs = data.get("runs", [])
        description["kind"] = (
            "quality_smoke"
            if runs and all(r.get("smoke") is True for r in runs)
            else "quality"
        )
    elif "command" in data:
        description["kind"] = "raw_launch"
    return description


def inventory(root, excluded, pinned):
    groups = defaultdict(list)
    previous_reviews = []
    for marker in (root / "results").rglob("verification.json"):
        try:
            if read(marker).get("status") == "review_only":
                previous_reviews.append(marker.parent)
        except (ValueError, AttributeError):
            pass
    for path in sorted((root / "results").rglob("*")):
        if (
            not path.is_file()
            or path.is_relative_to(excluded)
            or any(path.is_relative_to(p) for p in previous_reviews)
        ):
            continue
        relative = path.relative_to(root).as_posix()
        groups[digest(path)].append({"path": relative, "bytes": path.stat().st_size})
    rows = []
    for sha, aliases in sorted(groups.items()):
        aliases.sort(key=lambda x: (x["path"] not in pinned, len(x["path"]), x["path"]))
        rows.append(
            {
                "sha256": sha,
                "bytes": aliases[0]["bytes"],
                "canonical_path": aliases[0]["path"],
                "copies": len(aliases),
                "paths": [x["path"] for x in aliases],
                "role": "selected_for_review"
                if aliases[0]["path"] in pinned
                else "archived_or_supporting",
                **describe_artifact(root / aliases[0]["path"]),
            }
        )
    return rows


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    return (
        "\n".join(
            [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            + [
                "| "
                + " | ".join(str(v).replace("|", " / ").replace("\n", " ") for v in row)
                + " |"
                for row in rows
            ]
        )
        + "\n"
    )


def num(value, decimals=0):
    return f"{value:,.{decimals}f}" if value is not None else "Not measured"


def label(row):
    return f"{row['device'].upper().replace('CUDA', 'GPU')} {NAMES[row['model']]} {'training' if row['mode'] == 'train' else 'inference'}"


def render_review(selection, perf, start, deploy, catalog, facts):
    sections = [
        "# OpenNN and PyTorch: publication review\n",
        "Reviewed on 11 September 2026. **These observations are not ready for the next website release.**\n",
        "We compare OpenNN C++ applications with the PyTorch Python API. The tables below preserve the measured values, explain their limits and identify the work that remains. They do not claim that every comparison is valid or that OpenNN always wins.\n",
        "## What is available\n",
        table(
            ["Comparison", "Evidence", "Next step"],
            [
                [
                    "Throughput, memory and energy",
                    "12 historical configurations",
                    "Repeat with complete input records and resolve the workload differences below.",
                ],
                [
                    "CPU CNN and Transformer",
                    "No selected training or inference results",
                    "Measure the four missing configurations.",
                ],
                [
                    "Startup",
                    f"{start['timed_launches']} timed launches; {start['failures']} failed",
                    "Repeat under stable machine conditions.",
                ],
                [
                    "Deployment size",
                    "12 application configurations; file totals verified",
                    "Check the bundles on a clean target and record the exact build and packages.",
                ],
                [
                    "Prediction quality",
                    "Four model paths pass synthetic smoke checks",
                    "Train on real data and evaluate held-out predictions.",
                ],
                [
                    "Code size and dependencies",
                    "Dated source counts and installation inventories",
                    "Pin both source revisions and use the same counting rule.",
                ],
            ],
        ),
        "## Computers and software\n",
        "The performance records come from the reference computer: Intel Core i7-14700F, 32 GB RAM, NVIDIA RTX 5070 Ti, Linux, PyTorch 2.13.0+cu130 and NVIDIA driver 610.43.02. The CPU runs use FP32; GPU runs are labelled BF16, with an LSTM precision discrepancy described below.\n",
        "Eight performance records belong to session `2026-09-06-publish` at `e76425bd3`. The four GPU CNN and Transformer records belong to `2026-09-06-energy-publish` at `520f0f2f8`. Each ratio uses its own paired session. These records do not describe the current checkout.\n",
        "Startup and deployment come from a different computer: Intel Core i7-12700H and RTX 3060 Laptop GPU under WSL2. OpenNN uses CUDA 12.9; the PyTorch wheel uses CUDA 13.0, with separate cuDNN 9.20 distributions. These observations stay separate from reference-computer results.\n",
        "## Throughput\n",
        "Throughput counts completed work per second after warm-up. The value is the median of three launches. It does not measure how long a model takes to reach a target accuracy. Each model has its own unit.\n",
        table(
            [
                "Configuration",
                "Batch",
                "Precision",
                "Unit",
                "OpenNN",
                "PyTorch",
                "Ratio",
                "Variation ON / PT",
            ],
            [
                [
                    label(r),
                    r["batch"],
                    r["precision"],
                    UNITS[r["model"]],
                    num(r["engines"]["opennn"]["throughput"]["median"]),
                    num(r["engines"]["pytorch"]["throughput"]["median"]),
                    f"{r['engines']['opennn']['throughput']['median'] / r['engines']['pytorch']['throughput']['median']:.3f}×",
                    f"{r['engines']['opennn']['throughput']['cv_percent']:.3g}% / {r['engines']['pytorch']['throughput']['cv_percent']:.3g}%",
                ]
                for r in perf
            ],
        ),
        "OpenNN throughput ÷ PyTorch throughput. Higher is better. Variation is the sample standard deviation divided by the mean, expressed as a percentage. A value above 3% fails the current stability rule.\n",
        "CPU CNN training, CPU CNN inference, CPU Transformer training and CPU Transformer inference are **not measured in this selected result set**. They must remain visible as gaps.\n",
        "## Memory\n",
        "The table shows the highest measured memory use across three launches. CPU memory is the process's peak anonymous resident memory. GPU memory is total device memory in use minus the idle baseline. It includes runtime and allocator overhead. These are different measures and remain labelled separately.\n",
        table(
            ["Configuration", "OpenNN (MiB)", "PyTorch (MiB)", "OpenNN / PyTorch"],
            [
                [
                    label(r),
                    num(r["engines"]["opennn"]["memory"]["max"], 1),
                    num(r["engines"]["pytorch"]["memory"]["max"], 1),
                    f"{100 * r['engines']['opennn']['memory']['max'] / r['engines']['pytorch']['memory']['max']:.1f}%",
                ]
                for r in perf
            ],
        ),
        "OpenNN memory ÷ PyTorch memory × 100. Lower is better. For example, 60% means OpenNN uses 60% as much memory, or 40% less. These measurements do not establish a maximum model or dataset size.\n",
        "## Energy\n",
        "Energy is the median across three launches for the work recorded in each configuration. CPU readings measure the processor package. GPU readings measure the graphics board. Neither measures electricity at the wall socket.\n",
        table(
            [
                "Configuration",
                "Measured component",
                "OpenNN (Wh)",
                "PyTorch (Wh)",
                "OpenNN / PyTorch",
            ],
            [
                [
                    label(r),
                    "CPU package" if r["device"] == "cpu" else "GPU board",
                    num(r["engines"]["opennn"]["energy"]["median"], 4),
                    num(r["engines"]["pytorch"]["energy"]["median"], 4),
                    f"{100 * r['engines']['opennn']['energy']['median'] / r['engines']['pytorch']['energy']['median']:.1f}%",
                ]
                for r in perf
            ],
        ),
        "OpenNN energy ÷ PyTorch energy × 100. Lower is better. We do not average CPU package energy with GPU board energy, or infer an electricity-bill reduction from these component measurements.\n",
        "## Why the performance records need another run\n",
        "All twelve records lack content hashes for their prepared inputs. A filename, size and modification date do not prove that the engines received the same data. Ten records also have missing numerical quality values despite an old passing flag. That flag does not prove prediction quality or output agreement.\n",
        "The recomputed checks also flag PyTorch memory variation in CPU dense training. Full per-engine ranges, variation and check results are in the accompanying observations and readiness files.\n",
    ]
    for model in MODELS:
        sections += [
            f"### {NAMES[model]}\n",
            " ".join(selection["model_notes"][model]) + "\n",
        ]
    sections += [
        "The current code includes changes made after these runs. Fixes in the code cannot repair an old measurement; the affected comparison must run again. Different vendor libraries can also affect results, so conclusions must describe the tested configurations rather than assign every difference to framework code.\n",
        "## Startup\n",
        f"Every sample starts a fresh process and stops when its first prediction reaches host memory. Each configuration has 15 timed launches. **{start['unstable_cells']} of {len(start['cells'])} engine configurations exceed the 3% variation limit.** All paired startup comparisons therefore remain diagnostic.\n",
        table(
            [
                "Configuration",
                "Model",
                "OpenNN (ms)",
                "PyTorch (ms)",
                "OpenNN / PyTorch",
            ],
            [
                [
                    r["backend"]
                    + " / "
                    + r["precision"]
                    + (
                        " / "
                        + ("saved cache" if r["cache"] == "reused" else "empty cache")
                        if r["backend"] == "cuda"
                        else ""
                    ),
                    NAMES[r["family"]],
                    num(r["opennn_ms"], 3),
                    num(r["pytorch_python_ms"], 3),
                    f"{r['opennn_percent_of_pytorch']:.1f}%",
                ]
                for r in start["comparisons"]
            ],
        ),
        "OpenNN startup time ÷ PyTorch startup time × 100. Lower is better. Saved or empty cache refers to application tuning; filesystem caches are warm. No trained model file is loaded. The older footprint timer measured process lifetime and must not be used in this table.\n",
        "## Deployment size\n",
        "These values count files used to deploy each small application. OpenNN includes its executable and exercised native libraries. PyTorch includes the application, interpreter, standard library and complete installed runtime packages. This is a comparison with a standard Python installation, not two bundles reduced to their smallest possible size.\n",
        table(
            [
                "Backend",
                "Application",
                "OpenNN (MB)",
                "PyTorch (MB)",
                "OpenNN / PyTorch",
            ],
            [
                [
                    r["backend"],
                    NAMES[r["model"]],
                    num(r["opennn_MB"], 1),
                    num(r["pytorch_MB"], 1),
                    f"{r['percent_of_pytorch']:.1f}%",
                ]
                for r in deploy
            ],
        ),
        "OpenNN deployment size ÷ PyTorch deployment size × 100. Lower is better. MB means 1,000,000 bytes. GPU FP32 and BF16 rows share the same verified file inventory and are shown once. The figures exclude trained weights, datasets, operating-system and driver files, and generated caches. A clean-machine launch is still needed to show that each collected bundle is complete.\n",
        "## Prediction quality\n",
        table(
            ["Model", "Task", "Metric", "OpenNN", "PyTorch", "Status"],
            [
                [
                    "Dense",
                    "HIGGS classification",
                    "Test accuracy (%)",
                    "Not measured",
                    "Not measured",
                    "Train and evaluate",
                ],
                [
                    "LSTM",
                    "PM2.5 forecasting",
                    "Test RMSE in original units",
                    "Not measured",
                    "Not measured",
                    "Train and evaluate",
                ],
                [
                    "CNN",
                    "Image classification",
                    "Test top-1 accuracy (%)",
                    "Not measured",
                    "Not measured",
                    "Train and evaluate",
                ],
                [
                    "Transformer",
                    "English–German translation",
                    "Generated-translation BLEU",
                    "Not measured",
                    "Not measured",
                    "Train and evaluate",
                ],
            ],
        ),
        "Use five independent seeds and report the mean and standard deviation for each model. Set an acceptable quality difference before inspecting the final scores. Accuracy and BLEU are higher-is-better; RMSE is lower-is-better. Do not average them into one quality score. Synthetic smoke results and the older Rosenbrock task do not fill these rows.\n",
        "## Code and dependencies\n",
        f"The source audit at `{facts['git']['commit'][:9]}` counted {facts['source']['library']['code_lines']:,} OpenNN source lines after removing blanks and comments. Its application example counted {facts['source']['application_lines']['opennn']['code_lines']} OpenNN lines and {facts['source']['application_lines']['pytorch']['code_lines']} PyTorch lines. The separate counts of 13 and 29 describe selected statement lines under language-specific rules; they must not replace those line counts.\n",
        "The older PyTorch library figure of 834,319 lines has no matching raw source-count record in this selected evidence. Recount both repositories at pinned revisions with the same inclusion rules before publishing a codebase ratio. A smaller source tree is not evidence of equal functionality or easier maintenance.\n",
        "Count Python packages and native libraries separately. OpenNN requires no Python packages, but it still uses native dependencies. CPU and GPU PyTorch installations have different package lists. Use the saved package inventory for the chosen deployment configuration; do not state that OpenNN has zero dependencies.\n",
        "## Publication sequence\n",
        "1. Choose one release commit, record all runtime versions, and prepare inputs with content hashes.\n"
        "2. Confirm equal inputs, model behavior, precision, warm-up and measured work. Resolve the CNN, LSTM and Transformer differences listed above.\n"
        "3. Run the complete CPU/GPU training and inference matrix on the reference computer. Retain every launch, including failures.\n"
        "4. Complete quality training, repeat stable startup measurements, and test deployment bundles on a clean target.\n"
        "5. Recalculate every table from raw evidence, review the checks, and publish only the sections that meet their stated requirements.\n",
        "Keep unmeasured cells visible. A partial release must name its measured scope and must not claim that all models, devices or metrics improved. No overall improvement is reported while the intended comparison remains incomplete.\n",
        "## Evidence and preservation\n",
        f"The catalog covers {sum(x['copies'] for x in catalog):,} files and {len(catalog):,} unique contents. Files with identical SHA-256 hashes share one catalog entry with all their paths. No measurement file was edited or deleted. Old reports are preserved under `reports/archive/2026-09-11/`.\n",
        "The accompanying `performance.json`, `observations.csv`, `startup.json`, `deployment.json`, `readiness.csv` and `catalog.json` retain source paths, hashes, raw-derived statistics and pending checks. `publication/selection.json` pins the evidence; selecting a file does not approve it for publication.\n",
    ]
    return "\n".join(sections)


def inline(text):
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)


def preview(markdown):
    """Small escaped Markdown subset; no remote assets or executable source HTML."""
    output, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("| "):
            rows = []
            while i < len(lines) and lines[i].startswith("| "):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(c and set(c) <= {"-", ":"} for c in cells):
                    tag = "th" if not rows else "td"
                    rows.append(
                        "<tr>"
                        + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in cells)
                        + "</tr>"
                    )
                i += 1
            output.append(
                '<div class="table"><table>' + "".join(rows) + "</table></div>"
            )
            continue
        level = len(line) - len(line.lstrip("#"))
        if level and line[level : level + 1] == " ":
            output.append(f"<h{level}>" + inline(line[level:].strip()) + f"</h{level}>")
        elif line:
            output.append("<p>" + inline(line) + "</p>")
        i += 1
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OpenNN benchmark publication review</title><style>body{font:17px/1.6 system-ui,sans-serif;color:#27343c;max-width:1150px;margin:40px auto;padding:0 24px}h1,h2,h3{color:#287b9b;line-height:1.2}h2{margin-top:2.5em}p{max-width:95ch}.table{overflow-x:auto}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{padding:10px 12px;border-bottom:1px solid #dce3e7;text-align:left}th{background:#e9f4f8}tr:nth-child(even){background:#f6f8fa}@media print{body{font-size:11px;margin:0}table{font-size:9px}h2{break-after:avoid}.table{overflow:visible}}</style><main>'
        + "\n".join(output)
        + "</main></html>\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selection", type=Path, default=BENCHMARKS / "publication/selection.json"
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="New directory below benchmarks/results/scratch",
    )
    args = parser.parse_args(argv)
    out = args.out.resolve()
    scratch = (BENCHMARKS / "results/scratch").resolve()
    if out == scratch or not out.is_relative_to(scratch) or out.exists():
        parser.error("Output must be a new directory below benchmarks/results/scratch")
    selection = read(args.selection)
    sources = selection["performance"] + [
        selection[x] for x in ("startup", "deployment", "source_counts")
    ]
    for entry in sources:
        path = (BENCHMARKS / entry["path"]).resolve()
        if (
            not path.is_relative_to(BENCHMARKS / "results")
            or digest(path) != entry["sha256"]
        ):
            raise ValueError(f"Source path or hash mismatch: {entry['path']}")
    perf = [
        performance(read(BENCHMARKS / s["path"]), s) for s in selection["performance"]
    ]
    start_path = BENCHMARKS / selection["startup"]["path"]
    start = startup(read(start_path), start_path)
    deploy = deployment(read(BENCHMARKS / selection["deployment"]["path"]))
    catalog = inventory(BENCHMARKS, out, {s["path"] for s in sources})
    facts = read(BENCHMARKS / selection["source_counts"]["path"])
    out.mkdir(parents=True)
    for name, value in (
        ("performance", perf),
        ("startup", start),
        ("deployment", deploy),
        ("catalog", catalog),
    ):
        write_json(out / f"{name}.json", value)
    write_json(out / "selection.json", selection)
    provenance = []
    for source in sources:
        data = read(BENCHMARKS / source["path"])
        provenance.append(
            {
                "source": source["path"],
                "sha256": source["sha256"],
                "recorded": {
                    k: data[k]
                    for k in (
                        "git",
                        "session_id",
                        "configuration",
                        "config",
                        "frameworks",
                        "machine",
                        "cpu",
                        "host",
                        "datasets",
                        "provenance",
                    )
                    if k in data
                },
                "commands": [
                    r["command"]
                    for r in data.get("launches", data.get("cases", []))
                    if "command" in r
                ],
            }
        )
    write_json(out / "provenance.json", provenance)
    write_csv(
        out / "catalog.csv", [{**r, "paths": " ; ".join(r["paths"])} for r in catalog]
    )
    observations, readiness = [], []
    for r in perf:
        readiness.append(
            {
                "comparison": r["id"],
                "status": r["status"],
                "next_step": " ".join(r["reasons"]),
                "source": r["source"],
            }
        )
        for engine, metrics in r["engines"].items():
            for metric in ("throughput", "memory", "energy"):
                unit = (
                    UNITS[r["model"]]
                    if metric == "throughput"
                    else ("MiB" if metric == "memory" else "Wh")
                )
                observations.append(
                    {
                        "comparison": r["id"],
                        "engine": engine,
                        "metric": metric,
                        "unit": unit,
                        "model": r["model"],
                        "device": r["device"],
                        "mode": r["mode"],
                        "batch": r["batch"],
                        "precision": r["precision"],
                        "session": r["session"],
                        "commit": r["commit"],
                        **metrics[metric],
                        "source": r["source"],
                        "sha256": r["sha256"],
                    }
                )
    for model in ("cnn", "transformer"):
        for mode in ("train", "infer"):
            readiness.append(
                {
                    "comparison": f"cpu-{model}-{mode}",
                    "status": "Not measured",
                    "next_step": "Measure the paired CPU configuration.",
                    "source": "",
                }
            )
    for model in MODELS:
        readiness.append(
            {
                "comparison": f"quality-{model}",
                "status": "Not measured",
                "next_step": "Train and evaluate on held-out real data with five seeds.",
                "source": "",
            }
        )
    readiness += [
        {
            "comparison": "startup",
            "status": start["status"],
            "next_step": start["reason"],
            "source": selection["startup"]["path"],
        },
        {
            "comparison": "deployment",
            "status": "Needs packaging check",
            "next_step": "Verify each bundle on a clean target and pin its build and runtime.",
            "source": selection["deployment"]["path"],
        },
        {
            "comparison": "source-counts",
            "status": "Needs recount",
            "next_step": "Count both pinned repositories and matched applications with one rule.",
            "source": selection["source_counts"]["path"],
        },
    ]
    write_csv(out / "observations.csv", observations)
    write_csv(out / "readiness.csv", readiness)
    review = render_review(selection, perf, start, deploy, catalog, facts)
    (out / "review.md").write_text(review, encoding="utf-8")
    (out / "review.html").write_text(preview(review), encoding="utf-8")
    write_json(
        out / "verification.json",
        {
            "status": "review_only",
            "publication_ready": False,
            "raw_sources_verified": len(sources),
            "performance_cells": len(perf),
            "startup_timed_launches": start["timed_launches"],
            "deployment_rows": len(deploy),
            "files_cataloged": sum(x["copies"] for x in catalog),
            "unique_contents": len(catalog),
        },
    )
    print(f"Review: {out / 'review.html'}")
    print(
        f"Verified {len(sources)} pinned sources; {len(perf)} performance records; {start['timed_launches']} startup launches; {len(deploy)} deployment rows."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
