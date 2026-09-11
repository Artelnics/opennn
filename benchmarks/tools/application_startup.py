"""Linux/WSL startup protocol v2: launch -> first completed host prediction.

Fresh processes, warmed filesystem cache, separate reused/empty application
caches. No in-process inference warm-up. This is a local diagnostic protocol;
the repository's reference-machine/environmental publication gates still apply.
"""

import argparse, csv, datetime, hashlib, json, math, os, platform, random
import statistics, subprocess, time, sys
from pathlib import Path
from experiment import git_metadata, result_directory
from application_tables import write_table

FAMILIES = ["dense", "lstm", "cnn", "transformer"]
EXPECTED = {
    "dense": (20353, 2),
    "lstm": (73857, 2),
    "cnn": (268650, 20),
    "transformer": (33792, 2048),
}
THREAD_ENV = {
    "OMP_NUM_THREADS": "2",
    "OPENNN_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_THREADING_LAYER": "GNU",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
    "OMP_WAIT_POLICY": "PASSIVE",
    "GOMP_SPINCOUNT": "0",
}


def command_output(command):
    try:
        p = subprocess.run(command, capture_output=True, text=True, timeout=30)
        return {"returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def cpu_snapshot():
    return [int(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]]


def cpu_busy(before, after):
    delta = [b - a for a, b in zip(before, after)]
    return 100 * (sum(delta) - delta[3] - delta[4]) / sum(delta) if sum(delta) else None


def gpu_snapshot():
    return command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,temperature.gpu,utilization.gpu,memory.used,clocks.gr,clocks.mem,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )


def make_cases(config):
    groups = config.get("groups", [])
    if not groups:
        raise ValueError("Configuration must contain application groups")
    keys = [(g["engine"], g["backend"]) for g in groups]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate engine/backend group")
    for group in groups:
        if group["device"] not in ("cpu", "cuda") or group["engine"] not in (
            "opennn",
            "pytorch",
        ):
            raise ValueError("Unsupported engine/device")
        if group["engine"] == "pytorch" and not group.get("prefix_args"):
            raise ValueError("PyTorch baseline must use the Python application")
    for group in groups:
        if (
            group["engine"] == "opennn"
            and sum(
                g["engine"] == "pytorch" and g["device"] == group["device"]
                for g in groups
            )
            != 1
        ):
            raise ValueError(
                "Each OpenNN device needs exactly one PyTorch Python baseline"
            )
    cases = []
    for group in config["groups"]:
        for family in FAMILIES:
            for precision in (
                ["fp32", "bf16"] if group["device"] == "cuda" else ["fp32"]
            ):
                for cache in (
                    ["reused", "empty"] if group["device"] == "cuda" else ["reused"]
                ):
                    cases.append(
                        {
                            **group,
                            "family": family,
                            "precision": precision,
                            "cache": cache,
                            "binary": group["binary_pattern"].format(family=family),
                            "id": f"{group['engine']}-{group['backend']}-{family}-{precision}-{cache}",
                        }
                    )
    return cases


def launch(case, out, label, timeout, trace=False):
    binary = Path(case["binary"]).absolute()
    cache = (
        out / "caches" / case["id"] / (label if case["cache"] == "empty" else "shared")
    )
    cache.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            ("OPENNN_", "CUDA_", "CUDNN_", "TORCH_", "TRITON_", "MKL_", "OMP_", "GOMP_")
        ) or key in ["LD_DEBUG", "LD_PRELOAD"]:
            env.pop(key, None)
    env.update(THREAD_ENV)
    env.update(case.get("env", {}))
    env.update(
        {
            "OPENNN_LT_PLAN_CACHE_DIR": str(cache / "opennn-lt"),
            "CUDA_CACHE_PATH": str(cache / "cuda"),
            "XDG_CACHE_HOME": str(cache / "xdg"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
            "TRITON_CACHE_DIR": str(cache / "triton"),
        }
    )
    for key in [
        "OPENNN_LT_PLAN_CACHE_DIR",
        "CUDA_CACHE_PATH",
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ]:
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if trace:
        env["LD_DEBUG"] = "files"
    args = [
        str(binary),
        *[v.format(family=case["family"]) for v in case.get("prefix_args", [])],
        case["device"],
        case["precision"],
    ]
    time.sleep(
        0.2
    )  # Fixed idle gap outside the timed region; same policy for every engine.
    before_cpu = cpu_snapshot()
    started = time.monotonic_ns()
    try:
        p = subprocess.run(
            args, capture_output=True, text=True, env=env, timeout=timeout
        )
        exited = time.monotonic_ns()
        record = {
            "case": case["id"],
            "label": label,
            "started_ns": started,
            "exited_ns": exited,
            "returncode": p.returncode,
            "stdout": p.stdout,
            "stderr": p.stderr,
            "cpu_busy_during_process_percent": cpu_busy(before_cpu, cpu_snapshot()),
            "cache_directory": str(cache),
            "command": args,
        }
        markers = [
            json.loads(line[len("STARTUP_READY ") :])
            for line in p.stdout.splitlines()
            if line.startswith("STARTUP_READY ")
        ]
        if p.returncode or len(markers) != 1:
            raise ValueError("Failed process or missing/duplicate ready marker")
        marker = markers[0]
        if not started <= marker["main_ns"] <= marker["ready_ns"] <= exited:
            raise ValueError("Inconsistent monotonic timestamps")
        if (
            marker["engine"] != case["engine"]
            or marker["device"] != case["device"]
            or marker["precision"] != case["precision"]
        ):
            raise ValueError("Unexpected engine/device/precision")
        if (marker["parameters"], marker["output_values"]) != EXPECTED[case["family"]]:
            raise ValueError("Shape/parameter gate failed")
        if not math.isfinite(marker["first"]):
            raise ValueError("Nonfinite output")
        if case["engine"] == "pytorch":
            if marker.get("interface") != "python":
                raise ValueError("Wrong PyTorch interface")
            if Path(marker["python_prefix"]) != Path(case["binary"]).parent.parent:
                raise ValueError("Wrong Python environment")
        record.update(
            status="ok",
            marker=marker,
            latency_ms=(marker["ready_ns"] - started) / 1e6,
            launch_to_main_ms=(marker["main_ns"] - started) / 1e6,
            process_lifetime_ms=(exited - started) / 1e6,
        )
    except (subprocess.TimeoutExpired, ValueError) as error:
        if "record" not in locals():
            record = {"case": case["id"], "label": label, "command": args}
        record.update(status="failed", error=str(error))
    raw = out / "raw"
    raw.mkdir(exist_ok=True)
    (raw / f"{label}-{case['id']}.json").write_text(json.dumps(record, indent=2))
    return record


def summarize(records, cases):
    rows = []
    for case in cases:
        samples = [r for r in records if r["case"] == case["id"]]
        good = [r["latency_ms"] for r in samples if r["status"] == "ok"]
        row = {
            k: case[k]
            for k in [
                "id",
                "engine",
                "backend",
                "device",
                "family",
                "precision",
                "cache",
            ]
        }
        row.update(count=len(good), failures=len(samples) - len(good))
        if good:
            row.update(
                median_ms=statistics.median(good),
                min_ms=min(good),
                max_ms=max(good),
                mean_ms=statistics.mean(good),
                cv_percent=100 * statistics.stdev(good) / statistics.mean(good)
                if len(good) > 1
                else None,
                median_lifetime_ms=statistics.median(
                    r["process_lifetime_ms"] for r in samples if r["status"] == "ok"
                ),
            )
            row["variation_assessed"] = row["cv_percent"] is not None
            row["variation_over_3_percent"] = (
                row["cv_percent"] is not None and row["cv_percent"] > 3
            )
        rows.append(row)
    return rows


def comparisons(rows):
    answer = []
    for row in rows:
        if row["engine"] != "opennn" or not row["count"]:
            continue
        peer = next(
            r
            for r in rows
            if r["engine"] == "pytorch"
            and all(r[k] == row[k] for k in ["device", "family", "precision", "cache"])
        )
        if not peer["count"]:
            continue
        answer.append(
            {k: row[k] for k in ["backend", "family", "precision", "cache"]}
            | {
                "opennn_ms": row["median_ms"],
                "pytorch_python_ms": peer["median_ms"],
                "opennn_percent_of_pytorch": 100 * row["median_ms"] / peer["median_ms"],
                "pytorch_divided_by_opennn": peer["median_ms"] / row["median_ms"],
                "opennn_cv_percent": row["cv_percent"],
                "pytorch_cv_percent": peer["cv_percent"],
                "opennn_min_ms": row["min_ms"],
                "opennn_max_ms": row["max_ms"],
                "pytorch_min_ms": peer["min_ms"],
                "pytorch_max_ms": peer["max_ms"],
                "opennn_n": row["count"],
                "pytorch_n": peer["count"],
                "variation_assessed": row["variation_assessed"]
                and peer["variation_assessed"],
                "variation_over_3_percent": row["variation_over_3_percent"]
                or peer["variation_over_3_percent"],
            }
        )
    return answer


def export(out, config, provenance, records, cases):
    rows = summarize(records, cases)
    pairs = comparisons(rows)
    result = {
        "protocol": "startup-python-v2",
        "status": "local_diagnostic",
        "config": config,
        "provenance": provenance,
        "cells": rows,
        "comparisons": pairs,
    }
    (out / "results.json").write_text(json.dumps(result, indent=2))
    for name, values in [("cells.csv", rows), ("comparisons.csv", pairs)]:
        if values:
            with (out / name).open("w", newline="") as stream:
                fields = list(dict.fromkeys(k for row in values for k in row))
                writer = csv.DictWriter(stream, fields)
                writer.writeheader()
                writer.writerows(values)
    write_table(out, "startup", pairs)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["startup"], default="startup")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--out", type=Path, help="New directory under benchmarks/results/scratch"
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--affinity",
        help="Optional comma-separated Linux CPU IDs, same for all children",
    )
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Application startup measurements currently require Linux/WSL")
    if args.rounds < 1 or args.repeats < 1 or args.timeout <= 0:
        parser.error("Rounds, repeats and timeout must be positive")
    config = json.loads(args.config.read_text())
    out = result_directory("startup", args.out)
    if args.affinity:
        os.sched_setaffinity(0, {int(n) for n in args.affinity.split(",")})
    cases = make_cases(config)
    for case in cases:
        if not Path(case["binary"]).is_file():
            raise FileNotFoundError(case["binary"])
    provenance = {
        "git": git_metadata(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "platform": platform.platform(),
        "cpu": command_output(["lscpu"]),
        "gpu_before": gpu_snapshot(),
        "compiler": command_output(["g++", "--version"]),
        "between_launches_seconds": 0.2,
        "interface": "PyTorch Python vs OpenNN C++",
        "affinity": sorted(os.sched_getaffinity(0)),
        "threads": THREAD_ENV,
        "rounds": args.rounds,
        "repeats": args.repeats,
        "binary_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted({Path(c["binary"]) for c in cases})
        },
        "environment_control": "Diagnostic: this runner does not enforce clock locking or host idleness. Record and review external machine controls; WSL cannot fully observe Windows host activity.",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    records = []
    # The smoke also verifies resolved runtime libraries and exact work metadata.
    # Reused caches are warmed by this separate process; empty-policy samples
    # always receive new task-owned directories. Filesystem caches are not dropped.
    for case in cases:
        r = launch(case, out, "warmup", args.timeout, trace=args.smoke)
        print(
            "warmup",
            case["id"],
            r["status"],
            round(r.get("latency_ms", 0), 2),
            flush=True,
        )
        if r["status"] != "ok":
            print(json.dumps(r)[-3000:], flush=True)
            raise RuntimeError("Warmup/shape gate failed; see raw record")
    if args.smoke:
        print(
            "Smoke passed for",
            len(cases),
            "cases; smoke timings are not performance results.",
            flush=True,
        )
        return
    for round_index in range(args.rounds):
        idle_start = cpu_snapshot()
        time.sleep(1)
        (out / f"round-{round_index + 1}-environment.json").write_text(
            json.dumps(
                {
                    "idle_cpu_busy_percent": cpu_busy(idle_start, cpu_snapshot()),
                    "gpu": gpu_snapshot(),
                },
                indent=2,
            )
        )
        for repeat in range(args.repeats):
            # Randomized order is fixed in advance and changes every pass.
            order = cases.copy()
            random.Random(42000 + round_index * 100 + repeat).shuffle(order)
            for case in order:
                label = f"r{round_index + 1:02}-n{repeat + 1:02}"
                r = launch(case, out, label, args.timeout)
                records.append(r)
                if r["status"] != "ok":
                    print("FAILED", label, case["id"], r.get("error"), flush=True)
            print(
                "round",
                round_index + 1,
                "repeat",
                repeat + 1,
                "completed;",
                len(records),
                "timed launches",
                flush=True,
            )
            export(out, config, provenance, records, cases)
    provenance["gpu_after"] = gpu_snapshot()
    result = export(out, config, provenance, records, cases)
    print(
        "COMPLETE",
        len(records),
        "launches;",
        sum(r["status"] != "ok" for r in records),
        "failures",
        flush=True,
    )
    for row in result["comparisons"]:
        print(
            row["backend"],
            row["family"],
            row["precision"],
            row["cache"],
            round(row["opennn_ms"], 2),
            round(row["pytorch_python_ms"], 2),
            round(row["pytorch_divided_by_opennn"], 2),
            flush=True,
        )

    return 1 if any(r["status"] != "ok" for r in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
