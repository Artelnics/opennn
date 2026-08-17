#!/usr/bin/env python3
"""Peak-batch training throughput: each engine at its own best batch size.

The fixed-batch throughput benchmarks compare engines at one common batch. This
one asks the question the capacity benchmarks set up: **how fast can each
engine train when it is allowed to pick its own batch size?** For every
(engine, precision) it sweeps the batch geometrically upward from the standard
size, measures training samples/s with the SAME speed drivers the fixed-batch
benchmarks use (one fresh process per point, so an OOM never poisons the next
point), and reports each engine's throughput curve, its peak, and the batch
where its curve dies.

This is a hardware/runtime saturation benchmark, not a time-to-quality one:
giant batches change convergence behaviour (see quality/convergence for the
quality-gated comparison). No new engine code — the drivers are the ones in
../higgs-gpu, ../resnet50, and ../attention-speed.

  usage: run_peak_batch_speed.py --family higgs|resnet50|transformer
                                 [--precisions bf16,fp32] [--engines opennn,pytorch,tensorflow]
                                 [--epochs N] [--timeout-s 900] [--max-batch N] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BENCH_ROOT = HERE.parent.parent
RESULTS_DIR = (BENCH_ROOT / "results").resolve()
DEFAULT_BENCH_DATA = Path(
    os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data"))
)
PY = os.environ.get("BENCH_PYTHON", sys.executable)

OOM_MARKERS = (
    "out of memory", "OOM", "RESOURCE_EXHAUSTED", "CUDA_ERROR_OUT_OF_MEMORY",
    "cudaErrorMemoryAllocation", "bad_alloc", "Not enough memory",
    "CUDNN_STATUS_ALLOC_FAILED", "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDA Error: 2 ", "cudaMalloc(",
)

def run_text(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except Exception:
        return ""

def repo_root() -> Path:
    root = run_text(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else HERE.parents[3]

REPO_ROOT = repo_root()

def find_binary(base: str, env_override: str) -> str:
    override = os.environ.get(env_override)
    if override:
        return override
    for directory in ("build", "build-benchmarks"):
        candidate = REPO_ROOT / directory / "bin" / base
        if candidate.exists():
            return str(candidate)
    return str(REPO_ROOT / "build" / "bin" / base)

def tensorflow_library_dirs(py: str) -> list[str]:
    override = os.environ.get("TF_NV_LIBS")
    if override:
        return [p for p in override.split(os.pathsep) if p]
    code = ("import json, site\nfrom pathlib import Path\n"
            "roots = []\n"
            "for base in list(site.getsitepackages()) + [site.getusersitepackages()]:\n"
            "    nv = Path(base) / 'nvidia'\n"
            "    if nv.exists():\n"
            "        roots.extend(str(p) for p in nv.rglob('lib') if p.is_dir())\n"
            "print(json.dumps(roots))")
    try:
        out = subprocess.run([py, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        return json.loads(lines[-1]) if lines else []
    except Exception:
        return []

class Family:
    def __init__(self, name: str, start_batch: int, max_batch: int, epochs: int,
                 cwd: Path | None = None):
        self.name = name
        self.start_batch = start_batch
        self.max_batch = max_batch
        self.epochs = epochs
        self.cwd = cwd

    def cmd(self, engine: str, precision: str, batch: int, epochs: int
            ) -> tuple[list[str], dict[str, str]]:
        raise NotImplementedError

    def engine_paths(self) -> dict[str, str]:
        return {"opennn": "CUDA graphs, GPU-resident data",
                "pytorch": "eager", "tensorflow": "tf.function(jit_compile=True)"}

class HiggsFamily(Family):
    def __init__(self) -> None:
        super().__init__("higgs", start_batch=7000, max_batch=10_500_000, epochs=3)
        self.train = DEFAULT_BENCH_DATA / "higgs" / "higgs_train.csv"
        self.test = DEFAULT_BENCH_DATA / "higgs" / "higgs_test.csv"
        self.drivers = BENCH_ROOT / "throughput" / "higgs-gpu"

    def cmd(self, engine, precision, batch, epochs):
        if engine == "opennn":
            binary = find_binary("opennn_speed", "OPENNN_SPEED_BIN")
            return ([binary, str(self.train), str(epochs), str(batch), precision,
                     "1024", "relu", "2", str(self.test), "none", "none", "none"],
                    {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:"})
        script = "pytorch_speed.py" if engine == "pytorch" else "tensorflow_speed.py"
        env = {}
        if engine == "tensorflow":
            env["LD_LIBRARY_PATH"] = os.pathsep.join(tensorflow_library_dirs(PY))
        return ([PY, str(self.drivers / script), str(self.train), str(epochs),
                 str(batch), precision, "shuffle", "1024", "relu", "2",
                 str(self.test), "none", "none", "none"], env)

class ResnetFamily(Family):
    def __init__(self) -> None:
        super().__init__("resnet50", start_batch=128, max_batch=50_000, epochs=3)
        self.data_dir = DEFAULT_BENCH_DATA / "cifar10"
        self.drivers = BENCH_ROOT / "throughput" / "resnet50"

    def cmd(self, engine, precision, batch, epochs):
        if engine == "opennn":
            binary = find_binary("opennn_resnet50_speed", "OPENNN_RESNET50_BIN")
            return ([binary, str(self.data_dir / "train"), str(epochs), str(batch),
                     precision], {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:"})
        script = ("pytorch_resnet50_speed.py" if engine == "pytorch"
                  else "tensorflow_resnet50_speed.py")
        env: dict[str, str] = {}
        if engine == "pytorch":
            env["PT_FAST"] = "1"
            if precision == "bf16":
                env["PT_BF16"] = "1"
            # PyTorch's strongest one-line options on top of the fast path -
            # torch.compile(mode="reduce-overhead") (CUDA graphs) and
            # Adam(fused=True) - are the protocol: a PyTorch user gets both for
            # free, and without them the sweep understated PyTorch by 2.5x at
            # batch 128, 1.4x at 512, 1.13x at 2048 (RTX 3060, bf16,
            # 2026-08-16). PYTORCH_PLAIN=1 reverts to the compile-default /
            # foreach-Adam path for comparison with older tables. TensorFlow
            # already runs its whole step under XLA (jit_compile), which is its
            # equivalent.
            if not os.environ.get("PYTORCH_PLAIN"):
                env["PT_COMPILE_MODE"] = "reduce-overhead"
                env["PT_FUSED_ADAM"] = "1"
        else:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(tensorflow_library_dirs(PY))
            if precision == "bf16":
                env["TF_BF16"] = "1"
        return ([PY, str(self.drivers / script), str(epochs), str(batch),
                 str(self.data_dir)], env)

    def engine_paths(self):
        return {
            "opennn": "CUDA graphs, cuDNN autotune, GPU-resident data",
            "pytorch": ("channels_last + torch.compile default + foreach Adam (PYTORCH_PLAIN)"
                        if os.environ.get("PYTORCH_PLAIN") else
                        "channels_last + torch.compile(mode=reduce-overhead) + Adam(fused=True)"),
            "tensorflow": "NHWC + tf.function(jit_compile=True) over the whole step",
        }

class TransformerFamily(Family):
    def __init__(self) -> None:
        drivers = BENCH_ROOT / "throughput" / "attention-speed"
        super().__init__("transformer", start_batch=32, max_batch=4096, epochs=5,
                         cwd=drivers)
        self.drivers = drivers
        # TRANSFORMER_CORPUS points every engine at one corpus file (absolute
        # path); by default the one next to the drivers. On WSL keep it on the
        # Linux disk: OpenNN reads its token cache per batch, and a corpus under
        # /mnt/c turns that into 9p round trips (measured 20-30 ms per batch).
        self.corpus = os.environ.get("TRANSFORMER_CORPUS", "synthetic_corpus.txt")
        self.shape = ("256", "8", "1024", "2")

    def engine_paths(self):
        return {
            "opennn": "CUDA graphs, fused (cuDNN) attention in both precisions, GPU-resident data",
            "pytorch": ("autocast + eager + foreach Adam (PYTORCH_PLAIN)"
                        if os.environ.get("PYTORCH_PLAIN") else
                        "autocast + torch.compile(mode=reduce-overhead) + Adam(fused=True)"),
            "tensorflow": "mixed_bfloat16 + tf.function(jit_compile=True) over the whole step",
        }

    def cmd(self, engine, precision, batch, epochs):
        args = [self.corpus, *self.shape, str(batch), str(epochs)]
        if engine == "opennn":
            binary = find_binary("opennn_transformer_train", "OPENNN_TRANSFORMER_TRAIN_BIN")
            env = {"LD_LIBRARY_PATH": "/usr/local/cuda/lib64:"}
            if precision == "bf16":
                env["OPENNN_BF16"] = "1"
            return [binary, *args], env
        script = ("pytorch_transformer_train.py" if engine == "pytorch"
                  else "tensorflow_transformer_train.py")
        env = {}
        if engine == "pytorch":
            if precision == "bf16":
                env["PT_BF16"] = "1"
            # Same protocol as the ResNet family: torch.compile(reduce-overhead)
            # + Adam(fused=True) unless PYTORCH_PLAIN=1.
            if not os.environ.get("PYTORCH_PLAIN"):
                env["PT_COMPILE_MODE"] = "reduce-overhead"
                env["PT_FUSED_ADAM"] = "1"
        if engine == "tensorflow":
            env["LD_LIBRARY_PATH"] = os.pathsep.join(tensorflow_library_dirs(PY))
            if precision == "bf16":
                env["TF_BF16"] = "1"
        return [PY, str(self.drivers / script), *args], env

FAMILIES = {"higgs": HiggsFamily, "resnet50": ResnetFamily, "transformer": TransformerFamily}

def classify_failure(raw: str) -> str:
    lowered = raw.lower()
    for marker in OOM_MARKERS:
        if marker.lower() in lowered:
            return "oom"
    return "error"

def run_point(cmd: list[str], env_over: dict[str, str], timeout_s: int,
              cwd: Path | None) -> tuple[dict[str, str], str, str]:
    env = dict(os.environ)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env.update(env_over)
    try:
        out = subprocess.run(cmd, env=env, cwd=str(cwd) if cwd else None,
                             capture_output=True, text=True, check=False,
                             timeout=timeout_s)
        raw = out.stdout + out.stderr
        status = "ok" if out.returncode == 0 else classify_failure(raw)
    except subprocess.TimeoutExpired as exc:
        raw = ((exc.stdout or "") + (exc.stderr or "")) if isinstance(exc.stdout, str) else ""
        status = "timeout"
    fields: dict[str, str] = {}
    for line in raw.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            fields[key.strip()] = value.strip()
    if status == "ok" and "samples_per_sec" not in fields:
        status = "error"
    return fields, status, raw

def batch_ladder(start: int, cap: int) -> list[int]:
    ladder = []
    batch = start
    while batch < cap:
        ladder.append(batch)
        batch *= 2
    ladder.append(cap)
    return ladder

def git_metadata() -> dict[str, Any]:
    commit = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
    branch = run_text(["git", "-C", str(REPO_ROOT), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_text(["git", "-C", str(REPO_ROOT), "status", "--short", "--untracked-files=no"])
    return {"commit": commit or "unknown", "branch": branch or "unknown",
            "dirty": bool(status.splitlines()),
            "status_short_count": len(status.splitlines())}

def versions() -> dict[str, Any]:
    v: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    code = ("import json\ninfo={}\n"
            "try:\n import torch; info['torch']=torch.__version__\n"
            "except Exception as e: info['torch_error']=str(e)\n"
            "try:\n import tensorflow as tf; info['tensorflow']=tf.__version__\n"
            "except Exception as e: info['tensorflow_error']=str(e)\n"
            "print(json.dumps(info))\n")
    try:
        out = subprocess.run([PY, "-c", code], capture_output=True, text=True, check=False)
        lines = [line for line in out.stdout.splitlines() if line.strip()]
        if lines:
            v.update(json.loads(lines[-1]))
    except Exception as exc:
        v["version_error"] = str(exc)
    return v

def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--family", required=True, choices=sorted(FAMILIES))
    parser.add_argument("--engines", default="opennn,pytorch,tensorflow")
    parser.add_argument("--precisions", default="bf16,fp32")
    parser.add_argument("--epochs", type=int, default=0,
                        help="timed epochs per point (0 = family default)")
    parser.add_argument("--start-batch", type=int, default=0,
                        help="first batch of the ladder (0 = family default)")
    parser.add_argument("--max-batch", type=int, default=0,
                        help="ladder cap (0 = family default: the training-set size)")
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="print every trial command without executing anything")
    args = parser.parse_args()

    family: Family = FAMILIES[args.family]()
    epochs = args.epochs or family.epochs
    start_batch = args.start_batch or family.start_batch
    cap = args.max_batch or family.max_batch
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    precisions = [p.strip() for p in args.precisions.split(",") if p.strip()]
    ladder = batch_ladder(start_batch, cap)

    if args.dry_run:
        for precision in precisions:
            for engine in engines:
                cmd, env = family.cmd(engine, precision, ladder[0], epochs)
                env_bits = " ".join(f"{k}={v}" for k, v in sorted(env.items()))
                print(f"[{precision}/{engine}] ladder={ladder}")
                print(f"  {env_bits} {shlex.join(cmd)}")
        return

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    benchmark_id = f"gpu-{args.family}-peak-batch-speed"
    out_path = args.output or RESULTS_DIR / f"{benchmark_id}-{run_id}.json"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing result file: {out_path}")

    result: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": benchmark_id,
        "run_id": run_id,
        "git": git_metadata(),
        "protocol": {
            "style": "throughput_at_own_best_batch",
            "benchmark_class": "training_throughput_saturation",
            "measurement_rule": {
                "ladder": ladder,
                "timed_epochs_per_point": epochs,
                "isolation": "one fresh process per (engine, precision, batch) point",
                "stop_rule": "first oom/error/timeout ends that engine's ascent",
                "quality_rule": "not gated (saturation benchmark; see quality/convergence)",
            },
            "engine_paths": family.engine_paths(),
        },
        "configuration": {
            "family": args.family,
            "engines": engines,
            "precisions": precisions,
            "epochs": epochs,
            "start_batch": start_batch,
            "max_batch": cap,
            "timeout_s": args.timeout_s,
        },
        "machine": versions(),
        "results": {},
    }
    result["git_commit"] = result["git"]["commit"]

    for precision in precisions:
        per_precision: dict[str, Any] = {}
        for engine in engines:
            print(f"\n=== {args.family} {precision} {engine} ===")
            curve: list[dict[str, Any]] = []
            peak_sps = 0.0
            peak_batch = 0
            frontier: dict[str, Any] | None = None
            for batch in ladder:
                cmd, env_over = family.cmd(engine, precision, batch, epochs)
                fields, status, raw = run_point(cmd, env_over, args.timeout_s, family.cwd)
                if status == "ok":
                    sps = float(fields["samples_per_sec"])
                    point = {"batch": batch, "status": status, "samples_per_sec": sps}
                    if "tokens_per_sec" in fields:
                        point["tokens_per_sec"] = float(fields["tokens_per_sec"])
                    curve.append(point)
                    if sps > peak_sps:
                        peak_sps, peak_batch = sps, batch
                    print(f"  batch {batch:>10,}: {sps:>14,.0f} samples/s")
                else:
                    frontier = {"batch": batch, "status": status,
                                "raw_tail": raw[-1500:]}
                    curve.append({"batch": batch, "status": status})
                    print(f"  batch {batch:>10,}: {status.upper()} — ascent ends")
                    break
            entry: dict[str, Any] = {"curve": curve}
            if peak_batch:
                entry["peak_samples_per_sec"] = peak_sps
                entry["peak_batch"] = peak_batch
                entry["max_ok_batch"] = max(p["batch"] for p in curve if p["status"] == "ok")
            if frontier:
                entry["frontier"] = frontier
            per_precision[engine] = entry
        base = per_precision.get("opennn", {}).get("peak_samples_per_sec")
        if base:
            for engine in ("pytorch", "tensorflow"):
                other = per_precision.get(engine, {}).get("peak_samples_per_sec")
                if other:
                    per_precision[f"opennn_vs_{engine}"] = round(base / other, 3)
        result["results"][precision] = per_precision
        out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
