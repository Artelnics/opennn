#!/usr/bin/env python3
"""GPU fixed-work energy benchmark: ResNet-50 on CIFAR-10, OpenNN vs PyTorch vs TensorFlow.

Every engine trains the IDENTICAL ResNet-50 v1.5 at CIFAR geometry for the
IDENTICAL fixed work (same epochs and batch), using the engine drivers of the
ResNet speed track (../../throughput/resnet50). What differs is the engine and
its fastest execution path (OpenNN cuDNN + CUDA graph, PyTorch channels_last +
torch.compile, TensorFlow XLA train step), so energy-per-sample is an engine
comparison at equal work.

For every run we:
  * sample GPU power.draw AND clocks.current.sm at 20 Hz for the whole process,
  * integrate (trapezoid) ONLY between the TRAIN_START_UNIX / TRAIN_END_UNIX
    markers each engine prints around its timed training loop -- the 2 warmup
    epochs and one-time data loading stay outside the window,
  * report total and active energy (idle baseline subtracted), microjoules per
    nominal epoch-sample (energy / (samples x epochs), same divisor for every
    engine), average power, median SM clock, wall time, and samples/sec.

Results (per-run + aggregate, versions, git commit) go to ../../results/ as
immutable JSON per results/README.md. GPU-energy only (board sensor; sampled
power, not a HW joule counter). Run on a quiet GPU.

  usage: run_resnet50_energy.py [--data-dir DIR] [--epochs N] [--batch N]
                                [--precision fp32|bf16|both] [--runs N]
                                [--engines opennn,pytorch,tensorflow]
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRIVERS = (HERE.parent.parent / "throughput" / "resnet50").resolve()
RESULTS_DIR = (HERE.parent.parent / "results").resolve()
BENCH_DATA = Path(os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data")))
PY = os.environ.get("BENCH_PYTHON", sys.executable)

def repo_root() -> Path:
    try:
        r = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            return Path(r.stdout.strip()).resolve()
    except Exception:
        pass
    return HERE.parents[2]

REPO_ROOT = repo_root()

def find_opennn_bin() -> tuple[str, bool]:
    for env_name in ("OPENNN_RESNET50_SPEED_BIN", "OPENNN_BIN"):
        override = os.environ.get(env_name)
        if override:
            return override, Path(override).exists()
    names = (["opennn_resnet50_speed.exe", "opennn_resnet50_speed"] if os.name == "nt"
             else ["opennn_resnet50_speed", "opennn_resnet50_speed.exe"])
    dirs = [
        DRIVERS,
        REPO_ROOT / "build" / "bin",
        REPO_ROOT / "build" / "bin" / "Release",
        REPO_ROOT / "build-benchmarks" / "bin",
        REPO_ROOT / "build-benchmarks" / "bin" / "Release",
    ]
    for directory in dirs:
        for name in names:
            candidate = directory / name
            if candidate.exists():
                return str(candidate), True
    return str(REPO_ROOT / "build" / "bin" / names[0]), False

OPENNN_BIN, OPENNN_BIN_FOUND = find_opennn_bin()

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
        lines = [l for l in out.stdout.splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else []
    except Exception:
        return []

def cmd_env(engine: str, precision: str):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    bf16 = precision == "bf16"

    if engine == "opennn":
        cmd = [OPENNN_BIN, str(Path(args.data_dir) / "train"),
               str(args.epochs), str(args.batch), precision]
    elif engine == "pytorch":
        cmd = [PY, str(DRIVERS / "pytorch_resnet50_speed.py"),
               str(args.epochs), str(args.batch), args.data_dir]
        env["PT_FAST"] = "1"
        if bf16:
            env["PT_BF16"] = "1"
        else:
            env.pop("PT_BF16", None)
    elif engine == "tensorflow":
        cmd = [PY, str(DRIVERS / "tensorflow_resnet50_speed.py"),
               str(args.epochs), str(args.batch), args.data_dir]
        if bf16:
            env["TF_BF16"] = "1"
        else:
            env.pop("TF_BF16", None)
        libs = tensorflow_library_dirs(PY)
        if libs:
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                libs + [env.get("LD_LIBRARY_PATH", "")])
    else:
        raise ValueError(engine)
    return cmd, env

def parse_trace_csv(path):
    """(seconds-of-day, watts, sm_mhz) samples from an nvidia-smi
    timestamp,power.draw,clocks.sm log, unwrapped past midnight."""
    samples = []
    offset = 0.0
    prev = None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 3:
                continue
            try:
                hms = parts[0].split(" ")[1].split(":")
                sec = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
                w = float(parts[1])
                clk = float(parts[2])
            except (IndexError, ValueError):
                continue
            if prev is not None and sec < prev - 1:
                offset += 86400
            prev = sec
            samples.append((sec + offset, w, clk))
    return samples

def unix_to_trace_time(unix_ts, samples):
    dt = datetime.fromtimestamp(unix_ts)
    sod = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    if samples and sod < samples[0][0] - 43200:
        sod += 86400
    return sod

def integrate(samples, idle_w, t_lo=None, t_hi=None):
    """Trapezoidal integral of power dt over [t_lo, t_hi];
    returns (total_J, active_J, avg_w, clk_median, span_s, n_samples)."""
    if t_lo is not None:
        samples = [s for s in samples if t_lo <= s[0] <= t_hi]
    e_total = e_active = span = sumw = 0.0
    clocks = []
    prev = None
    for t, w, clk in samples:
        sumw += w
        clocks.append(clk)
        if prev is not None:
            pt, pw = prev
            dt = t - pt
            if 0 < dt < 2:
                e_total += 0.5 * (w + pw) * dt
                e_active += 0.5 * ((w - idle_w) + (pw - idle_w)) * dt
                span += dt
        prev = (t, w)
    avg_w = sumw / len(samples) if samples else 0.0
    clk_median = statistics.median(clocks) if clocks else None
    return e_total, e_active, avg_w, clk_median, span, len(samples)

def measure_idle(seconds=5.0):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
             "-lms", "100"],
            capture_output=True, text=True, timeout=seconds).stdout
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
    vals = [float(x) for x in out.split() if re.fullmatch(r"[0-9.]+", x)]
    return sum(vals) / len(vals) if vals else 30.0

def gpu_state():
    fields = ("clocks.current.sm,clocks.max.sm,clocks.current.memory,"
              "temperature.gpu,power.limit,power.draw,clocks_throttle_reasons.active")
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().split("\n")[0]
        keys = ["sm_clock_mhz", "sm_clock_max_mhz", "mem_clock_mhz",
                "temp_c", "power_limit_w", "power_draw_w", "throttle_reasons"]
        state = {}
        for k, v in zip(keys, [v.strip() for v in out.split(",")]):
            try:
                state[k] = float(v)
            except ValueError:
                state[k] = v
        return state
    except Exception:
        return {"error": "nvidia-smi gpu-state query failed"}

def parse_marker(pattern, text, cast=float):
    m = re.search(pattern, text, re.MULTILINE)
    return cast(m.group(1)) if m else None

def run_one(engine, precision, idle_w, trace_path):
    cmd, env = cmd_env(engine, precision)
    logf = open(trace_path, "w")
    logger = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw,clocks.current.sm",
         "--format=csv,noheader,nounits", "-lms", "50"],
        stdout=logf, stderr=subprocess.DEVNULL)
    time.sleep(0.3)
    state_before = gpu_state()
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=args.timeout_s)
        out = proc.stdout + proc.stderr
        rc = proc.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or "") + (e.stderr or "")
        rc = "timeout"
    proc_wall = time.perf_counter() - t0
    state_after = gpu_state()
    time.sleep(0.3)
    logger.terminate()
    logger.wait()
    logf.close()

    samples = parse_trace_csv(trace_path)
    train_start = parse_marker(r"^TRAIN_START_UNIX=([0-9.]+)", out)
    train_end = parse_marker(r"^TRAIN_END_UNIX=([0-9.]+)", out)

    m = {
        "rc": rc,
        "process_wall_s": round(proc_wall, 3),
        "power_samples": len(samples),
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "train_samples": parse_marker(r"^samples=(\d+)", out, int),
        "samples_per_sec": parse_marker(r"^samples_per_sec=([0-9.]+)", out),
        "epoch_s": parse_marker(r"^epoch_s=([0-9.eE+-]+)", out),
    }

    if train_start and train_end and samples:
        t_lo = unix_to_trace_time(train_start, samples)
        t_hi = unix_to_trace_time(train_end, samples)
        e_total, e_active, avg_w, clk_median, span, n = integrate(samples, idle_w, t_lo, t_hi)
        m.update({
            "train_window_s": round(train_end - train_start, 3),
            "window_power_samples": n,
            "avg_power_w": round(avg_w, 2),
            "active_power_w": round(avg_w - idle_w, 2),
            "sm_clock_median_mhz": clk_median,
            "energy_total_j": round(e_total, 1),
            "energy_active_j": round(e_active, 1),
        })
        if m["train_samples"]:
            nominal = m["train_samples"] * args.epochs
            m["uj_per_sample_total"] = round(e_total * 1e6 / nominal, 4)
            m["uj_per_sample_active"] = round(e_active * 1e6 / nominal, 4)
    et, ea, _, _, _, _ = integrate(samples, idle_w)
    m["process_energy_total_j"] = round(et, 1)
    m["process_energy_active_j"] = round(ea, 1)

    sparse_trace = (m.get("train_window_s") or 0) > 0 and (
        m.get("window_power_samples", 0) / m["train_window_s"] < 5.0)
    if sparse_trace:
        m["rejected"] = "power trace below 5 Hz across train window"

    ok = ("RESULT=OK" in out and m.get("energy_total_j") is not None
          and not sparse_trace)
    return ok, m, out

def cooldown(idle_w, seconds=20, mib_threshold=1200):
    """Wait for VRAM to drain and power to settle back near idle."""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True).stdout.strip().splitlines()[0]
            mib, watts = [float(x) for x in q.split(",")]
            if mib <= mib_threshold and watts <= idle_w + 12:
                return
        except Exception:
            return
        time.sleep(1.0)

def git_metadata():
    def g(*a):
        try:
            r = subprocess.run(["git", "-C", str(REPO_ROOT), *a],
                               capture_output=True, text=True)
            return r.stdout.strip()
        except Exception:
            return ""
    status = g("status", "--short")
    return {"commit": g("rev-parse", "HEAD") or "unknown",
            "branch": g("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "dirty": bool(status),
            "status_short": status.splitlines()}

def versions():
    v = {"python": sys.version.split()[0], "bench_python": PY}
    for mod in ("torch", "tensorflow"):
        try:
            r = subprocess.run([PY, "-c", f"import {mod}; print({mod}.__version__)"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                v[mod] = r.stdout.strip().splitlines()[-1]
        except Exception:
            pass
    try:
        v["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,power.limit",
             "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return v

def main():
    global args
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-dir", default=str(BENCH_DATA / "cifar10"))
    ap.add_argument("--epochs", type=int, default=20,
                    help="fixed work: identical timed epochs for every engine")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--precision", default="both", choices=["fp32", "bf16", "both"])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument("--idle", type=float, default=None, help="override idle W (else measured)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    if not (Path(args.data_dir) / "train").is_dir():
        raise SystemExit(f"CIFAR-10 train folder not found: {args.data_dir}/train "
                         f"(run throughput/resnet50/prepare_cifar10.py)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    precisions = ["fp32", "bf16"] if args.precision == "both" else [args.precision]

    idle_w = args.idle if args.idle is not None else measure_idle()
    print(f"idle_baseline_W={idle_w:.2f}")
    print(f"opennn_binary={OPENNN_BIN} (found={OPENNN_BIN_FOUND})")

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-resnet50-energy",
        "run_id": run_id,
        "git_commit": git_metadata()["commit"],
        "git": git_metadata(),
        "dataset": "CIFAR-10",
        "configuration": {
            "task": "ResNet-50 v1.5 at CIFAR geometry trained for fixed work "
                    "(identical epochs/batch for every engine)",
            "model": "ResNet-50 v1.5, 10-class softmax, cross-entropy, Adam",
            "data_dir": args.data_dir,
            "epochs": args.epochs,
            "batch": args.batch,
            "precisions": precisions,
            "runs": args.runs,
            "engines": engines,
            "idle_baseline_w": round(idle_w, 2),
            "power_source": "nvidia-smi power.draw+clocks.sm 20Hz, trapezoidal "
                            "integration over the TRAIN_START..TRAIN_END window",
            "note": "GPU energy only (board sensor); sampled power, not a HW joule "
                    "counter; uj_per_sample uses nominal samples x epochs, the "
                    "same divisor for every engine; 2 warmup epochs excluded",
        },
        "machine": versions(),
        "runner": {
            "path": os.path.relpath(__file__, REPO_ROOT),
            "argv": sys.argv,
            "opennn_binary": OPENNN_BIN,
        },
        "commands": {},
        "results": {},
    }

    for precision in precisions:
        result["results"][precision] = {}
        for eng in engines:
            print(f"\n=== {eng} {precision} ({args.runs} runs, {args.epochs} epochs) ===")
            cmd, env = cmd_env(eng, precision)
            result["commands"][f"{eng}_{precision}"] = " ".join(cmd)
            per_run, fails = [], []
            for r in range(args.runs):
                cooldown(idle_w)
                trace = RESULTS_DIR / f".trace-resnet50-energy-{eng}-{precision}-{run_id}-{r}.csv"
                ok, m, out = run_one(eng, precision, idle_w, trace)
                trace.unlink(missing_ok=True)
                if not ok:
                    print(f"  run {r}: FAILED rc={m['rc']}")
                    fails.append({"metrics": m, "tail": out[-1500:]})
                    continue
                per_run.append(m)
                print(f"  run {r}: {m['energy_total_j']:.0f} J total "
                      f"({m['energy_active_j']:.0f} active) in {m['train_window_s']:.1f}s, "
                      f"avg {m['avg_power_w']:.0f} W @ {m['sm_clock_median_mhz']:.0f} MHz, "
                      f"{m['uj_per_sample_total']:.1f} uJ/sample")
            agg = {"n_ok": len(per_run), "per_run": per_run, "failed": fails}
            if per_run:
                for key in ("energy_total_j", "energy_active_j", "uj_per_sample_total",
                            "uj_per_sample_active", "train_window_s", "avg_power_w",
                            "active_power_w", "sm_clock_median_mhz", "samples_per_sec",
                            "epoch_s"):
                    vals = [m[key] for m in per_run if isinstance(m.get(key), (int, float))]
                    if vals:
                        agg[f"{key}_median"] = round(statistics.median(vals), 4)
                        agg[f"{key}_stdev"] = (round(statistics.pstdev(vals), 4)
                                               if len(vals) > 1 else 0.0)
                print(f"  => median {agg['energy_total_j_median']:.0f} J total, "
                      f"{agg['uj_per_sample_total_median']:.1f} uJ/sample, "
                      f"{agg['avg_power_w_median']:.0f} W, "
                      f"{agg['train_window_s_median']:.1f}s")
            result["results"][precision][eng] = agg

        base = result["results"][precision].get("opennn", {})
        if base.get("n_ok"):
            for eng, r in result["results"][precision].items():
                if eng == "opennn" or not r.get("n_ok"):
                    continue
                r["energy_ratio_vs_opennn_total"] = round(
                    r["energy_total_j_median"] / base["energy_total_j_median"], 3)
                r["energy_ratio_vs_opennn_active"] = round(
                    r["energy_active_j_median"] / base["energy_active_j_median"], 3)
                r["time_ratio_vs_opennn"] = round(
                    r["train_window_s_median"] / base["train_window_s_median"], 3)

    out_path = RESULTS_DIR / f"gpu-resnet50-energy-{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
