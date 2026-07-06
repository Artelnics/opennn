#!/usr/bin/env python3
"""GPU energy-to-target benchmark: chat Transformer, OpenNN vs PyTorch vs TensorFlow.

Every engine trains the IDENTICAL model (encoder-decoder Transformer, paper base
512/8/2048/6 by default) on the IDENTICAL token ids (OpenNN's tokens.bin cache)
with the IDENTICAL convergence hyperparameters (batch, plain Adam, lr, shuffle,
no dropout/regularization/clipping) until the IDENTICAL gate: epoch-mean token
cross-entropy over non-PAD targets <= --target. What differs is the engine and
its fastest execution path (OpenNN bf16 + CUDA graph, PyTorch bf16 autocast +
fused Adam + SDPA, TensorFlow mixed_bfloat16 + XLA), so energy-to-target is an
engine comparison, not a hyperparameter lottery.

For every run we:
  * sample GPU power.draw at 20 Hz for the whole process,
  * integrate (trapezoid) ONLY between the TRAIN_START_UNIX / TRAIN_END_UNIX
    markers each engine prints around its training loop -- compile/warmup work
    inside the loop counts, one-time corpus tokenization does not,
  * report total energy and active energy (idle baseline subtracted), plus
    wall time, epochs to target, and the per-epoch loss history.

Results (per-run + aggregate, versions, git commit) go to ../results/ as
immutable JSON per results/README.md. GPU-energy only (board sensor; sampled
power, not a HW joule counter). Run on a quiet GPU.

  usage: run_transformer_energy.py --target T [--batch N] [--lr F] [--runs N]
                                   [--engines opennn,pytorch,tensorflow]
                                   [--precision bf16|fp32] [--max-epochs N]
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
RESULTS_DIR = os.path.normpath(os.path.join(HERE, "..", "results"))
VENV_PY = os.environ.get("BENCH_PYTHON", "/home/artelnics/.venvs/ml/bin/python")
CORPUS = os.environ.get("CHAT_CORPUS",
                        "/home/artelnics/Documents/datasets/chat/chat_pairs.txt")
DEFAULT_BIN = os.path.join(REPO, "build", "bin", "opennn_transformer_energy")
D, H, FF, LAYERS = 512, 8, 2048, 6


# TF needs the venv's bundled CUDA libs on LD_LIBRARY_PATH to see the GPU.
def tf_ld_path():
    site = os.path.join(os.path.dirname(os.path.dirname(VENV_PY)),
                        "lib", "python3.12", "site-packages", "nvidia")
    libs = []
    if os.path.isdir(site):
        for d in sorted(os.listdir(site)):
            p = os.path.join(site, d, "lib")
            if os.path.isdir(p):
                libs.append(p)
    return os.pathsep.join(libs)


TF_LD = tf_ld_path()


def cmd_env(engine, shape, seed):
    tokens_bin = CORPUS + ".cache/tokens.bin"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    common = ["--tokens-bin", tokens_bin,
              "--in-seq", str(shape["input_seq"]), "--dec-seq", str(shape["decoder_seq"]),
              "--in-vocab", str(shape["input_vocab"]), "--out-vocab", str(shape["output_vocab"]),
              "--target", str(args.target), "--batch", str(args.batch),
              "--max-epochs", str(args.max_epochs), "--lr", str(args.lr),
              "--d", str(D), "--h", str(H), "--ff", str(FF), "--layers", str(LAYERS),
              "--seed", str(seed)]
    if engine == "opennn":
        cmd = [args.opennn_bin, CORPUS, str(args.target), str(args.batch),
               str(args.max_epochs), str(args.lr), str(D), str(H), str(FF), str(LAYERS),
               str(seed)]
        if args.precision == "bf16":
            env["OPENNN_BF16"] = "1"
        else:
            env.pop("OPENNN_BF16", None)
        env["OPENNN_GRAPH"] = "0" if args.no_graph else "1"
    elif engine == "pytorch":
        cmd = [VENV_PY, os.path.join(HERE, "pytorch_transformer_energy.py")] + common
        if args.precision == "bf16":
            env["PT_BF16"] = "1"
        else:
            env.pop("PT_BF16", None)
    elif engine == "tensorflow":
        cmd = [VENV_PY, os.path.join(HERE, "tensorflow_transformer_energy.py")] + common
        if args.precision == "bf16":
            env["TF_BF16"] = "1"
        else:
            env.pop("TF_BF16", None)
        env["TF_XLA"] = "0" if args.no_xla else "1"
        if TF_LD:
            env["LD_LIBRARY_PATH"] = TF_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        raise ValueError(engine)
    return cmd, env


def parse_power_csv(path):
    """(seconds-of-day, watts) samples from an nvidia-smi timestamp,power log,
    unwrapped past midnight so time is monotonic."""
    samples = []
    offset = 0.0
    prev = None
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) < 2:
                continue
            try:
                hms = parts[0].split(" ")[1].split(":")
                sec = int(hms[0]) * 3600 + int(hms[1]) * 60 + float(hms[2])
                w = float(parts[1])
            except (IndexError, ValueError):
                continue
            if prev is not None and sec < prev - 1:
                offset += 86400
            prev = sec
            samples.append((sec + offset, w))
    return samples


def unix_to_trace_time(unix_ts, samples):
    """Map a unix timestamp onto the trace's (unwrapped) seconds-of-day axis."""
    dt = datetime.fromtimestamp(unix_ts)
    sod = dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6
    if samples and sod < samples[0][0] - 43200:
        sod += 86400
    return sod


def integrate(samples, idle_w, t_lo=None, t_hi=None):
    """Trapezoidal ∫power dt, optionally restricted to [t_lo, t_hi];
    returns (total_J, active_J, avg_w, span_s, n_samples)."""
    if t_lo is not None:
        samples = [(t, w) for t, w in samples if t_lo <= t <= t_hi]
    e_total = e_active = span = sumw = 0.0
    prev = None
    for t, w in samples:
        sumw += w
        if prev is not None:
            pt, pw = prev
            dt = t - pt
            if 0 < dt < 2:  # ignore logger startup/teardown gaps
                e_total += 0.5 * (w + pw) * dt
                e_active += 0.5 * ((w - idle_w) + (pw - idle_w)) * dt
                span += dt
        prev = (t, w)
    avg_w = sumw / len(samples) if samples else 0.0
    return e_total, e_active, avg_w, span, len(samples)


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
    # anchored to line start so e.g. "epochs=" cannot match inside "max_epochs="
    m = re.search(pattern, text, re.MULTILINE)
    return cast(m.group(1)) if m else None


def run_one(engine, shape, idle_w, trace_path, seed):
    cmd, env = cmd_env(engine, shape, seed)
    logf = open(trace_path, "w")
    logger = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=timestamp,power.draw",
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

    samples = parse_power_csv(trace_path)
    train_start = parse_marker(r"TRAIN_START_UNIX=([0-9.]+)", out)
    train_end = parse_marker(r"TRAIN_END_UNIX=([0-9.]+)", out)

    m = {
        "rc": rc,
        "process_wall_s": round(proc_wall, 3),
        "power_samples": len(samples),
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "epochs": parse_marker(r"^epochs=(\d+)", out, int),
        "final_error": parse_marker(r"^final_error=([0-9.eE+-]+)", out),
        "reached_goal": parse_marker(r"^reached_goal=(\d)", out, int),
        "wall_s": parse_marker(r"^wall_s=([0-9.]+)", out),
        "samples_per_sec": parse_marker(r"^samples_per_sec=([0-9.]+)", out),
    }
    lh = re.search(r"^loss_history=([0-9.,eE+-]+)", out, re.MULTILINE)
    m["loss_history"] = [round(float(v), 5) for v in lh.group(1).split(",")] if lh else None
    if m["loss_history"]:
        m["epochs"] = len(m["loss_history"]) - 1

    if train_start and train_end and samples:
        t_lo = unix_to_trace_time(train_start, samples)
        t_hi = unix_to_trace_time(train_end, samples)
        e_total, e_active, avg_w, span, n = integrate(samples, idle_w, t_lo, t_hi)
        m.update({
            "train_window_s": round(train_end - train_start, 3),
            "window_power_samples": n,
            "avg_power_w": round(avg_w, 2),
            "active_power_w": round(avg_w - idle_w, 2),
            "energy_total_j": round(e_total, 1),
            "energy_active_j": round(e_active, 1),
            "energy_total_wh": round(e_total / 3600, 3),
            "energy_active_wh": round(e_active / 3600, 3),
        })
    et, ea, aw, _, _ = integrate(samples, idle_w)
    m["process_energy_total_j"] = round(et, 1)
    m["process_energy_active_j"] = round(ea, 1)

    ok = (m["reached_goal"] == 1 and "RESULT=OK" in out
          and m.get("energy_total_j") is not None)
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


def derive_shape():
    cmd = [args.opennn_bin, CORPUS, "probe"]
    # probe mode never trains, so the seed is irrelevant here
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900).stdout
    if "RESULT=OK" not in out:
        raise RuntimeError(f"OpenNN probe failed:\n{out[-2000:]}")

    def g(k):
        return int(re.search(rf"{k}=(\d+)", out).group(1))

    shape = {k: g(k) for k in
             ("samples", "input_vocab", "output_vocab", "input_seq", "decoder_seq")}
    print(f"model shape: {shape}")
    return shape


def git_commit():
    try:
        r = subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"],
                           capture_output=True, text=True)
        return r.stdout.strip()[:12] or "unknown"
    except Exception:
        return "unknown"


def versions():
    v = {"python": sys.version.split()[0]}
    for mod in ("torch", "tensorflow"):
        try:
            r = subprocess.run([VENV_PY, "-c",
                                f"import {mod}; print({mod}.__version__)"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                v[mod] = r.stdout.strip()
        except Exception:
            pass
    try:
        v["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True).stdout.strip()
    except Exception:
        pass
    return v


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True,
                    help="epoch-mean token CE gate (same for every engine)")
    ap.add_argument("--batch", type=int, default=128)
    # lr 1e-4: 5e-4 (the ChatGPT example default) parks all three engines on the
    # unigram plateau (~6.77) at batch 128; 1e-4 descends steadily (calibrated).
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=42,
                    help="run r uses seed seed_base + r (per-seed runs, MLPerf-style)")
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--opennn-bin", default=DEFAULT_BIN)
    ap.add_argument("--no-graph", action="store_true", help="disable OpenNN CUDA graph")
    ap.add_argument("--no-xla", action="store_true", help="disable TensorFlow XLA")
    ap.add_argument("--timeout-s", type=int, default=7200)
    ap.add_argument("--idle", type=float, default=None, help="override idle W (else measured)")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    engines = args.engines.split(",")

    idle_w = args.idle if args.idle is not None else measure_idle()
    print(f"idle_baseline_W={idle_w:.2f}")

    shape = derive_shape()

    result = {
        "schema_version": 1,
        "benchmark_id": "gpu-transformer-energy-to-target",
        "run_id": run_id,
        "git_commit": git_commit(),
        "configuration": {
            "task": "chat Transformer (Stanford Alpaca pairs) trained to a fixed "
                    "epoch-mean token cross-entropy",
            "model": f"encoder-decoder Transformer d{D}/h{H}/ff{FF}/{LAYERS}L",
            "shape": shape,
            "target_epoch_mean_token_ce": args.target,
            "batch": args.batch,
            "lr": args.lr,
            "precision": args.precision,
            "max_epochs": args.max_epochs,
            "runs": args.runs,
            "opennn_cuda_graph": not args.no_graph,
            "tensorflow_xla": not args.no_xla,
            "idle_baseline_w": round(idle_w, 2),
            "power_source": "nvidia-smi power.draw 20Hz, trapezoidal integration "
                            "over the TRAIN_START..TRAIN_END window",
            "note": "GPU energy only (board sensor); sampled power, not a HW joule "
                    "counter; identical data/model/hyperparameters, per-engine "
                    "fastest execution path",
        },
        "machine": versions(),
        "results": {},
    }

    for eng in engines:
        print(f"\n=== {eng} ({args.runs} runs, target {args.target}) ===")
        per_run, fails = [], []
        for r in range(args.runs):
            cooldown(idle_w)
            trace = os.path.join(RESULTS_DIR, f".trace-energy-{eng}-{run_id}-{r}.csv")
            seed = args.seed_base + r
            ok, m, out = run_one(eng, shape, idle_w, trace, seed)
            m["seed"] = seed
            os.remove(trace)
            if not ok:
                print(f"  run {r}: FAILED rc={m['rc']} reached={m['reached_goal']} "
                      f"epochs={m['epochs']}")
                fails.append({"metrics": m, "tail": out[-1500:]})
                continue
            per_run.append(m)
            print(f"  run {r}: {m['energy_total_wh']:.1f} Wh total "
                  f"({m['energy_active_wh']:.1f} active) in {m['train_window_s']:.0f}s, "
                  f"{m['epochs']} epochs, avg {m['avg_power_w']:.0f} W, "
                  f"final CE {m['final_error']:.4f}")

        agg = {"n_ok": len(per_run), "per_run": per_run, "failed": fails}
        if per_run:
            for key in ("energy_total_wh", "energy_active_wh", "train_window_s",
                        "avg_power_w", "epochs"):
                vals = [m[key] for m in per_run]
                agg[f"{key}_median"] = round(statistics.median(vals), 3)
                agg[f"{key}_stdev"] = (round(statistics.pstdev(vals), 3)
                                       if len(vals) > 1 else 0.0)
            print(f"  => median {agg['energy_total_wh_median']} Wh total, "
                  f"{agg['energy_active_wh_median']} Wh active, "
                  f"{agg['train_window_s_median']:.0f}s, "
                  f"{agg['epochs_median']:.0f} epochs")
        result["results"][eng] = agg

    base = result["results"].get("opennn", {})
    if base.get("n_ok"):
        for eng, r in result["results"].items():
            if eng == "opennn" or not r.get("n_ok"):
                continue
            r["energy_ratio_vs_opennn_total"] = round(
                r["energy_total_wh_median"] / base["energy_total_wh_median"], 3)
            r["energy_ratio_vs_opennn_active"] = round(
                r["energy_active_wh_median"] / base["energy_active_wh_median"], 3)
            r["time_ratio_vs_opennn"] = round(
                r["train_window_s_median"] / base["train_window_s_median"], 3)

    out_path = os.path.join(RESULTS_DIR,
                            f"gpu-transformer-energy-to-target-{run_id}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
