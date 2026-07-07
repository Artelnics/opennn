#!/usr/bin/env python3
"""Max batch (training AND inference) for the HIGGS dense classifier, GPU.

OpenNN vs PyTorch vs TensorFlow, fp32 and bf16. Per framework, per precision,
per mode, the largest batch that completes within the physical-VRAM cap:

    train  - one full-batch training step (forward + backward + Adam update)
    infer  - one forward pass, no gradients / optimizer state (OpenNN
             device-resident path, PyTorch inference_mode, TF training=False)

Model: the canonical HIGGS dense contract (28 -> hidden -> hidden -> 1, ReLU
hidden, binary cross-entropy, Adam; docs/benchmarks/throughput/higgs/README.md), built
identically by every engine. Data is synthetic with the contract shapes --
capacity depends on the shapes and the step, not the feature values.

Every batch candidate runs in a fresh process (OOM-safe, allocator-state
safe). The search is exponential growth then binary search, recording the
largest passing batch. OpenNN runs with prefetch-pool depth 1 and CUDA graph
off (set in the trial binary): this is a capacity benchmark.

--device cpu runs the same matrix CPU-only (fp32; bf16 is skipped): each
trial process is capped with a hard RLIMIT_DATA limit (--mem-cap-gib,
default 8 -- the same budget as the published data-capacity benchmark),
which makes the out-of-memory boundary deterministic. RLIMIT_DATA counts
brk + anonymous mmap (the tensor/data allocations) but NOT file-backed
library mappings -- PyTorch/TF map several GiB of runtime libraries, so an
address-space cap (RLIMIT_AS) would charge them for code, not data, and the
Windows data-capacity benchmark's Job Object cap is committed memory, which
RLIMIT_DATA approximates far better. CPU mode requires Linux (kernel >= 4.7
for mmap accounting in RLIMIT_DATA); on Windows use a Job Object wrapper as
in docs/benchmarks/capacity/data-capacity/.

A JSON artifact is written to docs/benchmarks/results/ by default
(--result-json to override, --no-result-json to skip).
"""

import argparse, json, os, re, subprocess, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
VENV_PY = os.environ.get("BENCH_PYTHON", "/home/artelnics/.venvs/ml/bin/python")
DEFAULT_BIN = os.path.join(REPO, "build", "bin", "opennn_higgs_maxbatch_trial")
RESULTS_DIR = os.path.join(REPO, "docs", "benchmarks", "results")


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


def nvidia_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True)
    return int(out.stdout.strip().splitlines()[0])


class PeakMonitor:
    def __init__(self, interval=0.05):
        self.interval, self.peak = interval, 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
    def __enter__(self): self._t.start(); return self
    def __exit__(self, *a): self._stop.set(); self._t.join(timeout=2)
    def _run(self):
        while not self._stop.is_set():
            try: self.peak = max(self.peak, nvidia_used_mib())
            except Exception: pass
            self._stop.wait(self.interval)


def cmd_env(engine, precision, mode, batch):
    on_cpu = args.device == "cpu"
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "" if on_cpu else "0"
    if args.higgs_bin:
        env["OPENNN_HIGGS_BIN"] = args.higgs_bin   # opennn trial
        env["HIGGS_BIN"] = args.higgs_bin          # pytorch / tensorflow trials
    else:
        env.pop("OPENNN_HIGGS_BIN", None)
        env.pop("HIGGS_BIN", None)
    if engine == "opennn":
        cmd = [args.opennn_bin, mode, str(batch),
               str(args.hidden), str(args.layers), "1", args.device]
        if args.tile is not None:
            cmd.append(str(args.tile))
        if precision == "bf16" and not on_cpu: env["OPENNN_BF16"] = "1"
        else: env.pop("OPENNN_BF16", None)
    elif engine == "pytorch":
        cmd = [VENV_PY, os.path.join(HERE, "pytorch_higgs_maxbatch.py"),
               "--mode", mode, "--batch", str(batch),
               "--hidden", str(args.hidden), "--layers", str(args.layers),
               "--steps", "1", "--warmup", "1", "--device", args.device]
        if precision == "bf16" and not on_cpu: env["PT_BF16"] = "1"
        else: env.pop("PT_BF16", None)
    elif engine == "tensorflow":
        cmd = [VENV_PY, os.path.join(HERE, "tensorflow_higgs_maxbatch.py"),
               "--mode", mode, "--batch", str(batch),
               "--hidden", str(args.hidden), "--layers", str(args.layers),
               "--steps", "1", "--warmup", "1", "--device", args.device]
        if precision == "bf16" and not on_cpu: env["TF_BF16"] = "1"
        else: env.pop("TF_BF16", None)
        if TF_LD: env["LD_LIBRARY_PATH"] = TF_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        raise ValueError(engine)
    return cmd, env


def rlimit_preexec(cap_bytes):
    # Hard data cap for the child: brk + anonymous mmap (tensor allocations)
    # past the cap fail (bad_alloc / MemoryError) instead of swapping, so the
    # boundary is deterministic. File-backed library mappings are not charged
    # (see module docstring). Linux only.
    def fn():
        import resource
        resource.setrlimit(resource.RLIMIT_DATA, (cap_bytes, cap_bytes))
    return fn


def run_trial(engine, precision, mode, batch, cap_mib):
    cmd, env = cmd_env(engine, precision, mode, batch)
    on_cpu = args.device == "cpu"
    preexec = rlimit_preexec(int(args.mem_cap_gib * (1 << 30))) if on_cpu else None
    try:
        if on_cpu:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=args.timeout_s, preexec_fn=preexec)
            peak = None
        else:
            with PeakMonitor() as mon:
                proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                      timeout=args.timeout_s)
                peak = mon.peak
    except subprocess.TimeoutExpired:
        return {"ok": False, "peak": None, "reason": "timeout"}
    raw = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and "RESULT=OK" in raw
    if ok and peak and peak > cap_mib:
        ok, reason = False, "vram_cap"
    else:
        reason = "ok" if ok else f"exit_{proc.returncode}"
    m = re.search(r"samples_per_sec=([0-9.]+)", raw)
    rss = re.search(r"peak_rss_mib=(\d+)", raw)
    vmp = re.search(r"vm_peak_mib=(\d+)", raw)
    if on_cpu and rss:
        peak = int(rss.group(1))   # CPU mode: peak = process RSS high-water mark
    return {"ok": ok, "peak": peak,
            "vm_peak_mib": int(vmp.group(1)) if vmp else None,
            "sps": float(m.group(1)) if m else None,
            "reason": reason, "raw": raw[-1500:]}


def cooldown(threshold=1200, timeout=30):
    if args.device == "cpu":
        return
    start = time.time()
    while time.time() - start < timeout:
        try:
            if nvidia_used_mib() <= threshold: return
        except Exception: return
        time.sleep(0.5)


def search_max_batch(engine, precision, mode, cap_mib):
    cache = {}
    def trial(b):
        if b not in cache:
            cache[b] = run_trial(engine, precision, mode, b, cap_mib)
            r = cache[b]
            print(f"  {engine:11s} {precision} {mode:5s} batch={b:<9d} "
                  f"{'OK ' if r['ok'] else 'FAIL'} peak={r['peak']} MiB reason={r['reason']}")
            cooldown()
        return cache[b]["ok"]
    lo, hi = 0, args.start_batch
    while hi <= args.max_limit and trial(hi):
        lo = hi; hi *= 2
    left, right = lo + 1, min(hi - 1, args.max_limit)
    while left <= right:
        # --min-step trades boundary precision for search time; useful when
        # single trials take minutes (e.g. tiled CPU runs at 10^7+ samples).
        if right - left + 1 < args.min_step: break
        mid = (left + right) // 2
        if trial(mid): lo, left = mid, mid + 1
        else: right = mid - 1
    fail = lo + 1 if lo + 1 in cache and not cache[lo + 1]["ok"] else None
    best = cache.get(lo, {})
    return lo, best.get("peak"), fail, best.get("vm_peak_mib")


def main():
    global args
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--precisions", default="fp32,bf16")
    ap.add_argument("--modes", default="train,infer",
                    help="comma list of train,infer")
    ap.add_argument("--opennn-bin", default=DEFAULT_BIN)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--start-batch", type=int, default=65536)
    ap.add_argument("--max-limit", type=int, default=1 << 26)
    ap.add_argument("--min-step", type=int, default=1,
                    help="stop the binary search when the bracket is narrower "
                         "than this (coarser boundary, fewer long trials)")
    ap.add_argument("--tile", type=int, default=None,
                    help="opennn infer tile rows (0 = untiled protocol; "
                         "default: the trial's built-in tile)")
    ap.add_argument("--higgs-bin", default=None,
                    help="prepared HIGGS float32 binary (rows x 29, features "
                         "then label; see README): trials use these real rows, "
                         "repeated modulo beyond the file, instead of "
                         "synthetic data")
    ap.add_argument("--reserve-mib", type=int, default=512)
    ap.add_argument("--mem-cap-gib", type=float, default=8.0,
                    help="CPU mode: hard RLIMIT_DATA cap per trial process")
    ap.add_argument("--timeout-s", type=int, default=600)
    ap.add_argument("--result-json", default=None)
    ap.add_argument("--no-result-json", action="store_true")
    args = ap.parse_args()

    if args.device == "cpu":
        if os.name != "posix":
            raise SystemExit("--device cpu needs Linux (RLIMIT_DATA); on Windows "
                             "use a Job Object wrapper as in docs/benchmarks/capacity/data-capacity/.")
        total_mib = None
        cap_mib = int(args.mem_cap_gib * 1024)
        print(f"CPU mode, RLIMIT_DATA cap={args.mem_cap_gib} GiB per trial, "
              f"model=28->{args.hidden}x{args.layers}->1")
    else:
        total_mib = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
                        .stdout.strip().splitlines()[0])
        cap_mib = total_mib - args.reserve_mib
        print(f"GPU total={total_mib} MiB, cap={cap_mib} MiB, "
              f"model=28->{args.hidden}x{args.layers}->1")

    engines = args.engines.split(",")
    precisions = args.precisions.split(",")
    modes = args.modes.split(",")
    if args.device == "cpu" and "bf16" in precisions:
        print("CPU mode: dropping bf16 (fp32 only)")
        precisions = [p for p in precisions if p != "bf16"]

    results = {}
    for e in engines:
        for p in precisions:
            for m in modes:
                print(f"\n[max-batch] {e} {p} {m}")
                mb, peak, fail, vm_peak = search_max_batch(e, p, m, cap_mib)
                results[(e, p, m)] = {"max_batch": mb, "peak_at_max": peak,
                                      "next_batch_failed": fail,
                                      "vm_peak_at_max_mib": vm_peak}

    print("\n===================== SUMMARY =====================")
    print(f"{'engine':12s} {'prec':5s} {'mode':6s} {'max_batch':>12s} {'peak@max(MiB)':>14s}")
    for e in engines:
        for p in precisions:
            for m in modes:
                r = results[(e, p, m)]
                print(f"{e:12s} {p:5s} {m:6s} {r['max_batch']:>12d} "
                      f"{str(r['peak_at_max']):>14s}")

    if not args.no_result_json:
        stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        path = args.result_json or os.path.join(
            RESULTS_DIR, f"{'cpu' if args.device == 'cpu' else 'gpu'}-higgs-max-batch-{stamp}.json")
        artifact = {
            "benchmark": "higgs-max-batch",
            "device": args.device,
            "model": {"inputs": 28, "hidden": args.hidden,
                      "hidden_layers": args.layers, "outputs": 1,
                      "activation": "relu", "loss": "binary_cross_entropy",
                      "optimizer": "adam"},
            "gpu_total_mib": total_mib,
            "memory_cap_mib": cap_mib,
            "search_min_step": args.min_step,
            "data": (f"higgs_bin:{args.higgs_bin} (rows repeat modulo beyond the file)"
                     if args.higgs_bin else "synthetic contract-shaped"),
            "protocol": "fresh process per candidate; exponential + binary "
                        "search; largest batch completing one step under the "
                        "memory cap (GPU: physical VRAM minus reserve; CPU: "
                        "hard RLIMIT_DATA on brk+anonymous mmap); OpenNN "
                        "pool=1, CUDA graph off; synthetic contract-shaped data",
            "results": [{"engine": e, "precision": p, "mode": m, **results[(e, p, m)]}
                        for e in engines for p in precisions for m in modes],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(artifact, f, indent=1)
        print(f"\nresult JSON written to {path}")


if __name__ == "__main__":
    main()
