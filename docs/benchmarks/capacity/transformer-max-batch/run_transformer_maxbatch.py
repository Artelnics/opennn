#!/usr/bin/env python3
"""Max batch (training AND inference) + speed for the seq2seq Transformer, GPU.

OpenNN vs PyTorch vs TensorFlow, fp32 and bf16, CUDA graph OFF (does not help
the transformer path). Per framework, per precision, per mode:
    max_batch, speed (samples/s).

Modes:
    train  - forward + backward + Adam step (the original benchmark)
    infer  - forward only, no gradients / optimizer state (OpenNN resident
             path, PyTorch inference_mode, TF training=False)

Same model for all engines: encoder-decoder Transformer d512/h8/ff2048/6L, with
vocab and sequence lengths derived once from the OpenNN corpus so every engine
builds the identical network. Each batch candidate runs in a fresh process
(OOM-safe); max batch = largest that completes warmup+step within the VRAM cap.

The corpus is selected with --corpus (or the CHAT_CORPUS env var). Two
supported corpora:
    - Alpaca chat pairs (chat_pairs.txt) -- the original configuration, kept
      for regression continuity with previously measured numbers.
    - WMT14 En-De pairs prepared by prepare_wmt14.py -- the standard dataset
      of the *Attention Is All You Need* base model; use for publishable runs.
Capacity depends on the corpus only through the derived vocab / sequence
lengths, which the driver prints and shares across all engines.
"""

import argparse, json, os, re, subprocess, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
VENV_PY = os.environ.get("BENCH_PYTHON", "python3")
DEFAULT_CORPUS = os.environ.get(
    "CHAT_CORPUS",
    os.path.join(os.environ.get("OPENNN_BENCH_DATA", os.path.expanduser("~/opennn-benchmark-data")),
                 "chat", "chat_pairs.txt"))
CORPUS = DEFAULT_CORPUS   # overridden by --corpus in main()
DEFAULT_BIN = os.path.join(REPO, "build", "bin", "opennn_transformer_maxbatch_trial")
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


def cmd_env(engine, precision, batch, steps, shape, mode="train"):
    ivoc, ovoc, iseq, dseq, _ = shape
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    if engine == "opennn":
        if mode == "train":
            # steps encoded as train_samples = steps*batch, epochs=0 (one pass).
            cmd = [args.opennn_bin, CORPUS, str(D), str(H), str(FF), str(LAYERS),
                   str(batch), str(steps * batch), "0", "train"]
        else:
            # infer: epochs is reused as the timed forward-iteration count;
            # train_samples only bounds the (unused) training split.
            cmd = [args.opennn_bin, CORPUS, str(D), str(H), str(FF), str(LAYERS),
                   str(batch), str(batch), str(steps), "infer"]
            if args.opennn_infer_cuda_graph:
                env["OPENNN_TRANSFORMER_INFER_CUDA_GRAPH"] = "1"
        if precision == "bf16": env["OPENNN_BF16"] = "1"
        else: env.pop("OPENNN_BF16", None)
    elif engine == "pytorch":
        cmd = [VENV_PY, os.path.join(HERE, "pytorch_transformer_maxbatch.py"),
               "--in-vocab", str(ivoc), "--out-vocab", str(ovoc),
               "--in-seq", str(iseq), "--dec-seq", str(dseq),
               "--d", str(D), "--h", str(H), "--ff", str(FF), "--layers", str(LAYERS),
               "--batch", str(batch), "--steps", str(steps), "--warmup", "3",
               "--mode", mode]
        if precision == "bf16": env["PT_BF16"] = "1"
        else: env.pop("PT_BF16", None)
    elif engine == "tensorflow":
        cmd = [VENV_PY, os.path.join(HERE, "tensorflow_transformer_maxbatch.py"),
               "--in-vocab", str(ivoc), "--out-vocab", str(ovoc),
               "--in-seq", str(iseq), "--dec-seq", str(dseq),
               "--d", str(D), "--h", str(H), "--ff", str(FF), "--layers", str(LAYERS),
               "--batch", str(batch), "--steps", str(steps), "--warmup", "3",
               "--mode", mode]
        if precision == "bf16": env["TF_BF16"] = "1"
        else: env.pop("TF_BF16", None)
        if TF_LD: env["LD_LIBRARY_PATH"] = TF_LD + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    else:
        raise ValueError(engine)
    return cmd, env


def run_trial(engine, precision, batch, steps, shape, cap_mib, mode="train"):
    cmd, env = cmd_env(engine, precision, batch, steps, shape, mode)
    try:
        with PeakMonitor() as mon:
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                                  timeout=args.timeout_s)
            peak = mon.peak
    except subprocess.TimeoutExpired:
        return {"ok": False, "peak": None, "sps": None, "reason": "timeout"}
    raw = proc.stdout + proc.stderr
    ok = proc.returncode == 0 and "RESULT=OK" in raw
    if ok and peak and peak > cap_mib:
        ok, reason = False, "vram_cap"
    else:
        reason = "ok" if ok else f"exit_{proc.returncode}"
    m = re.search(r"samples_per_sec=([0-9.]+)", raw)
    w = re.search(r"wall_s=([0-9.]+)", raw)
    return {"ok": ok, "peak": peak, "sps": float(m.group(1)) if m else None,
            "wall": float(w.group(1)) if w else None,
            "reason": reason, "raw": raw[-1500:]}


def cooldown(threshold=1200, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if nvidia_used_mib() <= threshold: return
        except Exception: return
        time.sleep(0.5)


def search_max_batch(engine, precision, shape, cap_mib, mode="train"):
    cache = {}
    def trial(b):
        if b not in cache:
            cache[b] = run_trial(engine, precision, b, args.probe_steps, shape, cap_mib, mode)
            r = cache[b]
            print(f"  {engine:11s} {precision} {mode:5s} batch={b:<6d} "
                  f"{'OK ' if r['ok'] else 'FAIL'} peak={r['peak']} MiB reason={r['reason']}")
            cooldown()
        return cache[b]["ok"]
    lo, hi = 0, args.start_batch
    while hi <= args.max_limit and trial(hi):
        lo = hi; hi *= 2
    left, right = lo + 1, min(hi - 1, args.max_limit)
    while left <= right:
        mid = (left + right) // 2
        if trial(mid): lo, left = mid, mid + 1
        else: right = mid - 1
    return lo, cache.get(lo, {}).get("peak")


def measure_speed(engine, precision, batch, shape, cap_mib, mode="train"):
    # Steady-state throughput. PyTorch/TF time only their bare step loop, but
    # OpenNN's TRAIN wall_s covers the whole train() call, which includes a
    # fixed warmup (cuDNN plan selection, allocations) and teardown (parameter
    # D2H, frees) inside the timed window. Two-point differencing -- run 1x and
    # 4x the steps and divide the extra samples by the extra wall time --
    # cancels that fixed cost exactly, so all engines report the same quantity.
    # The OpenNN INFER trial times only its resident forward loop (warmup and
    # plan selection excluded in-process), so it needs no differencing.
    long_steps = args.speed_steps * 4
    if engine == "opennn" and mode == "train":
        short = run_trial(engine, precision, batch, args.speed_steps, shape, cap_mib, mode)
        cooldown()
        long = run_trial(engine, precision, batch, long_steps, shape, cap_mib, mode)
        cooldown()
        if short["ok"] and long["ok"] and short["wall"] and long["wall"]:
            return (long_steps - args.speed_steps) * batch / (long["wall"] - short["wall"])
        return long["sps"] if long["ok"] else None
    r = run_trial(engine, precision, batch, long_steps, shape, cap_mib, mode)
    cooldown()
    return r["sps"] if r["ok"] else None


def derive_shape():
    # Run the OpenNN trial tiny to read the model shape (vocab/seq) it builds.
    cmd, env = cmd_env("opennn", "fp32", 8, 1, (0, 0, 0, 0, 0))
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600).stdout
    def g(k): return int(re.search(rf"{k}=(\d+)", out).group(1))
    shape = (g("input_vocab"), g("output_vocab"), g("input_seq"), g("decoder_seq"), g("samples"))
    print(f"model shape: in_vocab={shape[0]} out_vocab={shape[1]} "
          f"in_seq={shape[2]} dec_seq={shape[3]}")
    return shape


def main():
    global args, CORPUS
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", default="opennn,pytorch,tensorflow")
    ap.add_argument("--precisions", default="fp32,bf16")
    ap.add_argument("--modes", default="train,infer",
                    help="comma list of train,infer")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS,
                    help="prompt<TAB>response pair file (Alpaca chat_pairs.txt "
                         "or a WMT14 file from prepare_wmt14.py)")
    ap.add_argument("--opennn-bin", default=DEFAULT_BIN)
    ap.add_argument("--start-batch", type=int, default=16)
    ap.add_argument("--max-limit", type=int, default=65536)
    ap.add_argument("--reserve-mib", type=int, default=512)
    ap.add_argument("--probe-steps", type=int, default=3)
    ap.add_argument("--speed-steps", type=int, default=40)
    ap.add_argument("--speed-batch", type=int, default=32)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--opennn-infer-cuda-graph", action="store_true",
                    help="capture/replay OpenNN resident inference forwards "
                         "(speed mode; keep off for eager like-for-like runs)")
    ap.add_argument("--result-json", default=None,
                    help="optional path for a JSON result artifact "
                         "(convention: docs/benchmarks/results/)")
    args = ap.parse_args()
    CORPUS = args.corpus

    total_mib = int(subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
                    .stdout.strip().splitlines()[0])
    cap_mib = total_mib - args.reserve_mib
    print(f"GPU total={total_mib} MiB, cap={cap_mib} MiB, "
          f"speed_batch={args.speed_batch}, corpus={CORPUS}")

    shape = derive_shape()
    engines = args.engines.split(",")
    precisions = args.precisions.split(",")
    modes = args.modes.split(",")

    results = {}
    for e in engines:
        for p in precisions:
            for m in modes:
                print(f"\n[max-batch] {e} {p} {m}")
                mb, peak = search_max_batch(e, p, shape, cap_mib, m)
                print(f"[speed]     {e} {p} {m} @batch {args.speed_batch}")
                sps = measure_speed(e, p, args.speed_batch, shape, cap_mib, m)
                results[(e, p, m)] = {"max_batch": mb, "peak_at_max": peak, "speed_sps": sps}

    print("\n===================== SUMMARY =====================")
    print(f"{'engine':12s} {'prec':5s} {'mode':6s} {'max_batch':>10s} "
          f"{'speed(sps)':>12s} {'peak@max(MiB)':>14s}")
    for e in engines:
        for p in precisions:
            for m in modes:
                r = results[(e, p, m)]
                print(f"{e:12s} {p:5s} {m:6s} {r['max_batch']:>10d} "
                      f"{(r['speed_sps'] or 0):>12.1f} {str(r['peak_at_max']):>14s}")

    if args.result_json:
        def _cap(cmd):
            try:
                return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip() or None
            except Exception:
                return None
        artifact = {
            "benchmark_id": "gpu-transformer-max-batch",
            "provenance": {
                "generated_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
                "git_commit": _cap(["git", "rev-parse", "HEAD"]),
                "git_dirty": bool(_cap(["git", "status", "--porcelain"])),
                "gpu_name": _cap(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
            },
            "corpus": CORPUS,
            "model": {"d_model": D, "heads": H, "ff": FF, "layers": LAYERS,
                      "input_vocab": shape[0], "output_vocab": shape[1],
                      "input_seq": shape[2], "decoder_seq": shape[3]},
            "gpu_total_mib": total_mib, "vram_cap_mib": cap_mib,
            "speed_batch": args.speed_batch,
            "results": [{"engine": e, "precision": p, "mode": m, **results[(e, p, m)]}
                        for e in engines for p in precisions for m in modes],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.result_json)), exist_ok=True)
        with open(args.result_json, "w") as f:
            json.dump(artifact, f, indent=1)
        print(f"\nresult JSON written to {args.result_json}")


if __name__ == "__main__":
    main()
