# PyTorch GPU HIGGS dense training-speed benchmark, the counterpart to
# opennn_speed (GPU HIGGS training).
#
# Mirrors the canonical HIGGS dense classifier (28 -> hidden -> hidden -> 1,
# ReLU hidden, sigmoid output, binary cross entropy -- see
# docs/benchmarks/throughput/higgs/README.md). The train and test CSVs are loaded
# (features then last-column label), the training tensors are made GPU-resident
# once (no host<->device copy per step), and Adam runs for N epochs at the given
# batch. After training the test set is scored and accuracy / log-loss / ROC-AUC
# are reported for the quality gate.
#
# "Highest performance" path (adapted from higgs/higgs_framework_cpu.py's
# run_pytorch to the GPU):
#   * whole dataset resident on the GPU,
#   * TF32 matmuls enabled (bf16 mode),
#   * autocast (bf16) mixed precision,
#   * per-epoch GPU-resident reshuffle (matches OpenNN).
#
#   usage:  python pytorch_speed.py <train_csv> <epochs> <batch> <precision>
#                                   <shuffle> <hidden> <activation>
#                                   <hidden_layers> <test_csv>
#                                   <min_accuracy> <max_log_loss> <min_auc>
#           precision  = "bf16" (autocast + TF32) or "fp32" (strict)
#           shuffle    = "shuffle" to reshuffle every epoch (matches OpenNN)
#           activation = "relu" (default) or "tanh"
#           thresholds = "none" when unset
#   env:    PT_COMPILE_MODE=default|reduce-overhead|max-autotune -> torch.compile
#                      the model (reduce-overhead adds CUDA graphs); unset = eager
#           PT_COMPILE_STEP=1 -> compile the whole train step (forward, backward
#                      and the optimizer update) instead of the model alone
#           PT_FUSED_ADAM=1 -> torch.optim.Adam(fused=True)

import contextlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "higgs"))
from metrics import binary_metrics, parse_optional_float, passes_quality_gate

def load_csv(path):
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    x = np.ascontiguousarray(data[:, :-1])
    y = np.ascontiguousarray(data[:, -1:].astype(np.float32))
    return x, y

def main():
    train_csv = sys.argv[1] if len(sys.argv) > 1 else "higgs_train.csv"
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
    precision = sys.argv[4] if len(sys.argv) > 4 else "bf16"
    shuffle = (sys.argv[5] if len(sys.argv) > 5 else "shuffle") in ("shuffle", "1", "true")
    hidden = int(sys.argv[6]) if len(sys.argv) > 6 else 1024
    activation = (sys.argv[7] if len(sys.argv) > 7 else "relu").lower()
    hidden_layers = int(sys.argv[8]) if len(sys.argv) > 8 else 2
    test_csv = sys.argv[9] if len(sys.argv) > 9 else "higgs_test.csv"

    min_accuracy = parse_optional_float(sys.argv[10] if len(sys.argv) > 10 else None)
    max_log_loss = parse_optional_float(sys.argv[11] if len(sys.argv) > 11 else None)
    min_auc = parse_optional_float(sys.argv[12] if len(sys.argv) > 12 else None)

    assert torch.cuda.is_available(), "CUDA GPU required"
    device = "cuda"
    torch.manual_seed(42)

    use_autocast = precision == "bf16"
    # fp32 runs with TF32 tensor cores in every engine of this benchmark
    # (OpenNN's fp32 GEMMs are CUBLAS_COMPUTE_32F_FAST_TF32); "strict" disables it.
    allow_tf32 = precision != "strict"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.benchmark = True

    x_np, y_np = load_csv(train_csv)
    xt_np, yt_np = load_csv(test_csv)
    features = x_np.shape[1]
    samples = x_np.shape[0]

    print("engine=pytorch")
    print("mode=train")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"samples={samples}")
    print(f"batch={batch}")
    print(f"epochs={epochs}")
    print(f"hidden={hidden}")
    print(f"hidden_layers={hidden_layers}")
    print(f"activation={activation}")
    print(f"precision={precision} autocast={use_autocast} tf32={allow_tf32} shuffle={shuffle}")

    x = torch.from_numpy(x_np).to(device).contiguous()
    y = torch.from_numpy(y_np).to(device).contiguous()

    act_layer = torch.nn.ReLU if activation == "relu" else torch.nn.Tanh
    layers = []
    current = features
    for _ in range(hidden_layers):
        layers.append(torch.nn.Linear(current, hidden))
        layers.append(act_layer())
        current = hidden
    layers.append(torch.nn.Linear(current, 1))
    model = torch.nn.Sequential(*layers).to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    loss_fn = torch.nn.BCEWithLogitsLoss()
    compile_mode = os.environ.get("PT_COMPILE_MODE")
    compile_step = os.environ.get("PT_COMPILE_STEP") is not None
    fused_adam = os.environ.get("PT_FUSED_ADAM") is not None
    optimizer = torch.optim.Adam(model.parameters(), fused=fused_adam)
    print(f"mode={'compile:' + compile_mode + (':step' if compile_step else ':model') if compile_mode else 'eager'}"
          f"{' +fused_adam' if fused_adam else ''}")

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast else contextlib.nullcontext())

    if compile_mode and not compile_step:
        model = torch.compile(model, mode=None if compile_mode == "default" else compile_mode)

    def train_step(xb, yb):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            pred = model(xb)
            loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        return loss

    if compile_mode and compile_step:
        train_step = torch.compile(train_step, mode=None if compile_mode == "default" else compile_mode)

    n = x.shape[0]
    starts = list(range(0, n - batch + 1, batch))

    def run_epoch():
        model.train()
        if shuffle:

            perm = torch.randperm(n, device=device)
            for s in starts:
                idx = perm[s:s + batch]
                train_step(x[idx], y[idx])
        else:
            for s in starts:
                train_step(x[s:s + batch], y[s:s + batch])
        torch.cuda.synchronize()

    print("warmup...")
    run_epoch()
    run_epoch()

    print(f"TRAIN_START_UNIX={time.time():.3f}", flush=True)
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        run_epoch()
        times.append(time.perf_counter() - t0)
    print(f"TRAIN_END_UNIX={time.time():.3f}", flush=True)

    times.sort()
    median_epoch_s = times[len(times) // 2]
    # An epoch runs whole batches only; dividing the full split by the epoch time
    # overstates throughput by up to one batch, which is 6.5% at batch 896,000.
    samples_per_epoch = len(starts) * batch
    samples_per_sec = samples_per_epoch / median_epoch_s

    processed = (xt_np.shape[0] // batch) * batch
    xt = torch.from_numpy(xt_np[:processed]).to(device).contiguous()
    model.eval()
    preds = []
    with torch.no_grad():
        for s in range(0, processed, batch):
            with ctx:
                logits = model(xt[s:s + batch])

            probs = torch.sigmoid(logits.float())
            preds.append(probs.cpu().numpy())
    pred_np = np.vstack(preds) if preds else np.empty((0, 1), dtype=np.float32)
    metrics = binary_metrics(yt_np[: pred_np.shape[0]], pred_np)

    print(f"median_epoch_s={median_epoch_s:.9g}")
    print(f"samples_per_sec={samples_per_sec:.0f}")
    print(f"test_samples={pred_np.shape[0]}")
    for key, value in metrics.items():
        print(f"{key}={value:.9g}")
    print(f"peak_vram_mb={torch.cuda.max_memory_allocated() / 1e6:.0f}")

    if min_accuracy is not None or max_log_loss is not None or min_auc is not None:
        gate = passes_quality_gate(metrics, min_accuracy, max_log_loss, min_auc)
        print(f"quality_gate={'PASS' if gate else 'FAIL'}")

    print("RESULT=OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error={exc}", file=sys.stderr)
        print("RESULT=ERROR")
        raise
