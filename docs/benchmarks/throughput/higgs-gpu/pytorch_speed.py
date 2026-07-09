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

import contextlib
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
    allow_tf32 = precision in ("bf16", "tf32")
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

    # Whole training set resident on the GPU once (no host<->device copy per step).
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
    layers.append(torch.nn.Sigmoid())
    model = torch.nn.Sequential(*layers).to(device)
    print(f"parameters={sum(p.numel() for p in model.parameters())}")

    loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())

    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
           if use_autocast else contextlib.nullcontext())

    def train_step(xb, yb):
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            pred = model(xb)
            loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        return loss

    n = x.shape[0]
    starts = list(range(0, n - batch + 1, batch))

    def run_epoch():
        model.train()
        if shuffle:
            # Fresh permutation each epoch, resident on the GPU. Per-batch index
            # gather is PyTorch's fast shuffle path.
            perm = torch.randperm(n, device=device)
            for s in starts:
                idx = perm[s:s + batch]
                train_step(x[idx], y[idx])
        else:
            for s in starts:
                train_step(x[s:s + batch], y[s:s + batch])
        torch.cuda.synchronize()

    # Warmup: cuDNN autotuning, workspace allocation.
    print("warmup...")
    run_epoch()
    run_epoch()

    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        run_epoch()
        times.append(time.perf_counter() - t0)

    times.sort()
    median_epoch_s = times[len(times) // 2]
    samples_per_sec = samples / median_epoch_s

    # Score the test set: whole batch-aligned slice on the GPU, forward-only.
    processed = (xt_np.shape[0] // batch) * batch
    xt = torch.from_numpy(xt_np[:processed]).to(device).contiguous()
    model.eval()
    preds = []
    with torch.no_grad(), ctx:
        for s in range(0, processed, batch):
            preds.append(model(xt[s:s + batch]).float().cpu().numpy())
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
