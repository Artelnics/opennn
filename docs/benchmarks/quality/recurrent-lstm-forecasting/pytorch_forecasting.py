"""Optimized PyTorch RNN/LSTM forecasting training benchmark.

The batch-size, seed-count and fixed-work controls are shared with the OpenNN
driver through OPENNN_FORECASTING_{BATCH_SIZES,SEEDS,EPOCHS}. With EPOCHS set,
every engine executes the same number of complete training epochs; without it,
the original early-stopping quality protocol is retained.
"""

import os
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from xf_common import SCENARIOS, make_windows


def env_ints(name, default):
    text = os.environ.get(name, "").strip()
    if not text:
        return list(default)
    values = [int(item) for item in text.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} must contain positive comma-separated integers")
    return values


def warm_training_path(net, opt, lossf, xtr, ytr, xva, yva, batch):
    """Materialize cuDNN/optimizer state without changing the measured run."""
    initial_state = {key: value.detach().clone()
                     for key, value in net.state_dict().items()}
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if DEV == "cuda" else None

    net.train()
    warm_sizes = [min(batch, xtr.shape[0])]
    tail = xtr.shape[0] % batch
    if tail and tail != warm_sizes[0]:
        warm_sizes.append(tail)
    for size in warm_sizes:
        opt.zero_grad(set_to_none=True)
        prediction = net(xtr[:size])
        loss = lossf(prediction.float(), ytr[:size].float())
        loss.backward()
        opt.step()

    net.eval()
    with torch.inference_mode():
        lossf(net(xva).float(), yva.float())
    if DEV == "cuda":
        torch.cuda.synchronize()

    net.load_state_dict(initial_state)
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    opt.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)


def profile_training_steps(net, opt, lossf, xtr, ytr, batch, steps, label):
    """Print an operator/CUDA-time profile, then restore the measured state."""
    if steps <= 0:
        return

    initial_state = {key: value.detach().clone()
                     for key, value in net.state_dict().items()}
    cpu_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if DEV == "cuda" else None

    activities = [torch.profiler.ProfilerActivity.CPU]
    if DEV == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    net.train()
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for step in range(steps):
            begin = (step * batch) % max(batch, xtr.shape[0] - batch + 1)
            opt.zero_grad(set_to_none=True)
            prediction = net(xtr[begin:begin + batch])
            loss = lossf(prediction.float(), ytr[begin:begin + batch].float())
            loss.backward()
            opt.step()
            prof.step()
        if DEV == "cuda":
            torch.cuda.synchronize()

    print(f"PROFILE engine=pytorch label={label} steps={steps}")
    sort_key = "self_cuda_time_total" if DEV == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_key, row_limit=40))

    net.load_state_dict(initial_state)
    for state in opt.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                value.zero_()
    opt.zero_grad(set_to_none=True)
    torch.random.set_rng_state(cpu_rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)


ALLOW_CPU = "--allow-cpu" in sys.argv[1:] or os.environ.get("CUDA_VISIBLE_DEVICES") == ""
DEV = "cuda" if torch.cuda.is_available() else "cpu"
if DEV == "cpu" and not ALLOW_CPU:
    print("ERROR device_mismatch engine=pytorch expected=cuda actual=cpu "
          "(pass --allow-cpu or CUDA_VISIBLE_DEVICES=\"\" for a deliberate CPU run)",
          file=sys.stderr)
    sys.exit(2)

PHASE = "GPU" if DEV == "cuda" else "CPU"
PRECISION = os.environ.get("OPENNN_FORECASTING_PRECISION", "fp32").strip().lower()
if PRECISION not in ("fp32", "bf16"):
    raise ValueError("OPENNN_FORECASTING_PRECISION must be fp32 or bf16")
if PRECISION == "bf16" and DEV != "cuda":
    raise ValueError("BF16 forecasting benchmark currently requires CUDA")
DTYPE = torch.bfloat16 if PRECISION == "bf16" else torch.float32
SEEDS = list(range(min(env_ints("OPENNN_FORECASTING_SEEDS", [5])[0], 5)))
FIXED_EPOCHS = int(os.environ.get("OPENNN_FORECASTING_EPOCHS", "0"))
COMPILE_MODE = os.environ.get("OPENNN_FORECASTING_PYTORCH_COMPILE", "0").lower()
CPU_THREADS = int(os.environ.get("OPENNN_FORECASTING_CPU_THREADS", "0"))
PROFILE_STEPS = int(os.environ.get("OPENNN_FORECASTING_PROFILE_STEPS", "0"))
if CPU_THREADS > 0:
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


class Net(nn.Module):
    def __init__(self, kind, n_feat, hidden, out):
        super().__init__()
        if kind == "Recurrent":
            self.rnn = nn.RNN(n_feat, hidden, batch_first=True, nonlinearity="tanh")
        else:
            self.rnn = nn.LSTM(n_feat, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


cli_scenarios = [arg for arg in sys.argv[1:] if arg != "--allow-cpu"]
env_scenarios = [item for item in os.environ.get(
    "OPENNN_FORECASTING_SCENARIOS", "").split(",") if item]
want = cli_scenarios or env_scenarios or [s[0] for s in SCENARIOS]

for sid, past, future, hidden, lr, default_batch, max_ep, patience, multi in SCENARIOS:
    if sid not in want:
        continue

    batch_sizes = env_ints("OPENNN_FORECASTING_BATCH_SIZES", [default_batch])
    epochs_limit = FIXED_EPOCHS or max_ep
    Xtr, Ytr, Xva, Yva, Xte, Yte, y_mean, y_std = make_windows(past, future, multi)
    xtr = torch.from_numpy(Xtr).to(DEV, dtype=DTYPE)
    ytr = torch.from_numpy(Ytr).to(DEV, dtype=DTYPE)
    xva = torch.from_numpy(Xva).to(DEV, dtype=DTYPE)
    yva = torch.from_numpy(Yva).to(DEV, dtype=DTYPE)
    xte = torch.from_numpy(Xte).to(DEV, dtype=DTYPE)
    n = Xtr.shape[0]

    for batch in batch_sizes:
        for kind in ("Recurrent", "LSTM"):
            rmses, times, epochs_l, throughputs = [], [], [], []
            for seed in SEEDS:
                if DEV == "cuda":
                    torch.cuda.empty_cache()
                torch.manual_seed(seed)
                net = Net(kind, Xtr.shape[2], hidden, Ytr.shape[1]).to(DEV, dtype=DTYPE)
                if COMPILE_MODE not in ("", "0", "false", "off"):
                    mode = "max-autotune" if COMPILE_MODE in ("1", "true", "on") else COMPILE_MODE
                    net = torch.compile(net, mode=mode)
                optimizer_mode = "fused" if DEV == "cuda" else "foreach"
                opt = torch.optim.Adam(
                    net.parameters(), lr=lr,
                    fused=DEV == "cuda", foreach=DEV == "cpu")
                lossf = nn.MSELoss()

                profile_training_steps(
                    net, opt, lossf, xtr, ytr, batch, PROFILE_STEPS,
                    f"{sid}/{kind}/batch={batch}")
                warm_training_path(net, opt, lossf, xtr, ytr, xva, yva, batch)

                best_val = float("inf")
                best_state = None
                failures = 0
                ran = 0
                validation_s = 0.0
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(epochs_limit):
                    net.train()
                    perm = torch.randperm(n, device=DEV)
                    for i in range(0, n, batch):
                        idx = perm[i:i + batch]
                        opt.zero_grad(set_to_none=True)
                        prediction = net(xtr[idx])
                        loss = lossf(prediction.float(), ytr[idx].float())
                        loss.backward()
                        opt.step()
                    ran += 1
                    if not FIXED_EPOCHS or ran == 1:
                        if DEV == "cuda":
                            torch.cuda.synchronize()
                        validation_t0 = time.perf_counter()
                        net.eval()
                        with torch.inference_mode():
                            validation_loss = lossf(net(xva).float(), yva.float()).item()
                        if DEV == "cuda":
                            torch.cuda.synchronize()
                        validation_s += time.perf_counter() - validation_t0
                        if validation_loss < best_val - 1e-7:
                            best_val = validation_loss
                            if not FIXED_EPOCHS:
                                best_state = {key: value.detach().clone()
                                              for key, value in net.state_dict().items()}
                            failures = 0
                        elif not FIXED_EPOCHS:
                            failures += 1
                            if failures >= patience:
                                break
                if DEV == "cuda":
                    torch.cuda.synchronize()
                train_s = time.perf_counter() - t0 - validation_s

                if best_state is not None:
                    net.load_state_dict(best_state)
                net.eval()
                with torch.inference_mode():
                    pred = net(xte).float().cpu().numpy()
                pred_orig = pred * y_std + y_mean
                true_orig = Yte * y_std + y_mean
                rmse = float(np.sqrt(np.mean((pred_orig - true_orig) ** 2)))
                params = sum(parameter.numel() for parameter in net.parameters())
                throughput = (n * ran) / train_s if train_s > 0 else 0.0
                rmses.append(rmse)
                times.append(train_s)
                epochs_l.append(ran)
                throughputs.append(throughput)
                print(f"METRIC engine=pytorch phase={PHASE} scenario={sid} net={kind} "
                      f"batch_size={batch} seed={seed} params={params} epochs={ran} "
                      f"test_rmse={rmse:.6f} time_s={train_s:.6f} "
                      f"samples_per_sec={throughput:.1f} train_windows={n} device={DEV} "
                      f"compile={COMPILE_MODE} optimizer={optimizer_mode} precision={PRECISION} "
                      f"warmup=full_and_tail")

                del loss, prediction, opt, net
                if DEV == "cuda":
                    torch.cuda.empty_cache()

            std = statistics.stdev(rmses) if len(rmses) > 1 else 0.0
            print(f"METRIC engine=pytorch phase={PHASE} scenario={sid} net={kind} "
                  f"batch_size={batch} seed=aggregate params={params} "
                  f"epochs_mean={round(statistics.fmean(epochs_l))} successful_runs={len(rmses)} "
                  f"test_rmse_mean={statistics.fmean(rmses):.6f} test_rmse_std={std:.6f} "
                  f"test_rmse_best={min(rmses):.6f} time_s_mean={statistics.fmean(times):.6f} "
                  f"samples_per_sec_mean={statistics.fmean(throughputs):.1f} "
                  f"train_windows={n} device={DEV} compile={COMPILE_MODE} "
                  f"optimizer={optimizer_mode} precision={PRECISION} "
                  f"warmup=full_and_tail", flush=True)
