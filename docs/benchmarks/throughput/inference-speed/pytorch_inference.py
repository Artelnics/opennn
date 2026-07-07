# PyTorch CPU inference-speed benchmark.
#
# Mirrors the training-speed model so the two read together: a 2-layer MLP
# (F -> F -> 1, tanh then linear) on the Rosenbrock dataset. Here the network
# only does inference, under torch.no_grad() and model.eval() (no autograd
# graph, no dropout) -- the apples-to-apples equivalent of OpenNN's
# calculate_outputs() forward pass.
#
# Reports median seconds per full pass over the dataset, samples/second
# (throughput), and milliseconds per batch (latency).
#
#   usage:  python pytorch_inference.py <samples> <features> [batch] [reps]

import sys
import time

import numpy as np
import torch


def rosenbrock(n, f, seed=1234):
    rng = np.random.default_rng(seed)
    x = (rng.random((n, f), dtype=np.float32) * 2.0 - 1.0)
    return x


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    torch.manual_seed(42)
    torch.set_num_threads(torch.get_num_threads())  # CPU; default thread pool

    x = torch.from_numpy(rosenbrock(samples, features)).contiguous()
    print(f"samples={samples} features={features} batch={batch} reps={reps}")

    model = torch.nn.Sequential(
        torch.nn.Linear(features, features),
        torch.nn.Tanh(),
        torch.nn.Linear(features, 1),
    )
    model.eval()

    starts = list(range(0, samples - batch + 1, batch))

    @torch.no_grad()
    def run_pass():
        for s in starts:
            out = model(x[s:s + batch])
            float(out[0, 0])  # touch result so nothing is optimized away

    # Warmup.
    run_pass()
    run_pass()

    batched_samples = len(starts) * batch
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        run_pass()
        times.append(time.perf_counter() - t0)

    times.sort()
    median = times[len(times) // 2]
    print(f"median_pass_s={median:.6f}")
    print(f"samples_per_sec={batched_samples / median:.0f}")
    print(f"ms_per_batch={median / len(starts) * 1000.0:.4f}")
    print("RESULT=OK")


if __name__ == "__main__":
    main()
