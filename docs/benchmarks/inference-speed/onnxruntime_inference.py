# ONNX Runtime CPU inference-speed benchmark.
#
# ONNX Runtime is a dedicated inference engine: it loads a pre-built model
# (model_inference.onnx, the same 2-layer MLP the other frameworks build --
# generate it once with export_onnx.py) and runs it. It cannot train, so this
# program only times the forward pass -- the workload ONNX Runtime exists for,
# and the natural CPU-deployment competitor to OpenNN's calculate_outputs().
#
# Reports median seconds per full pass over the dataset, samples/second
# (throughput), and milliseconds per batch (latency).
#
#   usage:  python onnxruntime_inference.py <samples> <features> [batch] [reps] [model_path]

import sys
import time

import numpy as np
import onnxruntime as ort


def rosenbrock(n, f, seed=1234):
    rng = np.random.default_rng(seed)
    return (rng.random((n, f), dtype=np.float32) * 2.0 - 1.0)


def main():
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    features = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    model_path = sys.argv[5] if len(sys.argv) > 5 else "model_inference.onnx"

    x = rosenbrock(samples, features)
    print(f"samples={samples} features={features} batch={batch} reps={reps}")

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    starts = list(range(0, samples - batch + 1, batch))
    batches = [np.ascontiguousarray(x[s:s + batch]) for s in starts]

    def run_pass():
        for xb in batches:
            out = session.run(None, {input_name: xb})
            float(out[0][0, 0])

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
