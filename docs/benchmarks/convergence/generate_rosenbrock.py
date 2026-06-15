# Generate the Rosenbrock approximation benchmark dataset, normalize it once,
# and write a shared train/test split that OpenNN, PyTorch, and TensorFlow all
# consume. Doing the generation, normalization, and split here (once) keeps the
# accuracy comparison strictly apples-to-apples: every framework sees the exact
# same numbers, with no per-framework scaling.
#
# Rosenbrock (Neural Designer benchmark):
#   x_i ~ U(-1, 1),   y = sum_i [ (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2 ]

import csv
import random

VARIABLES = 10
SAMPLES = 10000
SEED = 1234
TRAIN_FRACTION = 0.8


def rosenbrock(x):
    total = 0.0
    for i in range(len(x) - 1):
        total += (1.0 - x[i]) ** 2 + 100.0 * (x[i + 1] - x[i] * x[i]) ** 2
    return total


def main():
    rng = random.Random(SEED)

    inputs = [[rng.uniform(-1.0, 1.0) for _ in range(VARIABLES)] for _ in range(SAMPLES)]
    targets = [rosenbrock(row) for row in inputs]

    # Column-wise z-score normalization (computed on the whole set, applied to all).
    columns = list(zip(*inputs)) + [tuple(targets)]
    stats = []
    for col in columns:
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        std = var ** 0.5 or 1.0
        stats.append((mean, std))

    def normalize(row, target):
        values = list(row) + [target]
        return [(v - m) / s for v, (m, s) in zip(values, stats)]

    rows = [normalize(inputs[i], targets[i]) for i in range(SAMPLES)]

    # Deterministic split (no shuffle dependence across frameworks).
    n_train = int(SAMPLES * TRAIN_FRACTION)

    def write(path, subset):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerows(subset)

    write("rosenbrock_train.csv", rows[:n_train])
    write("rosenbrock_test.csv", rows[n_train:])
    print(f"wrote rosenbrock_train.csv ({n_train}) and rosenbrock_test.csv ({SAMPLES - n_train})")


if __name__ == "__main__":
    main()
