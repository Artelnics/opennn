# Generate the Rosenbrock precision benchmark dataset from the Neural Designer
# blog post (10 inputs, 10,000 samples), z-normalize it once, and write a single
# shared file that OpenNN, PyTorch, and TensorFlow all consume. The blog uses no
# train/test split: training and evaluation happen on the full dataset.
#
# Rosenbrock (Neural Designer benchmark):
#   x_i ~ U(-1, 1),   y = sum_i [ (1 - x_i)^2 + 100 (x_{i+1} - x_i^2)^2 ]

import csv
import random

VARIABLES = 10
SAMPLES = 10000
SEED = 1234


def rosenbrock(x):
    total = 0.0
    for i in range(len(x) - 1):
        total += (1.0 - x[i]) ** 2 + 100.0 * (x[i + 1] - x[i] * x[i]) ** 2
    return total


def main():
    rng = random.Random(SEED)

    inputs = [[rng.uniform(-1.0, 1.0) for _ in range(VARIABLES)] for _ in range(SAMPLES)]
    targets = [rosenbrock(row) for row in inputs]

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

    with open("rosenbrock.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"wrote rosenbrock.csv ({SAMPLES} samples, {VARIABLES} inputs)")


if __name__ == "__main__":
    main()
