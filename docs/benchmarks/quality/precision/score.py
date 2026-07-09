# Neutral scorer: compute the full-dataset MSE the same way for every
# framework, from the targets (rosenbrock.csv, last column) and a predictions
# file (one predicted value per line, same order as the dataset rows).
#
# Importable: run_precision.py calls mse() so every engine is scored identically.
#
# Usage: python score.py <label> <predictions_file> [targets_csv]

import sys
import csv


def mse(targets_csv, pred_path):
    targets = []
    with open(targets_csv, newline="") as f:
        for row in csv.reader(f):
            targets.append(float(row[-1]))

    preds = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(float(line))

    assert len(preds) == len(targets), f"{len(preds)} preds vs {len(targets)} targets"

    n = len(targets)
    return sum((p - t) ** 2 for p, t in zip(preds, targets)) / n


def main():
    label = sys.argv[1]
    pred_path = sys.argv[2]
    targets_csv = sys.argv[3] if len(sys.argv) > 3 else "rosenbrock.csv"

    print(f"{label:10s}  MSE={mse(targets_csv, pred_path):.6f}")


if __name__ == "__main__":
    main()
