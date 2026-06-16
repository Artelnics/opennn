# Neutral scorer: compute the full-dataset MSE the same way for every
# framework, from the targets (rosenbrock.csv, last column) and a predictions
# file (one predicted value per line, same order as the dataset rows).
#
# Usage: python score.py <label> <predictions_file>

import sys
import csv


def main():
    label = sys.argv[1]
    pred_path = sys.argv[2]

    targets = []
    with open("rosenbrock.csv", newline="") as f:
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
    mse = sum((p - t) ** 2 for p, t in zip(preds, targets)) / n
    print(f"{label:10s}  MSE={mse:.6f}")


if __name__ == "__main__":
    main()
