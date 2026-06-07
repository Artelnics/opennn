# Neutral scorer: compute MSE and R^2 the same way for every framework, from
# the test targets (rosenbrock_test.csv, last column) and a predictions file
# (one predicted value per line, same order as the test rows).
#
# Usage: python score.py <label> <predictions_file>

import sys
import csv


def main():
    label = sys.argv[1]
    pred_path = sys.argv[2]

    targets = []
    with open("rosenbrock_test.csv", newline="") as f:
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
    mean_t = sum(targets) / n
    ss_tot = sum((t - mean_t) ** 2 for t in targets)
    ss_res = sum((p - t) ** 2 for p, t in zip(preds, targets))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"{label:10s}  MSE={mse:.6f}  R2={r2:.6f}")


if __name__ == "__main__":
    main()
