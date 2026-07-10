# Generate the tiny synthetic dataset for the standalone-code-export benchmark:
# 100 inputs drawn from U(-1, 1) whose single target is their sum. It exists only
# to train a small model that is then exported to standalone source; the numbers
# are trivial, so no download and no train/test split are needed.

import csv
import random

VARIABLES = 100
SAMPLES = 200
SEED = 42


def main():
    rng = random.Random(SEED)

    rows = []
    for _ in range(SAMPLES):
        inputs = [rng.uniform(-1.0, 1.0) for _ in range(VARIABLES)]
        rows.append(inputs + [sum(inputs)])

    with open("sum.csv", "w", newline="") as f:
        csv.writer(f, delimiter=";").writerows(rows)
    print(f"wrote sum.csv ({SAMPLES} samples, {VARIABLES} inputs)")


if __name__ == "__main__":
    main()
