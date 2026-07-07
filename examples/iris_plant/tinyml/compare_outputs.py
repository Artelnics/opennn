#!/usr/bin/env python3
"""Parity check between OpenNN reference outputs and the exported C model.

Compares, per test vector and output:
  - OpenNN reference (from the CSV written by the iris example)
  - the exported model compiled on the PC (pc_harness output)
  - optionally the same model running on the AVR emulator (avr_harness output)

Harness outputs are lines of space-separated 8-digit hex IEEE-754 floats
between BEGIN and END markers.

Usage: compare_outputs.py <reference.csv> <pc_output.txt> [avr_output.txt]
                          [--tolerance 1e-4]
"""

import re
import struct
import sys

ANSI_ESCAPES = re.compile(r"\x1b\[[0-9;]*m")
HEX_FLOAT = re.compile(r"\b[0-9a-fA-F]{8}\b")


def parse_csv(path: str) -> list[list[float]]:
    rows = []
    with open(path) as csv_file:
        for line in csv_file:
            line = line.strip()
            if line:
                rows.append([float(x) for x in line.split(";")])
    return rows


def parse_harness(path: str) -> list[list[float]]:
    """Extract hex-encoded float rows between BEGIN and END markers.

    Tolerates simavr decorations: UART echoed on stderr, ANSI color codes
    and trailing characters per line.
    """
    rows = []
    inside = False
    with open(path, errors="replace") as harness_file:
        for line in harness_file:
            line = ANSI_ESCAPES.sub("", line)
            if "BEGIN" in line:
                inside = True
                continue
            if inside and "END" in line:
                break
            if inside:
                tokens = HEX_FLOAT.findall(line)
                if tokens:
                    rows.append([struct.unpack(">f", bytes.fromhex(tok))[0]
                                 for tok in tokens])
    return rows


def max_abs_diff(a: list[list[float]], b: list[list[float]]) -> float:
    return max(abs(x - y) for row_a, row_b in zip(a, b) for x, y in zip(row_a, row_b))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    tolerance = 1e-4
    for arg in sys.argv[1:]:
        if arg.startswith("--tolerance"):
            tolerance = float(arg.split("=", 1)[1] if "=" in arg else args.pop())

    if len(args) < 2:
        sys.exit(__doc__)

    csv_rows = parse_csv(args[0])
    pc_rows = parse_harness(args[1])
    avr_rows = parse_harness(args[2]) if len(args) > 2 else None

    if not pc_rows:
        sys.exit("FAIL: no vectors parsed from PC harness output")

    n_outputs = len(pc_rows[0])
    reference = [row[-n_outputs:] for row in csv_rows]

    if len(reference) != len(pc_rows):
        sys.exit(f"FAIL: {len(reference)} reference vectors vs {len(pc_rows)} PC vectors")

    failures = 0

    diff_pc = max_abs_diff(reference, pc_rows)
    print(f"OpenNN reference vs exported C on PC : max abs diff = {diff_pc:.3e}")
    if diff_pc > tolerance:
        failures += 1

    if avr_rows is not None:
        if len(avr_rows) != len(reference):
            sys.exit(f"FAIL: {len(reference)} reference vectors vs {len(avr_rows)} emulator vectors")
        diff_avr = max_abs_diff(reference, avr_rows)
        diff_pc_avr = max_abs_diff(pc_rows, avr_rows)
        print(f"OpenNN reference vs exported C on MCU: max abs diff = {diff_avr:.3e}")
        print(f"Exported C: PC vs MCU emulator       : max abs diff = {diff_pc_avr:.3e}")
        if diff_avr > tolerance:
            failures += 1

        agree = sum(
            max(range(n_outputs), key=lambda i: ref[i]) == max(range(n_outputs), key=lambda i: avr[i])
            for ref, avr in zip(reference, avr_rows))
        print(f"Predicted class agreement (MCU vs reference): {agree}/{len(reference)}")
        if agree != len(reference):
            failures += 1

    print(f"Tolerance: {tolerance:g}")
    if failures:
        sys.exit(f"FAIL: {failures} parity check(s) above tolerance")
    print("PARITY OK")


if __name__ == "__main__":
    main()
