#!/usr/bin/env python3
"""
Count logical source lines for the Iris API-comparison snippets.

The metric intentionally follows the rule used in the API comparison article:
ignore blank lines and comments, ignore syntactic C++ brace-only lines, and
count multi-line statements as one logical instruction.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = {
    "opennn_cpp": ROOT / "opennn_iris.cpp",
    "pytorch_python": ROOT / "pytorch_iris.py",
    "tensorflow_python": ROOT / "tensorflow_iris.py",
}


def strip_cpp_comment(line: str, in_block: bool) -> tuple[str, bool]:
    out = []
    i = 0
    while i < len(line):
        if in_block:
            end = line.find("*/", i)
            if end == -1:
                return "".join(out), True
            i = end + 2
            in_block = False
        elif line.startswith("/*", i):
            in_block = True
            i += 2
        elif line.startswith("//", i):
            break
        else:
            out.append(line[i])
            i += 1
    return "".join(out), in_block


def count_cpp_lsloc(path: Path) -> int:
    count = 0
    statement = ""
    in_block_comment = False

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line, in_block_comment = strip_cpp_comment(raw_line, in_block_comment)
        line = line.strip()

        if not line:
            continue
        if re.fullmatch(r"[{};]+", line):
            continue

        if line.startswith("#include"):
            count += 1
            continue

        if line in {"try", "else"} or line.startswith(
            ("catch", "for ", "for(", "while ", "while(", "if ", "if(", "int main")
        ):
            count += 1
            continue

        statement = f"{statement} {line}".strip()
        if ";" in line:
            count += 1
            statement = ""

    if statement:
        count += 1

    return count


def count_python_lsloc(path: Path) -> int:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return sum(1 for node in ast.walk(tree) if isinstance(node, ast.stmt))


def main() -> None:
    results = {
        "opennn_cpp": count_cpp_lsloc(FILES["opennn_cpp"]),
        "pytorch_python": count_python_lsloc(FILES["pytorch_python"]),
        "tensorflow_python": count_python_lsloc(FILES["tensorflow_python"]),
    }

    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
