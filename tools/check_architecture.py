#!/usr/bin/env python3
"""Reject new dependencies that cross OpenNN's established module direction."""

from __future__ import annotations

import re
from pathlib import Path

from check_code_quality import source_files

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "opennn"
INCLUDE = re.compile(r'^\s*#\s*include\s+["<]opennn/([^">]+)[">]')
CUDA_TYPE = re.compile(
    r"\b(?:cuda(?:Stream|Event|Graph|DataType)_t|"
    r"cublas[A-Za-z0-9_]*_t|cudnn[A-Za-z0-9_]*_t)\b"
)

ALLOWED = {
    "core": {"core"},
    "dataset": {"core", "dataset", "root"},
    "network": {"core", "network", "root"},
    "training": {"core", "dataset", "network", "training", "root"},
    "evaluation": {"core", "dataset", "network", "training", "evaluation", "root"},
    "model_selection": {"core", "dataset", "network", "training", "model_selection", "root"},
    "models": {"core", "network", "models", "root"},
    "response_optimization": {"core", "network", "response_optimization", "root"},
    "root": {"core", "dataset", "evaluation", "model_selection", "models",
             "network", "response_optimization", "root", "training"},
}

# These cycles predate the ratchet and require broader API moves to remove.
# Keep the exceptions path-specific so another file cannot copy the dependency.
EXCEPTIONS = {
    ("dataset/bert_dataset.cpp", "network"),
    ("dataset/correlations.cpp", "models"),
    ("dataset/correlations.cpp", "network"),
    ("dataset/correlations.cpp", "training"),
    ("dataset/language_dataset.h", "network"),
    ("dataset/text_generation_dataset.h", "network"),
    ("dataset/yolo_dataset.cpp", "network"),
    ("network/back_propagation.cpp", "training"),
    ("network/chat.cpp", "models"),
    ("network/standard_networks.h", "models"),
}


def module(path: Path, source: Path = SOURCE) -> str:
    parts = path.relative_to(source).parts
    return parts[0] if len(parts) > 1 else "root"


def check(source: Path = SOURCE) -> list[str]:
    violations = []
    for path in source_files(source):
        relative = path.relative_to(source).as_posix()
        source_module = module(path, source)
        for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
            # .cuh files are CUDA implementation interfaces and may name vendor
            # types. The portable .h boundary keeps its existing restriction.
            if (path.suffix == ".h"
                    and not relative.startswith("core/cuda/")
                    and relative != "core/opennn_types.h"
                    and CUDA_TYPE.search(line)):
                violations.append(
                    f"{relative}:{number}: expose an OpenNN backend type instead of a CUDA vendor type"
                )
            match = INCLUDE.match(line)
            if not match:
                continue
            include_parts = Path(match.group(1)).parts
            target_module = include_parts[0] if len(include_parts) > 1 else "root"
            if (target_module not in ALLOWED[source_module]
                    and (relative, target_module) not in EXCEPTIONS):
                violations.append(
                    f"{relative}:{number}: {source_module} must not depend on {target_module}"
                )
    return violations


def main() -> int:
    violations = check()
    if violations:
        print("Architecture dependency check failed:\n" + "\n".join(violations))
        return 1
    print("Architecture dependency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
