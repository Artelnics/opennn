"""What a result needs to record about the tree and the machine that produced it.

Every runner grew its own copy of these. They drifted, and not cosmetically:
`versions()` existed in nine distinct forms, recording a different set of fields
in each family, so what a result says about its own provenance depended on which
directory produced it. A convergence result recorded no framework versions and
no GPU at all; a capacity result recorded CUDA and cuDNN. `git_metadata()` had
five forms differing in how they record a dirty tree -- full status text, a
count, or a truncated sample -- which is why some results carry `git.dirty` and
others have no such field to be true or false.

These are the union, so a result records the same things wherever it was run.
Fields that cannot be collected are omitted rather than guessed; callers should
treat a missing key as "not available here", never as a value.
"""

from __future__ import annotations

import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

__all__ = [
    "run_text",
    "repo_root",
    "git_metadata",
    "framework_versions",
    "file_info",
]

# A command that fails is not an error here: provenance is best-effort, and a
# missing git or nvidia-smi must not take a benchmark down with it.
def run_text(cmd: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def repo_root(start: Path | None = None) -> Path:
    """The working tree containing `start`, or its third parent as a fallback."""
    here = (start or Path(__file__)).resolve().parent
    root = run_text(["git", "-C", str(here), "rev-parse", "--show-toplevel"])
    return Path(root).resolve() if root else here.parents[2]


REPO_ROOT = repo_root()


def git_metadata(root: Path | None = None) -> dict[str, Any]:
    """Commit, branch, and whether the tree was dirty when this ran.

    `dirty` is the field that decides whether a result can be regenerated, so it
    is always present -- along with enough of the status to see what was
    uncommitted, capped so a very dirty tree cannot dominate the artifact.
    """
    target = str(root or REPO_ROOT)

    commit = run_text(["git", "-C", target, "rev-parse", "HEAD"])
    branch = run_text(["git", "-C", target, "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_text(["git", "-C", target, "status", "--short", "--untracked-files=no"])

    status_lines = status.splitlines()

    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status_lines),
        "status_short_count": len(status_lines),
        "status_short_sample": status_lines[:50],
        "status_short_truncated": len(status_lines) > 50,
    }


def framework_versions(python: str | None = None) -> dict[str, Any]:
    """Python, the two frameworks, what they were built against, and the GPU.

    Imported in a subprocess rather than here: a runner that imports torch to
    ask its version has loaded CUDA into the process that is about to time
    something. `python` selects the interpreter the benchmark itself uses, which
    is not always the one running the driver.
    """
    interpreter = python or sys.executable

    versions: dict[str, Any] = {
        "python": sys.version.split()[0],
        # Named python_executable, not bench_python: both appear across the
        # stored results for the same fact, and this is the one the newer
        # artifacts use. Old results keep their key; step 7 covers the corpus.
        "python_executable": interpreter,
        "platform": platform.platform(),
    }

    probes = {
        "torch": "import torch;"
                 "print(torch.__version__);"
                 "print(torch.version.cuda);"
                 "print(torch.backends.cudnn.version())",
        "tensorflow": "import tensorflow as tf;"
                      "b = tf.sysconfig.get_build_info();"
                      "print(tf.__version__);"
                      "print(b.get('cuda_version'));"
                      "print(b.get('cudnn_version'))",
    }

    for name, script in probes.items():
        completed = subprocess.run([interpreter, "-c", script],
                                   capture_output=True, text=True, timeout=120)

        if completed.returncode != 0:
            # Why it is absent is provenance too: "not installed" and "installed
            # but failed to import" are different facts about the machine.
            reason = completed.stderr.strip().splitlines()
            versions[f"{name}_error"] = reason[-1] if reason else "probe failed"
            continue

        lines = completed.stdout.strip().splitlines()
        if not lines:
            continue

        versions[name] = lines[0]
        if len(lines) > 2:
            versions[f"{name}_built_cuda"] = lines[1]
            versions[f"{name}_built_cudnn"] = lines[2]

    nvcc = run_text(["nvcc", "--version"], timeout=15)
    match = re.search(r"release\s+([0-9.]+)", nvcc)
    if match:
        versions["cuda_nvcc"] = match.group(1)

    gpu = run_text(
        ["nvidia-smi", "--query-gpu=name,driver_version,power.limit",
         "--format=csv,noheader"],
        timeout=15,
    )
    if gpu:
        versions["gpu"] = gpu.splitlines()[0].strip()

    return versions


def file_info(path: Path) -> dict[str, Any]:
    """Identity of an input file, so a result names the data it was measured on."""
    info: dict[str, Any] = {"path": str(path)}

    if path.exists():
        stat = path.stat()
        info.update({"exists": True, "bytes": stat.st_size, "mtime": stat.st_mtime})
    else:
        info["exists"] = False

    return info
