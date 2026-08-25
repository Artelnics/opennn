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

import os
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

    # `status_short` keeps the name and the list type used by 46 of the stored
    # results -- more than any other spelling -- but capped, which is what the
    # count-and-sample variant was for: a very dirty tree should not dominate
    # the artifact. The count is the total, so truncation stays visible.
    return {
        "commit": commit or "unknown",
        "branch": branch or "unknown",
        "dirty": bool(status_lines),
        "status_short": status_lines[:50],
        "status_short_count": len(status_lines),
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


SESSION_ENV = "OPENNN_BENCH_SESSION"


def session_id() -> str:
    """The measurement session this run belongs to.

    Contract item 7.8: numbers may only be compared with others carrying the
    same session id. A session is one sitting on one machine with the clocks
    left alone -- section 6's warning about a 6,994 vs 8,682 samples/s swing an
    hour apart is what the id exists to make visible.

    A session normally spans several runner invocations, so it cannot be
    generated per process: export `$OPENNN_BENCH_SESSION` once and every runner
    launched under it agrees. Without it, each process gets its own
    `pid`-suffixed id, which is deliberately awkward -- an artifact whose
    session id nothing else shares is exactly what an un-anchored run is.
    """
    declared = os.environ.get(SESSION_ENV)
    if declared:
        return declared
    return f"adhoc-{os.getpid()}"


def result_destination(results_root: Path, dirty: bool | None = None) -> Path:
    """Where an artifact may be written: the evidence store, or `scratch/`.

    Contract item 7.8: a dirty tree writes to `results/scratch/`, never to the
    evidence store. That rule was written down and enforced by nothing --
    `git_metadata()` reported `dirty` and no code acted on it, which is how 39
    of the 107 existing artifacts came to be dirty-tree results sitting in
    `results/` as though they were reproducible.

    Pass `dirty` if the caller already has git metadata in hand; otherwise it is
    determined here. The directory is created, so a runner can write straight
    into the returned path.
    """
    if dirty is None:
        dirty = bool(git_metadata().get("dirty", True))

    destination = results_root / "scratch" if dirty else results_root
    destination.mkdir(parents=True, exist_ok=True)
    return destination
