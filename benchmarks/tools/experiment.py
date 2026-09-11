"""Shared artifact locations for the application and quality experiments."""

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import uuid

from common import RESULTS, REPO_ROOT


def git_metadata():
    """Use Windows Git for a Windows checkout accessed through WSL.

    Linux Git's per-file stat calls on an NTFS/OneDrive mount can time out.
    A failed provenance command must never be interpreted as a clean tree.
    """
    executable, root = "git", str(REPO_ROOT)
    if root.startswith("/mnt/") and shutil.which("git.exe"):
        executable = shutil.which("git.exe")
        root = subprocess.check_output(["wslpath", "-w", root], text=True).strip()

    def read(*arguments):
        return subprocess.check_output(
            [executable, "-C", root, *arguments], text=True, timeout=30
        ).strip()

    try:
        status = read("status", "--porcelain").splitlines()
        return {
            "commit": read("rev-parse", "HEAD"),
            "branch": read("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "dirty_count": len(status),
            "dirty_sample": status[:20],
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {"commit": None, "dirty": None, "error": str(error)}


def result_directory(kind, requested=None):
    """Keep diagnostic experiments in ignored scratch; never overwrite a run."""
    root = (RESULTS / "scratch").resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = (
        Path(requested).resolve()
        if requested
        else root / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"
    )
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Experiment output must be a new directory below {root}")
    path.mkdir(parents=True, exist_ok=False)
    return path


def dispatch(arguments):
    """Inspect --family itself, not arbitrary option values or paths."""
    family = None
    for index, value in enumerate(arguments):
        if value == "--family" and index + 1 < len(arguments):
            family = arguments[index + 1]
        elif value.startswith("--family="):
            family = value.split("=", 1)[1]
    modules = {
        "startup": "application_startup",
        "deployment": "application_deployment",
        "quality": "quality_runner",
    }
    if family not in modules:
        return None
    import importlib

    return importlib.import_module(modules[family]).main(arguments) or 0
