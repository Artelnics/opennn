"""Shared specialized-family dispatch, provenance and artifact locations."""

from datetime import datetime, timezone
from pathlib import Path
import uuid

from common import RESULTS, git_metadata as git_metadata

SPECIALIZED_FAMILIES = {
    "startup": "application_startup",
    "deployment": "application_deployment",
    "quality": "quality_runner",
    "qwen": "families.qwen",
}


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
    if family not in SPECIALIZED_FAMILIES:
        return None
    import importlib

    return importlib.import_module(SPECIALIZED_FAMILIES[family]).main(arguments) or 0
