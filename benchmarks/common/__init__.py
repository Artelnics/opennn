"""Helpers every benchmark runner needs, so each one stops growing its own.

Step 3 of REORGANIZATION_PLAN.md. What lives here is the part that must be the
same everywhere -- what a result records about the tree, the frameworks and the
machine -- rather than anything about a particular model or metric.

Import from a runner with:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from common import git_metadata, framework_versions

See DUPLICATION_LEDGER.md for what was measured before this existed: 2,217
duplicated lines across the runners, and helpers that had drifted rather than
merely been copied -- `versions()` in nine distinct forms, `engine_cmd()` in ten.
Only helpers that were already identical, or whose variants differ in ways this
module makes explicit, have been lifted. The drifted ones are a merge, not a
lift, and are listed in the ledger as still outstanding.
"""

from .provenance import (
    REPO_ROOT,
    file_info,
    framework_versions,
    git_metadata,
    repo_root,
    run_text,
)

from .gpu import (
    gpu_state,
    measure_idle,
    used_mib,
    wait_for_idle,
)

__all__ = [
    "REPO_ROOT",
    "file_info",
    "framework_versions",
    "git_metadata",
    "repo_root",
    "run_text",
    "gpu_state",
    "measure_idle",
    "used_mib",
    "wait_for_idle",
]
