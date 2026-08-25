"""Finding the compiled OpenNN benchmark programs, once.

Step 3 of REORGANIZATION_PLAN.md, closing open question 9.5.

Every runner that shells out to a C++ benchmark grew its own copy of this
lookup: five `find_opennn_*` functions over a hardcoded directory list, plus 16
distinct environment-variable names for the override. The list had drifted from
what the build actually produces -- none of the copies knew about `build-bench`,
which is the directory section 0.4 of the plan tells you to create, so following
the plan literally left every runner unable to find the executable it had just
had you compile.

Resolution order, most specific first:

1. `$OPENNN_<NAME>_BIN` -- the per-program override, derived from the program
   name, so `opennn_speed` reads `OPENNN_SPEED_BIN`. This is the spelling the
   existing runners already use for the five programs that had one.
2. `$OPENNN_BIN` -- the blanket override.
3. Any explicit aliases the caller passes, for the handful of programs whose
   historical variable name does not follow the pattern (`OPENNN_RESNET_BIN`
   beside `OPENNN_RESNET50_BIN`, and so on).
4. The build directories below.

An override is honoured whether or not the path exists, and the `found` flag
reports which -- a runner should be able to say "you pointed me at a binary that
is not there" rather than silently searching past it and reporting a different
one. That distinction is why this returns a pair rather than a path.
"""

from __future__ import annotations

import os
from pathlib import Path

from .provenance import REPO_ROOT

# build-bench first: it is what REORGANIZATION_PLAN.md section 0.4 builds.
# The rest are kept so existing trees and older instructions keep working.
BUILD_DIR_NAMES = (
    "build-bench",
    "build-benchmarks",
    "build",
    "build-gpu",
    "build-cuda",
    "build-release",
)

# Single-config generators put binaries in bin/; MSVC adds a per-config level.
CONFIG_SUBDIRS = ("", "Release", "RelWithDebInfo")

def candidate_names(base: str) -> list[str]:
    """`base` under both extensions, native spelling first."""
    return [base + ".exe", base] if os.name == "nt" else [base, base + ".exe"]

def env_names(base: str) -> list[str]:
    """Environment variables consulted for `base`, most specific first.

    `opennn_speed` -> `OPENNN_SPEED_BIN`, then the blanket `OPENNN_BIN`.
    """
    stem = base[len("opennn_"):] if base.startswith("opennn_") else base
    return [f"OPENNN_{stem.upper()}_BIN", "OPENNN_BIN"]

def search_dirs(root: Path | None = None) -> list[Path]:
    base = root or REPO_ROOT
    dirs: list[Path] = []
    for build in BUILD_DIR_NAMES:
        for config in CONFIG_SUBDIRS:
            dirs.append(base / build / "bin" / config if config else base / build / "bin")
    return dirs

def find_binary(
    base: str,
    *,
    aliases: tuple[str, ...] = (),
    extra_dirs: tuple[Path, ...] = (),
    root: Path | None = None,
) -> tuple[str, bool]:
    """Locate benchmark program `base`.

    Returns `(path, found)`. `found` is False when nothing matched, and also
    when an override names a path that does not exist -- in that case `path` is
    the override, so the caller can report the path the user actually asked for
    instead of a silently different one.
    """
    for name in list(env_names(base)) + list(aliases):
        override = os.environ.get(name)
        if override:
            return override, Path(override).exists()

    for directory in list(extra_dirs) + search_dirs(root):
        for name in candidate_names(base):
            candidate = directory / name
            if candidate.exists():
                return str(candidate), True

    # Nothing found: name the directory the plan says to build into, so the
    # error a runner prints points somewhere useful.
    fallback = (root or REPO_ROOT) / BUILD_DIR_NAMES[0] / "bin" / candidate_names(base)[0]
    return str(fallback), False
