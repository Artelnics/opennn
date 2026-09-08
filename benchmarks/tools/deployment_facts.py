#!/usr/bin/env python3
"""Deployment facts: what OpenNN costs before it computes anything.

The twelve performance cells have `reports/*.md` behind them, and every figure
in those documents traces to an artifact under `results/`.  The claims that
live beside them -- how large the library is, how much you write to use it,
what has to be installed, what a deployment weighs -- had no such document, and
they drifted.  A deck carried a first-prediction time of 36 ms against the
569 ms the footprint family measures, and an application-line count that no
file in this repository supported.  Neither was reproducible, so neither was
checkable, so nobody caught them.

This tool exists so that cannot recur.  It emits the same shape of artifact the
runner emits, under one rule: **nothing here is estimated.**  A fact this host
cannot measure is recorded as null with a note saying why, exactly as
`footprint.md` leaves the split of its 202 MiB unmeasured rather than guessing.

Two groups, because they have different validity:

  source   read from the repository tree.  The same on every machine, so they
           can be taken on a laptop and quoted anywhere.

  machine  read from this host.  Only valid for the machine that will deploy,
           and absent entirely on a checkout with no build.

Every count carries its `method` string in the artifact.  That is the point:
the previous numbers were unfalsifiable because nobody could say what they had
counted.

Usage:

    python3 tools/deployment_facts.py
    python3 tools/deployment_facts.py --binary ../build-bench/bin/footprint_opennn
    python3 tools/deployment_facts.py --binary ... --pip /home/artelnics/benchenv/bin/pip
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import git_metadata  # noqa: E402

BENCHMARKS = Path(__file__).resolve().parent.parent
ROOT = BENCHMARKS.parent

SOURCE_SUFFIXES = (".cpp", ".h", ".cu", ".cuh")

# Library names that ship with the CUDA toolkit or with cuDNN, as opposed to
# the C++ runtime and the host maths libraries.  Used only to split the NEEDED
# list into two counts; the full list is recorded either way, so a reader who
# disagrees with the split can redo it.
CUDA_LIBRARY_STEMS = ("libcudart", "libcublas", "libcublasLt", "libcudnn",
                      "libnvrtc", "libcuda", "libnvToolsExt", "libcufft",
                      "libcusolver", "libcusparse", "libnccl", "libcurand")


def classify_cpp_lines(text: str) -> dict[str, int]:
    """Split C/C++ source into blank, comment and code lines.

    Block comments are tracked across lines, which a naive line-comment filter
    does not do -- a header with a forty-line banner comment would otherwise
    read as forty lines of code.  A line holding both code and a trailing
    comment counts as code, which is what every SLOC tool does.
    """
    blank = comment = code = 0
    in_block = False

    for raw in text.splitlines():
        line = raw.strip()

        if not line:
            blank += 1
            continue

        if in_block:
            comment += 1
            if "*/" in line:
                in_block = False
                # Code after the close on the same line still counts as code.
                if line.split("*/", 1)[1].strip():
                    comment -= 1
                    code += 1
            continue

        if line.startswith("//"):
            comment += 1
            continue

        if line.startswith("/*"):
            if "*/" not in line:
                in_block = True
            comment += 1
            continue

        code += 1

    return {"blank": blank, "comment": comment, "code": code}


def count_tree(directory: Path) -> dict[str, Any]:
    """Line counts over one directory of C/C++/CUDA source."""
    totals = {"blank": 0, "comment": 0, "code": 0}
    files = 0

    for path in sorted(directory.rglob("*")):
        if path.suffix not in SOURCE_SUFFIXES or not path.is_file():
            continue
        files += 1
        for key, value in classify_cpp_lines(path.read_text(errors="replace")).items():
            totals[key] += value

    return {
        "files": files,
        "total_lines": totals["blank"] + totals["comment"] + totals["code"],
        "code_lines": totals["code"],
        "comment_lines": totals["comment"],
        "blank_lines": totals["blank"],
        "suffixes": list(SOURCE_SUFFIXES),
        "method": ("every .cpp/.h/.cu/.cuh under the directory; code_lines "
                   "excludes blank lines and whole-line comments, block "
                   "comments tracked across lines"),
    }


def read_models(models_header: Path) -> dict[str, Any]:
    """The ready-made networks, read from the one header that declares them."""
    if not models_header.is_file():
        return {"count": None, "names": [], "note": f"{models_header} not found"}

    names = re.findall(r"^class\s+(\w+)\s+(?:final\s+)?:\s*public\s+NeuralNetwork",
                       models_header.read_text(errors="replace"), re.M)

    return {
        "count": len(names),
        "names": names,
        "method": f"classes deriving from NeuralNetwork in {models_header.name}",
    }


def read_layers(layers_directory: Path) -> dict[str, Any]:
    """The layer types, one header each, minus the abstract base."""
    if not layers_directory.is_dir():
        return {"count": None, "names": [], "note": f"{layers_directory} not found"}

    names = sorted(path.stem.replace("_layer", "")
                   for path in layers_directory.glob("*.h")
                   if path.stem != "layer")

    return {
        "count": len(names),
        "names": names,
        "method": ("one *.h per layer type in neural_network/layers, "
                   "excluding the abstract base layer.h"),
    }


def measure_examples(examples_directory: Path) -> dict[str, Any]:
    """What a complete program costs to write.

    Three counts per example, because the honest answer depends on what you
    call a line and the disagreement should be visible rather than settled by
    whoever made the slide:

      total_lines  everything in the file, banner and licence included
      code_lines   non-blank, non-comment
      statements   lines ending in ';' that are not #include -- the closest
                   thing to "decisions the programmer had to make"

    The benchmark drivers under families/ are deliberately NOT counted here.
    They hand-write the training loop on both engines so the two are measured
    doing identical work, which is the opposite of how an application uses the
    library: it is why those files run longer in C++ than in Python, and why
    they are the wrong evidence for this question.
    """
    if not examples_directory.is_dir():
        return {"count": None, "examples": [], "note": f"{examples_directory} not found"}

    examples = []

    for main in sorted(examples_directory.glob("*/main.cpp")):
        text = main.read_text(errors="replace")
        counts = classify_cpp_lines(text)

        statements = 0
        for raw in text.splitlines():
            line = raw.strip()
            if line.endswith(";") and not line.startswith("#include") \
                    and not line.startswith("//"):
                statements += 1

        examples.append({
            "name": main.parent.name,
            "total_lines": counts["blank"] + counts["comment"] + counts["code"],
            "code_lines": counts["code"],
            "statements": statements,
        })

    return {
        "count": len(examples),
        "examples": examples,
        "method": ("examples/*/main.cpp; statements are lines ending in ';' "
                   "that are not #include; families/ drivers excluded on "
                   "purpose, see the docstring"),
    }


def classify_python_lines(text: str) -> dict[str, int]:
    """Blank, comment and code lines for Python, module docstring excluded.

    The docstring is dropped rather than counted as comment, because in these
    files it is prose about the comparison and not part of either program.
    """
    body = text.split('"""', 2)[2] if text.lstrip().startswith('"""') else text

    blank = comment = code = 0
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            blank += 1
        elif line.startswith("#"):
            comment += 1
        else:
            code += 1

    return {"blank": blank, "comment": comment, "code": code}


def measure_application_lines(opennn_main: Path, pytorch_main: Path) -> dict[str, Any]:
    """The same program in both libraries, counted the same way.

    The pair exists so this claim can be argued with rather than believed: both
    files are in the tree, they do the same job on the same data, and the
    PyTorch one was written to be as short as honesty allows -- no argument
    parsing, no logging, nothing the OpenNN example does not also have.

    `statements` is the figure to quote.  `code_lines` flatters C++ badly in
    both directions: it counts lone braces, `#include` lines and the try/catch
    frame as code, so the two languages are not comparable on it.

    What the difference is, mechanically: `TrainingStrategy` is the epoch loop
    and `TestingAnalysis` is the binary-classification report, so on the
    PyTorch side those become the split, the scaling, the loop with its
    zero_grad/backward/step, and four counted confusion-matrix terms.
    """
    result: dict[str, Any] = {
        "method": ("the same task written in both libraries; statements are "
                   "non-blank non-comment lines that are not imports (Python) "
                   "or that end in ';' and are not #include (C++)"),
    }

    for engine, path, classify in (("opennn", opennn_main, classify_cpp_lines),
                                   ("pytorch", pytorch_main, classify_python_lines)):
        if not path.is_file():
            result[engine] = {"statements": None, "note": f"{path} not found"}
            continue

        text = path.read_text(errors="replace")
        counts = classify(text)

        # The Python file opens with prose about the comparison; it is neither
        # program and must not reach the statement count.
        body = text
        if engine == "pytorch" and text.lstrip().startswith('"""'):
            body = text.split('"""', 2)[2]

        statements = 0
        for raw in body.splitlines():
            line = raw.strip()
            if not line or line.startswith("//") or line.startswith("#include"):
                continue
            if engine == "opennn":
                statements += 1 if line.endswith(";") else 0
            elif not line.startswith(("#", "import ", "from ")):
                statements += 1

        result[engine] = {
            "file": str(path.relative_to(ROOT)),
            "statements": statements,
            "code_lines": counts["code"],
        }

    return result


def read_needed_libraries(binary: Path | None) -> dict[str, Any]:
    """The shared objects a built binary asks the loader for.

    This is the honest version of "how many dependencies": not a package count
    but the NEEDED list in the ELF header, which is what actually has to be
    present on the target machine.
    """
    if binary is None:
        return {"count": None, "note": "no --binary given"}

    if not binary.is_file():
        return {"count": None, "note": f"{binary} not found"}

    if not shutil.which("readelf"):
        return {"count": None, "note": "readelf not on PATH (not a Linux host?)"}

    output = subprocess.run(["readelf", "-d", str(binary)],
                            capture_output=True, text=True, check=False).stdout

    names = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", output)
    cuda = [n for n in names if any(n.startswith(s) for s in CUDA_LIBRARY_STEMS)]

    return {
        "count": len(names),
        "cuda_count": len(cuda),
        "names": names,
        "cuda_names": cuda,
        "binary": str(binary),
        "method": "readelf -d, NEEDED entries; cuda_count by library-name prefix",
    }


def read_packages(pip: Path | None, requirement: str) -> dict[str, Any]:
    """What one `pip install` brings in, resolved without installing anything.

    The obvious method -- count what is installed in the environment that runs
    the PyTorch side -- is wrong here, and measurably so: `benchenv` holds 89
    packages because it carries both engines plus TensorFlow and the Intel
    oneAPI runtimes, and reporting that as PyTorch's cost would overstate it
    threefold.  A resolve answers the question actually being asked, which is
    what a machine has to take on to run PyTorch.

    The version is pinned by the caller so the figure matches the engine the
    twelve cells were measured against; an unpinned resolve reports whatever
    the index offers today, which moves.  `--ignore-installed` makes the answer
    independent of what the environment already has.
    """
    if pip is None:
        return {"count": None, "note": "no --pip given"}

    if not Path(pip).is_file():
        return {"count": None, "note": f"{pip} not found"}

    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "resolve.json"
        result = subprocess.run(
            [str(pip), "install", requirement, "--dry-run", "--ignore-installed",
             "--quiet", "--report", str(report)],
            capture_output=True, text=True, check=False)

        if result.returncode != 0 or not report.is_file():
            return {"count": None,
                    "note": f"resolve failed: {result.stderr.strip()[:200]}"}

        resolved = json.loads(report.read_text())

    versions = {item["metadata"]["name"]: item["metadata"]["version"]
                for item in resolved["install"]}
    names = sorted(versions)
    nvidia = [name for name in names if name.lower().startswith("nvidia")]

    return {
        "count": len(names),
        "nvidia_count": len(nvidia),
        "names": names,
        "nvidia_names": nvidia,
        "requirement": requirement,
        "resolved_version": versions.get(requirement.split("==")[0]),
        "resolver": str(pip),
        "method": ("pip install <requirement> --dry-run --ignore-installed, "
                   "counting the resolved install set; NOT a count of an "
                   "installed environment, see the docstring"),
    }


def measure_deployment_bytes(binary: Path | None) -> dict[str, Any]:
    """Disk cost of the binary plus every shared object it resolves to.

    `ldd` resolves what the loader would map, so this is the deployed set for
    this host's library layout.  Each resolved path is counted once: the
    symlink farms in a CUDA install would otherwise be counted several times
    over, which is how a deployment figure gets inflated without anyone lying.
    """
    if binary is None or not Path(binary).is_file():
        return {"bytes": None, "note": "no --binary given, or not found"}

    if not shutil.which("ldd"):
        return {"bytes": None, "note": "ldd not on PATH (not a Linux host?)"}

    output = subprocess.run(["ldd", str(binary)],
                            capture_output=True, text=True, check=False).stdout

    paths = {Path(binary).resolve()}
    for match in re.finditer(r"=>\s+(/\S+)", output):
        paths.add(Path(match.group(1)).resolve())

    total = 0
    missing = []
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            missing.append(str(path))

    return {
        "bytes": total,
        "mib": round(total / (1024 * 1024), 1),
        "objects": len(paths),
        "unreadable": missing,
        "method": ("binary plus every path ldd resolves, deduplicated by "
                   "realpath, sizes from stat"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--binary", type=Path, default=None,
                        help="a built OpenNN executable, for the linked-library "
                             "and disk facts")
    parser.add_argument("--pip", type=Path, default=None,
                        help="pip of the environment that runs the PyTorch side")
    parser.add_argument("--requirement", default="torch==2.13.0",
                        help="what to resolve for the package count; pinned by "
                             "default to the engine the twelve cells measured")
    parser.add_argument("--label", default=None, help="artifact label")
    parser.add_argument("--out", type=Path, default=BENCHMARKS / "results",
                        help="where the artifact lands")
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    artifact: dict[str, Any] = {
        "schema_version": 1,
        "benchmark_id": "deployment-facts",
        "run_id": run_id,
        "label": args.label,
        "git": git_metadata(ROOT),
        "host": {"platform": sys.platform, "python": sys.version.split()[0]},
        "source": {
            "library": count_tree(ROOT / "opennn"),
            "models": read_models(ROOT / "opennn" / "models" / "models.h"),
            "layers": read_layers(ROOT / "opennn" / "neural_network" / "layers"),
            "examples": measure_examples(ROOT / "examples"),
            "application_lines": measure_application_lines(
                ROOT / "examples" / "breast_cancer" / "main.cpp",
                BENCHMARKS / "application" / "breast_cancer_pytorch.py"),
        },
        "machine": {
            "linked_libraries": read_needed_libraries(args.binary),
            "packages": read_packages(args.pip, args.requirement),
            "deployment_size": measure_deployment_bytes(args.binary),
        },
    }

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"deployment-facts{'-' + args.label if args.label else ''}-{run_id}.json"
    path = args.out / name
    path.write_text(json.dumps(artifact, indent=2, default=str))

    source = artifact["source"]
    print(f"  library    {source['library']['code_lines']:>8,} code lines "
          f"({source['library']['total_lines']:,} total, "
          f"{source['library']['files']} files)")
    print(f"  models     {source['models']['count']:>8}")
    print(f"  layers     {source['layers']['count']:>8}")
    print(f"  examples   {source['examples']['count']:>8}")

    application = source["application_lines"]
    if application["opennn"].get("statements") and application["pytorch"].get("statements"):
        print(f"  same task  {application['opennn']['statements']:>8} statements "
              f"in OpenNN against {application['pytorch']['statements']} in PyTorch")

    for key, block in artifact["machine"].items():
        if block.get("note"):
            print(f"  {key:<10} not measured: {block['note']}")

    print(f"\n  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
