"""Deployment files for OpenNN applications and standard PyTorch Python installs.

Primary: OpenNN executable + exercised native dependency closure; Python script
+ interpreter + complete installed dependency packages + standard library +
exercised external native dependencies. Exclude OS/driver files and generated
bytecode/tuning caches. This is explicitly the standard Python installation,
not a claim about its minimum possible pruned size. Also report observed runtime
files under one consistent tracing method for both engines.
"""

import argparse, json, os, re, subprocess, sys, datetime, platform, csv
import hashlib
from pathlib import Path
from application_startup import make_cases, THREAD_ENV, EXPECTED
from common import cpu_state, gpu_state
from experiment import git_metadata, result_directory
from application_tables import write_table

OS_STEMS = (
    "ld-linux",
    "libc.so",
    "libm.so",
    "libdl.so",
    "librt.so",
    "libpthread.so",
    "libutil.so",
    "libresolv.so",
    "libnss_",
)
DRIVER_STEMS = (
    "libcuda.so",
    "libnvidia-",
    "libnvcuvid.so",
    "libdxcore.so",
    "libd3d12.so",
)


def included(path):
    return not path.name.startswith(OS_STEMS + DRIVER_STEMS) and "/wsl/" not in str(
        path
    )


def files_below(root):
    if not root.exists():
        return set()
    return {
        p.resolve()
        for p in root.rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix not in [".pyc", ".pyo"]
        and "site-packages" not in p.relative_to(root).parts
        and "dist-packages" not in p.relative_to(root).parts
    }


def native_closure(paths, env):
    todo = list(paths)
    seen = set()
    unresolved = []
    env = env.copy()
    env.pop("LD_DEBUG", None)
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            with path.open("rb") as stream:
                if stream.read(4) != b"\x7fELF":
                    continue
        except OSError:
            continue
        p = subprocess.run(["ldd", str(path)], capture_output=True, text=True, env=env)
        for line in p.stdout.splitlines():
            if "not found" in line:
                unresolved.append({"file": str(path), "dependency": line.strip()})
            match = re.search(r"(?:=>\s+)?(/.+?)\s+\(0x", line)
            if match:
                resolved = Path(match.group(1)).resolve()
                if resolved.is_file() and resolved not in seen:
                    todo.append(resolved)
    return seen, unresolved


def run_case(case, out):
    name = case["id"].removesuffix("-reused")
    folder = out / name
    folder.mkdir()
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(
            ("OPENNN_", "CUDA_", "CUDNN_", "TORCH_", "TRITON_", "MKL_", "OMP_", "GOMP_")
        ) or key in ["LD_DEBUG", "LD_PRELOAD"]:
            env.pop(key, None)
    env.update(THREAD_ENV)
    env.update(case.get("env", {}))
    env["LD_DEBUG"] = "files"
    env["OPENNN_LT_PLAN_CACHE_DIR"] = str(folder / "plans")
    binary = Path(case["binary"]).absolute()
    binary_sha256 = hashlib.sha256(binary.read_bytes()).hexdigest()
    cmd = [
        str(binary),
        *[v.format(family=case["family"]) for v in case.get("prefix_args", [])],
        case["device"],
        case["precision"],
    ]
    if case["engine"] == "pytorch":
        cmd.append(str(folder / "python-audit"))
    with (folder / "loader.log").open("w") as stream:
        p = subprocess.run(
            cmd, env=env, stdout=subprocess.PIPE, stderr=stream, text=True, timeout=180
        )
    (folder / "stdout.txt").write_text(p.stdout)
    markers = [
        json.loads(l[len("STARTUP_READY ") :])
        for l in p.stdout.splitlines()
        if l.startswith("STARTUP_READY ")
    ]
    if p.returncode or len(markers) != 1:
        raise RuntimeError(f"{name} failed; inspect {folder}")
    marker = markers[0]
    if any(marker.get(k) != case[k] for k in ("engine", "device", "precision")):
        raise RuntimeError("Engine/device/precision gate failed")
    if case["engine"] == "pytorch" and (
        marker.get("interface") != "python"
        or Path(marker["python_prefix"]) != binary.parent.parent
    ):
        raise RuntimeError("Expected the configured PyTorch Python environment")
    if (marker["parameters"], marker["output_values"]) != EXPECTED[case["family"]]:
        raise RuntimeError("Shape gate failed")
    paths = {binary.resolve()}
    for value in re.findall(
        r"calling init:\s*(/[^\n]+)", (folder / "loader.log").read_text()
    ):
        path = Path(value.strip())
        if path.is_file():
            paths.add(path.resolve())
    installed_extra = set()
    python_runtime = set()
    package_files = set()
    if case["engine"] == "pytorch":
        audit = folder / "python-audit"
        for line in (audit / "maps.txt").read_text().splitlines():
            parts = line.split(None, 5)
            if len(parts) == 6 and parts[5].startswith("/"):
                path = Path(parts[5].removesuffix(" (deleted)"))
                if path.is_file():
                    paths.add(path.resolve())
        for module in json.loads((audit / "modules.json").read_text()):
            paths.add(Path(module["file"]).resolve())
        script = Path(case["prefix_args"][1])
        paths.add(script.resolve())
        runtime = json.loads((audit / "python-runtime.json").read_text())
        prefix = Path(runtime["prefix"])
        # Full venv, including package metadata and normal entry points, without
        # generated bytecode. No pip installer was included in this runtime venv.
        installed_extra = {
            p.resolve()
            for p in prefix.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in [".pyc", ".pyo"]
        }
        site = Path(runtime["paths"]["purelib"])
        package_files = {
            p.resolve()
            for p in site.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and p.suffix not in [".pyc", ".pyo"]
        }
        python_runtime = files_below(Path(runtime["paths"]["stdlib"])) | {
            binary.resolve()
        }
        installed_extra |= python_runtime | {script.resolve()}
    observed, missing = native_closure(paths, env)
    # Some loader-resolved dependencies use per-library RPATH/preloading rather
    # than LD_LIBRARY_PATH. Report unresolved ldd entries; don't silently ignore.
    if missing:
        raise RuntimeError(f"{name}: unresolved dependencies {missing[:8]}")
    observed = {p for p in observed if included(p)}
    installed = {p for p in observed | installed_extra if included(p)}

    def manifest(paths):
        return [{"path": str(p), "bytes": p.stat().st_size} for p in sorted(paths)]

    result = {
        k: case[k] for k in ["engine", "backend", "device", "family", "precision"]
    }
    result.update(
        status="ok",
        command=cmd,
        executable_sha256=binary_sha256,
        application_sha256=hashlib.sha256(script.read_bytes()).hexdigest()
        if case["engine"] == "pytorch"
        else binary_sha256,
        marker=marker,
        observed_files=manifest(observed),
        deployment_files=manifest(installed),
        deployment_bytes=sum(p.stat().st_size for p in installed),
        observed_bytes=sum(p.stat().st_size for p in observed),
        complete_package_bytes=sum(p.stat().st_size for p in package_files),
        python_standard_library_and_executable_bytes=sum(
            p.stat().st_size for p in python_runtime
        ),
        unresolved_dependencies=missing,
    )
    result["python_packages"] = (
        json.loads((folder / "python-audit/packages.json").read_text())
        if case["engine"] == "pytorch"
        else []
    )
    result["python_package_count"] = len(result["python_packages"])
    (folder / "measurement.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=["deployment"], default="deployment")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux"):
        parser.error("Deployment tracing currently requires Linux/WSL")
    config = json.loads(args.config.read_text())
    out = result_directory("deployment", args.out)
    provenance = {
        "git": git_metadata(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "host": platform.platform(),
        "cpu": cpu_state(),
        "gpu": gpu_state()
        if any(g["device"] == "cuda" for g in config["groups"])
        else None,
    }
    rows = []
    for case in make_cases(config):
        if case["cache"] != "reused":
            continue
        result = run_case(case, out)
        rows.append(result)
        print(
            case["id"],
            result["status"],
            "deployment MB",
            round(result["deployment_bytes"] / 1e6, 3),
            "observed MB",
            round(result["observed_bytes"] / 1e6, 3),
            flush=True,
        )
    summary = {
        **provenance,
        "status": "local_diagnostic",
        "protocol": "deployment-python-v2",
        "method": __doc__,
        "config": config,
        "cases": rows,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    pairs = []
    for row in rows:
        if row["engine"] != "opennn":
            continue
        peer = next(
            r
            for r in rows
            if r["engine"] == "pytorch"
            and all(r[k] == row[k] for k in ["family", "device", "precision"])
        )
        pairs.append(
            {k: row[k] for k in ["backend", "family", "precision"]}
            | {
                "opennn_MB": row["deployment_bytes"] / 1e6,
                "pytorch_python_MB": peer["deployment_bytes"] / 1e6,
                "opennn_percent_of_pytorch": 100
                * row["deployment_bytes"]
                / peer["deployment_bytes"],
                "opennn_observed_MB": row["observed_bytes"] / 1e6,
                "pytorch_observed_MB": peer["observed_bytes"] / 1e6,
            }
        )
    with (out / "comparisons.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    print("Completed", len(rows), "application configurations.", flush=True)
    write_table(out, "deployment", pairs)


if __name__ == "__main__":
    main()
