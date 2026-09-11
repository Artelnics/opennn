"""Build application probes and isolated PyTorch environments on Linux/WSL.

Use Python 3.12 with venv support. Builds and downloaded packages stay in a new
external work directory. The checkout being prepared supplies the native code.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiment import git_metadata
from quality_data import digest

ROOT = Path(__file__).resolve().parents[2]


def run(command, log):
    with log.open("w") as stream:
        result = subprocess.run(
            [str(value) for value in command], stdout=stream, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"Command failed; see {log}\n{log.read_text()[-3000:]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument(
        "--backends", default="cpu-eigen", help="Comma-separated cpu-eigen,cpu-mkl,cuda"
    )
    parser.add_argument("--mkl-root", type=Path)
    parser.add_argument("--onednn-root", type=Path)
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--cudnn-include", type=Path)
    parser.add_argument("--cudnn-library", type=Path)
    parser.add_argument("--gpu-arch", default="native")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--eigen-source", type=Path)
    parser.add_argument("--cudnn-frontend-source", type=Path)
    args = parser.parse_args(argv)
    if not sys.platform.startswith("linux") or sys.version_info[:2] != (3, 12):
        parser.error("Use Python 3.12 inside Linux/WSL")
    backends = args.backends.split(",")
    if (
        not backends
        or len(set(backends)) != len(backends)
        or any(b not in ("cpu-eigen", "cpu-mkl", "cuda") for b in backends)
    ):
        parser.error("Choose distinct supported backends")
    if args.jobs < 1:
        parser.error("Jobs must be positive")
    work = args.work.resolve()
    if work.is_relative_to(ROOT):
        parser.error("Builds and environments must stay outside the checkout")
    if "cpu-mkl" in backends and not (args.mkl_root and args.onednn_root):
        parser.error("cpu-mkl requires --mkl-root and --onednn-root")
    if "cuda" in backends and not (
        args.cuda_root and args.cudnn_include and args.cudnn_library
    ):
        parser.error("cuda requires --cuda-root, --cudnn-include and --cudnn-library")
    work.mkdir(parents=True, exist_ok=False)
    groups = []
    families = ("dense", "lstm", "cnn", "transformer")
    binaries = [f"{f}_application_opennn" for f in families]
    for backend in backends:
        build = work / backend
        flags = [
            "cmake",
            "-S",
            ROOT,
            "-B",
            build,
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DOpenNN_BUILD_TESTS=OFF",
            "-DOpenNN_BUILD_EXAMPLES=OFF",
            "-DOpenNN_BUILD_BENCHMARKS=ON",
            "-DOpenNN_BUILD_SHARED=OFF",
            "-DOpenNN_ENABLE_LTO=ON",
            f"-DOpenNN_DISABLE_CUDA={'OFF' if backend == 'cuda' else 'ON'}",
            f"-DOpenNN_ENABLE_MKL={'ON' if backend == 'cpu-mkl' else 'OFF'}",
            f"-DOpenNN_ENABLE_ONEDNN={'ON' if backend == 'cpu-mkl' else 'OFF'}",
        ]
        libraries = []
        if backend == "cpu-mkl":
            flags += [
                f"-DOpenNN_MKL_ROOT={args.mkl_root.resolve()}",
                f"-DOpenNN_ONEDNN_ROOT={args.onednn_root.resolve()}",
            ]
            libraries = [
                str(p / part)
                for p in (args.mkl_root.resolve(), args.onednn_root.resolve())
                for part in ("lib", "lib/intel64")
            ]
        if backend == "cuda":
            flags += [
                "-DOpenNN_REQUIRE_CUDA=ON",
                f"-DCMAKE_CUDA_COMPILER={args.cuda_root.resolve() / 'bin/nvcc'}",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.gpu_arch}",
                f"-DCUDNN_INCLUDE_DIR={args.cudnn_include.resolve()}",
                f"-DCUDNN_LIBRARY={args.cudnn_library.resolve()}",
            ]
            libraries = [
                str(args.cudnn_library.resolve().parent),
                str(args.cuda_root.resolve() / "targets/x86_64-linux/lib"),
            ]
        if args.eigen_source:
            flags.append(
                f"-DFETCHCONTENT_SOURCE_DIR_EIGEN={args.eigen_source.resolve()}"
            )
        if args.cudnn_frontend_source:
            flags.append(
                f"-DFETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND={args.cudnn_frontend_source.resolve()}"
            )
        print("Building", backend, flush=True)
        run(flags, work / f"{backend}-configure.log")
        run(
            ["cmake", "--build", build, "--parallel", args.jobs, "--target", *binaries],
            work / f"{backend}-build.log",
        )
        groups.append(
            {
                "engine": "opennn",
                "backend": backend,
                "device": "cuda" if backend == "cuda" else "cpu",
                "binary_pattern": str(build / "bin/{family}_application_opennn"),
                "env": {"LD_LIBRARY_PATH": ":".join(libraries)},
            }
        )
    for device in sorted({g["device"] for g in groups}):
        environment = work / f"python-{device}"
        python = environment / "bin/python"
        run([sys.executable, "-m", "venv", environment], work / f"{device}-venv.log")
        requirements = (
            ROOT / f"benchmarks/manifests/application-{device}-requirements.txt"
        )
        index = "cu130" if device == "cuda" else "cpu"
        run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-compile",
                "--extra-index-url",
                f"https://download.pytorch.org/whl/{index}",
                "-r",
                requirements,
            ],
            work / f"{device}-install.log",
        )
        run([python, "-m", "pip", "check"], work / f"{device}-check.log")
        # This is a newly created, harness-owned environment; the installer is
        # excluded from the deployed application's declared dependency closure.
        run(
            [python, "-m", "pip", "uninstall", "--yes", "pip"],
            work / f"{device}-remove-installer.log",
        )
        groups.append(
            {
                "engine": "pytorch",
                "backend": device,
                "device": device,
                "binary_pattern": str(python),
                "prefix_args": [
                    "-I",
                    str(ROOT / "benchmarks/families/application.py"),
                    "{family}",
                ],
                "env": {"LD_LIBRARY_PATH": "", "PYTHONNOUSERSITE": "1"},
            }
        )
    config = {
        "groups": groups,
        "interface": "OpenNN C++ vs PyTorch Python API",
        "source": git_metadata(),
        "application_sha256": digest(ROOT / "benchmarks/families/application.py"),
        "build_arguments": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    (work / "applications.json").write_text(json.dumps(config, indent=2))
    print("Configuration:", work / "applications.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
