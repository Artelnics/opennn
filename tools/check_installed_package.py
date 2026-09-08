#!/usr/bin/env python3
"""Check required release documentation and notices in an installation prefix."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--data-dir", default="share", help="CMAKE_INSTALL_DATAROOTDIR used at build time")
    parser.add_argument("--expect-cpu-target", choices=("NATIVE", "PORTABLE"))
    args = parser.parse_args()
    docs = args.prefix / args.data_dir / "doc/OpenNN"
    required = ["LICENSE.txt", "LICENSE-GPL-3.0.txt", "THIRD_PARTY_NOTICES.md", "MIGRATION.md", "CHANGELOG.md", "QUALITY.md", "RELEASE_SCOPE.json",
                "build-info.json", "libjpeg-turbo/LICENSE.md", "libjpeg-turbo/README.ijg", "zlib/LICENSE"]
    metadata = json.loads((docs / "build-info.json").read_text())
    if metadata["version"] != "9.0.0":
        raise ValueError(f"Unexpected release version: {metadata['version']}")
    if metadata.get("cpu_target") not in {"NATIVE", "PORTABLE"}:
        raise ValueError(f"Missing or invalid CPU target: {metadata.get('cpu_target')}")
    if args.expect_cpu_target and metadata["cpu_target"] != args.expect_cpu_target:
        raise ValueError(
            f"Expected CPU target {args.expect_cpu_target}, got {metadata['cpu_target']}"
        )
    scope = json.loads((docs / "RELEASE_SCOPE.json").read_text())
    if scope["binary_package"]["includes_example_assets"]:
        raise ValueError("Installed binary package may not include example assets")
    if any((args.prefix / name).exists() for name in ("examples", "data", "nn")):
        raise ValueError("Installed binary package unexpectedly contains example assets")
    if (args.prefix / "include/eigen3/Eigen").exists():
        required += ["Eigen/" + name for name in ("COPYING.MPL2", "COPYING.APACHE", "COPYING.BSD", "COPYING.MINPACK", "COPYING.README")]
    if metadata["cuda"] == "ON":
        required += ["cudnn-frontend/" + name for name in ("LICENSE.txt", "LICENSE-MIT.txt", "LICENSING.md", "NOTICE", "THIRD_PARTY_LICENSES.txt")]
    missing = [name for name in required if not (docs / name).is_file() or (docs / name).stat().st_size == 0]
    if missing:
        raise ValueError("Missing installed notices: " + ", ".join(missing))
    print(f"Verified {len(required)} installed documents; {metadata['compiler']} {metadata['compiler_version']}, CUDA={metadata['cuda']}")


if __name__ == "__main__":
    main()
