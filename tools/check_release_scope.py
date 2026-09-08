#!/usr/bin/env python3
"""Enforce which OpenNN artifacts may be described as release-ready."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-kind", choices=("binary", "full-source"), required=True)
    args = parser.parse_args()

    scope = json.loads((ROOT / "RELEASE_SCOPE.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "datasets.manifest.json").read_text(encoding="utf-8"))
    unresolved = sorted(
        item["path"] for item in manifest["datasets"]
        if not item["redistribution_cleared"]
    )

    if scope.get("schema_version") != 1:
        raise ValueError("Unsupported RELEASE_SCOPE.json schema")
    if scope["historical_model_compatibility"]["claim"] != "not_claimed":
        raise ValueError("Historical model compatibility must remain unclaimed without a complete 8.x fixture")
    if scope["neural_designer_compatibility"]["claim"] != "not_evaluated":
        raise ValueError("Neural Designer must remain outside this review")

    if args.package_kind == "binary":
        if not scope["binary_package"]["publishable"]:
            raise ValueError("Binary package is not approved by the release scope")
        if scope["binary_package"]["includes_example_assets"]:
            raise ValueError("Binary package scope may not include unresolved example assets")
        print(f"Binary package scope verified; {len(unresolved)} unresolved asset groups are excluded.")
        return 0

    if unresolved or not scope["full_source_archive"]["publishable"]:
        print("Full source publication is blocked by unresolved redistribution records:")
        for path in unresolved:
            print(f"  {path}")
        return 1

    print("Full source publication scope verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
