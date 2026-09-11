"""A passing label, missing values or a changed bundle cannot certify a result."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from consolidate_results import deployment, inventory, performance, preview, stats


def performance_fixture():
    launches = []
    for engine in ("opennn", "pytorch"):
        for rate in (90, 100, 110):
            launches.append(
                {
                    "engine": engine,
                    "returncode": 0,
                    "samples_per_sec": rate,
                    "instruments": {
                        "peak_mib": rate,
                        "energy_wh": rate / 100,
                        "energy_measurable": True,
                        "window_samples": 100,
                    },
                }
            )
    return {
        "configuration": {
            "family": "dense",
            "mode": "train",
            "device": "cpu",
            "rounds": 3,
            "batch": "32",
            "precision": "fp32",
        },
        "benchmark_id": "cpu-dense-train",
        "session_id": "test",
        "git": {"commit": "abcdef", "dirty": False},
        "shape_gate": {"agrees": True},
        "machine_quiet": {"quiet": True},
        "datasets": {"train": {"path": "train.csv", "bytes": 100}},
        "quality_gate": {"agrees": True, "accuracies": {"32": [float("nan")] * 6}},
        "summary": {
            e: {"median_samples_per_sec": 100, "peak_mib": 110, "energy_wh": 1.0}
            for e in ("opennn", "pytorch")
        },
        "launches": launches,
    }


class BenchmarkPublicationTest(unittest.TestCase):
    def test_stats_use_all_samples_and_sample_variation(self):
        result = stats([90, 100, 110])
        self.assertEqual(result["median"], 100)
        self.assertEqual(result["cv_percent"], 10)
        self.assertEqual((result["min"], result["max"], result["n"]), (90, 110, 3))
        self.assertIsNone(stats([100])["cv_percent"])

    def test_invalid_measurements_cannot_be_silently_dropped(self):
        for values in ([], [1, float("nan")], [float("inf")], [0], [-1]):
            with self.assertRaises(ValueError):
                stats(values)

    def test_passing_quality_flag_with_nan_does_not_pass_review(self):
        result = performance(
            performance_fixture(), {"path": "raw.json", "sha256": "test"}
        )
        self.assertFalse(result["checks"]["finite_quality_values_recorded"])
        self.assertFalse(result["checks"]["input_hashes_recorded"])
        self.assertFalse(result["checks"]["pytorch_throughput_variation"])
        self.assertEqual(result["status"], "Needs repeat")

    def test_incomplete_pair_fails_and_wrong_summary_stays_visible(self):
        raw = performance_fixture()
        raw["launches"].pop()
        with self.assertRaises(ValueError):
            performance(raw, {"path": "raw.json", "sha256": "test"})
        raw = performance_fixture()
        raw["summary"]["opennn"]["peak_mib"] = 100
        result = performance(raw, {"path": "raw.json", "sha256": "test"})
        self.assertFalse(result["checks"]["opennn_saved_summary_matches"])
        self.assertEqual(result["engines"]["opennn"]["memory"]["max"], 110)

    def test_deployment_merge_requires_equal_file_inventories(self):
        cases = []
        for engine in ("opennn", "pytorch"):
            for precision in ("fp32", "bf16"):
                cases.append(
                    {
                        "engine": engine,
                        "precision": precision,
                        "backend": "cuda",
                        "device": "cuda",
                        "family": "dense",
                        "status": "ok",
                        "unresolved_dependencies": [],
                        "deployment_files": [{"path": engine, "bytes": 100}],
                        "deployment_bytes": 100,
                    }
                )
        self.assertEqual(len(deployment({"cases": cases})), 1)
        changed = copy.deepcopy(cases)
        changed[1]["deployment_files"][0]["path"] = "different-runtime"
        with self.assertRaises(ValueError):
            deployment({"cases": changed})
        changed = copy.deepcopy(cases)
        changed[0]["deployment_bytes"] = 200
        with self.assertRaises(ValueError):
            deployment({"cases": changed})

    def test_preview_escapes_source_html(self):
        result = preview("# Review\n\n**Pending** `<script>alert(1)</script>`")
        self.assertIn("<strong>Pending</strong>", result)
        self.assertNotIn("<script>", result)
        self.assertIn("&lt;script&gt;", result)

    def test_duplicate_evidence_and_previous_reviews_are_not_new_observations(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            results = root / "results"
            results.mkdir()
            raw = json.dumps({"benchmark_id": "cpu-dense-infer", "label": "publish"})
            (results / "a.json").write_text(raw)
            (results / "pinned.json").write_text(raw)
            old = results / "old-review"
            old.mkdir()
            (old / "verification.json").write_text(
                json.dumps({"status": "review_only"})
            )
            (old / "generated.json").write_text(raw)
            rows = inventory(root, results / "new-review", {"results/pinned.json"})
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["copies"], 2)
            self.assertEqual(rows[0]["canonical_path"], "results/pinned.json")
            self.assertEqual(rows[0]["recorded_status"], "publish")
            self.assertEqual(rows[0]["role"], "selected_for_review")


if __name__ == "__main__":
    unittest.main()
