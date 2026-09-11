"""Runner regression checks without models, datasets or GPU workloads.

Run with: python -m unittest discover -s tests -p benchmark_runner_test.py
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import run as runner
import compare as benchmark_compare
from families import footprint


class BenchmarkRunnerTest(unittest.TestCase):
    @staticmethod
    def artifact(rate=100.0, memory=50.0, batch=128):
        return {
            "benchmark_id": "cpu-dense-infer",
            "shape_gate": {"agrees": True},
            "quality_gate": {"agrees": True},
            "machine_quiet": {"quiet": True},
            "summary": {"opennn": {
                "median_samples_per_sec": rate,
                "workload_mib": memory,
                "max_batch": batch,
            }},
        }

    def test_benchmark_regression_limits(self):
        baseline = self.artifact()
        self.assertEqual(benchmark_compare.compare(baseline, self.artifact(96, 52)), [])
        failures = benchmark_compare.compare(baseline, self.artifact(94, 56, 64))
        self.assertEqual(len(failures), 3)

    def test_benchmark_regression_rejects_invalid_runs(self):
        baseline = self.artifact()
        candidate = self.artifact()
        candidate["machine_quiet"]["quiet"] = False
        candidate["shape_gate"]["agrees"] = False
        self.assertEqual(len(benchmark_compare.compare(baseline, candidate)), 2)

    def test_failure_classification(self):
        for code, stdout, stderr, expected in (
            (0, "RESULT=OK", "CUDA out of memory recovered", None),
            (1, "RESULT=OOM", "", "oom"),
            (1, "RESULT=ERROR\nreason=bad allocation", "", "oom"),
            (1, "", "CUDA error: out of memory", "oom"),
            (1, "", "DefaultCPUAllocator: can't allocate memory", "oom"),
            (1, "", "MemoryError", "oom"),
            (1, "RESULT=ERROR", "file not found", "error"),
            (2, "", "usage error", "error"),
            (0, "fits=0", "", "error"),
            (-11, "RESULT=OOM", "", "crash"),
            (0xC0000005, "", "CUDA out of memory", "crash"),
        ):
            with self.subTest(code=code, stdout=stdout, stderr=stderr):
                self.assertEqual(runner.failure_kind(code, stdout, stderr), expected)

    def test_capacity_requires_success_then_oom(self):
        success = {"fits": True, "batch": 16, "failure_kind": None}
        failure = {"fits": False, "batch": 32, "failure_kind": "error"}
        self.assertFalse(runner.capacity_summary([success, failure])["frontier_valid"])
        failure["failure_kind"] = "oom"
        self.assertTrue(runner.capacity_summary([success, failure])["frontier_valid"])
        self.assertIsNone(runner.capacity_summary([failure])["max_batch"])
        self.assertFalse(runner.capacity_summary([failure])["frontier_valid"])
        self.assertFalse(runner.capacity_summary([success])["frontier_valid"])

    def test_footprint_missing_memory_is_null(self):
        outcome = {"fields": {"baseline_ram_mb": "null"},
                   "process_lifetime_seconds": 1.0, "process_time_scope": "process"}
        self.assertIsNone(runner.footprint_metrics(outcome)["baseline_ram_mib"])
        outcome["fields"]["baseline_ram_mb"] = "12.5"
        self.assertEqual(runner.footprint_metrics(outcome)["baseline_ram_mib"], 12.5)

    def test_unavailable_resident_query(self):
        with patch.object(footprint.os, "name", "posix"), patch("builtins.open", side_effect=OSError):
            self.assertIsNone(footprint.resident_mb())

    def test_current_resident_query(self):
        if sys.platform.startswith("linux") or sys.platform == "win32":
            self.assertGreater(footprint.resident_mb(), 0)

    def test_process_timing_excludes_observer_shutdown_and_drains_pipes(self):
        class Monitor:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def watch_rss(self, pid): pass
            def summary(self, start, end): return {}

        class Foreign:
            def __init__(self, *args): pass
            def __enter__(self): return self
            def __exit__(self, *args): time.sleep(0.2)
            def worst(self, *args): return {}

        with patch.object(runner, "Monitor", Monitor), patch.object(runner, "ForeignActivity", Foreign), \
                patch.object(runner, "cpu_pinning", return_value=([], {}, {})):
            started = time.perf_counter()
            result = runner.launch([sys.executable, "-c",
                "import sys; print('x\\n'*50000); sys.stderr.write('y\\n'*50000); print('RESULT=OK')"], False, "cpu")
            elapsed = time.perf_counter() - started
        self.assertTrue(result["fits"])
        self.assertGreaterEqual(elapsed - result["process_lifetime_seconds"], 0.19)
        self.assertEqual(result["wall_seconds"], round(result["process_lifetime_seconds"], 3))


if __name__ == "__main__":
    unittest.main()
