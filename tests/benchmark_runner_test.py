"""Runner regression checks without models, datasets or GPU workloads.

Run with: python -m unittest discover -s tests -p benchmark_runner_test.py
"""

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import run as runner
import compare as benchmark_compare
import common
import deployment_facts
from experiment import dispatch
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

    def test_failed_git_queries_preserve_unknown_provenance(self):
        for error in (FileNotFoundError("git"),
                      subprocess.TimeoutExpired("git", 30),
                      subprocess.CalledProcessError(128, "git")):
            for successful_queries in ([], [""], ["", "commit"]):
                with self.subTest(error=error, successful_queries=successful_queries), \
                        patch.object(common.subprocess, "check_output",
                                     side_effect=[*successful_queries, error]):
                    metadata = common.git_metadata()
                self.assertIsNone(metadata["dirty"])
                self.assertIsNone(metadata["commit"])
                self.assertIn("error", metadata)

    def test_git_provenance_preserves_root_and_status(self):
        root = Path("checkout with spaces")
        for status, dirty in (("", False), (" M README.md\n?? generated.txt", True)):
            with self.subTest(status=status), \
                    patch.object(common.subprocess, "check_output",
                                 side_effect=[status, "commit\n", "dev\n"]) as query:
                metadata = common.git_metadata(root)
            self.assertEqual(metadata["dirty"], dirty)
            self.assertEqual(metadata["dirty_count"], len(status.splitlines()))
            self.assertEqual(metadata["branch"], "dev")
            self.assertEqual(query.call_args_list[0].args[0],
                             ["git", "-C", str(root), "status", "--porcelain"])

    def test_wsl_git_conversion_failure_is_unknown(self):
        root = "/mnt/c/checkout with spaces"
        with patch.object(common.shutil, "which", return_value="/usr/bin/git.exe"), \
                patch.object(common.subprocess, "check_output", side_effect=[
                    "C:\\checkout with spaces\n", "", "commit", "dev"
                ]) as query:
            self.assertFalse(common.git_metadata(root)["dirty"])
            self.assertEqual(query.call_args_list[1].args[0],
                             ["/usr/bin/git.exe", "-C", "C:\\checkout with spaces",
                              "status", "--porcelain"])
        with patch.object(common.shutil, "which", return_value="/usr/bin/git.exe"), \
                patch.object(common.subprocess, "check_output",
                             side_effect=subprocess.CalledProcessError(1, "wslpath")):
            self.assertIsNone(common.git_metadata(root)["dirty"])

    def test_unknown_provenance_always_uses_scratch(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(common, "RESULTS", Path(folder)), \
                patch.object(common, "git_metadata", return_value={"dirty": False}) as query, \
                patch.object(common, "clocks_locked", return_value=True):
            for dirty in (None, True, False):
                for busy in (False, True):
                    with self.subTest(dirty=dirty, busy=busy):
                        expected = Path(folder)
                        if dirty is not False or busy:
                            expected /= "scratch"
                        self.assertEqual(common.result_destination(dirty, "cpu", busy), expected)
            query.assert_not_called()

    def test_specialized_dispatch_uses_only_the_selected_family(self):
        for arguments in (["--family", "qwen"], ["--family=qwen"],
                          ["--family", "dense", "--family=qwen"]):
            with self.subTest(arguments=arguments), patch("importlib.import_module") as imported:
                imported.return_value.main.return_value = 7
                self.assertEqual(dispatch(arguments), 7)
                imported.assert_called_once_with("families.qwen")
                imported.return_value.main.assert_called_once_with(arguments)
        for arguments in (["--family", "dense", "--label", "qwen"],
                          ["--family=qwen", "--family=dense"], ["--label", "qwen"]):
            with self.subTest(arguments=arguments), patch("importlib.import_module") as imported:
                self.assertIsNone(dispatch(arguments))
                imported.assert_not_called()

    def test_qwen_label_keeps_standard_runner_and_help_lists_qwen(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run.py", "--family", "dense", "--label", "qwen", "--help"]), \
                contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            runner.main()
        self.assertEqual(stopped.exception.code, 0)
        self.assertIn("--epochs", output.getvalue())
        self.assertIn("qwen", output.getvalue().split("--family", 1)[1].split("}", 1)[0])

    def test_deployment_facts_routes_the_recorded_git_snapshot(self):
        probes = {
            "count_tree": {"code_lines": 1, "total_lines": 1, "files": 1},
            "read_models": {"count": 0},
            "read_layers": {"count": 0},
            "measure_examples": {"count": 0},
            "measure_application_lines": {"opennn": {}, "pytorch": {}},
            "read_needed_libraries": {},
            "read_packages": {},
            "measure_deployment_bytes": {},
        }
        for dirty in (None, True, False):
            for override in (False, True):
                with self.subTest(dirty=dirty, override=override), \
                        tempfile.TemporaryDirectory() as folder:
                    root = Path(folder)
                    arguments = ["deployment_facts.py"]
                    if override:
                        arguments += ["--out", str(root / "explicit")]
                    metadata = {"commit": "recorded", "dirty": dirty}
                    with patch.object(sys, "argv", arguments), \
                            patch.object(common, "RESULTS", root / "results"), \
                            patch.object(deployment_facts, "git_metadata", return_value=metadata) as query, \
                            patch.multiple(deployment_facts, **{
                                name: Mock(return_value=value) for name, value in probes.items()
                            }), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(deployment_facts.main(), 0)
                    query.assert_called_once_with(deployment_facts.ROOT)
                    destination = root / "explicit" if override else root / "results"
                    if not override and dirty is not False:
                        destination /= "scratch"
                    artifacts = list(destination.glob("*.json"))
                    self.assertEqual(len(artifacts), 1)
                    self.assertEqual(json.loads(artifacts[0].read_text())["git"], metadata)

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
