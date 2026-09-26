"""Quality runs must retain artifact identity and ignore inherited diagnostics."""

from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from quality_data import digest, prepare_one, read_array
import quality_runner


class BenchmarkQualityRunnerTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.root = Path(temporary.name).resolve()
        self.addCleanup(temporary.cleanup)

    def test_initialization_identity_detects_manifest_tensor_and_path_changes(self):
        tensor = self.root / "weight.bin"
        np.asarray([1, 2], dtype="<f4").tofile(tensor)
        path = self.root / "manifest.json"
        artifact = {"tensors": [{"name": "weight", "file": tensor.name,
                                 "sha256": digest(tensor)}]}
        path.write_text(json.dumps(artifact))
        identity = {"manifest": str(path), "manifest_sha256": digest(path)}
        quality_runner.verify_initialization_files(identity)
        original_manifest, original_tensor = path.read_bytes(), tensor.read_bytes()
        path.write_bytes(original_manifest + b" ")
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            quality_runner.verify_initialization_files(identity)
        path.write_bytes(original_manifest)
        tensor.write_bytes(original_tensor[:-1] + bytes([original_tensor[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "tensor changed"):
            quality_runner.verify_initialization_files(identity)
        tensor.write_bytes(original_tensor)
        artifact["tensors"][0]["file"] = "../outside.bin"
        path.write_text(json.dumps(artifact))
        identity["manifest_sha256"] = digest(path)
        with self.assertRaisesRegex(ValueError, "tensor changed"):
            quality_runner.verify_initialization_files(identity)

    def run_mocked_pair(self, diagnostic_only=False, seeds="42", during_process=None,
                        extra_args=None, history_text="epoch,training_loss\n1,0.5\n"):
        manifest_path = prepare_one("dense", self.root, self.root / "data", smoke=True)
        manifest = json.loads(manifest_path.read_text())
        binary = self.root / "quality_opennn"
        binary.write_bytes(b"mock executable; never launched")
        output = self.root / "run"
        output.mkdir()
        environments = []
        inherited = {
            "OPENNN_QUALITY_DIAGNOSTIC_STEPS": "99",
            "OPENNN_QUALITY_INITIALIZATION": "unrelated-initialization",
            "OPENNN_QUALITY_INITIAL_PREDICTIONS": "1",
            "OPENNN_QUALITY_EXPORT_INITIALIZATION": "1",
            "OPENNN_QUALITY_LOAD_CHECKPOINT": "unrelated-checkpoint",
            "OPENNN_QUALITY_ADAM_EPSILON": "0.5",
            "OPENNN_QUALITY_WEIGHT_DECAY": "0.25",
            "OPENNN_QUALITY_LR_MILESTONES": "9",
            "OPENNN_QUALITY_LR_GAMMA": "0.5",
            "OPENNN_DETERMINISTIC": "1",
            "OPENNN_LT_DETERMINISTIC": "1",
            "OPENNN_QUALITY_DETERMINISTIC": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":16:8",
        }

        def execute(command, **kwargs):
            environments.append(dict(kwargs["env"]))
            engine = "opennn" if command[0] == str(binary) else "pytorch"
            folder = Path(command[-7])
            read_array(manifest_path.parent, manifest["test"]["y"]).tofile(folder / "predictions.bin")
            driver = {
                "model": "dense", "engine": engine, "seed": int(command[-4]),
                "epochs": int(command[-6]), "device": "cpu", "precision": "fp32",
                "deterministic": kwargs["env"].get("OPENNN_QUALITY_DETERMINISTIC") == "1",
                "optimizer": kwargs["env"].get("OPENNN_QUALITY_OPTIMIZER", "adam"),
                "adam_epsilon": float(kwargs["env"]["OPENNN_QUALITY_ADAM_EPSILON"]),
                "weight_decay": float(kwargs["env"]["OPENNN_QUALITY_WEIGHT_DECAY"]),
                "learning_rate_milestones": [
                    int(value) for value in kwargs["env"]["OPENNN_QUALITY_LR_MILESTONES"].split(",")
                    if value
                ],
                "learning_rate_gamma": float(kwargs["env"]["OPENNN_QUALITY_LR_GAMMA"]),
                "batchnorm_recalibrated": kwargs["env"].get("OPENNN_QUALITY_RECALIBRATE_BATCHNORM") == "1",
                "train_samples": 4, "test_samples": 3, "parameters": 1,
                "interface": "python" if engine == "pytorch" else "cpp",
                "diagnostic_only": diagnostic_only, "status": "ok",
            }
            if during_process:
                during_process(driver, manifest_path, len(environments))
            (folder / "driver.json").write_text(json.dumps(driver))
            (folder / "history.csv").write_text(history_text)
            return subprocess.CompletedProcess(command, 0)

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, inherited))
            stack.enter_context(patch.object(quality_runner, "result_directory", return_value=output))
            stack.enter_context(patch.object(quality_runner, "git_metadata", return_value={}))
            stack.enter_context(patch.object(quality_runner, "cpu_state", return_value={}))
            stack.enter_context(patch.object(quality_runner.platform, "platform", return_value="test-platform"))
            stack.enter_context(patch.object(quality_runner.subprocess, "run", side_effect=execute))
            arguments = [
                "--model", "dense", "--manifest", str(manifest_path),
                "--epochs", "1", "--batch", "4", "--seeds", seeds,
                "--opennn-binary", str(binary), "--out", str(output),
            ]
            result = quality_runner.main(arguments + list(extra_args or []))
            self.assertTrue(all(os.environ[key] == value for key, value in inherited.items()))
        return result, environments, json.loads((output / "results.json").read_text())

    def test_deterministic_mode_is_explicit_and_forwarded_to_both_engines(self):
        result, environments, report = self.run_mocked_pair(extra_args=["--deterministic"])
        self.assertEqual(result, 0)
        self.assertTrue(report["provenance"]["arguments"]["deterministic"])
        self.assertTrue(all(run["driver"]["deterministic"] for run in report["runs"]))
        for environment in environments:
            self.assertEqual(environment["OPENNN_DETERMINISTIC"], "1")
            self.assertEqual(environment["OPENNN_LT_DETERMINISTIC"], "1")
            self.assertEqual(environment["OPENNN_QUALITY_DETERMINISTIC"], "1")
            self.assertEqual(environment["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")

    def test_sgd_mode_is_explicit_and_forwarded_to_both_engines(self):
        result, environments, report = self.run_mocked_pair(
            extra_args=["--optimizer", "sgd", "--momentum", "0.8"]
        )
        self.assertEqual(result, 0)
        self.assertEqual(report["provenance"]["arguments"]["optimizer"], "sgd")
        self.assertTrue(all(run["driver"]["optimizer"] == "sgd" for run in report["runs"]))
        for environment in environments:
            self.assertEqual(environment["OPENNN_QUALITY_OPTIMIZER"], "sgd")
            self.assertEqual(environment["OPENNN_QUALITY_MOMENTUM"], "0.8")

    def test_adam_epsilon_is_explicit_and_forwarded_to_both_engines(self):
        result, environments, report = self.run_mocked_pair(
            extra_args=["--adam-epsilon", "0.0001"]
        )
        self.assertEqual(result, 0)
        expected = float(np.float32(0.0001))
        self.assertEqual(report["provenance"]["arguments"]["adam_epsilon"], expected)
        self.assertTrue(all(run["driver"]["adam_epsilon"] == expected
                            for run in report["runs"]))
        for environment in environments:
            self.assertEqual(environment["OPENNN_QUALITY_ADAM_EPSILON"], str(expected))

    def test_sgd_learning_rate_schedule_is_explicit_and_forwarded(self):
        result, environments, report = self.run_mocked_pair(
            extra_args=["--epochs", "4", "--optimizer", "sgd",
                        "--learning-rate-milestones", "2,3",
                        "--learning-rate-gamma", "0.2"],
            history_text="epoch,training_loss\n1,0.5\n2,0.4\n3,0.3\n4,0.2\n",
        )
        self.assertEqual(result, 0)
        self.assertEqual(report["provenance"]["arguments"]["learning_rate_milestones"], [2, 3])
        expected_gamma = float(np.float32(0.2))
        self.assertEqual(report["provenance"]["arguments"]["learning_rate_gamma"], expected_gamma)
        for environment in environments:
            self.assertEqual(environment["OPENNN_QUALITY_LR_MILESTONES"], "2,3")
            self.assertEqual(environment["OPENNN_QUALITY_LR_GAMMA"], str(expected_gamma))
        self.assertTrue(all(run["driver"]["learning_rate_milestones"] == [2, 3]
                            for run in report["runs"]))

    def test_weight_decay_is_explicit_and_forwarded(self):
        result, environments, report = self.run_mocked_pair(
            extra_args=["--weight-decay", "0.0005"]
        )
        self.assertEqual(result, 0)
        expected = float(np.float32(0.0005))
        self.assertEqual(report["provenance"]["arguments"]["weight_decay"], expected)
        self.assertTrue(all(run["driver"]["weight_decay"] == expected
                            for run in report["runs"]))
        for environment in environments:
            self.assertEqual(environment["OPENNN_QUALITY_WEIGHT_DECAY"], str(expected))

    def test_truncated_training_history_is_rejected(self):
        result, _, report = self.run_mocked_pair(history_text="epoch,training_loss\n")
        self.assertEqual(result, 1)
        self.assertTrue(all(run["status"] == "error" for run in report["runs"]))
        self.assertTrue(all("Training history has 0 rows; expected 1" in run["error"]
                            for run in report["runs"]))

    def test_batchnorm_recalibration_is_explicit_and_forwarded(self):
        result, environments, report = self.run_mocked_pair(
            extra_args=["--recalibrate-batchnorm"]
        )
        self.assertEqual(result, 0)
        self.assertTrue(report["provenance"]["arguments"]["recalibrate_batchnorm"])
        self.assertTrue(all(run["driver"]["batchnorm_recalibrated"] for run in report["runs"]))
        for environment in environments:
            self.assertEqual(environment["OPENNN_QUALITY_RECALIBRATE_BATCHNORM"], "1")

    def test_cli_clears_inherited_diagnostic_and_initialization_modes(self):
        result, environments, report = self.run_mocked_pair()
        self.assertEqual(result, 0)
        self.assertEqual(len(environments), 2)
        for environment in environments:
            for key in ("OPENNN_QUALITY_DIAGNOSTIC_STEPS", "OPENNN_QUALITY_INITIALIZATION",
                        "OPENNN_QUALITY_INITIAL_PREDICTIONS", "OPENNN_QUALITY_EXPORT_INITIALIZATION",
                        "OPENNN_QUALITY_LOAD_CHECKPOINT", "OPENNN_DETERMINISTIC",
                        "OPENNN_LT_DETERMINISTIC", "OPENNN_QUALITY_DETERMINISTIC",
                        "CUBLAS_WORKSPACE_CONFIG"):
                self.assertNotIn(key, environment)
            self.assertEqual(
                environment["OPENNN_QUALITY_ADAM_EPSILON"],
                str(float(np.finfo(np.float32).eps)),
            )
            self.assertEqual(environment["OPENNN_QUALITY_WEIGHT_DECAY"], "0.0")
            self.assertEqual(environment["OPENNN_QUALITY_LR_MILESTONES"], "")
            self.assertEqual(
                environment["OPENNN_QUALITY_LR_GAMMA"],
                str(float(np.float32(0.1))),
            )
        self.assertTrue(all(run["status"] == "ok" for run in report["runs"]))

    def test_cli_rejects_diagnostic_only_output_as_quality_evidence(self):
        result, _, report = self.run_mocked_pair(diagnostic_only=True)
        self.assertEqual(result, 1)
        self.assertTrue(all(run["status"] == "error" for run in report["runs"]))
        self.assertTrue(all("Numerical diagnostics" in run["error"] for run in report["runs"]))

    def test_missing_work_metadata_retains_failed_trial_and_remaining_seeds(self):
        def omit_metadata(driver, manifest_path, launch):
            if launch == 1:
                del driver["epochs"]
            elif launch == 2:
                del driver["parameters"]

        result, environments, report = self.run_mocked_pair(seeds="42,43", during_process=omit_metadata)
        self.assertEqual(result, 1)
        self.assertEqual(len(environments), 4)
        self.assertEqual([run["status"] for run in report["runs"]], ["error", "error", "ok", "ok"])
        self.assertIn("work gate", report["runs"][0]["error"])
        self.assertIn("parameter count", report["runs"][1]["error"])
        comparison = report["paired_analysis"]["rows"][0]
        self.assertEqual(comparison["complete_pairs"], 1)
        self.assertEqual(comparison["pair_issues"][0]["seed"], 42)
        self.assertFalse(report["quality_gate"]["assessed"])

    def test_data_mutation_invalidates_current_run_and_blocks_later_launches(self):
        def change_data(driver, manifest_path, launch):
            manifest = json.loads(manifest_path.read_text())
            path = manifest_path.parent / manifest["train"]["x"]["file"]
            content = path.read_bytes()
            path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

        result, environments, report = self.run_mocked_pair(seeds="42,43", during_process=change_data)
        self.assertEqual(result, 1)
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(report["runs"]), 4)
        self.assertTrue(all(run["status"] == "error" for run in report["runs"]))
        self.assertIsNotNone(report["runs"][0]["process_seconds"])
        self.assertTrue(all(run["process_seconds"] is None for run in report["runs"][1:]))
        self.assertEqual(report["paired_analysis"]["rows"][0]["complete_pairs"], 0)

    def test_manifest_mutation_invalidates_current_run_and_blocks_later_launches(self):
        def change_manifest(driver, manifest_path, launch):
            manifest_path.write_bytes(manifest_path.read_bytes() + b" ")

        result, environments, report = self.run_mocked_pair(seeds="42,43", during_process=change_manifest)
        self.assertEqual(result, 1)
        self.assertEqual(len(environments), 1)
        self.assertEqual(len(report["runs"]), 4)
        self.assertTrue(all("manifest changed" in run["error"] for run in report["runs"]))

    def test_implementation_identity_includes_initialization_and_data_loaders(self):
        benchmarks = self.root / "benchmarks"
        (benchmarks / "families").mkdir(parents=True)
        (benchmarks / "tools").mkdir()
        binary = self.root / "driver"
        paths = {
            "opennn_binary_sha256": binary,
            "pytorch_driver_sha256": benchmarks / "families/quality.py",
            "quality_data_sha256": benchmarks / "tools/quality_data.py",
            "initialization_generator_sha256": benchmarks / "tools/quality_initialization.py",
        }
        for path in paths.values():
            path.write_bytes(b"original implementation")
        provenance = {key: digest(path) for key, path in paths.items()}
        with patch.object(quality_runner, "BENCHMARKS", benchmarks):
            quality_runner.verify_implementation_files(binary, provenance)
            for key, path in paths.items():
                with self.subTest(implementation=key):
                    path.write_bytes(b"changed implementation")
                    with self.assertRaisesRegex(ValueError, "implementation changed"):
                        quality_runner.verify_implementation_files(binary, provenance)
                    path.write_bytes(b"original implementation")

    def test_markdown_uses_complete_pairs_and_exposes_excluded_failure(self):
        provenance = {"arguments": {"seeds": "42,43"}, "manifests": {"dense": {}}}
        records = [
            {"model": "dense", "engine": "opennn", "seed": 42, "status": "ok", "metrics": {"accuracy_percent": 10}},
            {"model": "dense", "engine": "pytorch", "seed": 42, "status": "ok", "metrics": {"accuracy_percent": 20}},
            {"model": "dense", "engine": "opennn", "seed": 43, "status": "ok", "metrics": {"accuracy_percent": 100}},
            {"model": "dense", "engine": "pytorch", "seed": 43, "status": "error", "error": "test failure"},
        ]
        quality_runner.write_results(self.root, provenance, records)
        markdown = (self.root / "comparison.md").read_text(encoding="utf-8")
        self.assertIn("| 10 (one pair) | 20 (one pair) | 1/2 | Incomplete or invalid |", markdown)
        self.assertIn("dense seed 43", markdown)
        report = json.loads((self.root / "results.json").read_text())
        native = next(row for row in report["summary"] if row["engine"] == "opennn")
        self.assertEqual(native["mean"], 55)
        self.assertFalse(report["quality_gate"]["assessed"])


if __name__ == "__main__":
    unittest.main()
