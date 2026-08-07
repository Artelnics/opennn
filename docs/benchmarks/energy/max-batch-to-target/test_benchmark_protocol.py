#!/usr/bin/env python3
"""Regression tests for benchmark command construction."""

import importlib.util
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "run_max_batch_to_target", HERE / "run_max_batch_to_target.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

class CommandProtocolTest(unittest.TestCase):
    def test_resnet_tensorflow_honors_graph_mode(self):
        args = types.SimpleNamespace(
            workload="resnet50",
            tf_xla="0",
            bench_python="/python",
            data_dir="/cifar",
            precision="fp32",
            target=2.0,
            max_steps=20,
        )
        capacity = {
            "machine": {"gpu": {"memory_total_mib": 6144}},
            "configuration": {"vram_reserve_mib": 256},
        }
        command, env = RUNNER.command_for(
            args, "tensorflow", 1562, capacity, "tensorflow_graph", 42)
        self.assertEqual(env["TF_XLA"], "0")
        self.assertIn("tensorflow_resnet50_maxbatch.py", command[1])

    def test_resnet_tensorflow_honors_xla_mode(self):
        args = types.SimpleNamespace(
            workload="resnet50",
            tf_xla="1",
            bench_python="/python",
            data_dir="/cifar",
            precision="fp32",
            target=2.0,
            max_steps=20,
        )
        capacity = {
            "machine": {"gpu": {"memory_total_mib": 6144}},
            "configuration": {"vram_reserve_mib": 256},
        }
        _, env = RUNNER.command_for(
            args, "tensorflow", 1562, capacity, "tensorflow_xla", 42)
        self.assertEqual(env["TF_XLA"], "1")

if __name__ == "__main__":
    unittest.main()
