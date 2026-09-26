"""Numerical diagnostic comparisons must expose differences and missing work."""

import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from quality_data import digest, prepare_one, read_array
from quality_numerics import compare_arrays, compare_outputs, compare_run_artifacts, compare_tensor_sets, read_tensor_manifest


class BenchmarkQualityNumericsTest(unittest.TestCase):
    def test_array_statistics_and_zero_reference(self):
        result = compare_arrays(np.array([1, 2]), np.array([1, 4]))
        self.assertEqual(result["max_abs"], 2)
        self.assertAlmostEqual(result["rms"], math.sqrt(2))
        self.assertAlmostEqual(result["relative_l2"], 2 / math.sqrt(17))
        self.assertEqual(compare_arrays(np.zeros(3), np.zeros(3))["relative_l2"], 0)
        self.assertEqual(compare_arrays(np.ones(3), np.zeros(3))["relative_l2"], 1)

    def test_chunk_boundaries_and_scalar_arrays(self):
        left = np.zeros((1 << 20) + 7, dtype=np.float32)
        right = left.copy()
        right[-1] = 2
        result = compare_arrays(left, right)
        self.assertEqual(result["max_abs"], 2)
        self.assertAlmostEqual(result["rms"], 2 / math.sqrt(len(left)))
        self.assertEqual(compare_arrays(np.array(2), np.array(3))["shape"], [])

    def test_invalid_shapes_empty_and_nonfinite_are_rejected(self):
        for left, right in (([1], [1, 2]), ([], []), ([float("nan")], [0]), ([0], [float("inf")])):
            with self.assertRaises(ValueError):
                compare_arrays(left, right)

    def test_padding_differences_are_exposed_separately(self):
        left = np.zeros((1, 3, 2))
        right = left.copy()
        right[0, 2, :] = 100
        result = compare_outputs(left, right, model="transformer", targets=np.array([[2, 3, 0]]))
        self.assertEqual(result["all_positions"]["max_abs"], 100)
        self.assertEqual(result["active_positions"]["max_abs"], 0)
        self.assertEqual(result["padded_positions"]["max_abs"], 100)
        self.assertEqual(result["active_tokens"], 2)
        with self.assertRaises(ValueError):
            compare_outputs(left, right, model="transformer", targets=np.zeros((1, 3)))

    def test_frozen_lstm_omission_is_explicit_and_verified(self):
        initial = {"lstm.weight_ih_l0": np.ones((4, 2)), "lstm.bias_hh_l0": np.zeros(4)}
        native = {"lstm.weight_ih_l0": initial["lstm.weight_ih_l0"].copy()}
        result = compare_tensor_sets(native, initial, initial, model="lstm", kind="initialization")
        self.assertEqual(result["native_omissions"][0]["name"], "lstm.bias_hh_l0")
        self.assertTrue(result["native_omissions"][0]["verified"])
        invalid = dict(initial, **{"lstm.bias_hh_l0": np.ones(4)})
        with self.assertRaises(ValueError):
            compare_tensor_sets(native, invalid, initial, model="lstm", kind="initialization")

    def test_batchnorm_counter_requires_expected_update_count(self):
        initial = {"bn.weight": np.ones(2), "bn.num_batches_tracked": np.array(0)}
        native = {"bn.weight": np.ones(2)}
        updated = {"bn.weight": np.ones(2), "bn.num_batches_tracked": np.array(3)}
        result = compare_tensor_sets(native, updated, initial, model="cnn", kind="parameters", step=2)
        self.assertEqual(result["native_omissions"][0]["reference_value"], 3)
        with self.assertRaises(ValueError):
            compare_tensor_sets(native, updated, initial, model="cnn", kind="parameters", step=1)

    def test_unknown_missing_or_extra_tensor_cannot_be_ignored(self):
        initial = {"weight": np.ones(2), "bias": np.zeros(2)}
        for native, python in (({"weight": np.ones(2)}, initial),
                               (initial, {"weight": np.ones(2)}),
                               ({**initial, "other": np.zeros(2)}, initial)):
            with self.assertRaises(ValueError):
                compare_tensor_sets(native, python, initial, model="dense", kind="initialization")

    def test_gradients_require_all_trainable_parameters_and_no_buffers(self):
        initial = {"bn.weight": np.ones(2), "bn.running_mean": np.zeros(2),
                   "bn.running_var": np.ones(2), "bn.num_batches_tracked": np.array(0)}
        gradient = {"bn.weight": np.ones(2)}
        result = compare_tensor_sets(gradient, gradient, initial, model="cnn", kind="gradients", step=0)
        self.assertEqual(result["tensor_count"], 1)
        self.assertEqual(result["native_omissions"], [])
        with self.assertRaises(ValueError):
            compare_tensor_sets(initial, gradient, initial, model="cnn", kind="gradients", step=0)

    def test_initialization_roundtrip_catches_matching_but_wrong_weights(self):
        initial = {"weight": np.ones(2)}
        incorrect = {"weight": np.zeros(2)}
        result = compare_tensor_sets(incorrect, incorrect, initial, model="dense", kind="initialization")
        self.assertEqual(result["max_abs"], 0)
        self.assertEqual(result["source_roundtrip"]["opennn"][0]["max_abs"], 1)
        self.assertFalse(result["source_roundtrip_exact"]["opennn"])

    def test_stage_failures_do_not_discard_other_comparisons(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest_path = prepare_one("dense", root, root / "data", smoke=True)
            manifest = json.loads(manifest_path.read_text())

            def tensors(folder):
                folder.mkdir(parents=True, exist_ok=True)
                np.ones(2, dtype="<f4").tofile(folder / "tensor.bin")
                path = folder / "manifest.json"
                path.write_text(json.dumps({"schema": 1, "tensors": [{
                    "name": "weight", "shape": [2], "file": "tensor.bin", "dtype": "float32"}]}))
                return path

            initial = tensors(root / "shared")
            for engine in ("opennn", "pytorch"):
                folder = root / engine
                tensors(folder / "initialization")
                for step in (None, 0):
                    split = "test" if step is None else "train"
                    count = 8 if step is None else 2
                    target = np.array(read_array(manifest_path.parent, manifest[split]["y"])[:count], copy=True)
                    output = folder if step is None else folder / "diagnostic" / "step_0"
                    output.mkdir(parents=True, exist_ok=True)
                    prefix = "initial_predictions" if step is None else "outputs"
                    target.astype("<f4").tofile(output / f"{prefix}.bin")
                    (output / f"{prefix}.json").write_text(json.dumps({
                        "shape": list(target.shape), "dtype": "float32", "samples": len(target),
                        "output": "probabilities", "mode": "evaluation" if step is None else "training"}))
                    if step is not None:
                        tensors(output / "gradients")
                        tensors(output / "parameters")
                (folder / "diagnostic.json").write_text(json.dumps({
                    "mode": "fixed_batch_training", "steps": [{"step": 0, "loss": 0.5, "samples": 2}]}))
            result = compare_run_artifacts(root, manifest_path, initial, steps=1, batch=2)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["comparisons"]), 6)
            (root / "pytorch" / "diagnostic" / "step_0" / "gradients" / "manifest.json").write_text(
                json.dumps({"schema": 1, "tensors": []}))
            result = compare_run_artifacts(root, manifest_path, initial, steps=1, batch=2)
            self.assertFalse(result["complete"])
            self.assertEqual(result["errors"][0]["stage"], "step_0_gradients")
            self.assertIn("step_0_parameters", result["comparisons"])

    def test_tensor_manifest_validates_names_hashes_paths_and_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            values = root / "tensor.bin"
            np.array([1, 2], dtype="<f4").tofile(values)
            spec = {"name": "weight", "shape": [2], "dtype": "float32", "file": "tensor.bin",
                    "sha256": digest(values)}
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"schema": 1, "tensors": [spec]}))
            tensors, identity = read_tensor_manifest(manifest)
            self.assertEqual(list(tensors["weight"]), [1, 2])
            self.assertEqual(identity["sha256"], digest(manifest))
            del tensors
            for entries in ([spec, spec], [{**spec, "sha256": "bad"}],
                            [{**spec, "file": "../tensor.bin"}], [{**spec, "shape": [3]}]):
                manifest.write_text(json.dumps({"schema": 1, "tensors": entries}))
                with self.assertRaises(ValueError):
                    read_tensor_manifest(manifest)
            np.array([1, float("nan")], dtype="<f4").tofile(values)
            spec.pop("sha256")
            manifest.write_text(json.dumps({"schema": 1, "tensors": [spec]}))
            with self.assertRaises(ValueError):
                read_tensor_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
