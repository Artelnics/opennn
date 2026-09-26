"""Shared initial tensors must be reproducible, complete and unmodified."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from quality_data import profile
from quality_initialization import _build, create_initialization, load_initialization, validate_initialization


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class BenchmarkQualityInitializationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.root = Path(temporary.name).resolve()
        self.addCleanup(temporary.cleanup)

    def manifest(self, kind="dense"):
        shapes = {
            "dense": [4, 28], "lstm": [4, 4, 15],
            "cnn": [4, 32, 32, 3], "transformer": [4, 6],
        }
        return {
            "model": kind,
            "profile": profile(kind, smoke=True),
            "train": {"x": {"shape": shapes[kind]},
                      "y": {"shape": [4, 3] if kind == "cnn" else [4, 1]}},
            "vocabulary": [f"token{i}" for i in range(32)],
        }

    def fixture(self, kind="dense", directory="initialization"):
        manifest = self.manifest(kind)
        model = _build(manifest)
        path = create_initialization(manifest, self.root / directory, 42, model=model)
        return manifest, model, path

    def test_all_model_templates_reload_every_parameter_and_buffer_exactly(self):
        for kind in ("dense", "lstm", "transformer", "cnn"):
            with self.subTest(model=kind):
                if kind == "cnn" and importlib.util.find_spec("torchvision") is None:
                    self.skipTest("torchvision is unavailable")
                manifest, model, path = self.fixture(kind, kind)
                arrays, provenance = validate_initialization(manifest, path, model, 42)
                before_rng = torch.random.get_rng_state().clone()
                loaded = load_initialization(model, manifest, path, expected_seed=42)
                self.assertEqual(provenance, loaded)
                self.assertEqual(provenance["tensor_count"], len(model.state_dict()))
                self.assertEqual(provenance["manifest_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertTrue(torch.equal(before_rng, torch.random.get_rng_state()))
                for name, tensor in model.state_dict().items():
                    np.testing.assert_array_equal(tensor.detach().numpy(), arrays[name], err_msg=name)
                if kind == "lstm":
                    self.assertFalse(model.lstm.bias_hh_l0.requires_grad)
                    self.assertEqual(int(model.lstm.bias_hh_l0.count_nonzero()), 0)
                if kind == "cnn":
                    self.assertTrue(any(name.endswith("running_var") for name in arrays))
                    for name, values in arrays.items():
                        if name.endswith("running_var"):
                            np.testing.assert_array_equal(values, np.ones_like(values))
                        if name.endswith(("running_mean", "num_batches_tracked")):
                            self.assertEqual(int(np.count_nonzero(values)), 0)
                    classifier_bound = 1 / np.sqrt(model.fc.in_features)
                    self.assertLessEqual(
                        float(np.abs(arrays["fc.weight"]).max()), classifier_bound
                    )
                    self.assertLessEqual(
                        float(np.abs(arrays["fc.bias"]).max()), classifier_bound
                    )
                    self.assertGreater(int(np.count_nonzero(arrays["fc.bias"])), 0)
                if kind == "transformer":
                    self.assertIn("position", arrays)

    def test_same_seed_is_bitwise_reproducible_and_different_seed_changes_weights(self):
        manifest, model, first = self.fixture()
        second = create_initialization(manifest, self.root / "second", 42, model)
        different = create_initialization(manifest, self.root / "different", 43, model)
        self.assertEqual(first.read_bytes(), second.read_bytes())
        original = json.loads(first.read_text())["tensors"]
        changed = json.loads(different.read_text())["tensors"]
        self.assertTrue(any(a["sha256"] != b["sha256"] for a, b in zip(original, changed)))
        for entry in original:
            self.assertEqual((first.parent / entry["file"]).read_bytes(),
                             (second.parent / entry["file"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "not empty"):
            create_initialization(manifest, first.parent, 42, model)

    def test_shape_template_preserves_rng(self):
        before = torch.random.get_rng_state().clone()
        _build(self.manifest())
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_mutated_tensor_is_rejected_before_any_parameter_changes(self):
        manifest, model, path = self.fixture()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        record = json.loads(path.read_text())
        tensor_path = path.parent / record["tensors"][-1]["file"]
        contents = bytearray(tensor_path.read_bytes())
        contents[-1] ^= 1
        tensor_path.write_bytes(contents)
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            load_initialization(model, manifest, path)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_missing_duplicate_unexpected_and_wrong_shape_tensors_are_rejected(self):
        manifest, model, path = self.fixture()
        original = json.loads(path.read_text())
        for problem in ("missing", "duplicate", "unexpected", "shape", "escape"):
            with self.subTest(problem=problem):
                record = copy.deepcopy(original)
                if problem == "missing":
                    record["tensors"].pop()
                elif problem == "duplicate":
                    record["tensors"].append(record["tensors"][0])
                elif problem == "unexpected":
                    record["tensors"][0]["name"] = "unknown.weight"
                elif problem == "shape":
                    record["tensors"][0]["shape"][0] += 1
                else:
                    record["tensors"][0]["file"] = "../outside.bin"
                path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    load_initialization(model, manifest, path)

    def test_mismatched_seed_and_architecture_are_rejected(self):
        manifest, model, path = self.fixture()
        with self.assertRaisesRegex(ValueError, "trial seed"):
            validate_initialization(manifest, path, model, expected_seed=43)
        changed = copy.deepcopy(manifest)
        changed["profile"]["hidden"] += 1
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_initialization(changed, path, model)

    def test_nonzero_frozen_lstm_bias_rejected_even_with_matching_hash(self):
        manifest, model, path = self.fixture("lstm")
        record = json.loads(path.read_text())
        spec = next(item for item in record["tensors"] if item["name"] == "lstm.bias_hh_l0")
        target = path.parent / spec["file"]
        np.ones(spec["shape"], dtype="<f4").tofile(target)
        spec["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "recurrent bias"):
            validate_initialization(manifest, path, model)


if __name__ == "__main__":
    unittest.main()
