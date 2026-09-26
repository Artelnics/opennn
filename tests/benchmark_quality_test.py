"""Neutral scoring, data identity, dispatch and application baseline checks."""

import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/tools"))
from quality_data import prepare_one, validate_manifest, read_array
from quality_runner import score_predictions, summarize
from application_startup import make_cases, summarize as summarize_startup
from experiment import dispatch, result_directory


class BenchmarkQualityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)

    def fixture(self, model):
        path = prepare_one(model, self.root, self.root / model, smoke=True)
        manifest = validate_manifest(path)
        return path, manifest

    def test_perfect_classifications(self):
        for kind in ("dense", "cnn"):
            path, manifest = self.fixture(kind)
            predictions = self.root / f"{kind}.bin"
            read_array(path.parent, manifest["test"]["y"]).tofile(predictions)
            self.assertEqual(
                score_predictions(path, predictions)["accuracy_percent"], 100
            )

    def test_forecasting_error_uses_original_units(self):
        path, manifest = self.fixture("lstm")
        manifest["target_scale"] = 10
        path.write_text(json.dumps(manifest))
        target = np.asarray(read_array(path.parent, manifest["test"]["y"]))
        predictions = self.root / "forecast.bin"
        (target + 2).astype("<f4").tofile(predictions)
        self.assertAlmostEqual(
            score_predictions(path, predictions)["rmse"], 20, places=5
        )

    def test_missing_or_nonfinite_predictions_rejected(self):
        path, _ = self.fixture("dense")
        predictions = self.root / "bad.bin"
        for values in ([0.5], [0.5, float("nan"), 0.2]):
            np.asarray(values, dtype="<f4").tofile(predictions)
            with self.assertRaises(ValueError):
                score_predictions(path, predictions)

    def test_modified_tensor_rejected_before_training(self):
        path, manifest = self.fixture("dense")
        file = path.parent / manifest["train"]["x"]["file"]
        with file.open("r+b") as stream:
            stream.write(b"xxxx")
        with self.assertRaises(ValueError):
            validate_manifest(path)

    def test_same_train_and_test_data_rejected(self):
        path, manifest = self.fixture("dense")
        manifest["test"] = manifest["train"]
        path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            validate_manifest(path)

    def test_cnn_duplicate_content_across_classes_is_excluded(self):
        from PIL import Image

        source = self.root / "imagenet_subset/train"
        first, second = source / "class_a", source / "class_b"
        first.mkdir(parents=True)
        second.mkdir(parents=True)
        for directory, colors in ((first, (10, 20, 30)), (second, (40, 50))):
            for index, color in enumerate(colors):
                Image.new("RGB", (8, 8), (color, color, color)).save(
                    directory / f"{index}.jpg"
                )
        shutil.copyfile(first / "0.jpg", second / "duplicate.jpg")

        path = prepare_one("cnn", self.root, self.root / "cnn-real", smoke=False)
        manifest = validate_manifest(path)
        self.assertEqual(len(manifest["excluded_duplicate_images"]), 1)
        self.assertEqual(manifest["excluded_duplicate_images"][0]["path"],
                         "class_b/duplicate.jpg")
        identities = json.loads((path.parent / "image-split.json").read_text())
        self.assertEqual(len(identities), 5)
        train = {item["sha256"] for item in identities if item["split"] == "train"}
        test = {item["sha256"] for item in identities if item["split"] == "test"}
        self.assertFalse(train & test)

    def test_cnn_fixed_training_order_mixes_classes(self):
        from PIL import Image

        source = self.root / "imagenet_subset/train"
        for label in range(4):
            directory = source / f"class_{label}"
            directory.mkdir(parents=True)
            for index in range(8):
                color = label * 32 + index
                Image.new("RGB", (8, 8), (color, color, color)).save(
                    directory / f"{index}.png"
                )
        path = prepare_one("cnn", self.root, self.root / "cnn-mixed", smoke=False)
        manifest = validate_manifest(path)
        labels = np.asarray(read_array(path.parent, manifest["train"]["y"])).argmax(1)
        self.assertEqual(set(labels), set(range(4)))
        self.assertGreater(len(set(labels[:8])), 1)
        self.assertFalse(np.array_equal(labels, np.sort(labels)))

    def test_translation_generation_and_padding(self):
        path, manifest = self.fixture("transformer")
        predictions = self.root / "translation.bin"
        read_array(path.parent, manifest["test"]["y"]).tofile(predictions)
        self.assertAlmostEqual(score_predictions(path, predictions)["bleu"], 100)
        values = np.asarray(read_array(path.parent, manifest["test"]["y"])).copy()
        values[0, 0] = len(manifest["vocabulary"])
        values.tofile(predictions)
        with self.assertRaises(ValueError):
            score_predictions(path, predictions)

    def test_single_seed_has_no_estimated_variation(self):
        rows = summarize(
            [
                {
                    "model": "dense",
                    "engine": "opennn",
                    "status": "ok",
                    "metrics": {"accuracy_percent": 75},
                }
            ]
        )
        self.assertIsNone(rows[0]["stddev"])

    def test_single_startup_sample_has_no_estimated_variation(self):
        case = {
            "id": "dense-opennn",
            "engine": "opennn",
            "backend": "cpu-eigen",
            "device": "cpu",
            "family": "dense",
            "precision": "fp32",
            "cache": "reused",
        }
        sample = {
            "case": case["id"],
            "status": "ok",
            "latency_ms": 10,
            "process_lifetime_ms": 12,
        }
        row = summarize_startup([sample], [case])[0]
        self.assertIsNone(row["cv_percent"])
        self.assertFalse(row["variation_assessed"])
        second = dict(sample, latency_ms=20)
        row = summarize_startup([sample, second], [case])[0]
        self.assertTrue(row["variation_assessed"])
        self.assertGreater(row["cv_percent"], 0)

    def test_application_matrix_and_python_baseline(self):
        native = {
            "engine": "opennn",
            "backend": "cuda",
            "device": "cuda",
            "binary_pattern": "/bin/{family}",
        }
        python = {
            "engine": "pytorch",
            "backend": "cuda",
            "device": "cuda",
            "binary_pattern": "/venv/bin/python",
            "prefix_args": ["-I", "/application.py", "{family}"],
        }
        self.assertEqual(len(make_cases({"groups": [native, python]})), 32)
        with self.assertRaises(ValueError):
            make_cases({"groups": [native]})
        with self.assertRaises(ValueError):
            make_cases({"groups": [native, dict(python, prefix_args=[])]})

    def test_result_path_and_unrelated_dispatch(self):
        with self.assertRaises(ValueError):
            result_directory("quality", self.root / "outside")
        self.assertIsNone(dispatch(["--family", "dense", "--label", "quality"]))


if __name__ == "__main__":
    unittest.main()
