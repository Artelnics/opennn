"""The forecasting driver must match OpenNN's half-MSE objective and gradient."""

import csv
import copy
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import weakref
from unittest.mock import patch

import numpy as np

try:
    import torch
except ImportError:
    torch = None

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCHMARKS / "tools"))
from quality_data import array_file


@unittest.skipIf(torch is None, "PyTorch is unavailable")
class BenchmarkQualityDriverTest(unittest.TestCase):
    @staticmethod
    def driver_module():
        specification = importlib.util.spec_from_file_location("quality_loss_test_driver", BENCHMARKS / "families/quality.py")
        driver = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(driver)
        return driver

    def test_forecaster_fp32_path_preserves_outputs_and_gradients(self):
        driver = self.driver_module()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = driver.Forecaster(3, 4)
            reference = copy.deepcopy(model)
            model.explicit_recurrent_bf16 = True  # CPU still follows FP32.
            inputs = torch.randn(2, 5, 3)
            observed = model(inputs)
            expected = reference.head(reference.lstm(inputs)[0][:, -1])
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
            observed.square().mean().backward()
            expected.square().mean().backward()
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad, rtol=0, atol=0)

    def test_cuda_bf16_lstm_keeps_fp32_masters_and_releases_temporary_weights(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        if torch.backends.cudnn.version() < 92000:
            self.skipTest("This BF16 recurrent regression requires the validated cuDNN 9.20 runtime")
        driver = self.driver_module()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(42)
            model = driver.Forecaster(3, 4).cuda()
            model.explicit_recurrent_bf16 = True
            reference_lstm = copy.deepcopy(model.lstm).to(torch.bfloat16)
            reference_head = copy.deepcopy(model.head)
            identities = {name: id(parameter) for name, parameter in model.named_parameters()}
            original = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
            observations, temporary_weights = [], []

            def observe(module, inputs, outputs):
                sequence, (hidden, cell) = outputs
                observations.append((inputs[0].dtype, sequence.dtype, hidden.dtype, cell.dtype))
                temporary_weights[:] = [weakref.ref(parameter) for parameter in module.parameters()]

            hook = model.lstm.register_forward_hook(observe)
            self.addCleanup(hook.remove)
            optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            inputs = torch.randn(2, 5, 3, device="cuda")
            targets = torch.randn(2, 1, device="cuda")
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                reference_lstm.load_state_dict(model.lstm.state_dict())
                reference_head.load_state_dict(model.head.state_dict())
                reference_lstm.zero_grad(set_to_none=True)
                reference_head.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = model(inputs)
                self.assertEqual(prediction.dtype, torch.bfloat16)
                loss = .5 * torch.nn.functional.mse_loss(prediction.float(), targets)
                loss.backward()
                with torch.autocast("cuda", enabled=False):
                    reference_sequence = reference_lstm(inputs.to(torch.bfloat16))[0]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    reference_prediction = reference_head(reference_sequence[:, -1])
                reference_loss = .5 * torch.nn.functional.mse_loss(reference_prediction.float(), targets)
                reference_loss.backward()
                torch.testing.assert_close(prediction, reference_prediction, rtol=0, atol=0)
                expected_parameters = {
                    **{"lstm." + name: parameter for name, parameter in reference_lstm.named_parameters()},
                    **{"head." + name: parameter for name, parameter in reference_head.named_parameters()},
                }
                for name, parameter in model.named_parameters():
                    self.assertEqual(id(parameter), identities[name])
                    self.assertEqual(parameter.dtype, torch.float32)
                    if parameter.requires_grad:
                        self.assertEqual(parameter.grad.dtype, torch.float32)
                        self.assertTrue(torch.isfinite(parameter.grad).all())
                        torch.testing.assert_close(parameter.grad, expected_parameters[name].grad.float(), rtol=0, atol=0)
                    else:
                        self.assertIsNone(parameter.grad)
                        self.assertEqual(int(parameter.count_nonzero()), 0)
                optimizer.step()
                self.assertTrue(all(weight is getattr(model.lstm, name) for name, weight in
                                    zip(model.lstm._flat_weights_names, model.lstm._flat_weights)))
                self.assertTrue(all(value.dtype == torch.float32 for state in optimizer.state.values()
                                    for name, value in state.items() if name in ("exp_avg", "exp_avg_sq")))
                del prediction, loss, reference_sequence, reference_prediction, reference_loss
                gc.collect()
                self.assertTrue(all(reference() is None for reference in temporary_weights))
            self.assertEqual(observations, [(torch.bfloat16,) * 4] * 3)
            self.assertTrue(any(not torch.equal(parameter, original[name])
                                for name, parameter in model.named_parameters() if parameter.requires_grad))

    def test_forecasting_half_mse_loss_and_gradient_in_both_execution_paths(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        driver = self.driver_module()
        manifest = {"model": "lstm", "profile": {"hidden": 1}}
        for split in ("train", "test"):
            manifest[split] = {
                "x": array_file(root, f"{split}-x", np.asarray([[1], [2]], dtype="f4")),
                "y": array_file(root, f"{split}-y", np.asarray([[0], [1]], dtype="f4")),
            }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest))

        def known_model(_):
            model = torch.nn.Linear(1, 1, bias=False)
            with torch.no_grad():
                model.weight.fill_(2)
            return model

        environment = {
            "OPENNN_QUALITY_LOAD_CHECKPOINT": "",
            "OPENNN_QUALITY_INITIALIZATION": "",
            "OPENNN_QUALITY_INITIAL_PREDICTIONS": "",
            "OPENNN_QUALITY_EXPORT_INITIALIZATION": "",
            "OPENNN_QUALITY_SAVE_CHECKPOINT": "",
            "OPENNN_QUALITY_PROBE": "",
        }
        self.addCleanup(torch.set_num_threads, torch.get_num_threads())
        for mode, steps in (("diagnostic", "1"), ("training", "0")):
            with self.subTest(mode=mode):
                output = root / mode
                with patch.dict(os.environ, dict(environment, OPENNN_QUALITY_DIAGNOSTIC_STEPS=steps)), \
                     patch.object(driver, "build", side_effect=known_model), \
                     torch.random.fork_rng(devices=[]):
                    self.assertEqual(driver.main([
                        str(path), str(output), "1", "2", "42", "cpu", "fp32", ".001",
                    ]), 0)
                self.assertEqual(json.loads((output / "driver.json").read_text())["recurrent_precision_policy"], "fp32")
                # Predictions [2, 4] give residuals [2, 3]: half-MSE = 13/4.
                if mode == "diagnostic":
                    report = json.loads((output / "diagnostic.json").read_text())
                    self.assertEqual(report["steps"][0]["loss"], 3.25)
                    gradient_folder = output / "diagnostic/step_0/gradients"
                    tensors = json.loads((gradient_folder / "manifest.json").read_text())["tensors"]
                    gradient = next(item for item in tensors if item["name"] == "weight")
                    values = np.fromfile(gradient_folder / gradient["file"], dtype="<f4")
                    # dL/dw = mean(residual * input) = (2 + 6)/2 = 4.
                    np.testing.assert_array_equal(values, np.asarray([4], dtype="f4"))
                else:
                    with (output / "history.csv").open() as stream:
                        history = list(csv.DictReader(stream))
                    self.assertEqual(float(history[0]["training_loss"]), 3.25)


if __name__ == "__main__":
    unittest.main()
