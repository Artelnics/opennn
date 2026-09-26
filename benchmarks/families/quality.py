#!/usr/bin/env python3
"""PyTorch Python counterpart of quality.cpp; scoring lives in quality_runner.py."""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from quality_data import read_array
from quality_initialization import export_model_tensors, load_initialization


class Forecaster(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.lstm.bias_hh_l0.data.zero_()
        self.lstm.bias_hh_l0.requires_grad_(False)
        self.head = nn.Linear(hidden, 1)
        self.explicit_recurrent_bf16 = False

    def forward(self, x):
        if (
            self.explicit_recurrent_bf16
            and x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
        ):
            # CUDA autocast can dispatch cuDNN RNNs in FP16 even when BF16 was
            # requested. Explicit differentiable casts keep FP32 master weights.
            with torch.autocast("cuda", enabled=False):
                parameters = {
                    name: parameter.to(torch.bfloat16)
                    for name, parameter in self.lstm.named_parameters()
                }
                try:
                    sequence = torch.func.functional_call(
                        self.lstm, parameters, (x.to(torch.bfloat16),), strict=True
                    )[0]
                finally:
                    # functional_call restores parameters, but RNNBase caches
                    # their temporary flattened views until its next forward.
                    self.lstm._update_flat_weights()
        else:
            sequence = self.lstm(x)[0]
        return self.head(sequence[:, -1])


class Translator(nn.Module):
    def __init__(self, vocabulary, cfg):
        super().__init__()
        width, heads, ff = cfg["d_model"], cfg["heads"], cfg["ff"]
        self.source = nn.Embedding(vocabulary, width)
        self.target = nn.Embedding(vocabulary, width)
        enc = nn.TransformerEncoderLayer(
            width, heads, ff, dropout=0, batch_first=True, layer_norm_eps=1e-6
        )
        dec = nn.TransformerDecoderLayer(
            width, heads, ff, dropout=0, batch_first=True, layer_norm_eps=1e-6
        )
        self.encoder = nn.TransformerEncoder(
            enc, cfg["layers"], norm=None, enable_nested_tensor=False
        )
        self.decoder = nn.TransformerDecoder(dec, cfg["layers"], norm=None)
        self.output = nn.Linear(width, vocabulary)
        position = torch.arange(cfg["sequence"]).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, width, 2).float() * (-math.log(10000.0) / width)
        )
        encoding = torch.zeros(cfg["sequence"], width)
        encoding[:, 0::2], encoding[:, 1::2] = (
            torch.sin(position * div),
            torch.cos(position * div),
        )
        self.register_buffer("position", encoding)
        self.scale = math.sqrt(width)

    def forward(self, source, decoder):
        s = self.source(source.long()) * self.scale + self.position
        d = self.target(decoder.long()) * self.scale + self.position
        mask = torch.ones(
            decoder.shape[1], decoder.shape[1], dtype=torch.bool, device=decoder.device
        ).triu(1)
        memory = self.encoder(s, src_key_padding_mask=source == 0)
        return self.output(
            self.decoder(
                d,
                memory,
                tgt_mask=mask,
                tgt_key_padding_mask=decoder == 0,
                memory_key_padding_mask=source == 0,
            )
        )


def build(manifest):
    kind, cfg = manifest["model"], manifest["profile"]
    if kind == "dense":
        layers = []
        width = manifest["train"]["x"]["shape"][-1]
        for _ in range(cfg["layers"]):
            layers.extend([nn.Linear(width, cfg["hidden"]), nn.ReLU()])
            width = cfg["hidden"]
        layers.append(nn.Linear(width, 1))
        model = nn.Sequential(*layers)
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        return model
    if kind == "lstm":
        return Forecaster(manifest["train"]["x"]["shape"][-1], cfg["hidden"])
    if kind == "cnn":
        from torchvision.models.resnet import ResNet, Bottleneck

        return ResNet(
            Bottleneck, cfg["blocks"], num_classes=manifest["train"]["y"]["shape"][-1]
        )
    return Translator(len(manifest["vocabulary"]), cfg)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 8:
        raise ValueError(
            "usage: quality.py manifest out epochs batch seed cpu|cuda fp32|bf16 learning_rate"
        )
    manifest_path, out = Path(argv[0]), Path(argv[1])
    epochs, batch, seed = map(int, argv[2:5])
    device, precision = argv[5:7]
    rate = float(argv[7])
    if (
        epochs < 1
        or batch < 1
        or device not in ("cpu", "cuda")
        or precision not in ("fp32", "bf16")
    ):
        raise ValueError("Invalid quality options")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device == "cpu" and precision != "fp32":
        raise ValueError("CPU quality uses FP32")
    torch.manual_seed(seed)
    torch.set_num_threads(
        int(os.environ.get("TORCH_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "2")))
    )
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    deterministic = os.environ.get("OPENNN_QUALITY_DETERMINISTIC") == "1"
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    manifest = json.loads(manifest_path.read_text())
    kind = manifest["model"]
    data = {
        split: {
            key: read_array(manifest_path.parent, spec)
            for key, spec in manifest[split].items()
        }
        for split in ("train", "test")
    }
    model = build(manifest).to(device)
    recurrent_metadata = {}
    if kind == "lstm":
        model.explicit_recurrent_bf16 = device == "cuda" and precision == "bf16"
        recurrent_metadata["recurrent_precision_policy"] = (
            "explicit_bf16_with_fp32_masters" if model.explicit_recurrent_bf16 else "fp32"
        )
    checkpoint = os.environ.get("OPENNN_QUALITY_LOAD_CHECKPOINT")
    shared_initialization = os.environ.get("OPENNN_QUALITY_INITIALIZATION")
    if checkpoint and shared_initialization:
        raise ValueError("Checkpoint evaluation and shared initialization are mutually exclusive")
    initialization = (
        load_initialization(model, manifest, shared_initialization, expected_seed=seed)
        if shared_initialization
        else {"mode": "native"}
    )
    evaluation_only = bool(checkpoint)
    if evaluation_only:
        model.load_state_dict(
            torch.load(
                Path(checkpoint) / "parameters.pt",
                map_location=device,
                weights_only=True,
            )
        )
    optimizer_name = os.environ.get("OPENNN_QUALITY_OPTIMIZER", "adam")
    momentum = float(os.environ.get("OPENNN_QUALITY_MOMENTUM", "0.9"))
    weight_decay = float(np.float32(
        os.environ.get("OPENNN_QUALITY_WEIGHT_DECAY", "0")
    ))
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("Weight decay must be finite and nonnegative")
    learning_rate_milestones = [
        int(value) for value in
        os.environ.get("OPENNN_QUALITY_LR_MILESTONES", "").split(",") if value
    ]
    learning_rate_gamma = float(np.float32(
        os.environ.get("OPENNN_QUALITY_LR_GAMMA", "0.1")
    ))
    if (learning_rate_milestones != sorted(set(learning_rate_milestones))
            or any(not 0 < epoch < epochs for epoch in learning_rate_milestones)):
        raise ValueError("Learning-rate milestones must be distinct, increasing, and inside the training budget")
    if not math.isfinite(learning_rate_gamma) or not 0 < learning_rate_gamma < 1:
        raise ValueError("Learning-rate gamma must be finite and in (0, 1)")
    if learning_rate_milestones and optimizer_name != "sgd":
        raise ValueError("Learning-rate milestones currently require SGD")
    adam_epsilon = float(np.float32(
        os.environ.get("OPENNN_QUALITY_ADAM_EPSILON", np.finfo(np.float32).eps)
    ))
    if not math.isfinite(adam_epsilon) or adam_epsilon <= 0:
        raise ValueError("Adam epsilon must be finite and positive")
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=rate, betas=(0.9, 0.999),
            eps=adam_epsilon, weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=rate, momentum=momentum,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported quality optimizer: {optimizer_name}")
    out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("OPENNN_QUALITY_EXPORT_INITIALIZATION"):
        export_model_tensors(model, manifest, out / "initialization", seed)

    def context():
        return (
            torch.autocast(device_type=device, dtype=torch.bfloat16)
            if precision == "bf16"
            else contextlib.nullcontext()
        )

    def tensor(values):
        return torch.from_numpy(np.array(values, dtype=np.float32, copy=True)).to(
            device
        )

    def inputs(split, start, end):
        x = tensor(data[split]["x"][start:end])
        return x.permute(0, 3, 1, 2).contiguous() / 255 if kind == "cnn" else x

    if os.environ.get("OPENNN_QUALITY_INITIAL_PREDICTIONS"):
        count = min(8, len(data["test"]["x"]))
        model.eval()
        with torch.inference_mode(), context():
            x = inputs("test", 0, count)
            if kind == "transformer":
                initial_output = model(x, tensor(data["test"]["decoder"][:count]))
            else:
                initial_output = model(x)
                if kind == "dense":
                    initial_output = initial_output.sigmoid()
                elif kind == "cnn":
                    initial_output = initial_output.softmax(-1)
        if not torch.isfinite(initial_output).all():
            raise RuntimeError("Nonfinite initial prediction")
        initial_output.float().cpu().numpy().astype("<f4").tofile(
            out / "initial_predictions.bin"
        )
        (out / "initial_predictions.json").write_text(
            json.dumps(
                {
                    "shape": list(initial_output.shape),
                    "dtype": "float32",
                    "samples": count,
                    "output": "teacher_forced_logits" if kind == "transformer" else
                    "regression" if kind == "lstm" else "probabilities",
                    "mode": "evaluation",
                },
                indent=2,
            )
        )

    diagnostic_steps = int(os.environ.get("OPENNN_QUALITY_DIAGNOSTIC_STEPS", "0"))
    if diagnostic_steps < 0:
        raise ValueError("Diagnostic step count must be nonnegative")
    if diagnostic_steps:
        if evaluation_only:
            raise ValueError("Checkpoint evaluation cannot perform training diagnostics")
        model.train()
        count = min(batch, len(data["train"]["x"]))
        x, y = inputs("train", 0, count), tensor(data["train"]["y"][:count])
        decoder = tensor(data["train"]["decoder"][:count]) if kind == "transformer" else None
        diagnostic = {"mode": "fixed_batch_training", "steps": []}
        for step in range(diagnostic_steps):
            folder = out / "diagnostic" / f"step_{step}"
            folder.mkdir(parents=True, exist_ok=True)
            optimizer.zero_grad(set_to_none=True)
            with context():
                if kind == "transformer":
                    output = model(x, decoder)
                    loss = nn.functional.cross_entropy(
                        output.reshape(-1, output.shape[-1]), y.long().reshape(-1),
                        ignore_index=0,
                    )
                else:
                    output = model(x)
                    if kind == "dense":
                        loss = nn.functional.binary_cross_entropy_with_logits(output, y)
                    elif kind == "cnn":
                        loss = nn.functional.cross_entropy(output, y.argmax(1))
                    else:
                        # OpenNN's one-output squared error is sum/(2*batch).
                        loss = 0.5 * nn.functional.mse_loss(output, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite diagnostic loss")
            loss.backward()
            export_model_tensors(model, manifest, folder / "gradients", seed, gradients=True)
            observed = output.detach()
            if kind == "dense":
                observed = observed.sigmoid()
            elif kind == "cnn":
                observed = observed.softmax(-1)
            if not torch.isfinite(observed).all():
                raise RuntimeError("Nonfinite diagnostic output")
            observed.float().cpu().numpy().astype("<f4").tofile(folder / "outputs.bin")
            (folder / "outputs.json").write_text(json.dumps({
                "shape": list(observed.shape),
                "dtype": "float32",
                "samples": count,
                "output": "logits" if kind == "transformer" else
                "regression" if kind == "lstm" else "probabilities",
                "mode": "training",
            }, indent=2))
            optimizer.step()
            export_model_tensors(model, manifest, folder / "parameters", seed)
            diagnostic["steps"].append({"step": step, "loss": float(loss.detach()), "samples": count})
        (out / "diagnostic.json").write_text(json.dumps(diagnostic, indent=2))
        (out / "driver.json").write_text(json.dumps({
            "engine": "pytorch", "interface": "python", "model": kind,
            "device": device, "precision": precision, "seed": seed,
            "deterministic": deterministic,
            "optimizer": optimizer_name,
            "adam_epsilon": adam_epsilon,
            "batchnorm_recalibrated": False,
            "epochs": 0,
            "initialization": initialization, "diagnostic_only": True,
            "diagnostic_steps": diagnostic_steps, "status": "ok",
            **recurrent_metadata,
        }, indent=2))
        print("RESULT=DIAGNOSTIC", flush=True)
        return 0

    with (out / "history.csv").open("w") as history:
        history.write("epoch,training_loss\n")
        for epoch in range(0 if evaluation_only else epochs):
            model.train()
            total = 0.0
            weight = 0
            for start in range(0, len(data["train"]["x"]), batch):
                end = min(start + batch, len(data["train"]["x"]))
                x, y = (
                    inputs("train", start, end),
                    tensor(data["train"]["y"][start:end]),
                )
                optimizer.zero_grad(set_to_none=True)
                with context():
                    if kind == "transformer":
                        output = model(x, tensor(data["train"]["decoder"][start:end]))
                        loss = nn.functional.cross_entropy(
                            output.reshape(-1, output.shape[-1]),
                            y.long().reshape(-1),
                            ignore_index=0,
                        )
                        n = int((y != 0).sum())
                    else:
                        output = model(x)
                        n = end - start
                        if kind == "dense":
                            loss = nn.functional.binary_cross_entropy_with_logits(
                                output, y
                            )
                        elif kind == "cnn":
                            loss = nn.functional.cross_entropy(output, y.argmax(1))
                        else:
                            loss = 0.5 * nn.functional.mse_loss(output, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * n
                weight += n
            history.write(f"{epoch + 1},{total / weight:.9g}\n")
            history.flush()
            print(f"epoch={epoch + 1} training_loss={total / weight:.9g}", flush=True)
            if epoch + 1 in learning_rate_milestones:
                for group in optimizer.param_groups:
                    group["lr"] *= learning_rate_gamma
    if not evaluation_only and os.environ.get("OPENNN_QUALITY_SAVE_CHECKPOINT"):
        torch.save(model.state_dict(), out / "parameters.pt")
    batchnorm_recalibrated = (
        kind == "cnn" and os.environ.get("OPENNN_QUALITY_RECALIBRATE_BATCHNORM") == "1"
    )
    if batchnorm_recalibrated:
        normalizations = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
        for normalization in normalizations:
            normalization.reset_running_stats()
        seen = 0
        model.train()
        with torch.inference_mode(), context():
            for start in range(0, len(data["train"]["x"]), batch):
                end = min(start + batch, len(data["train"]["x"]))
                count = end - start
                seen += count
                for normalization in normalizations:
                    normalization.momentum = count / seen
                model(inputs("train", start, end))
        for normalization in normalizations:
            normalization.momentum = 0.1
    if os.environ.get("OPENNN_QUALITY_EXPORT_FINAL"):
        export_model_tensors(model, manifest, out / "final", seed)
    model.eval()
    if kind == "transformer" and os.environ.get("OPENNN_QUALITY_PROBE"):
        probe = {}
        with torch.inference_mode():
            for split in ("train", "test"):
                count = min(8, len(data[split]["x"]))
                with context():
                    logits = model(
                        inputs(split, 0, count),
                        tensor(data[split]["decoder"][:count]),
                    )
                targets = tensor(data[split]["y"][:count]).long()
                active = targets != 0
                active_count = int(active.sum())
                if active_count == 0:
                    raise RuntimeError("Teacher-forced probe has no target tokens")
                predictions = logits.argmax(-1)[active]
                unique, counts = predictions.unique(return_counts=True)
                dominant = counts.argmax()
                probe[split] = {
                    "samples": count,
                    "active_tokens": active_count,
                    "mean_cross_entropy": float(
                        nn.functional.cross_entropy(
                            logits.reshape(-1, logits.shape[-1]),
                            targets.reshape(-1),
                            ignore_index=0,
                            reduction="sum",
                        )
                    ) / active_count,
                    "token_accuracy_percent": float(
                        (predictions == targets[active]).sum()
                    ) * 100 / active_count,
                    "dominant_prediction_id": int(unique[dominant]),
                    "dominant_prediction_fraction": int(counts[dominant])
                    / active_count,
                    "unique_prediction_ids": int(len(unique)),
                }
        (out / "teacher_forced_probe.json").write_text(json.dumps(probe, indent=2))
    with torch.inference_mode(), (out / "predictions.bin").open("wb") as predictions:
        for start in range(0, len(data["test"]["x"]), batch):
            end = min(start + batch, len(data["test"]["x"]))
            x = inputs("test", start, end)
            with context():
                if kind == "transformer":
                    sequence = manifest["profile"]["sequence"]
                    decoder = torch.zeros((end - start, sequence), device=device)
                    decoder[:, 0] = 2
                    output = torch.zeros_like(decoder)
                    done = torch.zeros(end - start, dtype=torch.bool, device=device)
                    for position in range(sequence):
                        logits = model(x, decoder)
                        if not torch.isfinite(logits).all():
                            raise RuntimeError("Nonfinite generation logits")
                        token = logits[:, position].argmax(-1)
                        output[:, position] = torch.where(done, 0, token)
                        if position + 1 < sequence:
                            decoder[:, position + 1] = torch.where(done, 0, token)
                        done |= token == 3
                        if done.all():
                            break
                else:
                    output = model(x)
                    if kind == "dense":
                        output = output.sigmoid()
                    elif kind == "cnn":
                        output = output.softmax(-1)
            if not torch.isfinite(output).all():
                raise RuntimeError("Nonfinite prediction")
            output.float().cpu().numpy().astype("<f4").tofile(predictions)
    record = {
        "engine": "pytorch",
        "interface": "python",
        "device": device,
        "precision": precision,
        "model": kind,
        "seed": seed,
        "deterministic": deterministic,
        "optimizer": optimizer_name,
        "adam_epsilon": adam_epsilon,
        "weight_decay": weight_decay,
        "learning_rate_milestones": learning_rate_milestones,
        "learning_rate_gamma": learning_rate_gamma,
        "batchnorm_recalibrated": batchnorm_recalibrated,
        "epochs": epochs,
        "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_samples": len(data["train"]["x"]),
        "test_samples": len(data["test"]["x"]),
        "evaluation_only": evaluation_only,
        "initialization": initialization,
        "status": "ok",
        **recurrent_metadata,
    }
    (out / "driver.json").write_text(json.dumps(record, indent=2))
    print("RESULT=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
