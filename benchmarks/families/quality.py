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


class Forecaster(nn.Module):
    def __init__(self, features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(features, hidden, batch_first=True)
        self.lstm.bias_hh_l0.data.zero_()
        self.lstm.bias_hh_l0.requires_grad_(False)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.head(self.lstm(x)[0][:, -1])


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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=rate,
        betas=(0.9, 0.999),
        eps=float(np.finfo(np.float32).eps),
    )
    out.mkdir(parents=True, exist_ok=True)

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

    with (out / "history.csv").open("w") as history:
        history.write("epoch,training_loss\n")
        for epoch in range(epochs):
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
                            loss = nn.functional.mse_loss(output, y)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * n
                weight += n
            history.write(f"{epoch + 1},{total / weight:.9g}\n")
            history.flush()
            print(f"epoch={epoch + 1} training_loss={total / weight:.9g}", flush=True)
    model.eval()
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
        "epochs": epochs,
        "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_samples": len(data["train"]["x"]),
        "test_samples": len(data["test"]["x"]),
        "status": "ok",
    }
    (out / "driver.json").write_text(json.dumps(record, indent=2))
    print("RESULT=OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
