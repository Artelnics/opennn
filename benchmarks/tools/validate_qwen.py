#!/usr/bin/env python3
"""Prove that the OpenNN and GGUF Qwen3-4B files contain one BF16 model.

The OpenNN file is a headerless stream in layer/parameter order.  It adds a
zero padding row to the embedding and serializes the tied output projection;
neither is part of the canonical logical parameter count.  This validator
names and compares every logical tensor rather than treating equal file sizes
or model names as evidence.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST_PATH = HERE.parent / "manifests" / "qwen_manifest.json"
sys.path.insert(0, str(HERE))
from qwen_support import verify_file  # noqa: E402


class SafeTensorStore:
    def __init__(self, directory: Path):
        index = json.loads((directory / "model.safetensors.index.json").read_text())
        self.directory = directory
        self.weight_map: dict[str, str] = index["weight_map"]
        self.headers: dict[str, tuple[int, dict[str, Any]]] = {}

    def _header(self, filename: str) -> tuple[int, dict[str, Any]]:
        if filename not in self.headers:
            path = self.directory / filename
            with path.open("rb") as handle:
                header_bytes = struct.unpack("<Q", handle.read(8))[0]
                header = json.loads(handle.read(header_bytes))
            self.headers[filename] = (8 + header_bytes, header)
        return self.headers[filename]

    def bits(self, name: str) -> np.ndarray:
        filename = self.weight_map[name]
        data_start, header = self._header(filename)
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"canonical tensor {name} is {entry['dtype']}, expected BF16")
        begin, end = entry["data_offsets"]
        count = (end - begin) // 2
        return np.memmap(self.directory / filename, dtype="<u2", mode="r",
                         offset=data_start + begin, shape=(count,)).reshape(entry["shape"])

    def shape(self, name: str) -> tuple[int, ...]:
        filename = self.weight_map[name]
        _, header = self._header(filename)
        return tuple(header[name]["shape"])

    def logical_parameters(self) -> int:
        return sum(int(np.prod(self.shape(name))) for name in self.weight_map)


def chunks(array: np.ndarray, elements: int = 8 << 20) -> Iterator[np.ndarray]:
    flat = array.reshape(-1)
    for begin in range(0, flat.size, elements):
        yield np.asarray(flat[begin:begin + elements], dtype=np.uint16)


def compare_flat(actual: np.ndarray, expected: np.ndarray) -> tuple[bool, int | None]:
    if actual.size != expected.size:
        return False, 0
    offset = 0
    for expected_chunk in chunks(expected):
        actual_chunk = np.asarray(actual[offset:offset + expected_chunk.size], dtype=np.uint16)
        mismatch = np.flatnonzero(actual_chunk != expected_chunk)
        if mismatch.size:
            return False, offset + int(mismatch[0])
        offset += expected_chunk.size
    return True, None


def compare_transposed(actual: np.ndarray, expected: np.ndarray,
                       rows_per_chunk: int = 256) -> tuple[bool, int | None]:
    if expected.ndim != 2 or actual.size != expected.size:
        return False, 0
    output_columns = expected.shape[0]
    offset = 0
    for begin in range(0, expected.shape[1], rows_per_chunk):
        end = min(expected.shape[1], begin + rows_per_chunk)
        block = np.asarray(expected[:, begin:end].T, dtype=np.uint16).reshape(-1)
        candidate = np.asarray(actual[offset:offset + block.size], dtype=np.uint16)
        mismatch = np.flatnonzero(candidate != block)
        if mismatch.size:
            return False, offset + int(mismatch[0])
        offset += block.size
    assert offset == expected.size == expected.shape[1] * output_columns
    return True, None


def validate_opennn(store: SafeTensorStore, raw_path: Path) -> dict[str, Any]:
    raw = np.memmap(raw_path, dtype="<u2", mode="r")
    cursor = 0
    checks: list[dict[str, Any]] = []

    def consume(label: str, canonical: str, transpose: bool = False) -> None:
        nonlocal cursor
        expected = store.bits(canonical)
        count = expected.size
        actual = raw[cursor:cursor + count]
        valid, mismatch = (compare_transposed(actual, expected) if transpose
                           else compare_flat(actual, expected))
        checks.append({"storage": label, "canonical": canonical,
                       "elements": int(count), "transpose": transpose,
                       "valid": valid, "first_mismatch": mismatch})
        cursor += count

    embedding = store.bits("model.embed_tokens.weight")
    hidden = embedding.shape[1]
    padding = raw[cursor:cursor + hidden]
    checks.append({"storage": "embed_tokens.padding_row", "elements": int(hidden),
                   "valid": bool(np.all(padding == 0)),
                   "first_mismatch": None if np.all(padding == 0) else int(np.flatnonzero(padding)[0])})
    cursor += hidden
    consume("embed_tokens.weight", "model.embed_tokens.weight")

    for layer in range(36):
        prefix = f"model.layers.{layer}"
        consume(f"input_norm_{layer}", f"{prefix}.input_layernorm.weight")
        consume(f"attn_{layer}.q_proj", f"{prefix}.self_attn.q_proj.weight")
        consume(f"attn_{layer}.k_proj", f"{prefix}.self_attn.k_proj.weight")
        consume(f"attn_{layer}.v_proj", f"{prefix}.self_attn.v_proj.weight")
        consume(f"attn_{layer}.o_proj", f"{prefix}.self_attn.o_proj.weight")
        consume(f"attn_{layer}.q_norm", f"{prefix}.self_attn.q_norm.weight")
        consume(f"attn_{layer}.k_norm", f"{prefix}.self_attn.k_norm.weight")
        consume(f"post_norm_{layer}", f"{prefix}.post_attention_layernorm.weight")
        consume(f"gate_up_{layer}.gate", f"{prefix}.mlp.gate_proj.weight", True)
        consume(f"gate_up_{layer}.up", f"{prefix}.mlp.up_proj.weight", True)
        consume(f"down_{layer}", f"{prefix}.mlp.down_proj.weight", True)

    consume("final_norm", "model.norm.weight")

    # The tied lm_head occupies [hidden, vocab+1] in OpenNN storage.
    tied_count = (embedding.shape[0] + 1) * hidden
    tied = raw[cursor:cursor + tied_count]
    extended = np.empty((embedding.shape[0] + 1, hidden), dtype=np.uint16)
    extended[0].fill(0)
    for begin in range(0, embedding.shape[0], 8192):
        end = min(embedding.shape[0], begin + 8192)
        extended[begin + 1:end + 1] = embedding[begin:end]
    valid, mismatch = compare_transposed(tied, extended)
    checks.append({"storage": "lm_head.tied_copy", "canonical": "model.embed_tokens.weight",
                   "elements": int(tied_count), "transpose": True,
                   "excluded_from_logical_parameters": True,
                   "valid": valid, "first_mismatch": mismatch})
    cursor += tied_count

    return {
        "valid": cursor == raw.size and all(item["valid"] for item in checks),
        "serialized_elements": int(raw.size),
        "consumed_elements": int(cursor),
        "logical_elements_compared": int(sum(
            item["elements"] for item in checks
            if not item.get("excluded_from_logical_parameters")
            and item["storage"] != "embed_tokens.padding_row")),
        "checks": checks,
    }


def _gguf_expected_names() -> dict[str, str]:
    names = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
    }
    for layer in range(36):
        canonical = f"model.layers.{layer}"
        ggml = f"blk.{layer}"
        names.update({
            f"{ggml}.attn_norm.weight": f"{canonical}.input_layernorm.weight",
            f"{ggml}.attn_q.weight": f"{canonical}.self_attn.q_proj.weight",
            f"{ggml}.attn_k.weight": f"{canonical}.self_attn.k_proj.weight",
            f"{ggml}.attn_v.weight": f"{canonical}.self_attn.v_proj.weight",
            f"{ggml}.attn_output.weight": f"{canonical}.self_attn.o_proj.weight",
            f"{ggml}.attn_q_norm.weight": f"{canonical}.self_attn.q_norm.weight",
            f"{ggml}.attn_k_norm.weight": f"{canonical}.self_attn.k_norm.weight",
            f"{ggml}.ffn_norm.weight": f"{canonical}.post_attention_layernorm.weight",
            f"{ggml}.ffn_gate.weight": f"{canonical}.mlp.gate_proj.weight",
            f"{ggml}.ffn_up.weight": f"{canonical}.mlp.up_proj.weight",
            f"{ggml}.ffn_down.weight": f"{canonical}.mlp.down_proj.weight",
        })
    return names


def _gguf_bits(data: np.ndarray, tensor_type: Any) -> tuple[np.ndarray, str, bool]:
    array = np.asarray(data)
    if array.dtype == np.float32:
        raw = array.view(np.uint32)
        exact_widening = bool(np.all((raw & np.uint32(0xffff)) == 0))
        return (raw >> np.uint32(16)).astype(np.uint16), "f32", exact_widening
    # GGUFReader exposes unquantized BF16 payloads as byte arrays.  The
    # tensor type, rather than numpy's dtype, is the authoritative encoding.
    # Viewing pairs of bytes preserves the source BF16 bit patterns exactly.
    if int(tensor_type) == 30:  # GGMLQuantizationType.BF16
        return array.view("<u2"), "bf16", True
    if array.dtype.itemsize == 2:
        return array.view(np.uint16), "bf16", True
    return np.empty(0, dtype=np.uint16), str(array.dtype), False


def validate_gguf(store: SafeTensorStore, gguf_path: Path, llama: Path) -> dict[str, Any]:
    sys.path.insert(0, str(llama / "gguf-py"))
    import gguf  # type: ignore

    reader = gguf.GGUFReader(str(gguf_path), "r")
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    mapping = _gguf_expected_names()
    checks: list[dict[str, Any]] = []

    for name, canonical in mapping.items():
        tensor = tensors.get(name)
        if tensor is None:
            checks.append({"storage": name, "canonical": canonical,
                           "valid": False, "reason": "missing"})
            continue
        actual, dtype, exact = _gguf_bits(tensor.data, tensor.tensor_type)
        expected = store.bits(canonical)
        valid, mismatch = compare_flat(actual.reshape(-1), expected)
        orientation = "direct"
        if not valid and expected.ndim == 2:
            valid, mismatch = compare_transposed(actual.reshape(-1), expected)
            orientation = "transposed"
        checks.append({
            "storage": name, "canonical": canonical,
            "elements": int(expected.size), "storage_type": dtype,
            "exact_bf16_or_widening": exact, "orientation": orientation,
            "valid": bool(valid and exact), "first_mismatch": mismatch,
        })

    unexpected = sorted(set(tensors) - set(mapping) - {"output.weight"})
    return {
        "valid": not unexpected and len(tensors) in (len(mapping), len(mapping) + 1)
                 and all(item["valid"] for item in checks),
        "tensor_count": len(tensors),
        "unexpected_tensors": unexpected,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--write-report", type=Path)
    parser.add_argument("--skip-values", action="store_true",
                        help="validate manifests only (fast diagnostics)")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST_PATH.read_text())
    root = args.data_root / "qwen3"
    opennn_dir = root / "models" / "opennn"
    canonical_dir = root / "models" / "canonical"
    files = {
        "opennn": {name: verify_file(opennn_dir / name, spec)
                    for name, spec in manifest["opennn_model"]["files"].items()},
        "canonical": {name: verify_file(canonical_dir / name, spec)
                       for name, spec in manifest["canonical_model"]["files"].items()},
    }
    file_gate = all(result["valid"] for group in files.values() for result in group.values())
    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(MANIFEST_PATH),
        "files_valid": file_gate,
        "files": files,
    }

    if file_gate and not args.skip_values:
        store = SafeTensorStore(canonical_dir)
        report["canonical_logical_parameters"] = store.logical_parameters()
        report["logical_parameter_gate"] = (
            report["canonical_logical_parameters"] == manifest["logical_parameters"])
        report["opennn_values"] = validate_opennn(
            store, opennn_dir / "qwen3_bf16.bin")
        report["gguf_values"] = validate_gguf(
            store, root / "models" / "qwen3-4b-bf16.gguf",
            root / "tools" / "llama.cpp")
    else:
        report["logical_parameter_gate"] = False

    report["valid"] = bool(
        report["files_valid"] and report["logical_parameter_gate"]
        and report.get("opennn_values", {}).get("valid")
        and report.get("gguf_values", {}).get("valid"))

    encoded = json.dumps(report, indent=2)
    if args.write_report:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(encoded, encoding="utf-8")
    if args.write_report:
        print(json.dumps({
            "validation_report": str(args.write_report),
            "files_valid": report["files_valid"],
            "logical_parameter_gate": report["logical_parameter_gate"],
            "opennn_values_valid": report.get("opennn_values", {}).get("valid"),
            "gguf_values_valid": report.get("gguf_values", {}).get("valid"),
            "valid": report["valid"],
        }, indent=2))
    else:
        print(encoded)
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
