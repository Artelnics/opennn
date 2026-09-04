"""Shared, dependency-light helpers for the Qwen benchmark family."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, specification: dict[str, Any]) -> dict[str, Any]:
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_bytes": int(specification["bytes"]),
        "expected_sha256": specification["sha256"],
    }
    if not path.is_file():
        return result | {"valid": False, "reason": "missing"}
    result["bytes"] = path.stat().st_size
    if result["bytes"] != result["expected_bytes"]:
        return result | {"valid": False, "reason": "size"}
    result["sha256"] = sha256_file(path)
    result["valid"] = result["sha256"] == result["expected_sha256"]
    if not result["valid"]:
        result["reason"] = "sha256"
    return result


def last_json(text: str) -> Any:
    """Return the last complete JSON line/document printed by a tool."""
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    for line in reversed(stripped.splitlines()):
        line = line.strip()
        if line.startswith(("{", "[")):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise ValueError("process did not emit JSON")


def raw_chatml(content: str) -> str:
    return ("<|im_start|>user\n" + content + "<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


def llama_bench_rows(payload: Any) -> list[dict[str, Any]]:
    """Normalize the JSON and JSONL variants emitted by llama-bench."""
    if isinstance(payload, dict):
        for key in ("results", "benchmarks", "data"):
            if isinstance(payload.get(key), list):
                return payload[key]
        return [payload]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError("llama-bench JSON is neither an object nor an array")


def llama_bench_samples(payload: Any, prompt: int,
                        generated: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Recover every timed pp/tg repetition from llama-bench JSON output."""
    rows = llama_bench_rows(payload)
    prefill = next((row for row in rows
                    if int(row.get("n_prompt", -1)) == prompt
                    and int(row.get("n_gen", 0)) == 0), None)
    decode = next((row for row in rows
                   if int(row.get("n_prompt", 0)) == 0
                   and int(row.get("n_gen", -1)) == generated), None)
    if prefill is None or decode is None:
        raise ValueError(f"llama-bench did not emit independent pp{prompt} and tg{generated} rows")

    def timings(row: dict[str, Any], tokens: int) -> list[tuple[float, float]]:
        nanoseconds = row.get("samples_ns")
        rates = row.get("samples_ts")
        if isinstance(nanoseconds, list) and nanoseconds:
            return [(float(ns) / 1e6, 1e9 * tokens / float(ns))
                    for ns in nanoseconds]
        if isinstance(rates, list) and rates:
            return [(1000.0 * tokens / float(rate), float(rate)) for rate in rates]
        rate = float(row["avg_ts"])
        return [(1000.0 * tokens / rate, rate)]

    pp = timings(prefill, prompt)
    tg = timings(decode, generated)
    if len(pp) != len(tg):
        raise ValueError("llama-bench pp/tg repetition counts differ")
    samples = [{
        "prompt_tokens": prompt,
        "generated_tokens": generated,
        "prefill_ms": prefill_ms,
        "decode_ms": decode_ms,
        "prefill_tokens_per_second": prefill_rate,
        "decode_tokens_per_second": decode_rate,
    } for (prefill_ms, prefill_rate), (decode_ms, decode_rate) in zip(pp, tg)]
    return samples, rows


def sample_statistics(values: Iterable[float]) -> dict[str, Any]:
    data = [float(value) for value in values if math.isfinite(float(value))]
    if not data:
        return {"count": 0, "median": None, "min": None, "max": None, "cv": None}
    mean = statistics.fmean(data)
    cv = statistics.pstdev(data) / mean if len(data) > 1 and mean else 0.0
    return {
        "count": len(data),
        "median": statistics.median(data),
        "min": min(data),
        "max": max(data),
        "cv": cv,
    }


def common_prefix(strings: Iterable[str]) -> int:
    values = list(strings)
    if not values:
        return 0
    limit = min(map(len, values))
    for index in range(limit):
        if len({value[index] for value in values}) != 1:
            return index
    return limit
