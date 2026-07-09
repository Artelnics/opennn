#!/usr/bin/env python3
"""Prepare the shared chat corpus under $OPENNN_BENCH_DATA/chat/.

Downloads the Stanford Alpaca instruction dataset and writes a
`prompt<TAB>response` pair file (`chat_pairs.txt`) — the corpus used by the
transformer energy-to-target and transformer max-batch benchmarks. Large data
stays outside the repository; see ../DATA_POLICY.md.

    export OPENNN_BENCH_DATA="$HOME/opennn-benchmark-data"
    python prepare_chat.py                 # -> $OPENNN_BENCH_DATA/chat/chat_pairs.txt
    python prepare_chat.py --raw alpaca_data.json   # use a local copy instead of downloading
"""
import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

ALPACA_URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"


def data_root() -> Path:
    return Path(os.environ.get("OPENNN_BENCH_DATA", str(Path.home() / "opennn-benchmark-data")))


def load_alpaca(raw: str | None) -> list[dict]:
    if raw:
        return json.loads(Path(raw).read_text(encoding="utf-8"))
    print(f"Downloading Stanford Alpaca from {ALPACA_URL} ...", file=sys.stderr)
    with urllib.request.urlopen(ALPACA_URL) as resp:
        return json.loads(resp.read().decode("utf-8"))


def to_pairs(records: list[dict]):
    for r in records:
        instruction = (r.get("instruction") or "").strip()
        extra = (r.get("input") or "").strip()
        response = (r.get("output") or "").strip()
        prompt = f"{instruction}\n{extra}".strip() if extra else instruction
        # tab-separated single-line pairs; drop rows that would break the format
        prompt = " ".join(prompt.split())
        response = " ".join(response.split())
        if prompt and response:
            yield prompt, response


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(data_root() / "chat"),
                    help="output directory (default: $OPENNN_BENCH_DATA/chat)")
    ap.add_argument("--raw", default=None,
                    help="local alpaca_data.json instead of downloading")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chat_pairs.txt"

    records = load_alpaca(args.raw)
    n = 0
    with out_file.open("w", encoding="utf-8") as fh:
        for prompt, response in to_pairs(records):
            fh.write(f"{prompt}\t{response}\n")
            n += 1

    meta = {"source": "stanford_alpaca", "url": ALPACA_URL, "pairs": n,
            "format": "prompt<TAB>response"}
    (out_dir / "chat_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {n} prompt/response pairs to {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
