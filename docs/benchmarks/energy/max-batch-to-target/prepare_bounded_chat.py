#!/usr/bin/env python3
"""Create a deterministic bounded-sequence Transformer benchmark corpus."""

import argparse
import hashlib
import json
from pathlib import Path

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pairs", type=int, default=4096)
    parser.add_argument("--prompt-words", type=int, default=24)
    parser.add_argument("--response-words", type=int, default=48)
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with source.open(encoding="utf-8") as input_stream, \
            output.open("w", encoding="utf-8", newline="\n") as output_stream:
        for line in input_stream:
            if written >= args.pairs:
                break
            try:
                prompt, response = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            prompt = " ".join(prompt.split()[:args.prompt_words])
            response = " ".join(response.split()[:args.response_words])
            if not prompt or not response:
                continue
            output_stream.write(f"{prompt}\t{response}\n")
            written += 1

    metadata = {
        "source": str(source.resolve()),
        "source_sha256": sha256(source),
        "output": str(output.resolve()),
        "output_sha256": sha256(output),
        "pairs_requested": args.pairs,
        "pairs_written": written,
        "prompt_words_max": args.prompt_words,
        "response_words_max": args.response_words,
        "selection": "first valid pairs in source order",
        "truncation": "whitespace-token prefix",
    }
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0 if written == args.pairs else 1

if __name__ == "__main__":
    raise SystemExit(main())
