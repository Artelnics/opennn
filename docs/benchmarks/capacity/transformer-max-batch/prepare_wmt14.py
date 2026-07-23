#!/usr/bin/env python3
"""Prepare a WMT14 En-De pair corpus for the transformer max-batch benchmark.

WMT 2014 English-German is the dataset the *Attention Is All You Need* base
model (d512/h8/ff2048/6L) was trained and reported on, which makes it the
standard, citable corpus for this benchmark. This script downloads the
Stanford NMT preprocessed WMT14 En-De training text (~4.5M sentence pairs,
plain UTF-8, one sentence per line) and writes the `source <TAB> target` pair
file that `TextDataset` / run_transformer_maxbatch.py consume -- the same
format as the Alpaca chat_pairs.txt corpus.

Capacity depends on the corpus only through the derived vocabulary (capped at
30000 by the OpenNN trial) and sequence lengths, so the prepared file is
bounded on both axes:

  --max-tokens  truncates each side to N whitespace tokens (default 128, the
                conventional cap for the paper-base model), which bounds the
                padded sequence lengths every engine builds; and
  --max-pairs   keeps the first N valid pairs (default 200000) -- enough to
                saturate a 30k vocabulary cap while keeping the per-trial
                corpus load time small (every fresh-process trial re-reads
                the corpus).

Files land under OPENNN_BENCH_DATA (never in the repository); see
docs/benchmarks/DATA_POLICY.md.

  usage: python prepare_wmt14.py [--out PATH] [--max-pairs N] [--max-tokens N]
             [--train-en PATH] [--train-de PATH]

A `wmt14_metadata.json` is written next to the pair file recording the source
files, filters, and counts.
"""

import argparse
import json
import os
import sys
import urllib.request

STANFORD_BASE = "https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de"
TRAIN_EN_URL = f"{STANFORD_BASE}/train.en"
TRAIN_DE_URL = f"{STANFORD_BASE}/train.de"


def default_out():
    root = os.environ.get("OPENNN_BENCH_DATA")
    if not root:
        sys.exit("Set OPENNN_BENCH_DATA (large files live outside the repo; "
                 "see docs/benchmarks/DATA_POLICY.md) or pass --out.")
    return os.path.join(root, "wmt14", "wmt14_en_de_pairs.txt")


def download(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"reusing {dest}")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")

    def hook(blocks, block_size, total):
        done = blocks * block_size
        if total > 0 and blocks % 4096 == 0:
            print(f"  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="\r")

    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp, hook)
    os.replace(tmp, dest)
    print(f"  done ({os.path.getsize(dest) / 1e6:.0f} MB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None,
                    help="output pair file (default: $OPENNN_BENCH_DATA/wmt14/"
                         "wmt14_en_de_pairs.txt)")
    ap.add_argument("--max-pairs", type=int, default=200000,
                    help="keep the first N valid pairs (0 = all ~4.5M; note "
                         "every benchmark trial re-reads the corpus)")
    ap.add_argument("--max-tokens", type=int, default=128,
                    help="truncate each side to N whitespace tokens")
    ap.add_argument("--train-en", default=None,
                    help="local train.en (skips the download)")
    ap.add_argument("--train-de", default=None,
                    help="local train.de (skips the download)")
    args = ap.parse_args()

    out = args.out or default_out()
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)

    train_en = args.train_en or os.path.join(out_dir, "train.en")
    train_de = args.train_de or os.path.join(out_dir, "train.de")
    if not args.train_en:
        download(TRAIN_EN_URL, train_en)
    if not args.train_de:
        download(TRAIN_DE_URL, train_de)

    kept = skipped = truncated = 0
    with open(train_en, encoding="utf-8", errors="replace") as f_en, \
         open(train_de, encoding="utf-8", errors="replace") as f_de, \
         open(out, "w", encoding="utf-8", newline="\n") as f_out:
        for en, de in zip(f_en, f_de):
            en, de = en.strip(), de.strip()
            if not en or not de or "\t" in en or "\t" in de:
                skipped += 1
                continue
            en_tokens, de_tokens = en.split(), de.split()
            if len(en_tokens) > args.max_tokens or len(de_tokens) > args.max_tokens:
                en = " ".join(en_tokens[:args.max_tokens])
                de = " ".join(de_tokens[:args.max_tokens])
                truncated += 1
            f_out.write(f"{en}\t{de}\n")
            kept += 1
            if args.max_pairs and kept >= args.max_pairs:
                break

    metadata = {
        "dataset": "WMT14 English-German (Stanford NMT preprocessed)",
        "source_urls": [TRAIN_EN_URL, TRAIN_DE_URL],
        "reference": "Vaswani et al., Attention Is All You Need (2017)",
        "pair_format": "english<TAB>german",
        "pairs_kept": kept,
        "pairs_skipped_empty_or_tab": skipped,
        "pairs_truncated": truncated,
        "max_tokens_per_side": args.max_tokens,
        "max_pairs": args.max_pairs,
        "output": os.path.abspath(out),
    }
    metadata_path = os.path.join(out_dir, "wmt14_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=1)

    print(f"wrote {kept} pairs to {out} "
          f"(skipped {skipped}, truncated {truncated})")
    print(f"metadata: {metadata_path}")
    print("\nrun the benchmark against it with:")
    print(f"  python run_transformer_maxbatch.py --corpus {out} ...")


if __name__ == "__main__":
    main()
