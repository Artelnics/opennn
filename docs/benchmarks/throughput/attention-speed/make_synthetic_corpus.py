#!/usr/bin/env python3
"""Generate a synthetic tab-separated corpus for the Transformer TRAINING benchmark.

Each line is:  <input tokens, space-separated> \t <target tokens, space-separated>

OpenNN's LanguageDataset.read_txt sets
    maximum_input_sequence_length  = max(#input tokens)  + 2   (START/END)
    maximum_target_sequence_length = max(#target tokens) + 1
so to hit a desired model input length L we emit (L-2) input tokens per line, and
(L-1) target tokens, on every line (fixed shape -> no padding variance, a clean
controlled benchmark that the PyTorch side reproduces token-for-token).

Vocabulary is the set of distinct words actually emitted, so we draw tokens from a
fixed pool word_0..word_{V-1} and guarantee every pool word appears at least once.

Determinism: a simple LCG (no numpy/torch dep) so OpenNN-side corpus and the
PyTorch-side in-memory tensors can be regenerated identically from the same seed.

Usage:
    make_synthetic_corpus.py OUT.txt VOCAB SEQ_LEN N_SAMPLES [SEED]
"""
import sys


def lcg(seed):
    state = seed & 0xFFFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state


def main():
    if len(sys.argv) < 5:
        sys.exit(__doc__)
    out_path = sys.argv[1]
    vocab = int(sys.argv[2])          # distinct content words in the pool
    seq_len = int(sys.argv[3])        # desired model input sequence length L
    n_samples = int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 1234

    # tokens per line so OpenNN derives exactly seq_len for input and target.
    in_tokens = seq_len - 2           # +2 for START/END -> seq_len
    tgt_tokens = seq_len - 1          # +1 -> seq_len
    if in_tokens < 1 or tgt_tokens < 1:
        sys.exit("seq_len too small (need >= 3)")

    pool = [f"w{i}" for i in range(vocab)]
    rng = lcg(seed)

    # One flat token stream: first `vocab` tokens are the pool in order (so every
    # word appears -> stable vocabulary across regenerations), the rest random.
    per_line = in_tokens + tgt_tokens
    total = n_samples * per_line
    tokens = pool + [pool[next(rng) % vocab] for _ in range(total - vocab)]

    lines = []
    for s in range(n_samples):
        row = tokens[s * per_line:(s + 1) * per_line]
        lines.append(" ".join(row[:in_tokens]) + "\t" + " ".join(row[in_tokens:]))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"wrote {n_samples} samples to {out_path}: "
          f"vocab={vocab} seq_len={seq_len} "
          f"(in_tokens={in_tokens} tgt_tokens={tgt_tokens})")


if __name__ == "__main__":
    main()
