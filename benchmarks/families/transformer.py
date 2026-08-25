#!/usr/bin/env python3
"""The transformer family in PyTorch, defined once, driven four ways.

PLAN.md; the counterpart of transformer.cpp. Encoder-decoder, d_model 512, 8
heads, feed-forward 2048, 6 layers -- the "Attention Is All You Need" base
model -- with scaled token embeddings, sinusoidal positions and a final
projection to the vocabulary, which is what transformer.cpp builds.

  transformer.py train    <corpus> <corpus> [epochs] [batch,...] [d_model] [layers] [dev] [prec]
  transformer.py infer    <corpus>          [reps]   [batch,...] [d_model] [layers] [dev] [prec]
  transformer.py capacity <corpus>          [batch]              [d_model] [layers] [dev] [prec]

**On tokenisation.** The two engines do not share a tokeniser: OpenNN's
LanguageDataset derives its own vocabulary and sequence length, and
reimplementing that here would be a second thing to keep in step. What must
match is the tensor *shape*, because that is what the arithmetic depends on --
sequence length, vocabulary size and batch. Both are printed by both engines
and recorded in the artifact, so a divergence shows up as data rather than
hiding inside a throughput number. The token identities themselves are
irrelevant to a speed measurement; their count is not.
"""

from __future__ import annotations

import collections
import contextlib
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SEED = 42
VOCAB_CAP = 20_000          # LanguageDataset's own cap, so both sides agree

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-math.log(10000.0) / d_model))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]

class Seq2Seq(nn.Module):
    def __init__(self, vocab: int, sequence: int, d_model: int, layers: int):
        super().__init__()
        heads = max(1, d_model // 64)
        self.scale = math.sqrt(d_model)
        self.source_embedding = nn.Embedding(vocab, d_model)
        self.target_embedding = nn.Embedding(vocab, d_model)
        self.positions = PositionalEncoding(d_model, sequence + 1)
        # Encoder and decoder built directly rather than through
        # nn.Transformer, which appends a final LayerNorm to each stack.
        # Those two norms are 2 * (512 weight + 512 bias) = 2,048 parameters
        # OpenNN's Transformer does not have, and two extra normalisations per
        # forward pass over every position. norm=None drops them, so the two
        # engines run the same graph rather than merely a similar one.
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model,
                                                   batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model,
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, norm=None)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers, norm=None)
        self.output = nn.Linear(d_model, vocab)

    def forward(self, source, target):
        s = self.positions(self.source_embedding(source) * self.scale)
        t = self.positions(self.target_embedding(target) * self.scale)
        return self.output(self.decoder(t, self.encoder(s)))

def build(vocab: int, sequence: int, opts: dict) -> nn.Module:
    """The transformer family. Nothing else here constructs the network."""
    torch.manual_seed(SEED)
    return Seq2Seq(vocab, sequence, opts["d_model"], opts["layers"]).to(opts["device"])

def load_corpus(path: str, sequence: int | None = None) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """`source <TAB> target` to padded id tensors, and the shapes derived.

    Whitespace tokens, ranked by frequency and capped at VOCAB_CAP, with 0
    reserved for padding. Sequence length is the longest pair unless one is
    given, so the padded shape is a property of the corpus rather than a knob.
    """
    sources, targets = [], []
    counter: collections.Counter = collections.Counter()

    with open(path, encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            source, target = parts[0].split(), parts[1].split()
            counter.update(source)
            counter.update(target)
            sources.append(source)
            targets.append(target)

    # OpenNN's WordLevelTokenizer reserves [PAD]=0, [UNK]=1, [START]=2,
    # [END]=3, and brackets each sequence with START and END. Mirroring that
    # is not cosmetic: those two positions are two more rows of attention per
    # sequence, and without them the padded length differed by exactly 2.
    PAD, UNK, START, END = 0, 1, 2, 3
    vocabulary = {token: index + 4
                  for index, (token, _) in enumerate(counter.most_common(VOCAB_CAP - 4))}

    longest = max(max(len(s) for s in sources), max(len(t) for t in targets))
    length = sequence or longest + 2

    def encode(rows):
        tensor = torch.full((len(rows), length), PAD, dtype=torch.long)
        for row_index, row in enumerate(rows):
            ids = [START] + [vocabulary.get(t, UNK) for t in row[: length - 2]] + [END]
            tensor[row_index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        return tensor

    return encode(sources), encode(targets), VOCAB_CAP, length

def parse_opts(argv: list[str], first: int) -> dict:
    def at(index: int, default: str) -> str:
        return argv[index] if len(argv) > index else default

    device = at(first + 2, "cuda")
    precision = at(first + 3, "fp32")

    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    allow_tf32 = precision != "strict"
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    return {"d_model": int(at(first, "512")), "layers": int(at(first + 1, "6")),
            "device": device, "precision": precision,
            "autocast": precision == "bf16"}

def autocast_ctx(opts: dict):
    if not opts["autocast"]:
        return contextlib.nullcontext()
    return torch.autocast(device_type=opts["device"], dtype=torch.bfloat16)

def compiled(fn, opts: dict):
    mode = os.environ.get("PT_COMPILE_MODE", "default")
    if mode == "eager" or opts["device"] != "cuda":
        return fn, "eager"
    return torch.compile(fn, mode=None if mode == "default" else mode), f"compile:{mode}"

def batches_of(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part]

def sync(opts: dict) -> None:
    if opts["device"] == "cuda":
        torch.cuda.synchronize()

def report(batch: int, seconds: list[float], sequences: int, tokens_per_sequence: int) -> None:
    seconds = sorted(seconds)
    median = seconds[len(seconds) // 2]
    print(f"batch_{batch}_samples_per_sec={int(sequences / median)}"
          f" median_epoch_s={median:.6g}")
    print(f"batch_{batch}_tokens_per_sec={int(sequences * tokens_per_sequence / median)}")
    print(f"batch_{batch}_epoch_times={','.join(f'{t:.6g}' for t in seconds)}", flush=True)

def train_like(argv: list[str], mode: str) -> int:
    epochs = int(argv[4]) if len(argv) > 4 else 1
    batches = batches_of(argv[5] if len(argv) > 5 else "32")
    opts = parse_opts(argv, 6)

    print(f"engine=pytorch\nmode={mode}\ndevice={opts['device']}")
    source, target, vocab, sequence = load_corpus(argv[2])
    print(f"samples={source.shape[0]} sequence={sequence} "
          f"input_vocab={vocab} target_vocab={vocab} "
          f"d_model={opts['d_model']} layers={opts['layers']}")

    source = source.to(opts["device"])
    target = target.to(opts["device"])
    warmup = 1 if mode == "train" else 0

    for batch in batches:
        model = build(vocab, sequence, opts)
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
        ctx = autocast_ctx(opts)

        def step(s, t):
            optimizer.zero_grad(set_to_none=True)
            with ctx:
                logits = model(s, t)
                loss = loss_fn(logits.reshape(-1, vocab), t.reshape(-1))
            loss.backward()
            optimizer.step()

        step_fn, how = compiled(step, opts)
        starts = range(0, source.shape[0] - batch + 1, batch)

        def run_epoch():
            model.train()
            for start in starts:
                step_fn(source[start:start + batch], target[start:start + batch])
            sync(opts)

        for _ in range(warmup):
            run_epoch()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(epochs):
            mark = time.perf_counter()
            run_epoch()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        print(f"batch_{batch}_mode={how}")
        report(batch, times, len(list(starts)) * batch, sequence)

    print("RESULT=OK")
    return 0

def infer(argv: list[str]) -> int:
    reps = int(argv[3]) if len(argv) > 3 else 1
    batches = batches_of(argv[4] if len(argv) > 4 else "32")
    opts = parse_opts(argv, 5)

    print(f"engine=pytorch\nmode=infer\ndevice={opts['device']}")
    source, target, vocab, sequence = load_corpus(argv[2])
    print(f"samples={source.shape[0]} sequence={sequence} "
          f"input_vocab={vocab} target_vocab={vocab}")

    source = source.to(opts["device"])
    target = target.to(opts["device"])

    for batch in batches:
        model = build(vocab, sequence, opts).eval()
        if batch == batches[0]:
            print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        forward, _ = compiled(model, opts)
        processed = (source.shape[0] // batch) * batch

        s, t = source[:batch], target[:batch]

        def run_pass():
            with torch.no_grad(), autocast_ctx(opts):
                for _ in range(processed // batch):
                    forward(s, t)
            sync(opts)

        run_pass()

        print(f"TIMED_START_UNIX={time.time():.3f}", flush=True)
        times = []
        for _ in range(reps):
            mark = time.perf_counter()
            run_pass()
            times.append(time.perf_counter() - mark)
        print(f"TIMED_END_UNIX={time.time():.3f}", flush=True)

        report(batch, times, processed, sequence)

    print("RESULT=OK")
    return 0

def capacity(argv: list[str]) -> int:
    batch = int(argv[3]) if len(argv) > 3 else 32
    opts = parse_opts(argv, 4)

    print(f"engine=pytorch\nmode=capacity\ndevice={opts['device']}\nbatch={batch}")

    try:
        source, target, vocab, sequence = load_corpus(argv[2])
        s = source[:batch].to(opts["device"])
        t = target[:batch].to(opts["device"])

        model = build(vocab, sequence, opts)
        print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)

        with autocast_ctx(opts):
            loss = loss_fn(model(s, t).reshape(-1, vocab), t.reshape(-1))
        loss.backward()
        optimizer.step()
        sync(opts)
    except torch.cuda.OutOfMemoryError as error:
        print(f"fits=0\nreason={error}\nRESULT=OOM", flush=True)
        return 1
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print(f"fits=0\nreason={error}\nRESULT=OOM", flush=True)
        return 1

    print("fits=1\nRESULT=OK", flush=True)
    return 0

def main() -> int:
    argv = sys.argv
    mode = argv[1] if len(argv) > 1 else ""

    if mode in ("train", "quality") and len(argv) > 3:
        return train_like(argv, mode)
    if mode == "infer" and len(argv) > 2:
        return infer(argv)
    if mode == "capacity" and len(argv) > 2:
        return capacity(argv)

    print(__doc__, file=sys.stderr)
    return 2

if __name__ == "__main__":
    raise SystemExit(main())
