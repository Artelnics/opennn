# PyTorch Transformer TRAINING throughput counterpart to opennn_transformer_train.
#
# Mirrors the OpenNN encoder-decoder Transformer training loop: same architecture
# (token embeddings + sinusoidal positional encoding, N encoder + N decoder layers,
# linear projection to vocab), same optimizer (Adam), same token cross-entropy loss
# over the vocabulary, same synthetic corpus shape (vocab / sequence length / sample
# count are read from the SAME corpus file the OpenNN side trains on, so the FLOPs
# match token-for-token). Times a fixed number of epochs after a warmup epoch and
# reports samples/sec and tokens/sec.
#
#   usage: python pytorch_transformer_train.py CORPUS.txt [d_model] [heads] [ff] [layers] [batch] [epochs]
#   env:   PT_BF16=1 -> train under autocast(bf16) (matches OpenNN OPENNN_BF16)

import sys
import os
import math
import time

import torch
import torch.nn as nn

corpus  = sys.argv[1] if len(sys.argv) > 1 else "synthetic_corpus.txt"
d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 256
heads   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ff      = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
layers  = int(sys.argv[5]) if len(sys.argv) > 5 else 2
batch   = int(sys.argv[6]) if len(sys.argv) > 6 else 32
epochs  = int(sys.argv[7]) if len(sys.argv) > 7 else 30

assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device("cuda")
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True


# --- read the SAME corpus OpenNN trains on, to match shapes exactly -------------
# OpenNN's TextDataset: input_seq = max(#input tokens) + 2 (START/END),
# decoder_seq = max(#target tokens) + 1, vocab = distinct tokens + 4 reserved.
def read_corpus(path):
    in_lens, tgt_lens, vocab = [], [], set()
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        a, _, b = line.partition("\t")
        ina, tgta = a.split(), b.split()
        in_lens.append(len(ina))
        tgt_lens.append(len(tgta))
        vocab.update(ina); vocab.update(tgta)
    input_seq = max(in_lens) + 2
    decoder_seq = max(tgt_lens) + 1
    vocab_size = len(vocab) + 4   # [PAD] [UNK] [START] [END]
    return len(in_lens), input_seq, decoder_seq, vocab_size


samples, input_seq, decoder_seq, vocab = read_corpus(corpus)
print(f"precision={'bf16' if os.environ.get('PT_BF16') else 'fp32'} samples={samples} "
      f"input_seq={input_seq} decoder_seq={decoder_seq} input_vocab={vocab} output_vocab={vocab} "
      f"d_model={d_model} heads={heads} ff={ff} layers={layers} batch={batch} epochs={epochs}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.src_emb = nn.Embedding(vocab, d_model)
        self.tgt_emb = nn.Embedding(vocab, d_model)
        self.pos = PositionalEncoding(d_model, max(input_seq, decoder_seq) + 1)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=heads,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=ff, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, src, tgt):
        s = self.pos(self.src_emb(src) * self.scale)
        t = self.pos(self.tgt_emb(tgt) * self.scale)
        return self.out(self.transformer(s, t))


model = Seq2SeqTransformer().to(dev).train()
params = sum(p.numel() for p in model.parameters())
print(f"parameters={params}")

# Synthetic data matching OpenNN's corpus shape (same #samples, seq lengths, vocab).
# Throughput is shape/FLOP bound, so matched shapes make the comparison fair; the
# actual token values differ but the forward+backward+optimizer cost is identical.
src = torch.randint(0, vocab, (samples, input_seq), device=dev)
dec = torch.randint(0, vocab, (samples, decoder_seq), device=dev)
tgt = torch.randint(0, vocab, (samples, decoder_seq), device=dev)

lr = float(os.environ.get("OPENNN_LR", "0.0001"))
print(f"learning_rate={lr}")
opt = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss()

use_bf16 = os.environ.get("PT_BF16") is not None
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

n_batches = samples // batch


def run_epoch():
    for b in range(n_batches):
        i = b * batch
        s = src[i:i + batch]; d = dec[i:i + batch]; y = tgt[i:i + batch]
        opt.zero_grad(set_to_none=True)
        with ctx:
            logits = model(s, d)                       # (batch, decoder_seq, vocab)
            loss = loss_fn(logits.reshape(-1, vocab), y.reshape(-1))
        loss.backward()
        opt.step()
    return float(loss.detach())


# warmup epoch (matches OpenNN train()'s internal CUDA warmup), excluded from timing
run_epoch()
torch.cuda.synchronize()

t0 = time.perf_counter()
last = 0.0
timed_passes = max(1, epochs)                          # OpenNN's train() runs max(1, maximum_epochs) passes
for _ in range(timed_passes):
    last = run_epoch()
torch.cuda.synchronize()
wall_s = time.perf_counter() - t0

total_samples = n_batches * batch * timed_passes
samples_per_s = total_samples / wall_s
tokens_per_s = samples_per_s * (input_seq + decoder_seq)

print(f"final_loss={last:.5f}")
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print(f"tokens_per_sec={tokens_per_s:.2f}")
print("RESULT=OK")
