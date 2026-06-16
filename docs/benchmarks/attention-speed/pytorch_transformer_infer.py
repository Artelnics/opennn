# PyTorch Transformer inference throughput counterpart to opennn_transformer_infer.
#
# Mirrors the OpenNN encoder-decoder Transformer: token embeddings (scaled) +
# sinusoidal positional encoding, N encoder layers and N decoder layers
# (multi-head attention + position-wise feed-forward, post-LayerNorm), and a
# final linear projection to the vocabulary. Times the steady-state forward pass
# on the GPU after a warmup; reports tokens/sec. fp32 to match OpenNN's config.
#
#   usage: python pytorch_transformer_infer.py [seq] [d_model] [heads] [ff] [layers] [vocab] [batch] [iters]

import sys
import math
import time

import torch
import torch.nn as nn

seq     = int(sys.argv[1]) if len(sys.argv) > 1 else 64
d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 512
heads   = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ff      = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
layers  = int(sys.argv[5]) if len(sys.argv) > 5 else 6
vocab   = int(sys.argv[6]) if len(sys.argv) > 6 else 10000
batch   = int(sys.argv[7]) if len(sys.argv) > 7 else 8
iters   = int(sys.argv[8]) if len(sys.argv) > 8 else 50

assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device("cuda")
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True


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
        self.pos = PositionalEncoding(d_model, seq + 1)
        self.transformer = nn.Transformer(
            d_model=d_model, nhead=heads,
            num_encoder_layers=layers, num_decoder_layers=layers,
            dim_feedforward=ff, dropout=0.0, batch_first=True)
        self.out = nn.Linear(d_model, vocab)

    def forward(self, src, tgt):
        s = self.pos(self.src_emb(src) * self.scale)
        t = self.pos(self.tgt_emb(tgt) * self.scale)
        return self.out(self.transformer(s, t))


model = Seq2SeqTransformer().to(dev).eval()
params = sum(p.numel() for p in model.parameters())
print(f"config seq={seq} d_model={d_model} heads={heads} ff={ff} layers={layers} vocab={vocab} batch={batch}")
print(f"parameters={params}")

src = torch.randint(0, vocab, (batch, seq), device=dev)
tgt = torch.randint(0, vocab, (batch, seq), device=dev)

# OPENNN_BF16-equivalent: autocast to bf16 (flash-attention, like OpenNN's bf16 SDPA path).
import os
use_bf16 = os.environ.get("PT_BF16") is not None
print(f"precision={'bf16' if use_bf16 else 'fp32'}")
ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cuda", enabled=False)

with torch.no_grad(), ctx:
    model(src, tgt)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(src, tgt)
    torch.cuda.synchronize()
    per = (time.perf_counter() - t0) / iters

tokens = batch * seq
print(f"step_s={per:.6f}")
print(f"tokens_per_sec={int(tokens / per)}")
print(f"sequences_per_sec={int(batch / per)}")
print("RESULT=OK")
