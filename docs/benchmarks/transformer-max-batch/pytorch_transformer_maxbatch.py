# PyTorch encoder-decoder Transformer max-batch / training-speed counterpart to
# opennn_transformer_maxbatch_trial. Same architecture (token embeddings +
# sinusoidal positional encoding, N encoder + N decoder layers, linear vocab
# projection), same optimizer (Adam), same token cross-entropy, same model shape
# (vocab / sequence lengths passed in so the model is identical to OpenNN's).
#
# Most-optimized PyTorch config for this case: TF32 matmul/cudnn (fp32 path),
# autocast(bf16) (bf16 path), fused Adam, SDPA/flash attention (nn.Transformer
# uses F.scaled_dot_product_attention internally in torch 2.x), cudnn.benchmark,
# and optional torch.compile (PT_COMPILE=1). No CUDA-graph flag is needed; this
# is the eager/compiled steady-state path.
#
#   usage: python pytorch_transformer_maxbatch.py --in-vocab V --out-vocab V \
#              --in-seq S --dec-seq S --d 512 --h 8 --ff 2048 --layers 6 \
#              --batch B --steps N [--warmup W]
#   env:   PT_BF16=1 -> autocast(bf16);  PT_COMPILE=1 -> torch.compile

import argparse, math, os, time
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--in-vocab", type=int, required=True)
ap.add_argument("--out-vocab", type=int, required=True)
ap.add_argument("--in-seq", type=int, required=True)
ap.add_argument("--dec-seq", type=int, required=True)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--ff", type=int, default=2048)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--steps", type=int, default=30)
ap.add_argument("--warmup", type=int, default=5)
args = ap.parse_args()

assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device("cuda")
torch.manual_seed(0)

# --- most-optimized fp32/bf16 knobs -------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

use_bf16 = os.environ.get("PT_BF16") is not None
use_compile = os.environ.get("PT_COMPILE") is not None
print(f"precision={'bf16' if use_bf16 else 'fp32'} in_vocab={args.in_vocab} "
      f"out_vocab={args.out_vocab} in_seq={args.in_seq} dec_seq={args.dec_seq} "
      f"d_model={args.d} heads={args.h} ff={args.ff} layers={args.layers} "
      f"batch={args.batch} steps={args.steps} compile={use_compile}")


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
        self.scale = math.sqrt(args.d)
        self.src_emb = nn.Embedding(args.in_vocab, args.d)
        self.tgt_emb = nn.Embedding(args.out_vocab, args.d)
        self.pos = PositionalEncoding(args.d, max(args.in_seq, args.dec_seq) + 1)
        self.transformer = nn.Transformer(
            d_model=args.d, nhead=args.h,
            num_encoder_layers=args.layers, num_decoder_layers=args.layers,
            dim_feedforward=args.ff, dropout=0.0, batch_first=True)
        self.out = nn.Linear(args.d, args.out_vocab)

    def forward(self, src, tgt):
        s = self.pos(self.src_emb(src) * self.scale)
        t = self.pos(self.tgt_emb(tgt) * self.scale)
        return self.out(self.transformer(s, t))


model = Seq2SeqTransformer().to(dev).train()
print(f"parameters={sum(p.numel() for p in model.parameters())}")

try:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, fused=True)
except (RuntimeError, ValueError):
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss()

step_fn = model
if use_compile:
    step_fn = torch.compile(model)

ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 \
    else torch.autocast("cuda", enabled=False)

# Small pool of distinct random batches (matched shape); cycle through it.
pool = max(8, args.warmup + args.steps)
src = torch.randint(0, args.in_vocab, (pool, args.batch, args.in_seq), device=dev)
dec = torch.randint(0, args.out_vocab, (pool, args.batch, args.dec_seq), device=dev)
tgt = torch.randint(0, args.out_vocab, (pool, args.batch, args.dec_seq), device=dev)


def step(i):
    j = i % pool
    opt.zero_grad(set_to_none=True)
    with ctx:
        logits = step_fn(src[j], dec[j])
        loss = loss_fn(logits.reshape(-1, args.out_vocab), tgt[j].reshape(-1))
    loss.backward()
    opt.step()
    return loss

for i in range(args.warmup):
    step(i)
torch.cuda.synchronize()

t0 = time.perf_counter()
last = None
for i in range(args.steps):
    last = step(args.warmup + i)
torch.cuda.synchronize()
wall_s = time.perf_counter() - t0

samples_per_s = args.steps * args.batch / wall_s
print(f"final_loss={float(last):.5f}")
print(f"wall_s={wall_s:.5f}")
print(f"samples_per_sec={samples_per_s:.2f}")
print(f"tokens_per_sec={samples_per_s * (args.in_seq + args.dec_seq):.2f}")
print("RESULT=OK")
