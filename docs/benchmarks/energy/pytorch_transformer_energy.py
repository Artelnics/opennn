# PyTorch energy-to-target counterpart to opennn_transformer_energy. Trains the
# SAME model (encoder-decoder Transformer, token embeddings + sinusoidal
# positional encoding, linear vocab projection) on the SAME token ids (OpenNN's
# tokens.bin cache: per sample [input_seq | target_seq] int32, PAD=0, decoder
# input = START(2) + target shifted right) to the SAME gate (epoch-mean token
# cross-entropy over non-PAD targets <= target), in PyTorch's fastest honest
# configuration: autocast(bf16), fused Adam, SDPA attention, TF32, and the
# attention masks OpenNN also applies (PAD keys masked everywhere, causal
# decoder self-attention).
#
#   usage: python pytorch_transformer_energy.py --tokens-bin F --in-seq S \
#              --dec-seq S --in-vocab V --out-vocab V --target T [--batch B]
#              [--max-epochs N] [--lr LR] [--d 512] [--h 8] [--ff 2048] [--layers 6]
#   env:   PT_BF16=1 -> autocast(bf16);  PT_COMPILE=1 -> torch.compile

import argparse, math, os, time
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument("--tokens-bin", required=True)
ap.add_argument("--in-seq", type=int, required=True)
ap.add_argument("--dec-seq", type=int, required=True)
ap.add_argument("--in-vocab", type=int, required=True)
ap.add_argument("--out-vocab", type=int, required=True)
ap.add_argument("--target", type=float, required=True)
ap.add_argument("--batch", type=int, default=128)
ap.add_argument("--max-epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--d", type=int, default=512)
ap.add_argument("--h", type=int, default=8)
ap.add_argument("--ff", type=int, default=2048)
ap.add_argument("--layers", type=int, default=6)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

assert torch.cuda.is_available(), "CUDA GPU required"
dev = torch.device("cuda")
torch.manual_seed(args.seed)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

use_bf16 = os.environ.get("PT_BF16") is not None
use_compile = os.environ.get("PT_COMPILE") is not None

START_INDEX = 2

records = np.fromfile(args.tokens_bin, dtype=np.int32)
record_len = args.in_seq + args.dec_seq
assert records.size % record_len == 0, \
    f"tokens.bin size {records.size} not divisible by record {record_len}"
records = records.reshape(-1, record_len)
n_samples = records.shape[0]

src = torch.from_numpy(records[:, :args.in_seq].astype(np.int64)).to(dev)
tgt = torch.from_numpy(records[:, args.in_seq:].astype(np.int64)).to(dev)
dec = torch.cat([torch.full((n_samples, 1), START_INDEX, dtype=torch.int64, device=dev),
                 tgt[:, :-1]], dim=1)

print(f"precision={'bf16' if use_bf16 else 'fp32'} compile={use_compile} "
      f"samples={n_samples} in_seq={args.in_seq} dec_seq={args.dec_seq} "
      f"in_vocab={args.in_vocab} out_vocab={args.out_vocab} "
      f"target={args.target} batch={args.batch} max_epochs={args.max_epochs} "
      f"lr={args.lr} d_model={args.d} heads={args.h} ff={args.ff} layers={args.layers}",
      flush=True)


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
        self.src_emb = nn.Embedding(args.in_vocab, args.d, padding_idx=0)
        self.tgt_emb = nn.Embedding(args.out_vocab, args.d, padding_idx=0)
        self.pos = PositionalEncoding(args.d, max(args.in_seq, args.dec_seq) + 1)
        self.transformer = nn.Transformer(
            d_model=args.d, nhead=args.h,
            num_encoder_layers=args.layers, num_decoder_layers=args.layers,
            dim_feedforward=args.ff, dropout=0.0, batch_first=True)
        self.out = nn.Linear(args.d, args.out_vocab)

    def forward(self, src_b, dec_b, causal, src_pad, dec_pad):
        s = self.pos(self.src_emb(src_b) * self.scale)
        t = self.pos(self.tgt_emb(dec_b) * self.scale)
        y = self.transformer(s, t, tgt_mask=causal,
                             src_key_padding_mask=src_pad,
                             tgt_key_padding_mask=dec_pad,
                             memory_key_padding_mask=src_pad)
        return self.out(y)


model = Seq2SeqTransformer().to(dev).train()

# Match OpenNN's initialization exactly (set_parameters_glorot): every weight
# matrix uniform +/-sqrt(6/(fan_in+fan_out)), biases zero, PAD embedding row
# zero (nn.Embedding padding_idx already keeps row 0 at zero), LayerNorm 1/0.
with torch.no_grad():
    for name, p in model.named_parameters():
        if p.dim() >= 2:
            nn.init.xavier_uniform_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
    # OpenNN draws Q/K/V as three separate d x d Glorot matrices; PyTorch fuses
    # them into one (3d, d) in_proj whose joint fan would halve the limit.
    for m in model.modules():
        if isinstance(m, nn.MultiheadAttention):
            d = m.embed_dim
            for i in range(3):
                nn.init.xavier_uniform_(m.in_proj_weight[i * d:(i + 1) * d])
    model.src_emb.weight[0].zero_()
    model.tgt_emb.weight[0].zero_()

print(f"parameters={sum(p.numel() for p in model.parameters())}", flush=True)

# eps matches OpenNN's Adam (EPSILON = FLT_EPSILON)
ADAM_EPS = 1.1920929e-07
try:
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=ADAM_EPS, fused=True)
except (RuntimeError, ValueError):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=ADAM_EPS)

# ignore_index=0: OpenNN's CrossEntropyError3d averages over non-PAD targets.
loss_fn = nn.CrossEntropyLoss(ignore_index=0)

step_fn = model
if use_compile:
    step_fn = torch.compile(model)

ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 \
    else torch.autocast("cuda", enabled=False)

# bool masks throughout (same type as the key-padding masks -> SDPA fast path)
causal = torch.triu(torch.ones(args.dec_seq, args.dec_seq, dtype=torch.bool, device=dev),
                    diagonal=1)
gen = torch.Generator(device="cpu").manual_seed(args.seed)

# The energy window starts here: it includes any compile/warmup work, exactly
# as OpenNN's window includes its in-train() warmup and graph capture.
print(f"TRAIN_START_UNIX={time.time():.3f}", flush=True)
t0 = time.perf_counter()

loss_history = []
reached = False
epochs_run = 0

for epoch in range(args.max_epochs + 1):
    perm = torch.randperm(n_samples, generator=gen).to(dev)
    epoch_loss = torch.zeros((), device=dev)
    n_batches = 0
    for start in range(0, n_samples, args.batch):
        idx = perm[start:start + args.batch]
        src_b, dec_b, tgt_b = src[idx], dec[idx], tgt[idx]
        opt.zero_grad(set_to_none=True)
        with ctx:
            logits = step_fn(src_b, dec_b, causal, src_b == 0, dec_b == 0)
            loss = loss_fn(logits.reshape(-1, args.out_vocab), tgt_b.reshape(-1))
        loss.backward()
        opt.step()
        epoch_loss += loss.detach().float()
        n_batches += 1

    mean_loss = float(epoch_loss.item() / n_batches)
    loss_history.append(mean_loss)
    epochs_run = epoch
    print(f"epoch={epoch} loss={mean_loss:.6f} elapsed={time.perf_counter() - t0:.1f}s",
          flush=True)
    if mean_loss < args.target:
        reached = True
        break

torch.cuda.synchronize()
wall_s = time.perf_counter() - t0
print(f"TRAIN_END_UNIX={time.time():.3f}", flush=True)

print("loss_history=" + ",".join(f"{v:.6f}" for v in loss_history))
print(f"epochs={epochs_run}")
print(f"final_error={loss_history[-1]:.6f}")
print(f"reached_goal={1 if reached else 0}")
print(f"wall_s={wall_s:.3f}")
print(f"samples_per_sec={n_samples * (epochs_run + 1) / wall_s:.2f}")
print("RESULT=OK")
