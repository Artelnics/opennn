#!/usr/bin/env python3
"""Attention-kernel probe for the transformer training benchmark's shape.

Times PyTorch's scaled_dot_product_attention backends (flash-attention 2,
memory-efficient, cuDNN) forward and backward at (B, 8 heads, 256, 32) in bf16,
non-causal and causal, plus the step's GEMM shapes, so a GPU's ceiling for the
fused attention and the linear layers is known before reading the sweep:
OpenNN's fused attention is cuDNN's SDPA graph, PyTorch's default here is FA2.
See transformer-training-batch-sweep.md.

  usage: sdpa_probe.py [BATCH]        (default 128)
"""
import torch, time, sys
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

B, H, S, D = int(sys.argv[1]) if len(sys.argv) > 1 else 128, 8, 256, 32
dt = torch.bfloat16
q = torch.randn(B, H, S, D, device="cuda", dtype=dt, requires_grad=True)
k = torch.randn(B, H, S, D, device="cuda", dtype=dt, requires_grad=True)
v = torch.randn(B, H, S, D, device="cuda", dtype=dt, requires_grad=True)
do = torch.randn(B, H, S, D, device="cuda", dtype=dt)

def bench(backend, causal, iters=50):
    with sdpa_kernel(backend):
        for _ in range(5):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            o.backward(do)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        fwd = (time.perf_counter() - t0) / iters
        t0 = time.perf_counter()
        for _ in range(iters):
            o = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            o.backward(do)
        torch.cuda.synchronize()
        both = (time.perf_counter() - t0) / iters
    flop_fwd = 4 * B * H * S * S * D * (0.5 if causal else 1)
    print(f"{backend.name:16s} causal={causal}: fwd {fwd*1e3:6.3f} ms ({flop_fwd/fwd/1e12:5.1f} TFLOPS)  "
          f"fwd+bwd {both*1e3:6.3f} ms  bwd {(both-fwd)*1e3:6.3f} ms ({2.5*flop_fwd/(both-fwd)/1e12:5.1f} TFLOPS)")

for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION):
    for causal in (False, True):
        try:
            bench(backend, causal)
        except Exception as e:
            print(backend.name, causal, "unavailable:", str(e)[:80])

# GEMM reference: (B*S, 256) x (256, 1024) bf16
M, K, N = B * S, 256, 1024
a = torch.randn(M, K, device="cuda", dtype=dt); w = torch.randn(K, N, device="cuda", dtype=dt)
for _ in range(5): c = a @ w
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): c = a @ w
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50
print(f"GEMM {M}x{K}x{N}: {t*1e3:.3f} ms {2*M*K*N/t/1e12:.1f} TFLOPS")
# wgrad-like: (K, M) x (M, N) -> (256, 1024) with K=M rows reduction
g = torch.randn(M, N, device="cuda", dtype=dt)
for _ in range(5): dw = a.t() @ g
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): dw = a.t() @ g
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50
print(f"WGRAD {K}x{M}x{N}: {t*1e3:.3f} ms {2*M*K*N/t/1e12:.1f} TFLOPS")
dw32 = torch.empty(K, N, device="cuda", dtype=torch.float32)
for _ in range(5): torch.matmul(a.t(), g, out=dw)
M2, K2, N2 = B * S, 1024, 256
a2 = torch.randn(M2, K2, device="cuda", dtype=dt); w2 = torch.randn(K2, N2, device="cuda", dtype=dt)
for _ in range(5): c = a2 @ w2
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): c = a2 @ w2
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50
print(f"GEMM {M2}x{K2}x{N2}: {t*1e3:.3f} ms {2*M2*K2*N2/t/1e12:.1f} TFLOPS")
M3, K3, N3 = B * S, 256, 256
a3 = torch.randn(M3, K3, device="cuda", dtype=dt); w3 = torch.randn(K3, N3, device="cuda", dtype=dt)
for _ in range(5): c = a3 @ w3
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): c = a3 @ w3
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50
print(f"GEMM {M3}x{K3}x{N3}: {t*1e3:.3f} ms {2*M3*K3*N3/t/1e12:.1f} TFLOPS")
g3 = torch.randn(M3, N3, device="cuda", dtype=dt)
for _ in range(5): dw = a3.t() @ g3
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(50): dw = a3.t() @ g3
torch.cuda.synchronize(); t = (time.perf_counter() - t0) / 50
print(f"WGRAD {K3}x{M3}x{N3}: {t*1e3:.3f} ms {2*M3*K3*N3/t/1e12:.1f} TFLOPS")
