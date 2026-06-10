# Establish the raw GEMM floor on this GPU for the shapes OpenNN runs per step.
# A Dense(1000->1000) forward is one (B x K) x (K x N) matmul with B=1000,
# K=N=1000; backward adds two more of the same size (dX and dW). We time these
# in isolation at bf16 (tensor cores) and tf32 to see how close OpenNN's
# measured fwd/bwd are to the hardware floor.
import time
import torch

assert torch.cuda.is_available()
dev = "cuda"
B = K = N = 1000
iters = 2000

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def bench(dtype, label):
    a = torch.randn(B, K, device=dev, dtype=dtype)
    w = torch.randn(K, N, device=dev, dtype=dtype)
    g = torch.randn(B, N, device=dev, dtype=dtype)
    # warmup
    for _ in range(50):
        y = a @ w            # fwd
        dx = g @ w.t()       # bwd dX
        dw = a.t() @ g       # bwd dW
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        y = a @ w
        dx = g @ w.t()
        dw = a.t() @ g
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    # 3 GEMMs per "step", 1000 steps per 500k-sample epoch
    epoch_ms = dt * 1000 * 500  # 500 batches/epoch
    print(f"{label:10}: {dt*1e6:8.1f} us / 3-GEMM step  ->  {epoch_ms:7.1f} ms/epoch (500 steps)  -> {500000/(dt*500):,.0f} samples/s ceiling")


bench(torch.bfloat16, "bf16")
bench(torch.float32, "tf32/fp32")
print(f"\nGPU: {torch.cuda.get_device_name(0)}")
