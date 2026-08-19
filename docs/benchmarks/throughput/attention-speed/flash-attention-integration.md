# FlashAttention-2 in OpenNN's attention (working note)

*Working note, 2026-08-19, on the laptop (RTX 3060 Laptop 6 GB, sm_86, WSL2,
CUDA 12.9, cuDNN 9.24). Companion to
[`transformer-training-batch-sweep.md`](transformer-training-batch-sweep.md),
where the sweep left one structural remainder: the attention kernel itself.
A standalone probe put FlashAttention-2 at 1.61x our cuDNN SDPA per layer on
this GPU (~+11% per training step), which is what this integration is for.*

## What stage one covers, and why not more

FA2 is behind a build option, off by default:

```sh
cmake -DOpenNN_WITH_FLASH_ATTENTION=ON \
      -DOpenNN_FLASH_ATTENTION_HEAD_DIMS=32 \        # default 32;64;128
      [-DOpenNN_FLASH_ATTENTION_SOURCE_DIR=~/fa2]    # default: fetched
```

Each kernel takes minutes to compile and there is one per head dimension per
mask per direction, hence the head-dimension list. The kernels exist for
sm_80 through sm_90 only, so the option builds whichever of the requested
architectures FA2 covers and quietly does nothing when none is covered -
Blackwell (sm_120, the RTX 5070 Ti) has no FA2 kernel in v2.8.3, so the
product GPU is not a target of this work; the RTX 4080 (sm_89) is.

The rung (`device::AttentionRung`, `OPENNN_ATTENTION_RUNG=auto|cudnn|flash` in
the transformer benchmark) takes FA2 when all of this holds, and cuDNN's graph
otherwise:

| Condition | Why |
| --- | --- |
| head dimension 32, 64 or 128, and built | FA2 ships kernels for those |
| bf16 tensors | the only dtype FA2 has kernels for; a fp32 layer arrives through the same fp32-via-bf16 pack cuDNN's graph reads |
| heads interleaved, (B, S, H, D) | what the projections write and FA2 reads unchanged; the separated layout would need strides of its own for the merged attention output |
| dropout off | the kernels are compiled without it |
| an lse slot planned | FA2's forward writes the log-sum-exp its backward re-reads; that is the statistics state |
| **not (causal and padded)** | see below |

In the encoder-decoder transformer that is every encoder self-attention and
every cross-attention; decoder self-attention keeps cuDNN.

## The two things the probe had to establish

**Padded batches need no packing.** FA2's dense entry points assume every
sample fills its slot. `seqused_k` looks like the way out, but the launch
decides its even-shape fast path from `cu_seqlens` alone, so a sequence length
divisible by the block size takes the unmasked path and reads the padding as
data - wrong results, silently. What does work is `cu_seqlens_k` holding the
*lengths* with `is_seqlens_k_cumulative = false`: `BlockInfo` then leaves the
offsets dense and still clamps the key range, and the launch is off its fast
path, so the last block masks itself. Measured against the definition on the
host (batch 4, seqlen 64, lengths 64/61/33/7, bf16): worst absolute error
0.0093 on the forward over |out| <= 1.43, and 0.0112 on dQ/dK/dV over
|grad| <= 2.31. Nothing is gathered, scattered or copied.

**Causal and padded disagree about the corner.** With the key range shorter
than the query range, FA2 anchors the causal diagonal to the bottom-right;
OpenNN's tokens are left-aligned, which is the top-left. The probe is
unambiguous: against a top-left reference the error is 3.09 over |out| <= 3.09
(i.e. unrelated), against a bottom-right reference 0.0101. So the rung refuses
that combination outright rather than approximating it. Lifting it means the
varlen kernels, where the query range is per-sample too - stage two.

Only the key side is masked: a padded *query* row still produces an output.
Nothing downstream reads those rows (the loss that would make their delta
ignores them), and because dQ is linear in dO, a zero delta there keeps them
out of every other row's gradient as well. That is the same reason this
network could drop `set_zero_padded_queries` when it moved to fused attention.

## Guards

* `GpuComparison.FlashAttentionRungMatchesCudnnAttention` - the same
  transformer gradient under each rung, over a corpus of ragged samples so the
  padding path is what runs, and it asserts the call count moved, because two
  rungs that quietly resolve to the same kernel would agree perfectly and
  prove nothing.
* The probe itself (`~/fa2-probe`, not in the repo) covers forward and
  backward numerics, dense and padded, against the definition on the host.

## Measurements

Transformer training on the laptop, same network as the sweep (d_model 256,
8 heads, ff 1024, 2 encoder + 2 decoder layers, sequence 256, corpus on ext4,
5 timed epochs), `OPENNN_ATTENTION_RUNG=auto` against `=cudnn`, the two
alternated inside each round so a thermal drift lands on both:

| point | round | FA2 tokens/s | cuDNN tokens/s | ratio |
| --- | --- | --- | --- | --- |
| bf16, batch 128 | 1 | 876,544 | 745,738 | **1.175** |
| bf16, batch 128 | 2 | 613,721 | 541,875 | **1.133** |
| bf16, batch 256 | 1 | 822,267 | 672,489 | **1.223** |
| bf16, batch 256 | 2 | 649,885 | 570,027 | **1.140** |
| fp32, batch 128 | 1 | 362,331 | 339,893 | **1.066** |
| fp32, batch 128 | 2 | 367,347 | 338,559 | **1.085** |

The absolute numbers fall by ~30% between rounds and the ratios do not: the
laptop throttles hard over a sustained sweep (86 C by round two), which is the
whole reason the pairs are alternated. Two rounds, not three, for the same
reason - a third would have measured the cooling system.

The loss agrees to five digits at every point (bf16 128: 0.0537738 against
0.0537927), and `cuda_graph=captured` throughout, so the kernels take the
capture and the grouped 8-step training graph is unaffected. fp32 gains less
because both rungs pay the same fp32-via-bf16 cast around an attention that is
the only thing that changed.

**Passing the lengths costs nothing.** The rung passes them always, since
whether a batch is padded is not knowable on the host, and the expectation was
that this would cost something, as it takes the launch off its even-shape path.
Measured on the probe at the benchmark's shape (batch 128, sequence 256, 8
heads, head dim 32, every sequence full, so the lengths change no result):
1.991 ms forward+backward with them against 2.290 ms without, reproducible, the
difference almost all in the forward (0.409 against 0.585 ms). Faster, not
slower, and I have no mechanism to offer for it - what it settles is the
practical question, that a "this dataset never pads" fast path is not a lever.

## Next

* **The RTX 4080** (sm_89), where these kernels also run and which is not
  thermally limited; the numbers above are a laptop's.
* **Varlen**, which lifts the causal-and-padded refusal and would put decoder
  self-attention on FA2 as well.
* **Grouped-query attention**, whose operator has a graph cache of its own and
  is untouched here; FA2 takes h_k != h natively.
* A CI job that compiles with the option on. It is off by default, so nothing
  currently builds the FA2 side of `core/cuda/flash_attention.cu`.
