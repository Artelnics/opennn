# The contract

Fixed. Changing any item here invalidates every result taken under the old
version, so a change means a re-run, not a footnote.

## 1. The reference machine

Every published number comes from this machine. Others may replicate; their
numbers are never compared against these.

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti · `sm_120` · 16,303 MiB · 300 W |
| Driver | 610.43.02 |
| CPU | Intel i7-14700F |
| OS | Linux, native — not WSL |
| CUDA · cuDNN | 13.3.73 · 9.25.0 |
| PyTorch | 2.13.0+cu130, arch list includes `sm_120` |
| Python | 3.12.3 |

Native Linux is deliberate. The CNN family lazy-loads images per batch, so it
measures convolution throughput *plus* input-pipeline efficiency, and WSL's
filesystem layer would contaminate exactly that.

## 2. TensorFlow is not in the matrix

No TensorFlow build ships `sm_120` kernels — not 2.21.0, the current release,
and not the nightly, which pulls CUDA 12.9 and still targets only `sm_60`
through `sm_90`. It runs here on driver-JIT'd PTX and says so at import.

Measured cost: ~25% on a 4096³ matmul. That figure is *unattributable*, because
there is no native build to difference against, and being unattributable is
what disqualifies it — a published deficit would partly measure NVIDIA's
release schedule and would move on a rebuild.

Every cell records `engines: [opennn, pytorch]` until an NGC container changes
this. Verify any candidate build by dumping its kernels, not by reading
`build_info`, which the nightly leaves empty:

```bash
cuobjdump --list-elf .../tensorflow/libtensorflow_cc.so.2 | grep -oE "sm_[0-9]+" | sort -u
```

## 3. Each engine at its best

Not "as it comes". An engine measured below its own ceiling makes the other
look good for the wrong reason, and this is the item most likely to drift
silently.

| | |
|---|---|
| OpenNN | captured CUDA graph, device-resident split, whole batches only |
| PyTorch | `torch.compile`, `channels_last` for convolutions, TF32 enabled |

This is not hypothetical. Measured against an *eager* PyTorch, dense training
read 1.29×; against a compiled one, 1.06×. The first number was not a lie about
OpenNN — it was a lie about PyTorch.

## 4. Same work, or no comparison

Both gates run before any number is reported, and both are recorded in the
artifact.

**Shape.** Sample count, sequence length, vocabulary and parameter count must
agree across engines. Two findings that arrived this way:

- OpenNN's tokeniser emitted 158-position sequences where PyTorch's emitted
  128 — 23% more arithmetic for the same corpus, invisible in samples/s. Fixed
  by pre-tokenising with OpenNN's own rule, so both read the same stream.
- `nn.Transformer` appends a final `LayerNorm` to encoder and decoder, 2,048
  parameters OpenNN's model does not have. Fixed by building the stacks
  directly with `norm=None`.

Current parameter counts, matching exactly: dense 1,080,321 · CNN 25,557,032 ·
transformer 74,878,496 · LSTM 73,857.

**Quality.** Accuracies must agree within tolerance, per batch, across engines.
A speed win bought by computing something different is not a speed win.

## 5. Metrics

**Throughput.** Samples/s, warmup excluded, median of repeated rounds, every
round kept in the artifact so a drifting session is visible rather than
averaged into a claim. Ratios within one session are the durable output;
absolute samples/s is provenance.

**Peak memory.** `nvidia-smi` device-used, sampled at 20 ms, minus the idle
reading taken before the run. Whole-device on purpose: it counts the CUDA
context and the allocator's cached blocks, because that memory really is
unavailable to anything else.

> `torch.cuda.max_memory_allocated()` never appears in a comparison. It excludes
> both, has no OpenNN equivalent, and flatters PyTorch by construction. Keep it
> as a labelled PyTorch diagnostic if wanted.

**Energy.** Board power integrated over the engine's own timed window, between
the marks it prints. Reported only when the window held at least four samples;
otherwise the artifact says so. A short run has no energy figure, and `0.0000
Wh` would be a claim rather than an absence.

The dense family's epochs are milliseconds, so its energy cell needs `--epochs`
raised until the window clears about a second. That is a per-family setting,
not a matter of taste.

## 6. Sessions

**Lock the clock before measuring anything published:**

```bash
sudo ./gpu_clocks.sh lock 2700
```

This card idles near 400 MHz and drifts about 8% across a day. Margins under
~2% are not resolvable while it floats, however many rounds are taken. A
transformer run on unlocked clocks read 986/s and 482/s for identical work
fifteen minutes apart.

**One session id per sitting**, exported so every launch under it agrees:

```bash
export OPENNN_BENCH_SESSION=2026-08-25-baseline
```

Numbers compare only within a session. An artifact whose session id nothing
else shares is an un-anchored run, and the `adhoc-<pid>` default is deliberately
awkward for that reason.

**A dirty tree writes to `results/scratch/`.** Enforced in code, because the
previous suite stated this rule in prose and checked it nowhere — 39 of its 107
artifacts were dirty-tree results filed as reproducible ones.
