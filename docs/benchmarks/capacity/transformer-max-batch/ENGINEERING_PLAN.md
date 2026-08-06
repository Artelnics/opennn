# Transformer memory plan (simplified)

Status: revised after auditing the runtime against the original proposal. The
original draft introduced a joint training arena, external-storage binding for
`BackPropagation` and an `SDPAWorkspace` interface. The audit showed the
existing spec/planner machinery already provides the needed mechanism, so those
pieces are dropped. Estimates remain hypotheses until measured.

## Key code facts driving the simplification

1. `BackPropagation::setup_delta_pool` already performs lifetime-based
   planning: every backward spec gets a `[first_step, last_step]` range and
   `plan_memory_pool` places entries so that only overlapping lifetimes exclude
   each other. Scratch specs (index >= source count) get a lifetime confined to
   their own layer's backward step, so scratch from different layers already
   shares offsets. Both `Chronological` and `Compact` strategies do this; they
   differ only in placement order.
2. `ForwardPropagation::set` already accepts optional external storage.
3. Specs carry their own dtype (`TensorSpec{shape, dtype}`), so BF16 scratch
   can live in the FP32 delta pool without planner changes.
4. `AttentionOperator::SDPACache` keys entries by batch size and shape. Any
   batch-size change (e.g. a final partial batch) duplicates the full private
   buffer set per layer. Pool-planned specs are re-planned per batch instead.

Consequence: the SDPA memory work reduces to declaring the private buffers as
backward specs and passing the planned views into the SDPA calls. No new
arena, no new ownership types, no `BackPropagation` interface change.

## Audit snapshot (unchanged)

Measurements collected with `OPENNN_MEMORY_DEBUG`; raw evidence under
`docs/benchmarks/results/evidence/audit-universal-runtime/`.

| Workload | Forward arena | Backward delta arena | Relevant observation |
| --- | ---: | ---: | --- |
| HIGGS, CUDA tile 65,536 | 512.25 MiB | 512.00 MiB | Logical batch is gradient accumulation over a fixed tile. |
| ResNet-50, batch 5,163 | 3,983.57 MiB | 1,048.73 MiB | Forward recomputation already saves 5,031.91 logical MiB. |
| Transformer, batch 80 | 3,527.33 MiB | 215.47 MiB | Delta lower bound is 206.56 MiB: only ~8.9 MiB fragmentation. |

`AttentionOperator::SDPACache` holds ~1,119 MiB of private BF16 tensors at
batch 80 (18 attention layers), outside both arenas. Roughly half is backward
conversion scratch (dO/dQ/dK/dV), half is forward BF16 copies (Q/K/V/O).

The ~8.9 MiB fragmentation figure means delta-pool compaction changes are not
worth pursuing; the joint forward+backward arena (estimated batch 80 -> 82-85)
is deferred for the same reason: highest complexity, lowest return.

## Measured results (2026-08-06, RTX 3060 Laptop 6144 MiB, WSL, FP32)

Reference config: `chat_pairs_bounded_4096_p16_r32.txt`, 512/8/2048/6,
input_seq 71, decoder_seq 128. Archived baseline (commit 2ad3ee706):
max batch 80 under the 5632 MiB cap.

After phases 0-1 (shared dO/dQ/dK/dV scratch in the delta pool):

| Batch | Result | Peak VRAM (1 Hz sampling) |
| ---: | --- | ---: |
| 80 | OK | 5,289 MiB |
| 88 | OK | 5,741 MiB |
| 92 | OK | 5,941 MiB |
| 96 | OK | 5,943 MiB |

Ledger at batch 80: delta pool 215.5 -> 255.5 MiB (+40 MiB, the planned SDPA
scratch), forward arena unchanged at 3,527.3 MiB.

After phase 2 (Q/K/V rematerialization):

| Batch | Result | Peak VRAM (1 Hz sampling) |
| ---: | --- | ---: |
| 80 | OK | 5,149 MiB |
| 96 | OK | 5,975 MiB |
| 104 | OK | 5,985 MiB |
| 112 | OK | 5,985 MiB |

Ledger at batch 80: delta pool 285.5 MiB (+30, remat specs), forward arena
3,537.3 MiB (+10, transient pack). Batch-80 peak dropped 5,289 -> 5,149 MiB.
Micro-run wall time at batch 80 was 1.63 s -> 1.67 s (within noise for a
256-sample run; the accepted A/B is the full time/energy-to-target benchmark).

Two caveats discovered while measuring:

1. **Only decoder self-attention runs SDPA in this benchmark.** The threshold
   is 128 and the sequences are 71/128, so encoder self-attention and
   cross-attention (min side 71) take the unfused path. The private-buffer
   pool that phases 1-2 recover is therefore 6 layers, not 18: ~250 MiB
   (phase 1) + ~190 MiB (phase 2) at batch 80 - which the measured peaks and
   ledger deltas match almost exactly. The plan's original ~1,119 MiB figure
   assumed SDPA on all 18 layers; lowering `sdpa_min_sequence_length` below
   71 for this workload would widen both the savings and possibly speed, and
   should be evaluated as its own change.
2. **Raw "OK" above ~batch 92 is not a clean capacity signal on this machine.**
   Sampled memory plateaus at ~5,985 MiB while batches keep growing: WDDM is
   almost certainly paging device memory. The comparable capacity metric is
   the benchmark driver's 5,632 MiB cap; the official rerun below uses it.

## Official benchmark rerun (2026-08-06, driver protocol, target loss 7.5)

SDPA-threshold A/B with the energy driver (batch 80, fp32, 3 epochs, 2 runs):
192 (no SDPA) 33.1/34.3 s; 128 (decoder self-attention on SDPA) 33.8/35.2 s;
64 (all 18 layers on SDPA) 37.3/38.5 s. The default threshold of 128 is well
calibrated on this GPU; 64 is clearly worse (+16%), and 128 vs 192 is within
noise. Result JSONs: `gpu-transformer-max-batch-opennn-sdpa{128,192}-*` and
`gpu-transformer-max-batch-to-target-phase2-sdpa{128,192}-*`.

| Config | Max batch (cap 5,632 MiB) | Time-to-target median | Energy median |
| --- | ---: | ---: | ---: |
| Archived baseline (threshold 128, commit 2ad3ee706) | 80 | 34.0 s | 3,452.6 J |
| Phases 0-2, threshold 128 | 89 | 38.3 s | 3,081.8 J |
| Phases 0-2, threshold 192 | 91 | 38.5 s | 3,093.5 J |

Conclusions:

- Capacity rose 80 -> 89-91 (+11-14%). The archived maximum was limited by
  the private SDPA allocations phases 1-2 removed; with them gone, the SDPA
  and unfused configurations converge (the remaining 2-batch gap is the
  private O copy plus cuDNN workspaces).
- Training at the larger maximum batch cuts energy-to-target by ~10.5% but
  raises time-to-target by ~13%: fewer optimizer steps draw less total board
  power, while per-sample throughput near the VRAM cap does not improve. At
  fixed batch 80 the time-to-target is unchanged (33-34 s). Batch choice is
  therefore metric-dependent; both configurations beat PyTorch (47.1 s /
  3,914 J) and TensorFlow (105.1 s / 7,159 J) on both metrics.
- `opennn_transformer_energy` previously never configured the SDPA threshold,
  so archived energy runs used the library default of 192 (no SDPA); it now
  calls `benchmark::configure_transformer_sdpa` like the capacity trial.

## Phase 0 - correctness of attention valid lengths (implemented)

Bug: `refresh_sdpa_sequence_lengths` skipped recomputing per-sample sequence
lengths when `source_input.data` matched the cached pointer. Forward arenas are
reused across batches, so the pointer stays stable while padding changes: the
second batch silently trains with the first batch's masks.

Related asymmetry: forward falls back to the unfused path when explicit
`attention_valid_lengths` are present, but backward dispatched on `use_sdpa`
alone, so it could call SDPA backward for a forward that never ran SDPA.

Changes:

1. Always derive the sequence lengths from the current batch content; the
   pointer-identity cache is removed (the kernel is negligible next to SDPA).
2. Backward mirrors forward's dispatch condition.
3. Regression test: one `ForwardPropagation` reused across two batches with
   different padding patterns (guaranteed stable pointers) must match a fresh
   `ForwardPropagation` on the second batch.

Follow-up (separate change): feed explicit valid lengths into the SDPA graph
via the existing pinned-staging path, so networks that export valid lengths can
use SDPA instead of the unfused fallback.

Gate: convergence benchmarks are only valid after this phase.

## Phase 1 - SDPA backward gradient scratch into the delta pool (implemented)

`apply_sdpa_backward` allocated private per-layer, per-shape BF16 buffers for
dO/dQ/dK/dV (~560 MiB total at batch 80). They are live only during one
layer's backward step.

Change: `AttentionOperator::sdpa_gradient_scratch_specs` declares the four
BF16 tensors as additional `MultiHeadAttention` backward specs (empty when
SDPA is off or the compute dtype is natively BF16). The existing planner
overlaps them across layers automatically; the views are passed into
`apply_sdpa_backward`, which no longer allocates.

Expected effect: ~560 MiB of private allocations replaced by ~30 MiB inside
the delta pool (largest attention layer). Time-neutral: no operations added.

Gate: identical outputs/loss/gradients; VRAM peak drop of roughly the
predicted size; no step-time regression at fixed batch; HIGGS and ResNet-50
unchanged.

## Phase 2 - BF16 Q/K/V rematerialization (implemented)

The forward BF16 copies of Q/K/V (~420 MiB) persisted from forward to
backward. Their FP32 sources stay in the forward arena for backward anyway,
and FP32->BF16 rounding is deterministic, so re-casting during backward
reproduces bit-identical inputs - this is exact, not an approximation.

Implementation:

- Forward: the BF16 Q/K/V casts live in one transient forward spec
  (`sdpa_qkv_pack_spec`). It aliases the shared transient block because
  TransposeScratch is only written after the SDPA graph finishes, so their
  live ranges are disjoint within the layer.
- Backward: `sdpa_gradient_scratch_specs` grew from 4 to 7 entries (dO, dQ,
  dK, dV plus rematerialized Q, K, V), all delta-pool planned with
  single-step lifetimes; three extra cast kernels per attention layer.
- O remat IMPLEMENTED (2026-08-07): the private per-entry `SDPACache` O copy
  is gone. Forward writes the graph's BF16 O into a new section of the
  transient Q/K/V/O pack; backward rematerializes O by flat-casting the
  retained merged (B,S,E) output (bit-exact: the FP32 output is the exact
  image of the graph's BF16 O) into an 8th SDPA scratch slot, and the
  backward graph declares O with BSHD strides — no permuting kernel needed.
  Prerequisite planner change: one layer's transient slots are now placed
  back to back in the shared transient block (block = max per-layer sum, was
  max individual slot), so the pack's O section and TransposeScratch never
  alias during the post-graph cast. Measured at OPENNN_SDPA_MIN=128, batch
  80: arena 3,557.3 -> 3,587.3 (+30 transient), private buffers -63, peak
  VRAM 4,883 -> 4,853 MiB (-30 net); batches 96/100/104 train OK. ResNet
  unchanged (bit-identical plan). At lower SDPA thresholds (more attention
  layers fused) the freed private total grows toward the original ~140 MiB
  estimate. All SDPA VRAM is now visible to the ledger/arena planner.

No runtime flag: the change is bit-exact and the A/B comparison is done
against the archived baseline runs; revert the commit if step time or energy
regress.

`stats_buf`, dropout seeds, sequence-length buffers, descriptors and cuDNN
graphs stay in `SDPACache` (small, must survive or are per-shape compiled
state).

## Deferred work

- Joint forward+backward training arena: MEASURED (2026-08-06) with the
  lifetime dump (`memory_debug::record_pool_lifetimes`) and
  `docs/benchmarks/analysis/analyze_joint_arena.py`, including the
  recompute-overlay correction (overlays displaced by deltas fall back to
  the shared transient block):
  Transformer batch 80: 3,823 -> 3,557 MiB (saves 265 MiB, -6.9%; no
  recompute, uncorrected).
  ResNet-50 batch 5,163: 5,032 -> 4,134 MiB (saves 898 MiB, -17.8%; 2 of 52
  recompute overlays displaced, transient block +40.3 MiB).
  IMPLEMENTED AND MEASURED (2026-08-06/07). Final architecture: the forward
  planner is one lifetime-driven path for every layout (the cursor branch
  was deleted; non-compact nets use the conservative [i, 2L-1-i] bound,
  which reproduces the cursor footprint exactly);
  `BackPropagation::build_delta_entries` is a static, self-contained entry
  builder; `ForwardPropagation::set(..., Loss* joint_loss)` appends the
  delta entries to its own timeline and first-fit (recompute overlays then
  see them as occupants — the ordering hazard is structurally gone), storing
  the delta offsets in `joint_delta_plan`;
  `BackPropagation::set(..., ForwardPropagation*)` binds its delta views
  into the forward arena via the shared `bind_delta_views`. Standalone FP/BP
  degenerate to today's behavior; `Optimizer::train` passes the Loss and the
  FP. Both test suites fully green with the joint arena live.

  Measured (WSL, RTX 3060, ledger per instance):
  Transformer batch 80: arena 3,557.3 MiB vs 3,537.3 + 285.5 separate —
  saving 265 MiB, peak VRAM 5,149 -> 4,883 MiB (analyzer prediction 3,557.4:
  exact). Batches 96 and 100 train OK (official cap-based max pending the
  driver rerun; estimate ~93-98 vs 89).
  ResNet-50 batch 5,163: arena 4,114.5 MiB vs 3,983.6 + 1,048.7 separate —
  saving 917.8 MiB (better than the corrected 898 prediction; joint plan
  fragmentation is ZERO, transient block absorbed the 2 displaced recompute
  overlays at +40 MiB), peak VRAM 4,971 MiB.
  Remaining: official capacity/energy driver reruns (other machine).
- Chunked vocabulary projection + cross entropy (~415 MiB upper bound): most
  invasive; only after the above and only with time/energy gates.
- Fusion work (residual+LayerNorm, dense epilogues, delta accumulation):
  profile again after the memory changes; 5-10% time/energy is the measurement
  target.

## Benchmark matrix and acceptance (unchanged)

For every performance phase, archive:

1. maximum train batch in a fresh process;
2. fixed-batch step time, average power and joules;
3. time and joules to the benchmark target;
4. peak device-memory ledger by owner;
5. numerical comparison of outputs, loss and gradients;
6. profiler breakdown and build/driver metadata.

Run the matrix for HIGGS, ResNet-50 and Transformer. A change is accepted as a
general runtime improvement only if it does not materially regress the other
two workloads. SDPA and projection changes are evaluated separately and must
improve either time-to-target or joules-to-target, not only maximum batch.
