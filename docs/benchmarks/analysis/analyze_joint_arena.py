"""Joint forward+backward arena feasibility analysis from a lifetime dump.

Usage:
    OPENNN_MEMORY_DEBUG=1 <trial binary> ... > trial.log
    python analyze_joint_arena.py trial.log

Parses the [MEMORY_DEBUG] ledger emitted by any training trial (the
`forward.lifetime_entry` / `backward.lifetime_entry` rows recorded by
memory_debug::record_pool_lifetimes) and computes the peak of a hypothetical
single arena holding both the forward activation pool and the backward delta
pool, using the same first-fit placement as opennn/memory_pool.cpp.

Timeline: forward of layer i at step i, backward of layer i at 2L-1-i.
Backward-pool step s maps to global step (2L-1-last_trainable) + s.

Ledger rows repeat when a process constructs the propagation objects more than
once (e.g. a warmup train followed by the timed train); sizes are normalized
by the row count, so that is handled automatically.

Measured results on the RTX 3060 Laptop reference machine (2026-08-06):
ResNet-50 batch 5,163: separate pools 5,032 MiB -> joint 4,094 MiB (-18.6%).
Transformer batch 80:  separate pools 3,823 MiB -> joint 3,557 MiB (-6.9%).
"""
import re
import sys


def parse(path):
    fwd, bwd, recompute = [], [], []
    L = last_trainable = None
    fwd_pool = delta_pool = transient = None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.search(r"forward\.recompute_entry,[\d:]+,(\d+),([\d.]+),"
                      r"first=(-?\d+),second=(-?\d+),overlaid=(\d)", line)
        if m:
            recompute.append((float(m.group(2)) / int(m.group(1)),
                              int(m.group(3)), int(m.group(4)),
                              bool(int(m.group(5)))))
            continue
        m = re.search(r"forward\.lifetime_meta,timeline,\d+,[\d.]+,layers=(\d+)", line)
        if m: L = int(m.group(1))
        m = re.search(r"backward\.lifetime_meta,timeline,\d+,[\d.]+,"
                      r"first_trainable=(-?\d+),last_trainable=(-?\d+)", line)
        if m: last_trainable = int(m.group(2))
        m = re.search(r"forward\.lifetime_entry,[\d:]+,(\d+),([\d.]+),"
                      r"first=(-?\d+),last=(-?\d+)", line)
        if m: fwd.append((float(m.group(2)) / int(m.group(1)),
                          int(m.group(3)), int(m.group(4))))
        m = re.search(r"backward\.lifetime_entry,[\d:]+,(\d+),([\d.]+),"
                      r"first=(-?\d+),last=(-?\d+)", line)
        if m: bwd.append((float(m.group(2)) / int(m.group(1)),
                          int(m.group(3)), int(m.group(4))))
        m = re.search(r"forward,ForwardPropagation::data,(\d+),([\d.]+)", line)
        if m: fwd_pool = float(m.group(2)) / int(m.group(1))
        m = re.search(r"backward,BackPropagation::delta_pool,(\d+),([\d.]+)", line)
        if m: delta_pool = float(m.group(2)) / int(m.group(1))
        m = re.search(r"forward\.transient_pool,shared_block,(\d+),([\d.]+)", line)
        if m: transient = float(m.group(2)) / int(m.group(1))
    return fwd, bwd, recompute, L, last_trainable, fwd_pool, delta_pool, transient


def first_fit(entries, order):
    """entries: list of (mib, first, last). order: index permutation.
    Returns (peak, offsets)."""
    offsets = [None] * len(entries)
    placed = []
    peak = 0.0
    for idx in order:
        size, first, last = entries[idx]
        if size <= 0: continue
        blocks = sorted(
            (offsets[j], offsets[j] + entries[j][0])
            for j in placed
            if entries[j][1] <= last and first <= entries[j][2])
        offset = 0.0
        for begin, end in blocks:
            if begin >= offset + size: break
            if end > offset: offset = end
        offsets[idx] = offset
        peak = max(peak, offset + size)
        placed.append(idx)
    return peak, offsets


def overlay_fits(entries, offsets, peak, size, instants):
    """Mirror of find_memory_pool_overlay: is there a byte range inside the
    plan where no entry live at either instant overlaps?"""
    candidates = {0.0}
    for (entry_size, _, _), offset in zip(entries, offsets):
        if offset is None or entry_size <= 0: continue
        candidates.add(offset)
        candidates.add(offset + entry_size)
    for candidate in sorted(candidates):
        if candidate + size > peak: continue
        collision = False
        for (entry_size, first, last), offset in zip(entries, offsets):
            if offset is None or entry_size <= 0: continue
            if candidate < offset + entry_size and offset < candidate + size \
               and any(first <= t <= last for t in instants):
                collision = True
                break
        if not collision:
            return True
    return False


def lower_bound(entries):
    events = {}
    for size, first, last in entries:
        events[first] = events.get(first, 0.0) + size
        events[last + 1] = events.get(last + 1, 0.0) - size
    live = peak = 0.0
    for step in sorted(events):
        live += events[step]
        peak = max(peak, live)
    return peak


def compact_order(entries):
    return sorted(range(len(entries)),
                  key=lambda i: (-entries[i][0], entries[i][1], -entries[i][2], i))


def chrono_order(entries):
    return sorted(range(len(entries)), key=lambda i: (entries[i][1], i))


def main():
    path = sys.argv[1]
    fwd, bwd, recompute, L, last_trainable, fwd_pool, delta_pool, transient = parse(path)
    if not fwd or not bwd or L is None or last_trainable is None:
        sys.exit("no lifetime dump found: run the trial with OPENNN_MEMORY_DEBUG=1")
    print(f"forward entries={len(fwd)} backward entries={len(bwd)} "
          f"recompute entries={len(recompute)} "
          f"layers={L} last_trainable={last_trainable}")
    print(f"ledger: forward data={fwd_pool} MiB (incl transient={transient}), "
          f"delta pool={delta_pool} MiB")


    fwd_peak, _ = first_fit(fwd, compact_order(fwd))
    bwd_peak_c, _ = first_fit(bwd, compact_order(bwd))
    bwd_peak_h, _ = first_fit(bwd, chrono_order(bwd))
    print(f"replica: forward pool peak={fwd_peak:.2f} MiB, "
          f"delta pool peak compact={bwd_peak_c:.2f} / chrono={bwd_peak_h:.2f} MiB")


    offset = (2 * L - 1) - last_trainable
    joint = list(fwd) + [(s, f + offset, l + offset) for (s, f, l) in bwd]
    if transient:
        joint.append((transient, 0, 2 * L - 1))

    lb = lower_bound(joint)
    jc, offsets_c = first_fit(joint, compact_order(joint))
    jh, offsets_h = first_fit(joint, chrono_order(joint))
    current = (fwd_pool or 0.0) + (delta_pool or 0.0)
    best, best_offsets = (jc, offsets_c) if jc <= jh else (jh, offsets_h)
    print(f"joint lower bound = {lb:.2f} MiB")
    print(f"joint first-fit: compact={jc:.2f} MiB, chrono={jh:.2f} MiB")





    correction = 0.0
    if recompute:
        displaced = [
            (size, first, second)
            for (size, first, second, was_overlaid) in recompute
            if was_overlaid
            and not overlay_fits(joint, best_offsets, best, size, (first, second))
        ]
        if displaced:
            largest = max(size for size, _, _ in displaced)
            correction = max(0.0, largest - (transient or 0.0))
        print(f"recompute overlays displaced by deltas = {len(displaced)}"
              f" of {sum(1 for r in recompute if r[3])}"
              f" (transient-block growth = {correction:.2f} MiB)")
    best += correction

    print(f"current separate pools = {current:.2f} MiB")
    print(f"saving = {current - best:.2f} MiB "
          f"({100.0 * (current - best) / current:.1f}% of the pooled memory)")


if __name__ == "__main__":
    main()
