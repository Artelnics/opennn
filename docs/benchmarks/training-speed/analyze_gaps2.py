# Second pass: phase-aware gap analysis of the nsys sqlite export.
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()

def name_of(name_id, cache={}):
    if name_id not in cache:
        row = cur.execute("SELECT value FROM StringIds WHERE id=?", (name_id,)).fetchone()
        cache[name_id] = row[0] if row else str(name_id)
    return cache[name_id]

kernels = cur.execute(
    "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
memcpys = cur.execute(
    "SELECT start, end, copyKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()

t_origin = kernels[0][0]
ms = lambda t: (t - t_origin) / 1e6

# top-15 biggest kernel-to-kernel gaps over the WHOLE run, with context
print("largest gaps (whole run), with surrounding kernels:")
prev = None
all_gaps = []
for i, (s, e, nid) in enumerate(kernels):
    if prev is not None and s > prev[1]:
        all_gaps.append((s - prev[1], prev, (s, e, nid)))
    if prev is None or e > prev[1]:
        prev = (s, e, nid)
for gap, before, after in sorted(all_gaps, key=lambda g: -g[0])[:15]:
    print(f"  {gap/1e6:10.2f} ms gap at t={ms(before[1]):9.1f} ms:"
          f" after {name_of(before[2])[:40]} -> before {name_of(after[2])[:40]}")

# steps = adam_update occurrences; report step period and idle per step in the
# longest dense region (gaps > 5 ms treated as phase boundaries)
adam_starts = [s for s, e, nid in kernels if "adam_update_capturable" in name_of(nid)]
print(f"\nadam (step) count whole run: {len(adam_starts)}")

# dense segments
segments = []
seg_start_idx = 0
prev_end = kernels[0][1]
for i in range(1, len(kernels)):
    s, e, nid = kernels[i]
    if s - prev_end > 5e6:
        segments.append((seg_start_idx, i))
        seg_start_idx = i
    prev_end = max(prev_end, e)
segments.append((seg_start_idx, len(kernels)))

print("\ndense segments (>5ms boundaries):")
for a, b in segments:
    seg = kernels[a:b]
    if len(seg) < 10:
        continue
    t0, t1 = seg[0][0], seg[-1][1]
    steps = sum(1 for s, e, nid in seg if "adam_update_capturable" in name_of(nid))
    busy = 0
    cs, ce = None, None
    for s, e, nid in seg:
        if ce is None or s > ce:
            if ce is not None:
                busy += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    wall = t1 - t0
    label = f"steps={steps}"
    if steps:
        label += f" wall/step={wall/steps/1e3:.0f}us busy/step={busy/steps/1e3:.0f}us idle/step={(wall-busy)/steps/1e3:.0f}us"
    print(f"  t={ms(t0):9.1f}..{ms(t1):9.1f} ms  wall={wall/1e6:8.1f} ms  busy={100.0*busy/wall:5.1f}%  {label}")

# memcpy structure: big H2D events
print("\nmemcpys > 1 MB:")
big = [(s, e, k, b) for s, e, k, b in memcpys if b > 1e6]
tot = defaultdict(lambda: [0, 0, 0])
for s, e, k, b in big:
    rec = tot[k]
    rec[0] += 1
    rec[1] += b
    rec[2] += e - s
for k, (cnt, byts, dur) in sorted(tot.items()):
    print(f"  kind={k}  n={cnt}  {byts/1e6:.0f} MB  {dur/1e6:.1f} ms")
print("first 8 big memcpys (t, ms, MB):")
for s, e, k, b in big[:8]:
    print(f"  t={ms(s):9.1f} ms  dur={(e-s)/1e6:7.2f} ms  {b/1e6:7.1f} MB  kind={k}")
