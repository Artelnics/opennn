# Post-process an nsys sqlite export: GPU busy vs idle, gap structure.
import sqlite3
import sys
from collections import defaultdict

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()

tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]

def demangled(name_id, cache={}):
    if name_id not in cache:
        row = cur.execute("SELECT value FROM StringIds WHERE id=?", (name_id,)).fetchone()
        cache[name_id] = row[0] if row else str(name_id)
    return cache[name_id]

kernels = cur.execute(
    "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
print(f"kernels: {len(kernels)}")

memcpy_rows = []
if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
    memcpy_rows = cur.execute(
        "SELECT start, end, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
print(f"memcpys: {len(memcpy_rows)}")

# steady-state window: middle 60% of the kernel timeline by index
n = len(kernels)
lo, hi = int(n * 0.2), int(n * 0.8)
window = kernels[lo:hi]
t0, t1 = window[0][0], window[-1][1]
wall = t1 - t0

# union busy time over kernels only, then kernels+memcpy
def union_busy(intervals):
    busy = 0
    cur_s, cur_e = None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    return busy

kern_iv = [(s, e) for s, e, _ in window]
busy_k = union_busy(kern_iv)
mem_iv = [(s, e) for s, e, _ in memcpy_rows if s >= t0 and e <= t1]
busy_all = union_busy(kern_iv + mem_iv)

print(f"\nsteady window: {wall/1e6:.1f} ms wall")
print(f"GPU busy (kernels):        {busy_k/1e6:.1f} ms  ({100.0*busy_k/wall:.1f}%)")
print(f"GPU busy (kernels+memcpy): {busy_all/1e6:.1f} ms  ({100.0*busy_all/wall:.1f}%)")
print(f"GPU idle:                  {(wall-busy_all)/1e6:.1f} ms  ({100.0*(wall-busy_all)/wall:.1f}%)")

# gap analysis: time between consecutive kernel intervals (merged), and the
# name of the kernel that STARTS after each gap
gaps = defaultdict(lambda: [0, 0])  # follower name -> [count, total ns]
prev_end = None
for s, e, name_id in window:
    if prev_end is not None and s > prev_end:
        g = gaps[demangled(name_id)]
        g[0] += 1
        g[1] += s - prev_end
    prev_end = max(prev_end or e, e)

print("\nidle attributed to the kernel that follows the gap:")
for name, (count, total) in sorted(gaps.items(), key=lambda kv: -kv[1][1])[:12]:
    print(f"  {total/1e6:9.2f} ms  {count:6d} gaps  avg {total/count/1e3:7.1f} us  -> {name[:70]}")

# per-kernel busy sums in the window
sums = defaultdict(lambda: [0, 0])
for s, e, name_id in window:
    rec = sums[demangled(name_id)]
    rec[0] += 1
    rec[1] += e - s
print("\nper-kernel busy in window:")
for name, (count, total) in sorted(sums.items(), key=lambda kv: -kv[1][1])[:15]:
    print(f"  {total/1e6:9.2f} ms  {count:6d} calls  avg {total/count/1e3:7.1f} us  {name[:70]}")
