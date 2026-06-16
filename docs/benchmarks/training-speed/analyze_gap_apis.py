# What CUDA API calls the host makes during the big inter-train() stall.
import sqlite3
import sys

db = sqlite3.connect(sys.argv[1])
cur = db.cursor()

def name_of(name_id, cache={}):
    if name_id not in cache:
        row = cur.execute("SELECT value FROM StringIds WHERE id=?", (name_id,)).fetchone()
        cache[name_id] = row[0] if row else str(name_id)
    return cache[name_id]

k0 = cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
ms = lambda t: (t - k0) / 1e6

# stall window: t = 816 .. 4085 ms relative to first kernel
a = k0 + int(816e6)
b = k0 + int(4090e6)

rows = cur.execute(
    "SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME "
    "WHERE end > ? AND start < ? ORDER BY (end-start) DESC LIMIT 25", (a, b)).fetchall()
print("longest CUDA API calls inside the stall window (816..4090 ms):")
total = 0
for s, e, nid in rows:
    print(f"  t={ms(s):9.1f}..{ms(e):9.1f} ms  dur={(e-s)/1e6:9.2f} ms  {name_of(nid)[:60]}")

# coverage: how much of the window is inside ANY runtime API call
rows = cur.execute(
    "SELECT start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME "
    "WHERE end > ? AND start < ? ORDER BY start", (a, b)).fetchall()
busy = 0
cs = ce = None
for s, e in rows:
    s, e = max(s, a), min(e, b)
    if ce is None or s > ce:
        if ce is not None:
            busy += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
if ce is not None:
    busy += ce - cs
print(f"\nwindow {((b-a)/1e6):.0f} ms: inside CUDA API calls {busy/1e6:.0f} ms "
      f"({100.0*busy/(b-a):.1f}%), outside (pure host work) {(b-a-busy)/1e6:.0f} ms")
