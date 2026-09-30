# Electrocardiogram heartbeat data

Source: [BIDMC Congestive Heart Failure Database, version 1.0.0](https://physionet.org/content/chfdb/1.0.0/),
published by PhysioNet, record `chf07`.
Dataset DOI: [10.13026/C29G60](https://doi.org/10.13026/C29G60).

Original publication: Baim, D. S., Colucci, W. S., Monrad, E. S., et al. (1986),
*Survival of patients with severe congestive heart failure treated with oral
milrinone*, Journal of the American College of Cardiology, 7(3), 661-670.
PhysioNet citation: Pollard, T., Moody, B. E., Lehman, L., et al. (2026),
*PhysioNet as a global platform for biomedical research*, Nature Health,
DOI [10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

Data licence: [Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/),
as specified by PhysioNet for this database. The bundled derivative is
distributed under those terms. Preserve this notice, attribution and licence
link with the data when redistributing it. Publicly shared outputs must also
acknowledge that they use information from the BIDMC database under ODC-By.
Licence checked on 2026-09-30.

## Source and OpenNN adaptation

OpenNN derives this subset directly from the versioned original files
[`chf07.dat`](https://physionet.org/files/chfdb/1.0.0/chf07.dat),
[`chf07.ecg`](https://physionet.org/files/chfdb/1.0.0/chf07.ecg) and
[`chf07.hea`](https://physionet.org/files/chfdb/1.0.0/chf07.hea).
Their hashes match PhysioNet's
[`SHA256SUMS.txt`](https://physionet.org/files/chfdb/1.0.0/SHA256SUMS.txt).
PhysioNet also distributes these identical files through its public S3 bucket.

The first ECG channel is decoded from WFDB format 212 into digital sample
values. Each interior heartbeat is bounded by the integer midpoints between
its annotation and the preceding and following annotations. The segment is
linearly interpolated to 140 values, then standardized to zero mean and unit
population standard deviation. The CSV rounds these values to eight decimals.

Keep every interior beat annotated as premature ventricular contraction
(code 5), premature/ectopic supraventricular beat (9), or R-on-T premature
ventricular contraction (41). Draw 2,943 normal beats (code 1) without replacement
using NumPy `RandomState(21)`, and order the selected 5,000 beats chronologically.
Unclassifiable beats (code 13) and the first and last annotations are excluded.
The source annotations are automated and have not been manually corrected.
"Normal" refers to the beat annotation, not to the patient's overall health.

`chf07_heartbeats.csv` has no header: each row holds 140 values followed by
label `1` for a normal beat or `0` for a non-normal beat.

| Source annotation | Bundled beats | Binary label |
| --- | ---: | ---: |
| Normal (1) | 2,943 | 1 |
| R-on-T premature ventricular contraction (41) | 1,767 | 0 |
| Premature ventricular contraction (5) | 96 | 0 |
| Premature/ectopic supraventricular beat (9) | 194 | 0 |

`test_indices.csv` holds the first 1,000 entries of a fresh
`RandomState(21).permutation(5000)`, using zero-based CSV row indices.
Testing contains 608 normal and 392 non-normal beats. The example trains on
the remaining 2,335 normal beats; other development beats are unused.
Evaluation results apply to this OpenNN-derived subset.

## Reproduce the bundled files

Run the following Python 3.12 snippet from the repository root with NumPy
2.3.5. Original recordings are cached outside the checkout; the output replaces
the two bundled CSV files. It requires internet access on the first run.

```python
import hashlib
from pathlib import Path
from urllib.request import urlopen
import numpy as np

cache = Path("../opennn-example-sources/chfdb-1.0.0")
cache.mkdir(parents=True, exist_ok=True)
base = "https://physionet-open.s3.amazonaws.com/chfdb/1.0.0"
hashes = {
    "chf07.dat": "4136216e2cde35d4773fe3face5c14fe80e0c0de30ac753b35833b68bbd8bd8b",
    "chf07.ecg": "d73603dfe69383c02267ab03e99b23eea6ca4e1e2de28c24e720a95267e88f8e",
    "chf07.hea": "8784600b73070fa480422223d332d30aac00a5eb81a72cfe4dcae6f17ca5c5a7",
}
for name, digest in hashes.items():
    path = cache / name
    if not path.exists():
        with urlopen(f"{base}/{name}", timeout=60) as response:
            path.write_bytes(response.read())
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Source checksum mismatch: {name}")

packed = np.frombuffer((cache / "chf07.dat").read_bytes(), dtype=np.uint8)
packed = packed.reshape(-1, 3).astype(np.int16)
signal = packed[:, 0] | ((packed[:, 1] & 15) << 8)
signal = np.where(signal >= 2048, signal - 4096, signal)
words = np.frombuffer((cache / "chf07.ecg").read_bytes(), dtype="<u2")
assert words[-1] == 0
words = words[:-1]
codes = words >> 10
assert np.all(np.isin(codes, [1, 5, 9, 13, 41]))
times = np.cumsum(words & 1023, dtype=np.int64)
interior = np.arange(1, len(times) - 1)
abnormal = interior[np.isin(codes[interior], [5, 9, 41])]
normal = interior[codes[interior] == 1]
draw = np.random.RandomState(21).choice(normal, 5000 - len(abnormal), replace=False)
selected = np.sort(np.concatenate([draw, abnormal]))

rows = []
for beat in selected:
    start = (times[beat - 1] + times[beat]) // 2
    stop = (times[beat] + times[beat + 1]) // 2
    segment = signal[start:stop]
    values = np.interp(np.linspace(0, len(segment) - 1, 140),
                       np.arange(len(segment)), segment)
    assert values.std() > 0
    values = (values - values.mean()) / values.std()
    rows.append(np.append(values, int(codes[beat] == 1)))

output = Path("examples/electrocardiogram_anomaly_detection/data")
output.mkdir(parents=True, exist_ok=True)
np.savetxt(output / "chf07_heartbeats.csv", np.array(rows), delimiter=",",
           fmt=["%.8f"] * 140 + ["%d"])
testing = np.random.RandomState(21).permutation(len(selected))[:1000]
np.savetxt(output / "test_indices.csv", testing, fmt="%d")
```

SHA-256 of `chf07_heartbeats.csv` (LF line endings):
`b326f4ef9ea8dea02038bd5102c13996fb06d0993f8ee53bd24d6e8840a03701`.

SHA-256 of `test_indices.csv` (LF line endings):
`2cc30a39e9af6e72b2b851e2cf2e51a2900256db4361f2ad00ad2edbdf671031`.
