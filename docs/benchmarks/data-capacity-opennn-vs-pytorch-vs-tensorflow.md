# Data capacity: how much can you load and train in a fixed RAM budget — OpenNN vs the default `pandas` load path (Windows)

*Benchmark note for [opennn.net/benchmarks](https://www.opennn.net/benchmarks/). Last updated 2026-06-08. Measured on Windows 11 x86_64, CPU-only.*

Before a model trains, the data has to fit in memory. On a fixed-RAM machine — a
laptop, a memory-capped container, a modest cloud instance — the practical question
is blunt: **how large a dataset can you load and start training before the process
runs out of memory?**

This note measures exactly that, comparing each tool's **default in-memory loading
path**. We take a regression dataset (the Rosenbrock benchmark), grow the number of
samples, and find the largest one each path can load **and** train one step on, inside
a fixed committed-memory budget. On the OpenNN side that is its native CSV reader; on
the Python side it is `pandas.read_csv` → tensors — the load most PyTorch and
TensorFlow projects actually write. This is a default-vs-default comparison, not a
claim about the leanest possible Python loader (see *Caveats*).

## The numbers

Under an **8 GB committed-memory cap** (a Windows Job Object, identical for both),
loading a Rosenbrock CSV of **100 inputs + 1 target** and training a 100 → 100 → 1
MLP one epoch with Adam:

| | OpenNN (native loader) | `pandas.read_csv` → tensors |
|---|---|---|
| **Max samples loaded + trained** | **16,000,000** | **6,000,000** |
| **Max values** (samples × 101) | **1.62 billion** | **0.61 billion** |
| **OOM at** | 17,000,000 | 7,000,000 |
| vs OpenNN | 1× | **2.67× less** |

**OpenNN fits ~2.7× more data than the default `pandas` load path** in the same
memory budget — it trained on 16 million samples where the `pandas` load exhausted the
cap at 7 million. (`pandas` is a stand-in for the common PyTorch/TensorFlow tabular
load; both frameworks can also stream — see *Caveats*.)

## Why OpenNN holds more

Both tools end up training on the same numbers, but they pay very differently to get
the data into memory.

**OpenNN — a compact float32 matrix, fed by a memory-mapped file.** The dataset is
parsed once into a single dense `float` (4-byte) matrix. Two properties of the loader
keep the footprint low:

* **Memory-mapped input (quote-free numeric CSV).** When the file contains no `"`
  characters — the case for this benchmark's plain numeric Rosenbrock data — the CSV
  is mapped read-only and parsed in place, so its bytes are *file-backed* pages, not
  heap memory. Under a committed-memory budget they cost nothing (the OS can drop and
  re-read them), so only the parsed matrix counts against the limit. If the file
  contains quoted fields, the loader falls back to reading the whole file into a heap
  buffer first (to strip quoted separators), which forfeits this saving; the
  capacity advantage above applies to the unquoted-numeric path.
* **Per-row tokenization.** Rows are tokenized one at a time as the matrix is filled,
  rather than building a table of every row's fields up front. There is no large
  transient index of the whole file.

The result is that OpenNN's memory while loading is essentially *the matrix itself*
(`samples × 101 × 4 bytes`) plus a small fixed overhead.

**`pandas` — the whole file becomes a DataFrame first.** The standard path,
`pandas.read_csv(...)`, materializes the entire file as a DataFrame before any tensor
exists, and the conversion to tensors holds further copies transiently. That DataFrame
is all committed memory, and it is what hits the cap first. (Measured with pandas'
default float64 read; forcing float32 moves the crash point only slightly — the
overhead is the DataFrame and its copies, not the element width.)

## Why it matters

* **Fixed-RAM machines:** on a laptop or a memory-capped instance, ~2.7× more data
  is the difference between training on your whole dataset and having to subsample.
* **The lean path is the default path:** OpenNN reaches this with a plain "load the
  CSV" call. To match it from Python you reach for a streaming pipeline
  (`IterableDataset` / `tf.data`); OpenNN's *default* already behaves frugally, with no
  pipeline to write.
* **Headroom for the model:** every gigabyte the loader doesn't spend is a gigabyte
  left for parameters, activations, and the OS.

## Caveats

* **`pandas` is the *common* default, not the leanest or only loader.** This is a
  default-vs-default comparison. A leaner *eager* load (`numpy.loadtxt` → tensor, no
  DataFrame) would narrow the gap; and both PyTorch and TensorFlow can stream data
  larger than RAM with `IterableDataset` / `tf.data` (reading the CSV in batches from
  disk), which **removes the ceiling entirely** at the cost of writing a data pipeline.
  The point here is what the out-of-the-box `read_csv` path the large majority of
  tutorials and projects use costs in memory. OpenNN likewise has an opt-in
  disk-streaming mode (`StorageMode::BinaryFile`) for datasets larger than RAM; this
  note measures the **default in-memory mode on both sides**.
* **This is a loading/footprint benchmark, not a throughput one.** It measures how
  much data fits and trains, not how fast. PyTorch and TensorFlow are general-purpose
  frameworks; OpenNN is a focused native library.
* **Measured at 100 input variables under an 8 GB committed cap**, on Windows 11
  x86_64, CPU-only (OpenNN built with MSVC Release; PyTorch 2.10.0+cpu, pandas 3.0.1,
  CPython 3.13). The *ratio* is what generalizes: it is set by per-sample memory
  efficiency, so it holds across RAM sizes and scales to wider datasets (more
  variables shifts both crash points down together).
* The crash point is made deterministic with a Job Object committed-memory limit so
  it does not depend on what else is running on the machine.

## Relation to the earlier Neural Designer note

An earlier Neural Designer article
([Data capacity of TensorFlow, PyTorch, and Neural Designer](https://www.neuraldesigner.com/blog/capacity-comparison-tensorflow-vs-pytorch-vs-neural-designer/),
2023) reported a ~1.8× capacity advantage on a 16 GB machine. This note reproduces
the *spirit* of that result with current versions and a reproducible, capped
methodology — and the measured advantage here is larger (~2.7×), reflecting loader
improvements in the OpenNN engine (memory-mapped, per-row CSV parsing) rather than the
original setup.

## Reproducing

The programs and driver are in [`docs/benchmarks/capacity/`](capacity/):

* `generate_rosenbrock.c` — streams a Rosenbrock CSV of any size to disk.
* `opennn_capacity.cpp` — loads a CSV into a `TabularDataset`, trains one Adam epoch,
  prints peak/sustained RAM. Built as the `opennn_capacity` CMake target.
* `pytorch_capacity.py` — the pandas → tensors equivalent.
* `run_capped.c` — runs a child under a Job Object committed-memory cap (Windows).
* `capacity_search.ps1` — grows the sample count until each engine OOMs under the cap.

```powershell
# build the generator, the cap launcher, and the OpenNN test
clang -O2 -o generate_rosenbrock.exe generate_rosenbrock.c
clang -O2 -municode -o run_capped.exe run_capped.c
cmake --build build --config Release --target opennn_capacity

# sweep both engines under an 8 GB cap, 100 variables
powershell -File capacity_search.ps1 -CapGB 8 -Variables 100
# -> capacity_results.csv : max samples each engine loads + trains before OOM
```
