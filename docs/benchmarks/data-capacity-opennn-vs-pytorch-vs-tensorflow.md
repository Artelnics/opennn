# Data capacity: OpenNN vs PyTorch vs TensorFlow

OpenNN loads and trains on about 2.7x more tabular data than the common pandas-to-tensor path under
the same fixed RAM budget.

Before a model trains, the data has to fit in memory. On a fixed-RAM machine — a laptop, a
memory-capped container, or a modest cloud instance — the practical question is simple: how large a
dataset can you load and start training before the process runs out of memory?

This benchmark compares OpenNN’s native in-memory CSV loading path with the default Python workflow
commonly used before PyTorch or TensorFlow training: `pandas.read_csv` followed by conversion to
tensors.

> This is a default-vs-default loading benchmark. PyTorch and TensorFlow can stream data from disk
> with custom pipelines, but the common eager path in many tabular projects starts with pandas.

## Contents

- [Introduction](#introduction)
- [Benchmark application](#benchmark-application)
- [Reference computer](#reference-computer)
- [Methodology](#methodology)
- [Results](#results)
- [Discussion](#discussion)
- [Conclusions](#conclusions)
- [References](#references)

## Introduction

Framework size, startup latency and runtime memory all matter, but there is another constraint that
appears even earlier: the dataset itself. If loading the data exhausts memory, training never
starts.

For tabular machine learning in Python, the most common first step is still to load the whole CSV
into a pandas DataFrame and then convert it into tensors for PyTorch or TensorFlow. That workflow is
convenient, but it carries a significant memory cost.

OpenNN uses a native C++ loader that parses the CSV into a compact dense matrix. This benchmark
measures how much larger a dataset OpenNN can load and train on within the same memory budget.

## Benchmark application

The benchmark uses a regression dataset generated from the Rosenbrock function. Each sample has 100
input variables and 1 target variable, for a total of 101 numeric values per row.

The model is a small multilayer perceptron:

| Item | Value |
| --- | --- |
| Problem type | Tabular regression |
| Dataset | Rosenbrock synthetic benchmark |
| Input variables | 100 |
| Target variables | 1 |
| Network | 100 → 100 → 1 MLP |
| Training | One epoch with Adam |

## Reference computer

| Item | Value |
| --- | --- |
| Operating system | Windows 11 x86_64 |
| Execution mode | CPU-only |
| Memory cap | 8 GB committed-memory limit |
| OpenNN build | MSVC Release |
| Python stack | CPython 3.13, pandas 3.0.1, PyTorch 2.10.0+cpu |

## Methodology

The test grows the number of samples in the CSV and checks whether each implementation can load the
dataset and train one epoch before hitting the memory cap.

The cap is enforced with a Windows Job Object committed-memory limit. This makes the failure point
deterministic and independent of what else is running on the machine.

The two loading paths are:

- **OpenNN:** native CSV reader into a compact float32 matrix.
- **Python path:** `pandas.read_csv` followed by tensor conversion, representing the common eager
  loading workflow used before PyTorch or TensorFlow training.

The benchmark measures capacity, not speed: the result is the largest dataset that can be loaded and
used for training inside the fixed memory budget.

## Results

| Metric | OpenNN | Pandas (PyTorch / TensorFlow) |
| --- | --- | --- |
| Max samples loaded and trained | **16,000,000** | 6,000,000 |
| Max values loaded and trained | **1.62 billion** | 0.61 billion |
| Out of memory at | 17,000,000 samples | 7,000,000 samples |
| Relative capacity | **1x** | 2.67x less than OpenNN |

Maximum samples loaded and trained

8 GB committed-memory cap · 100 inputs + 1 target · Windows 11 CPU-only HIGHER IS BETTER ->

OpenNN

16.0M

pandas → tensors

6.0M

OpenNN fits about 2.7x more samples under the same memory limit. The comparison measures default
eager loading paths, not custom streaming pipelines.

## Discussion

### Why OpenNN holds more data

OpenNN parses the CSV directly into a dense float32 matrix. The input file is memory-mapped and read
as file-backed pages, so the loader avoids building a large intermediate table of strings or
objects. Rows are tokenized one at a time as the matrix is filled.

That keeps OpenNN’s memory footprint close to the final matrix itself: samples × variables × 4
bytes, plus a small fixed overhead.

### Why the pandas path reaches the limit earlier

The pandas workflow first materializes the entire CSV as a DataFrame. Then the data is converted
into tensors for training. During that process, the DataFrame and tensor copies can coexist, and the
DataFrame itself carries overhead beyond the raw numeric values.

Forcing pandas to read float32 can reduce element size, but it does not remove the DataFrame and
conversion overhead. That is why the memory ceiling remains much lower than OpenNN’s native path.

### What this does and does not claim

This benchmark does not say PyTorch or TensorFlow cannot train larger-than-RAM datasets. They can,
using custom data pipelines such as `IterableDataset` or `tf.data`. The point is that OpenNN’s lean
loading behavior is the default path, while matching it in Python usually requires writing a
streaming pipeline.

## Conclusions

- OpenNN loaded and trained on 16 million samples under an 8 GB committed-memory cap.
- The common pandas-to-tensor path loaded and trained on 6 million samples under the same cap.
- That gives OpenNN about 2.7x more data capacity for this tabular workload.
- The advantage comes from OpenNN’s compact native loader and direct dense matrix storage.
- For fixed-RAM machines, that extra capacity can be the difference between training on the full
  dataset and having to subsample.

## References

- [The Rosenbrock benchmark for machine
  learning](https://www.neuraldesigner.com/blog/the-rosenbrock-benchmark-for-machine-learning/),
  Neural Designer.
- [pandas.read_csv
  documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html).
- [PyTorch data loading documentation](https://pytorch.org/docs/stable/data.html).
- [TensorFlow tf.data guide](https://www.tensorflow.org/guide/data).
- [Data capacity of TensorFlow, PyTorch, and Neural
  Designer](https://www.neuraldesigner.com/blog/capacity-comparison-tensorflow-vs-pytorch-vs-neural-designer/),
  Neural Designer, 2023.
