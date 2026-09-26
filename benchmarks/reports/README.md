# Benchmark results and publication review

Reviewed through 16 September 2026. **These observations are not ready for the next website release.**

We compare OpenNN C++ applications with the PyTorch Python API. The tables below preserve the measured values, explain their limits and identify the work that remains. They do not claim that every comparison is valid or that OpenNN always wins.

[Commands](../README.md) · [Measurement protocol](../PROTOCOL.md) ·
[Publication procedure](../PROTOCOL.md#15-publication-procedure)

Contents: [Throughput](#throughput), [memory](#memory), [energy](#energy),
[workload evidence limits](#limits-of-the-selected-performance-records),
[startup](#startup), [deployment](#deployment-size), [quality](#prediction-quality),
[code and dependencies](#code-and-dependencies), [earlier footprint tests](#earlier-footprint-tests),
[website outline](#website-outline), [history](#evidence-and-preservation).

## What is available

| Comparison | Evidence | Next step |
| --- | --- | --- |
| Throughput, memory and energy | 12 historical configurations selected here; further cells are reported complete | Locate and review the existing artifacts; do not repeat completed cells. |
| CPU CNN and Transformer performance | No results in this selected historical set | Locate completed training and inference artifacts before changing this table. |
| Startup | Historical laptop matrix plus 660 successful reference-machine timed launches | Resolve high-CV cells and desktop GPU activity before publication. |
| Deployment size | 12 historical laptop configurations; 28 reference-machine variants measured and isolated-root replayed | Test a different physical target before claiming cross-machine portability. |
| Prediction quality | Five-seed CUDA FP32 real-data runs for the four primary models, including corrected-kernel locked translation replay | Declare prospective quality margins and collect untouched confirmation before any four-model parity claim. |
| Code size and dependencies | Matched pinned source counts and reference-machine runtime inventories | Review native closures and clean-target completeness. |

## Computers and software

The performance records come from the reference computer: Intel Core i7-14700F, 32 GB RAM, NVIDIA RTX 5070 Ti, Linux, PyTorch 2.13.0+cu130 and NVIDIA driver 610.43.02. The CPU runs use FP32; GPU runs are labelled BF16, with an LSTM precision discrepancy described below.

Eight performance records belong to session `2026-09-06-publish` at `e76425bd3`. The four GPU CNN and Transformer records belong to `2026-09-06-energy-publish` at `520f0f2f8`. Each ratio uses its own paired session. These records do not describe the current checkout.

The historical startup and deployment tables come from a different computer:
Intel Core i7-12700H and RTX 3060 Laptop GPU under WSL2. OpenNN uses CUDA
12.9 there; the PyTorch wheel uses CUDA 13.0, with separate cuDNN 9.20
distributions. The later reference-machine startup and deployment tables are
labelled separately below.

## Throughput

Throughput counts completed work per second after warm-up. The value is the median of three launches. It does not measure how long a model takes to reach a target accuracy. Each model has its own unit.

| Configuration | Batch | Precision | Unit | OpenNN | PyTorch | Ratio | Variation ON / PT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CPU Dense inference | 4096 | fp32 | samples/s | 220,512 | 170,329 | 1.295× | 0.0381% / 0.268% |
| CPU Dense training | 4096 | fp32 | samples/s | 70,120 | 54,520 | 1.286× | 0.139% / 1.86% |
| CPU LSTM inference | 256 | fp32 | windows/s | 80,285 | 69,304 | 1.158× | 0.276% / 4.2% |
| CPU LSTM training | 256 | fp32 | windows/s | 23,251 | 13,103 | 1.774× | 0.809% / 1.08% |
| GPU CNN inference | 128 | bf16 | images/s | 7,060 | 5,634 | 1.253× | 0% / 0.0875% |
| GPU CNN training | 64 | bf16 | images/s | 1,667 | 1,401 | 1.190× | 0.104% / 0.109% |
| GPU Dense inference | 8192 | bf16 | samples/s | 39,387,890 | 38,689,107 | 1.018× | 0.0033% / 0.0231% |
| GPU Dense training | 8192 | bf16 | samples/s | 11,406,741 | 10,048,603 | 1.135× | 0.154% / 3.38% |
| GPU LSTM inference | 256 | bf16 | windows/s | 2,716,548 | 513,813 | 5.287× | 0.00194% / 1.57% |
| GPU LSTM training | 256 | bf16 | windows/s | 823,255 | 95,842 | 8.590× | 1.02% / 0.592% |
| GPU Transformer inference | 32 | bf16 | sequences/s | 5,413 | 4,694 | 1.153× | 0% / 0.0325% |
| GPU Transformer training | 32 | bf16 | sequences/s | 1,352 | 1,145 | 1.181× | 0.0427% / 0% |

OpenNN throughput ÷ PyTorch throughput. Higher is better. Variation is the sample standard deviation divided by the mean, expressed as a percentage. A value above 3% fails the current stability rule.

CPU CNN training, CPU CNN inference, CPU Transformer training and CPU Transformer inference are **not measured in this selected result set**. Existing completed artifacts must be located and reviewed before filling these historical gaps; these cells are not scheduled for another run.
The accessible result stores contain CPU CNN and Transformer startup and
deployment records, but no paired workload-performance files for those four
slots. Their completed measurements may be stored elsewhere; this report
does not substitute smoke timings or repeat the cells.

## Memory

The table shows the highest measured memory use across three launches. CPU memory is the process's peak anonymous resident memory. GPU memory is total device memory in use minus the idle baseline. It includes runtime and allocator overhead. These are different measures and remain labelled separately.

| Configuration | OpenNN (MiB) | PyTorch (MiB) | OpenNN / PyTorch |
| --- | --- | --- | --- |
| CPU Dense inference | 280.4 | 568.7 | 49.3% |
| CPU Dense training | 305.2 | 787.3 | 38.8% |
| CPU LSTM inference | 156.5 | 458.8 | 34.1% |
| CPU LSTM training | 209.5 | 591.4 | 35.4% |
| GPU CNN inference | 845.8 | 1,267.2 | 66.7% |
| GPU CNN training | 3,572.9 | 4,227.9 | 84.5% |
| GPU Dense inference | 371.3 | 411.3 | 90.3% |
| GPU Dense training | 507.6 | 631.6 | 80.4% |
| GPU LSTM inference | 293.6 | 441.6 | 66.5% |
| GPU LSTM training | 315.6 | 511.6 | 61.7% |
| GPU Transformer inference | 627.2 | 1,161.2 | 54.0% |
| GPU Transformer training | 2,321.2 | 3,349.6 | 69.3% |

OpenNN memory ÷ PyTorch memory × 100. Lower is better. For example, 60% means OpenNN uses 60% as much memory, or 40% less. These measurements do not establish a maximum model or dataset size.

## Energy

Energy is the median across three launches for the work recorded in each configuration. CPU readings measure the processor package. GPU readings measure the graphics board. Neither measures electricity at the wall socket.

| Configuration | Measured component | OpenNN (Wh) | PyTorch (Wh) | OpenNN / PyTorch |
| --- | --- | --- | --- | --- |
| CPU Dense inference | CPU package | 0.0979 | 0.1063 | 92.1% |
| CPU Dense training | CPU package | 0.0862 | 0.0974 | 88.5% |
| CPU LSTM inference | CPU package | 0.0216 | 0.0233 | 92.4% |
| CPU LSTM training | CPU package | 0.0431 | 0.0608 | 70.8% |
| GPU CNN inference | GPU board | 2.3976 | 3.1327 | 76.5% |
| GPU CNN training | GPU board | 3.7574 | 4.2545 | 88.3% |
| GPU Dense inference | GPU board | 0.1731 | 0.1830 | 94.6% |
| GPU Dense training | GPU board | 0.1317 | 0.1520 | 86.7% |
| GPU LSTM inference | GPU board | 0.2575 | 0.6478 | 39.8% |
| GPU LSTM training | GPU board | 0.0292 | 0.1320 | 22.1% |
| GPU Transformer inference | GPU board | 11.4501 | 15.0000 | 76.3% |
| GPU Transformer training | GPU board | 16.2770 | 22.7057 | 71.7% |

OpenNN energy ÷ PyTorch energy × 100. Lower is better. We do not average CPU package energy with GPU board energy, or infer an electricity-bill reduction from these component measurements.

## Limits of the selected performance records

All twelve records lack content hashes for their prepared inputs. A filename, size and modification date do not prove that the engines received the same data. Ten records also have missing numerical quality values despite an old passing flag. That flag does not prove prediction quality or output agreement.

The recomputed checks also flag PyTorch memory variation in CPU dense training. Full per-engine ranges, variation and check results are in the accompanying observations and readiness files.

### Dense

Both engines use a 28 → 1024 → 1024 → 1 classifier on HIGGS. The historical training paths use different sample-order policies. GPU dense training exceeds the throughput variation limit on the PyTorch side. Its energy and throughput windows also differ in duration. These limits must be resolved from the existing artifacts before a stronger claim is made.

### LSTM

Both engines use 128 LSTM units and 24-step windows of 15 weather features. The old GPU records are labelled BF16, but the archived analysis identifies FP16 recurrent kernels on the PyTorch side. The old PyTorch inference driver kept a full dataset on the GPU while replaying only one batch. Training also differs in scaling layers and data residency. CPU LSTM inference exceeds the throughput variation limit on the PyTorch side. These are unresolved limits of the selected historical evidence.

### CNN

The historical model is ResNet-50 v1.5 with 25,557,032 parameters, using 224 × 224 images from a 50,000-image ImageNet subset. It is a different workload from the older CIFAR-sized website article. The old training runs use different image transforms, decoding paths, transfer sizes and warm-up counts. The selected records do not establish matched pixels or timing boundaries.

### Transformer

The historical model has six encoder and six decoder layers, width 512, eight attention heads and a 20,000-token vocabulary on WMT14 text. The old drivers differ in causal and padding masks, dropout, and the loss denominator. These selected performance records do not establish equivalent work. Translation quality must use generated text on held-out data; equal tensor shapes are not sufficient.

The current code includes changes made after these runs. Fixes in the code cannot repair an old measurement. Completed throughput, workload-memory and energy cells are being preserved; this review does not repeat them. Different vendor libraries can also affect results, so conclusions must describe the tested configurations rather than assign every difference to framework code.

## Startup

The historical WSL2 laptop matrix below starts a fresh process for each sample and stops when its first prediction reaches host memory. Each configuration had 15 timed launches. **43 of 44 engine configurations exceeded the 3% variation limit** on that computer, so its paired comparisons remain diagnostic.

| Configuration | Model | OpenNN (ms) | PyTorch (ms) | OpenNN / PyTorch |
| --- | --- | --- | --- | --- |
| cpu-eigen / fp32 | Dense | 1.739 | 661.609 | 0.3% |
| cpu-eigen / fp32 | LSTM | 4.804 | 652.580 | 0.7% |
| cpu-eigen / fp32 | CNN | 3.427 | 670.453 | 0.5% |
| cpu-eigen / fp32 | Transformer | 3.479 | 648.294 | 0.5% |
| cpu-mkl / fp32 | Dense | 9.863 | 661.609 | 1.5% |
| cpu-mkl / fp32 | LSTM | 16.025 | 652.580 | 2.5% |
| cpu-mkl / fp32 | CNN | 11.198 | 670.453 | 1.7% |
| cpu-mkl / fp32 | Transformer | 11.325 | 648.294 | 1.7% |
| cuda / fp32 / saved cache | Dense | 787.974 | 1,538.936 | 51.2% |
| cuda / fp32 / empty cache | Dense | 872.797 | 1,531.146 | 57.0% |
| cuda / bf16 / saved cache | Dense | 914.584 | 1,589.930 | 57.5% |
| cuda / bf16 / empty cache | Dense | 910.337 | 1,633.804 | 55.7% |
| cuda / fp32 / saved cache | LSTM | 848.630 | 1,533.287 | 55.3% |
| cuda / fp32 / empty cache | LSTM | 875.581 | 1,553.056 | 56.4% |
| cuda / bf16 / saved cache | LSTM | 928.217 | 1,613.484 | 57.5% |
| cuda / bf16 / empty cache | LSTM | 868.236 | 1,620.082 | 53.6% |
| cuda / fp32 / saved cache | CNN | 928.344 | 1,591.334 | 58.3% |
| cuda / fp32 / empty cache | CNN | 947.681 | 1,625.366 | 58.3% |
| cuda / bf16 / saved cache | CNN | 1,061.431 | 1,732.044 | 61.3% |
| cuda / bf16 / empty cache | CNN | 1,047.034 | 1,732.080 | 60.4% |
| cuda / fp32 / saved cache | Transformer | 871.942 | 1,649.388 | 52.9% |
| cuda / fp32 / empty cache | Transformer | 855.075 | 1,645.411 | 52.0% |
| cuda / bf16 / saved cache | Transformer | 922.554 | 1,760.148 | 52.4% |
| cuda / bf16 / empty cache | Transformer | 944.072 | 1,757.436 | 53.7% |

OpenNN startup time ÷ PyTorch startup time × 100. Lower is better. Saved or empty cache refers to application tuning; filesystem caches are warm. No trained model file is loaded. The older footprint timer measured process lifetime and must not be used in this table.

On the reference machine, all 44 application configurations passed a fresh
process warm-up and first-prediction shape gate in
`startup-20260915T180208Z-63ad794e`. Those smoke times are not speed results.
The later timed matrix `startup-20260915T221250Z-8b337801` completed all
660 launches with zero failures: 15 samples in each of 44 engine cells,
forming 24 OpenNN/PyTorch pairs. Each time is launch to the first completed
host prediction.

| Backend | Model | Precision | Cache | OpenNN (ms) | PyTorch (ms) | OpenNN/PyTorch | OpenNN CV | PyTorch CV |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| cpu-eigen | dense | fp32 | reused | 1.113 | 497.857 | 0.2% | 9.59% | 0.85% |
| cpu-eigen | lstm | fp32 | reused | 3.576 | 497.310 | 0.7% | 3.22% | 0.70% |
| cpu-eigen | cnn | fp32 | reused | 2.473 | 496.376 | 0.5% | 4.89% | 1.80% |
| cpu-eigen | transformer | fp32 | reused | 1.757 | 500.766 | 0.4% | 7.40% | 0.77% |
| cpu-mkl | dense | fp32 | reused | 8.965 | 497.857 | 1.8% | 2.81% | 0.85% |
| cpu-mkl | lstm | fp32 | reused | 11.818 | 497.310 | 2.4% | 2.33% | 0.70% |
| cpu-mkl | cnn | fp32 | reused | 10.298 | 496.376 | 2.1% | 2.47% | 1.80% |
| cpu-mkl | transformer | fp32 | reused | 9.582 | 500.766 | 1.9% | 4.74% | 0.77% |
| cuda | dense | bf16 | empty | 254.301 | 812.135 | 31.3% | 1.17% | 0.59% |
| cuda | dense | bf16 | reused | 253.000 | 812.686 | 31.1% | 0.82% | 1.69% |
| cuda | dense | fp32 | empty | 238.094 | 784.303 | 30.4% | 0.84% | 1.36% |
| cuda | dense | fp32 | reused | 237.149 | 783.675 | 30.3% | 3.63% | 1.03% |
| cuda | lstm | bf16 | empty | 262.143 | 816.944 | 32.1% | 1.15% | 1.62% |
| cuda | lstm | bf16 | reused | 258.148 | 809.430 | 31.9% | 3.36% | 0.61% |
| cuda | lstm | fp32 | empty | 253.695 | 800.677 | 31.7% | 0.52% | 2.96% |
| cuda | lstm | fp32 | reused | 249.067 | 794.700 | 31.3% | 1.56% | 0.83% |
| cuda | cnn | bf16 | empty | 341.286 | 916.751 | 37.2% | 5.88% | 1.70% |
| cuda | cnn | bf16 | reused | 330.693 | 909.975 | 36.3% | 0.49% | 0.90% |
| cuda | cnn | fp32 | empty | 316.379 | 849.485 | 37.2% | 0.59% | 0.95% |
| cuda | cnn | fp32 | reused | 312.750 | 845.224 | 37.0% | 4.53% | 3.12% |
| cuda | transformer | bf16 | empty | 260.924 | 897.158 | 29.1% | 1.05% | 1.40% |
| cuda | transformer | bf16 | reused | 255.880 | 899.398 | 28.5% | 0.97% | 3.18% |
| cuda | transformer | fp32 | empty | 243.277 | 877.875 | 27.7% | 1.10% | 1.72% |
| cuda | transformer | fp32 | reused | 242.281 | 877.892 | 27.6% | 1.10% | 2.40% |

The table reports each engine's coefficient of variation (CV). Eleven of
44 engine cells and ten of 24 pairs exceed the 3% stability limit.
The isolation control `startup-isolation-20260915T221238Z.json` moved
23 IDE/sync processes and 553 threads to CPUs 16–27, pinned the runner and
children to CPUs 0–15, and restored all affinities afterward. Before the
run, performance-core busy fraction was 0.19%. Desktop Xorg, GNOME and VS
Code graphics activity kept the GPU at 8% in five pre-run samples; no other
GPU compute process was present. The three round snapshots reported GPU
utilization of 4%, 4% and 3% and all-core CPU busy fractions of 4.15%,
0.68% and 4.15%. GPU clocks were not locked. These reference-machine
startup ratios are completed local diagnostics, not publication-ready
GPU comparisons; high-CV pairs likewise need a quieter rerun if a stable
ratio is required.

## Deployment size

These values count files used to deploy each small application. OpenNN includes its executable and exercised native libraries. PyTorch includes the application, interpreter, standard library and complete installed runtime packages. This is a comparison with a standard Python installation, not two bundles reduced to their smallest possible size.

| Backend | Application | OpenNN (MB) | PyTorch (MB) | OpenNN / PyTorch |
| --- | --- | --- | --- | --- |
| cpu-eigen | Dense | 4.9 | 809.0 | 0.6% |
| cpu-eigen | LSTM | 4.9 | 809.0 | 0.6% |
| cpu-eigen | CNN | 4.9 | 809.0 | 0.6% |
| cpu-eigen | Transformer | 4.9 | 809.0 | 0.6% |
| cpu-mkl | Dense | 285.2 | 809.0 | 35.2% |
| cpu-mkl | LSTM | 285.2 | 809.0 | 35.2% |
| cpu-mkl | CNN | 285.2 | 809.0 | 35.2% |
| cpu-mkl | Transformer | 285.2 | 809.0 | 35.2% |
| cuda | Dense | 1,086.5 | 4,806.9 | 22.6% |
| cuda | LSTM | 1,359.2 | 4,806.9 | 28.3% |
| cuda | CNN | 1,694.7 | 4,806.9 | 35.3% |
| cuda | Transformer | 1,086.5 | 4,806.9 | 22.6% |

OpenNN deployment size ÷ PyTorch deployment size × 100. Lower is better. MB means 1,000,000 bytes. GPU FP32 and BF16 rows share the same verified file inventory and are shown once. The figures exclude trained weights, datasets, operating-system and driver files, and generated caches. A clean-machine launch is still needed to show that each collected bundle is complete.

The new reference-machine deployment inventory at `84bd178e8` is
`deployment-20260915T180602Z-4626baf7/results.json`. All 28 FP32/BF16
variants launched, passed their shape checks, and had no unresolved native
dependencies. FP32 and BF16 variants share one file inventory, so the table
shows each backend/model once. The figures use decimal MB and compare the
OpenNN executable plus exercised native library closure with a standard,
separately pinned PyTorch Python installation. They remain local diagnostics
because the benchmark worktree has uncommitted changes. The separate
`deployment-isolated-replay-20260915.json` created a new bubblewrap mount root
containing the collected application inventories and 13 explicit baseline
OS/driver files. All 28 FP32/BF16 variants launched there and reproduced the
parameter and output-shape gates. This validates local inventory completeness
against the installed OS/driver; another physical target is still needed for
a cross-machine portability claim.

| Backend | Application | OpenNN (MB) | PyTorch (MB) | OpenNN / PyTorch |
| --- | --- | ---: | ---: | ---: |
| cpu-eigen | Dense | 4.933 | 814.633 | 0.6% |
| cpu-eigen | LSTM | 4.934 | 814.633 | 0.6% |
| cpu-eigen | CNN | 4.942 | 814.633 | 0.6% |
| cpu-eigen | Transformer | 4.933 | 814.633 | 0.6% |
| cpu-mkl | Dense | 286.193 | 814.633 | 35.1% |
| cpu-mkl | LSTM | 286.194 | 814.633 | 35.1% |
| cpu-mkl | CNN | 286.193 | 814.633 | 35.1% |
| cpu-mkl | Transformer | 286.189 | 814.633 | 35.1% |
| cuda | Dense | 881.074 | 4,811.046 | 18.3% |
| cuda | LSTM | 987.605 | 4,811.046 | 20.5% |
| cuda | CNN | 1,443.964 | 4,811.046 | 30.0% |
| cuda | Transformer | 881.065 | 4,811.046 | 18.3% |

The OpenNN CPU Eigen Dense closure contained four deployed files including
the executable; CPU MKL Dense contained eleven, including MKL and oneDNN;
CUDA Dense contained twelve, including cuDNN. OpenNN needed no Python
packages. The clean PyTorch CPU and CUDA installations contained ten and
twenty-nine pinned Python packages respectively; their complete installation,
interpreter and standard library are included in the totals. Package counts
and native file counts are different units and should not be compared as one
dependency-count ratio. The first inventory pass included the Linux
distribution's optional `/etc/python3.12/sitecustomize.py`; the runner now
excludes that OS file, and the table uses the corrected second pass.

## Prediction quality

| Model | Task | Metric | OpenNN | PyTorch | Status |
| --- | --- | --- | --- | --- | --- |
| Dense | HIGGS classification | Test accuracy (%) | 68.807 ± 0.160 | 68.798 ± 0.145 | Five CUDA FP32 seeds, diagnostic |
| LSTM | PM2.5 forecasting | Test RMSE in original units | 23.074 ± 0.529 | 22.870 ± 0.552 | Five CUDA FP32 seeds, diagnostic |
| LSTM | PM2.5 forecasting | Test RMSE in original units | 23.111 ± 0.544 | 23.112 ± 0.788 | Twenty shared-initialization CUDA FP32 pairs; equivalence established within ±0.902 RMSE |
| CNN | ImageNet subset classification | Test top-1 accuracy (%) | 3.196 ± 1.107 | 3.402 ± 1.264 | Five mixed-order, scaled CUDA FP32 seeds; low-quality diagnostic |
| CNN | CIFAR-10 classification | Test accuracy (%) | 67.108 ± 3.665 | 66.734 ± 3.390 | Five CUDA FP32 seeds, diagnostic |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 3.334 | 3.687 | Pre-fix two-epoch pilot; low achieved quality |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 0.0203 | 7.532 | Pre-fix ten-epoch pilot; OpenNN quality collapse |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 0.0167 | 7.468 | Pre-fix checkpoints; locked disjoint 1,000-row replay |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 7.025 | 7.532 | Corrected-kernel ten-epoch pilot; one seed, low achieved quality |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 7.177 | 7.468 | Corrected checkpoint; locked disjoint 1,000-row repair replay |
| Transformer | News Commentary v9 English–German | Generated-translation BLEU | 7.389 ± 0.122 | 7.541 ± 0.130 | Five corrected ten-epoch seeds; excluded locked 1,000-row replay, low-quality diagnostic |

All rows used CUDA FP32 with TF32 disabled. That identifies the arithmetic
setting; the table tests achieved held-out quality, not bitwise equality of
weights or predictions. Similar means alone do not establish training-quality
equivalence. The twenty-pair LSTM row is the exception: it passed the frozen
equivalence rule described below.

The five-seed Dense, earlier LSTM, ImageNet CNN, CIFAR-10 CNN and corrected
locked Transformer values are held-out means and sample standard deviations.
The ImageNet source is `quality-imagenet-mixed-scaled-five-seeds-20260916`;
the corrected Transformer scores and checkpoint hashes are in
`quality-transformer-five-seed-locked-comparison-20260916.json`.

The five-seed Dense, earlier LSTM and CIFAR-10 CNN values use seeds 42–46,
with their raw records under sessions
`quality-20260915T114824Z-79f389ec`,
`quality-20260915T115431Z-c18c0b2b` and
`quality-20260915T152121Z-081de320`, respectively. They are descriptive; no equivalence
margin was predeclared and the quality gate remains unassessed. Accuracy and
BLEU are higher-is-better; RMSE is lower-is-better. Do not average them into
one quality score. Synthetic smoke results and the older Rosenbrock task do
not fill these rows.

An exploratory independent-seed Welch 95% interval for OpenNN minus PyTorch
mean quality (`quality-uncertainty-20260915.json`) spans −0.214 to +0.233
accuracy points for Dense, −0.584 to +0.993 original-unit RMSE for LSTM,
and −4.780 to +5.528 accuracy points for CIFAR-10 CNN. The CNN interval
allows a several-point difference in either direction despite its close
observed means. These are descriptive intervals computed after the runs,
not a predeclared equivalence test; they cannot establish four-model parity.

An earlier word-vocabulary WMT14 run generated `[UNK]` for every test sentence,
so its BLEU cannot support translation-quality parity. A later teacher-forced
probe improved but still decoded one generic hypothesis; teacher-forced loss
alone is insufficient. The intact News Commentary v9 preparation retains
Unicode text and uses a shared training-only SentencePiece BPE vocabulary.
Its predeclared 1,000-row test subset has zero target `[UNK]` IDs, and replaying
its target IDs through the neutral decoder scores 100 BLEU. The two-epoch
paired pilot scores 3.334 BLEU for OpenNN and 3.687 for PyTorch; both engines
generated 1,000 distinct hypotheses with zero `[UNK]` tokens, and 998 and 999
respectively terminated at EOS. Their 0.353-BLEU difference is small, but
both achieved scores remain too low for a translation-quality parity claim.
The raw record is `quality-20260915T153931Z-d5c027c8`; these are News
Commentary split scores, not standard WMT14 newstest scores. The two engines
used the same 74,878,496-parameter profile, training tensors, test rows and
neutral scorer.
A locked second 1,000-row subset (`--transformer-test-seed 20260916
--transformer-exclude-test-seed 20260915`) excludes every pilot row. Its
training tensor hashes and tokenizer pieces/scores match the pilot, and a
target-token replay scores 99.96 BLEU. The training budget was fixed before
this subset was scored. Its generated hypotheses were inspected after both
pre-fix checkpoints had been frozen; a corrected-kernel replay is therefore
an exploratory repair check, not an untouched confirmatory test.
The two serialized tokenizer files have different byte hashes because their
stored training-input and model-prefix paths differ; their 20,000 pieces,
normalizer and checked encode/decode behavior match.

The paired ten-epoch run on the original pilot rows is recorded under
`quality-20260915T175100Z-65fec1b6`. PyTorch scored 7.532 BLEU and produced
1,000 distinct hypotheses, while OpenNN scored 0.0203 BLEU and produced the
same hypothesis for all 1,000 rows. Neither engine emitted `[UNK]`; no OpenNN
hypothesis reached EOS, versus 998 of 1,000 for PyTorch. OpenNN used only four
distinct generated token IDs, versus 5,790 for PyTorch. On the same eight
teacher-forced test rows, OpenNN had 7.06 mean cross-entropy and 7.28% token
accuracy versus PyTorch's 3.09 and 45.01%. PyTorch's online training-loss
history fell through epoch ten
to 1.571, while OpenNN's post-epoch full-train loss rose after epoch five to
7.113; these two history definitions differ, so their levels should not be
subtracted. The matched test probe and generated scores establish a severe
quality gap for this seed and budget, despite the earlier `[UNK]` repair.

The disjoint locked subset was evaluated by loading both ten-epoch checkpoints,
with no additional updates or budget selection. The raw replay is
`quality-20260915T202842Z-86038c6d`: OpenNN scored 0.0167 BLEU and generated
one repeated hypothesis, while PyTorch scored 7.468 BLEU and generated 1,000
distinct hypotheses. Neither generated `[UNK]`; zero OpenNN rows and 999
PyTorch rows reached EOS. On the same eight teacher-forced locked test rows,
OpenNN had 6.74 mean cross-entropy and 10.04% token accuracy versus PyTorch's
2.81 and 54.44%. The locked score confirms the failure is not peculiar to the
pilot test sentences; it still describes one training seed and two 1,000-row
test selections rather than model-wide quality equivalence.

Those ten-epoch OpenNN weights were trained before a CUDA LayerNorm defect was
fixed. At embedding widths 128 and above, the LayerNorm warp kernels reduced
two sums into lane 0 but the forward and backward computations on every other
lane used incomplete sums. A same-weight CPU/CUDA full-profile forward check
initially differed by about 30% on the six-layer model. Broadcasting both
completed sums to every lane repaired the width-128 standalone LayerNorm
forward and padded training-gradient regressions. With the quality-run
initializer, the six-layer forward check now differs by 0.74% and passes its
1% tolerance; the small remaining gap is a numerical diagnostic, not quality
equivalence. Loading the old OpenNN checkpoint into the corrected binary on
CPU and CUDA produces nearly identical test cross-entropy (8.91575 and
8.91582) and identical 1.89% token accuracy; the old weights themselves are
poor. The corrected OpenNN-only seed-42 ten-epoch run used the same 180,102
training rows, original pilot manifest and budget as the archived PyTorch
checkpoint. Its post-epoch full-train loss fell from 5.22 to 1.49, whereas
the pre-fix OpenNN history rose after epoch five to 7.11 by epoch ten.
On the pilot 1,000 test rows, corrected OpenNN generated 1,000 distinct
hypotheses, all reached EOS, none contained `[UNK]`, and BLEU rose to 7.025
versus PyTorch's archived 7.532. The aligned eight-row test probe showed
3.124 cross-entropy and 42.59% token accuracy, versus 3.089 and 45.01%.

Loading this corrected checkpoint on the disjoint locked 1,000 rows scored
7.177 BLEU for OpenNN versus 7.468 for the frozen PyTorch checkpoint.
Both produced 1,000 distinct hypotheses and zero `[UNK]` rows; OpenNN and
PyTorch reached EOS on 998 and 999 rows. The eight-row locked probe showed
3.182 cross-entropy and 45.56% token accuracy for OpenNN, versus 2.807 and
54.44% for PyTorch. The neutral scorer, binary hash, checkpoint hashes,
manifest hashes and archived PyTorch prediction hashes are recorded in
`quality-transformer-warp-fixed-seed42-20260915-comparison.json` and its
provenance record. These repair checks fix the observed collapse but cannot
establish model-wide quality equivalence: there is one training seed, low
absolute BLEU, a post-repair look at an already inspected locked subset,
and no predeclared equivalence margin. The earlier paired and locked scores
remain pre-fix evidence. The seed-42 repair replay used the default OpenNN host thread count; the five-seed diagnostic reran corrected seed 42 with eight host threads to match PyTorch, while seeds 43–46 used the paired runner with eight threads.

Across the five corrected checkpoints on the excluded locked subset, OpenNN scored 7.389 ± 0.122 BLEU and PyTorch 7.541 ± 0.130. The paired OpenNN-minus-PyTorch difference has a descriptive 95% interval of -0.222 to -0.082 BLEU. OpenNN produced 1000–1000 distinct hypotheses per 1,000 rows, with 0 total `[UNK]` rows and 996–1000 EOS rows; PyTorch produced 1000–1000 distinct hypotheses, 0 `[UNK]` rows and 998–1000 EOS rows. These are repair diagnostics with low absolute BLEU, previously inspected locked rows and no prospective equivalence margin; they do not establish translation-quality parity. The corrected scorer checks the test-row exclusion, manifests, checkpoint hashes and pinned binary.

For CNN, the original ImageNet subset was sorted by class, giving nearly
single-class training batches: all 1,250 batches of 32 held at most two
classes, and 277 held just one. After deterministic global row mixing, batches
averaged 31.52 distinct classes and none held just one. This repaired that
ordering, but OpenNN remained at chance until its quality-data loader also
pre-scaled training images by 255 to match inference. The optimizer skips the
network's leading image-scaling layer during training. On official CIFAR-10,
an unscaled OpenNN control scored 10.00%. The corrected five-seed paired run
averaged 67.11% for OpenNN and 66.73% for PyTorch, a 0.37-point mean gap;
individual seeds and prior runs varied by several points. The corrected
one-seed ImageNet control scored 3.66% top-1 and predicted 529 classes, versus OpenNN's
unscaled 0.10% one-class collapse and PyTorch's mixed-order 5.49% on the same
test rows. With about 40 training images per class and ten epochs, these
ImageNet scores remain a defect diagnostic rather than adequate achieved
quality or a parity comparison. The primary mixed-order, scaled five-seed run
scored 3.196 ± 1.107% top-1 for OpenNN and 3.402 ± 1.264% for PyTorch; its descriptive paired OpenNN-minus-PyTorch 95% interval spans -2.132 to +1.720 accuracy points. These sparse-data scores remain low, and the interval and post hoc inspection cannot certify parity. The earlier 3.66%/5.49% pair remains a one-seed control, while CIFAR-10 is a separate supplemental task.

### Controlled initialization and numerical investigation, September 16

The follow-up quality tools now support one neutral, hashed initialization
artifact shared by both engines. They explicitly convert parameter layouts,
retain a trial plan and every failed run, and compare initial predictions,
losses, gradients and updated parameters before full training. This removes
different random draws as an explanation for the first-step differences.
It does not make different floating-point implementations bitwise equivalent.

The controlled CPU FP32 checks identified two systematic discrepancies:

- The PyTorch LSTM benchmark used mean squared error, while OpenNN used half
  mean squared error. The Python driver now uses the same half-MSE objective;
  held-out RMSE is unchanged. Historical scores retain the earlier objective.
- Transformer temporary attention buffers could overlap a live encoder
  gradient during backward propagation. The allocator now checks scratch
  lifetimes in both forward and backward passes. On the fixed synthetic
  fixture, the largest first-step gradient difference fell from 0.0988344
  to 0.000000193715, matching a separate control with isolated scratch buffers.

The binary classification loss also gained a stable logits path for ordinary
single-output sigmoid heads. It preserves gradients that were lost through
saturated probabilities; inference still returns probabilities. These changes
require new training runs and do not retrospectively improve historical scores.

A twelve-case CPU FP32 audit compared both binary labels at logits
`-1000, -20, -8, 8, 20, 1000` against a high-precision analytic reference.
At `+20`, target `1`, corrected OpenNN retained the gradient
`-2.061153692e-9`; installed PyTorch `2.13.0+cu130` returned zero.
At `-20`, target `0`, OpenNN retained the loss `2.061153692e-9`, while
PyTorch returned zero. The retained values have about `3.59e-8` relative
error. Both underflow on correct predictions at magnitude 1000, below
FP32's representable range. This establishes a numerical accuracy advantage
for the named cases and implementation versions, not superior task accuracy.
All cases and reference calculations are retained in the external
`quality-parity-development-20260916/bce-accuracy/` evidence directory.

The CNN gradient discrepancy in one synthetic fixture was traced to rounding
around a ReLU boundary. A five-seed follow-up retained mixed outcomes: OpenNN
was closer to an FP64 reference on four seeds and PyTorch on one. OpenNN's mean
relative gradient error was about 20% lower, but the exploratory paired 95%
interval for the error difference included zero. This is a scoped numerical
observation, not established superiority or prediction-quality parity.

The final local CUDA diagnostic matrix completed all four models in FP32 and
BF16 using cuDNN 9.20.0 in both engines. Shared initial tensors matched exactly.
Parameter comparisons now read FP32 optimizer master values, rather than
comparing OpenNN's rounded BF16 compute view with PyTorch's FP32 master values.
The PyTorch LSTM BF16 driver explicitly executes the recurrent operation in
BF16: its ordinary autocast path had selected FP16. These corrections change
the meaning of the precision comparison and do not relabel earlier runs.

Across three fixed-batch updates, BF16 Dense and LSTM outputs matched exactly
on the small diagnostic fixture. CNN and Transformer BF16 differences remain:
the largest third-step master-parameter gaps were approximately 0.00560 and
0.000590 respectively. Matching initial tensors or a few updates does not
establish parity after full training. The compact matrix and per-tensor
measurements are in `cuda-matched-summary.json` in the external evidence
directory below.

The completed code batch passed the full repository verification wrappers:
CPU, 1,123 passed and 54 skipped; CUDA with the system cuDNN, 1,308 passed and
14 skipped; CUDA with the matched cuDNN 9.20.0, 1,310 passed and 12 skipped.
Each configuration also passed its response-scenario test. All 55 quality-tool
Python tests passed, including the CUDA BF16 LSTM regression. These checks
validate the implementation and measurement tools; they are not quality scores.

Two corrected real-data studies on the laptop RTX 3060 retained five
shared-initialization pairs each (seeds 201–205), batch 256, and matched
objectives. Dense used 250,000 HIGGS training rows and 500,000 held-out rows
for 100 epochs. LSTM used 35,035 PM2.5 training windows and 8,741 held-out
windows for 50 epochs. Both engines used the same prepared sample order,
including every partial final batch.

| Task | OpenNN mean | PyTorch mean | OpenNN advantage, paired 95% interval | OpenNN wins / ties / losses |
|---|---:|---:|---:|---:|
| Dense accuracy, % (higher is better) | 68.79888 | 68.67580 | +0.12308 points [-0.30107, +0.54723] | 3 / 0 / 2 |
| LSTM RMSE (lower is better) | 23.26794 | 22.81262 | -0.45532 [-0.82045, -0.09019] | 0 / 0 / 5 |

Positive advantage favors OpenNN. Dense has a small observed mean advantage
with mixed seeds and an interval spanning zero. LSTM has an observed deficit
on every seed. The literal requirement of no observed losses is not met.
No equivalence margin was declared, so these exploratory studies do not
establish formal parity or superiority. Their intervals describe seed
variation on the fixed held-out data, not uncertainty over new datasets.
They use the configured FP32 paths with TF32 disabled through NVIDIA's
override; matching initial predictions on eight examples does not by itself
establish the arithmetic policy at training batch size 256.
All 20 planned runs passed the independent retained-file integrity audit;
scores were recomputed from saved predictions. The audit includes every seed,
epoch history, initial tensor identity and recorded implementation hash in
`real-dense-lstm-five-seed-audit.json` and its Markdown companion. Fresh
full-training quality results under these controls cover Dense and LSTM only;
the CNN and Transformer numerical diagnostics do not replace quality reruns.

A separate real-profile LSTM runtime audit tested seed 201 at the actual
training batch size. Across three fixed-batch updates, the native default,
explicit default-math and forced-FMA variants produced identical numerical
results. Native and PyTorch first-step outputs differed by at most 1.19e-7,
and gradients by 7.45e-9. A subsequent one-epoch trace covered native restored
warm-up, 136 full training batches plus the 219-row tail, and 34 full evaluation
batches plus the 37-row tail. All 178 native and 173 PyTorch recurrent forwards
succeeded using FP32 persistent-small-H cuDNN execution. Native used default
math and PyTorch FMA; neither entered conversion-permitted math or fell back at
the tails. This does not support reduced-precision fallback as an explanation
for the observed training-quality deficit.

Forcing the native standard algorithm with FMA improved its first-step error
against an FP64 reference, but moved it farther from PyTorch's persistent
path. This is an algorithm-specific numerical diagnostic, not a trained-quality
advantage or a basis for changing the benchmark's reported outcomes. The
one-epoch trace is diagnostic and is excluded from the five-seed quality table.
The compact report, complete forward traces and hashed probe sources are
preserved under `lstm-arithmetic-audit/` in the external evidence directory.

### LSTM reference-machine equivalence audit, September 17

A prospective reference-machine run froze twenty seeds (301–320), exact shared
initial tensors, 50 epochs, batch 256, fixed row order, every tail sample, FP32
Adam and TF32 disabled before observing the new full-training scores. The
equivalence margin was ±0.902128 RMSE, one percent of the training-target
standard deviation. The separate achieved-quality gate required both mean
RMSE values to remain at or below 27.063849.

All forty launches completed. OpenNN averaged 23.111391 RMSE and PyTorch
23.111892. Defining positive advantage as PyTorch RMSE minus OpenNN RMSE, the
paired mean was +0.000501 and its 90% confidence interval was
[-0.275683, +0.276684]. Both bounds are strictly inside the frozen margin and
both engines passed the achieved-quality gate, so LSTM training-quality
equivalence is established for this protocol. The descriptive 95% interval,
[-0.333805, +0.334806], includes zero; neither engine established superiority.
OpenNN won nine seeds and lost eleven, which is consistent with equivalence of
the aggregate training-quality distribution rather than identical trajectories.

The numerical audit loaded bit-exact initial tensors into both engines. Its
first two losses were identical, and after three actual-size batch updates the
largest parameter difference was 7.82311e-8. The runtime audit found FP32
persistent cuDNN execution for full and tail batches without reduced-precision
fallback. OpenNN's setup warm-up restores the parameters, recurrent state,
Adam moment buffers and Adam step counter before counted training. These checks
did not identify an OpenNN LSTM precision defect requiring a source change.

Every saved prediction was hash-checked and independently rescored. The frozen
plan is `../opennn-benchmark-results/scratch/lstm-precision-parity-20260917-plan.json`;
the numerical audit and complete result are in the sibling directories
`lstm-precision-numerics-20260917/` and `lstm-precision-parity-20260917/`.
This local prospective result covers one dataset, machine and configuration;
it is not an externally registered study and does not imply bitwise identity or
equivalence on new datasets.

Compact audit evidence is preserved outside Git under
`../opennn-benchmark-results/scratch/quality-parity-development-20260916/`.
Full local WSL tensors, builds and logs are under
`/home/artelnics/opennn-benchmark-data/quality-parity-20260916/`.
The earlier reference-machine scores above remain a separate result set.

## Code and dependencies

The source audit at `5b4dac2cc` counted 64,700 OpenNN source lines after removing blanks and comments. Its breast-cancer application example counted 28 OpenNN lines and 32 PyTorch lines. The separate counts of 13 and 29 describe selected statement lines under language-specific rules; they must not replace those line counts.

The older PyTorch library figure of 834,319 lines has no matching raw source-count record in this selected evidence. The pinned audit below uses one counting tool and states both source scopes. A smaller source tree is not evidence of equal functionality or easier maintenance.

A matched source audit (`source-audit-matched-20260915.json`) used `cloc 1.98`
with C++, C/C++ headers, CUDA and Python included for both source trees.
OpenNN's unmodified core subtree at pulled revision `84bd178e8` contained
64,351 code lines in 254 files. The PyTorch `v2.13.0` tag at `cf30153c4`
contained 1,630,144 code lines in 6,416 files across `aten/src`, `c10`,
`caffe2`, `torch` and `functorch`, excluding named test, example, tool,
benchmark and third-party directories. Its Python files contributed 707,116
lines and its C++/header/CUDA files 923,028 lines. Both raw `cloc` JSON files
and their SHA-256 hashes are recorded beside the audit. These selected library
scopes have different features and language mixes; their count difference
does not measure equivalent functionality or developer effort. The historical
64,700 and 834,319 figures used different evidence and must remain separate.

Count Python packages and native libraries separately. OpenNN requires no Python packages, but it still uses native dependencies. CPU and GPU PyTorch installations have different package lists. Use the saved package inventory for the chosen deployment configuration; do not state that OpenNN has zero dependencies.

## Publication sequence

1. Choose one release commit, record all runtime versions, and prepare inputs with content hashes.
2. Confirm equal inputs, model behavior, precision, warm-up and measured work. Resolve the CNN, LSTM and Transformer differences listed above.
3. Locate and verify the completed CPU/GPU training and inference artifacts from the reference computer. Retain every launch, including failures.
4. Complete quality training, measure the reference-machine startup matrix, and test deployment bundles on a clean target.
5. Recalculate every table from raw evidence, review the checks, and publish only the sections that meet their stated requirements.

Keep unmeasured cells visible. A partial release must name its measured scope and must not claim that all models, devices or metrics improved. No overall improvement is reported while the intended comparison remains incomplete.

## Earlier footprint tests

Earlier footprint measurements cover baseline memory, process lifetime and
model export. The startup processes used different devices (OpenNN GPU,
PyTorch CPU), and their wall times included teardown. They cannot replace the
matching-device, first-prediction [startup comparison](#startup).

The single-process baseline readings are neither peak training memory nor
maximum model capacity. Keep RSS, anonymous memory and device memory separate.
The export test wrote files without executing both exported models: a new claim
needs a load-and-predict comparison and an explicit supported-architecture list.
Dense or recurrent source export does not establish CNN or Transformer export.
Keep those historical observations separate from current runtime results.

A pulled-upstream CPU FP32 Footprint pilot at `84bd178e8` ran one launch per
engine for memory, startup and export (`cpu-footprint-pulled-20260915-
20260915T104337Z.json`). The empty-object memory readings were 216.2 MiB
and 805.4 MiB total resident memory respectively; the monitor's separate
peak anonymous RSS readings were 119.0 MiB and 449.4 MiB. The startup launch took
0.157 s and 0.930 s respectively from process start to exit; the drivers'
internal first-prediction timers have different starting boundaries and should
not be compared as the same latency. The export launch wrote 15,104 bytes of
OpenNN C source and 14,972 bytes of OpenNN Python source, versus a 7,747-byte
PyTorch TorchScript file. The C source compiled and ran, the Python source
imported and predicted with NumPy and pandas installed, and TorchScript loaded
and predicted with PyTorch installed. The two OpenNN exports agreed to
1.4 × 10⁻⁸ on one common input. Their file sizes exclude those Python
dependencies and their formats and execution requirements differ. The old
OpenNN export driver also trained for 50 epochs while the PyTorch driver
serialized an untrained model; their export process times did unequal work.
Each observation has one launch only, and the saved machine-quiet gate failed
before the run. These are pilot costs rather than stable framework ratios or
an export-execution comparison.

A 15-round CPU FP32 repeat at the same pulled core revision
(`cpu-footprint-pulled-isolated-repeat-20260915-20260915T170815Z.json`)
returned successfully for all 90 launches and passed its machine-quiet gate:
0.13% performance-core activity before, at most 0.25% background activity
during and 0% after. VS Code and OneDrive threads were temporarily scheduled
on the efficiency cores and their original affinity was restored. Median
empty-object total RSS was 219.32 MiB for OpenNN and 809.45 MiB for PyTorch;
the separate monitor peak anonymous RSS medians were 115.2 and 449.8 MiB.
Median whole-process startup lifetimes, including teardown, were 0.05423 and
0.85161 seconds, with sample CVs of 1.96% and 3.18%. This artifact remains
in `scratch/` because the benchmark worktree contains reviewed protocol and
runner edits. Its export times retain the unequal-training defect and are
excluded from an export-cost comparison.

The corrected CPU FP32 Footprint run (`cpu-footprint-pulled-aligned-export-
20260915-20260915T172834Z.json`) serialized a fresh seeded, untrained
3 → 64 → 1 Tanh model in each engine. All 90 launches passed and the
performance-core quiet gate passed: 0.13% before and after, at most 0.31%
during. Median baseline total RSS was 219.305 MiB for OpenNN and 809.531 MiB
for PyTorch; separate peak anonymous RSS medians were 115.4 and 449.8 MiB.
Median whole-process startup lifetimes were 0.05474 and 0.85262 s, with
CVs of 1.57% and 2.32%. These lifetimes include teardown, so they are not
first-response latencies. Each engine's internal first-prediction timer has
a different starting boundary.

OpenNN's aligned export launch produced 12,404 bytes of C source and 12,209
bytes of Python source; PyTorch produced a 7,747-byte TorchScript file. The
OpenNN C and Python outputs compiled/imported and agreed on a nonzero input
to within 1.2 × 10⁻⁸; TorchScript loaded and predicted. The formats and
execution dependencies differ: generated Python imports NumPy and pandas.
The full-run median process lifetimes were 0.05612 and 0.85866 s, but
OpenNN's 3.55% CV exceeded the 3% variation rule. An export-only 15-round
repeat (`cpu-footprint-pulled-export-only-20260915-20260915T174307Z.json`)
passed all 30 launches and the quiet gate, with 0.19% performance-core
activity before/after and no measured background activity during. Its median
whole-process export lifetimes were 0.05652 s for OpenNN and 0.85783 s for
PyTorch, with CVs of 2.13% and 0.79%. These include runtime startup, model
construction and serialization. Because the artifact formats differ, they
are separate deployment costs rather than a like-for-like serialization-speed
ratio. Both artifacts remain diagnostic in `scratch/` while the benchmark
worktree contains uncommitted review changes.

The named Footprint model exported valid predictions. A separate core
`ModelExpression` defect affected unnamed networks: its body invented default
output names, but the C and Python emitters still counted zero named outputs.
The working-tree fix fills missing input and output names before emission.
An unnamed-network regression now executes both generated C and Python files
and checks their predictions against OpenNN; the expression execution target
passed all eleven available C/Python checks. This core fix was made after the
Footprint and source-count artifacts above, so their pinned binaries and
revision remain the measurement evidence.

## Website outline

The proposed page remains unpublished. Use the sections of this report for
training and inference, memory, energy, startup, deployment, prediction quality,
and code/dependencies. Each section should have a short explanation, a table
with units and ratios, the tested configuration, and downloadable raw evidence.
Show missing configurations explicitly. Publication wording and ratio rules
are in [the protocol](../PROTOCOL.md#15-publication-procedure).

The September 11 website review used an indexed copy; a direct request returned
a browser-verification page. It was not a fresh authenticated WordPress snapshot.
The older RTX 4080 articles, RTX 5070 Ti records and laptop studies remain separate
result sets. Before updating the index and articles:

- Match the named task: keep Iris distinct from breast cancer, CIFAR-sized CNNs
  from ImageNet, and Rosenbrock regression from the four new quality tasks.
- Keep framework baseline memory distinct from training peaks and dataset
  capacity. The earlier pandas loading workflow does not establish a universal
  PyTorch dataset limit; retain its memory cap and loading method.
- Distinguish CPU Eigen from MKL + oneDNN deployment; verify Windows separately
  and compare FP32/BF16 on the same model with the actual precision recorded.
- Resolve the model and timing differences above, keep dense runtime settings,
  thread counts and affinity visible, and execute exports before claiming parity.
- Add LSTM, CPU CNN/Transformer, energy and real-data quality results as their
  evidence becomes available. A PyTorch comparison does not remeasure TensorFlow
  or Qwen runtime results.

Explain practical implications within the measured scope: shorter iterations
can permit more experiments, but do not establish time to target quality;
component energy does not measure an electricity bill. Source size alone does
not establish equivalent features or ease of use. Replace the relevant articles
and index cards together once the evidence is approved.

## Evidence and preservation

The catalog covers 2,868 files and 2,634 unique contents. Files with identical SHA-256 hashes share one catalog entry with all their paths. No measurement file was edited or deleted. Original reports are preserved in [Git history](https://github.com/Artelnics/opennn/tree/a379ec5e634d65436b8b175fcd03c044bc98182b/benchmarks/reports/archive/2026-09-11/).

The accompanying `performance.json`, `observations.csv`, `startup.json`, `deployment.json`, `readiness.csv` and `catalog.json` retain source paths, hashes, raw-derived statistics and pending checks. [`selection.json`](selection.json) pins the evidence; selecting a file does not approve it for publication.


The [former family reports](https://github.com/Artelnics/opennn/tree/a379ec5e634d65436b8b175fcd03c044bc98182b/benchmarks/reports/),
[website audit](https://github.com/Artelnics/opennn/blob/a379ec5e634d65436b8b175fcd03c044bc98182b/benchmarks/publication/website-audit.md) and
[editorial draft](https://github.com/Artelnics/opennn/blob/a379ec5e634d65436b8b175fcd03c044bc98182b/benchmarks/publication/website-draft.md) are retained
at the same immutable commit, `a379ec5e6`. The detailed historical analyses keep
their original dates and configurations; their explanations are not evidence
about later runs. Inspect them locally without creating another checkout:

```sh
git show a379ec5e6:benchmarks/reports/archive/2026-09-11/dense.md
```

Generated review files and every raw observation remain under the external
`../opennn-benchmark-results/` tree. This documentation consolidation changes no raw
measurement, source hash, selection record or publication decision.
