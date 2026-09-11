# OpenNN and PyTorch: publication review

Reviewed on 11 September 2026. **These observations are not ready for the next website release.**

We compare OpenNN C++ applications with the PyTorch Python API. The tables below preserve the measured values, explain their limits and identify the work that remains. They do not claim that every comparison is valid or that OpenNN always wins.

## What is available

| Comparison | Evidence | Next step |
| --- | --- | --- |
| Throughput, memory and energy | 12 historical configurations | Repeat with complete input records and resolve the workload differences below. |
| CPU CNN and Transformer | No selected training or inference results | Measure the four missing configurations. |
| Startup | 660 timed launches; 0 failed | Repeat under stable machine conditions. |
| Deployment size | 12 application configurations; file totals verified | Check the bundles on a clean target and record the exact build and packages. |
| Prediction quality | Four model paths pass synthetic smoke checks | Train on real data and evaluate held-out predictions. |
| Code size and dependencies | Dated source counts and installation inventories | Pin both source revisions and use the same counting rule. |

## Computers and software

The performance records come from the reference computer: Intel Core i7-14700F, 32 GB RAM, NVIDIA RTX 5070 Ti, Linux, PyTorch 2.13.0+cu130 and NVIDIA driver 610.43.02. The CPU runs use FP32; GPU runs are labelled BF16, with an LSTM precision discrepancy described below.

Eight performance records belong to session `2026-09-06-publish` at `e76425bd3`. The four GPU CNN and Transformer records belong to `2026-09-06-energy-publish` at `520f0f2f8`. Each ratio uses its own paired session. These records do not describe the current checkout.

Startup and deployment come from a different computer: Intel Core i7-12700H and RTX 3060 Laptop GPU under WSL2. OpenNN uses CUDA 12.9; the PyTorch wheel uses CUDA 13.0, with separate cuDNN 9.20 distributions. These observations stay separate from reference-computer results.

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

CPU CNN training, CPU CNN inference, CPU Transformer training and CPU Transformer inference are **not measured in this selected result set**. They must remain visible as gaps.

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

## Why the performance records need another run

All twelve records lack content hashes for their prepared inputs. A filename, size and modification date do not prove that the engines received the same data. Ten records also have missing numerical quality values despite an old passing flag. That flag does not prove prediction quality or output agreement.

The recomputed checks also flag PyTorch memory variation in CPU dense training. Full per-engine ranges, variation and check results are in the accompanying observations and readiness files.

### Dense

Both engines use a 28?1024?1024?1 classifier on HIGGS. The historical training paths use different sample-order policies; align and record those policies for the new comparison. GPU dense training exceeds the throughput variation limit on the PyTorch side. Its energy and throughput windows also differ in duration, so both boundaries need checking. Record library versions and the selected kernel configuration before the new session.

### LSTM

Both engines use 128 LSTM units and 24-step windows of 15 weather features. The old GPU records are labelled BF16, but the archived analysis identifies FP16 recurrent kernels on the PyTorch side. Verify the actual precision and compare the same prediction task. The old PyTorch inference driver kept a full dataset on the GPU while replaying only one batch. Training also differs in scaling layers and data residency. Remove unused data from the inference comparison and decide whether each timing includes input transfers. CPU LSTM inference exceeds the throughput variation limit on the PyTorch side. Repeat it after stabilizing the machine.

### CNN

The historical model is ResNet-50 v1.5 with 25,557,032 parameters, using 224 ? 224 images from a 50,000-image ImageNet subset. It is a different workload from the older CIFAR-sized website article. The old training runs use different image transforms, decoding paths, transfer sizes and warm-up counts. Feed both engines the same prepared pixels and match the timing boundary and warm-up. Separate resident-model inference from application input handling.

### Transformer

The historical model has six encoder and six decoder layers, width 512, eight attention heads and a 20,000-token vocabulary on WMT14 text. The old drivers differ in causal and padding masks, dropout, and the loss denominator. Align these before repeating training or inference. Translation quality must use generated text on held-out data; equal tensor shapes are not sufficient.

The current code includes changes made after these runs. Fixes in the code cannot repair an old measurement; the affected comparison must run again. Different vendor libraries can also affect results, so conclusions must describe the tested configurations rather than assign every difference to framework code.

## Startup

Every sample starts a fresh process and stops when its first prediction reaches host memory. Each configuration has 15 timed launches. **43 of 44 engine configurations exceed the 3% variation limit.** All paired startup comparisons therefore remain diagnostic.

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

## Prediction quality

| Model | Task | Metric | OpenNN | PyTorch | Status |
| --- | --- | --- | --- | --- | --- |
| Dense | HIGGS classification | Test accuracy (%) | Not measured | Not measured | Train and evaluate |
| LSTM | PM2.5 forecasting | Test RMSE in original units | Not measured | Not measured | Train and evaluate |
| CNN | Image classification | Test top-1 accuracy (%) | Not measured | Not measured | Train and evaluate |
| Transformer | English–German translation | Generated-translation BLEU | Not measured | Not measured | Train and evaluate |

Use five independent seeds and report the mean and standard deviation for each model. Set an acceptable quality difference before inspecting the final scores. Accuracy and BLEU are higher-is-better; RMSE is lower-is-better. Do not average them into one quality score. Synthetic smoke results and the older Rosenbrock task do not fill these rows.

## Code and dependencies

The source audit at `5b4dac2cc` counted 64,700 OpenNN source lines after removing blanks and comments. Its application example counted 28 OpenNN lines and 32 PyTorch lines. The separate counts of 13 and 29 describe selected statement lines under language-specific rules; they must not replace those line counts.

The older PyTorch library figure of 834,319 lines has no matching raw source-count record in this selected evidence. Recount both repositories at pinned revisions with the same inclusion rules before publishing a codebase ratio. A smaller source tree is not evidence of equal functionality or easier maintenance.

Count Python packages and native libraries separately. OpenNN requires no Python packages, but it still uses native dependencies. CPU and GPU PyTorch installations have different package lists. Use the saved package inventory for the chosen deployment configuration; do not state that OpenNN has zero dependencies.

## Publication sequence

1. Choose one release commit, record all runtime versions, and prepare inputs with content hashes.
2. Confirm equal inputs, model behavior, precision, warm-up and measured work. Resolve the CNN, LSTM and Transformer differences listed above.
3. Run the complete CPU/GPU training and inference matrix on the reference computer. Retain every launch, including failures.
4. Complete quality training, repeat stable startup measurements, and test deployment bundles on a clean target.
5. Recalculate every table from raw evidence, review the checks, and publish only the sections that meet their stated requirements.

Keep unmeasured cells visible. A partial release must name its measured scope and must not claim that all models, devices or metrics improved. No overall improvement is reported while the intended comparison remains incomplete.

## Evidence and preservation

The catalog covers 2,868 files and 2,634 unique contents. Files with identical SHA-256 hashes share one catalog entry with all their paths. No measurement file was edited or deleted. Old reports are preserved under `reports/archive/2026-09-11/`.

The accompanying `performance.json`, `observations.csv`, `startup.json`, `deployment.json`, `readiness.csv` and `catalog.json` retain source paths, hashes, raw-derived statistics and pending checks. `publication/selection.json` pins the evidence; selecting a file does not approve it for publication.
