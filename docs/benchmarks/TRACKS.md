# Benchmark Tracks

This file is the human-readable consolidation layer for the benchmark suite.
The machine-readable source is [`benchmark_manifest.json`](benchmark_manifest.json);
this page groups the same evidence by decision status.

## Headline Candidates

These tracks can support presentation or website claims after their raw commands,
versions, and result artifacts are archived.

| Track | Evidence | Current gap |
|---|---|---|
| CPU runtime size | [`size-cpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](size-cpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Archive measurement commands and versions. |
| GPU deployment size | [`size-gpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](size-gpu-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Archive CUDA/cuDNN library list. |
| Startup latency | [`startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](startup-latency-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Store result JSON and cold-cache protocol. |
| Peak training memory | [`peak-memory-opennn-vs-pytorch-vs-tensorflow.md`](peak-memory-opennn-vs-pytorch-vs-tensorflow.md) | Store result JSON and process measurement method. |
| Data capacity | [`data-capacity-opennn-vs-pytorch-vs-tensorflow.md`](data-capacity-opennn-vs-pytorch-vs-tensorflow.md) | Store result JSON and TensorFlow capacity measurement. |
| Install dependencies | [`dependencies-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](dependencies-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Archive package commands and exact versions. |
| Standalone export | [`code-export-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](code-export-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Archive generated source proof. |
| Transformer inference | [`transformer-inference-gpu-opennn-vs-pytorch.md`](transformer-inference-gpu-opennn-vs-pytorch.md) | Rerun with complete provenance on the reference machine. |

## Supporting Evidence

These tracks are useful context, but should not be the primary performance
headline.

| Track | Evidence | Notes |
|---|---|---|
| Rosenbrock accuracy | [`accuracy-opennn-vs-pytorch-vs-tensorflow.md`](accuracy-opennn-vs-pytorch-vs-tensorflow.md) | Quality sanity check on a small synthetic task. |
| Rosenbrock precision | [`precision-opennn-vs-pytorch-vs-tensorflow.md`](precision-opennn-vs-pytorch-vs-tensorflow.md) | Optimizer/workflow comparison, not a deep-learning scale claim. |
| Native source LOC | [`loc-opennn-vs-pytorch-vs-tensorflow.md`](loc-opennn-vs-pytorch-vs-tensorflow.md) | Context metric with counting caveats. |
| Application LOC | [`application-lines-of-code-opennn-vs-pytorch-vs-tensorflow.md`](application-lines-of-code-opennn-vs-pytorch-vs-tensorflow.md) | API ergonomics metric; style-sensitive. |
| Native Windows GPU capability | [`gpu-on-windows-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](gpu-on-windows-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Capability matrix, not throughput. |

## Internal Tracks

These tracks are useful for engineering and migration, but not public claims.

| Track | Evidence | Next action |
|---|---|---|
| CPU HIGGS dense training | [`higgs-cpu-training-opennn-vs-pytorch-vs-tensorflow.md`](higgs-cpu-training-opennn-vs-pytorch-vs-tensorflow.md) | Rerun full HIGGS split with repeated samples and committed result JSON. |
| CPU HIGGS dense inference | [`higgs-cpu-inference-opennn-vs-pytorch-vs-tensorflow.md`](higgs-cpu-inference-opennn-vs-pytorch-vs-tensorflow.md) | Rerun full HIGGS test split and measure CPU energy outside WSL. |
| HIGGS dense max batch, GPU + capped CPU (train + infer) | [`higgs-max-batch/README.md`](higgs-max-batch/README.md) | Harness ready; run on the reference GPU (fp32/bf16) and the CPU RLIMIT_DATA-capped matrix, archive the result JSONs. |
| GPU Transformer max batch (train + infer) | [`transformer-max-batch/README.md`](transformer-max-batch/README.md) | Alpaca train mode measured (RTX 4080); run the new infer mode and the WMT14 corpus, archive result JSON. |
| Recurrent/LSTM forecasting | [`recurrent-lstm-forecasting-opennn.md`](recurrent-lstm-forecasting-opennn.md) | Run on the reference Linux GPU and archive raw output. |

## Hold Back

These tracks should stay out of public headline slides until the listed issue is
resolved.

| Track | Evidence | Blocker |
|---|---|---|
| GPU CNN training | [`cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md`](cnn-training-speed-gpu-opennn-vs-pytorch-vs-tensorflow.md) | Competitor paths need optimized or explicitly scoped comparison. |
| GPU ResNet-50 training | [`resnet50-training-speed-gpu-opennn-vs-pytorch.md`](resnet50-training-speed-gpu-opennn-vs-pytorch.md) | Needs repeated result JSON and reference-machine rerun. |
| GPU ResNet-50 max batch | [`resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md`](resnet50-max-batch-gpu-opennn-vs-pytorch-vs-tensorflow.md) | Current result is memory-regression evidence, not an OpenNN headline win. |
| CPU inference speed | [`inference-speed-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md`](inference-speed-opennn-vs-onnxruntime-vs-pytorch-vs-tensorflow.md) | Needs reference Linux rerun and checksum gate. |
| Transformer training | [`transformer-training-gpu-opennn-vs-pytorch.md`](transformer-training-gpu-opennn-vs-pytorch.md) | Needs repeated-run JSON and quality gate. |

## Historical

These tracks are retained for engineering history and should be replaced by
HIGGS or newer model-specific benchmarks for new claims.

| Track | Evidence | Replacement path |
|---|---|---|
| GPU dense Rosenbrock speed/capacity | [`rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md`](rosenbrock-maxbatch-and-speed-gpu-opennn-vs-pytorch.md) | Replace with HIGGS dense CPU/GPU evidence. |
| GPU dense Rosenbrock energy | [`energy-consumption-gpu-opennn-vs-pytorch.md`](energy-consumption-gpu-opennn-vs-pytorch.md) | Replace with HIGGS/Transformer energy evidence. |
