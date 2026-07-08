# Image Classification Max-Batch Probes

Purpose: low-level image-classification capacity probes for OpenNN, PyTorch, and
TensorFlow. For the full ResNet-50 capacity benchmark, use
[`../resnet50-max-batch/`](../resnet50-max-batch/).

Build the OpenNN trial and run the per-framework probe scripts in this folder:

```bash
cmake --build build-benchmarks --target opennn_image_classification_maxbatch_trial
python pytorch_image_classification_maxbatch.py
python tensorflow_image_classification_maxbatch.py
```
