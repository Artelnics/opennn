# CNN Training Speed Benchmark

Purpose: GPU training-speed comparison for a small CNN (one convolutional + one
pooling layer) on MNIST, batch 128, fp32 — OpenNN (GPU-resident data) vs PyTorch
vs TensorFlow.

Prepare MNIST with `prepare_mnist.py`, then run:

```bash
python run_cnn.py
```

Reports samples/s, epoch time, and final loss per framework.
