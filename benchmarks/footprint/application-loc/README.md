# Application LOC Benchmark

Purpose: compare the logical source lines of code needed for the same Iris
classification workflow across OpenNN, PyTorch, and TensorFlow/Keras.

Run:

```bash
python count_lsloc.py
```

Counts the logical lines of `opennn_iris.cpp`, `pytorch_iris.py`, and
`tensorflow_iris.py` (kept aligned to the same workflow).
