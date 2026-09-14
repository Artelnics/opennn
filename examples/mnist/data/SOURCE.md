# MNIST test images

Copyright Yann LeCun and Corinna Cortes. MNIST derives from the original NIST
datasets. The [Keras MNIST publisher](https://keras.io/api/datasets/mnist/)
records the data licence as
[Creative Commons Attribution-ShareAlike 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

These BMP images are distributed under the same data licence. Preserve this
attribution, source and licence link when sharing them, and apply the same
licence to adaptations as required. OpenNN's software licence does not replace
the dataset licence.

OpenNN adaptation: the 10,000 test images are converted to 28x28 grayscale BMP
and grouped by digit. Within each digit folder, `{digit}_{ordinal}.bmp` uses
the zero-based occurrence of that digit in the original test split. All pixels,
labels and filename mappings were verified against the Keras distribution on
2026-09-08; no training-set images or unmatched images were found.

[Source archive](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
SHA-256: `731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1`.
