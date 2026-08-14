# ECG5000 anomaly detection

This example follows the ECG autoencoder workflow from TensorFlow's official
[autoencoder tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder):

- the 140 signal values are normalized with one global minimum and maximum
  calculated from the training partition;
- the deterministic 80/20 partition is the one produced by
  `train_test_split(test_size=0.2, random_state=21)`;
- only normal samples in the training partition fit the autoencoder;
- the model is `140-32-16-8-16-32-140`, with ReLU hidden activations and a
  sigmoid output;
- Adam trains for 20 epochs with batches of 512 and mean absolute error;
- the anomaly threshold is the normal-training reconstruction-error mean plus
  one population standard deviation.

The tutorial data is stored in `data/ecg.csv` from the tutorial's
[official download](https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv).
Although the dataset is commonly called ECG5000, this prepared CSV contains
4,998 rows. Its SHA-256 checksum is:

```text
72ce7b040ca0c6ed36c3368e570c6ac4ddf20100476e47373c63b2395e012df1
```

`data/test_indices.csv` stores the 1,000 fixed test indices. This keeps the
example reproducible without making Python or scikit-learn runtime dependencies.

The executable writes the trained network, external normalization/threshold
metadata, reconstruction-error data, and one normal and anomalous reconstruction
CSV to its working directory.
