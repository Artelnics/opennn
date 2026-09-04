# ECG5000 anomaly detection

This example trains an autoencoder to recognize normal ECG signals and flags
signals with unusually large reconstruction errors:

- OpenNN creates a reproducible 80/20 training and testing split;
- only normal samples are used to train the autoencoder;
- input and output scaling are prepared automatically from the training data;
- the anomaly threshold is the mean training reconstruction error plus one
  population standard deviation.

The tutorial data is stored in `data/ecg.csv` from the tutorial's
[official download](https://storage.googleapis.com/download.tensorflow.org/data/ecg.csv).
Although the dataset is commonly called ECG5000, this prepared CSV contains
4,998 rows. Its SHA-256 checksum is:

```text
72ce7b040ca0c6ed36c3368e570c6ac4ddf20100476e47373c63b2395e012df1
```

The executable prints the learned threshold and the number of anomalies found
in the testing partition.
