# Prediction quality

**Status: real-data training and held-out evaluation are pending.**

The new code trains both libraries on shared prepared data and scores their
predictions with one scorer. All four model paths pass CPU and GPU smoke tests.
Those tests use synthetic data and reduced models; their scores are not
benchmark evidence.

| Model | Task | Metric | OpenNN | PyTorch |
|---|---|---|---|---|
| Dense | HIGGS classification | Test accuracy (%) | Not measured | Not measured |
| LSTM | PM2.5 forecasting | Test RMSE in original units | Not measured | Not measured |
| CNN | Image classification | Test top-1 accuracy (%) | Not measured | Not measured |
| Transformer | English-German translation | Generated-translation BLEU | Not measured | Not measured |

Run five independent seeds. Choose the training budget and acceptable quality
difference before inspecting the final scores. Report the mean and sample
standard deviation for each task. Accuracy and BLEU are higher-is-better;
RMSE is lower-is-better. There is no combined quality percentage.

The older Rosenbrock regression test is a different task. It cannot fill these
rows. Successful execution, equal parameter counts and similar training losses
do not establish prediction-quality parity.

See [QUALITY.md](../QUALITY.md) for the prepared splits, models, optimizer
settings, decoding method and commands. These quality runs also do not repair
the different workloads in the historical throughput records.
