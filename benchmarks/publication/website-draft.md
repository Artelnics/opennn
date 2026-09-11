# OpenNN benchmarks

Editorial draft for the next release. Add numerical claims only after the
corresponding measurements pass review. This page has not been published.

We compare OpenNN and PyTorch on specific models and computers. Each test shows
the results, the settings and the data needed to repeat it.

## Training and inference

We measure how much work each library completes per second.

Show training and inference in separate tables, with CPU and GPU columns. Include
Dense, LSTM, CNN and Transformer. Label the model-specific units and display
"Not measured" for missing configurations. Link each table to the hardware,
batch sizes, precision, runtime settings and downloadable launches.

Shorter training iterations can let companies test more model configurations.
Higher inference throughput can help an application handle more requests.

## Memory

We measure how much memory each application uses during training and inference.

Show CPU and GPU tables separately. Define the memory measurement in the caption
and include both absolute values and the OpenNN-to-PyTorch ratio.

Lower memory use can leave more room for other applications on the same device.

## Energy

We measure the energy each processor uses to complete the same work.

Show CPU package energy and GPU board energy separately. Give the amount of work
and the energy in Wh. Leave unavailable readings marked "Not measured".

Lower energy per workload can reduce the energy needed for repeated training
and inference jobs.

## Startup

We measure the time from starting an application to its first completed prediction.

Show the four small applications, their CPU/GPU backends and their cache
settings. Give milliseconds and ratios. Describe whether the test constructs
a model or loads a saved model.

Shorter startup times can reduce the wait before an application is ready.

## Deployment size

We count the files each application needs on its target computer.

Show each application and backend with its deployment size in MB. Explain that
the PyTorch column includes a standard Python installation. State the exclusions
and provide the file and package inventories.

Smaller bundles can reduce download sizes and storage requirements.

## Prediction quality

We train both implementations and evaluate their predictions on held-out data.

| Model | Metric | OpenNN | PyTorch |
|---|---|---|---|
| Dense | Test accuracy (%) | Awaiting training | Awaiting training |
| LSTM | Test RMSE | Awaiting training | Awaiting training |
| CNN | Test top-1 accuracy (%) | Awaiting training | Awaiting training |
| Transformer | Generated-translation BLEU | Awaiting training | Awaiting training |

Use the measured scores and declared tolerance to describe agreement. Do not
prewrite a claim that the libraries produce the same quality.

## Code and dependencies

We count the source lines and runtime dependencies in the tested configurations.

Show library source and application source as separate rows. Count both
repositories with one rule. Keep Python packages separate from native libraries.
Show feature differences alongside these counts.

These measurements help readers judge the size and integration requirements of
the tested applications.

## How to read the results

These results describe the models, software and computers listed in each test.
They do not show that one library is faster for every application. We keep the
raw measurements and reproduction instructions beside each comparison so that
other people can check the results.

Keep older tests accessible with their original dates and methods. Do not merge
their numbers into the new tables. After this release, extend the comparison to
additional models, hardware and frameworks using the same measurement rules.
