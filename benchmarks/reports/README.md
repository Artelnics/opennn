# OpenNN benchmark results

The benchmark evidence has been consolidated and reviewed for the next website
release. **The current result set needs further work before publication.**

Read the [publication review](publication-review.md) for all selected values,
measurement definitions and outstanding checks. It recalculates the tables from
raw records. Read the [publication guide](../publication/README.md) for the
release requirements and the [website draft](../publication/website-draft.md)
for the proposed page.

| Report | What it covers | Current status |
|---|---|---|
| [Dense](dense.md) | HIGGS training and inference on CPU/GPU | Repeat with complete provenance and stability checks |
| [LSTM](lstm.md) | Forecasting training and inference on CPU/GPU | Resolve precision, memory scope and workload differences |
| [CNN](cnn.md) | ResNet-50 training and inference | Match the input pipeline; add CPU results |
| [Transformer](transformer.md) | Translation-model training and inference | Match masks, dropout and loss; add CPU results |
| [Startup](startup.md) | First completed prediction in small applications | Repeat under stable machine conditions |
| [Deployment](deployment.md) | Application files, source counts and dependencies | Verify target bundles and recount source consistently |
| [Prediction quality](quality.md) | Four tasks on held-out data | Full training still pending |
| [Earlier footprint tests](footprint.md) | Baseline memory, process lifetime and export | Historical evidence; different measurement scopes |

The primary comparison uses OpenNN C++ and the PyTorch Python API. Older
LibTorch and TensorFlow observations stay in the archive with their original
scope. Results from the reference desktop, the laptop and older website tests
remain separate. Missing results are not filled with smoke-test scores.

The previous reports are preserved in [the September 11 archive](archive/2026-09-11/).
Their passing labels and broad claims are historical; use the current review to
judge publication readiness. Raw results remain ignored under `../results/`.
The relocation manifest and content catalog retain the original paths and hashes.
No observation was removed to improve an average or a comparison.
