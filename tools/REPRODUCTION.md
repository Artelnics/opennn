# Reproducing example data and reference models

These tools produce files outside the checkout and preserve the bundled assets.
Use Python 3.12 or 3.13 in a virtual environment:

```sh
python -m pip install -r tools/reproduction-requirements.txt
python tools/reproduce_datasets.py --cache ../dataset-downloads --output ../reproduced-data
```

Downloads are pinned by SHA-256. A changed or corrupted archive is rejected,
including an already-cached archive. The tool does not extract archive paths.
It reads reviewed Git index blobs, so stage reviewed asset changes first.
`--datasets iris concrete` selects a subset; the default checks all seven recipes.
The output `reproduction.json` records source URLs/checksums and generated hashes.

| Recipe | Reconstructed and compared against the index |
| --- | --- |
| Airfoil | All 1,503 rows, with the existing headers and semicolon delimiter |
| Breast cancer | All 683 complete rows; ID removal, binary labels, and the eleven synthetic missing cells |
| Iris | All 150 corrected `bezdekIris.data` rows and normalized category labels |
| Concrete | All 1,030 XLS rows and nine columns, in existing feature order |
| Amazon | The original 1,000 labelled sentences, ten-row reduced sample and five-row small sample, including their first-sentence edits |
| MNIST | All 10,000 test images, every pixel, digit folder and per-digit ordinal filename |
| ECG | The exact TensorFlow CSV and the seed-21 NumPy test permutation |

Numeric CSVs are compared cell by cell with absolute tolerance 1e-12 and no
relative tolerance. Generated CSV number formatting may differ from the original;
headers, categorical cells, missing values, row order and shapes must agree.
MNIST uses decoded grayscale pixels for equivalence, not BMP encoder metadata.
Attribution notices present in each bundle are copied alongside generated data.

This does not reproduce the Amazon numeric/tokenized derivatives, SST-2, emotion,
melanoma, translation, Madrid, or wastewater assets. Reproducing a transformation
does not grant redistribution permission. The unresolved entries in
[DATASETS.md](../DATASETS.md) and the `--release` inventory gate still apply.

## Training new reference models

Build the separate consumer against an installed OpenNN 9.0 package:

```sh
cmake -S tools/reproduce_models -B ../model-generator -DCMAKE_PREFIX_PATH=/absolute/path/to/opennn-prefix
cmake --build ../model-generator --config Release
python tools/reproduce_models/verify.py --executable ../model-generator/opennn_reproduce_models --data-root ../reproduced-data --output ../reference-models
```

For Visual Studio, the executable is under `Release/` and has the `.exe` suffix.
If the OpenNN installation uses an externally installed Eigen package, provide
its prefix or `Eigen3_DIR`, as for any other installed-package consumer.

The generator trains Iris (4-16-3, Tanh/Softmax) and concrete (8-32-16-1,
Tanh/Identity) reference networks. It uses CPU FP32, Eigen, one thread, seed
1729, Adam at 0.01, batch size 128, no shuffle, 1,000 epochs, and restoration
of the best validation epoch. Each row's zero-based index modulo ten assigns
0-5 to training, 6-7 to validation and 8-9 to testing. This keeps Iris's ordered
classes represented in every split. Full optimizer settings and split rows are
saved, together with JSON/binary model pairs and held-out prediction references.

Verification requires at least 85% held-out Iris accuracy and 0.70 concrete R²,
identical model bytes across two runs in the same environment, successful native
save/reload, and Python export agreement (`rtol=1e-4`, `atol=1e-5`). The report
records the source archive, training CSV, generator executable/source, model
hashes, platform and measured results. Identical bytes across different compilers
or standard libraries are not promised; compare predictions and recorded settings.

These are newly trained reference models, not reconstructed copies of the
historical bundled models or substitutes for their missing provenance. They are
not production-model migration evidence. Generated models remain outside Git.
