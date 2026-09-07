# Example data: provenance and redistribution review

Reviewed 2026-09-08. OpenNN's software licence does **not** automatically
license third-party datasets, images, text, or trained artifacts. This review
preserves the existing files and distinguishes identified sources from
unresolved permission and transformation records. It does not clear every
asset for a public release.

`datasets.manifest.json` records portable SHA-256 content inventories for all
Git-tracked example `data/` and `nn/` files. Run:

```sh
python tools/check_dataset_manifest.py
python tools/check_dataset_manifest.py --release
```

The first checks Git index content against this review. Stage reviewed asset changes before running it; unstaged changes are not part of this check. The second intentionally fails
while any group lacks redistribution clearance. Updating a hash alone does
not constitute a provenance review. Text hashes normalize CRLF to LF.

## Identified UCI datasets

UCI publishes the following datasets under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Preserve the creator
credit, source link, licence link, and description of changes when sharing
these adaptations.

| Local bundle | Source and attribution | Local transformation and evidence |
| --- | --- | --- |
| `airfoil_self_noise/data` | Brooks, T., Pope, D., Marcolini, M. (1989), [Airfoil Self-Noise, UCI](https://archive.ics.uci.edu/dataset/291/airfoil+self+noise), DOI 10.24432/C5VW2C | All 1,503 rows match the upstream `.dat` numerically in order. Header added; whitespace changed to semicolons. Cleared with this attribution. |
| `breast_cancer/data` | Wolberg, W. (1990), [Breast Cancer Wisconsin (Original), UCI](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), DOI 10.24432/C5HP4Z | Clean CSV matches all 683 complete upstream rows: remove ID, omit rows containing `?`, map 2/4 to 0/1, add header and semicolons. The separate missing-values variant is a local adaptation requiring its modification record. |
| `concrete/data` | Yeh, I. (1998), [Concrete Compressive Strength, UCI](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), DOI 10.24432/C5PK67 | All 1,030 rows and nine columns match `Concrete_Data.xls` within 8e-15 absolute error. XLS converted to CSV; headers renamed. See existing `data/SOURCE.md`. Bundled `nn/` model needs a training/provenance record. |
| `iris_plant/data` | Fisher, R. (1936), [Iris, UCI](https://archive.ics.uci.edu/dataset/53/iris), DOI 10.24432/C56C76 | All 150 rows match `bezdekIris.data` numerically; labels lowercased with underscores, header and semicolons added. This includes corrected rows 35 and 38. `iris_model.json` needs a model-generation record. |
| `amazon_reviews/data` | Kotzias, D. (2015), [Sentiment Labelled Sentences, UCI](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), DOI 10.24432/C57604; Kotzias et al., *From Group to Individual Labels using Deep Features*, KDD 2015 | The 1,000 labelled sentences match after 0/1 to Bad/Good mapping, except an added leading apostrophe on the first local sentence. Small/reduced samples and tokenized files are derivatives; reconstruction settings are missing. `amazon_cells_reduced_data.txt` contains NUL bytes and malformed tabular fields and is not a validated replacement for the original text. |

Upstream ZIP SHA-256 values from this review:

```text
15   3f91e49bceb30c0de8ea988344357236f26ed8bd536415ac594226015b6fd84d
53   d11fe30213d36434a0879aab7cb00ce3c812eb7ba2495874438abff7b7b762e9
165  dad85d14de8aee4e07479daa774e6b569a313715b71a3b92c95a07cf91c2c9a7
291  5c7767ba53ad827d3f48ba1eb9434117f4892df8f10bc4c99e118a9e8a7ae07c
331  afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343
```

Download links are on the cited UCI pages. Compare the ZIP checksum before
repeating the transformations above; a changed upstream file needs review.
These records describe verified equivalence, not a byte-for-byte regeneration
script for every derived file. No existing dataset is removed by this review.

## Assets still requiring clearance

| Bundle | Evidence | Required resolution |
| --- | --- | --- |
| `bert/data/sst2.txt` | Content is consistent with SST-2. The [Stanford Sentiment Treebank publisher](https://nlp.stanford.edu/sentiment/) identifies Socher et al., EMNLP 2013. | Verify exact split/transformation and obtain applicable dataset redistribution terms. A code licence does not establish a text licence. |
| `emotion_analysis/data` | Content is consistent with Saravia et al.'s emotion corpus. The [author's README](https://github.com/dair-ai/emotion_dataset/blob/master/README.md) specifies educational and research purposes. | Resolve permitted redistribution/use for the intended release or provide a separately licensed reproducible replacement. Do not describe this bundle as unrestricted commercial data. |
| `ecg5000_anomaly_detection/data` | The [UCR/UEA publisher](https://www.timeseriesclassification.com/description.php?Dataset=ECG5000) traces ECG5000 to PhysioNet `chf07`, with heartbeat extraction and interpolation. [PhysioNet's source](https://physionet.org/content/chfdb/1.0.0/) lists ODC Attribution 1.0. | Verify the local binary-label conversion, derivative distribution terms, and origin/generation of `test_indices.csv`. The original record's licence alone does not document every subsequent transformation. |
| `mnist/data` | 10,000 BMP images, presented as MNIST. [Original publisher](https://yann.lecun.com/exdb/mnist/). | Verify source split, BMP conversion and an authoritative retained licence notice. The original publisher page did not expose usable licence text during this review. The generated `mnist_data.bin` cache from master was excluded; the BMP originals remain. |
| `melanoma_cancer/data` | 102 BMP images in benign/malignant folders; original image IDs and source collection unverified. | Obtain original source, attribution, transformations and redistribution permission from the contributor. Do not infer an ISIC/HAM10000 licence from filenames. |
| `translation/data/ES-EN-small.txt` | Spanish-English sentence pairs; authorship not recorded. | Confirm authorship/source and redistribution permission. |
| `legacy_8/n2o_forecast/data` | Source named in master example matches Hansen, L. D., Rani, A., Ortiz Arroyo, D., Durdevic, P. (2024), [Mendeley Data V4](https://data.mendeley.com/datasets/xmbxhscgpr/4), DOI 10.17632/xmbxhscgpr.4, CC BY 4.0. | Verify local column selection, renaming, resampling and shifted controls against the 234 MB source. Source identification and licence are established; local derivation is not yet reproduced. |
| `legacy_8/forecasting/data` | Madrid NO2 binary data and parameter files from master. | Establish original dataset, licence, producer, preprocessing and binary format. A municipal-data origin cannot be assumed. |
| `legacy_8/wwt_optimization/data` | Column-modification JavaScript and a results text file. The CSV required by the example is absent. | Obtain input source and permissions, and reproduce results. This is an incomplete historical application. |

The owner has been asked for missing source/permission records. Until they are
provided or replacements are reviewed, these rows remain unresolved. Do not
publish a source archive containing them as though all data were covered by
OpenNN's LGPL. The installed CMake library package does not install example
datasets, but GitHub's automatic source archives include tracked files.
