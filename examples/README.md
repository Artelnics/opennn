# OpenNN examples

Each folder is one example: a `main.cpp` and, when it needs one, a `data/`
folder. Start with [`blank`](blank/main.cpp), an empty template, and then train
your first model with [`iris_plant`](#train-your-first-model).

## Examples

| Example | What it shows | Data |
| --- | --- | --- |
| [`blank`](blank/main.cpp) | Empty template for a new application | None |
| [`iris_plant`](iris_plant/main.cpp) | Classification, evaluation and export to C and Python | Bundled CSV |
| [`airfoil_self_noise`](airfoil_self_noise/main.cpp) | Regression | Bundled CSV |
| [`yacht_hydrodynamics`](yacht_hydrodynamics/main.cpp) | Regression with neuron selection and quasi-Newton training | Bundled CSV |
| [`breast_cancer`](breast_cancer/main.cpp) | Binary classification | Bundled CSV |
| [`concrete`](concrete/main.cpp) | Trains a model of concrete strength, then optimizes the mix under constraints | Bundled CSV |
| [`electrocardiogram_anomaly_detection`](electrocardiogram_anomaly_detection/main.cpp) | Heartbeat anomaly detection with an autoencoder | 5,000 PhysioNet/BIDMC heartbeats; [ODC-By 1.0 and attribution](electrocardiogram_anomaly_detection/data/SOURCE.md) |
| [`amazon_reviews`](amazon_reviews/main.cpp) | Sentiment classification | 1,000 Amazon/UCI reviews; [CC BY 4.0 and attribution](amazon_reviews/data/SOURCE.md) |
| [`emotion_analysis`](emotion_analysis/main.cpp) | Six-class emotion classification with a transformer encoder | 5,272 GoEmotions messages; [CC BY 4.0 and attribution](emotion_analysis/data/SOURCE.md) |
| [`mnist`](mnist/main.cpp) | Image classification | Bundled images |
| [`melanoma_cancer`](melanoma_cancer/main.cpp) | Binary image classification | 102 ISIC 2016 images; [CC0 1.0 and provenance](melanoma_cancer/data/SOURCE.md) |
| [`bert`](bert/main.cpp) | Fine-tuning BERT for sentiment classification | 1,000 Amazon/UCI reviews (CC BY 4.0); downloads BERT-Base Uncased (Apache 2.0, about 437 MB); [licences and notices](bert/data/SOURCE.md) |
| [`translation`](translation/main.cpp) | Spanish-English encoder-decoder translation | 1,000 Tatoeba/ManyThings pairs; [CC BY 2.0 France and attribution](translation/data/SOURCE.md) |
| [`gpt2`](gpt2/main.cpp) | Text generation with a pretrained model | Downloads GPT-2 Small (about 650 MB); [Modified MIT licence and notices](gpt2/data/SOURCE.md) |
| [`qwen3`](qwen3/main.cpp) | Chat with a pretrained model | Downloads Qwen3-4B (about 8.83 GB); [Apache 2.0 and notices for 4B and 0.6B](qwen3/data/SOURCE.md) |
| [`yolo`](yolo/main.cpp) | Object detection training | External dataset and optional pretrained weights; [sources, licences and unresolved inputs](yolo/data/SOURCE.md) |

The examples run on the GPU when one is available and on the CPU otherwise.
`translation`, `gpt2` and `yolo` require a CUDA GPU. `iris_plant`,
`breast_cancer` and `yacht_hydrodynamics` always run on the CPU, because they
train with the quasi-Newton method, which is CPU-only.

## Build and run

From the repository root, configure a build with examples enabled and build one
example:

```sh
cmake -S . -B ../opennn-build -DCMAKE_BUILD_TYPE=Release -DOpenNN_BUILD_TESTS=OFF -DOpenNN_BUILD_EXAMPLES=ON
cmake --build ../opennn-build --config Release --target iris_plant --parallel
```

Run it from the executable folder, where the build places its data in
`../data/<example>`:

```sh
cd ../opennn-build/bin
./iris_plant
```

With Visual Studio the executable is `Release/iris_plant.exe`. Replace
`iris_plant` with any example from the table.

## Train your first model

[`iris_plant`](iris_plant/main.cpp) shows the complete flow:

1. Load the CSV into a `TabularDataset`.
2. Create a `ClassificationNetwork` matching its inputs and targets.
3. Train it with `Training`.
4. Print the classification results with `Evaluation`.
5. Save the model and export it to C and Python with `ModelExpression`.

The saved model and the exported code are written next to the executable.

Where a dataset comes from a public source, its `data/SOURCE.md` gives the
attribution and licence. Keep that notice with the data when redistributing it.
OpenNN's software licence does not replace the terms for datasets or pretrained
models. Keep this catalog and the affected `SOURCE.md` notices updated whenever
an example or its data changes.

`bert` and `amazon_reviews` use the same 1,000 reviews from UCI's *Sentiment
Labelled Sentences*. BERT's default input is
`../data/bert/amazon_cells_labelled.txt`, with `Bad`/`Good` labels. The reviews
are bundled with each example so either target can be built independently.

The BERT model and WordPiece vocabulary are downloaded from
[Artelnics/bert-base-uncased-opennn](https://huggingface.co/Artelnics/bert-base-uncased-opennn)
and use Apache 2.0. CMake places
`LICENSE_BERT.txt`, `NOTICE_BERT.txt` and `SOURCE.md` beside the downloaded
files in `../data/bert`; keep those notices with redistributed model files.
Run `bert [reviews.txt] [model_dir]` from the executable directory. It defaults
to the bundled reviews and model cache above, fine-tunes for three epochs
with batches of 32, then prints binary classification metrics.

`emotion_analysis` uses the single-label GoEmotions messages for `anger`,
`fear`, `joy`, `love`, `sadness` and `surprise`. Its default input is
`../data/emotion_analysis/goemotions.txt`; it creates its own stratified
80/10/10 training/validation/testing split. To run one epoch on the CPU:

```sh
./emotion_analysis --device cpu --epochs 1
```

The data notice links to the original files, licence and regeneration command.

`electrocardiogram_anomaly_detection` uses 5,000 heartbeats extracted directly
from PhysioNet's BIDMC `chf07` recording. Each row contains 140 signal values
and a normal/non-normal label. It trains an autoencoder on normal development
beats and evaluates anomaly detection on 1,000 held-out beats. Its data notice
records the extraction, labels, split and regeneration procedure.

Build target `electrocardiogram_anomaly_detection` and run it from the executable
folder without arguments; it trains for 20 epochs and prints the anomaly
threshold, confusion matrix and classification metrics, including ROC AUC.

`melanoma_cancer` uses 102 ISIC 2016 images: 50 benign lesions and 52 melanomas.
They are RGB BMPs resized to 300 x 300, with the original ISIC identifiers and
labels recorded in `data/manifest.csv`. CMake extracts `data/images.zip` into
`../data/melanoma_cancer/isic2016`, which is the example's input directory.
Its data notice records the source, CC0 dedication and regeneration procedure.
Run the executable without arguments; it trains an image classifier and prints
the binary classification metrics.

`translation` uses 1,000 Spanish-English sentence pairs from Tatoeba, compiled
by ManyThings. Its input is `../data/translation/tatoeba_es_en.tsv`, with
Spanish in the first column and English in the second. The sentence owners,
original identifiers and licence links are preserved in `data/ATTRIBUTION.tsv`;
keep that file and the data notices with any redistributed copy.
Run it without arguments on a CUDA GPU. It trains for 50 epochs with batches
of 16, then prints a translation of `Vete.`. Its data notice gives
the recorded source version, selection and regeneration procedure.

`gpt2` downloads the converted GPT-2 Small weights and byte-pair tokenizer from
[Artelnics/gpt2-small-opennn](https://huggingface.co/Artelnics/gpt2-small-opennn) to
`../data/gpt2`. CMake copies the model's original `LICENSE_GPT2.txt` and
`SOURCE.md` into that directory; keep both notices with any redistributed model
or tokenizer files. The model uses OpenAI's Modified MIT License, independently
of OpenNN's software licence.

Run it on a CUDA GPU with an optional quoted prompt:

```sh
./gpt2 "Artificial intelligence"
```

It labels the generated text and produces at most 40 new tokens. Without an
argument it uses the same prompt shown above.

`qwen3` downloads the converted Qwen3-4B BF16 weights and tokenizer to
`../data/qwen3`. CMake places `LICENSE_QWEN3.txt`, `NOTICE_QWEN3.txt` and
`SOURCE.md` in that directory; keep these notices with any redistributed
model or tokenizer files. Qwen3-4B and the library's Qwen3-0.6B variant use
Apache 2.0, including the original Alibaba Cloud attribution.

Run `qwen3 [model_dir]` from the executable directory. Enter a prompt,
then `exit`, `quit` or an empty line to finish. The example selects 4B;
choosing 0.6B requires selecting `Qwen3::Variant::B0_6` in the calling code.
Pass `../data` explicitly to reuse a cache from the earlier default path,
and copy the model notices into that cache as well.

`yolo [experiment]` requires CUDA FP32. The default `v3-pretrained` uses
Darknet53 and externally supplied 416 x 416 synthetic images and labels.
Set `SYNTHETIC_YOLO_IMAGES` and `SYNTHETIC_YOLO_LABELS`, or supply
`synthetic_yolo/images` and `synthetic_yolo/labels` relative to the executable
directory. Those files have no recorded generator or licence in OpenNN.
`v3-scratch` skips pretrained-backbone loading; use a fresh experiment directory
to avoid resuming a checkpoint imported earlier.

An experiment name containing `voc` selects PASCAL VOC (`VOC_ROOT`, optional
`VOC12_ROOT`); one containing `bccd` selects BCCD (`BCCD_IMAGES`, `BCCD_LABELS`).
The source has additional COCO and Raccoon paths disabled by default.
No external dataset or YOLO weights are bundled or published by OpenNN.

The original Darknet licence is a custom public-domain declaration, preserved
in `yolo/data/LICENSE_DARKNET.txt`; the source notice records the upstream
release clarification and the limits of the checkpoint verification.
The optional `v8-pretrained` ONNX import and `yolov8s_to_nd.py` conversion use
Ultralytics checkpoints under AGPL-3.0 unless separately licensed. Conversion
does not replace that licence with OpenNN's LGPL. The complete AGPL text and
the required attribution, source and conversion records are linked from the
source notice. The internal synthetic-image generator is currently unused.

## Remaining licence review work

| Example | Remaining work |
| --- | --- |
| `yolo` | Replace or establish provenance for the default external synthetic dataset. Obtain checkpoint-specific confirmation before redistributing `darknet53.conv.74` or `yolov3-tiny.weights`; optional external datasets and Ultralytics conversions need their actual file manifests and applicable notices/source records. See the reviewed source notice above. |

Keep this table current as each remaining item is resolved.

GPT-2 and BERT weights, tokenizers, original licences and source notices are
published on Hugging Face: [GPT-2 publication](https://huggingface.co/Artelnics/gpt2-small-opennn/commit/1201797a74b0d3f352dbdf7a3c4dfa18b2c5dc0c)
and [BERT publication](https://huggingface.co/Artelnics/bert-base-uncased-opennn/commit/5d118ff0e833af441ecc54a83302c2a3b2b4b6ec).
The model and tokenizer bytes match the former GitHub release assets. Current
OpenNN downloads from Hugging Face; older OpenNN versions retain their GitHub
download URLs and need those URLs updated if the old assets are removed.

The Qwen3 notices and model-card links are published and their public downloads
verified: [0.6B publication](https://huggingface.co/Artelnics/qwen3-0.6b-opennn/commit/80398f97dd796e80ee57fc6a014c9eb144bd0886)
and [4B publication](https://huggingface.co/Artelnics/qwen3-4b-opennn/commit/7b6c010b04027c7cdef23172cbdb3e1b1c6561a6).
