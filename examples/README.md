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
| [`electrocardiogram_anomaly_detection`](electrocardiogram_anomaly_detection/main.cpp) | Heartbeat anomaly detection with an autoencoder | 5,000 bundled heartbeats |
| [`amazon_reviews`](amazon_reviews/main.cpp) | Sentiment classification | 1,000 bundled reviews |
| [`emotion_analysis`](emotion_analysis/main.cpp) | Six-class emotion classification with a transformer encoder | 5,272 bundled messages |
| [`mnist`](mnist/main.cpp) | Image classification | Bundled images |
| [`melanoma_cancer`](melanoma_cancer/main.cpp) | Binary image classification | 102 bundled images |
| [`bert`](bert/main.cpp) | Fine-tuning BERT for sentiment classification | 1,000 bundled reviews; downloads BERT-Base Uncased (about 437 MB) |
| [`translation`](translation/main.cpp) | Spanish-English encoder-decoder translation | 1,000 bundled sentence pairs |
| [`gpt2`](gpt2/main.cpp) | Text generation with a pretrained model | Downloads GPT-2 Small (about 650 MB) |
| [`qwen3`](qwen3/main.cpp) | Chat with a pretrained model | Downloads Qwen3-4B (about 8.83 GB) |
| [`yolo`](yolo/main.cpp) | Object detection training | External dataset; optional pretrained weights |

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
`iris_plant` with any example from the table. Most examples take no arguments;
the others are listed below.

| Example | Usage |
| --- | --- |
| `emotion_analysis` | `emotion_analysis [--device auto\|cpu\|cuda] [--seed N] [--epochs E] [--data FILE]`; for example `--device cpu --epochs 1` for a quick run. |
| `bert` | `bert [reviews.txt] [model_dir]`; the model is cached in `../data/bert`. |
| `gpt2` | `gpt2 ["prompt"]`; the model is cached in `../data/gpt2`. |
| `qwen3` | `qwen3 [model_dir]`; the model is cached in `../data/qwen3`. Type a prompt, and `exit`, `quit` or an empty line to finish. |
| `yolo` | `yolo [experiment]`, default `v3-pretrained`. No dataset is bundled: set `SYNTHETIC_YOLO_IMAGES` and `SYNTHETIC_YOLO_LABELS`, or use an experiment name containing `voc` (`VOC_ROOT`) or `bccd` (`BCCD_IMAGES`, `BCCD_LABELS`). |

Pretrained models are downloaded on the first run.

## Train your first model

[`iris_plant`](iris_plant/main.cpp) shows the complete flow:

1. Load the CSV into a `TabularDataset`.
2. Create a `ClassificationNetwork` matching its inputs and targets.
3. Train it with `Training`.
4. Print the classification results with `Evaluation`.
5. Save the model and export it to C and Python with `ModelExpression`.

The saved model and the exported code are written next to the executable.

## Data and model licences

Each example's `data/SOURCE.md` gives the source, attribution and licence of
its data or pretrained model, and how the bundled files were prepared. The
`bert`, `gpt2` and `qwen3` folders also hold the model's original licence and
notice files; the build copies them next to the downloaded model. Keep these
notices with any redistributed copy. OpenNN's software licence does not replace
the terms of the data or the pretrained models.
