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
| [`ecg5000_anomaly_detection`](ecg5000_anomaly_detection/main.cpp) | Anomaly detection with an autoencoder | Bundled CSV |
| [`amazon_reviews`](amazon_reviews/main.cpp) | Sentiment classification | Bundled text |
| [`emotion_analysis`](emotion_analysis/main.cpp) | Text classification with a transformer encoder | Bundled text |
| [`mnist`](mnist/main.cpp) | Image classification | Bundled images |
| [`melanoma_cancer`](melanoma_cancer/main.cpp) | Binary image classification | Bundled images |
| [`bert`](bert/main.cpp) | Fine-tuning a pretrained text classifier | Bundled text; downloads the model |
| [`translation`](translation/main.cpp) | Encoder-decoder translation | Bundled text |
| [`gpt2`](gpt2/main.cpp) | Text generation with a pretrained model | Downloads the model |
| [`qwen3`](qwen3/main.cpp) | Chat with a pretrained model | Downloads the model (about 9 GB) |
| [`yolo`](yolo/main.cpp) | Object detection training | External detection dataset |

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
attribution and licence.
