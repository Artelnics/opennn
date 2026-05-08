@mainpage OpenNN API Reference

OpenNN is a high-performance C++ library for advanced analytics built around
neural networks. This is the **API reference**; for tutorials and step-by-step
guides please visit [opennn.net](https://www.opennn.net).

The library is developed by [Artelnics](https://artelnics.com). The source
code lives at [github.com/Artelnics/OpenNN](https://github.com/Artelnics/OpenNN).

@section mainpage_core Core classes

A typical OpenNN program revolves around five classes that work together:

| Class                                  | Role                                                          |
|----------------------------------------|---------------------------------------------------------------|
| @ref opennn::Dataset "Dataset"         | Data loading, partitioning, scaling, statistics               |
| @ref opennn::NeuralNetwork "NeuralNetwork" | Architecture definition and layer management              |
| @ref opennn::TrainingStrategy "TrainingStrategy" | Pairs an optimizer with a loss and runs training    |
| @ref opennn::ModelSelection "ModelSelection" | Hyperparameter and architecture search                  |
| @ref opennn::TestingAnalysis "TestingAnalysis" | Model evaluation on held-out data                     |

@section mainpage_layers Layer types

@ref opennn::Dense "Dense", @ref opennn::DenseRelu "DenseRelu",
@ref opennn::Convolutional "Convolutional", @ref opennn::ConvolutionalRelu "ConvolutionalRelu",
@ref opennn::Pooling "Pooling", @ref opennn::Pooling3d "Pooling3d",
@ref opennn::Recurrent "Recurrent", @ref opennn::Embedding "Embedding",
@ref opennn::MultiHeadAttention "MultiHeadAttention", @ref opennn::Normalization3d "Normalization3d",
@ref opennn::Scaling "Scaling", @ref opennn::Unscaling "Unscaling",
@ref opennn::Bounding "Bounding", @ref opennn::Addition "Addition",
@ref opennn::Flatten "Flatten".

@section mainpage_optimizers Optimizers

@ref opennn::AdaptiveMomentEstimation "AdaptiveMomentEstimation" (Adam),
@ref opennn::StochasticGradientDescent "StochasticGradientDescent" (SGD),
@ref opennn::QuasiNewtonMethod "QuasiNewtonMethod" (BFGS),
@ref opennn::LevenbergMarquardtAlgorithm "LevenbergMarquardtAlgorithm".

@section mainpage_first_program First program

The example below loads `iris.csv`, builds a small MLP for regression and
trains it with Adam. See @ref opennn::ApproximationNetwork "ApproximationNetwork"
for the architecture used here.

```cpp
#include "opennn/dataset.h"
#include "opennn/standard_networks.h"
#include "opennn/training_strategy.h"

using namespace opennn;

int main()
{
    // 1. Load and partition the data.
    Dataset data("iris.csv", ",", /*has_header=*/true);

    // 2. Build a 4-input, hidden(3), 3-output approximation MLP.
    ApproximationNetwork network({4}, {3}, {3});

    // 3. Pair the network with the data, choose an optimizer and a loss.
    TrainingStrategy training_strategy(&network, &data);
    training_strategy.set_optimization_algorithm("AdaptiveMomentEstimation");
    training_strategy.set_loss("MeanSquaredError");

    // 4. Train.
    training_strategy.train();

    return 0;
}
```

@section mainpage_navigation Navigating the reference

- [Class List](annotated.html) — every class in alphabetical order.
- [Class Index](classes.html) — alphabetical index by name.
- [File List](files.html) — by header file.

@section mainpage_links Resources

- Tutorials: <https://www.opennn.net/tutorials/>
- License: GNU Lesser General Public License v2.1 or later.
- Issues / contributions: <https://github.com/Artelnics/OpenNN>
