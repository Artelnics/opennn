#include "pch.h"

#include "opennn/registry.h"
#include "opennn/layer.h"
#include "opennn/optimizer.h"
#include "opennn/inputs_selection.h"

using namespace opennn;

namespace
{

template<typename T>
vector<string> sorted_registered_names()
{
    vector<string> names = Registry<T>::instance().registered_names();
    sort(names.begin(), names.end());
    return names;
}

}


TEST(RegistryTest, AllRegistrableComponentsAreRegistered)
{
    register_classes();

    const vector<string> expected_layers = {
        "Activation",
        "Addition",
        "Bounding",
        "C2PSA",
        "Concatenate",
        "Concatenation",
        "Convolutional",
        "Dense",
        "Detection",
        "DetectionV8",
        "Embedding",
        "Flatten",
        "GroupedQueryAttention",
        "LongShortTermMemory",
        "MultiHeadAttention",
        "NonMaxSuppression",
        "Normalization3d",
        "Pooling",
        "Pooling3d",
        "RMSNormalization3d",
        "Recurrent",
        "Scaling",
        "Tokenizer",
        "Unscaling",
        "Upsample"
    };

    const vector<string> expected_optimizers = {
        "AdaptiveMomentEstimation",
        "LevenbergMarquardt",
        "QuasiNewtonMethod",
        "StochasticGradientDescent"
    };

    const vector<string> expected_inputs_selection = {
        "GeneticAlgorithm",
        "GrowingInputs"
    };

    EXPECT_EQ(sorted_registered_names<Layer>(), expected_layers);
    EXPECT_EQ(sorted_registered_names<Optimizer>(), expected_optimizers);
    EXPECT_EQ(sorted_registered_names<InputsSelection>(), expected_inputs_selection);
}
