#include "pch.h"

#include "opennn/registry.h"
#include "opennn/layer.h"
#include "opennn/optimizer.h"
#include "opennn/inputs_selection.h"
#include "opennn/neuron_selection.h"

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
        "Concatenate",
        "Concatenation",
        "Convolutional",
        "Dense",
        "Detection",
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

    const vector<string> expected_neuron_selection = {
        "GrowingNeurons"
    };

    EXPECT_EQ(sorted_registered_names<Layer>(), expected_layers);
    EXPECT_EQ(sorted_registered_names<Optimizer>(), expected_optimizers);
    EXPECT_EQ(sorted_registered_names<InputsSelection>(), expected_inputs_selection);
    EXPECT_EQ(sorted_registered_names<NeuronSelection>(), expected_neuron_selection);
}
