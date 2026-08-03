#include "pch.h"

#include "opennn/registry.h"
#include "opennn/layer.h"
#include "opennn/optimizer.h"
#include "opennn/inputs_selection.h"

using namespace opennn;

TEST(RegistryTest, AllComponentNamesConstruct)
{
    const vector<string> layer_names = {
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

    const vector<string> optimizer_names = {
        "AdaptiveMomentEstimation",
        "LevenbergMarquardt",
        "QuasiNewtonMethod",
        "StochasticGradientDescent"
    };

    const vector<string> inputs_selection_names = {
        "GeneticAlgorithm",
        "GrowingInputs"
    };

    for (const string& name : layer_names)
        EXPECT_NE(create_layer(name), nullptr) << name;

    for (const string& name : optimizer_names)
        EXPECT_NE(create_optimizer(name), nullptr) << name;

    for (const string& name : inputs_selection_names)
        EXPECT_NE(create_inputs_selection(name), nullptr) << name;
}

TEST(RegistryTest, AliasesConstructConfiguredComponents)
{
    EXPECT_EQ(create_layer("Concatenate")->get_type(), LayerType::Concatenation);
    EXPECT_EQ(create_layer("RMSNormalization3d")->get_type(), LayerType::RMSNormalization3d);
}

TEST(RegistryTest, UnknownComponentThrows)
{
    EXPECT_THROW(create_layer("Unknown"), runtime_error);
    EXPECT_THROW(create_optimizer("Unknown"), runtime_error);
    EXPECT_THROW(create_inputs_selection("Unknown"), runtime_error);
}
