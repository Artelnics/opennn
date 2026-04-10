#include "pch.h"
#include "../opennn/tensor_utilities.h"
#include "../opennn/loss.h"
#include "../opennn/dense_layer.h"
#include "../opennn/dataset.h"
#include "gtest/gtest.h"

using namespace opennn;

TEST(CrossEntropyError3DTest, DefaultConstructor)
{
    NeuralNetwork neural_network;
    Dataset dataset;

    Loss loss(&neural_network, &dataset);
    loss.set_error(Loss::Error::CrossEntropy);

    EXPECT_TRUE(loss.get_neural_network() != nullptr);
    EXPECT_TRUE(loss.get_dataset() != nullptr);
}


// Batch class has been removed from the API.
// The MultipleClassification_CPU_vs_GPU and BinaryClassification_CPU_vs_GPU
// tests depended on Batch and are disabled until the API is restored.
/*
TEST(CrossEntropyError3DTest, MultipleClassification_CPU_vs_GPU)
{
    ...
}

TEST(CrossEntropyError3DTest, BinaryClassification_CPU_vs_GPU)
{
    ...
}
*/
