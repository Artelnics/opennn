#include "pch.h"

#include "../opennn/dataset.h"
#include "../opennn/training_strategy.h"
#include "../opennn/model_selection.h"
#include "../opennn/standard_networks.h"
#include "../opennn/growing_neurons.h"

using namespace opennn;


TEST(ModelSelectionTest, DefaultConstructor)
{
    ModelSelection model_selection;

    EXPECT_EQ(model_selection.has_training_strategy(), false);
}


TEST(ModelSelectionTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    ModelSelection model_selection(&training_strategy);

    EXPECT_EQ(model_selection.has_training_strategy(), true);
}

TEST(ModelSelectionTest, NeuronsSelection)
{
    Dataset dataset(21, {1}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {2}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    ModelSelection model_selection(&training_strategy);
    model_selection.set_neurons_selection("GrowingNeurons");
    model_selection.set_inputs_selection("GrowingInputs");

    EXPECT_NE(model_selection.get_neurons_selection(), nullptr);
    EXPECT_NE(model_selection.get_inputs_selection(), nullptr);
    EXPECT_TRUE(model_selection.has_training_strategy());
}

TEST(ModelSelectionTest, PerformNeuronsSelection)
{
    const Index samples = 30;
    MatrixR data(samples, 2);
    for(Index i = 0; i < samples; i++)
    {
        type x = type(i) / samples * 2 - 1;
        data(i, 0) = x;
        data(i, 1) = x;
    }

    Dataset dataset(samples, {1}, {1});
    dataset.set_data(data);
    dataset.split_samples_random();

    ApproximationNetwork neural_network({1}, {1}, {1});
    TrainingStrategy training_strategy(&neural_network, &dataset);

    ModelSelection model_selection(&training_strategy);
    model_selection.set_neurons_selection("GrowingNeurons");

    GrowingNeurons* growing_neurons = static_cast<GrowingNeurons*>(model_selection.get_neurons_selection());
    growing_neurons->set_maximum_neurons(5);
    growing_neurons->set_display(false);

    NeuronsSelectionResults results = model_selection.perform_neurons_selection();

    EXPECT_GE(results.optimal_neurons_number, 1);
    EXPECT_LE(results.optimal_neurons_number, 5);
    EXPECT_GE(results.optimum_validation_error, 0);
}