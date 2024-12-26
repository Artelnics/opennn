#include "pch.h"

#include "../opennn/model_selection.h"


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

//    data_set.generate_sum_data(20, 2);

    NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, { 1 }, { 2 }, { 1 });
/*
    //training_strategy.set_display(false);

    //model_selection.set_display(false);

    GrowingNeurons* incremental_neurons = model_selection.get_growing_neurons();

    incremental_neurons->set_maximum_selection_failures(2);

    incremental_neurons->set_display(false);

    NeuronsSelectionResults results;

    results = model_selection.perform_neurons_selection();

    EXPECT_EQ(model_selection.get_inputs_selection_method() == ModelSelection::InputsSelectionMethod::GROWING_INPUTS);
    EXPECT_EQ(model_selection.get_neurons_selection_method() == ModelSelection::NeuronsSelectionMethod::GROWING_NEURONS);
    EXPECT_EQ(results.optimum_selection_error != 0.0);
    EXPECT_EQ(results.optimal_neurons_number >= 1);
*/
}
