#include "pch.h"

#include "opennn/model_selection/growing_inputs.h"
#include "opennn/training_strategy/training_strategy.h"
#include "opennn/dataset/dataset.h"
#include "opennn/dataset/tabular_dataset.h"
#include "opennn/neural_network/layers/dense_layer.h"
#include "opennn/neural_network/standard_networks.h"

using namespace opennn;

TEST(GrowingInputsTest, DefaultConstructor)
{
    GrowingInputs growing_inputs;
}

TEST(GrowingInputsTest, GeneralConstructor)
{
    TrainingStrategy training_strategy;

    GrowingInputs growing_inputs(&training_strategy);
}

TEST(GrowingInputsTest, InputSelection)
{
    TabularDataset dataset(20, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{2}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    InputsSelectionResult input_selection_results = growing_inputs.perform_input_selection();
    EXPECT_GE(input_selection_results.optimal_input_variables_indices[0], 0);
}

TEST(GrowingInputsTest, InputSelectionKnownResult)
{
    const Index samples = 50;

    TabularDataset dataset(samples, {2}, {1});

    MatrixR data(samples, 3);
    for(Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;
        data(i, 1) = type(rand()) / RAND_MAX;
        data(i, 2) = data(i, 0);
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    growing_inputs.set_maximum_inputs_number(1);
    InputsSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), 1);
    EXPECT_EQ(results.optimal_input_variables_indices[0], 0);
}

TEST(GrowingInputsTest, CrossValidationKeepsPersistentRoles)
{
    const Index samples = 60;

    TabularDataset dataset(samples, {2}, {1});
    MatrixR data(samples, 3);
    for (Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;
        data(i, 1) = type(rand()) / RAND_MAX;
        data(i, 2) = data(i, 0);
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    const vector<SampleRole> roles_before = dataset.get_sample_roles();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense>(Shape{1}, Shape{1}));
    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);
    growing_inputs.set_maximum_inputs_number(1);
    growing_inputs.set_folds_number(3);

    InputsSelectionResult results = growing_inputs.perform_input_selection();

    EXPECT_EQ(results.optimal_input_variables_indices.size(), 1);
    EXPECT_EQ(results.optimal_input_variables_indices[0], 0);

    EXPECT_TRUE(dataset.get_sample_roles() == roles_before);
}
