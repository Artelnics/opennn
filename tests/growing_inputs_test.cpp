#include "pch.h"

#include "../opennn/growing_inputs.h"
#include "../opennn/training_strategy.h"
#include "../opennn/dataset.h"
#include "../opennn/dense_layer.h"
#include "../opennn/standard_networks.h"


using namespace opennn;


TEST(GrowingInputsTest, DefaultConstructor)
{
    GrowingInputs growing_inputs;

    EXPECT_EQ(growing_inputs.has_training_strategy(), false);
}


TEST(GrowingInputsTest, GeneralConstructor)
{

    TrainingStrategy training_strategy;

    GrowingInputs growing_inputs(&training_strategy);

    EXPECT_EQ(growing_inputs.has_training_strategy(), true);

}

TEST(GrowingInputsTest, InputSelection)
{
    Dataset dataset(20, {2}, {1});
    dataset.set_data_random();
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{2}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    EXPECT_EQ(growing_inputs.has_training_strategy(), true);

    InputsSelectionResults input_selection_results = growing_inputs.perform_input_selection();
    EXPECT_GE(input_selection_results.optimal_input_variables_indices[0], 0);
}


TEST(GrowingInputsTest, InputSelectionKnownResult)
{
    const Index samples = 50;

    Dataset dataset(samples, {2}, {1});

    // Generamos datos donde output = input1, input2 es ruido puro
    MatrixR data(samples, 3);
    for(Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;           // input1: señal clara
        data(i, 1) = type(rand()) / RAND_MAX;      // input2: ruido puro
        data(i, 2) = data(i, 0);                   // output = input1
    }
    dataset.set_data(data);
    dataset.split_samples_random();

    NeuralNetwork neural_network;
    neural_network.add_layer(make_unique<opennn::Dense<2>>(Shape{1}, Shape{1}));

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    growing_inputs.set_maximum_inputs_number(1);
    InputsSelectionResults results = growing_inputs.perform_input_selection();

    // El índice 0 (input1) debería ser seleccionado
    EXPECT_EQ(results.optimal_input_variables_indices.size(), 1);
    EXPECT_EQ(results.optimal_input_variables_indices[0], 0);
}


TEST(GrowingInputsTest, InputSelectionMultipleInputs)
{
    const Index samples = 30;

    Dataset dataset(samples, {2}, {1});

    MatrixR data(samples, 3);
    for(Index i = 0; i < samples; i++)
    {
        data(i, 0) = type(i) / samples;
        data(i, 1) = type(i) / samples * type(2);
        data(i, 2) = data(i, 0) + data(i, 1);
    }

    dataset.set_data(data);
    dataset.set_variable_roles({"Input", "Input", "Target"});
    dataset.split_samples_random();

    ApproximationNetwork neural_network({2}, {1}, {1});

    TrainingStrategy training_strategy(&neural_network, &dataset);

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    InputsSelectionResults results = growing_inputs.perform_input_selection();

    EXPECT_TRUE(results.optimal_input_variables_indices[0] < 2);
}
