#include "pch.h"

#include "../opennn/growing_inputs.h"


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
    TrainingStrategy training_strategy;

    GrowingInputs growing_inputs(&training_strategy);
    growing_inputs.set_display(false);

    EXPECT_EQ(growing_inputs.has_training_strategy(), true);

//    InputsSelectionResults inputs_selection_results = growing_inputs.perform_inputs_selection();
}

/*

void GrowingInputsTest::test_perform_inputs_selection()
{


    // Test

    data_set.generate_random_data(30, 3);

    Tensor<string, 1> columns_uses(3);
    columns_uses.setValues({"Input","Input","Target"});

    data_set.set_raw_variable_uses(columns_uses);

    data_set.split_samples_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2,1,1});


    EXPECT_EQ(inputs_selection_results.optimal_input_raw_variables_indices[0] < 2);

    // Test

    data_set.generate_sum_data(20,3);

    neural_network.set();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2,6,1});

    TrainingStrategy training_strategy1(&neural_network, &data_set);

    //inputs_selection_results = growing_inputs.perform_inputs_selection();

    EXPECT_EQ(inputs_selection_results.optimal_input_raw_variables_indices[0] < 2);

}

}
*/