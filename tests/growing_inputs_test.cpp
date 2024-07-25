//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   I N P U T S   T E S T   C L A S S   H E A D E R       
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "growing_inputs_test.h"


GrowingInputsTest::GrowingInputsTest() : UnitTesting()
{
    training_strategy.set(&neural_network, &data_set);
}


GrowingInputsTest::~GrowingInputsTest()
{
}


void GrowingInputsTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    GrowingInputs growing_inputs_1(&training_strategy);

    assert_true(growing_inputs_1.has_training_strategy(), LOG);

    // Test

    GrowingInputs growing_inputs_2;

    assert_true(!growing_inputs_2.has_training_strategy(), LOG);
}


void GrowingInputsTest::test_destructor()
{
    cout << "test_destructor\n";

    GrowingInputs* growing_inputs = new GrowingInputs;

    delete growing_inputs;
}


void GrowingInputsTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

    GrowingInputs growing_inputs(&training_strategy);

    growing_inputs.set_display(false);

    InputsSelectionResults inputs_selection_results;

    // Test
/*
    data_set.generate_random_data(30, 3);

    Tensor<string, 1> columns_uses(3);
    columns_uses.setValues({"Input","Input","Target"});

    data_set.set_raw_variables_uses(columns_uses);

    data_set.split_samples_random();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2,1,1});

    //inputs_selection_results = growing_inputs.perform_inputs_selection();

    assert_true(inputs_selection_results.optimal_input_raw_variables_indices[0] < 2, LOG);

    // Test

    data_set.generate_sum_data(20,3);

    neural_network.set();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {2,6,1});

    TrainingStrategy training_strategy1(&neural_network, &data_set);

    //inputs_selection_results = growing_inputs.perform_inputs_selection();

    assert_true(inputs_selection_results.optimal_input_raw_variables_indices[0] < 2, LOG);
*/
}


void GrowingInputsTest::run_test_case()
{
    cout << "Running growing inputs test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Input selection methods

    test_perform_inputs_selection();

    cout << "End of growing input test case.\n\n";

}
