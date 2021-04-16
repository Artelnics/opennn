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
}


GrowingInputsTest::~GrowingInputsTest()
{
}


void GrowingInputsTest::test_constructor()
{
    cout << "test_constructor\n";

    NeuralNetwork neural_network;
    DataSet data_set;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    GrowingInputs growing_inputs_1(&training_strategy);

    assert_true(growing_inputs_1.has_training_strategy(), LOG);

    GrowingInputs growing_inputs_2;

    assert_true(!growing_inputs_2.has_training_strategy(), LOG);

}


void GrowingInputsTest::test_destructor()
{
    cout << "test_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}


void GrowingInputsTest::test_set_default()
{
    cout << "test_set_default\n";
}


void GrowingInputsTest::test_perform_inputs_selection()
{
    cout << "test_perform_inputs_selection\n";

    DataSet data_set;

    Tensor<type, 2> data;
    

    NeuralNetwork neural_network;

    TrainingStrategy training_strategy(&neural_network, &data_set);

    ModelSelection model_selection(&training_strategy);

    GrowingInputs growing_inputs(&training_strategy);

    InputsSelectionResults inputs_selection_results;

    // Test

//    ds.generate_inputs_selection_data(30,3);

//    data_set.set_columns_uses({"Input","Input","Target"});

    data_set.split_samples_random();

    neural_network.set(NeuralNetwork::Approximation, {2,1,1});

    inputs_selection_results = growing_inputs.perform_inputs_selection();

//    assert_true(gir->optimal_inputs_indices[0] == 0, LOG);

//    gi.delete_selection_history();
//    gi.delete_parameters_history();
//    gi.delete_loss_history();

    // Test

    data_set.generate_sum_data(20,3);

    neural_network.set();

    neural_network.set(NeuralNetwork::Approximation, {2,6,1});

    TrainingStrategy training_strategy1(&neural_network, &data_set);

    inputs_selection_results = growing_inputs.perform_inputs_selection();

//    assert_true(gir->optimal_inputs_indices[0] == 0, LOG);

//    gi1.delete_selection_history();
//    gi1.delete_parameters_history();
//    gi1.delete_loss_history();
}


void GrowingInputsTest::test_to_XML()
{
    cout << "test_to_XML\n";

    GrowingInputs growing_inputs;

//    tinyxml2::XMLDocument* document = growing_inputs.to_XML();
//    assert_true(document != nullptr, LOG);

//    delete document;
}

void GrowingInputsTest::test_from_XML()
{
    cout << "test_from_XML\n";

//    GrowingInputs growing_inputs;

//    tinyxml2::XMLDocument* document = growing_inputs.to_XML();
//    growing_inputs.from_XML(*document);

//    delete document;
}


void GrowingInputsTest::run_test_case()
{
    cout << "Running growing inputs test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_default();

    // Input selection methods

    test_perform_inputs_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    cout << "End of growing input test case.\n\n";
}
