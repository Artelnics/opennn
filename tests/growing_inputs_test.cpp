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


void GrowingInputsTest::test_destructor() // @todo
{
    cout << "test_destructor\n";

    GrowingInputs* gi = new GrowingInputs;

    delete gi;
}


void GrowingInputsTest::test_set_default() // @todo
{
    cout << "test_set_default\n";
}


void GrowingInputsTest::test_perform_inputs_selection() // @todo
{
    cout << "test_perform_inputs_selection\n";

    DataSet data_set;

    Tensor<type, 2> data;
    Tensor<Index, 1> architecture;

    NeuralNetwork neural_network;

    SumSquaredError sum_squared_error(&neural_network ,&data_set);

    InputsSelectionResults* gir;

    // Test

//    ds.generate_inputs_selection_data(30,3);

//    ds.set_columns_uses({"Input","Input","Target"});

    data_set.split_samples_random();

    architecture.resize(3);
    architecture.setValues({2,1,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    TrainingStrategy ts(&neural_network, &data_set);

    ModelSelection ms(&ts);

    GrowingInputs gi(&ts);

    ts.set_display(false);

    gi.set_display(false);

    gi.set_approximation(true);

    gir = gi.perform_inputs_selection();

//    assert_true(gir->optimal_inputs_indices[0] == 0, LOG);

//    gi.delete_selection_history();
//    gi.delete_parameters_history();
//    gi.delete_loss_history();

    // Test

    data_set.generate_sum_data(20,3);

    neural_network.set();

    architecture.setValues({2,6,1});

    neural_network.set(NeuralNetwork::Approximation, architecture);

    TrainingStrategy ts1(&neural_network, &data_set);

    ModelSelection ms1(&ts);

    GrowingInputs gi1(&ts);

    ts1.set_display(false);

    gi1.set_display(false);

    gi1.set_approximation(false);

    gir = gi1.perform_inputs_selection();

//    assert_true(gir->optimal_inputs_indices[0] == 0, LOG);

//    gi1.delete_selection_history();
//    gi1.delete_parameters_history();
//    gi1.delete_loss_history();
}

// Serialization methods

void GrowingInputsTest::test_to_XML() // @todo
{
    cout << "test_to_XML\n";

    GrowingInputs growing_inputs;

//    tinyxml2::XMLDocument* document = growing_inputs.to_XML();
//    assert_true(document != nullptr, LOG);

//    delete document;
}

void GrowingInputsTest::test_from_XML() // @todo
{
    cout << "test_from_XML\n";

//    GrowingInputs growing_inputs;

//    tinyxml2::XMLDocument* document = growing_inputs.to_XML();
//    growing_inputs.from_XML(*document);

//    delete document;
}


void GrowingInputsTest::run_test_case() // @todo
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
