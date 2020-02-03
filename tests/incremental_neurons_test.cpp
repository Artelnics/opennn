//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I N C R E M E N T A L   N E U R O N S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "incremental_neurons_test.h"

IncrementalNeuronsTest::IncrementalNeuronsTest() : UnitTesting()
{
}


IncrementalNeuronsTest::~IncrementalNeuronsTest()
{
}


void IncrementalNeuronsTest::test_constructor()
{
    cout << "test_constructor\n";

    NeuralNetwork nn;
    DataSet ds;

    TrainingStrategy training_strategy(&nn, &ds);

    IncrementalNeurons io1(&training_strategy);

    assert_true(io1.has_training_strategy(), LOG);

    IncrementalNeurons io2;

    assert_true(!io2.has_training_strategy(), LOG);

}


void IncrementalNeuronsTest::test_destructor()
{
    cout << "test_destructor\n";

    IncrementalNeurons* io = new IncrementalNeurons;

    delete io;
}


void IncrementalNeuronsTest::test_set_default()
{
    cout << "test_set_default\n";
}


void IncrementalNeuronsTest::test_perform_neurons_selection()
{
    cout << "test_perform_neurons_selection\n";

    string str;
    Tensor<type, 2> data(21,2);

    NeuralNetwork neural_network;

    DataSet data_set;
    TrainingStrategy ts(&neural_network, &data_set);

    IncrementalNeurons io(&ts);

    IncrementalNeurons::IncrementalNeuronsResults* results = nullptr;

    // Test

    str =   "-1 0\n"
            "-0.9 0\n"
            "-0.8 0\n"
            "-0.7 0\n"
            "-0.6 0\n"
            "-0.5 0\n"
            "-0.4 0\n"
            "-0.3 0\n"
            "-0.2 0\n"
            "-0.1 0\n"
            "0.0 0\n"
            "0.1 0\n"
            "0.2 0\n"
            "0.3 0\n"
            "0.4 0\n"
            "0.5 0\n"
            "0.6 0\n"
            "0.7 0\n"
            "0.8 0\n"
            "0.9 0\n"
            "1 0\n";

//    data.parse(str);
    data.setValues({{-1,0},{-0.9,0},{-0.9,0},{-0.7,0},{-0.6,0},{-0.5,0},{-0.4,0},
                    {-0.3,0},{-0.2,0},{-0.1,0},{0.0,0},{0.1,0},{0.2,0},{0.3,0},{0.4,0},
                   {0.5,0},{0.6,0},{0.7,0},{0.8,0},{0.9,0},{1,0}});

    data_set.set(data);

//    Tensor<DataSet::VariableUse, 1> uses(2);
//    uses.setValues({DataSet::Input, DataSet::Target});

//    data_set.set_columns_uses(uses);

    Tensor<Index, 1 > architecture(3);
    architecture.setValues({1,3,1});

    neural_network.set(NeuralNetwork::Approximation,architecture);
    neural_network.set_parameters_constant(0.0);

    ts.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    ts.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    io.set_trials_number(1);
    io.set_maximum_order(7);
    io.set_selection_error_goal(1.0e-3);
    io.set_display(false);

    results = io.perform_neurons_selection();

    assert_true(neural_network.get_layers_neurons_numbers()[0] == 1, LOG);
    assert_true(results->stopping_condition ==
                NeuronsSelection::SelectionErrorGoal, LOG);

    // Test

    str =
            "-1 -1\n"
            "-0.9 -0.9\n"
            "-0.8 -0.8\n"
            "-0.7 -0.7\n"
            "-0.6 -0.6\n"
            "-0.5 -0.5\n"
            "-0.4 -0.4\n"
            "-0.3 -0.3\n"
            "-0.2 -0.2\n"
            "-0.1 -0.1\n"
            "0.0 0.0\n"
            "0.1 0.1\n"
            "0.2 0.2\n"
            "0.3 0.3\n"
            "0.4 0.4\n"
            "0.5 0.5\n"
            "0.6 0.6\n"
            "0.7 0.7\n"
            "0.8 0.8\n"
            "0.9 0.9\n"
            "1 1\n";

//    data.parse(str);
    data.setValues({{-1,1},{-0.9, -0.9},{-0.9,-0.8},{-0.7,-0.7},{-0.6,-0.6},{-0.5,-0.5},{-0.4,-0.4},
                    {-0.3,-0.3},{-0.2,-0.2},{-0.1,-0.1},{0.0,0.0},{0.1,0.1},{0.2,0.2},{0.3,0.3},{0.4,0.4},
                   {0.5,0.5},{0.6,0.6},{0.7,0.7},{0.8,0.8},{0.9,0.9},{1,1}});
    data_set.set(data);

    neural_network.set(NeuralNetwork::Approximation, architecture);
    neural_network.set_parameters_constant(0.0);

    ts.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    ts.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    ts.set_display(false);

    io.set_trials_number(1);
    io.set_maximum_order(7);
    io.set_selection_error_goal(0.0);
    io.set_maximum_selection_failures(1);
    io.set_display(false);

    results = io.perform_neurons_selection();

    assert_true(neural_network.get_layers_neurons_numbers()[0] == 1, LOG);
    assert_true(results->stopping_condition == NeuronsSelection::AlgorithmFinished, LOG);

}


void IncrementalNeuronsTest::test_to_XML()
{
    cout << "test_to_XML\n";

    IncrementalNeurons io;

    tinyxml2::XMLDocument* document = io.to_XML();
    assert_true(document != nullptr, LOG);

    delete document;

}

void IncrementalNeuronsTest::test_from_XML()
{
    cout << "test_from_XML\n";

    IncrementalNeurons io;

    tinyxml2::XMLDocument* document = io.to_XML();
    io.from_XML(*document);

    delete document;

}


void IncrementalNeuronsTest::run_test_case()
{
    cout << "Running incremental order test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Set methods

    test_set_default();

    // Order selection methods

    test_perform_neurons_selection();

    // Serialization methods

    test_to_XML();

    test_from_XML();

    cout << "End of incremental order test case.\n";
}
