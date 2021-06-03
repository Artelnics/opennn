//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   G R O W I N G   N E U R O N S   T E S T   C L A S S   H E A D E R
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com                                           

#include "growing_neurons_test.h"

GrowingNeuronsTest::GrowingNeuronsTest() : UnitTesting()
{
}


GrowingNeuronsTest::~GrowingNeuronsTest()
{
}


void GrowingNeuronsTest::test_constructor()
{
    cout << "test_constructor\n";

    GrowingNeurons growing_neurons_1(&training_strategy);

    assert_true(growing_neurons_1.has_training_strategy(), LOG);

    GrowingNeurons growing_neurons_2;

    assert_true(!growing_neurons_2.has_training_strategy(), LOG);
}


void GrowingNeuronsTest::test_destructor()
{
    cout << "test_destructor\n";

    GrowingNeurons* growing_neurons_pointer = new GrowingNeurons;

    delete growing_neurons_pointer;
}


void GrowingNeuronsTest::test_perform_neurons_selection()
{
    cout << "test_perform_neurons_selection\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 2> data;

    NeuronsSelectionResults neurons_selection_results;

    // Test

    data.resize(21,2);

    data.setValues({{-1,0},{-0.9f,0},{-0.9f,0},{-0.7f,0},{-0.6f,0},{-0.5,0},{-0.4f,0},
                    {-0.3f,0},{-0.2f,0},{-0.1f,0},{0.0,0},{0.1f,0},{0.2f,0},{0.3f,0},{0.4f,0},
                   {0.5f,0},{0.6f,0},{0.7f,0},{0.8f,0},{0.9f,0},{1,0}});

    data_set.set(data);

    Tensor<DataSet::VariableUse, 1> uses(2);
    uses.setValues({DataSet::Input, DataSet::Target});

    data_set.set_columns_uses(uses);

    neural_network.set(NeuralNetwork::Approximation, {1,3,1});
    neural_network.set_parameters_constant(0.0);

    training_strategy.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(7);
    growing_neurons.set_selection_error_goal(1.0e-3f);
    growing_neurons.set_display(false);

    neurons_selection_results = growing_neurons.perform_neurons_selection();

    assert_true(neural_network.get_layers_neurons_numbers()[0] == 1, LOG);

    assert_true(neurons_selection_results.stopping_condition == NeuronsSelection::SelectionErrorGoal, LOG);

    // Test

    data.setValues({{-1,1},{-0.9f, -0.9f},{-0.9f,-0.8f},{-0.7f,-0.7f},{-0.6f,-0.6f},{-0.5,-0.5},{-0.4f, -0.4f},
                    {-0.3f,-0.3f},{-0.2f,-0.2f},{-0.1f,-0.1f},{0.0,0.0},{0.1f,0.1f},{0.2f,0.2f},{0.3f,0.3f},{0.4f,0.4f},
                   {0.5,0.5},{0.6f,0.6f},{0.7f,0.7f},{0.8f,0.8f},{0.9f,0.9f},{1,1}});

    data_set.set(data);

    neural_network.set(NeuralNetwork::Approximation, {inputs_number, targets_number});
    neural_network.set_parameters_constant(0.0);

    training_strategy.set_loss_method(TrainingStrategy::SUM_SQUARED_ERROR);

    training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(7);
    growing_neurons.set_selection_error_goal(0.0);
    growing_neurons.set_maximum_selection_failures(1);
    growing_neurons.set_display(false);

    neurons_selection_results = growing_neurons.perform_neurons_selection();

    assert_true(neural_network.get_layers_neurons_numbers()[0] == 1, LOG);

    assert_true(neurons_selection_results.stopping_condition == NeuronsSelection::MaximumEpochs, LOG);

}


void GrowingNeuronsTest::run_test_case()
{
    cout << "Running incremental order test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Order selection methods

    test_perform_neurons_selection();

    cout << "End of incremental order test case.\n\n";
}
