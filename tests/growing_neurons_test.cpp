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
    training_strategy.set(&neural_network, &data_set);
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

    growing_neurons.set_training_strategy_pointer(&training_strategy);

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Tensor<type, 2> data;

    NeuronsSelectionResults neurons_selection_results;

    // Test

    data.resize(21,2);
    data.setValues({{type(-1),type(0)},
                    {type(-0.9),type(0)},
                    {type(-0.9),type(0)},
                    {type(-0.7),type(0)},
                    {type(-0.6),type(0)},
                    {type(-0.5),type(0)},
                    {type(-0.4),type(0)},
                    {type(-0.3),type(0)},
                    {type(-0.2),type(0)},
                    {type(-0.1),type(0)},
                    {type(0),type(0)},
                    {type(0.1),type(0)},
                    {type(0.2),type(0)},
                    {type(0.3),type(0)},
                    {type(0.4),type(0)},
                    {type(0.5),type(0)},
                    {type(0.6),type(0)},
                    {type(0.7),type(0)},
                    {type(0.8),type(0)},
                    {type(0.9),type(0)},
                    {type(1),type(0)}});
    data_set.set(data);

    Tensor<DataSet::VariableUse, 1> uses(2);
    uses.setValues({DataSet::VariableUse::Input, DataSet::VariableUse::Target});
    data_set.set_columns_uses(uses);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {1,3,1});
    neural_network.set_parameters_constant(type(0));

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::SUM_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(7);
    growing_neurons.set_selection_error_goal(type(1.0e-3f));
    growing_neurons.set_display(false);

    assert_true(neural_network.get_layers_neurons_numbers()[0] == 1, LOG);

    // Test

    samples_number = 21;
    inputs_number = 1;
    targets_number = 1;

    data.resize(samples_number,inputs_number + targets_number);

    data.setValues({{type(-1),type(1)},
                    {type(-0.9), type(-0.9)},
                    {type(-0.9),type(-0.8)},
                    {type(-0.7),type(-0.7)},
                    {type(-0.6),type(-0.6)},
                    {type(-0.5),type(-0.5)},
                    {type(-0.4), type(-0.4)},
                    {type(-0.3),type(-0.3)},
                    {type(-0.2),type(-0.2)},
                    {type(-0.1),type(-0.1)},
                    {type(0),type(0)},
                    {type(0.1),type(0.1)},
                    {type(0.2),type(0.2)},
                    {type(0.3),type(0.3)},
                    {type(0.4),type(0.4)},
                    {type(0.5),type(0.5)},
                    {type(0.6),type(0.6)},
                    {type(0.7),type(0.7)},
                    {type(0.8),type(0.8)},
                    {type(0.9),type(0.9)},
                    {type(1),type(1)}});

    data_set.set(data);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number, 3, targets_number});
    neural_network.set_parameters_constant(type(0));

    training_strategy.set_loss_method(TrainingStrategy::LossMethod::SUM_SQUARED_ERROR);
    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
    training_strategy.set_display(false);

    growing_neurons.set_trials_number(1);
    growing_neurons.set_maximum_neurons_number(5);
    growing_neurons.set_selection_error_goal(type(0));
    growing_neurons.set_maximum_selection_failures(1);
    growing_neurons.set_display(false);

    assert_true(neural_network.get_layers_neurons_numbers()[0] == inputs_number, LOG);

    neurons_selection_results = growing_neurons.perform_neurons_selection();

    assert_true(neurons_selection_results.stopping_condition == NeuronsSelection::StoppingCondition::MaximumNeurons, LOG);

}


void GrowingNeuronsTest::run_test_case()
{
    cout << "Running growing neurons test case...\n";

    // Constructor and destructor methods

    test_constructor();
    test_destructor();

    // Order selection methods

    test_perform_neurons_selection();

    cout << "End of growing neurons test case.\n\n";
}
