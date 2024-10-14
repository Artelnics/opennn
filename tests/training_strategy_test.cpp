//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   T R A I N I N G   S T R A T E G Y   T E S T   C L A S S               
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include "training_strategy_test.h"

namespace opennn
{

TrainingStrategyTest::TrainingStrategyTest() : UnitTesting() 
{
    training_strategy.set(&neural_network, &data_set);
}


TrainingStrategyTest::~TrainingStrategyTest()
{
}


void TrainingStrategyTest::test_constructor()
{
    cout << "test_constructor\n";

    // Test

    TrainingStrategy training_strategy_1(&neural_network, &data_set);

    assert_true(training_strategy.get_neural_network() != nullptr, LOG);
    assert_true(training_strategy.get_data_set() != nullptr, LOG);
}


void TrainingStrategyTest::test_destructor()
{
    cout << "test_destructor\n";

    TrainingStrategy* training_strategy = new TrainingStrategy(&neural_network, &data_set);

    delete training_strategy;
}


void TrainingStrategyTest::test_perform_training()
{
    cout << "test_perform_training\n";

    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index neurons_number;

    Tensor<type, 2> data;
/*
    // Test

    samples_number = 5;
    inputs_number = 2;
    targets_number = 1;
    neurons_number = 4;

    data.resize(samples_number, inputs_number+targets_number);

    data.setValues({{type(0),type(1), type(2)},
                    {type(0),type(1), type(2)},
                    {type(0),type(1), type(2)},
                    {type(0),type(1), type(2)},
                    {type(0),type(1), type(2)}});

    data_set.set(samples_number, inputs_number, targets_number);
    data_set.set_data(data);
    data_set.set_training();

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});
    neural_network.set_parameters_random();

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    training_strategy.set_maximum_epochs_number(10);
    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L1);
    training_strategy.set_display(false);

    training_strategy.perform_training();
*/
}


void TrainingStrategyTest::test_to_XML()
{
    cout << "test_to_XML\n";

    FILE *pFile;

    string file_name = "../data/training_strategy.xml";

    pFile = fopen(file_name.c_str(), "w");

    if(pFile)
    {
        tinyxml2::XMLPrinter document(pFile);

        training_strategy.to_XML(document);

        fclose(pFile);
    }
}


void TrainingStrategyTest::test_from_XML()
{
    cout << "test_from_XML\n";

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT);

    training_strategy.set_default();

    tinyxml2::XMLDocument document;

    string file_name = "../data/training_strategy.xml";

    if(document.LoadFile(file_name.c_str()))
        throw invalid_argument("Cannot load XML file " + file_name + ".\n");

    training_strategy.from_XML(document);
}


void TrainingStrategyTest::test_save()
{
    cout << "test_save\n";

    string file_name = "../data/training_strategy.xml";

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT);

    training_strategy.save(file_name);
}


void TrainingStrategyTest::test_load()
{
    cout << "test_load\n";

    string file_name = "../data/training_strategy.xml";

    // Test

    training_strategy.save(file_name);
    training_strategy.load(file_name);
}


void TrainingStrategyTest::run_test_case()
{
    cout << "Running training strategy test case...\n";

     and destructor

    test_constructor();
    test_destructor();

    // Training

    test_perform_training();

    // Serialization

    test_to_XML();
    test_from_XML();

    test_save();
    test_load();

    cout << "End of training strategy test case.\n\n";
}

}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques, SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.

// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
