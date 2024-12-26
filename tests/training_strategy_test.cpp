#include "pch.h"

/*

void TrainingStrategyTest::test_constructor()
{
    // Test

    TrainingStrategy training_strategy_1(&neural_network, &data_set);

    EXPECT_EQ(training_strategy.get_neural_network());
    EXPECT_EQ(training_strategy.get_data_set());
}


void TrainingStrategyTest::test_perform_training()
{
    Index samples_number;
    Index inputs_number;
    Index targets_number;

    Index neurons_number;

    Tensor<type, 2> data;

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
    data_set.set(DataSet::SampleUse::Training);

    neural_network.set(NeuralNetwork::ModelType::Approximation, {inputs_number}, {neurons_number}, {targets_number});
    neural_network.set_parameters_random();

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::STOCHASTIC_GRADIENT_DESCENT);
    training_strategy.set_maximum_epochs_number(10);
    training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L1);
    training_strategy.set_display(false);

    training_strategy.perform_training();

}


void TrainingStrategyTest::test_to_XML()
{
    string file_name = "../data/training_strategy.xml";

    ofstream file(file_name);

    if (!file.is_open())
    {
        cerr << "Error: Could not open file " << file_name << "\n";
        return;
    }

    tinyxml2::XMLPrinter document;

    training_strategy.to_XML(document);

    file << document.CStr();
}


void TrainingStrategyTest::test_from_XML()
{
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
    string file_name = "../data/training_strategy.xml";

    training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::GRADIENT_DESCENT);

    training_strategy.save(file_name);
}


void TrainingStrategyTest::test_load()
{
    string file_name = "../data/training_strategy.xml";

    // Test

    training_strategy.save(file_name);
    training_strategy.load(file_name);
}
*/


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
