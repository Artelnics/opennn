//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <time.h>

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        // Data set

        DataSet data_set("/Users/artelnics/Documents/opennn/examples/iris_plant/data/iris_plant_original.csv", ";", true);

        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);

//        data_set.save("../data/data_set.xml");
//        data_set.load("../data/data_set.xml");

        // Neural network


        const Index hidden_neurons_number = 6;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                     {input_variables_number}, {}, {target_variables_number});

//        neural_network.save("../data/neural_network.xml");
//        neural_network.load("../data/neural_network.xml");

        neural_network.print();

        CrossEntropyError cross_entropy_error(&neural_network, &data_set);

        cross_entropy_error.calculate_numerical_gradient();

        TrainingStrategy training_strategy(&neural_network, &data_set);
//        training_strategy.print();
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.perform_training();


        // Testing analysis

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.print_binary_classification_tests();
/*
        const Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();
        cout << "\nConfusion matrix:\n" << confusion << endl;

        Tensor<type, 2> inputs(3, neural_network.get_inputs_number());

        inputs.setValues({{type(5.1),type(3.5),type(1.4),type(0.2)},
                          {type(6.4),type(3.2),type(4.5),type(1.5)},
                          {type(6.3),type(2.7),type(4.9),type(1.8)}});

        const Tensor<type, 2> outputs = neural_network.calculate_outputs(inputs);

        cout << "\nInputs:\n" << inputs << endl;
        cout << "\nOutputs:\n" << outputs << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");

        neural_network.save_expression_c("data/neural_network.c");
        neural_network.save_expression_python("data/neural_network.py");
*/
        cout << "Bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }   
}  

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
