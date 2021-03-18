//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   C L A S S I F I C A T I O N    A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a pattern recognition problem. 

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Simple classification example." << endl;

        // Data set

        DataSet data_set("../data/simple_pattern_recognition.csv", ';', true);

        data_set.split_samples_random();

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();

        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        Tensor<string, 1> scaling_inputs_methods(inputs_names.dimension(0));

        scaling_inputs_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_input_variables(scaling_inputs_methods);

        // Neural network

        Tensor<Index, 1> neural_network_architecture(3);

        neural_network_architecture.setValues({2, 10, 1});

        NeuralNetwork neural_network(NeuralNetwork::Classification, neural_network_architecture);

        neural_network.set_inputs_names(inputs_names);

        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::NORMALIZED_SQUARED_ERROR);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

        const TrainingResults training_results = training_strategy.perform_training();

        // Testing analysis

        data_set.unscale_input_variables_minimum_maximum(inputs_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<type, 1> binary_classification_tests = testing_analysis.calculate_binary_classification_tests();

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/expression.py");

        training_strategy.save("../data/training_strategy.xml");
        training_results.save("../data/training_results.dat");

        cout << "Bye" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}  


// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2019 Artificial Intelligence Techniques SL
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
