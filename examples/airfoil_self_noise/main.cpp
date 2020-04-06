//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F - N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// This is an approximation application.

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Airfoil Self-Noise Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        ///@todo Does not work in linux

        // Data set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);

        const Vector<string> inputs_names = data_set.get_input_variables_names();
        const Vector<string> targets_names = data_set.get_target_variables_names();

        data_set.split_instances_random();

        const Vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
        const Vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();

        // Neural network

        const size_t inputs_number = data_set.get_input_variables_number();
        const size_t hidden_perceptrons_number = 9;
        const size_t outputs_number = data_set.get_target_variables_number();

        NeuralNetwork neural_network(NeuralNetwork::Approximation, {inputs_number, hidden_perceptrons_number, outputs_number});

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

        unscaling_layer_pointer->set_descriptives(targets_descriptives);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);
        training_strategy.get_quasi_Newton_method_pointer()->set_loss_goal(1.0e-3);

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy.perform_training();

        // Testing analysis

        data_set.unscale_inputs_minimum_maximum(inputs_descriptives);
        data_set.unscale_targets_minimum_maximum(targets_descriptives);

        const TestingAnalysis testing_analysis(&neural_network, &data_set);

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
//        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

        optimization_algorithm_results.save("../data/optimization_algorithm_results.dat");

        linear_regression_analysis.save("../data/linear_regression_analysis.dat");

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
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
