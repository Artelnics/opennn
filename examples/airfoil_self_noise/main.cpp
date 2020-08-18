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
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Airfoil Self-Noise Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        data_set.split_samples_random();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_inputs_methods(input_variables_number);
        scaling_inputs_methods.setConstant("MinimumMaximum");

        Tensor<string, 1> scaling_target_methods(target_variables_number);
        scaling_target_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives =  data_set.scale_input_variables(scaling_inputs_methods);
        const Tensor<Descriptives, 1> target_descriptives = data_set.scale_target_variables(scaling_target_methods);

        // Neural network

        const Index hidden_neurons_number = 7;

        Tensor<Index, 1> neural_network_architecture(3);
        neural_network_architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_network_architecture);

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        neural_network.set_parameters_random();

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_descriptives(inputs_descriptives);
        scaling_layer_pointer->set_scaling_methods(scaling_inputs_methods);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_descriptives(target_descriptives);
        unscaling_layer_pointer->set_unscaling_methods(scaling_target_methods);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy.perform_training();

        data_set.unscale_input_variables(scaling_inputs_methods, inputs_descriptives);
        data_set.unscale_targets(scaling_target_methods, target_descriptives);

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

        cout << "Intercept: " << linear_regression_analysis.intercept << endl;
        cout << "Slope: " << linear_regression_analysis.slope << endl;
        cout << "Correlation: " << linear_regression_analysis.correlation << endl;

        // Save results

//        data_set.save("../data/data_set.xml");

//        neural_network.save("../data/neural_network.xml");

//        training_strategy.save("../data/training_strategy.xml");

//        optimization_algorithm_results.save("../data/optimization_algorithm_results.dat");

//        linear_regression_analysis.save("../data/linear_regression_analysis.dat");

        cout << "End" << endl;

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
