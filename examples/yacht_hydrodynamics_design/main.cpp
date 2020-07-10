//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   Y A C H T   R E S I S T A N C E   D E S I G N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is an approximation application. 

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <omp.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Yacht Resistance Design Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool  = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/yachtresistance.csv", ';', true);
        data_set.set_thread_pool_device(thread_pool_device);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        data_set.split_instances_random();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_inputs_methods(input_variables_number);
        scaling_inputs_methods.setConstant("MinimumMaximum");

        Tensor<string, 1> scaling_target_methods(target_variables_number);
        scaling_target_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs(scaling_inputs_methods);

        const Tensor<Descriptives, 1> targets_descriptives = data_set.scale_targets(scaling_target_methods);

        // Neural network

        const Index hidden_neurons_number = 10;

        Tensor<Index, 1> neural_network_architecture(3);
        neural_network_architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_network_architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);
        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();
        unscaling_layer_pointer->set_unscaling_methods(UnscalingLayer::MinimumMaximum);
        unscaling_layer_pointer->set_descriptives(targets_descriptives);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.set_thread_pool_device(thread_pool_device);

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

        cout<<linear_regression_analysis.correlation;

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");

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
