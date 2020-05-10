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

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);
        data_set.set_thread_pool_device(thread_pool_device);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        data_set.split_instances_random();

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
        const Tensor<Descriptives, 1> targets_descriptives = data_set.scale_targets_minimum_maximum();

//        data_set.set_batch_instances_number(100);

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();
        const Index hidden_neurons_number = 100;
        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> neural_network_architecture(3);
        neural_network_architecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_network_architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

        unscaling_layer_pointer->set_descriptives(targets_descriptives);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_batch_size(5);

        training_strategy.set_thread_pool_device(thread_pool_device);

        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy.perform_training();

        // Testing analysis

        data_set.unscale_inputs_minimum_maximum(inputs_descriptives);
        data_set.unscale_targets_minimum_maximum(targets_descriptives);

        const TestingAnalysis testing_analysis(&neural_network, &data_set);        

        const TestingAnalysis::LinearRegressionAnalysis linear_regression_analysis = testing_analysis.perform_linear_regression_analysis()[0];

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression("../data/expression.txt");

        training_strategy.save("../data/training_strategy.xml");

        optimization_algorithm_results.save("../data/optimization_algorithm_results.dat");

        linear_regression_analysis.save("../data/linear_regression_analysis.dat");

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
