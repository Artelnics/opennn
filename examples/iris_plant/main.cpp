//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a classical pattern recognition problem.

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
        cout << "OpenNN. Iris Plant Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/iris_plant_original.csv", ';', true);

        data_set.set_thread_pool_device(thread_pool_device);

        const Tensor<string, 1> inputs_names = data_set.get_input_variables_names();
        const Tensor<string, 1> targets_names = data_set.get_target_variables_names();

        data_set.split_instances_random();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string, 1> scaling_inputs_methods(input_variables_number);
        scaling_inputs_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.calculate_input_variables_descriptives();
        data_set.scale_inputs(scaling_inputs_methods, inputs_descriptives);

        // Neural network

        const Index hidden_neurons_number = 5;

        Tensor<Index, 1> architecture(3);
        architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        NeuralNetwork neural_network(NeuralNetwork::Classification, architecture);
        neural_network.set_thread_pool_device(thread_pool_device);

        neural_network.set_inputs_names(inputs_names);
        neural_network.set_outputs_names(targets_names);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.set_loss_method(TrainingStrategy::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

        StochasticGradientDescent* stochastic_gradient_descent = training_strategy.get_stochastic_gradient_descent_pointer();

        stochastic_gradient_descent->set_loss_goal(1.0e-3);
        stochastic_gradient_descent->set_maximum_epochs_number(10000);
        stochastic_gradient_descent->set_display_period(100);

        training_strategy.perform_training();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);
        scaling_layer_pointer->set_scaling_methods(ScalingLayer::MinimumMaximum);

        // Testing analysis

        data_set.unscale_inputs(scaling_inputs_methods, inputs_descriptives);

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.set_thread_pool_device(thread_pool_device);

        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

        cout << "Confusion: " << endl;
        cout << confusion << endl;

        // Save results

        data_set.save("../data/data_set.xml");

        neural_network.save("../data/neural_network.xml");

        training_strategy.save("../data/training_strategy.xml");

        cout << "Bye" << endl;

        return 0;
    }
    catch(exception& e)
    {
        cout << e.what() << endl;

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
