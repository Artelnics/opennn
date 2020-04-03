//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R O S E N B R O C K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

#include <../eigen/unsupported/Eigen/CXX11/Tensor>

#include <../../eigen/unsupported/Eigen/CXX11/ThreadPool>

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main(void)
{          
    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;

        // Data Set

        const Index samples = 1000000;
        const Index variables = 1000;

        Device device(Device::EigenThreadPool);

        DataSet data_set;

        data_set.generate_Rosenbrock_data(samples, variables+1);

        data_set.set_device_pointer(&device);

        data_set.set_training();

        const Tensor<Descriptives, 1> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
        const Tensor<Descriptives, 1> targets_descriptives = data_set.scale_targets_minimum_maximum();

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();

        const Index hidden_neurons_number = variables;

        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> arquitecture(3);

        arquitecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, arquitecture);
        neural_network.set_device_pointer(&device);

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

        scaling_layer_pointer->set_descriptives(inputs_descriptives);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(10);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

        training_strategy.set_device_pointer(&device);

        StochasticGradientDescent* stochastic_gradient_descent_pointer
                = training_strategy.get_stochastic_gradient_descent_pointer();

        stochastic_gradient_descent_pointer->set_batch_size(variables);

        stochastic_gradient_descent_pointer->perform_training();

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
