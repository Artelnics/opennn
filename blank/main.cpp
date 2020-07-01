//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com


// System includes

#include <iostream>
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <stdint.h>
#include <limits.h>
#include <statistics.h>
#include <regex>

// Systems Complementaries

#include <cmath>
#include <cstdlib>
#include <ostream>


// OpenNN includes

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace chrono;


int main(void)
{
    try
    {
        cout << "OpenNN. Blank Application." << endl;

        // Device

        const int n = 4;
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        DataSet data_set("C:/Users/Usuario/Documents/airfoil_self_noise.csv", ';', true);

        data_set.set_thread_pool_device(thread_pool_device);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        Tensor<string,1> scaling_methods(input_variables_number);
        scaling_methods.setConstant("MinimumMaximum");

        Tensor<string,1> unscaling_methods(target_variables_number);
        unscaling_methods.setConstant("MinimumMaximum");

        const Tensor<Descriptives,1> inputs_descriptives = data_set.scale_inputs(scaling_methods);
        const Tensor<Descriptives,1> targets_descriptives = data_set.scale_targets(unscaling_methods);

        Tensor<Index, 1> architecture(3);
        architecture(0) = input_variables_number;
        architecture(1) = 7;
        architecture(2) = target_variables_number;

        NeuralNetwork neural_network(NeuralNetwork::Approximation, architecture);

        neural_network.set_thread_pool_device(thread_pool_device);

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.get_normalized_squared_error_pointer()->set_normalization_coefficient();

        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.perform_training();

        return 0;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return 1;
    }
}


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2020 Artificial Intelligence Techniques, SL.
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
