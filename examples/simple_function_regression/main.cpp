//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   F U N C T I O N   R E G R E S S I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is an approximation application. 

// System includes

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdexcept>

#include <omp.h>
// OpenNN includes

#include "../../opennn/opennn.h"

using namespace std;
using namespace OpenNN;

int main(void)
{
    try
    {
        cout << "OpenNN. Simple Function Regression Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Device

        const int n = omp_get_max_threads();
        NonBlockingThreadPool* non_blocking_thread_pool = new NonBlockingThreadPool(n);
        ThreadPoolDevice* thread_pool_device = new ThreadPoolDevice(non_blocking_thread_pool, n);

        // Data set

        DataSet data_set("../data/simple_function_regression.csv", ';', true);

        data_set.set_thread_pool_device(thread_pool_device);

        data_set.set_training();

        // Neural network

        Tensor<Index, 1> neural_network_architecture(3);

        neural_network_architecture.setValues({1, 3, 1});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, neural_network_architecture);

        neural_network.set_thread_pool_device(thread_pool_device);

        dynamic_cast<PerceptronLayer*>(neural_network.get_trainable_layers_pointers()(0))->set_activation_function(PerceptronLayer::Linear);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_thread_pool_device(thread_pool_device);

        training_strategy.set_loss_method(TrainingStrategy::MINKOWSKI_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_loss_index_pointer()->set_regularization_method("NO_REGULARIZATION");

        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

        training_strategy.perform_training();

        neural_network.save_expression_python("simple_function_regresion.py");

        cout<<"bye";

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
