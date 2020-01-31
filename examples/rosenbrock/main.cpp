//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   R O S E N B R O C K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#ifndef EIGEN_USE_THREADS
#define EIGEN_USE_THREADS
#endif

//#define EIGEN_USE_BLAS

//#define EIGEN_TEST_NO_LONGDOUBLE

//#define EIGEN_TEST_NO_COMPLEX

//#define EIGEN_TEST_FUNC cxx11_tensor_cuda

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE Index

//#define EIGEN_USE_GPU

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


#include "../opennn/config.h"
//#include "../opennn/device.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;


int main(void)
{

    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;

//      data_set.generate_Rosenbrock_data(1000, 4);

        // Data set

        Tensor<type, 2> data(10000,3);

//        data.setValues({{0,1,2},
//                       {3,4,5},
//                       {6,7,8},
//                       {9,10,11},
//                       {12,13,14},
//                       {15,16,17}});

        data.setRandom();

        DataSet data_set(data);

        data_set.set_training();

        data_set.set_batch_instances_number(10000);

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();

        const Index hidden_neurons_number = 3;

        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> arquitecture(3);

        arquitecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, arquitecture);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(MeanSquaredError::NoRegularization);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(10000);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

        training_strategy.get_stochastic_gradient_descent_pointer()->perform_training();

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
