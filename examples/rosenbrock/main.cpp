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

//          Eigen::initParallel();

//        Eigen::array<Index, 10> a;

        // Data set
/*
        Device device(Device::EigenSimpleThreadPool);

        ThreadPoolDevice* thread_pool_device = device.get_eigen_thread_pool_device();

        const Eigen::array<IndexPair<Index>, 1> transposed_product_dimensions = {IndexPair<Index>(0, 1)};

        Tensor<type, 2> a(1000,1000);
        a.setRandom();
        Tensor<type, 2> b(1000,1000);
        b.setRandom();
        Tensor<type, 2> c(1000,1000);
        c.setRandom();

        MatrixXf a(10, 10);
        a.setRandom();
        MatrixXf b(10, 10);
        b.setRandom();
        MatrixXf c(10, 10);
        c.setRandom();

        time_t tstart, tend;
        tstart = time(0);

        Index rows;

        for(int i = 0; i < 1000; i++)
        {
            //cout << i << endl;

//            cout << (a*b.transpose()).rows() << endl;

            c.device(*device.get_eigen_thread_pool_device()) = a.contract(b, transposed_product_dimensions);
        }


//        cout << a*b.transpose() << endl;

//        cout << rows << endl;

        tend = time(0);
        cout << "Time: "<< difftime(tend, tstart) <<" seconds."<< endl;

        cout << c.dimension(0) << endl;
//        cout << c.dimension(1) << endl;

//      data_set.generate_Rosenbrock_data(1000, 4);


        // Device

*/

        Device device(Device::EigenSimpleThreadPool);

        // Data set

        Tensor<type, 2> data(1000000,1001);

        data.setRandom();

        DataSet data_set(data);

        data_set.set_device_pointer(&device);

        data_set.set_training();

        data_set.set_batch_instances_number(1000);

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();

        const Index hidden_neurons_number = 1000;

        const Index outputs_number = data_set.get_target_variables_number();

        Tensor<Index, 1> arquitecture(3);

        arquitecture.setValues({inputs_number, hidden_neurons_number, outputs_number});

        NeuralNetwork neural_network(NeuralNetwork::Approximation, arquitecture);
        neural_network.set_device_pointer(&device);

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(1);

        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

        training_strategy.set_device_pointer(&device);

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
