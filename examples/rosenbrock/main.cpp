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

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

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

//#include "../eigen/Eigen/Eigen"
#include "../opennn/config.h"
#include "../opennn/device.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;


int main(void)
{

    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;
/*
        int n = 4;

        SimpleThreadPool* simple_thread_pool;

        simple_thread_pool = new SimpleThreadPool(n);

        ThreadPoolDevice* thread_pool_device;

        thread_pool_device = new ThreadPoolDevice(simple_thread_pool, n);

        Tensor<type, 2> inputs(1000, 1000);
        inputs.setRandom();

        Tensor<type, 2> deltas(1000, 1000);
        deltas.setRandom();

        Tensor<type, 2> derivatives(1000, 1000);

        const Eigen::array<IndexPair<int>, 1> dimensions = {IndexPair<int>(1, 0)};

        derivatives.device(*thread_pool_device) = inputs.contract(deltas, dimensions);

        cout << derivatives.dimension(0) << endl;


//        ThreadPoolDevice thread_pool_device(&simple_thread_pool, n);

    derivatives.device(thread_pool_device) = inputs.contract(deltas, dimensions);

    time_t beginning_time, current_time;
    time(&beginning_time);
    double elapsed_time = 0.0;

    for(int i = 0; i < 1000; i++)
    {
        //cout << i << endl;

        derivatives.device(thread_pool_device) = inputs.contract(deltas, dimensions);

//            c.device(thread_pool_device) = a.contract(b, product_dimensions);

//            c.device(thread_pool_device) = a;

        //c.setConstant(1.0);

    }
  */
/*

        MatrixXf a(1000, 1000);
        MatrixXf b(1000, 1000);
        MatrixXf c(1000, 1000);

        int n = 8;//omp_get_max_threads();

        omp_set_num_threads(n);
        Eigen::setNbThreads(n);

        time_t beginning_time, current_time;
        time(&beginning_time);
        double elapsed_time = 0.0;

        for(int i = 0; i < 1000; i++)
        {
            //cout << i << endl;

            c = a*b;
        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        cout <<  " = " <<  elapsed_time << endl;



        cout << c.rows() << endl;
        cout << c.cols() << endl;
*/
/*
        Tensor<type, 2> inputs(1000, 1000);
        inputs.setRandom();

        Tensor<type, 2> deltas(1000, 1000);
        deltas.setRandom();

        Tensor<type, 2> derivatives(1000, 1000);

//        for(size_t iii = 2; iii < 10; iii++)
        {

            int n = 16;//iii+1;//omp_get_max_threads();

            NonBlockingThreadPool simple_thread_pool(n);

            ThreadPoolDevice thread_pool_device(&simple_thread_pool, n);

        const Eigen::array<IndexPair<int>, 1> dimensions = {IndexPair<int>(1, 0)};

        derivatives.device(thread_pool_device) = inputs.contract(deltas, dimensions);

        time_t beginning_time, current_time;
        time(&beginning_time);
        double elapsed_time = 0.0;

        for(int i = 0; i < 1000; i++)
        {
            //cout << i << endl;

            derivatives.device(thread_pool_device) = inputs.contract(deltas, dimensions);

//            c.device(thread_pool_device) = a.contract(b, product_dimensions);

//            c.device(thread_pool_device) = a;

            //c.setConstant(1.0);

        }

        time(&current_time);
        elapsed_time = difftime(current_time, beginning_time);

        cout << elapsed_time << endl;
    }

        cout << derivatives.dimension(0) << endl;
//        system("pause");
*/
/*
        Eigen::Tensor<int, 2> a(1, 2);
        a.setValues({{0, 1}});

        cout << "a" << endl << a << endl;

        Eigen::array<int, 2> bcast;
        bcast[0] = 5;
        bcast[1] = 1;
        //({3, 2});
        Eigen::Tensor<int, 2> b = a.broadcast(bcast);

        cout << "b" << endl << b << endl;
*/
        // Data set

        DataSet data_set;

        data_set.generate_Rosenbrock_data(1000000, 1001);

//        const vector<string> inputs_names = data_set.get_input_variables_names();
//        const vector<string> targets_names = data_set.get_target_variables_names();

//        const vector<Descriptives> inputs_descriptives = data_set.scale_inputs_minimum_maximum();
//        const vector<Descriptives> targets_descriptives = data_set.scale_targets_minimum_maximum();

        data_set.set_batch_instances_number(1000);

        data_set.set_training();

        // Neural network

        const int inputs_number = data_set.get_input_variables_number();
        const int hidden_neurons_number = 1000;
        const int outputs_number = data_set.get_target_variables_number();

        NeuralNetwork neural_network(NeuralNetwork::Approximation, {inputs_number, hidden_neurons_number, outputs_number});

//        neural_network.set_inputs_names(inputs_names);
//        neural_network.set_outputs_names(targets_names);

//        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();

//        scaling_layer_pointer->set_descriptives(inputs_descriptives);

//        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

//        unscaling_layer_pointer->set_descriptives(targets_descriptives);

        // Training strategy object

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::STOCHASTIC_GRADIENT_DESCENT);
        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(0);
        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

//        const OptimizationAlgorithm::Results optimization_algorithm_results = training_strategy.perform_training();

        training_strategy.get_stochastic_gradient_descent_pointer()->perform_training();

//        optimization_algorithm_results.save("../data/optimization_algorithm_results.dat");

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
