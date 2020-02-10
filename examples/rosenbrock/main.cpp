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

using Eigen::Tensor;


#define EIGEN_TEST_NO_LONGDOUBLE

#define EIGEN_TEST_NO_COMPLEX

#define EIGEN_TEST_FUNC cxx11_tensor_cuda

#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#define EIGEN_USE_GPU

int main(void)
{          
    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;
/*
        Tensor<float, 1> in1(2);

        Tensor<float, 1> in2(2);

        Tensor<float, 1> out(2);

        in1.setRandom();

        in2.setRandom();



        std::size_t in1_bytes = in1.size() * sizeof(float);

        std::size_t in2_bytes = in2.size() * sizeof(float);

        std::size_t out_bytes = out.size() * sizeof(float);



        float* d_in1;

        float* d_in2;

        float* d_out;

        cudaMalloc((void**)(&d_in1), in1_bytes);

        cudaMalloc((void**)(&d_in2), in2_bytes);

        cudaMalloc((void**)(&d_out), out_bytes);



        cudaMemcpy(d_in1, in1.data(), in1_bytes, cudaMemcpyHostToDevice);

        cudaMemcpy(d_in2, in2.data(), in2_bytes, cudaMemcpyHostToDevice);

        Eigen::CudaStreamDevice stream;

        Eigen::GpuDevice gpu_device(&stream);

        Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in1(d_in1, 2);

        Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_in2(d_in2, 2);

        Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> gpu_out(d_out, 2);

<<<<<<< Updated upstream
*/


//        gpu_out.device(gpu_device) = gpu_in1 + gpu_in2;

/*

//        Eigen::CudaStreamDevice stream;

//        Eigen::GpuDevice gpu_device(&stream);

//        GpuDevice gpu_device();

//        combinations.device(*gpu_device) = inputs.contract(synaptic_weights, product_dimensions);


        Tensor<type, 2> a(1000, 1000);
        Tensor<type, 2> b(1000, 1000);
        Tensor<type, 2> c(1000, 1000);

        a.setRandom();
        b.setRandom();

        std::size_t bytes_a = a.size()  * sizeof(float);
        std::size_t bytes_b = b.size() * sizeof(float);
        std::size_t bytes_c = c.size() * sizeof(float);



        float* data_a;

        float* data_b;

        float* data_c;

        cudaMalloc((void**)(&data_a), bytes_a);
        cudaMalloc((void**)(&data_b), bytes_b);
        cudaMalloc((void**)(&data_c), bytes_c);
        cudaMemcpy(data_a, a.data(), bytes_a, cudaMemcpyHostToDevice);
        cudaMemcpy(data_b, b.data(), bytes_b, cudaMemcpyHostToDevice);

        cudaStream_t stream;

        assert(cudaStreamCreate(&stream) == cudaSuccess);

        Eigen::GpuDevice gpu_device(&stream);

        CudaStreamDevice stream;

        GpuDevice gpu_device(&stream);

        TensorMap< Tensor<float, 2> > gpu_t_left(data_a, 1000, 1000);

        TensorMap< Tensor<float, 2> > gpu_t_right(data_b, 1000, 1000);

        TensorMap< Tensor<float, 2> > gpu_t_result(data_c, 1000, 1000);

        typedef Tensor<float, 1>::DimensionPair DimPair;

        Eigen::array<DimPair, 2> dims;

        dims[0] = DimPair(2, 0);

        dims[1] = DimPair(3, 1);

//        const Eigen::array<IndexPair<Index>, 1> product_dimensions = {IndexPair<Index>({1, 0})};

        //        typedef Eigen::Map<Eigen::Matrix<float, Dynamic, Dynamic> > MapXf;

        //        MapXf m_left(a.data(), 1000, 1000);

        //        MapXf m_right(b.data(), 1000, 1000);

        //        Eigen::Matrix<float, Dynamic, Dynamic> m_result(1000, 1000);

        // m_result = m_left * m_right;

        //        gpu_t_result.device(gpu_device) = gpu_t_left.contract(gpu_t_right, dims);

//        gpu_t_result.device(gpu_device) = gpu_t_left + gpu_t_right;

        cout << "hello" << endl;
*/
/*
        cudaMemcpy(c.data(), data_c, bytes_c, cudaMemcpyDeviceToHost);

        for (size_t i = 0; i < c.dimensions().TotalSize(); i++) {

          if (fabs(c.data()[i] - m_result.data()[i]) >= 1e-4) {

            cout << "mismatch detected at index " << i << ": " << c.data()[i] << " vs " <<  m_result.data()[i] << endl;

            assert(false);

          }
        }
*/

//      Eigen::initParallel();

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
*/
        // Device

        Device device(Device::EigenSimpleThreadPool);

        // Data set

//        Tensor<type, 2> data(1000000, 1001);
        Tensor<type, 2> data(100000, 51);

        data.setRandom();
//        data.setConstant(1);

        DataSet data_set(data);

        data_set.set_device_pointer(&device);

        data_set.set_training();
//        data_set.split_instances_random();

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

        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(LossIndex::NoRegularization);

//        training_strategy.get_stochastic_gradient_descent_pointer()->set_maximum_epochs_number(20);

//        training_strategy.get_stochastic_gradient_descent_pointer()->set_display_period(1);

        training_strategy.set_device_pointer(&device);

//        training_strategy.get_stochastic_gradient_descent_pointer()->perform_training();
        training_strategy.get_quasi_Newton_method_pointer()->perform_training();

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
