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

/*
        const Eigen::array<IndexPair<Index>, 1> A_B = {IndexPair<Index>(1, 0)};

        Tensor<type, 2, RowMajor> t1(2,2);
        Tensor<type, 2, RowMajor> t2(2,2);

//          Tensor<type, 2, RowMajor> t3 = t1.contract(t2, A_B);


//        for(Index i = 0; i < t.size(); i++) t.data()[i] = i;

//        cout << t.swap_layout() << endl;

        vector<int> ar = { 10, 20, 30, 40, 50 };

            // Declaring iterator to a vector
            vector<int>::iterator ptr = ar.begin();

            // Using advance() to increment iterator position
            // points to 4

            advance(ptr, 3);

            // Displaying iterator position
            cout << "The position of iterator after advancing is : ";
            cout << *ptr << " " << endl;


        srand(static_cast<unsigned>(time(nullptr)));

        MatrixXf M1 = MatrixXf::Random(3,8);

        vector<Index> rows(1,1);

//        cout <<M1(1,rows) << endl;

        cout << "Column major input:" << endl << M1 << "\n";

        Map<MatrixXf, 0, OuterStride<> > M2(M1.data(), M1.rows(), (M1.cols()+2)/3, OuterStride<>(M1.outerStride()*3));

        cout << "1 column over 3:" << endl << M2 << "\n";

        typedef Matrix<float,Dynamic,Dynamic,RowMajor> RowMajorMatrixXf;
        RowMajorMatrixXf M3(M1);
        cout << "Row major input:" << endl << M3 << "\n";

        Map<RowMajorMatrixXf,0,Stride<Dynamic,3> > M4(M3.data(), M3.rows(), (M3.cols()+2)/3,
                                                      Stride<Dynamic,3>(M3.outerStride(),3));
        cout << "1 column over 3:" << endl << M4 << "\n";
*/
        // Data Set

        const Index samples = 10;
        const Index variables = 3;
/*
        DataSet data_set;

        data_set.generate_Rosenbrock_data(samples, variables+1);

        data_set.set_separator(DataSet::Comma);
        data_set.set_data_file_name("D:/rosenbrock_400000_100.csv");


        data_set.save_data();


        // Read Data

//        DataSet data_set("D:/rosenbrock_1000000_1000.csv", ',', false);

        // Generate Data
*/
        // Device

        Device device(Device::EigenSimpleThreadPool);

        DataSet data_set;

        data_set.generate_Rosenbrock_data(samples, variables+1);

        data_set.set_device_pointer(&device);

        data_set.set_training();
//        data_set.split_instances_random();

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
