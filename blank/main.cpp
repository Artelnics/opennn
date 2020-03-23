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

void get_batches()
{
    Index buffer_size = 3;
    Index batches_number = 2;
    Index batch_size = 5;

    Tensor<type, 1> data(10);
    data.setValues({1,2,3,4,5,6,7,8,9,10});

    Tensor<type, 2> batches(batches_number, batch_size);

    TensorMap< Tensor<type, 1> > buffer(data.data(), buffer_size);
    TensorMap< Tensor<type, 1> > rest_data(data.data() + buffer_size, data.size()-buffer_size);

    Index count = 0;

    cout << "Initial buffer" << buffer <<endl;

    for(Index i = 0; i < batches_number; i++)
    {
        if(i == batches_number-1)
        {
            random_shuffle(buffer.data(), buffer.data() +  buffer.size());

            for(Index j = 0; j < buffer_size;j++)
            {
                batches(i,j) = buffer(j);
            }
            for(Index j = buffer_size; j < batch_size; j++)
            {
                batches(i,j) = rest_data(count);
                count++;
            }

            break;
        }
        for(Index j = 0; j < batch_size; j++)
        {
            Index random_index = static_cast<Index>(rand()% buffer_size);
cout << "Index" << random_index << endl;
            batches(i, j) = buffer(random_index);

            for(Index k = random_index; k < buffer_size-1; k++)
            {
                buffer(k) = buffer(k+1);
            }

            buffer(buffer_size-1) = rest_data(count);
cout << "buffer" << buffer << endl;
            count++;
        }
    }

    cout << batches << endl;
}


int main(void)
{
    try
    {
        cout << "Hello Blank Application" << endl;

        srand(static_cast<unsigned>(time(nullptr)));

//        get_batches();
		
        // Data Set

        const Index samples = 2000;
        const Index variables = 20;

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

        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        training_strategy.get_mean_squared_error_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.get_Levenberg_Marquardt_algorithm_pointer()->set_display_period(1);

        training_strategy.set_device_pointer(&device);

        training_strategy.perform_training();

        cout << "Bye Blank Application" << endl;

        return 0;

    }
       catch(exception& e)
    {
       cerr << e.what() << endl;
    }
  }


// OpenNN: Open Neural Networks Library.
// Copyright(C) 2005-2019 Artificial Intelligence Techniques, SL.
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
