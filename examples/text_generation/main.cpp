//   OpenNN: Open Neural Networks Library
//   www.opennn.org
//
//   T E X T   G E N E R A T I O N   E X A M P L E
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

static Eigen::IndexList<Eigen::type2index<0>> alongClass; // 1st dimension
Eigen::IndexList<int, Eigen::type2index<1>> bcast;
template<typename T>
void ETensorLogSoftmax(const Tensor<T, 2>& input, Tensor<T, 2>& output) {
  bcast.set(0, input.dimension(0));
  Eigen::IndexList<Eigen::type2index<1>, typename Eigen::Index> dims2d;
  dims2d.set(1, input.dimension(1));
  // creating a real tensor is faster thant auto which would resolve to some TensorExpr
  Eigen::Tensor<T,2> wMinusMax = input - input.maximum(alongClass).eval().reshape(dims2d).broadcast(bcast);

  cout <<"wMinusMax:\n" <<  wMinusMax << endl;

  output = wMinusMax - wMinusMax.exp().sum(alongClass).log().eval().reshape(dims2d).broadcast(bcast);
}

int main(void)
{
    try
    {
        Tensor<type,2> matrix(3,3);

        Tensor<type,2> softmax(3,3);

        matrix.setValues({{1,1,1},{2,2,2},{3,3,3}});

        cout << matrix << endl;

        tensor_softmax(matrix,softmax);

        cout << softmax << endl;

        system("pause");

        // Dataset

        DataSet data_set;

        data_set.set_data_file_name("../data/text_generation.csv");

        data_set.read_text();

        const Index lags_number = 1;
        data_set.set_lags_number(lags_number);

        const Index steps_ahead_number = 1;
        data_set.set_steps_ahead_number(steps_ahead_number);

        data_set.transform_time_series();

        data_set.print();
        data_set.print_data_preview();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index hidden_neurons_number = 1;

        Tensor<Index, 1> architecture(3);
        architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

        //NeuralNetwork neural_network(NeuralNetwork::Forecasting, architecture);

        NeuralNetwork neural_network;

        LongShortTermMemoryLayer lstm_layer(input_variables_number,hidden_neurons_number);
        RecurrentLayer rnn_layer(input_variables_number,hidden_neurons_number);
        PerceptronLayer perceptron_layer(input_variables_number,hidden_neurons_number);
        ProbabilisticLayer probabilistic_layer(hidden_neurons_number,target_variables_number);

        //neural_network.add_layer(&lstm_layer);
        //neural_network.add_layer(&rnn_layer);
        neural_network.add_layer(&perceptron_layer);
        neural_network.add_layer(&probabilistic_layer);

        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);

        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();

        adam->set_loss_goal(1.0e-3);
        adam->set_maximum_epochs_number(5);
        adam->set_display_period(1);

        training_strategy.perform_training();

//        // Save results

//        data_set.save("../data/data_set.xml");
//        neural_network.save("../data/neural_network.xml");
//        training_strategy.save("../data/training_strategy.xml");

        cout << "End Text Generation Example" << endl;

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
