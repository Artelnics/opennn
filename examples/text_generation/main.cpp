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

int main(void)
{
    try
    {
        // Dataset

        DataSet data_set;

        data_set.set_data_file_name("../data/el_quijote.txt");

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

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

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
