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

        cout << "OpenNN. Text Generation Example." << endl;

        // Dataset

        DataSet data_set;

        data_set.set_data_file_name("../data/text_generation.csv");

        data_set.read_text();

        const int lags_number = 1;
        data_set.set_lags_number(lags_number);
        //data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();
        data_set.print();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "Input variables number: " << input_variables_number << endl;
        cout << "Target variables number: " << target_variables_number << endl;

//        // Neural network

//        const Index hidden_neurons_number = 1;

//        Tensor<Index, 1> architecture(3);
//        architecture.setValues({input_variables_number, hidden_neurons_number, target_variables_number});

//        //NeuralNetwork neural_network(NeuralNetwork::Forecasting, architecture);

//        NeuralNetwork neural_network;

//        LongShortTermMemoryLayer lstm_layer(input_variables_number,hidden_neurons_number);
//        ProbabilisticLayer probabilistic_layer(hidden_neurons_number,target_variables_number);

//        neural_network.add_layer(&lstm_layer);
//        neural_network.add_layer(&probabilistic_layer);

//        neural_network.print();

//        // Training strategy

//        TrainingStrategy training_strategy(&neural_network, &data_set);

//        training_strategy.set_loss_method(TrainingStrategy::CROSS_ENTROPY_ERROR);
//        training_strategy.set_optimization_method(TrainingStrategy::ADAPTIVE_MOMENT_ESTIMATION);

//        AdaptiveMomentEstimation* adam = training_strategy.get_adaptive_moment_estimation_pointer();

//        adam->set_loss_goal(1.0e-3);
//        adam->set_maximum_epochs_number(5);
//        adam->set_display_period(1);

//        training_strategy.perform_training();

//        // Testing analysis

//        Tensor<type, 2> inputs(3,6);

//        inputs.setValues({{1,0,0,0,0,0},
//                          {1,1,1,0.5,0.5,1},
//                          {0,1,0,1,0,1}});

//        cout << "inputs: " << endl;
//        cout << inputs << endl;

//        cout << "outputs: " << endl;
//        cout << neural_network.calculate_outputs(inputs) << endl;

//        data_set.unscale_input_variables(scaling_inputs_methods, inputs_descriptives);

//        TestingAnalysis testing_analysis(&neural_network, &data_set);

//        Tensor<Index, 2> confusion = testing_analysis.calculate_confusion();

//        cout << "Confusion: " << endl;
//        cout << confusion << endl;

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
