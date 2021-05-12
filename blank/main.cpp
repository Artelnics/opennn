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

#include "../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Write your code here

        // Data set

        DataSet data_set("C:/Users/Usuario/Documents/AirPassengers.csv", ';', true);

        data_set.set_lags_number(2);
        data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();

        const Tensor<type, 2> testing_inputs = data_set.get_testing_input_data();

        const Index inputs_number = data_set.get_input_variables_number();
        const Index targets_number = data_set.get_target_variables_number();

        // Neural network

        const Index lstm_neurons_number = 8;

        NeuralNetwork neural_network;

        ScalingLayer scaling_layer(inputs_number);
        LongShortTermMemoryLayer long_short_term_memory_layer(inputs_number, lstm_neurons_number);
        RecurrentLayer recurrent_layer(inputs_number, lstm_neurons_number);
        PerceptronLayer output_perceptron_layer(lstm_neurons_number, targets_number);
        UnscalingLayer unscaling_layer(targets_number);

        neural_network.add_layer(&scaling_layer);
        neural_network.add_layer(&long_short_term_memory_layer);
//        neural_network.add_layer(&recurrent_layer);
        neural_network.add_layer(&output_perceptron_layer);
        neural_network.add_layer(&unscaling_layer);

        output_perceptron_layer.set_activation_function(PerceptronLayer::Linear);

        long_short_term_memory_layer.set_timesteps(3);
        recurrent_layer.set_timesteps(2);

        neural_network.set_parameters_random();

        neural_network.print();

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

//        training_strategy.set_maximum_epochs_number(500);

        training_strategy.perform_training();

        cout << "Outputs: " << endl << neural_network.calculate_outputs(data_set.get_input_data()) << endl;






        cout << "Good bye!" << endl;

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
