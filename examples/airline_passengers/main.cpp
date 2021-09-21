//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R L I N E  P A S S E N G E R S  A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// This is a forecasting application.

// System includes

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>
#include <math.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace OpenNN;

int main()
{
    try
    {
        cout << "OpenNN. Airline Passengers Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/airline_passengers.csv", ',', true);

        const int lags_number = 2;

        data_set.set_lags_number(lags_number);

        data_set.set_steps_ahead_number(1);

        data_set.transform_time_series();

        data_set.print();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        data_set.set_columns_scalers(MinimumMaximum);

        // Neural network

        const Index hidden_neurons_number = 16;

        NeuralNetwork neural_network(NeuralNetwork::Forecasting, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::QUASI_NEWTON_METHOD);

        const TrainingResults training_results = training_strategy.perform_training();

        // Testing Analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        Tensor<type,1> testing_errors = testing_analysis.calculate_testing_errors();

        // Calculate and export outputs

        data_set.get_training_input_data();

        Tensor<type, 2> testing_input_data = data_set.get_testing_input_data();
        Tensor<type, 2> testing_output_data = neural_network.calculate_outputs(testing_input_data);

        ofstream output_data_file;
        output_data_file.open("output_LSTM.csv");

        for(Index i= 0; i < lags_number; i++){
            output_data_file << testing_input_data(i) << ";" << ""  <<endl;
        }
        for(Index i= lags_number; i < testing_input_data.dimensions()[0]; i++){
            output_data_file << testing_input_data(i) << ";" << testing_output_data(i-lags_number) <<endl;
        }

        // Save network results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/neural_network.py");

        cout << "End Airline Passengers Example" << endl;

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
