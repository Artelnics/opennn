//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R L I N E   P A S S E N G E R S
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

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Ailine passengers example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/airline_passengers.csv", ',', true);

        const Index lags_number = 2;
        const Index steps_ahead_number = 1;

        data_set.set_lags_number(lags_number);
        data_set.set_steps_ahead_number(steps_ahead_number);

        data_set.transform_time_series();

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "Input variables number: " << input_variables_number << endl;
        cout << "Target variables number: " << target_variables_number << endl;

        // Neural network

        const Index hidden_neurons_number = 10;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Forecasting, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

//        neural_network.print();

        const TrainingResults training_results = training_strategy.perform_training();

        // Calculate outputs

        Tensor<type, 2> input(4,2);
        Tensor<Index, 1> input_dims = get_dimensions(input);
        input.setValues({
                                 {150,146},
                                 {124,253},
                                 {124,264},
                                 {124,221}
                             });

        Tensor<type, 2> output;

        output = neural_network.calculate_outputs(input.data(), input_dims);

        cout << "Input data:\n" << input << "\nPredictions:\n" << output << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_python("../data/neural_network.py");

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
