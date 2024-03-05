//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
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
        cout << "OpenNN. Airfoil self noise example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/airfoil_self_noise.csv", ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        // Neural network

        const Index neurons_number = 10;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {input_variables_number, neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::MEAN_SQUARED_ERROR);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        training_strategy.set_maximum_epochs_number(10);

        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);

        testing_analysis.print_goodness_of_fit_analysis();

        // Save results

        neural_network.save("../data/neural_network.xml");
        neural_network.save_expression_c("../data/airfoil_self_noise.c");

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(std::exception& e)
    {
        std::cerr << e.what() << endl;

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
