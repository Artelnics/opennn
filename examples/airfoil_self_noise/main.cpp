//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A I R F O I L   S E L F   N O I S E   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <iostream>
#include <string>
#include <time.h>

#include "../../opennn/testing_analysis.h"
#include "../../opennn/model_selection.h"
#include "../../opennn/testing_analysis.h"

int main()
{
    try
    {
        cout << "Airfoil self noise " << endl;

        // Data set
        
        Dataset dataset("../data/airfoil_self_noise.csv", ";", true, false);

        const Index inputs_number = dataset.get_variables_number(Dataset::VariableUse::Input);
        const Index targets_number = dataset.get_variables_number(Dataset::VariableUse::Target);

        // Neural network

        const Index neurons_number = 12;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
                                     {inputs_number}, {neurons_number}, {targets_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &dataset);

        training_strategy.perform_training();

        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &dataset);

        testing_analysis.print_goodness_of_fit_analysis();

        neural_network.print();
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
