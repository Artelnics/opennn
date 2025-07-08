//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <string>

#include "../../opennn/dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/standard_networks.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/optimization_algorithm.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Iris Plant Example." << endl;

        // Data set

        Dataset dataset("../data/iris_plant_original.csv", ";", true, false);

        const Index inputs_number = dataset.get_variables_number(Dataset::VariableUse::Input);
        const Index targets_number = dataset.get_variables_number(Dataset::VariableUse::Target);

        // Neural network

        const Index neurons_number = 6;

        ClassificationNetwork classification_network({inputs_number}, {neurons_number}, {targets_number});

        TrainingStrategy training_strategy(&classification_network, &dataset);

        training_strategy.perform_training();

        // Testing analysis

        const TestingAnalysis testing_analysis(&classification_network, &dataset);

        cout << "Confusion matrix:\n"
             << testing_analysis.calculate_confusion() << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }   
}  

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
