//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>

#include "../../opennn/dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Example." << endl;

        // Data set

        Dataset dataset("../data/breast_cancer.csv", ";", true, false);

        const Index inputs_number = dataset.get_variables_number(Dataset::VariableUse::Input);
        const Index targets_number = dataset.get_variables_number(Dataset::VariableUse::Target);
        
        // Neural network

        const Index neurons_number = 3;
        
        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                    { inputs_number }, { neurons_number}, { targets_number });

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &dataset);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &dataset);
        testing_analysis.print_binary_classification_tests();

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

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
