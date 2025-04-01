//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B R E A S T   C A N C E R   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <time.h>

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Breast Cancer Example." << endl;

        // Data set

        DataSet data_set("data/breast_cancer.csv", ";", true, false);

        const Index input_variables_number = data_set.get_variables_number(DataSet::VariableUse::Input);
        const Index target_variables_number = data_set.get_variables_number(DataSet::VariableUse::Target);
        /*
        // Neural network

        const Index neurons_number = 3;
        
        NeuralNetwork neural_network(NeuralNetwork::ModelType::Classification,
                                    { input_variables_number }, { neurons_number, neurons_number }, { target_variables_number });

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::WEIGHTED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::QUASI_NEWTON_METHOD);
        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::L2);

        TrainingResults results = training_strategy.perform_training();

        const Tensor<string, 2> final_results = results.write_override_results(4);

        cout << "Final results:\n" << final_results << endl;
        */
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
