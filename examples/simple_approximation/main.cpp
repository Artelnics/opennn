//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   S I M P L E   F U N C T I O N   R E G R E S S I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
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
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Simple Function Regression Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data Set

        DataSet data_set("../data/simple_function_regression.csv", ';', true);

        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();
        const Index hidden_neurons_number = 3;

        // Neural Network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation,
                                     {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.set_display_period(1000);
        training_strategy.perform_training();

        // Save results

        neural_network.save_expression_python("simple_function_regresion.py");

        cout << "Bye Simple Function Regression" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cerr << e.what() << endl;

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
