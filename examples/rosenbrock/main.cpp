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

#include "../../opennn/opennn.h"

using namespace OpenNN;
using namespace std;
using namespace Eigen;

int main()
{          
    try
    {
        cout << "OpenNN. Rosenbrock Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data Set

        const Index samples = 10000;
        const Index variables = 10;
        const Index hidden_neurons_number = 10;

        DataSet data_set("C:/rosenbrock.csv", ',', true);

        data_set.set_training();

        // Neural network

        const Index inputs_number = data_set.get_input_variables_number();
        const Index outputs_number = data_set.get_target_variables_number();

        NeuralNetwork neural_network(NeuralNetwork::Approximation, {inputs_number, hidden_neurons_number, outputs_number});     

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::MEAN_SQUARED_ERROR);

        training_strategy.get_loss_index_pointer()->set_regularization_method(LossIndex::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::LEVENBERG_MARQUARDT_ALGORITHM);

        training_strategy.set_display_period(10);
        //training_strategy.set_maximum_epochs_number(1000000);

        training_strategy.get_quasi_Newton_method_pointer()->set_minimum_loss_decrease(-1000);
/*
        LevenbergMarquardtAlgorithm* optimization_algorithm_pointer = training_strategy.get_Levenberg_Marquardt_algorithm_pointer();

        optimization_algorithm_pointer->set_minimum_loss_decrease(-1000);
        optimization_algorithm_pointer->set_display_period(1000);
        optimization_algorithm_pointer->set_maximum_epochs_number(10000);
*/
        training_strategy.perform_training();

        cout << "End Rosenbrock" << endl;

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
