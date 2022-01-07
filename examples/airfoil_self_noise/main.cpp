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

using namespace opennn;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        DataSet data_set("../data/R_100000_samples_101_variables.csv", ',', true);

        const Index inputs_number = data_set.get_input_variables_number();
        const Index targets_number = data_set.get_target_variables_number();
        const Index neurons_number = 100;

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Approximation, {inputs_number, neurons_number, targets_number});

        TrainingStrategy training_strategy(&neural_network, &data_set);

        ModelSelection model_selection(&training_strategy);

        model_selection.set_inputs_selection_method(ModelSelection::InputsSelectionMethod::GENETIC_ALGORITHM);

        GeneticAlgorithm* genetic_algorithm_pointer = model_selection.get_genetic_algorithm_pointer();

        genetic_algorithm_pointer->set_individuals_number(10);

        model_selection.perform_inputs_selection();

        cout << "Good bye!" << endl;

        system("pause");

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
