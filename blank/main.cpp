//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
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

#include "../opennn/opennn.h"
#include "../opennn/layer.h"

using namespace opennn;
using namespace std;
using namespace Eigen;

int main()
{
    try
    {
        cout << "OpenNN. Blank application." << endl;

        // Data Set

        DataSet data_set("/home/artelnics2020/Escritorio/datasets/5_year_mortality.csv",',',true);

        const Index input_number_variables = data_set.get_input_variables_number();
        const Index target_number_variables = data_set.get_target_variables_number();

        // Neural network

        NeuralNetwork neural_network(NeuralNetwork::ProjectType::Classification,{input_number_variables, 10, target_number_variables});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        // Model Selection

        ModelSelection model_selection(&training_strategy);

        model_selection.set_inputs_selection_method(ModelSelection::InputsSelectionMethod::GROWING_INPUTS);

        model_selection.get_growing_inputs_pointer()->set_display(true);

        model_selection.get_growing_inputs_pointer()->set_maximum_inputs_number(15);
        model_selection.get_growing_inputs_pointer()->set_minimum_inputs_number(10);

        InputsSelectionResults input_selection_results = model_selection.perform_inputs_selection();

        cout << "selection error history" << input_selection_results.selection_error_history << endl;

        cout << "selection error history" << input_selection_results.training_error_history << endl;

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
