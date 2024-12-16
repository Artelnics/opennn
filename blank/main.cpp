//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   I R I S   P L A N T   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

#include "../opennn/opennn.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Blank project." << endl;

        const Index samples_number = get_random_index(1, 10);
        const Index inputs_number = get_random_index(1, 10);
        const Index targets_number = get_random_index(1, 10);
        const Index neurons_number = get_random_index(1, 10);

        DataSet data_set(samples_number, { inputs_number }, { targets_number });
        data_set.set_data_random();
        data_set.set(DataSet::SampleUse::Training);
        
        //Batch batch(samples_number, &data_set);
        /*
        batch.fill(data_set.get_sample_indices(DataSet::SampleUse::Training),
            data_set.get_variable_indices(DataSet::VariableUse::Input),
            data_set.get_variable_indices(DataSet::VariableUse::Target));
        /*
        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation,
            { inputs_number }, { neurons_number }, { targets_number });

        neural_network.set_parameters_random();

        ForwardPropagation forward_propagation(samples_number, &neural_network);

//        neural_network.forward_propagate(batch.get_input_pairs(), forward_propagation, true);

        // Loss index

//        NormalizedSquaredError normalized_squared_error(&neural_network, &data_set);

//        BackPropagation back_propagation(samples_number, &normalized_squared_error);
//        normalized_squared_error.back_propagate(batch, forward_propagation, back_propagation);
*/
        cout << "Bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

        return 1;
    }
}

// OpenNN: Open Neural Networks Library.
// Copyright (C) 2005-2024 Artificial Intelligence Techniques SL
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
