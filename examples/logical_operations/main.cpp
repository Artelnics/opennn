//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   L O G I C A L   O P E R A T O R S   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// This is a classical learning problem.

// System includes

#include <iostream>
#include <math.h>
#include <time.h>

// OpenNN includes

#include "../../opennn/opennn.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Logical Operations Example." << endl;

        srand(static_cast<unsigned>(time(nullptr)));

        // Data set

        DataSet data_set("../data/logical_operations.csv", ';', true);

        Tensor<string, 1> uses(8);
        uses.setValues({"Input","Input","Target","Target","Target","Target","Target","Target"});

        data_set.set_columns_uses(uses);
        data_set.set_training();

        const Index input_variables_number = data_set.get_input_numeric_variables_number();
        const Index target_variables_number = data_set.get_target_numeric_variables_number();

        // Neural network

        const Index hidden_neurons_number = 6;

        NeuralNetwork neural_network(NeuralNetwork::ModelType::Approximation, {input_variables_number, hidden_neurons_number, target_variables_number});

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.perform_training();

        // Print results to screen

        Tensor<type, 2> inputs(1,2);
        Tensor<type, 2> outputs;

        Tensor<Index, 1> inputs_dimensions = get_dimensions(inputs);

        cout << "\nX Y\tAND\tOR\tNAND\tNOR\tXOR\tXNOR\n" << endl;

        inputs(0,0) = type(1);
        inputs(0,1) = type(1);

        outputs = neural_network.calculate_outputs(inputs);

        cout << inputs << " " << outputs << endl;

        inputs(0,0) = type(1);
        inputs(0,1) = type(0.0);

        outputs = neural_network.calculate_outputs(inputs);

        cout << inputs << " " << outputs << endl;

        inputs(0,0) = type(0.0);
        inputs(0,1) = type(1);

        outputs = neural_network.calculate_outputs(inputs);

        cout << inputs << " " << outputs << endl;

        inputs(0,0) = type(0.0);
        inputs(0,1) = type(0.0);

        outputs = neural_network.calculate_outputs(inputs);

        cout << inputs << " " << outputs << endl;

        // Save results

        neural_network.save("../data/neural_network.xml");

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
