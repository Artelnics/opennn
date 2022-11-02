//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   B L A N K   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

// System includes

#include <stdio.h>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

// OpenNN includes

#include "../opennn/opennn.h"
using namespace opennn;

int main()
{
    try
    {
        srand(time(NULL));

        cout << "Hello OpenNN!" << endl;

        Tensor<Tensor<type, 1>, 1> image_test = read_bmp_image_data("Z:/Images/DatasetRedDots-bmp/3.bmp");

        /*
        DataSet data_set;

        data_set.set_data_file_name("Z:/Images/DatasetRedDots-bmp/ground_truth.xml");

        data_set.read_ground_truth();

//        data_set.print_data();


//        Index categories_number = 10;

//        Index regions_number = 2000;
//        Index channels_number = 227;
//        Index region_rows = 227;
//        Index region_columns = 227;

//        Index inputs_number = regions_number*channels_number*region_rows*region_columns;

//        Tensor<type, 3> image;


        const Index input_variables_number = data_set.get_input_variables_number();
        const Index target_variables_number = data_set.get_target_variables_number();

        cout << "input_variables_number: " << input_variables_number << endl;
        cout << "target_variables_number: " << target_variables_number << endl;

        const Tensor<Index, 1> input_variables_dimensions = data_set.get_input_variables_dimensions();

        cout << "input_variables_dimensions: " << input_variables_dimensions << endl;

        NeuralNetwork neural_network;

//        RegionProposalLayer region_proposal_layer;
//        neural_network.add_layer(&region_proposal_layer);

        FlattenLayer flatten_layer(input_variables_dimensions);
        neural_network.add_layer(&flatten_layer);

//        cout << "Flatten layer outputs dimensions: " << flatten_layer.get_outputs_dimensions() << endl;

        ProbabilisticLayer probabilistic_layer(input_variables_number, target_variables_number);
        neural_network.add_layer(&probabilistic_layer);

//        cout << "Neural network architecture: " << endl << neural_network.get_layers_names() << endl;

        // Training strategy

        TrainingStrategy training_strategy(&neural_network, &data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::NORMALIZED_SQUARED_ERROR);
        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);
        training_strategy.perform_training();
*/
        cout << "Bye OpenNN!" << endl;

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

