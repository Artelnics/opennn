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

int main(int argc, char *argv[])
{
    try
    {
        srand(time(NULL));

        cout << "Hello OpenNN!" << endl;

        DataSet data_set;

        data_set.set_data_file_name("Z:/Images/DatasetRedDots-bmp/ground_truth.xml");

        data_set.read_ground_truth();

        data_set.print_data();
/*
        Index categories_number = 10;

        Index regions_number = 2000;
        Index channels_number = 227;
        Index region_rows = 227;
        Index region_columns = 227;

        Index inputs_number = regions_number*channels_number*region_rows*region_columns;

        Tensor<type, 3> image;

        NeuralNetwork neural_network;
/*
        RegionProposalLayer region_proposal_layer;
        neural_network.add_layer(&region_proposal_layer);

        FlattenLayer flatten_layer;
        neural_network.add_layer(&flatten_layer);

        ProbabilisticLayer probabilistric_layer(inputs_number, categories_number);
        neural_network.add_layer(&probabilistric_layer);
*/
        cout << "Bye OpenNN!" << endl;
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

