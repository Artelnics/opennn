//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//   A M A Z O N   R E V I E W S
//
//   Artificial Intelligence Techniques SL
//   artelnics@artelnics.com

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <time.h>

#include "../../opennn/language_dataset.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"
#include "../../opennn/embedding_layer.h"
#include "../../opennn/flatten_layer_3d.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

//        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index batch_size = 1;

        const Index vocabulary_size = 4;
        const Index sequence_length = 3;
        const Index embedding_dimension = 2;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Embedding>(vocabulary_size, sequence_length, embedding_dimension));



        //neural_network.add_layer(make_unique<Flatten3d>(dimensions({sequence_length, embedding_dimension})));

        Tensor<type, 2> inputs(batch_size, sequence_length);
        inputs.setConstant(1);

        cout << neural_network.calculate_outputs(inputs) << endl;

        cout << "Good bye!" << endl;

        return 0;
    }
    catch(const exception& e)
    {
        cout << e.what() << endl;

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
// License along with this library; if not, write to the Free Software Foundation.
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
