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

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index sequence_length = language_dataset.get_input_length();
        const Index embedding_dimension = 32;

        const Index neurons_number = 64;

        const Index targets_number = language_dataset.get_target_length();

        dimensions input_dimensions      = {input_vocabulary_size, sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions     = {targets_number};

        NeuralNetwork neural_network(
            NeuralNetwork::ModelType::TextClassification,
            input_dimensions,
            complexity_dimensions,
            output_dimensions);

        Tensor<type, 2> inputs(1,1);
        inputs.setRandom();

        Tensor<type, 3> outputs = neural_network.calculate_outputs_2_3(inputs);

        CrossEntropyError3d cross_entropy_error_3d(&neural_network, &language_dataset);

        cout << cross_entropy_error_3d.calculate_error_xxx() << endl;

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
