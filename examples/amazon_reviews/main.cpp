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

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

<<<<<<< HEAD
        language_dataset.print_data();
=======
        const Index batch_size = 1;
        language_dataset.print_data();

/*
        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index sequence_length = language_dataset.get_input_length();
        const Index embedding_dimension = 32;
>>>>>>> 6256d37335b57d7210ba7e2a5bb48ca3ec4116d4

/*
        const Index vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index sequence_length = language_dataset.get_input_length();
        const Index embedding_dimension = 32;

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Embedding>(vocabulary_size, sequence_length, embedding_dimension));
        const Index targets_number = language_dataset.get_target_length();

        dimensions input_dimensions      = {input_vocabulary_size, sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions     = {targets_number};

<<<<<<< HEAD
        const Index batch_size = 8;
=======
        NeuralNetwork neural_network(
            NeuralNetwork::ModelType::TextClassification,
            input_dimensions,
            complexity_dimensions,
            output_dimensions);

        neural_network.print();

        // Training strategy
>>>>>>> 57ef74e1e (temporal commit)

        TrainingStrategy training_strategy(&neural_network, &language_dataset);
        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_2D);
        training_strategy.set_maximum_epochs_number(1000);

<<<<<<< HEAD

        //neural_network.add_layer(make_unique<Flatten3d>(dimensions({sequence_length, embedding_dimension})));
>>>>>>> 6256d37335b57d7210ba7e2a5bb48ca3ec4116d4

        Tensor<type, 2> inputs(batch_size, sequence_length);
        inputs.setConstant(1);

        cout << neural_network.calculate_outputs_2_3(inputs) << endl;

        //neural_network.add_layer(make_unique<Flatten3d>(dimensions({sequence_length, embedding_dimension})));
*/

        training_strategy.print();

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &language_dataset);

        testing_analysis.print_binary_classification_tests();
*/
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
