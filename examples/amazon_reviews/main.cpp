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
#include "../../opennn/perceptron_layer.h"
#include "../../opennn/multihead_attention_layer.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        //const Index sequence_length = 3;
        const Index embedding_dimension = 4;
        const Index heads_number = 2;
        //const Index batch_size = 3;

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Embedding>(language_dataset.get_input_dimensions(), embedding_dimension));
        neural_network.add_layer(make_unique<MultiHeadAttention>(neural_network.get_output_dimensions(), heads_number), {0,0});
        neural_network.add_layer(make_unique<Flatten3d>(neural_network.get_output_dimensions()));
        neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));

        MeanSquaredError mean_squared_error(&neural_network, &language_dataset);

        cout << (mean_squared_error.calculate_gradient().abs() - mean_squared_error.calculate_numerical_gradient().abs()).maximum()<< endl;
/*
        TrainingStrategy training_strategy(&neural_network, &language_dataset);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(200);

        training_strategy.perform_training();

        TestingAnalysis testing_analysis(&neural_network, &language_dataset);

        testing_analysis.print_binary_classification_tests();

        //Tensor<type, 3> inputs()
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
