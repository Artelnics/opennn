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
#include "../../opennn/standard_networks.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        /*
        cout << "OpenNN. Amazon reviews example." << endl;

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        const Index embedding_dimension = 32;
        const Index neurons_number = 1;

        const Index input_vocabulary_size = language_dataset.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_dataset.get_target_vocabulary_size();

        const Index input_sequence_length = language_dataset.get_input_sequence_length();
        const Index targets_number = language_dataset.get_target_sequence_length();

        dimensions input_dimensions = {input_vocabulary_size, input_sequence_length, embedding_dimension};
        dimensions complexity_dimensions = {neurons_number};
        dimensions output_dimensions = {targets_number};


        // std::cout << "embedding_dimension = " << embedding_dimension << std::endl;
        // std::cout << "neurons_number = " << neurons_number << std::endl;

        // std::cout << "input_vocabulary_size = " << input_vocabulary_size << std::endl;
        // std::cout << "target_vocabulary_size = " << target_vocabulary_size << std::endl;

        // std::cout << "input_sequence_length = " << input_sequence_length << std::endl;
        // std::cout << "targets_number = " << targets_number << std::endl;

        // cout << "input_dimensions" << endl;
        // print_vector(input_dimensions);

        // cout << "complexity_dimensions" << endl;
        // print_vector(complexity_dimensions);

        // cout << "output_dimensions" << endl;
        // print_vector(output_dimensions);

        // TextClassificationNetwork text_classification_network(
        //     input_dimensions,
        //     complexity_dimensions,
        //     output_dimensions
        //     );

        NeuralNetwork neural_network;

        neural_network.add_layer(make_unique<Embedding>(dimensions({input_vocabulary_size, input_sequence_length}),
                                         embedding_dimension,
                                         "embedding_layer"
                                         ));

        cout << "neural_network " << endl;
        print_vector(neural_network.get_output_dimensions());

        // TrainingStrategy training_strategy(&text_classification_network, &language_dataset);

        // AdaptiveMomentEstimation* adam = static_cast<AdaptiveMomentEstimation*>(training_strategy.get_optimization_algorithm());
        // adam->set_maximum_epochs_number(50);

        // training_strategy.perform_training();

        const Index batch_size = 1;
        Tensor<type, 2> input(batch_size, input_sequence_length);
        input.setRandom();

        const Tensor<type, 3> outputs = neural_network.calculate_outputs<2,3>(input);
        // const Tensor<type, 3> outputs = text_classification_network.calculate_outputs<2, 3>(input);

        cout << "Outputs:\n" << outputs << endl;

        // const Tensor<type, 2> outputs = text_classification_network.calculate_outputs<3, 2>(input);

        // cout << "Outputs:\n" << outputs << endl;


        // const TestingAnalysis testing_analysis(&text_classification_network, &language_dataset);


        // cout << "Confusion matrix:\n"
        //      << testing_analysis.calculate_confusion() << endl;

        // MeanSquaredError mean_squared_error(&text_classification_network, &language_dataset);
        // cout << mean_squared_error.calculate_numerical_error() << endl;
        // cout << (mean_squared_error.calculate_gradient().abs() - mean_squared_error.calculate_numerical_gradient().abs()).maximum()<< endl;


        // const Tensor<type, 2> inputs = language_dataset.get_data(Dataset::VariableUse::Input);

        // cout << "inputs_dataset: " << inputs.dimensions() << endl;

        // const Eigen::array<Index, 3> new_dims = {1, inputs.dimension(0), inputs.dimension(1)};
        // const Tensor<type, 3> input = inputs.reshape(new_dims);

        // const Tensor<type, 2> outputs = text_classification_network.calculate_outputs<3, 2>(input);

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
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA), language_dataset.get_target_dimensions(), Dense2d::Activation::Logistic));
