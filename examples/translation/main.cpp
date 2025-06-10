//   OpenNN: Open Neural Networks Library
//   www.opennn.net
//
//  T R A N S L A T I O N   A P P L I C A T I O N
//
//   Artificial Intelligence Techniques SL (Artelnics)
//   artelnics@artelnics.com

// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "../../opennn/training_strategy.h"
#include "../../opennn/language_dataset.h"
#include "embedding_layer.h"
#include "flatten_layer_3d.h"
#include "multihead_attention_layer.h"
#include "normalization_layer_3d.h"
#include "perceptron_layer.h"

using namespace std;
using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Translation Example." << endl;

        // Data set

        LanguageDataset language_dataset("/Users/artelnics/Documents/opennn/examples/translation/data/ENtoES_dataset_reduced_6.txt");

        // Sentiment analysis case

        const Index sequence_length = 10;
        const Index vocabulary_size = 50;
        const Index embedding_dimension = 32;
        const Index heads_number = 4;
        const dimensions outputs_number = { 1 };

        NeuralNetwork neural_network;
        neural_network.add_layer(make_unique<Embedding>(vocabulary_size, sequence_length, embedding_dimension, "Embedding"));
        neural_network.add_layer(make_unique<Normalization3d>(sequence_length, embedding_dimension, "Normalization"));
        neural_network.add_layer(make_unique<MultiHeadAttention>(sequence_length, sequence_length, embedding_dimension, heads_number, false, "Multihead_attention"));
        neural_network.set_layer_inputs_indices("Multihead_attention",{"Normalization", "Normalization"});
        neural_network.add_layer(make_unique<Flatten3d>(neural_network.get_output_dimensions()));
        neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), outputs_number));

        cout << "Parameters number: " << neural_network.get_parameters_number() << endl;

        TrainingStrategy training_strategy(&neural_network, &language_dataset);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_2D);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        AdaptiveMomentEstimation* adaptive_moment_estimation = training_strategy.get_adaptive_moment_estimation();

        language_dataset.set(Dataset::SampleUse::Training);
        adaptive_moment_estimation->set_loss_goal(0.3);
        adaptive_moment_estimation->set_maximum_epochs_number(100);
        adaptive_moment_estimation->set_maximum_time(59400);
        adaptive_moment_estimation->set_batch_samples_number(12);
        adaptive_moment_estimation->set_display_period(1);

        training_strategy.perform_training();

        // Prediction test
        cout << "Vocabulary:" << endl;
        language_dataset.print_input_vocabulary();

        Tensor<type,2> testing_data(3,10);
        testing_data(0,0) = 2;
        testing_data(0,1) = 4;
        testing_data(0,2) = 29;
        testing_data(0,3) = 12;
        testing_data(0,4) = 13;
        testing_data(0,5) = 17;
        testing_data(0,6) = 3;
        testing_data(0,7) = 0;
        testing_data(0,8) = 0;
        testing_data(0,9) = 0;
        testing_data(1,0) = 2;
        testing_data(1,1) = 4;
        testing_data(1,2) = 5;
        testing_data(1,3) = 6;
        testing_data(1,4) = 7;
        testing_data(1,5) = 8;
        testing_data(1,6) = 3;
        testing_data(1,7) = 0;
        testing_data(1,8) = 0;
        testing_data(1,9) = 0;
        testing_data(2,0) = 2;
        testing_data(2,1) = 4;
        testing_data(2,2) = 29;
        testing_data(2,3) = 12;
        testing_data(2,4) = 30;
        testing_data(2,5) = 31;
        testing_data(2,6) = 17;
        testing_data(2,7) = 3;
        testing_data(2,8) = 0;
        testing_data(2,9) = 0;

        cout << "Outputs:\n" << neural_network.calculate_outputs(testing_data).round()<<endl;

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
// Copyright (C) 2005-2025 Artificial Intelligence Techniques SL
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
