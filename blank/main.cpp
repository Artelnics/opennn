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

#include "../opennn/language_data_set.h"
#include "../opennn/embedding_layer.h"
#include "../opennn/multihead_attention_layer.h"
#include "../opennn/probabilistic_layer_3d.h"
#include "../opennn/training_strategy.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        // Data set

        LanguageDataSet language_data_set("/Users/artelnics/Desktop/masked_large_huge.txt");

        const Index input_length = language_data_set.get_input_length();
        const Index target_length = language_data_set.get_target_length();
        const Index input_vocabulary_size = language_data_set.get_input_vocabulary_size();
        const Index target_vocabulary_size = language_data_set.get_target_vocabulary_size();

        cout << input_length << endl;

        // language_data_set.print();

        // language_data_set.print_vocabulary(text_data_set.get_target_vocabulary());
        // throw runtime_error("");

        // NeuralNetwork

        const Index embedding_dimension = 32;
        const Index heads_number = 2;
        const dimensions outputs_number = { 2 };

        NeuralNetwork neural_network;

        neural_network.add_layer(make_unique<Embedding>(input_vocabulary_size, input_length, embedding_dimension, "Embedding"));
        neural_network.add_layer(make_unique<MultiHeadAttention>(input_length, input_length, embedding_dimension, heads_number, false, "Multihead_attention"));
        neural_network.set_layer_inputs_indices("Multihead_attention", {"Embedding", "Embedding"});
        neural_network.add_layer(make_unique<Probabilistic3d>(target_length, embedding_dimension, target_vocabulary_size));

        // Transformer transformer()

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &language_data_set);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR_3D);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        language_data_set.split_samples_sequential(0.8, 0.2, 0);
        // training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.3);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(3000);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(244800);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(32);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        Tensor<type, 2> one_word = language_data_set.get_data(DataSet::VariableUse::Input).chip(1000,0).reshape(Eigen::array<Index,2>{1,15});
        cout << "one_word:\n " << one_word << endl;

        cout <<"One word output:\n" << neural_network.calculate_output(one_word).argmax(2) << endl;

        Tensor<type, 2> one_word_2 = language_data_set.get_data(DataSet::VariableUse::Input).chip(11000,0).reshape(Eigen::array<Index,2>{1,15});
        cout << "one_word_2:\n" << one_word_2 << endl;

        cout << "One word output_2:\n" << neural_network.calculate_output(one_word_2).argmax(2) << endl;

        Tensor<type, 2> one_word_3 = language_data_set.get_data(DataSet::VariableUse::Input).chip(1450,0).reshape(Eigen::array<Index,2>{1,15});
        cout << "one_word_3:\n" << one_word_3 << endl;

        cout <<"One word output_3:\n" << neural_network.calculate_output(one_word).argmax(2) << endl;

        // text_data_set.set(DataSet::SampleUse::Testing);

        // const TestingAnalysis testing_analysis(&neural_network, &text_data_set);

        // TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        // cout << "TESTING ANALYSIS:" << endl;
        // cout << "Roc curve: " << roc_analysis.area_under_curve << endl;

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
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
