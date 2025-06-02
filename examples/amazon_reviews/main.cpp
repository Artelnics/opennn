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

#include "../../opennn/language_data_set.h"
#include "../../opennn/neural_network.h"
#include "../../opennn/training_strategy.h"
#include "../../opennn/testing_analysis.h"

using namespace opennn;

int main()
{
    try
    {
        cout << "OpenNN. Amazon reviews example." << endl;

        // Data set

        LanguageDataset language_dataset("../data/amazon_cells_labelled.txt");

        // const vector<string> input_words = language_dataset.get_raw_variable_names(Dataset::VariableUse::Input);
        // const vector<string> targets_names = language_dataset.get_variable_names(Dataset::VariableUse::Target);

        // const Index words_number = language_dataset.get_variables_number(Dataset::VariableUse::Input);
        // const Index target_variables_number = language_dataset.get_variables_number(Dataset::VariableUse::Target);

        // cout<<words_number<<endl;
        // cout<<target_variables_number<<endl;

        // Neural Network


        NeuralNetwork neural_network(NeuralNetwork::ModelType::TextClassification,
                                     {  }, { }, {  });

        neural_network.print();
/*
        // const Index maximum_sequence_length = language_dataset.get_input_length();
        // const Index vocabulary_size = language_dataset.get_input_vocabulary_size();
        // const Index embedding_dimension = 64;
        // const Index heads_number = 4;
        // const dimensions outputs_number = { 1 };

        //cout << "Maximum seq: " << maximum_sequence_length << endl;
        //cout << "Vocab size: " << vocabulary_size << endl;

        //NeuralNetwork neural_network;
        //neural_network.add_layer(make_unique<Embedding>(vocabulary_size, maximum_sequence_length, embedding_dimension, "Embedding"));
        //neural_network.add_layer(make_unique<MultiHeadAttention>(maximum_sequence_length, maximum_sequence_length, embedding_dimension, heads_number, false, "Multihead_attention"));
        //neural_network.set_layer_inputs_indices("Multihead_attention",{"Embedding", "Embedding"});
        // neural_network.add_layer(make_unique<Addition3d>(maximum_sequence_length, embedding_dimension, "Addition"));
        // neural_network.set_layer_inputs_indices("Addition", {"Embedding", "Multihead_attention"});
        // neural_network.add_layer(make_unique<Normalization3d>(maximum_sequence_length, embedding_dimension, "Normalization"));
        //neural_network.add_layer(make_unique<Flatten3d>(neural_network.get_output_dimensions()));
        //neural_network.add_layer(make_unique<Dense2d>(neural_network.get_output_dimensions(), outputs_number));

        // Training Strategy

        TrainingStrategy training_strategy(&neural_network, &language_dataset);

        training_strategy.set_loss_method(TrainingStrategy::LossMethod::CROSS_ENTROPY_ERROR);

        training_strategy.get_loss_index()->set_regularization_method(LossIndex::RegularizationMethod::NoRegularization);

        training_strategy.set_optimization_method(TrainingStrategy::OptimizationMethod::ADAPTIVE_MOMENT_ESTIMATION);

        // training_strategy.get_adaptive_moment_estimation()->set_custom_learning_rate(depth);

        language_dataset.split_samples_sequential(0.8,0,0.2);
        // training_strategy.get_adaptive_moment_estimation()->set_loss_goal(0.3);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_epochs_number(100);
        training_strategy.get_adaptive_moment_estimation()->set_maximum_time(244800);
        training_strategy.get_adaptive_moment_estimation()->set_batch_samples_number(32);

        training_strategy.get_adaptive_moment_estimation()->set_display(true);
        training_strategy.get_adaptive_moment_estimation()->set_display_period(1);

        TrainingResults training_results = training_strategy.perform_training();

        // language_dataset.set(Dataset::SampleUse::Testing);
/*
        const TestingAnalysis testing_analysis(&neural_network, &language_dataset);

        TestingAnalysis::RocAnalysis roc_analysis = testing_analysis.perform_roc_analysis();

        cout << "TESTING ANALYSIS:" << endl;
        cout << "Roc curve: " << roc_analysis.area_under_curve << endl;

        // pair<type, type> transformer_error_accuracy = testing_analysis.test_transformer();

        // cout << "TESTING ANALYSIS:" << endl;
        // cout << "Testing error: " << transformer_error_accuracy.first << endl;
        // cout << "Testing accuracy: " << transformer_error_accuracy.second << endl;

        // transformer.save("/home/artelnics/Escritorio/andres_alonso/ViT/ENtoES_model.xml");
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
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
